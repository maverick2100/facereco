#!/usr/bin/env python3
"""Multi-camera face recognition for Apple Silicon (SCRFD + ArcFace on the Neural Engine).

Usage:
    python main.py run                       # start all enabled cameras
    python main.py enroll                    # enrol everyone in data/enrollment/<name>/*.jpg
    python main.py enroll --name "Alice" --images a.jpg b.jpg
    python main.py persons                   # list enrolled people
    python main.py delete-person 3
    python main.py export --since 2026-11-13 # CSV into data/exports/
    python main.py info                      # model + environment info
"""
from __future__ import annotations

import os

# Keep CPU-side math libraries single-threaded: the heavy lifting belongs on the ANE,
# and 20 cameras' worth of small numpy/OpenCV ops are faster without thread-pool churn.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional  # noqa: E402

import cv2  # noqa: E402

cv2.setNumThreads(1)
cv2.setUseOptimized(True)

from config.settings import Settings, load_settings  # noqa: E402
from constants import ENROLLMENT_DIR  # noqa: E402
from database.camera_repository import CameraRepository  # noqa: E402
from database.connection import Database  # noqa: E402
from database.embedding_repository import EmbeddingRepository  # noqa: E402
from database.event_repository import EventRepository  # noqa: E402
from database.person_repository import PersonRepository  # noqa: E402
from database.schema import init_schema  # noqa: E402
from ingestion.camera_health import CameraHealthRegistry  # noqa: E402
from ingestion.frame_receiver import FrameReceiver, OpenCVManager  # noqa: E402
from inference.inference_manager import InferenceManager  # noqa: E402
from monitoring.performance import PerformanceTracker  # noqa: E402
from monitoring.system_health import SystemHealth  # noqa: E402
from recognition.duplicate_filter import DuplicateFilter, TrackCache  # noqa: E402
from recognition.embedding_manager import EmbeddingGallery  # noqa: E402
from recognition.face_matcher import FaceMatcher  # noqa: E402
from recognition.identity_manager import IdentityManager  # noqa: E402
from scheduler.frame_scheduler import FrameScheduler  # noqa: E402
from services.enrollment_service import EnrollmentService  # noqa: E402
from services.event_service import EventService  # noqa: E402
from services.recognition_service import RecognitionService  # noqa: E402
from utils.logger import setup_logging  # noqa: E402
from utils.time_utils import parse_datetime  # noqa: E402

log = logging.getLogger("main")


class FaceRecognitionApp:
    """Composes and runs the full system."""

    def __init__(self, settings: Settings, cameras=None) -> None:
        self.settings = settings
        self.cameras = list(cameras) if cameras is not None else settings.enabled_cameras
        self.perf = PerformanceTracker()
        self.db = Database(settings.storage.db_path)
        init_schema(self.db)
        self.camera_repo = CameraRepository(self.db)
        self.person_repo = PersonRepository(self.db)
        self.embedding_repo = EmbeddingRepository(self.db)
        self.event_repo = EventRepository(self.db)

        self.inference = InferenceManager(settings.models, settings.detection, self.perf)
        self.gallery = EmbeddingGallery()
        self.identity = IdentityManager(self.person_repo, self.embedding_repo, self.gallery,
                                        settings.recognition.gallery_reload_s)
        self.events = EventService(self.event_repo, settings.storage, self.perf)
        self.health_registry = CameraHealthRegistry()
        self.scheduler = FrameScheduler(max_frame_age_s=settings.detection.max_frame_age_s)
        self.receivers: Dict[str, FrameReceiver] = {}
        self.ingestion = None
        self.recognition: Optional[RecognitionService] = None
        self.system_health: Optional[SystemHealth] = None
        self._stop = threading.Event()

    def _make_ingestion(self):
        backend = self.settings.ingestion.backend
        if backend in ("auto", "gstreamer"):
            from ingestion.gst_manager import GstManager, gstreamer_available

            if gstreamer_available():
                return GstManager(self.cameras, self.receivers, self.health_registry, self.settings.ingestion)
            if backend == "gstreamer":
                raise RuntimeError("GStreamer (PyGObject) not available - see README 'Install'")
            log.warning("GStreamer not available - falling back to OpenCV (higher CPU)")
        return OpenCVManager(self.cameras, self.receivers, self.health_registry, self.settings.ingestion)

    def start(self) -> None:
        s = self.settings
        if not self.cameras:
            raise RuntimeError("No enabled cameras in config/cameras.yaml")
        log.info("Starting with %d cameras, compute units=%s", len(self.cameras), s.models.compute_units)
        self.camera_repo.sync(s.cameras)

        self.inference.load()
        self.identity.reload()
        if self.gallery.size == 0:
            log.warning("Gallery is empty - every face will be 'unknown'. Run: python main.py enroll")

        matcher = FaceMatcher(self.gallery, s.recognition.match_threshold, s.recognition.match_margin)
        duplicates = DuplicateFilter(s.recognition.event_cooldown_s, s.recognition.unknown_cooldown_s,
                                     s.recognition.unknown_similarity)
        tracks = TrackCache(s.recognition.track_iou, s.recognition.track_reverify_s)
        self.events.start()

        for cam in self.cameras:
            q = self.scheduler.register(cam.camera_id, cam.priority, s.ingestion.queue_depth)
            h = self.health_registry.add(cam.camera_id, cam.name)
            self.receivers[cam.camera_id] = FrameReceiver(cam.camera_id, q, h)

        self.recognition = RecognitionService(
            self.inference, self.scheduler, self.identity, matcher, duplicates, tracks, self.events,
            self.cameras, s.detection, s.recognition, self.perf, s.storage.save_snapshots,
        )
        self.recognition.start()

        self.ingestion = self._make_ingestion()
        self.ingestion.start()

        self.system_health = SystemHealth(
            s.monitoring, self.perf, self.health_registry,
            extra={
                "scheduler": lambda: {k: v for k, v in self.scheduler.stats().items() if k != "per_camera"},
                "pipeline": self.recognition.stats,
                "gallery": lambda: {"persons": self.gallery.num_persons, "embeddings": self.gallery.size},
                "models": self.inference.info,
            },
        )
        self.system_health.start()
        log.info("System running - Ctrl+C to stop")

    def run_forever(self) -> None:
        signal.signal(signal.SIGINT, lambda *_: self._stop.set())
        signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
        self.start()
        try:
            while not self._stop.wait(1.0):
                pass
        finally:
            self.stop()

    def stop(self) -> None:
        log.info("Shutting down...")
        for component in (self.ingestion, self.recognition, self.system_health):
            if component is not None:
                try:
                    component.stop()
                except Exception:
                    log.exception("error stopping %s", type(component).__name__)
        self.events.stop()
        self.db.close()
        log.info("Stopped")


# ---------------------------------------------------------------------- commands
def _repos(settings: Settings):
    db = Database(settings.storage.db_path)
    init_schema(db)
    return db, PersonRepository(db), EmbeddingRepository(db), EventRepository(db)


def cmd_run(settings: Settings, args) -> int:
    cams = settings.enabled_cameras
    if args.cameras:
        wanted = set(args.cameras.split(","))
        cams = [c for c in settings.cameras if c.camera_id in wanted]
    FaceRecognitionApp(settings, cams).run_forever()
    return 0


def cmd_enroll(settings: Settings, args) -> int:
    db, persons, embeddings, _ = _repos(settings)
    inference = InferenceManager(settings.models, settings.detection).load()
    svc = EnrollmentService(inference, persons, embeddings)
    if args.name:
        if not args.images:
            print("--images is required with --name")
            return 2
        results = [svc.enroll_images(args.name, [Path(p) for p in args.images], args.external_id, args.replace)]
    else:
        results = svc.enroll_directory(Path(args.dir), replace=args.replace)
    for r in results:
        print(f"{r.name:<30} id={r.person_id:<5} added={r.added:<3} skipped={r.skipped:<3} errors={len(r.errors)}")
    db.close()
    return 0 if results else 1


def cmd_persons(settings: Settings, args) -> int:
    db, persons, embeddings, _ = _repos(settings)
    for p in persons.list(active_only=not args.all):
        print(f"{p['id']:>5}  {p['name']:<30} embeddings={embeddings.count_for(p['id']):<4} "
              f"active={bool(p['active'])}  external_id={p['external_id'] or ''}")
    db.close()
    return 0


def cmd_delete_person(settings: Settings, args) -> int:
    db, persons, _, _ = _repos(settings)
    ok = persons.delete(args.person_id)
    print("deleted" if ok else "not found")
    db.close()
    return 0 if ok else 1


def cmd_export(settings: Settings, args) -> int:
    db, _, _, events = _repos(settings)
    svc = EventService(events, settings.storage)
    path = svc.export_csv(Path(args.out) if args.out else None, parse_datetime(args.since),
                          parse_datetime(args.until), args.camera, args.known_only)
    print(path)
    db.close()
    return 0


def cmd_info(settings: Settings, args) -> int:
    inference = InferenceManager(settings.models, settings.detection).load()
    print(json.dumps(inference.info(), indent=2))
    print(f"cameras: {len(settings.cameras)} configured, {len(settings.enabled_cameras)} enabled")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", help="path to .env file")
    p.add_argument("--cameras-file", help="path to cameras.yaml")
    sub = p.add_subparsers(dest="command")

    r = sub.add_parser("run", help="run real-time recognition")
    r.add_argument("--cameras", help="comma-separated camera ids (default: all enabled)")

    e = sub.add_parser("enroll", help="enrol faces")
    e.add_argument("--dir", default=str(ENROLLMENT_DIR), help="folder with one sub-folder per person")
    e.add_argument("--name", help="person name (enrol specific images)")
    e.add_argument("--images", nargs="*", help="image files for --name")
    e.add_argument("--external-id", help="optional external/employee id")
    e.add_argument("--replace", action="store_true", help="replace existing embeddings")

    ps = sub.add_parser("persons", help="list persons")
    ps.add_argument("--all", action="store_true", help="include inactive")

    d = sub.add_parser("delete-person", help="delete a person and their embeddings")
    d.add_argument("person_id", type=int)

    x = sub.add_parser("export", help="export events to CSV")
    x.add_argument("--since")
    x.add_argument("--until")
    x.add_argument("--camera")
    x.add_argument("--known-only", action="store_true")
    x.add_argument("--out")

    sub.add_parser("info", help="show model/environment info")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(Path(args.env) if args.env else None,
                             Path(args.cameras_file) if args.cameras_file else None)
    setup_logging(settings.logging_config, settings.log_level)
    commands = {
        None: cmd_run, "run": cmd_run, "enroll": cmd_enroll, "persons": cmd_persons,
        "delete-person": cmd_delete_person, "export": cmd_export, "info": cmd_info,
    }
    if args.command is None:
        args.cameras = None
    return commands[args.command](settings, args)


if __name__ == "__main__":
    sys.exit(main())
