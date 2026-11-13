#!/usr/bin/env python3
"""Multi-camera face recognition + person counting for Apple Silicon (Neural Engine).

Each camera's `tasks` in config/cameras.yaml picks what runs on it:
    face   -> SCRFD + ArcFace recognition      person -> YOLOv7 person detection + counting

Usage:
    python main.py run                       # start all enabled cameras
    python main.py enroll                    # enrol everyone in data/enrollment/<name>/*.jpg
    python main.py enroll --name "Alice" --images a.jpg b.jpg
    python main.py persons                   # list enrolled people
    python main.py delete-person 3
    python main.py export --since 2026-11-13 # CSV into data/exports/
    python main.py counts --since 2026-11-13 # person totals per camera (--csv for all rows)
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
import csv  # noqa: E402
import time  # noqa: E402
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
from constants import ENROLLMENT_DIR, EXPORTS_DIR  # noqa: E402
from database.camera_repository import CameraRepository  # noqa: E402
from database.connection import Database  # noqa: E402
from database.embedding_repository import EmbeddingRepository  # noqa: E402
from database.event_repository import EventRepository  # noqa: E402
from database.person_count_repository import PersonCountRepository  # noqa: E402
from database.person_repository import PersonRepository  # noqa: E402
from database.schema import init_schema  # noqa: E402
from ingestion.camera_health import CameraHealthRegistry  # noqa: E402
from ingestion.frame_receiver import FrameReceiver, OpenCVManager  # noqa: E402
from inference.inference_manager import InferenceManager  # noqa: E402
from monitoring.performance import PerformanceTracker  # noqa: E402
from monitoring.system_health import SystemHealth  # noqa: E402
from person_detection.person_count_service import PersonCountService, load_yolo  # noqa: E402
from recognition.duplicate_filter import DuplicateFilter, TrackCache  # noqa: E402
from recognition.embedding_manager import EmbeddingGallery  # noqa: E402
from recognition.face_matcher import FaceMatcher  # noqa: E402
from recognition.identity_manager import IdentityManager  # noqa: E402
from scheduler.frame_scheduler import FrameScheduler  # noqa: E402
from services.enrollment_service import EnrollmentService  # noqa: E402
from services.event_service import EventService  # noqa: E402
from services.recognition_service import RecognitionService  # noqa: E402
from utils.logger import setup_logging  # noqa: E402
from utils.time_utils import iso, parse_datetime  # noqa: E402

log = logging.getLogger("main")


class FaceRecognitionApp:
    """Composes and runs the full system."""

    def __init__(self, settings: Settings, cameras=None) -> None:
        self.settings = settings
        self.cameras = list(cameras) if cameras is not None else settings.enabled_cameras
        self.face_cameras = [c for c in self.cameras if c.runs_face]
        self.person_cameras = [c for c in self.cameras if c.runs_person]
        self.perf = PerformanceTracker()
        self.db = Database(settings.storage.db_path)
        init_schema(self.db)
        self.camera_repo = CameraRepository(self.db)
        self.person_repo = PersonRepository(self.db)
        self.embedding_repo = EmbeddingRepository(self.db)
        self.event_repo = EventRepository(self.db)
        self.person_count_repo = PersonCountRepository(self.db)

        self.inference = InferenceManager(settings.models, settings.detection, self.perf)
        self.gallery = EmbeddingGallery()
        self.identity = IdentityManager(self.person_repo, self.embedding_repo, self.gallery,
                                        settings.recognition.gallery_reload_s)
        self.events = EventService(self.event_repo, settings.storage, self.perf)
        self.health_registry = CameraHealthRegistry()
        self.scheduler = FrameScheduler(max_frame_age_s=settings.detection.max_frame_age_s)
        person_fps = settings.person.max_fps
        self.person_scheduler = FrameScheduler(max_frame_age_s=settings.detection.max_frame_age_s,
                                               min_interval_s=1.0 / person_fps if person_fps > 0 else 0.0)
        self.receivers: Dict[str, FrameReceiver] = {}
        self.ingestion = None
        self.recognition: Optional[RecognitionService] = None
        self.person_service: Optional[PersonCountService] = None
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
        log.info("Starting with %d cameras (face: %d, person: %d), compute units=%s", len(self.cameras),
                 len(self.face_cameras), len(self.person_cameras), s.models.compute_units)
        self.camera_repo.sync(s.cameras)

        if self.face_cameras:
            self.inference.load()
            self.identity.reload()
            if self.gallery.size == 0:
                log.warning("Gallery is empty - every face will be 'unknown'. Run: python main.py enroll")
        if self.person_cameras:
            self.person_service = PersonCountService(
                load_yolo(s.models, s.person), self.person_scheduler, self.person_count_repo,
                self.person_cameras, s.person, self.perf, retention_days=s.storage.retention_days,
            )
            self.person_service.warmup(s.models.warmup_iterations)

        for cam in self.cameras:
            queues = []
            if cam.runs_face:
                queues.append(self.scheduler.register(cam.camera_id, cam.priority, s.ingestion.queue_depth))
            if cam.runs_person:
                queues.append(self.person_scheduler.register(cam.camera_id, cam.priority, s.ingestion.queue_depth))
            h = self.health_registry.add(cam.camera_id, cam.name)
            self.receivers[cam.camera_id] = FrameReceiver(cam.camera_id, queues, h)

        if self.face_cameras:
            matcher = FaceMatcher(self.gallery, s.recognition.match_threshold, s.recognition.match_margin)
            duplicates = DuplicateFilter(s.recognition.event_cooldown_s, s.recognition.unknown_cooldown_s,
                                         s.recognition.unknown_similarity)
            tracks = TrackCache(s.recognition.track_iou, s.recognition.track_reverify_s)
            self.events.start()
            self.recognition = RecognitionService(
                self.inference, self.scheduler, self.identity, matcher, duplicates, tracks, self.events,
                self.face_cameras, s.detection, s.recognition, self.perf, s.storage.save_snapshots,
            )
            self.recognition.start()
        if self.person_service is not None:
            self.person_service.start()

        self.ingestion = self._make_ingestion()
        self.ingestion.start()

        extra = {"models": self._models_info}
        if self.recognition is not None:
            extra.update({
                "scheduler": lambda: {k: v for k, v in self.scheduler.stats().items() if k != "per_camera"},
                "pipeline": self.recognition.stats,
                "gallery": lambda: {"persons": self.gallery.num_persons, "embeddings": self.gallery.size},
            })
        if self.person_service is not None:
            extra.update({
                "person_scheduler": lambda: {k: v for k, v in self.person_scheduler.stats().items()
                                             if k != "per_camera"},
                "person_counts": self.person_service.stats,
            })
        self.system_health = SystemHealth(s.monitoring, self.perf, self.health_registry, extra=extra)
        self.system_health.start()
        log.info("System running - Ctrl+C to stop")

    def _models_info(self) -> dict:
        info = self.inference.info()
        if self.person_service is not None:
            info["yolov7"] = self.person_service.engine.model.info()
        return info

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
        for component in (self.ingestion, self.recognition, self.person_service, self.system_health):
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


def cmd_counts(settings: Settings, args) -> int:
    db = Database(settings.storage.db_path)
    init_schema(db)
    repo = PersonCountRepository(db)
    since, until = parse_datetime(args.since), parse_datetime(args.until)
    if args.csv is not None:
        rows = repo.query(since, until, args.camera, limit=0)
        path = Path(args.csv) if args.csv else EXPORTS_DIR / f"person_counts_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["period_start_time", "period_end_time", "camera_id", "camera_name", "camera_location", "frames",
                "current_count", "min_count", "max_count", "avg_count", "new_persons"]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in reversed(rows):
                r["period_start_time"], r["period_end_time"] = iso(r["period_start"]), iso(r["period_end"])
                w.writerow(r)
        print(path)
    else:
        totals = repo.totals(since, until, args.camera)
        print(f"{'camera':<12} {'name':<24} {'persons':>8} {'peak':>6} {'frames':>9}  from -> to")
        for t in totals:
            print(f"{t['camera_id']:<12} {(t['camera_name'] or ''):<24} {t['total_persons']:>8} "
                  f"{t['peak_count']:>6} {t['frames']:>9}  {iso(t['first_ts'])} -> {iso(t['last_ts'])}")
        print(f"{'TOTAL':<12} {'':<24} {sum(t['total_persons'] for t in totals):>8}")
    db.close()
    return 0


def cmd_info(settings: Settings, args) -> int:
    if settings.face_cameras or not settings.person_cameras:
        inference = InferenceManager(settings.models, settings.detection).load()
        print(json.dumps(inference.info(), indent=2))
    if settings.person_cameras or settings.models.yolo_path.exists():
        print(json.dumps({"yolov7": load_yolo(settings.models, settings.person).model.info()}, indent=2))
    print(f"cameras: {len(settings.cameras)} configured, {len(settings.enabled_cameras)} enabled "
          f"(face: {len(settings.face_cameras)}, person: {len(settings.person_cameras)})")
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

    c = sub.add_parser("counts", help="person-count totals per camera (or --csv for every row)")
    c.add_argument("--since")
    c.add_argument("--until")
    c.add_argument("--camera")
    c.add_argument("--csv", nargs="?", const="", help="export rows to CSV (optional output path)")

    sub.add_parser("info", help="show model/environment info")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(Path(args.env) if args.env else None,
                             Path(args.cameras_file) if args.cameras_file else None)
    setup_logging(settings.logging_config, settings.log_level)
    commands = {
        None: cmd_run, "run": cmd_run, "enroll": cmd_enroll, "persons": cmd_persons,
        "delete-person": cmd_delete_person, "export": cmd_export, "counts": cmd_counts, "info": cmd_info,
    }
    if args.command is None:
        args.cameras = None
    return commands[args.command](settings, args)


if __name__ == "__main__":
    sys.exit(main())
