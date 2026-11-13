"""Real-time person detection + counting for cameras with `tasks: [person]`.

  FrameScheduler (person) --(batch)--> [person-detector thread]
      YOLOv7 (ANE) -> person filter / ROI -> tracker -> PersonCounter
      every FR_PERSON_FLUSH_S: one row per camera appended to `person_counts`

Runs alongside the face pipeline; both share the Neural Engine.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

from config.settings import CameraConfig, ModelSettings, PersonSettings
from constants import MODEL_CACHE_DIR
from services.live_view import LiveView
from database.person_count_repository import PersonCountRepository
from inference.coreml_loader import CoreMLModel
from monitoring.performance import PerformanceTracker
from person_detection.person_counter import CountRecord, PersonCounter
from person_detection.yolo_engine import YOLOv7Engine, filter_persons
from scheduler.batch_scheduler import BatchScheduler
from scheduler.frame_scheduler import FrameScheduler

log = logging.getLogger(__name__)


def load_yolo(models: ModelSettings, person: PersonSettings) -> YOLOv7Engine:
    cache = MODEL_CACHE_DIR if models.cache_compiled else None
    model = CoreMLModel(models.yolo_path, models.compute_units, cache, name="yolov7")
    return YOLOv7Engine(model, conf_threshold=person.conf_threshold, nms_threshold=person.nms_threshold,
                        class_id=person.class_id, num_classes=person.num_classes, input_std=person.input_std)


class PersonCountService:
    def __init__(self, engine: YOLOv7Engine, frames: FrameScheduler, repo: PersonCountRepository,
                 cameras: List[CameraConfig], settings: PersonSettings,
                 perf: Optional[PerformanceTracker] = None, retention_days: int = 0,
                 live: Optional[LiveView] = None) -> None:
        self.engine = engine
        self.frames = frames
        self.repo = repo
        self.cameras: Dict[str, CameraConfig] = {c.camera_id: c for c in cameras}
        self.settings = settings
        self.perf = perf or PerformanceTracker()
        self.retention_days = retention_days
        self.live = live
        self._last_retention = 0.0
        self.counter = PersonCounter(self.cameras, settings.flush_interval_s, settings.track_iou,
                                     settings.track_max_age_s, settings.track_min_hits)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ lifecycle
    def warmup(self, iterations: int = 3) -> None:
        import numpy as np

        w, h = self.engine.input_size
        t0 = time.perf_counter()
        for _ in range(max(0, iterations)):
            self.engine.detect_batch([np.zeros((h, w, 3), np.uint8)] * self.settings.batch_size)
        if iterations:
            log.info("YOLOv7 warm-up done in %.2fs", time.perf_counter() - t0)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="person-detector", daemon=True)
        self._thread.start()
        log.info("Person counting started on %d cameras (batch %d, DB row every %.0fs)",
                 len(self.cameras), self.settings.batch_size, self.settings.flush_interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        self._write(self.counter.flush())  # keep the partial last interval

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        batcher = BatchScheduler(self.frames.get_frames, self.settings.batch_size, self.settings.max_wait_ms)
        while not self._stop.is_set():
            if self.counter.due():
                self._write(self.counter.flush())
                self._apply_retention()
            frames = batcher.next_batch(self._stop)
            if not frames:
                continue
            t0 = time.perf_counter()
            try:
                results = self.engine.detect_batch([f.image for f in frames])
            except Exception:
                log.exception("Person detection failed")
                time.sleep(0.5)
                continue
            dt = time.perf_counter() - t0
            self.perf.record_latency("yolo_batch_ms", dt * 1000.0)
            self.perf.record_latency("yolo_per_frame_ms", dt * 1000.0 / len(frames))
            self.perf.add_busy("inference", dt)
            self.perf.add_busy("yolo", dt)
            self.perf.incr("person_frames_processed", len(frames))
            for frame, dets in zip(frames, results):
                cam = self.cameras.get(frame.camera_id)
                dets = filter_persons(dets, self.settings.min_box_height, cam.roi if cam else None)
                self.perf.incr("persons_detected", int(dets.shape[0]))
                new = self.counter.update(frame.camera_id, dets, frame.timestamp, frame.mono_ts)
                if new:
                    self.perf.incr("persons_new", new)
                if self.live is not None:
                    self.live.publish_persons(frame.camera_id, frame.image, frame.timestamp, dets)

    def _write(self, records: List[CountRecord]) -> None:
        if not records:
            return
        try:
            self.repo.insert_many(r.as_row() for r in records)
            self.perf.incr("person_count_rows", len(records))
            log.debug("Wrote %d person-count rows", len(records))
        except Exception:
            log.exception("Failed to write %d person-count rows", len(records))

    def _apply_retention(self) -> None:
        now = time.monotonic()
        if self.retention_days <= 0 or now - self._last_retention < 3600:
            return
        self._last_retention = now
        try:
            n = self.repo.delete_older_than(time.time() - self.retention_days * 86400)
            if n:
                log.info("Retention: deleted %d person-count rows older than %d days", n, self.retention_days)
        except Exception:
            log.exception("Person-count retention cleanup failed")

    def stats(self) -> dict:
        return self.counter.stats()
