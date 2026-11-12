"""Asynchronous event sink: batches DB inserts, writes snapshots, notifies listeners."""
from __future__ import annotations

import csv
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from config.settings import StorageSettings
from constants import EVENT_LOGGER_NAME, EXPORTS_DIR, SNAPSHOTS_DIR
from database.event_repository import EventRepository
from monitoring.performance import PerformanceTracker
from utils.image_utils import save_jpeg
from utils.time_utils import date_folder, iso

log = logging.getLogger(__name__)
event_log = logging.getLogger(EVENT_LOGGER_NAME)


@dataclass
class RecognitionEvent:
    camera_id: str
    person_id: Optional[int]          # None for unknown
    label: str
    similarity: float
    det_score: float
    quality: float
    bbox: Sequence[int]
    ts: float
    snapshot: Optional[np.ndarray] = None
    snapshot_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "person_id": self.person_id,
            "label": self.label,
            "similarity": round(self.similarity, 4),
            "det_score": round(self.det_score, 4),
            "quality": round(self.quality, 4),
            "bbox": list(self.bbox),
            "ts": self.ts,
            "time": iso(self.ts),
            "snapshot_path": self.snapshot_path,
        }


Listener = Callable[[List[RecognitionEvent]], None]


class EventService:
    def __init__(self, repo: EventRepository, settings: StorageSettings,
                 perf: Optional[PerformanceTracker] = None) -> None:
        self.repo = repo
        self.settings = settings
        self.perf = perf or PerformanceTracker()
        self._q: "queue.Queue[RecognitionEvent]" = queue.Queue(maxsize=settings.event_queue_size)
        self._listeners: List[Listener] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_retention = 0.0

    def add_listener(self, fn: Listener) -> None:
        """Hook for integrations (webhooks, MQTT, door relays...). Called from the writer thread."""
        self._listeners.append(fn)

    def submit(self, event: RecognitionEvent) -> bool:
        try:
            self._q.put_nowait(event)
            return True
        except queue.Full:
            self.perf.incr("events_dropped")
            return False

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="event-writer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        batch: List[RecognitionEvent] = []
        last_flush = time.monotonic()
        while not (self._stop.is_set() and self._q.empty()):
            try:
                batch.append(self._q.get(timeout=0.2))
            except queue.Empty:
                pass
            due = time.monotonic() - last_flush >= self.settings.event_flush_interval_s
            if batch and (len(batch) >= self.settings.event_batch_size or due or self._stop.is_set()):
                self._flush(batch)
                batch = []
                last_flush = time.monotonic()
            self._apply_retention()
        if batch:
            self._flush(batch)

    def _flush(self, batch: List[RecognitionEvent]) -> None:
        for ev in batch:
            if ev.snapshot is not None and self.settings.save_snapshots:
                name = f"{time.strftime('%H%M%S', time.localtime(ev.ts))}_{int((ev.ts % 1) * 1000):03d}_{ev.label}.jpg"
                path = SNAPSHOTS_DIR / date_folder(ev.ts) / ev.camera_id / name.replace(" ", "_")
                if save_jpeg(path, ev.snapshot, self.settings.snapshot_quality):
                    ev.snapshot_path = str(path)
            ev.snapshot = None  # free memory early
        try:
            self.repo.insert_many(
                (ev.camera_id, ev.person_id, ev.label, ev.similarity, ev.det_score, ev.quality, list(ev.bbox),
                 ev.snapshot_path, ev.ts) for ev in batch
            )
            self.perf.incr("events_written", len(batch))
        except Exception:
            log.exception("Failed to write %d events", len(batch))
        for ev in batch:
            event_log.info(json.dumps(ev.to_dict()))
        for fn in self._listeners:
            try:
                fn(batch)
            except Exception:
                log.exception("Event listener failed")

    def _apply_retention(self) -> None:
        if self.settings.retention_days <= 0:
            return
        now = time.monotonic()
        if now - self._last_retention < 3600:
            return
        self._last_retention = now
        try:
            n = self.repo.delete_older_than(time.time() - self.settings.retention_days * 86400)
            if n:
                log.info("Retention: deleted %d events older than %d days", n, self.settings.retention_days)
        except Exception:
            log.exception("Retention cleanup failed")

    # ------------------------------------------------------------------ export
    def export_csv(self, path: Optional[Path] = None, since: Optional[float] = None, until: Optional[float] = None,
                   camera_id: Optional[str] = None, known_only: bool = False) -> Path:
        rows = self.repo.query(since=since, until=until, camera_id=camera_id, known_only=known_only, limit=0)
        if path is None:
            path = EXPORTS_DIR / f"events_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["time", "ts", "camera_id", "camera_name", "camera_location", "person_id", "label",
                "similarity", "det_score", "quality", "bbox", "snapshot_path"]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in reversed(rows):
                r["time"] = iso(r["ts"])
                w.writerow(r)
        log.info("Exported %d events to %s", len(rows), path)
        return path
