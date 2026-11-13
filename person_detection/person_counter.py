"""Per-camera count aggregation.

Each analysed frame updates the camera's current interval; every
`interval_s` the interval is closed into one `CountRecord` (one DB row).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from person_detection.person_tracker import PersonTracker


@dataclass
class CountRecord:
    camera_id: str
    period_start: float      # epoch seconds
    period_end: float
    frames: int
    current_count: int
    min_count: int
    max_count: int
    avg_count: float
    new_persons: int

    def as_row(self) -> Tuple:
        return (self.camera_id, self.period_start, self.period_end, self.frames, self.current_count,
                self.min_count, self.max_count, round(self.avg_count, 3), self.new_persons, self.period_end)


class _Interval:
    def __init__(self, start: float) -> None:
        self.start = start
        self.end = start
        self.frames = 0
        self.total = 0
        self.min = 0
        self.max = 0
        self.last = 0
        self.new = 0

    def add(self, count: int, new: int, ts: float) -> None:
        self.min = count if self.frames == 0 else min(self.min, count)
        self.max = max(self.max, count)
        self.frames += 1
        self.total += count
        self.last = count
        self.new += new
        self.end = max(self.end, ts)


class CameraCounter:
    def __init__(self, camera_id: str, tracker: PersonTracker) -> None:
        self.camera_id = camera_id
        self.tracker = tracker
        self.current = 0
        self.total_persons = 0     # since start-up
        self.peak = 0
        self.last_update: Optional[float] = None
        self._interval: Optional[_Interval] = None

    def update(self, boxes: np.ndarray, ts: float, mono_ts: float) -> int:
        new = self.tracker.update(boxes, mono_ts)
        count = int(boxes.shape[0])
        if self._interval is None:
            self._interval = _Interval(ts)
        self._interval.add(count, new, ts)
        self.current = count
        self.total_persons += new
        self.peak = max(self.peak, count)
        self.last_update = ts
        return new

    def close(self, now: float) -> Optional[CountRecord]:
        iv, self._interval = self._interval, None
        if iv is None or iv.frames == 0:
            return None
        return CountRecord(self.camera_id, iv.start, max(iv.end, now), iv.frames, iv.last, iv.min, iv.max,
                           iv.total / iv.frames, iv.new)

    def stats(self) -> dict:
        return {"current": self.current, "total_persons": self.total_persons, "peak": self.peak,
                "tracks": self.tracker.active, "last_update": self.last_update}


class PersonCounter:
    """All cameras' counters plus the flush clock."""

    def __init__(self, camera_ids, interval_s: float = 60.0, track_iou: float = 0.3,
                 track_max_age_s: float = 2.0, track_min_hits: int = 2) -> None:
        self.interval_s = interval_s
        self._lock = threading.Lock()
        self.cameras: Dict[str, CameraCounter] = {
            cid: CameraCounter(cid, PersonTracker(track_iou, track_max_age_s, track_min_hits)) for cid in camera_ids
        }
        self._last_flush = time.monotonic()

    def update(self, camera_id: str, boxes: np.ndarray, ts: float, mono_ts: float) -> int:
        with self._lock:
            return self.cameras[camera_id].update(boxes, ts, mono_ts)

    def due(self) -> bool:
        return time.monotonic() - self._last_flush >= self.interval_s

    def flush(self) -> List[CountRecord]:
        now = time.time()
        with self._lock:
            self._last_flush = time.monotonic()
            return [r for r in (c.close(now) for c in self.cameras.values()) if r is not None]

    def stats(self) -> dict:
        with self._lock:
            per = {cid: c.stats() for cid, c in self.cameras.items()}
        return {
            "current_total": sum(p["current"] for p in per.values()),
            "persons_since_start": sum(p["total_persons"] for p in per.values()),
            "per_camera": per,
        }
