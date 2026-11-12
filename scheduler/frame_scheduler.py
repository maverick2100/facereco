"""Fair frame scheduling across many cameras.

The detector pulls frames from here. Cameras that have waited longest
(weighted by priority) are served first, so a single busy camera can never
starve the other 19. Frames older than `max_frame_age_s` are discarded.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List

from scheduler.camera_queue import CameraQueue, Frame


class FrameScheduler:
    def __init__(self, max_frame_age_s: float = 1.0, min_interval_s: float = 0.0) -> None:
        self.max_frame_age_s = max_frame_age_s
        self.min_interval_s = min_interval_s  # optional per-camera rate cap (0 = rely on GStreamer)
        self._cond = threading.Condition()
        self._queues: Dict[str, CameraQueue] = {}
        self._priority: Dict[str, float] = {}
        self._last_served: Dict[str, float] = {}
        self.stale_dropped = 0
        self.served = 0

    # ------------------------------------------------------------------ setup
    def register(self, camera_id: str, priority: float = 1.0, depth: int = 1) -> CameraQueue:
        q = CameraQueue(camera_id, depth, on_put=self._notify)
        with self._cond:
            self._queues[camera_id] = q
            self._priority[camera_id] = max(0.01, float(priority))
            self._last_served[camera_id] = 0.0
        return q

    def queue(self, camera_id: str) -> CameraQueue:
        return self._queues[camera_id]

    @property
    def camera_ids(self) -> List[str]:
        return list(self._queues)

    def _notify(self) -> None:
        with self._cond:
            self._cond.notify()

    # ------------------------------------------------------------------ consume
    def _ready(self, now: float) -> List[str]:
        return [
            cid
            for cid, q in self._queues.items()
            if q.has_frame() and (now - self._last_served[cid]) >= self.min_interval_s
        ]

    def get_frames(self, max_n: int, timeout: float) -> List[Frame]:
        """Return up to `max_n` frames (at most one per camera), waiting up to `timeout`."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while True:
                now = time.monotonic()
                ready = self._ready(now)
                if ready:
                    break
                remaining = deadline - now
                if remaining <= 0:
                    return []
                # Short wait: a rate-limited camera may become eligible without a notify.
                self._cond.wait(min(remaining, 0.02) if self.min_interval_s > 0 else remaining)

        now = time.monotonic()
        ready.sort(key=lambda c: (now - self._last_served[c]) * self._priority[c], reverse=True)
        out: List[Frame] = []
        for cid in ready:
            frame = self._queues[cid].pop_latest()
            if frame is None:
                continue
            if now - frame.mono_ts > self.max_frame_age_s:
                self.stale_dropped += 1
                continue
            self._last_served[cid] = now
            out.append(frame)
            if len(out) >= max_n:
                break
        self.served += len(out)
        return out

    def stats(self) -> dict:
        return {
            "served": self.served,
            "stale_dropped": self.stale_dropped,
            "dropped_total": sum(q.dropped for q in self._queues.values()),
            "per_camera": {cid: q.stats() for cid, q in self._queues.items()},
        }
