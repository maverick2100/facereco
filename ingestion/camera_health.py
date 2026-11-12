"""Per-camera health tracking (fps, last frame, reconnects, errors)."""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

STARTING = "starting"
STREAMING = "streaming"
RECONNECTING = "reconnecting"
STALE = "stale"
ERROR = "error"
STOPPED = "stopped"


class CameraHealth:
    def __init__(self, camera_id: str, name: str = "") -> None:
        self.camera_id = camera_id
        self.name = name or camera_id
        self.status = STARTING
        self.frames = 0
        self.reconnects = 0
        self.errors = 0
        self.last_error: Optional[str] = None
        self.last_frame_mono: Optional[float] = None
        self.started_mono = time.monotonic()
        self._fps_ema = 0.0
        self._lock = threading.Lock()

    def on_frame(self) -> None:
        t = time.monotonic()
        with self._lock:
            if self.last_frame_mono is not None:
                dt = t - self.last_frame_mono
                if dt > 0:
                    inst = 1.0 / dt
                    self._fps_ema = inst if self._fps_ema == 0 else 0.9 * self._fps_ema + 0.1 * inst
            self.last_frame_mono = t
            self.frames += 1
            self.status = STREAMING

    def on_error(self, message: str) -> None:
        with self._lock:
            self.errors += 1
            self.last_error = message[:500]
            self.status = ERROR

    def on_reconnect(self, reason: str = "") -> None:
        with self._lock:
            self.reconnects += 1
            self.status = RECONNECTING
            if reason:
                self.last_error = reason[:500]
            self._fps_ema = 0.0

    def set_status(self, status: str) -> None:
        with self._lock:
            self.status = status

    def seconds_since_frame(self) -> Optional[float]:
        if self.last_frame_mono is None:
            return None
        return time.monotonic() - self.last_frame_mono

    def is_stale(self, timeout_s: float) -> bool:
        since = self.seconds_since_frame()
        if since is None:
            return time.monotonic() - self.started_mono > timeout_s
        return since > timeout_s

    def mark_restarted(self) -> None:
        with self._lock:
            self.started_mono = time.monotonic()
            self.last_frame_mono = None

    def snapshot(self) -> dict:
        since = self.seconds_since_frame()
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "status": self.status,
            "fps": round(self._fps_ema, 2),
            "frames": self.frames,
            "reconnects": self.reconnects,
            "errors": self.errors,
            "last_error": self.last_error,
            "seconds_since_frame": None if since is None else round(since, 1),
        }


class CameraHealthRegistry:
    def __init__(self) -> None:
        self._items: Dict[str, CameraHealth] = {}

    def add(self, camera_id: str, name: str = "") -> CameraHealth:
        h = CameraHealth(camera_id, name)
        self._items[camera_id] = h
        return h

    def get(self, camera_id: str) -> CameraHealth:
        return self._items[camera_id]

    def all(self) -> List[CameraHealth]:
        return list(self._items.values())

    def snapshot(self) -> List[dict]:
        return [h.snapshot() for h in self._items.values()]

    def summary(self) -> dict:
        snaps = self.snapshot()
        streaming = sum(1 for s in snaps if s["status"] == STREAMING)
        return {
            "total": len(snaps),
            "streaming": streaming,
            "down": [s["camera_id"] for s in snaps if s["status"] != STREAMING],
            "total_fps": round(sum(s["fps"] for s in snaps), 1),
        }
