"""Per-camera 'latest frame wins' slot.

Each camera keeps at most `maxlen` frames (default 1). When inference is slower
than the cameras, old frames are dropped instead of building a backlog, so
latency and memory stay bounded no matter how many cameras are attached.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Optional

import numpy as np


@dataclass
class Frame:
    camera_id: str
    frame_id: int
    image: np.ndarray        # BGR uint8 (H, W, 3); treat as read-only
    timestamp: float         # wall clock (epoch seconds)
    mono_ts: float           # monotonic seconds at arrival
    meta: dict = field(default_factory=dict)


class CameraQueue:
    def __init__(self, camera_id: str, maxlen: int = 1, on_put: Optional[Callable[[], None]] = None) -> None:
        self.camera_id = camera_id
        self.maxlen = max(1, int(maxlen))
        self._dq: Deque[Frame] = deque(maxlen=self.maxlen)
        self._lock = threading.Lock()
        self._on_put = on_put
        self.received = 0
        self.dropped = 0

    def put(self, frame: Frame) -> None:
        with self._lock:
            if len(self._dq) == self.maxlen:
                self.dropped += 1
            self._dq.append(frame)
            self.received += 1
        if self._on_put is not None:
            self._on_put()

    def pop_latest(self) -> Optional[Frame]:
        """Return the newest frame and discard any older ones."""
        with self._lock:
            if not self._dq:
                return None
            frame = self._dq.pop()
            self.dropped += len(self._dq)
            self._dq.clear()
            return frame

    def has_frame(self) -> bool:
        return bool(self._dq)

    def __len__(self) -> int:
        return len(self._dq)

    def stats(self) -> dict:
        return {"received": self.received, "dropped": self.dropped, "pending": len(self._dq)}
