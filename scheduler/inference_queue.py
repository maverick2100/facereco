"""Bounded, thread-safe queue that drops the *oldest* item when full."""
from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Generic, List, TypeVar

T = TypeVar("T")


class InferenceQueue(Generic[T]):
    def __init__(self, maxsize: int = 256) -> None:
        self.maxsize = max(1, int(maxsize))
        self._dq: Deque[T] = deque()
        self._cond = threading.Condition()
        self.dropped = 0
        self.enqueued = 0

    def put(self, item: T) -> bool:
        """Enqueue; returns False if an old item had to be dropped to make room."""
        ok = True
        with self._cond:
            if len(self._dq) >= self.maxsize:
                self._dq.popleft()
                self.dropped += 1
                ok = False
            self._dq.append(item)
            self.enqueued += 1
            self._cond.notify()
        return ok

    def get_batch(self, max_n: int, timeout: float) -> List[T]:
        with self._cond:
            if not self._dq and timeout > 0:
                self._cond.wait(timeout)
            out: List[T] = []
            while self._dq and len(out) < max_n:
                out.append(self._dq.popleft())
            return out

    def __len__(self) -> int:
        return len(self._dq)

    def stats(self) -> dict:
        return {"pending": len(self._dq), "enqueued": self.enqueued, "dropped": self.dropped}
