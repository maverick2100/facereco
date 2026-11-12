"""Micro-batching: gather up to `max_batch` items but never wait more than `max_wait_ms`.

Batching amortises Core ML dispatch overhead and keeps the Neural Engine busy
with fewer, larger requests (less CPU time spent per inference).
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Generic, List, Optional, TypeVar

T = TypeVar("T")

FetchFn = Callable[[int, float], List[T]]  # (max_n, timeout_s) -> items


class BatchScheduler(Generic[T]):
    def __init__(self, fetch: FetchFn, max_batch: int, max_wait_ms: float, idle_timeout_s: float = 0.2) -> None:
        self._fetch = fetch
        self.max_batch = max(1, int(max_batch))
        self.max_wait_s = max(0.0, max_wait_ms / 1000.0)
        self.idle_timeout_s = idle_timeout_s
        self.batches = 0
        self.items = 0

    def next_batch(self, stop_event: Optional[threading.Event] = None) -> List[T]:
        items = self._fetch(self.max_batch, self.idle_timeout_s)
        if not items:
            return []
        deadline = time.monotonic() + self.max_wait_s
        while len(items) < self.max_batch:
            if stop_event is not None and stop_event.is_set():
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            more = self._fetch(self.max_batch - len(items), remaining)
            if not more:
                break
            items.extend(more)
        self.batches += 1
        self.items += len(items)
        return items

    @property
    def mean_batch_size(self) -> float:
        return self.items / self.batches if self.batches else 0.0
