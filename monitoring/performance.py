"""Lightweight, thread-safe performance counters."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, Tuple

import numpy as np


class LatencyStat:
    def __init__(self, window: int = 512) -> None:
        self._values: Deque[float] = deque(maxlen=window)
        self.count = 0
        self.total = 0.0
        self._lock = threading.Lock()

    def record(self, ms: float) -> None:
        with self._lock:
            self._values.append(ms)
            self.count += 1
            self.total += ms

    def snapshot(self) -> dict:
        with self._lock:
            vals = np.fromiter(self._values, dtype=np.float64) if self._values else None
        if vals is None:
            return {"count": self.count}
        return {
            "count": self.count,
            "mean_ms": round(float(vals.mean()), 2),
            "p50_ms": round(float(np.percentile(vals, 50)), 2),
            "p95_ms": round(float(np.percentile(vals, 95)), 2),
            "max_ms": round(float(vals.max()), 2),
        }


class RateCounter:
    """Total count + rate over a sliding window."""

    def __init__(self, window_s: float = 10.0) -> None:
        self.window_s = window_s
        self.total = 0
        self._events: Deque[Tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def add(self, n: int = 1) -> None:
        t = time.monotonic()
        with self._lock:
            self.total += n
            self._events.append((t, n))
            self._trim(t)

    def _trim(self, t: float) -> None:
        cutoff = t - self.window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def rate(self) -> float:
        t = time.monotonic()
        with self._lock:
            self._trim(t)
            return sum(n for _, n in self._events) / self.window_s


class BusyTracker:
    """Fraction of wall time a resource was busy over a sliding window."""

    def __init__(self, window_s: float = 10.0) -> None:
        self.window_s = window_s
        self._spans: Deque[Tuple[float, float]] = deque()
        self._lock = threading.Lock()

    def add(self, seconds: float) -> None:
        t = time.monotonic()
        with self._lock:
            self._spans.append((t, seconds))
            cutoff = t - self.window_s
            while self._spans and self._spans[0][0] < cutoff:
                self._spans.popleft()

    def utilization(self) -> float:
        t = time.monotonic()
        with self._lock:
            busy = sum(s for ts, s in self._spans if ts >= t - self.window_s)
        return min(1.0, busy / self.window_s)


class PerformanceTracker:
    def __init__(self, window_s: float = 10.0) -> None:
        self.window_s = window_s
        self._counters: Dict[str, RateCounter] = {}
        self._latency: Dict[str, LatencyStat] = {}
        self._busy: Dict[str, BusyTracker] = {}
        self._lock = threading.Lock()
        self.started = time.monotonic()

    def incr(self, name: str, n: int = 1) -> None:
        c = self._counters.get(name)
        if c is None:
            with self._lock:
                c = self._counters.setdefault(name, RateCounter(self.window_s))
        c.add(n)

    def record_latency(self, name: str, ms: float) -> None:
        s = self._latency.get(name)
        if s is None:
            with self._lock:
                s = self._latency.setdefault(name, LatencyStat())
        s.record(ms)

    def add_busy(self, resource: str, seconds: float) -> None:
        b = self._busy.get(resource)
        if b is None:
            with self._lock:
                b = self._busy.setdefault(resource, BusyTracker(self.window_s))
        b.add(seconds)

    def utilization(self, resource: str) -> float:
        b = self._busy.get(resource)
        return b.utilization() if b else 0.0

    def count(self, name: str) -> int:
        c = self._counters.get(name)
        return c.total if c else 0

    def rate(self, name: str) -> float:
        c = self._counters.get(name)
        return c.rate() if c else 0.0

    def snapshot(self) -> dict:
        return {
            "uptime_s": round(time.monotonic() - self.started, 1),
            "counters": {k: {"total": c.total, "per_s": round(c.rate(), 2)} for k, c in sorted(self._counters.items())},
            "latency": {k: s.snapshot() for k, s in sorted(self._latency.items())},
            "utilization_pct": {k: round(b.utilization() * 100, 1) for k, b in sorted(self._busy.items())},
        }
