"""Time helpers."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional, Union


def now() -> float:
    """Wall-clock epoch seconds (for persisted timestamps)."""
    return time.time()


def mono() -> float:
    """Monotonic seconds (for intervals / cooldowns)."""
    return time.monotonic()


def iso(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(time.time() if ts is None else ts).astimezone().isoformat(timespec="milliseconds")


def date_folder(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(time.time() if ts is None else ts).strftime("%Y-%m-%d")


def parse_datetime(value: Union[str, float, int, None]) -> Optional[float]:
    """Parse epoch seconds or ISO-8601 ('2026-11-13', '2026-11-13T08:00') to epoch seconds."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except ValueError:
        pass
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.astimezone()  # interpret as local time
    return dt.timestamp()


class Stopwatch:
    """`with Stopwatch() as sw: ...; sw.ms`"""

    def __init__(self) -> None:
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Stopwatch":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self.start

    @property
    def ms(self) -> float:
        return self.elapsed * 1000.0


class RateLimiter:
    """Returns True from `ready()` at most once every `interval_s`."""

    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self._last = 0.0

    def ready(self) -> bool:
        t = time.monotonic()
        if t - self._last >= self.interval_s:
            self._last = t
            return True
        return False
