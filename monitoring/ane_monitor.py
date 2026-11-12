"""Neural Engine monitoring.

macOS has no public API for ANE utilisation. We report:
  * `inference_busy_pct` - fraction of wall time spent inside Core ML predict
    calls (always available, from PerformanceTracker).
  * ANE / CPU / GPU power from `powermetrics` - only when running as root
    (sudo python main.py run). Non-zero ANE power proves the ANE is in use.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
from typing import Optional

from monitoring.performance import PerformanceTracker

log = logging.getLogger(__name__)

_POWER_RE = re.compile(r"^(ANE|CPU|GPU) Power:\s*([\d.]+)\s*mW", re.MULTILINE)


def read_powermetrics(sample_ms: int = 1000) -> Optional[dict]:
    if not shutil.which("powermetrics") or os.geteuid() != 0:
        return None
    try:
        out = subprocess.run(["powermetrics", "--samplers", "cpu_power", "-i", str(sample_ms), "-n", "1"],
                             capture_output=True, text=True, timeout=sample_ms / 1000 + 10).stdout
    except Exception as exc:
        log.debug("powermetrics failed: %s", exc)
        return None
    vals = {k.lower() + "_power_mw": float(v) for k, v in _POWER_RE.findall(out)}
    return vals or None


class ANEMonitor:
    def __init__(self, perf: PerformanceTracker, enable_powermetrics: bool = True, interval_s: float = 15.0) -> None:
        self.perf = perf
        self.interval_s = interval_s
        self.powermetrics_enabled = enable_powermetrics and os.geteuid() == 0 and bool(shutil.which("powermetrics"))
        self._latest: Optional[dict] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.powermetrics_enabled:
            log.info("ANE power metrics unavailable (run with sudo to enable powermetrics)")
            return
        self._thread = threading.Thread(target=self._run, name="ane-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._latest = read_powermetrics()
            self._stop.wait(self.interval_s)

    def sample(self) -> dict:
        d = {
            "inference_busy_pct": round(self.perf.utilization("inference") * 100, 1),
            "scrfd_busy_pct": round(self.perf.utilization("scrfd") * 100, 1),
            "arcface_busy_pct": round(self.perf.utilization("arcface") * 100, 1),
            "powermetrics": self.powermetrics_enabled,
        }
        if self._latest:
            d.update(self._latest)
        return d
