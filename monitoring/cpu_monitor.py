"""Process / system CPU usage."""
from __future__ import annotations

import os

import psutil


class CPUMonitor:
    def __init__(self) -> None:
        self.proc = psutil.Process(os.getpid())
        self.ncpu = psutil.cpu_count(logical=True) or 1
        self.proc.cpu_percent(None)  # prime the counters
        psutil.cpu_percent(None)

    def sample(self) -> dict:
        p = self.proc.cpu_percent(None)  # % of one core (can exceed 100)
        try:
            load = [round(x, 2) for x in os.getloadavg()]
        except OSError:
            load = []
        return {
            "process_pct_of_core": round(p, 1),
            "process_pct_of_machine": round(p / self.ncpu, 1),
            "system_pct": round(psutil.cpu_percent(None), 1),
            "threads": self.proc.num_threads(),
            "cores": self.ncpu,
            "load_avg": load,
        }
