"""Process / system memory usage (16 GB unified memory is shared with the ANE/GPU)."""
from __future__ import annotations

import os

import psutil

MB = 1024 * 1024


class MemoryMonitor:
    def __init__(self, low_available_mb: float = 1500.0) -> None:
        self.proc = psutil.Process(os.getpid())
        self.low_available_mb = low_available_mb

    def sample(self) -> dict:
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        rss = self.proc.memory_info().rss
        return {
            "process_rss_mb": round(rss / MB, 1),
            "system_total_mb": round(vm.total / MB),
            "system_available_mb": round(vm.available / MB),
            "system_used_pct": vm.percent,
            "swap_used_mb": round(sw.used / MB),
            "low_memory": vm.available / MB < self.low_available_mb,
        }
