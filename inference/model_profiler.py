"""Model benchmarking: latency, throughput and *CPU cost* per inference.

`cpu_ms_per_item` is the key ANE indicator: process CPU time consumed per
inference. When a model runs on the Neural Engine it is far lower than with
CPU_ONLY, even when wall-clock latency is similar.
"""
from __future__ import annotations

import time
from typing import Callable, Dict

import numpy as np


def benchmark(fn: Callable[[], object], items_per_call: int = 1, iterations: int = 50,
              warmup: int = 5) -> Dict[str, float]:
    for _ in range(max(0, warmup)):
        fn()
    lat = []
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    for _ in range(iterations):
        t = time.perf_counter()
        fn()
        lat.append((time.perf_counter() - t) * 1000.0)
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    arr = np.asarray(lat)
    total_items = iterations * items_per_call
    return {
        "ms_per_call": float(arr.mean()),
        "p95_ms": float(np.percentile(arr, 95)),
        "ms_per_item": float(wall * 1000.0 / total_items),
        "items_per_s": float(total_items / wall) if wall > 0 else 0.0,
        "cpu_ms_per_item": float(cpu * 1000.0 / total_items),
        "cpu_util_pct": float(cpu / wall * 100.0) if wall > 0 else 0.0,
    }


def format_table(rows: Dict[str, Dict[str, float]]) -> str:
    header = f"{'config':<28}{'ms/call':>10}{'p95':>9}{'ms/item':>10}{'items/s':>10}{'CPU ms/item':>13}{'CPU %':>8}"
    lines = [header, "-" * len(header)]
    for name, r in rows.items():
        if "error" in r:
            lines.append(f"{name:<28}  ERROR: {r['error']}")
            continue
        lines.append(f"{name:<28}{r['ms_per_call']:>10.2f}{r['p95_ms']:>9.2f}{r['ms_per_item']:>10.2f}"
                     f"{r['items_per_s']:>10.1f}{r['cpu_ms_per_item']:>13.2f}{r['cpu_util_pct']:>8.0f}")
    return "\n".join(lines)
