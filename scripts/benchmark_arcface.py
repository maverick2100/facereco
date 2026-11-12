#!/usr/bin/env python3
"""Benchmark ArcFace per compute unit and batch size.

    python scripts/benchmark_arcface.py
    python scripts/benchmark_arcface.py --units CPU_AND_NE --batches 1,8,16,32
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from config.settings import load_settings  # noqa: E402
from inference.arcface_engine import ArcFaceEngine  # noqa: E402
from inference.coreml_loader import CoreMLModel  # noqa: E402
from inference.model_profiler import benchmark, format_table  # noqa: E402
from utils.logger import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="CPU_ONLY,CPU_AND_GPU,CPU_AND_NE,ALL")
    ap.add_argument("--batches", default="1,8,16,32")
    ap.add_argument("--iterations", type=int, default=30)
    args = ap.parse_args()

    settings = load_settings()
    setup_logging(settings.logging_config, "WARNING")
    crop = np.random.randint(0, 255, (112, 112, 3), np.uint8)

    rows = {}
    for unit in args.units.split(","):
        try:
            model = CoreMLModel(settings.models.arcface_path, unit, cache_dir=None, name="arcface")
            eng = ArcFaceEngine(model, settings.models.arcface_input_mean, settings.models.arcface_input_std)
        except Exception as exc:
            rows[unit] = {"error": str(exc)}
            continue
        for b in [int(x) for x in args.batches.split(",")]:
            crops = [crop] * b
            name = f"{unit} b={b}"
            try:
                rows[name] = benchmark(lambda: eng.embed_batch(crops), b, args.iterations, warmup=3)
            except Exception as exc:
                rows[name] = {"error": str(exc)}
            print(f"done {name}", flush=True)
    print("\nArcFace (per face crop)\n")
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
