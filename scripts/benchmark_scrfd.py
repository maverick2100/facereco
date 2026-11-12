#!/usr/bin/env python3
"""Benchmark SCRFD per compute unit and batch size.

Look at 'CPU ms/item': on the ANE it should be a small fraction of CPU_ONLY.

    python scripts/benchmark_scrfd.py
    python scripts/benchmark_scrfd.py --units CPU_AND_NE,ALL --batches 1,4,8 --image crowd.jpg
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from config.settings import load_settings  # noqa: E402
from inference.coreml_loader import CoreMLModel  # noqa: E402
from inference.model_profiler import benchmark, format_table  # noqa: E402
from inference.scrfd_engine import SCRFDEngine  # noqa: E402
from utils.image_utils import load_image  # noqa: E402
from utils.logger import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="CPU_ONLY,CPU_AND_GPU,CPU_AND_NE,ALL")
    ap.add_argument("--batches", default="1,4,8")
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--image", help="test image (default: random 1280x720 noise)")
    args = ap.parse_args()

    settings = load_settings()
    setup_logging(settings.logging_config, "WARNING")
    img = load_image(args.image) if args.image else np.random.randint(0, 255, (720, 1280, 3), np.uint8)

    rows = {}
    for unit in args.units.split(","):
        try:
            model = CoreMLModel(settings.models.scrfd_path, unit, cache_dir=None, name="scrfd")
            eng = SCRFDEngine(model, settings.detection.det_threshold, settings.detection.nms_threshold,
                              settings.models.scrfd_input_mean, settings.models.scrfd_input_std,
                              settings.models.scrfd_apply_sigmoid)
        except Exception as exc:
            rows[unit] = {"error": str(exc)}
            continue
        for b in [int(x) for x in args.batches.split(",")]:
            batch = [img] * b
            name = f"{unit} b={b}"
            try:
                rows[name] = benchmark(lambda: eng.detect_batch(batch), b, args.iterations, warmup=3)
            except Exception as exc:
                rows[name] = {"error": str(exc)}
            print(f"done {name}", flush=True)
    print("\nSCRFD (end-to-end incl. letterbox + decode + NMS)\n")
    print(format_table(rows))
    print("\nframes/s needed for 20 cams @ 4 fps = 80")
    return 0


if __name__ == "__main__":
    sys.exit(main())
