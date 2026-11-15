#!/usr/bin/env python3
"""Show whether each Core ML model actually runs on the Apple Neural Engine (ANE), and if not, why.

Uses Core ML's own compute plan (macOS 14.4+), which says for every layer which device
Core ML will run it on: ANE, CPU or GPU. Then times the model on CPU_ONLY vs CPU_AND_NE.

    python scripts/check_ane.py                      # the models from .env (SCRFD, ArcFace, YOLOv7 if present)
    python scripts/check_ane.py models/yolov7.mlpackage
    python scripts/check_ane.py --units ALL          # plan for another compute-unit setting

Reading the result:
    ANE share near 100%         -> runs on the Neural Engine
    big CPU/GPU share           -> those layers (listed) are not ANE-compatible; re-convert (see README)
    CPU_AND_NE not faster and CPU ms/item close to CPU_ONLY -> effectively running on the CPU
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from config.settings import load_settings  # noqa: E402
from inference.ane_config import check_environment, resolve_compute_units  # noqa: E402
from inference.coreml_loader import parse_feature  # noqa: E402


def _device_name(dev) -> str:
    n = type(dev).__name__
    return "ANE" if "NeuralEngine" in n else "GPU" if "GPU" in n else "CPU" if "CPU" in n else n


def _walk_ops(block):
    for op in block.operations:
        yield op
        for b in getattr(op, "blocks", []) or []:
            yield from _walk_ops(b)


def compute_plan(path: Path, units: str) -> dict:
    """{'by_device': {dev: cost share}, 'ops': Counter, 'off_ane': {op_type: Counter(dev)}, 'kind': str}"""
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    # keep `model` alive: its compiled .mlmodelc is a temp folder deleted when the object goes away
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
    plan = MLComputePlan.load_from_path(path=str(model.get_compiled_model_path()),
                                        compute_units=resolve_compute_units(units))
    structure = plan.model_structure

    cost = defaultdict(float)
    count = Counter()
    off_ane = defaultdict(Counter)
    supported_ane_but_not_chosen = Counter()

    def add(op_type, usage, weight):
        if usage is None:          # constants etc. - not executed on a device
            return
        dev = _device_name(usage.preferred_compute_device)
        cost[dev] += weight
        count[dev] += 1
        if dev != "ANE":
            off_ane[op_type][dev] += 1
            if any(_device_name(d) == "ANE" for d in usage.supported_compute_devices):
                supported_ane_but_not_chosen[op_type] += 1

    def visit(struct, prefix=""):
        if struct.program is not None:
            for fn in struct.program.functions.values():
                for op in _walk_ops(fn.block):
                    usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                    est = plan.get_estimated_cost_for_mlprogram_operation(op)
                    add(op.operator_name, usage, est.weight if est is not None else 0.0)
            return "mlprogram"
        if struct.neuralnetwork is not None:
            for layer in struct.neuralnetwork.layers:
                usage = plan.get_compute_device_usage_for_neuralnetwork_layer(layer)
                add(layer.type, usage, 1.0)   # no cost estimate for neuralnetwork: count layers
            return "neuralnetwork"
        if struct.pipeline is not None:
            kinds = [visit(sub) for _, sub in struct.pipeline.sub_models]
            return "pipeline(" + ", ".join(k or "?" for k in kinds) + ")"
        return "unknown"

    kind = visit(structure)
    del model
    total = sum(cost.values()) or 1.0
    return {"kind": kind, "by_device": {d: c / total for d, c in cost.items()}, "ops": count,
            "off_ane": off_ane, "ane_capable": supported_ane_but_not_chosen}


def _dummy_feeds(path: Path):
    import coremltools as ct
    from PIL import Image

    spec = ct.utils.load_spec(str(path))
    feeds = {}
    for f in spec.description.input:
        fs = parse_feature(f)
        if fs.kind == "image":
            feeds[fs.name] = Image.fromarray(np.random.randint(0, 255, (fs.height, fs.width, 3), np.uint8))
        elif fs.kind == "multiarray":
            feeds[fs.name] = np.random.rand(*fs.shape).astype(fs.dtype)
        else:
            return None
    return feeds


def time_units(path: Path, units: str, iters: int = 30) -> dict:
    import coremltools as ct

    model = ct.models.MLModel(str(path), compute_units=resolve_compute_units(units))
    feeds = _dummy_feeds(path)
    if feeds is None:
        return {"error": "unsupported input type"}
    for _ in range(3):
        model.predict(feeds)
    w0, c0 = time.perf_counter(), time.process_time()
    for _ in range(iters):
        model.predict(feeds)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    return {"ms": wall * 1000 / iters, "cpu_ms": cpu * 1000 / iters}


def spec_warnings(path: Path) -> list:
    import coremltools as ct

    spec = ct.utils.load_spec(str(path))
    out = []
    kind = spec.WhichOneof("Type")
    if kind == "neuralNetwork":
        out.append("old 'neuralnetwork' format: re-convert as mlprogram (convert_to='mlprogram') with FP16")
    for f in spec.description.input:
        t = f.type
        w = t.WhichOneof("Type")
        if w == "multiArrayType":
            ma = t.multiArrayType
            if len(ma.shapeRange.sizeRanges) or len(ma.enumeratedShapes.shapes) > 1:
                out.append(f"input '{f.name}' has a flexible shape: fix it (e.g. 1x3x640x640) - flexible "
                           "shapes often keep the model off the ANE")
            if int(ma.dataType) == 65568 and kind != "mlProgram":
                out.append(f"input '{f.name}' is float32")
        elif w == "imageType":
            it = t.imageType
            if it.HasField("imageSizeRange") or len(it.enumeratedSizes.sizes) > 1:
                out.append(f"image input '{f.name}' has flexible size: fix it")
    if kind == "mlProgram" and "FLOAT16" not in str(spec.mlProgram):
        out.append("FP32 model: the ANE only runs FP16, so Core ML puts the whole model on the CPU. "
                   "Re-convert with compute_precision=ct.precision.FLOAT16")
    if kind == "pipeline":
        out.append("pipeline model (e.g. with a Core ML NMS stage): the NMS stage always runs on the CPU; "
                   "that is fine if the network itself is on the ANE")
    return out


def time_units_isolated(path: Path, units: str) -> dict:
    """time_units() in a child process: Core ML can abort the whole process on some graphs
    (e.g. a YOLO export with argmax on CPU_ONLY), which should be reported, not kill this script."""
    import json
    import subprocess

    r = subprocess.run([sys.executable, __file__, "--bench-one", units, str(path)], capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or [""])[-1]
    if r.returncode == 0 and last.startswith("{"):
        return json.loads(last)
    if r.returncode < 0 or r.returncode > 128:
        return {"error": f"Core ML crashed running this model on {units} (signal {abs(r.returncode) % 128}) - "
                         f"don't use FR_COMPUTE_UNITS={units} with it"}
    return {"error": (r.stderr.strip().splitlines() or ["failed"])[-1]}


def report(path: Path, units: str, bench: bool) -> None:
    print(f"\n=== {path}")
    if not path.exists():
        print("  not found")
        return
    for w in spec_warnings(path):
        print(f"  ! {w}")
    try:
        p = compute_plan(path, units)
    except Exception as exc:
        print(f"  compute plan unavailable ({exc}). Needs macOS 14.4+ and coremltools >= 8.")
        p = None
    if p:
        share = "  ".join(f"{d} {v * 100:5.1f}%" for d, v in sorted(p["by_device"].items(), key=lambda x: -x[1]))
        n = "  ".join(f"{d}:{c}" for d, c in p["ops"].most_common())
        print(f"  format       : {p['kind']}")
        print(f"  plan ({units}): {share}   (share of estimated compute; ops {n})")
        ane = p["by_device"].get("ANE", 0.0)
        verdict = ("RUNS ON ANE" if ane >= 0.9 else "MOSTLY ANE (some layers fall back)" if ane >= 0.5
                   else "NOT ON ANE")
        print(f"  verdict      : {verdict}")
        if ane < 0.9:
            print("  fix          : re-convert as mlprogram, FP16, fixed input shape, minimum_deployment_target "
                  "macOS14 (scripts/convert_models.py does this for SCRFD/ArcFace)")
        if p["off_ane"]:
            print("  layers not on the ANE (op type: device x count):")
            for op, devs in sorted(p["off_ane"].items(), key=lambda kv: -sum(kv[1].values()))[:15]:
                extra = "  <- ANE-capable, Core ML chose otherwise" if p["ane_capable"].get(op) else ""
                print(f"     {op:<28} " + ", ".join(f"{d} x{c}" for d, c in devs.items()) + extra)
    if bench:
        rows = {}
        for u in ("CPU_ONLY", "CPU_AND_NE"):
            rows[u] = time_units_isolated(path, u)
        for u, r in rows.items():
            print(f"  {u:<11}: " + (r["error"] if "error" in r else f"{r['ms']:7.2f} ms/inference, "
                                                                      f"CPU {r['cpu_ms']:6.2f} ms"))
        a, b = rows.get("CPU_ONLY", {}), rows.get("CPU_AND_NE", {})
        if "ms" in a and "ms" in b:
            print(f"  speed-up     : {a['ms'] / b['ms']:.1f}x faster, {a['cpu_ms'] / max(b['cpu_ms'], 1e-3):.1f}x "
                  "less CPU with CPU_AND_NE" + ("" if b["cpu_ms"] < 0.5 * a["cpu_ms"] else
                                                 "  <- CPU cost barely drops: not really on the ANE"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help=".mlpackage / .mlmodel paths (default: from .env)")
    ap.add_argument("--units", default=None, help="compute units for the plan (default FR_COMPUTE_UNITS)")
    ap.add_argument("--no-bench", action="store_true", help="skip the CPU_ONLY vs CPU_AND_NE timing")
    ap.add_argument("--bench-one", nargs=2, metavar=("UNITS", "MODEL"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.bench_one:
        import json

        try:
            print(json.dumps(time_units(Path(args.bench_one[1]), args.bench_one[0])))
            return 0
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1

    s = load_settings()
    units = args.units or s.models.compute_units
    print(f"FR_COMPUTE_UNITS = {s.models.compute_units}" +
          ("   <- set FR_COMPUTE_UNITS=CPU_AND_NE in .env to use the ANE" if s.models.compute_units in
           ("CPU_ONLY", "CPU_AND_GPU") else ""))
    for w in check_environment():
        print(f"! {w}")
    paths = [Path(m) for m in args.models] or [p for p in (s.models.scrfd_path, s.models.arcface_path,
                                                            s.models.yolo_path) if p.exists()]
    if not paths:
        print("No models found. Put them in models/ (see README step 4) or pass paths.")
        return 1
    for p in paths:
        report(p, units, not args.no_bench)
    print("\nAlso: 'sudo powermetrics --samplers cpu_power -i 1000' while the system runs; ANE Power > 0 mW "
          "means the ANE is working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
