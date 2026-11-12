#!/usr/bin/env python3
"""Convert InsightFace ONNX models (SCRFD / ArcFace) to ANE-optimised Core ML .mlpackage.

Optional - only needed if you start from ONNX (e.g. InsightFace `buffalo_l`:
det_10g.onnx + w600k_r50.onnx). Needs extra packages, best in a separate venv:

    pip install coremltools torch==2.5.1 onnx onnx2torch
    python scripts/convert_models.py --scrfd det_10g.onnx --arcface w600k_r50.onnx --out models/

ANE optimisations applied:
  * fixed input shapes, FP16 mlprogram, macOS 14 target
  * ImageType input -> mean/std normalisation runs inside the model
  * SCRFD: output heads are cut *before* the Transpose/Reshape ops, which Core ML
    otherwise runs on the CPU (~10% of the model cost). The raw NCHW heads are
    decoded by inference/scrfd_engine.py.
"""
import argparse
import sys
import tempfile
from pathlib import Path


def strip_scrfd_heads(onnx_path: Path, height: int, width: int) -> Path:
    """Re-point every SCRFD output to the NCHW tensor feeding its Transpose."""
    import onnx
    from onnx import helper, utils

    model = onnx.load(str(onnx_path))
    g = model.graph
    producer = {o: n for n in g.node for o in n.output}
    new_outputs = []
    for out in list(g.output):
        name, sigmoid = out.name, False
        node = producer.get(name)
        if node is not None and node.op_type == "Sigmoid":
            sigmoid, node = True, producer.get(node.input[0])
        if node is None or node.op_type != "Reshape":
            raise SystemExit(f"unexpected SCRFD head for output {out.name}")
        transpose = producer.get(node.input[0])
        if transpose is None or transpose.op_type != "Transpose":
            raise SystemExit(f"unexpected SCRFD head for output {out.name}")
        nchw = transpose.input[0]
        if sigmoid:
            sig_out = f"{nchw}_sigmoid"
            g.node.append(helper.make_node("Sigmoid", [nchw], [sig_out]))
            nchw = sig_out
        new_outputs.append(nchw)

    # Fix the input shape (ANE needs static shapes) and prune the unused heads.
    dims = g.input[0].type.tensor_type.shape.dim
    for d, v in zip(dims, (1, 3, height, width)):
        d.ClearField("dim_param")
        d.dim_value = v
    tmp = Path(tempfile.mkdtemp()) / "scrfd_full.onnx"
    onnx.save(model, str(tmp))
    pruned = tmp.with_name("scrfd_nchw.onnx")
    utils.extract_model(str(tmp), str(pruned), [g.input[0].name], new_outputs)
    print(f"SCRFD heads cut before Transpose: {len(new_outputs)} NCHW outputs")
    return pruned


def to_coreml(onnx_path: Path, shape, scale: float, bias, out_path: Path) -> None:
    import coremltools as ct
    import onnx
    import torch
    from onnx2torch import convert

    torch_model = convert(onnx.load(str(onnx_path))).eval()
    example = torch.rand(*shape) * 2 - 1
    with torch.no_grad():
        traced = torch.jit.trace(torch_model, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="input", shape=shape, scale=scale, bias=bias, color_layout=ct.colorlayout.RGB)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
    )
    mlmodel.save(str(out_path))
    outs = [o.name for o in mlmodel.get_spec().description.output]
    print(f"saved {out_path} (input {shape}, {len(outs)} outputs)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scrfd", type=Path, help="SCRFD ONNX (e.g. det_10g.onnx)")
    ap.add_argument("--arcface", type=Path, help="ArcFace ONNX (e.g. w600k_r50.onnx)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    ap.add_argument("--det-size", type=int, default=640, help="SCRFD input size (square)")
    ap.add_argument("--keep-scrfd-heads", action="store_true", help="do not cut the Transpose/Reshape heads")
    args = ap.parse_args()
    if not args.scrfd and not args.arcface:
        ap.error("pass --scrfd and/or --arcface")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.scrfd:
        src = args.scrfd if args.keep_scrfd_heads else strip_scrfd_heads(args.scrfd, args.det_size, args.det_size)
        # SCRFD normalisation: (x - 127.5) / 128, RGB
        to_coreml(src, (1, 3, args.det_size, args.det_size), 1 / 128.0, [-127.5 / 128.0] * 3,
                  args.out / "scrfd.mlpackage")
    if args.arcface:
        # ArcFace normalisation: (x - 127.5) / 127.5, RGB
        to_coreml(args.arcface, (1, 3, 112, 112), 1 / 127.5, [-1.0] * 3, args.out / "arcface.mlpackage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
