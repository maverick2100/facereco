#!/usr/bin/env python3
"""Convert ONNX models (SCRFD / ArcFace / YOLOv7) to ANE-optimised Core ML .mlpackage.

Optional - only needed if you start from ONNX (e.g. InsightFace `buffalo_l`:
det_10g.onnx + w600k_r50.onnx, and a YOLOv7 export). The packages it needs (torch, onnx, onnx2torch,
coremltools) are in requirements.txt:

    python scripts/convert_models.py --scrfd scrfd.onnx --arcface arcface.onnx --yolo yolov7.onnx --out models/

YOLOv7: export WITHOUT --end2end (the NMS ops inside an end2end graph don't convert; NMS runs in
person_detection/yolo_engine.py anyway), e.g. `python export.py --weights yolov7.pt --grid --img-size 640`.

ANE optimisations applied:
  * fixed input shapes, FP16 mlprogram, macOS 14 target
  * ImageType input -> mean/std normalisation runs inside the model
  * YOLOv7: 9x9 / 13x13 SPP max-pools rewritten as chained 5x5 (identical output, ANE instead of CPU)
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


def onnx_input_size(onnx_path: Path, default: int) -> int:
    """Square input size from the ONNX graph when it is static, else `default`."""
    import onnx

    dims = onnx.load(str(onnx_path), load_external_data=False).graph.input[0].type.tensor_type.shape.dim
    h, w = (d.dim_value for d in list(dims)[2:4]) if len(dims) == 4 else (0, 0)
    if h and w:
        if h != w:
            raise SystemExit(f"{onnx_path}: non-square input {h}x{w} is not supported")
        return int(h)
    return default


def cascade_large_maxpools(onnx_path: Path) -> Path:
    """YOLOv7's SPP block max-pools with 9x9 and 13x13 kernels, which the ANE can't run: Core ML moves
    them (plus the copies in and out) to the CPU. A stride-1 k x k max-pool equals (k-1)/4 chained
    5x5 max-pools (9 = 5+5-1, 13 = 5+5+5-2), so the output is bit-identical and stays on the ANE.
    Measured on the M4: 13.7 -> 4.1 ms per frame and 12.2 -> 2.8 ms CPU per frame."""
    import onnx
    from onnx import helper

    model = onnx.load(str(onnx_path))
    g, nodes, changed = model.graph, [], 0
    for n in g.node:
        a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
        k = list(a.get("kernel_shape", [0, 0]))
        if (n.op_type == "MaxPool" and k[0] == k[1] and k[0] > 5 and (k[0] - 1) % 4 == 0
                and list(a.get("strides", [1, 1])) == [1, 1] and list(a.get("pads", [0] * 4)) == [k[0] // 2] * 4
                and not a.get("ceil_mode", 0) and list(a.get("dilations", [1, 1])) == [1, 1]):
            x, reps = n.input[0], (k[0] - 1) // 4
            for i in range(reps):
                y = n.output[0] if i == reps - 1 else f"{n.output[0]}__mp5_{i}"
                nodes.append(helper.make_node("MaxPool", [x], [y], name=f"{n.name}__mp5_{i}",
                                              kernel_shape=[5, 5], pads=[2, 2, 2, 2], strides=[1, 1]))
                x = y
            changed += 1
        else:
            nodes.append(n)
    if not changed:
        return onnx_path
    del g.node[:]
    g.node.extend(nodes)
    out = Path(tempfile.mkdtemp()) / f"{onnx_path.stem}_mp5.onnx"
    onnx.save(model, str(out))
    print(f"{onnx_path.name}: {changed} large max-pools -> chained 5x5 (ANE-compatible, identical output)")
    return out


def onnx_output_names(onnx_path: Path):
    """Keep the ONNX output names (e.g. boxes / scores / class_idx) in the Core ML model."""
    import onnx

    return [o.name for o in onnx.load(str(onnx_path), load_external_data=False).graph.output]


def check_convertible(onnx_path: Path) -> None:
    import onnx

    ops = {n.op_type for n in onnx.load(str(onnx_path), load_external_data=False).graph.node}
    bad = ops & {"NonMaxSuppression", "TopK", "NonZero"}
    if bad:
        raise SystemExit(f"{onnx_path} contains {sorted(bad)} (an --end2end / NMS export). Re-export YOLOv7 "
                         "without --end2end, e.g.: python export.py --weights yolov7.pt --grid --img-size 640")


def to_coreml(onnx_path: Path, shape, scale: float, bias, out_path: Path, output_names=None) -> None:
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
        outputs=[ct.TensorType(name=n) for n in output_names] if output_names else None,
    )
    mlmodel.save(str(out_path))
    outs = [o.name for o in mlmodel.get_spec().description.output]
    print(f"saved {out_path} (input {shape}, {len(outs)} outputs)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scrfd", type=Path, help="SCRFD ONNX (e.g. det_10g.onnx)")
    ap.add_argument("--arcface", type=Path, help="ArcFace ONNX (e.g. w600k_r50.onnx)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    ap.add_argument("--yolo", type=Path, help="YOLOv7 ONNX (exported without --end2end)")
    ap.add_argument("--det-size", type=int, default=640, help="SCRFD input size (square)")
    ap.add_argument("--yolo-size", type=int, default=0, help="YOLOv7 input size (default: from the ONNX, else 640)")
    ap.add_argument("--keep-scrfd-heads", action="store_true", help="do not cut the Transpose/Reshape heads")
    args = ap.parse_args()
    if not (args.scrfd or args.arcface or args.yolo):
        ap.error("pass --scrfd, --arcface and/or --yolo")
    for p in (args.scrfd, args.arcface, args.yolo):
        if p is not None and not p.exists():
            ap.error(f"{p} not found")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.scrfd:
        src = args.scrfd if args.keep_scrfd_heads else strip_scrfd_heads(args.scrfd, args.det_size, args.det_size)
        # SCRFD normalisation: (x - 127.5) / 128, RGB
        to_coreml(src, (1, 3, args.det_size, args.det_size), 1 / 128.0, [-127.5 / 128.0] * 3,
                  args.out / "scrfd.mlpackage")
    if args.arcface:
        # ArcFace normalisation: (x - 127.5) / 127.5, RGB
        to_coreml(args.arcface, (1, 3, 112, 112), 1 / 127.5, [-1.0] * 3, args.out / "arcface.mlpackage")
    if args.yolo:
        check_convertible(args.yolo)
        size = args.yolo_size or onnx_input_size(args.yolo, 640)
        src = cascade_large_maxpools(args.yolo)
        # YOLOv7 normalisation: x / 255, RGB (built into the model, so FR_PERSON_INPUT_STD is not used)
        to_coreml(src, (1, 3, size, size), 1 / 255.0, [0.0] * 3, args.out / "yolov7.mlpackage",
                  output_names=onnx_output_names(args.yolo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
