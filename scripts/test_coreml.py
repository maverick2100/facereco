#!/usr/bin/env python3
"""Load both .mlpackage models, print their I/O spec, run a test inference.

    python scripts/test_coreml.py
    python scripts/test_coreml.py --image photo.jpg        # detect + embed faces in a photo
    python scripts/test_coreml.py --compute-units ALL
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from config.settings import load_settings  # noqa: E402
from inference.ane_config import check_environment  # noqa: E402
from inference.inference_manager import InferenceManager  # noqa: E402
from processing.face_alignment import norm_crop  # noqa: E402
from utils.image_utils import draw_faces, load_image  # noqa: E402
from utils.logger import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="image with faces")
    ap.add_argument("--compute-units", help="override FR_COMPUTE_UNITS")
    ap.add_argument("--out", help="save annotated image here")
    args = ap.parse_args()

    settings = load_settings()
    setup_logging(settings.logging_config, "INFO")
    if args.compute_units:
        settings.models.compute_units = args.compute_units

    for w in check_environment():
        print("WARNING:", w)

    im = InferenceManager(settings.models, settings.detection).load()
    info = im.info()
    for key in ("scrfd", "arcface"):
        m = info[key]
        print(f"\n[{key}] {m['path']}  type={m['type']}  load={m['load_time_s']}s  units={m['compute_units']}")
        for i in m["inputs"]:
            print("   in :", i)
        for o in m["outputs"]:
            print("   out:", o)

    if not args.image:
        w, h = im.scrfd.input_size
        dets, _ = im.detect([np.zeros((h, w, 3), np.uint8)])[0]
        emb = im.embed([np.zeros((112, 112, 3), np.uint8)])
        print(f"\nblank image -> {len(dets)} faces; embedding dim {emb.shape[1]}, norm {np.linalg.norm(emb[0]):.3f}")
        print("OK - pass --image to test on a real photo")
        return 0

    img = load_image(args.image)
    if img is None:
        print("cannot read", args.image)
        return 1
    dets, kps = im.detect([img])[0]
    print(f"\n{len(dets)} face(s) in {args.image}")
    if len(dets):
        crops = [norm_crop(img, k) for k in kps]
        emb = im.embed(crops)
        for i, d in enumerate(dets):
            print(f"  face {i}: box={d[:4].astype(int).tolist()} score={d[4]:.3f}")
        if len(emb) > 1:
            sims = emb @ emb.T
            print("pairwise cosine similarity:\n", np.round(sims, 3))
        out = args.out or str(Path(args.image).with_suffix(".detected.jpg"))
        import cv2

        cv2.imwrite(out, draw_faces(img, dets, [f"{d[4]:.2f}" for d in dets]))
        print("annotated image:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
