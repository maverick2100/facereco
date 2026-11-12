"""Model input preparation.

If a Core ML model was converted with an *ImageType* input, the normalisation
(scale/bias) runs inside the model on the Neural Engine and we only hand over
a PIL image - the cheapest path for the CPU. For *MultiArray* inputs we
normalise with a single `cv2.dnn.blobFromImage` call (vectorised C++).
"""
from __future__ import annotations

from typing import Any, Tuple

import cv2
import numpy as np


def letterbox(img: np.ndarray, size: Tuple[int, int]) -> Tuple[np.ndarray, float]:
    """Resize keeping aspect ratio into the top-left of a (w, h) canvas (SCRFD convention).

    Returns (canvas, scale) where original_coord = model_coord / scale.
    """
    target_w, target_h = size
    h, w = img.shape[:2]
    im_ratio = h / float(w)
    model_ratio = target_h / float(target_w)
    if im_ratio > model_ratio:
        new_h = target_h
        new_w = max(1, int(new_h / im_ratio))
    else:
        new_w = target_w
        new_h = max(1, int(new_w * im_ratio))
    scale = new_h / float(h)
    if (new_w, new_h) == (w, h):
        resized = img
    else:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if (new_w, new_h) == (target_w, target_h):
        return np.ascontiguousarray(resized), scale
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    return canvas, scale


def infer_layout(shape: Tuple[int, ...]) -> str:
    if len(shape) == 4:
        return "NCHW" if shape[1] in (1, 3) else "NHWC"
    if len(shape) == 3:
        return "CHW" if shape[0] in (1, 3) else "HWC"
    return "NCHW"


def input_size_of(spec: Any, default: Tuple[int, int]) -> Tuple[int, int]:
    """(width, height) expected by a model input FeatureSpec."""
    if spec.kind == "image" and spec.width and spec.height:
        return int(spec.width), int(spec.height)
    shape = tuple(spec.shape or ())
    if len(shape) >= 3:
        layout = infer_layout(shape)
        if layout in ("NCHW", "CHW"):
            h, w = shape[-2], shape[-1]
        else:
            h, w = shape[-3], shape[-2]
        if h > 0 and w > 0:
            return int(w), int(h)
    return default


def to_model_input(img_bgr: np.ndarray, spec: Any, mean: float, std: float, swap_rb: bool = True) -> Any:
    """Convert a BGR image already at model resolution into the value Core ML expects."""
    if spec.kind == "image":
        from PIL import Image

        if str(spec.color_space).startswith("GRAYSCALE"):
            return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), "L")
        # PIL images are RGB; Core ML converts to the model's declared colour space itself.
        return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), "RGB")

    if spec.kind == "multiarray":
        shape = tuple(spec.shape or ())
        layout = infer_layout(shape)
        if layout in ("NCHW", "CHW"):
            blob = cv2.dnn.blobFromImage(img_bgr, scalefactor=1.0 / std, mean=(mean, mean, mean), swapRB=swap_rb)
            if layout == "CHW":
                blob = blob[0]
        else:
            x = img_bgr[..., ::-1] if swap_rb else img_bgr
            blob = (x.astype(np.float32) - mean) * (1.0 / std)
            if layout == "NHWC":
                blob = blob[None]
        return np.ascontiguousarray(blob, dtype=spec.dtype or np.float32)

    raise ValueError(f"unsupported model input kind: {spec.kind}")
