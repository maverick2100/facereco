"""ArcFace embedding extractor on Core ML."""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

import cv2
import numpy as np

from constants import ARCFACE_INPUT_MEAN, ARCFACE_INPUT_SIZE, ARCFACE_INPUT_STD
from processing.image_preprocessing import input_size_of, to_model_input

log = logging.getLogger(__name__)


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


class ArcFaceEngine:
    def __init__(self, model: Any, input_mean: float = ARCFACE_INPUT_MEAN, input_std: float = ARCFACE_INPUT_STD,
                 output_name: Optional[str] = None) -> None:
        self.model = model
        self.input_mean = input_mean
        self.input_std = input_std
        self.input_spec = model.inputs[0]
        self.input_w, self.input_h = input_size_of(self.input_spec, (ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE))
        # A model converted with a fixed batch dimension > 1 (e.g. [16,3,112,112]) gets real
        # tensor batching, which is the most ANE-efficient option.
        shape = tuple(self.input_spec.shape or ())
        self.native_batch = int(shape[0]) if (self.input_spec.kind == "multiarray" and len(shape) == 4
                                              and shape[0] > 1) else 1
        self.output_name = output_name or self._pick_output()
        self.dim: Optional[int] = None
        log.info("ArcFace input %dx%d (%s), native batch %d, output '%s'",
                 self.input_w, self.input_h, self.input_spec.kind, self.native_batch, self.output_name)

    def _pick_output(self) -> str:
        outs = self.model.outputs
        if len(outs) == 1:
            return outs[0].name
        for o in outs:
            if any(k in o.name.lower() for k in ("embed", "fc1", "feature", "output")):
                return o.name
        return outs[0].name

    def _prepare(self, crop_bgr: np.ndarray) -> Any:
        if crop_bgr.shape[1] != self.input_w or crop_bgr.shape[0] != self.input_h:
            crop_bgr = cv2.resize(crop_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        return to_model_input(crop_bgr, self.input_spec, self.input_mean, self.input_std)

    def embed_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Return (N, D) L2-normalised float32 embeddings."""
        if len(crops) == 0:
            return np.zeros((0, self.dim or 512), dtype=np.float32)
        if self.native_batch > 1:
            vecs = self._embed_native(crops)
        else:
            feeds = [{self.input_spec.name: self._prepare(c)} for c in crops]
            outs = self.model.predict_batch(feeds)
            vecs = np.stack([np.asarray(o[self.output_name], dtype=np.float32).reshape(-1) for o in outs])
        self.dim = vecs.shape[1]
        return l2_normalize(vecs).astype(np.float32)

    def _embed_native(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        b = self.native_batch
        results: List[np.ndarray] = []
        for start in range(0, len(crops), b):
            chunk = [self._prepare(c) for c in crops[start:start + b]]
            n = len(chunk)
            chunk += [chunk[-1]] * (b - n)  # pad to the fixed batch size
            tensor = np.concatenate(chunk, axis=0)
            out = self.model.predict({self.input_spec.name: tensor})
            arr = np.asarray(out[self.output_name], dtype=np.float32).reshape(b, -1)
            results.append(arr[:n])
        return np.concatenate(results, axis=0)

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        return self.embed_batch([crop_bgr])[0]
