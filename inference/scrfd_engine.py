"""SCRFD face detector on Core ML.

Works with the usual InsightFace SCRFD exports (strides 8/16/32, 2 anchors,
outputs score/bbox/kps per stride). Output names differ between conversions,
so outputs are identified by shape:
    (..., N, 1)  -> scores      (..., N, 4) -> bbox distances
    (..., N, 10) -> landmarks   or raw NCHW (1, A*k, H, W)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from constants import SCRFD_DEFAULT_INPUT, SCRFD_INPUT_MEAN, SCRFD_INPUT_STD
from processing.face_detection import nms
from processing.image_preprocessing import input_size_of, letterbox, to_model_input

log = logging.getLogger(__name__)

Detections = Tuple[np.ndarray, np.ndarray]  # dets (N,5) [x1,y1,x2,y2,score], kps (N,5,2)


@dataclass
class _Level:
    stride: int
    score: str
    bbox: str
    kps: str
    fh: int
    fw: int


def _empty() -> Detections:
    return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 5, 2), dtype=np.float32)


class SCRFDEngine:
    def __init__(self, model: Any, det_threshold: float = 0.5, nms_threshold: float = 0.4,
                 input_mean: float = SCRFD_INPUT_MEAN, input_std: float = SCRFD_INPUT_STD,
                 apply_sigmoid: bool = False, max_faces: int = 0) -> None:
        self.model = model
        self.det_threshold = det_threshold
        self.nms_threshold = nms_threshold
        self.input_mean = input_mean
        self.input_std = input_std
        self.apply_sigmoid = apply_sigmoid
        self.max_faces = max_faces
        self.input_spec = model.inputs[0]
        self.input_w, self.input_h = input_size_of(self.input_spec, SCRFD_DEFAULT_INPUT)
        self.num_anchors = 2
        self._levels: List[_Level] = []
        self._centers: Dict[Tuple[int, int, int], np.ndarray] = {}
        log.info("SCRFD input %dx%d (%s)", self.input_w, self.input_h, self.input_spec.kind)

    @property
    def input_size(self) -> Tuple[int, int]:
        return self.input_w, self.input_h

    # ------------------------------------------------------------------ public
    def preprocess(self, img_bgr: np.ndarray) -> Tuple[Dict[str, Any], float]:
        canvas, scale = letterbox(img_bgr, (self.input_w, self.input_h))
        value = to_model_input(canvas, self.input_spec, self.input_mean, self.input_std)
        return {self.input_spec.name: value}, scale

    def detect_batch(self, images: Sequence[np.ndarray]) -> List[Detections]:
        if not images:
            return []
        prepared = [self.preprocess(img) for img in images]
        outputs = self.model.predict_batch([p[0] for p in prepared])
        return [self.postprocess(out, scale) for out, (_, scale) in zip(outputs, prepared)]

    def detect(self, img_bgr: np.ndarray) -> Detections:
        return self.detect_batch([img_bgr])[0]

    # ------------------------------------------------------------------ decoding
    def _flatten(self, outputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        flat: Dict[str, np.ndarray] = {}
        raw4d: Dict[Tuple[int, int], List[Tuple[str, np.ndarray]]] = {}
        for name, value in outputs.items():
            a = np.asarray(value, dtype=np.float32)
            if a.ndim == 4:  # raw conv head (1, A*k, H, W)
                raw4d.setdefault((a.shape[2], a.shape[3]), []).append((name, a))
            elif a.ndim >= 2:
                flat[name] = a.reshape(-1, a.shape[-1])
            else:
                flat[name] = a.reshape(-1, 1)
        for _, group in raw4d.items():
            anchors = min(a.shape[1] for _, a in group)  # score head has A channels
            for name, a in group:
                c, h, w = a.shape[1:]
                flat[name] = a[0].transpose(1, 2, 0).reshape(h * w * anchors, c // anchors)
        return flat

    def _infer_anchors(self, counts: List[int]) -> int:
        hw = self.input_w * self.input_h
        for a in (2, 1, 3, 4):
            ok = True
            for n in counts:
                s = math.sqrt(hw * a / float(n))
                if abs(s - round(s)) > 0.05 or int(round(s)) & (int(round(s)) - 1):
                    ok = False
                    break
            if ok:
                return a
        return 2

    def _build_levels(self, flat: Dict[str, np.ndarray]) -> List[_Level]:
        by_k: Dict[int, List[Tuple[int, str]]] = {1: [], 4: [], 10: []}
        for name, a in flat.items():
            if a.shape[1] in by_k:
                by_k[a.shape[1]].append((a.shape[0], name))
        if not by_k[1] or not (len(by_k[1]) == len(by_k[4]) == len(by_k[10])):
            shapes = {k: v.shape for k, v in flat.items()}
            raise RuntimeError(f"Unrecognised SCRFD outputs (need score/bbox/kps per stride): {shapes}")
        for k in by_k:
            by_k[k].sort(reverse=True)  # most anchors first == smallest stride
        self.num_anchors = self._infer_anchors([n for n, _ in by_k[1]])
        levels = []
        for (n, s_name), (n4, b_name), (n10, k_name) in zip(by_k[1], by_k[4], by_k[10]):
            if not n == n4 == n10:
                raise RuntimeError(f"SCRFD output size mismatch: {n}/{n4}/{n10}")
            stride = int(round(math.sqrt(self.input_w * self.input_h * self.num_anchors / float(n))))
            fh, fw = math.ceil(self.input_h / stride), math.ceil(self.input_w / stride)
            if fh * fw * self.num_anchors != n:
                fh, fw = self.input_h // stride, self.input_w // stride
            levels.append(_Level(stride, s_name, b_name, k_name, fh, fw))
        log.info("SCRFD heads: %s, anchors=%d",
                 ", ".join(f"s{l.stride}({l.fh}x{l.fw})" for l in levels), self.num_anchors)
        return levels

    def _anchor_centers(self, lv: _Level) -> np.ndarray:
        key = (lv.stride, lv.fh, lv.fw)
        c = self._centers.get(key)
        if c is None:
            ys, xs = np.mgrid[: lv.fh, : lv.fw]
            c = (np.stack([xs, ys], axis=-1).reshape(-1, 2) * lv.stride).astype(np.float32)
            if self.num_anchors > 1:
                c = np.repeat(c, self.num_anchors, axis=0)
            self._centers[key] = c
        return c

    def postprocess(self, outputs: Dict[str, Any], det_scale: float) -> Detections:
        flat = self._flatten(outputs)
        if not self._levels:
            self._levels = self._build_levels(flat)

        all_scores, all_boxes, all_kps = [], [], []
        for lv in self._levels:
            scores = flat[lv.score][:, 0]
            if self.apply_sigmoid:
                scores = 1.0 / (1.0 + np.exp(-scores))
            idx = np.nonzero(scores >= self.det_threshold)[0]
            if idx.size == 0:
                continue
            centers = self._anchor_centers(lv)[idx]
            d = flat[lv.bbox][idx] * lv.stride
            boxes = np.stack([centers[:, 0] - d[:, 0], centers[:, 1] - d[:, 1],
                              centers[:, 0] + d[:, 2], centers[:, 1] + d[:, 3]], axis=1)
            k = flat[lv.kps][idx].reshape(-1, 5, 2) * lv.stride + centers[:, None, :]
            all_scores.append(scores[idx])
            all_boxes.append(boxes)
            all_kps.append(k)

        if not all_scores:
            return _empty()
        scores = np.concatenate(all_scores)
        boxes = np.concatenate(all_boxes) / det_scale
        kps = np.concatenate(all_kps) / det_scale
        keep = nms(boxes, scores, self.nms_threshold)
        if self.max_faces and keep.size > self.max_faces:
            keep = keep[: self.max_faces]
        dets = np.hstack([boxes[keep], scores[keep, None]]).astype(np.float32)
        return dets, kps[keep].astype(np.float32)
