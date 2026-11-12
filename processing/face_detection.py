"""Detected-face container and filtering."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class DetectedFace:
    bbox: np.ndarray   # (4,) x1, y1, x2, y2
    score: float
    kps: np.ndarray    # (5, 2) landmarks

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def size(self) -> float:
        return min(self.width, self.height)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0])

    def bbox_int(self) -> List[int]:
        return [int(round(v)) for v in self.bbox[:4]]


def faces_from_detections(dets: np.ndarray, kps: np.ndarray) -> List[DetectedFace]:
    return [DetectedFace(dets[i, :4].astype(np.float32), float(dets[i, 4]), kps[i].astype(np.float32))
            for i in range(dets.shape[0])]


def filter_faces(faces: Sequence[DetectedFace], min_size: float = 0, roi: Optional[Sequence[int]] = None,
                 max_faces: int = 0) -> List[DetectedFace]:
    out = []
    for f in faces:
        if f.size < min_size:
            continue
        if roi is not None:
            cx, cy = f.center
            if not (roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]):
                continue
        out.append(f)
    out.sort(key=lambda f: f.score, reverse=True)
    if max_faces and len(out) > max_faces:
        out = out[:max_faces]
    return out


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """Greedy NMS; returns kept indices sorted by score."""
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        ovr = (w * h) / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(ovr <= threshold)[0] + 1]
    return np.asarray(keep, dtype=np.int64)
