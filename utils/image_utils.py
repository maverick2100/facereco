"""Small image helpers (OpenCV, BGR uint8 everywhere)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


def load_image(path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # honours EXIF orientation
    return img


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def crop_with_margin(img: np.ndarray, bbox: Sequence[float], margin: float = 0.25) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - bw * margin))
    y1 = int(max(0, y1 - bh * margin))
    x2 = int(min(w, x2 + bw * margin))
    y2 = int(min(h, y2 + bh * margin))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return img[y1:y2, x1:x2].copy()


def save_jpeg(path, img: np.ndarray, quality: int = 85) -> bool:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]))


def draw_faces(img: np.ndarray, boxes: Sequence[Sequence[float]], labels: Sequence[str] = ()) -> np.ndarray:
    out = img.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in b[:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if i < len(labels) and labels[i]:
            cv2.putText(out, labels[i], (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out
