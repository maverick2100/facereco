"""5-point similarity alignment to the ArcFace 112x112 template (InsightFace-compatible)."""
from __future__ import annotations

import cv2
import numpy as np

from constants import ARCFACE_DST, ARCFACE_INPUT_SIZE


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (Umeyama 1991), same as skimage SimilarityTransform."""
    num, dim = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_d = src - src_mean
    dst_d = dst - dst_mean
    A = dst_d.T @ src_d / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1, dtype=np.float64)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.full((dim + 1, dim + 1), np.nan)
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V
    scale = 1.0 / src_d.var(axis=0).sum() * (S @ d)
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale
    return T


def estimate_norm(kps: np.ndarray, image_size: int = ARCFACE_INPUT_SIZE) -> np.ndarray:
    """2x3 affine matrix mapping image landmarks onto the ArcFace template."""
    assert kps.shape == (5, 2), f"expected (5,2) landmarks, got {kps.shape}"
    if image_size % 112 == 0:
        ratio, diff_x = image_size / 112.0, 0.0
    else:
        ratio, diff_x = image_size / 128.0, 8.0 * image_size / 128.0
    dst = ARCFACE_DST * ratio
    dst[:, 0] += diff_x
    T = _umeyama(kps.astype(np.float64), dst.astype(np.float64))
    return T[:2, :].astype(np.float32)


def norm_crop(img: np.ndarray, kps: np.ndarray, image_size: int = ARCFACE_INPUT_SIZE) -> np.ndarray:
    M = estimate_norm(kps, image_size)
    return cv2.warpAffine(img, M, (image_size, image_size), flags=cv2.INTER_LINEAR, borderValue=0.0)
