"""Cheap face-quality gate run *before* ArcFace so bad crops never cost ANE time."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from processing.face_detection import DetectedFace


@dataclass
class QualityConfig:
    min_face_size: float = 40.0
    blur_threshold: float = 40.0
    max_yaw_ratio: float = 0.45
    min_quality: float = 0.25


@dataclass
class QualityResult:
    score: float
    blur: float
    yaw_ratio: float
    size: float
    passed: bool
    reason: str = ""


def blur_score(crop_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def yaw_ratio(kps: np.ndarray) -> float:
    """|nose offset from eye midpoint| / eye distance. ~0 frontal, >0.5 strong profile."""
    le, re, nose = kps[0], kps[1], kps[2]
    eye_dist = float(np.linalg.norm(re - le))
    if eye_dist < 1e-3:
        return 1.0
    mid = (le + re) / 2.0
    return abs(float(nose[0] - mid[0])) / eye_dist


def assess_quality(face: DetectedFace, crop_bgr: np.ndarray, cfg: QualityConfig) -> QualityResult:
    size = face.size
    blur = blur_score(crop_bgr)
    yaw = yaw_ratio(face.kps)

    size_f = min(1.0, size / max(1.0, cfg.min_face_size * 2.0))
    blur_f = min(1.0, blur / max(1.0, cfg.blur_threshold * 3.0))
    pose_f = max(0.0, 1.0 - yaw / max(1e-3, cfg.max_yaw_ratio * 2.0))
    score = float(face.score * (0.3 + 0.7 * size_f) * (0.3 + 0.7 * blur_f) * (0.3 + 0.7 * pose_f))

    reason = ""
    if size < cfg.min_face_size:
        reason = "small"
    elif blur < cfg.blur_threshold:
        reason = "blurry"
    elif yaw > cfg.max_yaw_ratio:
        reason = "pose"
    elif score < cfg.min_quality:
        reason = "low_score"
    return QualityResult(score=score, blur=blur, yaw_ratio=yaw, size=size, passed=not reason, reason=reason)
