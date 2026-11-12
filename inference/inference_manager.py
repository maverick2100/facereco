"""Owns both Core ML models and exposes timed, batched detect/embed calls."""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence

import numpy as np

from config.settings import DetectionSettings, ModelSettings
from constants import MODEL_CACHE_DIR
from inference.ane_config import log_environment
from inference.arcface_engine import ArcFaceEngine
from inference.coreml_loader import CoreMLModel
from inference.scrfd_engine import Detections, SCRFDEngine
from monitoring.performance import PerformanceTracker

log = logging.getLogger(__name__)


class InferenceManager:
    def __init__(self, models: ModelSettings, detection: DetectionSettings,
                 perf: Optional[PerformanceTracker] = None) -> None:
        self.models_cfg = models
        self.det_cfg = detection
        self.perf = perf or PerformanceTracker()
        self.scrfd: Optional[SCRFDEngine] = None
        self.arcface: Optional[ArcFaceEngine] = None
        self._scrfd_model: Optional[CoreMLModel] = None
        self._arcface_model: Optional[CoreMLModel] = None

    def load(self, detector: bool = True, recognizer: bool = True) -> "InferenceManager":
        log_environment()
        cache = MODEL_CACHE_DIR if self.models_cfg.cache_compiled else None
        cu = self.models_cfg.compute_units
        if detector:
            self._scrfd_model = CoreMLModel(self.models_cfg.scrfd_path, cu, cache, name="scrfd")
            self.scrfd = SCRFDEngine(
                self._scrfd_model,
                det_threshold=self.det_cfg.det_threshold,
                nms_threshold=self.det_cfg.nms_threshold,
                input_mean=self.models_cfg.scrfd_input_mean,
                input_std=self.models_cfg.scrfd_input_std,
                apply_sigmoid=self.models_cfg.scrfd_apply_sigmoid,
                max_faces=self.det_cfg.max_faces_per_frame,
            )
        if recognizer:
            self._arcface_model = CoreMLModel(self.models_cfg.arcface_path, cu, cache, name="arcface")
            self.arcface = ArcFaceEngine(
                self._arcface_model,
                input_mean=self.models_cfg.arcface_input_mean,
                input_std=self.models_cfg.arcface_input_std,
            )
        self.warmup(self.models_cfg.warmup_iterations)
        return self

    def warmup(self, iterations: int = 3) -> None:
        """First predictions trigger ANE program compilation; do it before cameras start."""
        t0 = time.perf_counter()
        for _ in range(max(0, iterations)):
            if self.scrfd is not None:
                w, h = self.scrfd.input_size
                self.scrfd.detect_batch([np.zeros((h, w, 3), dtype=np.uint8)] * max(1, self.det_cfg.batch_size))
            if self.arcface is not None:
                self.arcface.embed_batch([np.zeros((112, 112, 3), dtype=np.uint8)] * 4)
        if iterations:
            log.info("Model warm-up done in %.2fs", time.perf_counter() - t0)

    # ------------------------------------------------------------------ inference
    def detect(self, images: Sequence[np.ndarray]) -> List[Detections]:
        assert self.scrfd is not None, "detector not loaded"
        t0 = time.perf_counter()
        res = self.scrfd.detect_batch(images)
        dt = time.perf_counter() - t0
        self.perf.record_latency("scrfd_batch_ms", dt * 1000.0)
        self.perf.record_latency("scrfd_per_frame_ms", dt * 1000.0 / max(1, len(images)))
        self.perf.add_busy("inference", dt)
        self.perf.add_busy("scrfd", dt)
        return res

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        assert self.arcface is not None, "recognizer not loaded"
        t0 = time.perf_counter()
        emb = self.arcface.embed_batch(crops)
        dt = time.perf_counter() - t0
        self.perf.record_latency("arcface_batch_ms", dt * 1000.0)
        self.perf.record_latency("arcface_per_face_ms", dt * 1000.0 / max(1, len(crops)))
        self.perf.add_busy("inference", dt)
        self.perf.add_busy("arcface", dt)
        return emb

    def info(self) -> dict:
        return {
            "compute_units": self.models_cfg.compute_units,
            "scrfd": self._scrfd_model.info() if self._scrfd_model else None,
            "arcface": self._arcface_model.info() if self._arcface_model else None,
        }
