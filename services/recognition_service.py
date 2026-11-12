"""The real-time pipeline.

  FrameScheduler --(batch of frames)--> [detector thread]
        SCRFD (ANE) -> filter -> track-cache skip -> align -> quality gate
        --> InferenceQueue of face crops --> [recognizer thread]
        ArcFace (ANE, batched) -> match (BLAS) -> de-dup -> EventService

Only two Python threads touch the models; everything else is queues.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from config.settings import CameraConfig, DetectionSettings, RecognitionSettings
from constants import UNKNOWN_PERSON_ID
from inference.inference_manager import InferenceManager
from monitoring.performance import PerformanceTracker
from processing.face_alignment import norm_crop
from processing.face_detection import DetectedFace, faces_from_detections, filter_faces
from processing.face_quality import QualityConfig, assess_quality
from recognition.duplicate_filter import DuplicateFilter, TrackCache
from recognition.face_matcher import FaceMatcher
from recognition.identity_manager import IdentityManager
from scheduler.batch_scheduler import BatchScheduler
from scheduler.camera_queue import Frame
from scheduler.frame_scheduler import FrameScheduler
from scheduler.inference_queue import InferenceQueue
from services.event_service import EventService, RecognitionEvent
from utils.image_utils import crop_with_margin

log = logging.getLogger(__name__)


@dataclass
class FaceTask:
    camera_id: str
    frame_id: int
    timestamp: float
    face: DetectedFace
    crop: np.ndarray            # aligned 112x112 BGR
    quality: float
    snapshot: Optional[np.ndarray] = None


class RecognitionService:
    def __init__(self, inference: InferenceManager, frames: FrameScheduler, identity: IdentityManager,
                 matcher: FaceMatcher, duplicates: DuplicateFilter, tracks: TrackCache, events: EventService,
                 cameras: List[CameraConfig], det_cfg: DetectionSettings, rec_cfg: RecognitionSettings,
                 perf: Optional[PerformanceTracker] = None, save_snapshots: bool = False) -> None:
        self.inference = inference
        self.frames = frames
        self.identity = identity
        self.matcher = matcher
        self.duplicates = duplicates
        self.tracks = tracks
        self.events = events
        self.cameras: Dict[str, CameraConfig] = {c.camera_id: c for c in cameras}
        self.det_cfg = det_cfg
        self.rec_cfg = rec_cfg
        self.perf = perf or PerformanceTracker()
        self.save_snapshots = save_snapshots
        self.quality_cfg = QualityConfig(
            min_face_size=det_cfg.min_face_size,
            blur_threshold=rec_cfg.blur_threshold,
            max_yaw_ratio=rec_cfg.max_yaw_ratio,
            min_quality=rec_cfg.min_quality,
        )
        self.rec_queue: InferenceQueue[FaceTask] = InferenceQueue(rec_cfg.queue_size)
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._stop.clear()
        for target, name in ((self._detection_loop, "detector"), (self._recognition_loop, "recognizer")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        log.info("Recognition pipeline started (det batch %d, rec batch %d)",
                 self.det_cfg.batch_size, self.rec_cfg.batch_size)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=10)
        self._threads.clear()

    # ------------------------------------------------------------------ detection
    def _detection_loop(self) -> None:
        batcher = BatchScheduler(self.frames.get_frames, self.det_cfg.batch_size, self.det_cfg.max_wait_ms)
        while not self._stop.is_set():
            frames = batcher.next_batch(self._stop)
            if not frames:
                continue
            try:
                results = self.inference.detect([f.image for f in frames])
            except Exception:
                log.exception("Detection failed")
                time.sleep(0.5)
                continue
            self.perf.incr("frames_processed", len(frames))
            now = time.monotonic()
            for frame, (dets, kps) in zip(frames, results):
                try:
                    self._handle_detections(frame, dets, kps, now)
                except Exception:
                    log.exception("[%s] post-detection failed", frame.camera_id)

    def _handle_detections(self, frame: Frame, dets: np.ndarray, kps: np.ndarray, now: float) -> None:
        if dets.shape[0] == 0:
            return
        cam = self.cameras.get(frame.camera_id)
        faces = filter_faces(faces_from_detections(dets, kps), min_size=self.det_cfg.min_face_size,
                             roi=cam.roi if cam else None, max_faces=self.det_cfg.max_faces_per_frame)
        self.perf.incr("faces_detected", len(faces))
        for face in faces:
            hit = self.tracks.lookup(frame.camera_id, face.bbox, now)
            if hit is not None:
                self.duplicates.touch(frame.camera_id, hit[0], now)
                self.perf.incr("faces_track_reused")
                continue
            crop = norm_crop(frame.image, face.kps)
            q = assess_quality(face, crop, self.quality_cfg)
            if not q.passed:
                self.perf.incr(f"faces_rejected_{q.reason}")
                continue
            snap = crop_with_margin(frame.image, face.bbox, 0.4) if self.save_snapshots else None
            task = FaceTask(frame.camera_id, frame.frame_id, frame.timestamp, face, crop, q.score, snap)
            if not self.rec_queue.put(task):
                self.perf.incr("faces_dropped_backpressure")

    # ------------------------------------------------------------------ recognition
    def _recognition_loop(self) -> None:
        batcher = BatchScheduler(self.rec_queue.get_batch, self.rec_cfg.batch_size, self.rec_cfg.max_wait_ms)
        while not self._stop.is_set():
            try:
                self.identity.maybe_reload()
            except Exception:
                log.exception("Gallery reload failed")
            tasks = batcher.next_batch(self._stop)
            if not tasks:
                continue
            try:
                emb = self.inference.embed([t.crop for t in tasks])
                matches = self.matcher.match_batch(emb)
            except Exception:
                log.exception("Embedding/matching failed")
                time.sleep(0.5)
                continue
            self.perf.incr("faces_embedded", len(tasks))
            now = time.monotonic()
            for i, (task, m) in enumerate(zip(tasks, matches)):
                pid = m.person_id if m.is_known else UNKNOWN_PERSON_ID
                self.tracks.update(task.camera_id, task.face.bbox, pid, m.similarity, now)
                if pid == UNKNOWN_PERSON_ID and not self.rec_cfg.record_unknown:
                    continue
                if not self.duplicates.should_emit(task.camera_id, pid, emb[i], now):
                    self.perf.incr("events_suppressed")
                    continue
                self.perf.incr("recognized" if m.is_known else "unknown")
                self.events.submit(RecognitionEvent(
                    camera_id=task.camera_id,
                    person_id=None if pid == UNKNOWN_PERSON_ID else pid,
                    label=self.identity.name_of(pid),
                    similarity=m.similarity,
                    det_score=task.face.score,
                    quality=task.quality,
                    bbox=task.face.bbox_int(),
                    ts=task.timestamp,
                    snapshot=task.snapshot,
                ))

    def stats(self) -> dict:
        return {"rec_queue": self.rec_queue.stats(), "track_cache": self.tracks.stats()}
