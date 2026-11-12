"""Event de-duplication and recognition caching.

DuplicateFilter: emit one event per (camera, person) until the person has been
    absent for `cooldown_s`. Unknown faces are de-duplicated by embedding
    similarity.
TrackCache: if a face overlaps (IoU) a face identified on the same camera a
    moment ago, reuse that identity and skip ArcFace altogether. Each track is
    re-verified every `reverify_s`, so mistakes do not persist. With people
    standing in view this removes most ArcFace calls.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from constants import UNKNOWN_PERSON_ID
from processing.face_detection import iou


class DuplicateFilter:
    def __init__(self, cooldown_s: float = 10.0, unknown_cooldown_s: float = 30.0,
                 unknown_similarity: float = 0.5, max_unknown_per_camera: int = 200) -> None:
        self.cooldown_s = cooldown_s
        self.unknown_cooldown_s = unknown_cooldown_s
        self.unknown_similarity = unknown_similarity
        self.max_unknown = max_unknown_per_camera
        self._last_seen: Dict[Tuple[str, int], float] = {}
        self._unknown: Dict[str, List[Tuple[np.ndarray, float]]] = {}
        self._lock = threading.Lock()
        self._last_prune = time.monotonic()

    def touch(self, camera_id: str, person_id: int, now: Optional[float] = None) -> None:
        """Person still present (e.g. via track cache) - extend the cooldown."""
        if person_id == UNKNOWN_PERSON_ID:
            return
        with self._lock:
            self._last_seen[(camera_id, person_id)] = time.monotonic() if now is None else now

    def should_emit(self, camera_id: str, person_id: int, embedding: Optional[np.ndarray] = None,
                    now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._maybe_prune(now)
            if person_id != UNKNOWN_PERSON_ID:
                key = (camera_id, person_id)
                last = self._last_seen.get(key)
                self._last_seen[key] = now
                return last is None or (now - last) >= self.cooldown_s
            return self._unknown_should_emit(camera_id, embedding, now)

    def _unknown_should_emit(self, camera_id: str, embedding: Optional[np.ndarray], now: float) -> bool:
        if embedding is None:
            return True
        entries = [(e, t) for e, t in self._unknown.get(camera_id, []) if now - t < self.unknown_cooldown_s]
        if entries:
            mat = np.stack([e for e, _ in entries])
            sims = mat @ embedding
            j = int(sims.argmax())
            if sims[j] >= self.unknown_similarity:
                entries[j] = (entries[j][0], now)
                self._unknown[camera_id] = entries
                return False
        entries.append((np.asarray(embedding, np.float32), now))
        self._unknown[camera_id] = entries[-self.max_unknown:]
        return True

    def _maybe_prune(self, now: float) -> None:
        if now - self._last_prune < 60:
            return
        self._last_prune = now
        horizon = max(self.cooldown_s, 1.0) * 3
        self._last_seen = {k: t for k, t in self._last_seen.items() if now - t < horizon}
        for cam in list(self._unknown):
            self._unknown[cam] = [(e, t) for e, t in self._unknown[cam] if now - t < self.unknown_cooldown_s]


@dataclass
class _Track:
    bbox: np.ndarray
    person_id: int
    similarity: float
    verified: float   # last time ArcFace confirmed it
    seen: float       # last time it was matched by IoU


class TrackCache:
    def __init__(self, iou_threshold: float = 0.45, reverify_s: float = 2.0, ttl_s: float = 1.5) -> None:
        self.iou_threshold = iou_threshold
        self.reverify_s = reverify_s
        self.ttl_s = ttl_s
        self._tracks: Dict[str, List[_Track]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _best(self, tracks: List[_Track], bbox: Sequence[float], now: float) -> Tuple[Optional[_Track], float]:
        best, best_iou = None, 0.0
        for t in tracks:
            if now - t.seen > self.ttl_s:
                continue
            v = iou(t.bbox, bbox)
            if v > best_iou:
                best, best_iou = t, v
        return best, best_iou

    def lookup(self, camera_id: str, bbox: Sequence[float], now: Optional[float] = None) -> Optional[Tuple[int, float]]:
        """Return (person_id, similarity) if this face can skip ArcFace, else None."""
        if self.reverify_s <= 0:
            return None
        now = time.monotonic() if now is None else now
        with self._lock:
            t, v = self._best(self._tracks.get(camera_id, []), bbox, now)
            if t is not None and v >= self.iou_threshold and (now - t.verified) < self.reverify_s:
                t.bbox = np.asarray(bbox[:4], np.float32)
                t.seen = now
                self.hits += 1
                return t.person_id, t.similarity
        self.misses += 1
        return None

    def update(self, camera_id: str, bbox: Sequence[float], person_id: int, similarity: float,
               now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        box = np.asarray(bbox[:4], np.float32)
        with self._lock:
            tracks = [t for t in self._tracks.get(camera_id, []) if now - t.seen <= self.ttl_s]
            t, v = self._best(tracks, box, now)
            if t is not None and v >= self.iou_threshold:
                t.bbox, t.person_id, t.similarity, t.verified, t.seen = box, person_id, similarity, now, now
            else:
                tracks.append(_Track(box, person_id, similarity, now, now))
            self._tracks[camera_id] = tracks

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "hit_rate": round(self.hits / total, 3) if total else 0.0}
