"""Light-weight per-camera person tracker (greedy IoU + centre-distance matching).

Its only job is to count *distinct* people: a person standing in view for a
minute is counted once, not once per frame. A track is counted after
`min_hits` detections, and forgotten after `max_age_s` without a match.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray         # (4,) x1, y1, x2, y2
    last_seen: float         # monotonic seconds
    hits: int = 1
    counted: bool = False


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-6), 0.0)


class PersonTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age_s: float = 2.0, min_hits: int = 2,
                 max_center_shift: float = 0.6) -> None:
        self.iou_threshold = iou_threshold
        self.max_age_s = max_age_s
        self.min_hits = max(1, int(min_hits))
        # At low fps people move far between frames; also accept a match whose centre moved
        # less than this fraction of the box height.
        self.max_center_shift = max_center_shift
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, boxes: np.ndarray, now: float) -> int:
        """Feed one frame's person boxes (N, >=4). Returns how many *new* people were confirmed."""
        self.tracks = [t for t in self.tracks if now - t.last_seen <= self.max_age_s]
        boxes = np.asarray(boxes, dtype=np.float32)
        boxes = boxes[:, :4] if boxes.size else np.zeros((0, 4), np.float32)
        matched_tracks = set()
        unmatched = set(range(boxes.shape[0]))
        if self.tracks and boxes.shape[0]:
            tb = np.stack([t.bbox for t in self.tracks])
            score = _iou_matrix(tb, boxes)
            # Centre-distance fallback, scored below any real IoU match.
            tc = (tb[:, :2] + tb[:, 2:]) / 2.0
            bc = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            dist = np.linalg.norm(tc[:, None, :] - bc[None, :, :], axis=2)
            th = np.maximum(tb[:, 3] - tb[:, 1], 1.0)[:, None]
            near = (score < self.iou_threshold) & (dist <= self.max_center_shift * th)
            score = np.where(near, 0.01 * (1.0 - dist / (self.max_center_shift * th)), score)
            score = np.where((score >= self.iou_threshold) | near, score, -1.0)
            for flat in np.argsort(-score, axis=None):
                ti, bi = np.unravel_index(flat, score.shape)
                if score[ti, bi] < 0:
                    break
                if ti in matched_tracks or bi not in unmatched:
                    continue
                t = self.tracks[ti]
                matched_tracks.add(ti)
                t.bbox, t.last_seen = boxes[bi].copy(), now
                t.hits += 1
                unmatched.discard(bi)

        for bi in sorted(unmatched):
            self.tracks.append(Track(self._next_id, boxes[bi].copy(), now))
            self._next_id += 1

        new = 0
        for t in self.tracks:
            if not t.counted and t.hits >= self.min_hits:
                t.counted = True
                new += 1
        return new

    @property
    def active(self) -> int:
        return len(self.tracks)
