"""IN / OUT counting for one drawn region (polygon) per camera.

Each tracked person is reduced to an anchor point (by default the bottom-centre of the box,
i.e. the feet, which is what actually crosses a doorway on an angled camera). Every frame we
test whether that point is inside the polygon:

    outside -> inside  = one IN
    inside  -> outside = one OUT

A side change only counts once it has held for `min_frames` consecutive frames, so a person
standing on the edge (box jitter) doesn't produce IN/OUT/IN/OUT. A person first seen already
inside (or outside) the zone sets the starting side but isn't counted: we didn't see them cross.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

IN = "in"
OUT = "out"
ANCHORS = ("bottom", "center")


def points_in_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd ray casting. points (N, 2), poly (M, 2) -> bool (N,)."""
    if points.shape[0] == 0:
        return np.zeros(0, bool)
    x, y = points[:, 0:1], points[:, 1:2]                 # (N, 1)
    x1, y1 = poly[:, 0], poly[:, 1]                       # (M,)
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    crosses = (y1 > y) != (y2 > y)                        # edge straddles the horizontal ray
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return np.count_nonzero(crosses & (x < x_at), axis=1) % 2 == 1


def anchor_points(boxes: np.ndarray, anchor: str = "bottom") -> np.ndarray:
    boxes = np.asarray(boxes, np.float32)
    if boxes.size == 0:
        return np.zeros((0, 2), np.float32)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = boxes[:, 3] if anchor == "bottom" else (boxes[:, 1] + boxes[:, 3]) / 2.0
    return np.stack([cx, cy], axis=1)


@dataclass
class Crossing:
    camera_id: str
    track_id: int
    direction: str           # "in" | "out"
    ts: float                # epoch seconds
    x: float                 # anchor point at the moment the crossing was confirmed
    y: float

    def as_row(self) -> Tuple:
        return (self.camera_id, self.track_id, self.direction, self.ts, round(self.x, 1), round(self.y, 1))


@dataclass
class _State:
    inside: bool
    pending: Optional[bool] = None
    streak: int = 0


class ZoneCounter:
    def __init__(self, camera_id: str, polygon: Sequence[Sequence[float]], anchor: str = "bottom",
                 min_frames: int = 2) -> None:
        if anchor not in ANCHORS:
            raise ValueError(f"zone anchor must be one of {ANCHORS}, got {anchor!r}")
        self.camera_id = camera_id
        self.polygon = np.asarray(polygon, np.float32)
        if self.polygon.ndim != 2 or self.polygon.shape[0] < 3 or self.polygon.shape[1] != 2:
            raise ValueError(f"camera {camera_id}: count_zone needs at least 3 [x, y] points")
        self.anchor = anchor
        self.min_frames = max(1, int(min_frames))
        self._states: Dict[int, _State] = {}
        self.inside_now = 0
        self.total_in = 0
        self.total_out = 0

    def update(self, track_ids: Iterable[int], boxes: np.ndarray, ts: float,
               alive_ids: Optional[Iterable[int]] = None) -> List[Crossing]:
        """Feed the tracks matched in this frame. `alive_ids` = every track the tracker still keeps
        (state for the others is dropped). Returns the crossings confirmed on this frame."""
        ids = list(track_ids)
        pts = anchor_points(boxes, self.anchor) if ids else np.zeros((0, 2), np.float32)
        inside = points_in_polygon(pts, self.polygon)
        out: List[Crossing] = []
        for tid, (px, py), now_in in zip(ids, pts, inside):
            now_in = bool(now_in)
            st = self._states.get(tid)
            if st is None:
                self._states[tid] = _State(now_in)
                continue
            if now_in == st.inside:
                st.pending, st.streak = None, 0
                continue
            if st.pending is now_in:
                st.streak += 1
            else:
                st.pending, st.streak = now_in, 1
            if st.streak >= self.min_frames:
                st.inside, st.pending, st.streak = now_in, None, 0
                direction = IN if now_in else OUT
                if now_in:
                    self.total_in += 1
                else:
                    self.total_out += 1
                out.append(Crossing(self.camera_id, int(tid), direction, ts, float(px), float(py)))
        if alive_ids is not None:
            alive = set(alive_ids)
            for tid in [t for t in self._states if t not in alive]:
                del self._states[tid]
        self.inside_now = sum(1 for s in self._states.values() if s.inside)
        return out
