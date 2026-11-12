"""In-memory embedding gallery.

Embeddings are kept as one contiguous float32 matrix sorted by person id, so
matching M probe faces against N gallery vectors is a single matrix multiply
(Accelerate BLAS) followed by a per-person max (`np.maximum.reduceat`).
Updates build a new immutable state and swap the reference, so matching
never blocks on a lock.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


@dataclass(frozen=True)
class GalleryState:
    matrix: np.ndarray        # (N, D) float32, rows L2-normalised, sorted by person id
    person_ids: np.ndarray    # (N,)
    unique_ids: np.ndarray    # (P,)
    starts: np.ndarray        # (P,) start row of each person in `matrix`
    version: int

    @property
    def size(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def num_persons(self) -> int:
        return int(self.unique_ids.shape[0])


class EmbeddingGallery:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self._lock = threading.Lock()
        self._state = self._build(np.zeros((0,), np.int64), np.zeros((0, dim), np.float32), 0)

    @staticmethod
    def _build(ids: np.ndarray, vecs: np.ndarray, version: int) -> GalleryState:
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            d = vecs.shape[1] if vecs.ndim == 2 and vecs.shape[1] else 512
            return GalleryState(np.zeros((0, d), np.float32), ids, ids.copy(), ids.copy(), version)
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        mat = np.ascontiguousarray(l2_normalize(vecs[order]))
        unique, starts = np.unique(ids, return_index=True)
        return GalleryState(mat, ids, unique, starts, version)

    def snapshot(self) -> GalleryState:
        return self._state

    def replace(self, person_ids: Sequence[int], vectors: np.ndarray) -> None:
        with self._lock:
            self._state = self._build(np.asarray(person_ids), np.asarray(vectors, np.float32), self._state.version + 1)
            if self._state.size:
                self.dim = self._state.matrix.shape[1]

    def add(self, person_id: int, vectors: np.ndarray) -> None:
        vectors = np.atleast_2d(np.asarray(vectors, np.float32))
        with self._lock:
            s = self._state
            ids = np.concatenate([s.person_ids, np.full(vectors.shape[0], person_id, np.int64)])
            mat = vectors if s.size == 0 else np.vstack([s.matrix, vectors])
            self._state = self._build(ids, mat, s.version + 1)

    def remove_person(self, person_id: int) -> None:
        with self._lock:
            s = self._state
            keep = s.person_ids != person_id
            self._state = self._build(s.person_ids[keep], s.matrix[keep], s.version + 1)

    @property
    def size(self) -> int:
        return self._state.size

    @property
    def num_persons(self) -> int:
        return self._state.num_persons
