"""Cosine-similarity matching of probe embeddings against the gallery."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from constants import UNKNOWN_PERSON_ID
from recognition.embedding_manager import EmbeddingGallery


@dataclass
class MatchResult:
    person_id: int
    similarity: float
    second_similarity: float
    is_known: bool


class FaceMatcher:
    def __init__(self, gallery: EmbeddingGallery, threshold: float = 0.45, margin: float = 0.05) -> None:
        self.gallery = gallery
        self.threshold = threshold
        self.margin = margin

    def match_batch(self, embeddings: np.ndarray) -> List[MatchResult]:
        emb = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        state = self.gallery.snapshot()
        if emb.shape[0] == 0:
            return []
        if state.size == 0 or state.matrix.shape[1] != emb.shape[1]:
            return [MatchResult(UNKNOWN_PERSON_ID, 0.0, 0.0, False) for _ in range(emb.shape[0])]

        sims = emb @ state.matrix.T                                   # (M, N)
        per_person = np.maximum.reduceat(sims, state.starts, axis=1)  # (M, P)
        if per_person.shape[1] >= 2:
            top2 = np.partition(per_person, -2, axis=1)[:, -2:]
            second = top2[:, 0]
        else:
            second = np.full(per_person.shape[0], -1.0, dtype=np.float32)
        best_idx = per_person.argmax(axis=1)
        best = per_person[np.arange(per_person.shape[0]), best_idx]

        out = []
        for i in range(emb.shape[0]):
            b, s = float(best[i]), float(second[i])
            known = b >= self.threshold and (s < 0 or (b - s) >= self.margin)
            pid = int(state.unique_ids[best_idx[i]]) if known else UNKNOWN_PERSON_ID
            out.append(MatchResult(pid, b, max(s, 0.0), known))
        return out

    def match(self, embedding: np.ndarray) -> MatchResult:
        return self.match_batch(embedding[None])[0]
