"""Face embedding storage (float32 BLOBs)."""
from __future__ import annotations

import time
from typing import Iterable, Optional, Tuple

import numpy as np

from database.connection import Database


def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).reshape(-1).tobytes()


def from_blob(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)


class EmbeddingRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(self, person_id: int, vector: np.ndarray, model: str = "arcface", quality: float = 0.0,
            source: Optional[str] = None) -> int:
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        new_id = self.db.insert(
            "INSERT INTO embeddings(person_id, vector, dim, model, quality, source, created_at) VALUES (?,?,?,?,?,?,?)",
            (person_id, to_blob(v), int(v.size), model, float(quality), source, time.time()),
        )
        self.db.execute("UPDATE persons SET updated_at = ? WHERE id = ?", (time.time(), person_id))
        return new_id

    def add_many(self, rows: Iterable[Tuple[int, np.ndarray, float, Optional[str]]], model: str = "arcface") -> int:
        now = time.time()
        payload = [(pid, to_blob(v), int(np.asarray(v).size), model, float(q), src, now) for pid, v, q, src in rows]
        if payload:
            self.db.executemany(
                "INSERT INTO embeddings(person_id, vector, dim, model, quality, source, created_at) VALUES (?,?,?,?,?,?,?)",
                payload,
            )
        return len(payload)

    def load_all(self, active_only: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Return (person_ids (N,), vectors (N, D))."""
        sql = ("SELECT e.person_id, e.vector, e.dim FROM embeddings e JOIN persons p ON p.id = e.person_id"
               + (" WHERE p.active = 1" if active_only else "") + " ORDER BY e.person_id, e.id")
        rows = self.db.fetchall(sql)
        if not rows:
            return np.zeros((0,), dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
        ids = np.fromiter((r["person_id"] for r in rows), dtype=np.int64, count=len(rows))
        vecs = np.stack([from_blob(r["vector"], r["dim"]) for r in rows]).astype(np.float32)
        return ids, vecs

    def count_for(self, person_id: int) -> int:
        r = self.db.fetchone("SELECT COUNT(*) AS n FROM embeddings WHERE person_id = ?", (person_id,))
        return int(r["n"])

    def has_source(self, person_id: int, source: str) -> bool:
        r = self.db.fetchone("SELECT 1 FROM embeddings WHERE person_id = ? AND source = ? LIMIT 1", (person_id, source))
        return r is not None

    def delete_for_person(self, person_id: int) -> int:
        return self.db.execute("DELETE FROM embeddings WHERE person_id = ?", (person_id,)).rowcount

    def fingerprint(self) -> tuple:
        """Changes whenever embeddings or person activity change (used for hot reload)."""
        r = self.db.fetchone(
            "SELECT (SELECT COUNT(*) FROM embeddings) AS n, (SELECT MAX(id) FROM embeddings) AS m, "
            "(SELECT MAX(updated_at) FROM persons) AS u, (SELECT COUNT(*) FROM persons WHERE active = 1) AS a"
        )
        return (r["n"], r["m"], r["u"], r["a"])
