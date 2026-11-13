"""Person CRUD."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from database.connection import Database


class PersonRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    @staticmethod
    def _row(r) -> Optional[dict]:
        if r is None:
            return None
        d = dict(r)
        d["metadata"] = json.loads(d["metadata"]) if d.get("metadata") else {}
        return d

    def create(self, name: str, external_id: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None) -> int:
        now = time.time()
        return self.db.insert(
            "INSERT INTO persons(name, external_id, metadata, active, created_at, updated_at) VALUES (?,?,?,1,?,?)",
            (name, external_id, json.dumps(metadata or {}), now, now),
        )

    def get(self, person_id: int) -> Optional[dict]:
        return self._row(self.db.fetchone("SELECT * FROM persons WHERE id = ?", (person_id,)))

    def get_by_name(self, name: str) -> Optional[dict]:
        return self._row(self.db.fetchone("SELECT * FROM persons WHERE name = ? ORDER BY id LIMIT 1", (name,)))

    def get_by_external_id(self, external_id: str) -> Optional[dict]:
        return self._row(self.db.fetchone("SELECT * FROM persons WHERE external_id = ?", (external_id,)))

    def get_or_create(self, name: str, external_id: Optional[str] = None) -> int:
        existing = self.get_by_external_id(external_id) if external_id else self.get_by_name(name)
        if existing:
            if not existing["active"]:
                self.set_active(existing["id"], True)
            return int(existing["id"])
        return self.create(name, external_id)

    def list(self, active_only: bool = True) -> List[dict]:
        sql = "SELECT * FROM persons" + (" WHERE active = 1" if active_only else "") + " ORDER BY name"
        return [self._row(r) for r in self.db.fetchall(sql)]

    def rename(self, person_id: int, name: str) -> None:
        self.db.execute("UPDATE persons SET name = ?, updated_at = ? WHERE id = ?", (name, time.time(), person_id))

    def set_active(self, person_id: int, active: bool) -> None:
        self.db.execute("UPDATE persons SET active = ?, updated_at = ? WHERE id = ?",
                        (int(active), time.time(), person_id))

    def delete(self, person_id: int) -> bool:
        cur = self.db.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        return cur.rowcount > 0
