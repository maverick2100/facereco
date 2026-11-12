"""Recognition events."""
from __future__ import annotations

import json
from typing import Iterable, List, Optional, Sequence

from database.connection import Database

EVENT_COLUMNS = ("camera_id", "person_id", "label", "similarity", "det_score", "quality", "bbox", "snapshot_path", "ts")


class EventRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert_many(self, rows: Iterable[Sequence]) -> int:
        """rows: tuples in EVENT_COLUMNS order (bbox may be a list; it is JSON-encoded)."""
        payload = []
        for r in rows:
            r = list(r)
            if not isinstance(r[6], (str, type(None))):
                r[6] = json.dumps([int(v) for v in r[6]])
            payload.append(r)
        if payload:
            cols = ", ".join(EVENT_COLUMNS)
            marks = ", ".join("?" * len(EVENT_COLUMNS))
            self.db.executemany(f"INSERT INTO events({cols}) VALUES ({marks})", payload)
        return len(payload)

    def query(self, since: Optional[float] = None, until: Optional[float] = None, camera_id: Optional[str] = None,
              person_id: Optional[int] = None, known_only: bool = False, limit: int = 1000) -> List[dict]:
        where, params = [], []
        if since is not None:
            where.append("e.ts >= ?")
            params.append(since)
        if until is not None:
            where.append("e.ts < ?")
            params.append(until)
        if camera_id:
            where.append("e.camera_id = ?")
            params.append(camera_id)
        if person_id is not None:
            where.append("e.person_id = ?")
            params.append(person_id)
        if known_only:
            where.append("e.person_id IS NOT NULL")
        sql = ("SELECT e.*, c.name AS camera_name, c.location AS camera_location FROM events e "
               "LEFT JOIN cameras c ON c.id = e.camera_id")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY e.ts DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [dict(r) for r in self.db.fetchall(sql, params)]

    def count(self) -> int:
        return int(self.db.fetchone("SELECT COUNT(*) AS n FROM events")["n"])

    def delete_older_than(self, ts: float) -> int:
        return self.db.execute("DELETE FROM events WHERE ts < ?", (ts,)).rowcount
