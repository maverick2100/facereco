"""Person-count rows (appended once per camera per interval)."""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from database.connection import Database

COUNT_COLUMNS = ("camera_id", "period_start", "period_end", "frames", "current_count", "min_count",
                 "max_count", "avg_count", "new_persons", "ts")


class PersonCountRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert_many(self, rows: Iterable[Sequence]) -> int:
        """rows: tuples in COUNT_COLUMNS order."""
        payload = [tuple(r) for r in rows]
        if payload:
            cols = ", ".join(COUNT_COLUMNS)
            marks = ", ".join("?" * len(COUNT_COLUMNS))
            self.db.executemany(f"INSERT INTO person_counts({cols}) VALUES ({marks})", payload)
        return len(payload)

    @staticmethod
    def _where(since: Optional[float], until: Optional[float], camera_id: Optional[str]):
        where, params = [], []
        if since is not None:
            where.append("p.ts >= ?")
            params.append(since)
        if until is not None:
            where.append("p.ts < ?")
            params.append(until)
        if camera_id:
            where.append("p.camera_id = ?")
            params.append(camera_id)
        return (" WHERE " + " AND ".join(where)) if where else "", params

    def query(self, since: Optional[float] = None, until: Optional[float] = None,
              camera_id: Optional[str] = None, limit: int = 1000) -> List[dict]:
        where, params = self._where(since, until, camera_id)
        sql = ("SELECT p.*, c.name AS camera_name, c.location AS camera_location FROM person_counts p "
               "LEFT JOIN cameras c ON c.id = p.camera_id" + where + " ORDER BY p.ts DESC")
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [dict(r) for r in self.db.fetchall(sql, params)]

    def totals(self, since: Optional[float] = None, until: Optional[float] = None,
               camera_id: Optional[str] = None) -> List[dict]:
        """Per camera: total distinct people, peak occupancy, frames analysed."""
        where, params = self._where(since, until, camera_id)
        sql = ("SELECT p.camera_id, c.name AS camera_name, SUM(p.new_persons) AS total_persons, "
               "MAX(p.max_count) AS peak_count, SUM(p.frames) AS frames, "
               "MIN(p.period_start) AS first_ts, MAX(p.period_end) AS last_ts "
               "FROM person_counts p LEFT JOIN cameras c ON c.id = p.camera_id" + where +
               " GROUP BY p.camera_id ORDER BY p.camera_id")
        return [dict(r) for r in self.db.fetchall(sql, params)]

    def delete_older_than(self, ts: float) -> int:
        return self.db.execute("DELETE FROM person_counts WHERE ts < ?", (ts,)).rowcount
