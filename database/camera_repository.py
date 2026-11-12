"""Camera registry (credentials are never stored - URLs are masked)."""
from __future__ import annotations

import time
from typing import Iterable, List, Optional

from config.settings import CameraConfig
from database.connection import Database
from ingestion.camera_pipeline import mask_url


class CameraRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert(self, cam: CameraConfig) -> None:
        now = time.time()
        self.db.execute(
            """INSERT INTO cameras(id, name, url_masked, location, enabled, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET name=excluded.name, url_masked=excluded.url_masked,
                   location=excluded.location, enabled=excluded.enabled, updated_at=excluded.updated_at""",
            (cam.camera_id, cam.name, mask_url(cam.url), cam.location, int(cam.enabled), now, now),
        )

    def sync(self, cameras: Iterable[CameraConfig]) -> int:
        n = 0
        for cam in cameras:
            self.upsert(cam)
            n += 1
        return n

    def list(self) -> List[dict]:
        return [dict(r) for r in self.db.fetchall("SELECT * FROM cameras ORDER BY id")]

    def get(self, camera_id: str) -> Optional[dict]:
        r = self.db.fetchone("SELECT * FROM cameras WHERE id = ?", (camera_id,))
        return dict(r) if r else None
