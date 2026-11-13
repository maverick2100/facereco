"""Database schema + initialisation."""
from __future__ import annotations

import logging
import time

from constants import SCHEMA_VERSION
from database.connection import Database

log = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS cameras (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    url_masked  TEXT,
    location    TEXT,
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS persons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    external_id TEXT UNIQUE,
    metadata    TEXT,
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(name);

CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    vector      BLOB NOT NULL,
    dim         INTEGER NOT NULL,
    model       TEXT,
    quality     REAL,
    source      TEXT,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings(person_id);

CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id     TEXT NOT NULL,
    person_id     INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    label         TEXT NOT NULL,
    similarity    REAL,
    det_score     REAL,
    quality       REAL,
    bbox          TEXT,
    snapshot_path TEXT,
    ts            REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_camera_ts ON events(camera_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_person_ts ON events(person_id, ts);

-- One row per camera per interval (FR_PERSON_FLUSH_S), appended by person_detection/.
CREATE TABLE IF NOT EXISTS person_counts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id     TEXT NOT NULL,
    period_start  REAL NOT NULL,
    period_end    REAL NOT NULL,
    frames        INTEGER NOT NULL,   -- frames analysed in the interval
    current_count INTEGER NOT NULL,   -- people in the last analysed frame
    min_count     INTEGER NOT NULL,
    max_count     INTEGER NOT NULL,   -- peak occupancy in the interval
    avg_count     REAL NOT NULL,
    new_persons   INTEGER NOT NULL,   -- distinct people that appeared (tracked) in the interval
    ts            REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_person_counts_ts ON person_counts(ts);
CREATE INDEX IF NOT EXISTS idx_person_counts_camera_ts ON person_counts(camera_id, ts);
"""


def init_schema(db: Database) -> None:
    db.executescript(SCHEMA_SQL)
    row = db.fetchone("SELECT MAX(version) AS v FROM schema_version")
    current = row["v"] if row and row["v"] is not None else 0
    if current < SCHEMA_VERSION:
        db.execute("INSERT OR REPLACE INTO schema_version(version, applied_at) VALUES (?, ?)",
                   (SCHEMA_VERSION, time.time()))
        log.info("Database schema at version %d (%s)", SCHEMA_VERSION, db.path)
