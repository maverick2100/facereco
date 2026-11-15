"""Database schema + initialisation (SQLite and PostgreSQL)."""
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
    in_count      INTEGER NOT NULL DEFAULT 0,  -- count_zone: outside -> inside crossings in the interval
    out_count     INTEGER NOT NULL DEFAULT 0,  -- count_zone: inside -> outside crossings
    zone_count    INTEGER NOT NULL DEFAULT 0,  -- people inside the zone at the end of the interval
    ts            REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_person_counts_ts ON person_counts(ts);
CREATE INDEX IF NOT EXISTS idx_person_counts_camera_ts ON person_counts(camera_id, ts);

-- One row per confirmed zone crossing (written every few seconds, in batches).
CREATE TABLE IF NOT EXISTS person_crossings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT NOT NULL,
    track_id    INTEGER NOT NULL,
    direction   TEXT NOT NULL CHECK (direction IN ('in', 'out')),
    ts          REAL NOT NULL,
    x           REAL,
    y           REAL
);
CREATE INDEX IF NOT EXISTS idx_person_crossings_camera_ts ON person_crossings(camera_id, ts);
CREATE INDEX IF NOT EXISTS idx_person_crossings_ts ON person_crossings(ts);
"""


# Same tables for PostgreSQL. Timestamps stay epoch seconds, as DOUBLE PRECISION
# (PostgreSQL REAL is only 4 bytes, which would round timestamps to minutes).
PG_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS cameras (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    url_masked  TEXT,
    location    TEXT,
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  DOUBLE PRECISION NOT NULL,
    updated_at  DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS persons (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    external_id TEXT UNIQUE,
    metadata    TEXT,
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  DOUBLE PRECISION NOT NULL,
    updated_at  DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(name);

CREATE TABLE IF NOT EXISTS embeddings (
    id          BIGSERIAL PRIMARY KEY,
    person_id   BIGINT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    vector      BYTEA NOT NULL,
    dim         INTEGER NOT NULL,
    model       TEXT,
    quality     DOUBLE PRECISION,
    source      TEXT,
    created_at  DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings(person_id);

CREATE TABLE IF NOT EXISTS events (
    id            BIGSERIAL PRIMARY KEY,
    camera_id     TEXT NOT NULL,
    person_id     BIGINT REFERENCES persons(id) ON DELETE SET NULL,
    label         TEXT NOT NULL,
    similarity    DOUBLE PRECISION,
    det_score     DOUBLE PRECISION,
    quality       DOUBLE PRECISION,
    bbox          TEXT,
    snapshot_path TEXT,
    ts            DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_camera_ts ON events(camera_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_person_ts ON events(person_id, ts);

CREATE TABLE IF NOT EXISTS person_counts (
    id            BIGSERIAL PRIMARY KEY,
    camera_id     TEXT NOT NULL,
    period_start  DOUBLE PRECISION NOT NULL,
    period_end    DOUBLE PRECISION NOT NULL,
    frames        INTEGER NOT NULL,
    current_count INTEGER NOT NULL,
    min_count     INTEGER NOT NULL,
    max_count     INTEGER NOT NULL,
    avg_count     DOUBLE PRECISION NOT NULL,
    new_persons   INTEGER NOT NULL,
    in_count      INTEGER NOT NULL DEFAULT 0,
    out_count     INTEGER NOT NULL DEFAULT 0,
    zone_count    INTEGER NOT NULL DEFAULT 0,
    ts            DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_person_counts_ts ON person_counts(ts);
CREATE INDEX IF NOT EXISTS idx_person_counts_camera_ts ON person_counts(camera_id, ts);

CREATE TABLE IF NOT EXISTS person_crossings (
    id          BIGSERIAL PRIMARY KEY,
    camera_id   TEXT NOT NULL,
    track_id    INTEGER NOT NULL,
    direction   TEXT NOT NULL CHECK (direction IN ('in', 'out')),
    ts          DOUBLE PRECISION NOT NULL,
    x           REAL,
    y           REAL
);
CREATE INDEX IF NOT EXISTS idx_person_crossings_camera_ts ON person_crossings(camera_id, ts);
CREATE INDEX IF NOT EXISTS idx_person_crossings_ts ON person_crossings(ts);
"""

# Columns added after a table first shipped: (table, column, type). CREATE TABLE IF NOT EXISTS
# doesn't touch an existing table, so these are added to older databases on start-up.
_ADDED_COLUMNS = [
    ("person_counts", "in_count", "INTEGER NOT NULL DEFAULT 0"),
    ("person_counts", "out_count", "INTEGER NOT NULL DEFAULT 0"),
    ("person_counts", "zone_count", "INTEGER NOT NULL DEFAULT 0"),
]


def _add_missing_columns(db: Database) -> None:
    pg = getattr(db, "backend", "sqlite") == "postgres"
    for table, col, typ in _ADDED_COLUMNS:
        if pg:
            db.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {typ}")
        elif col not in {r["name"] for r in db.fetchall(f"PRAGMA table_info({table})")}:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
            log.info("Added column %s.%s", table, col)


def init_schema(db: Database) -> None:
    db.executescript(PG_SCHEMA_SQL if getattr(db, "backend", "sqlite") == "postgres" else SCHEMA_SQL)
    _add_missing_columns(db)
    row = db.fetchone("SELECT MAX(version) AS v FROM schema_version")
    current = row["v"] if row and row["v"] is not None else 0
    if current < SCHEMA_VERSION:
        db.execute("INSERT INTO schema_version(version, applied_at) VALUES (?, ?) "
                   "ON CONFLICT(version) DO NOTHING", (SCHEMA_VERSION, time.time()))
        log.info("Database schema at version %d (%s)", SCHEMA_VERSION, db.path)
