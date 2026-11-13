"""Database connections: SQLite (default, WAL, one connection per thread) or PostgreSQL (pooled).

Both backends expose the same small API, and repositories write SQL with `?` placeholders,
which the PostgreSQL backend translates to `%s`:

    execute(sql, params) -> result with .rowcount
    insert(sql, params)  -> new row id
    executemany(sql, rows)   (one transaction)
    fetchall / fetchone      -> rows usable as dicts (row["col"], dict(row))
    executescript(sql), bucket_expr(col, seconds), close()

Use `open_database(storage_settings)` to get the one configured in .env.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Sequence, Union
from urllib.parse import urlsplit, urlunsplit

log = logging.getLogger(__name__)


class Database:
    """SQLite backend."""

    backend = "sqlite"

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._all: List[sqlite3.Connection] = []
        self._lock = threading.Lock()
        # ':memory:' databases are per-connection, so share a single one across threads.
        self._shared: Optional[sqlite3.Connection] = None
        self._shared_lock = threading.RLock()
        if self.path == ":memory:":
            self._shared = self._connect()

    @property
    def description(self) -> str:
        return f"sqlite:{self.path}"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=self.path != ":memory:",
                               isolation_level=None)  # autocommit; explicit BEGIN in transaction()
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        if self.path != ":memory:":
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        with self._lock:
            self._all.append(conn)
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        if self._shared is not None:
            return self._shared
        c = getattr(self._local, "conn", None)
        if c is None:
            c = self._connect()
            self._local.conn = c
        return c

    @contextmanager
    def _guard(self) -> Iterator[None]:
        if self._shared is not None:
            with self._shared_lock:
                yield
        else:
            yield

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._guard():
            c = self.conn
            c.execute("BEGIN IMMEDIATE")
            try:
                yield c
                c.execute("COMMIT")
            except BaseException:
                c.execute("ROLLBACK")
                raise

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._guard():
            return self.conn.execute(sql, params)

    def insert(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Run an INSERT into a table with an `id` primary key and return the new id."""
        with self._guard():
            return int(self.conn.execute(sql, params).lastrowid)

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        with self.transaction() as c:
            c.executemany(sql, rows)

    def executescript(self, sql: str) -> None:
        with self._guard():
            self.conn.executescript(sql)

    def fetchall(self, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
        with self._guard():
            return self.conn.execute(sql, params).fetchall()

    def fetchone(self, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        with self._guard():
            return self.conn.execute(sql, params).fetchone()

    @staticmethod
    def bucket_expr(col: str, seconds: int) -> str:
        """SQL for the start (epoch seconds) of the `seconds`-wide bucket containing `col`."""
        s = int(seconds)
        return f"(CAST({col} / {s} AS INTEGER) * {s})"

    def close(self) -> None:
        with self._lock:
            for c in self._all:
                try:
                    c.close()
                except Exception:
                    pass
            self._all.clear()
        self._local = threading.local()


# --------------------------------------------------------------------------- PostgreSQL
@dataclass
class _Result:
    rowcount: int


def _to_pg(sql: str) -> str:
    """`?` placeholders -> `%s` (and escape literal `%`). Queries here never contain a literal `?`."""
    return sql.replace("%", "%%").replace("?", "%s")


def mask_dsn(dsn: str) -> str:
    """Hide the password in a postgres URL or key=value conninfo string."""
    if "://" in dsn:
        parts = urlsplit(dsn)
        if parts.password is not None:
            host = parts.hostname or ""
            if parts.port:
                host = f"{host}:{parts.port}"
            netloc = f"{parts.username or ''}:***@{host}"
            return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
        return dsn
    try:  # key=value conninfo -> user@host:port/dbname
        from psycopg.conninfo import conninfo_to_dict

        d = conninfo_to_dict(dsn)
        user = f"{d['user']}@" if d.get("user") else ""
        port = f":{d['port']}" if d.get("port") else ""
        return f"{user}{d.get('host', 'localhost')}{port}/{d.get('dbname', '')}"
    except Exception:
        return " ".join("password=***" if t.lower().startswith("password=") else t for t in dsn.split())


class PostgresDatabase:
    """PostgreSQL backend: a thread-safe connection pool, autocommit, rows as dicts."""

    backend = "postgres"

    def __init__(self, conninfo: str, min_size: int = 1, max_size: int = 10, connect_timeout_s: float = 15.0,
                 application_name: str = "facereco") -> None:
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError("PostgreSQL needs psycopg: pip install 'psycopg[binary]' psycopg-pool") from exc

        self.path = mask_dsn(conninfo)
        self._pool = ConnectionPool(
            conninfo,
            min_size=max(1, min_size),
            max_size=max(min_size, max_size),
            kwargs={"autocommit": True, "row_factory": dict_row, "application_name": application_name},
            timeout=30.0,
            name="facereco-db",
            open=False,
        )
        self._pool.open(wait=False)
        try:
            self._pool.wait(timeout=connect_timeout_s)
        except Exception as exc:
            self._pool.close()
            raise RuntimeError(f"Could not connect to PostgreSQL at {self.path}: {exc}. "
                               "Check FR_DATABASE_URL / FR_PG_* in .env.") from exc
        log.info("Connected to PostgreSQL %s (pool %d-%d)", self.path, min_size, max_size)

    @property
    def description(self) -> str:
        return f"postgres:{self.path}"

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        with self._pool.connection() as conn:
            with conn.transaction():
                yield conn

    def execute(self, sql: str, params: Sequence[Any] = ()) -> _Result:
        with self._pool.connection() as conn:
            cur = conn.execute(_to_pg(sql), tuple(params)) if params else conn.execute(sql)
            return _Result(cur.rowcount)

    def insert(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self._pool.connection() as conn:
            row = conn.execute(_to_pg(sql.rstrip().rstrip(";") + " RETURNING id"), tuple(params)).fetchone()
            return int(row["id"])

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        payload = [tuple(r) for r in rows]
        if not payload:
            return
        with self.transaction() as conn:
            with conn.cursor() as cur:
                cur.executemany(_to_pg(sql), payload)

    def executescript(self, sql: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(sql)

    def fetchall(self, sql: str, params: Sequence[Any] = ()) -> List[dict]:
        with self._pool.connection() as conn:
            cur = conn.execute(_to_pg(sql), tuple(params)) if params else conn.execute(sql)
            return cur.fetchall()

    def fetchone(self, sql: str, params: Sequence[Any] = ()) -> Optional[dict]:
        with self._pool.connection() as conn:
            cur = conn.execute(_to_pg(sql), tuple(params)) if params else conn.execute(sql)
            return cur.fetchone()

    @staticmethod
    def bucket_expr(col: str, seconds: int) -> str:
        s = int(seconds)
        return f"(FLOOR({col} / {s})::bigint * {s})"

    def close(self) -> None:
        self._pool.close()


AnyDatabase = Union[Database, PostgresDatabase]


def open_database(storage) -> AnyDatabase:
    """Open the database configured in StorageSettings (PostgreSQL when a URL/host is set, else SQLite)."""
    conninfo = storage.postgres_conninfo()
    if conninfo:
        return PostgresDatabase(conninfo, min_size=storage.pg_pool_min, max_size=storage.pg_pool_max,
                                connect_timeout_s=storage.pg_connect_timeout_s)
    return Database(storage.db_path)
