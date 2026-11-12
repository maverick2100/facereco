"""SQLite connection management (WAL mode, one connection per thread)."""
from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Sequence, Union

log = logging.getLogger(__name__)


class Database:
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

    def close(self) -> None:
        with self._lock:
            for c in self._all:
                try:
                    c.close()
                except Exception:
                    pass
            self._all.clear()
        self._local = threading.local()
