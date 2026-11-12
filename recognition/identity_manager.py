"""Keeps the in-memory gallery and person names in sync with the database (hot reload)."""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

from constants import UNKNOWN_LABEL, UNKNOWN_PERSON_ID
from database.embedding_repository import EmbeddingRepository
from database.person_repository import PersonRepository
from recognition.embedding_manager import EmbeddingGallery

log = logging.getLogger(__name__)


class IdentityManager:
    def __init__(self, persons: PersonRepository, embeddings: EmbeddingRepository, gallery: EmbeddingGallery,
                 reload_interval_s: float = 30.0) -> None:
        self.persons = persons
        self.embeddings = embeddings
        self.gallery = gallery
        self.reload_interval_s = reload_interval_s
        self._names: Dict[int, str] = {}
        self._fingerprint: Optional[tuple] = None
        self._last_check = 0.0
        self._lock = threading.Lock()

    def reload(self) -> None:
        with self._lock:
            people = self.persons.list(active_only=True)
            ids, vecs = self.embeddings.load_all(active_only=True)
            self._names = {int(p["id"]): p["name"] for p in people}
            self.gallery.replace(ids, vecs)
            self._fingerprint = self.embeddings.fingerprint()
            self._last_check = time.monotonic()
        log.info("Gallery loaded: %d persons, %d embeddings", self.gallery.num_persons, self.gallery.size)

    def maybe_reload(self) -> bool:
        """Cheap periodic check; reloads only if the DB changed (e.g. enrolment from another process)."""
        now = time.monotonic()
        if now - self._last_check < self.reload_interval_s:
            return False
        self._last_check = now
        try:
            fp = self.embeddings.fingerprint()
        except Exception as exc:
            log.warning("Gallery fingerprint check failed: %s", exc)
            return False
        if fp != self._fingerprint:
            self.reload()
            return True
        return False

    def name_of(self, person_id: Optional[int]) -> str:
        if person_id is None or person_id == UNKNOWN_PERSON_ID:
            return UNKNOWN_LABEL
        return self._names.get(int(person_id), f"person-{person_id}")
