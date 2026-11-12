import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import CameraConfig, StorageSettings  # noqa: E402
from database.camera_repository import CameraRepository  # noqa: E402
from database.connection import Database  # noqa: E402
from database.embedding_repository import EmbeddingRepository  # noqa: E402
from database.event_repository import EventRepository  # noqa: E402
from database.person_repository import PersonRepository  # noqa: E402
from database.schema import init_schema  # noqa: E402
from services.event_service import EventService, RecognitionEvent  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "t.db")
    init_schema(d)
    yield d
    d.close()


def test_schema_idempotent(db):
    init_schema(db)
    assert db.fetchone("SELECT MAX(version) v FROM schema_version")["v"] >= 1


def test_camera_url_is_masked(db):
    repo = CameraRepository(db)
    repo.sync([CameraConfig(camera_id="c1", name="Door", url="rtsp://admin:secret@10.0.0.1/stream")])
    row = repo.get("c1")
    assert "secret" not in row["url_masked"] and "10.0.0.1" in row["url_masked"]


def test_person_and_embeddings_roundtrip(db):
    persons, emb = PersonRepository(db), EmbeddingRepository(db)
    pid = persons.get_or_create("Alice")
    assert persons.get_or_create("Alice") == pid
    v = np.random.rand(512).astype(np.float32)
    emb.add(pid, v, source="a.jpg")
    emb.add_many([(pid, v * 2, 0.9, "b.jpg")])
    ids, vecs = emb.load_all()
    assert ids.tolist() == [pid, pid]
    np.testing.assert_allclose(vecs[0], v)
    assert emb.has_source(pid, "a.jpg") and not emb.has_source(pid, "c.jpg")

    fp = emb.fingerprint()
    persons.set_active(pid, False)
    assert emb.fingerprint() != fp
    assert emb.load_all()[0].size == 0

    assert persons.delete(pid)
    assert emb.count_for(pid) == 0  # cascade


def test_event_service_writes_and_exports(db, tmp_path):
    repo = EventRepository(db)
    svc = EventService(repo, StorageSettings(db_path=tmp_path / "t.db", event_flush_interval_s=0.05))
    svc.start()
    now = time.time()
    for i in range(5):
        svc.submit(RecognitionEvent("c1", None, "unknown", 0.1, 0.9, 0.5, [1, 2, 3, 4], now + i))
    svc.stop()
    assert repo.count() == 5
    rows = repo.query(camera_id="c1")
    assert rows[0]["ts"] >= rows[-1]["ts"]
    out = svc.export_csv(tmp_path / "e.csv")
    assert len(out.read_text().strip().splitlines()) == 6


def test_memory_db_shared_across_threads():
    import threading

    d = Database(":memory:")
    init_schema(d)
    repo = PersonRepository(d)
    t = threading.Thread(target=lambda: repo.create("Bob"))
    t.start()
    t.join()
    assert repo.get_by_name("Bob") is not None
