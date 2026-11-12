import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import CameraConfig, DetectionSettings, RecognitionSettings, StorageSettings  # noqa: E402
from constants import UNKNOWN_PERSON_ID  # noqa: E402
from database.connection import Database  # noqa: E402
from database.embedding_repository import EmbeddingRepository  # noqa: E402
from database.event_repository import EventRepository  # noqa: E402
from database.person_repository import PersonRepository  # noqa: E402
from database.schema import init_schema  # noqa: E402
from fakes import FakeArcFaceModel, FakeSCRFDModel  # noqa: E402
from inference.arcface_engine import ArcFaceEngine  # noqa: E402
from inference.scrfd_engine import SCRFDEngine  # noqa: E402
from ingestion.frame_receiver import FrameReceiver  # noqa: E402
from monitoring.performance import PerformanceTracker  # noqa: E402
from recognition.duplicate_filter import DuplicateFilter, TrackCache  # noqa: E402
from recognition.embedding_manager import EmbeddingGallery  # noqa: E402
from recognition.face_matcher import FaceMatcher  # noqa: E402
from recognition.identity_manager import IdentityManager  # noqa: E402
from scheduler.frame_scheduler import FrameScheduler  # noqa: E402
from services.event_service import EventService  # noqa: E402
from services.recognition_service import RecognitionService  # noqa: E402


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_matcher_picks_best_person_and_rejects_unknown():
    g = EmbeddingGallery(dim=4)
    g.replace([2, 1, 2], np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0.9, 0.1, 0]], np.float32))
    m = FaceMatcher(g, threshold=0.5, margin=0.05)
    res = m.match_batch(np.stack([_unit([1, 0.1, 0, 0]), _unit([0, 1, 0, 0]), _unit([0, 0, 0, 1])]))
    assert [r.person_id for r in res] == [1, 2, UNKNOWN_PERSON_ID]
    assert res[1].similarity > 0.99 and not res[2].is_known


def test_matcher_margin_rejects_ambiguous():
    g = EmbeddingGallery(dim=2)
    g.replace([1, 2], np.array([[1, 0.05], [1, -0.05]], np.float32))
    assert not FaceMatcher(g, threshold=0.5, margin=0.05).match(_unit([1, 0])).is_known


def test_empty_gallery_is_unknown():
    assert FaceMatcher(EmbeddingGallery()).match(_unit(np.ones(512))).person_id == UNKNOWN_PERSON_ID


def test_gallery_add_remove():
    g = EmbeddingGallery(dim=2)
    g.add(5, [[1, 0]])
    g.add(3, [[0, 1], [0, 2]])
    assert g.num_persons == 2 and g.size == 3
    g.remove_person(3)
    assert g.num_persons == 1 and g.snapshot().unique_ids.tolist() == [5]


def test_duplicate_filter_cooldown():
    f = DuplicateFilter(cooldown_s=10, unknown_cooldown_s=10, unknown_similarity=0.8)
    assert f.should_emit("c1", 1, now=0)
    assert not f.should_emit("c1", 1, now=5)
    assert f.should_emit("c2", 1, now=5)  # other camera
    assert not f.should_emit("c1", 1, now=14)  # still present (last seen 5)
    assert f.should_emit("c1", 1, now=30)
    e = _unit([1, 0, 0])
    assert f.should_emit("c1", UNKNOWN_PERSON_ID, e, now=0)
    assert not f.should_emit("c1", UNKNOWN_PERSON_ID, _unit([1, 0.1, 0]), now=1)
    assert f.should_emit("c1", UNKNOWN_PERSON_ID, _unit([0, 1, 0]), now=2)


def test_track_cache_reuse_and_reverify():
    t = TrackCache(iou_threshold=0.5, reverify_s=2.0, ttl_s=1.5)
    assert t.lookup("c1", [0, 0, 100, 100], now=0) is None
    t.update("c1", [0, 0, 100, 100], 7, 0.8, now=0)
    assert t.lookup("c1", [5, 5, 105, 105], now=0.5) == (7, 0.8)
    assert t.lookup("c2", [5, 5, 105, 105], now=0.5) is None
    assert t.lookup("c1", [500, 500, 600, 600], now=0.5) is None
    assert t.lookup("c1", [5, 5, 105, 105], now=2.5) is None  # needs re-verification


def test_identity_manager_hot_reload():
    db = Database(":memory:")
    init_schema(db)
    persons, emb = PersonRepository(db), EmbeddingRepository(db)
    g = EmbeddingGallery()
    idm = IdentityManager(persons, emb, g, reload_interval_s=0)
    idm.reload()
    assert g.size == 0
    pid = persons.create("Carol")
    emb.add(pid, np.ones(512, np.float32))
    assert idm.maybe_reload() and g.size == 1 and idm.name_of(pid) == "Carol"
    assert not idm.maybe_reload()
    assert idm.name_of(UNKNOWN_PERSON_ID) == "unknown"


class _FakeInference:
    def __init__(self):
        self.scrfd = SCRFDEngine(FakeSCRFDModel())
        self.arcface = ArcFaceEngine(FakeArcFaceModel())

    def detect(self, images):
        return self.scrfd.detect_batch(images)

    def embed(self, crops):
        return self.arcface.embed_batch(crops)


def test_end_to_end_pipeline_with_fake_models():
    db = Database(":memory:")
    init_schema(db)
    persons, emb_repo, ev_repo = PersonRepository(db), EmbeddingRepository(db), EventRepository(db)
    inf = _FakeInference()

    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (720, 1280, 3)).astype(np.uint8)
    dets, kps = inf.detect([frame])[0]
    from processing.face_alignment import norm_crop

    pid = persons.create("Dave")
    emb_repo.add(pid, inf.embed([norm_crop(frame, kps[0])])[0])

    gallery = EmbeddingGallery()
    identity = IdentityManager(persons, emb_repo, gallery, reload_interval_s=1e9)
    identity.reload()
    perf = PerformanceTracker()
    events = EventService(ev_repo, StorageSettings(event_flush_interval_s=0.05), perf)
    sched = FrameScheduler(max_frame_age_s=5)
    cams = [CameraConfig(camera_id="c1", name="c1", url="test://")]
    recv = FrameReceiver("c1", sched.register("c1"))
    det_cfg = DetectionSettings(min_face_size=20)
    rec_cfg = RecognitionSettings(blur_threshold=1.0, min_quality=0.0, max_yaw_ratio=1.0)
    svc = RecognitionService(inf, sched, identity, FaceMatcher(gallery, 0.45, 0.05), DuplicateFilter(60),
                             TrackCache(0.45, 2.0), events, cams, det_cfg, rec_cfg, perf)
    events.start()
    svc.start()
    for _ in range(4):
        recv(frame)
        time.sleep(0.1)
    time.sleep(0.3)
    svc.stop()
    events.stop()

    rows = ev_repo.query()
    assert len(rows) == 1, rows  # de-duplicated
    assert rows[0]["person_id"] == pid and rows[0]["label"] == "Dave"
    assert perf.count("faces_track_reused") >= 1  # later frames skipped ArcFace
