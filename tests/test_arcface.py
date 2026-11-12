import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import ARCFACE_DST  # noqa: E402
from fakes import FakeArcFaceModel  # noqa: E402
from inference.arcface_engine import ArcFaceEngine  # noqa: E402
from processing.face_alignment import estimate_norm, norm_crop  # noqa: E402
from processing.face_detection import DetectedFace  # noqa: E402
from processing.face_quality import QualityConfig, assess_quality, yaw_ratio  # noqa: E402


def _crop(color):
    return np.full((112, 112, 3), color, np.uint8)


def test_embeddings_are_normalised_and_batched():
    model = FakeArcFaceModel()
    eng = ArcFaceEngine(model)
    emb = eng.embed_batch([_crop((10, 20, 30)), _crop((200, 100, 50)), _crop((10, 20, 31))])
    assert emb.shape == (3, 512) and emb.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    assert emb[0] @ emb[2] > emb[0] @ emb[1]


def test_native_batch_pads_and_trims():
    model = FakeArcFaceModel(native_batch=4)
    eng = ArcFaceEngine(model)
    assert eng.native_batch == 4
    crops = [_crop((i * 20, 50, 100)) for i in range(6)]
    emb = eng.embed_batch(crops)
    assert emb.shape == (6, 512) and model.calls == 2
    ref = ArcFaceEngine(FakeArcFaceModel()).embed_batch(crops)
    np.testing.assert_allclose(emb, ref, atol=1e-5)


def test_empty_batch():
    assert ArcFaceEngine(FakeArcFaceModel()).embed_batch([]).shape[0] == 0


def test_alignment_maps_landmarks_to_template():
    # landmarks = template scaled x2 and shifted -> alignment must undo it exactly
    kps = ARCFACE_DST * 2.0 + np.array([100, 50], np.float32)
    M = estimate_norm(kps)
    mapped = (M[:, :2] @ kps.T).T + M[:, 2]
    np.testing.assert_allclose(mapped, ARCFACE_DST, atol=1e-3)
    img = np.random.randint(0, 255, (480, 640, 3), np.uint8)
    assert norm_crop(img, kps).shape == (112, 112, 3)


def test_quality_gate():
    kps = ARCFACE_DST.copy()
    assert yaw_ratio(kps) < 0.1
    face = DetectedFace(np.array([0, 0, 120, 120], np.float32), 0.9, kps)
    sharp = np.random.default_rng(0).integers(0, 255, (112, 112, 3)).astype(np.uint8)
    cfg = QualityConfig(min_face_size=40, blur_threshold=40)
    assert assess_quality(face, sharp, cfg).passed
    assert assess_quality(face, _crop((128, 128, 128)), cfg).reason == "blurry"
    small = DetectedFace(np.array([0, 0, 20, 20], np.float32), 0.9, kps)
    assert assess_quality(small, sharp, cfg).reason == "small"
    profile = kps.copy()
    profile[2, 0] += 30
    assert assess_quality(DetectedFace(face.bbox, 0.9, profile), sharp, cfg).reason == "pose"
