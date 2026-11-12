"""Fake Core ML models for tests (same interface as inference.coreml_loader.CoreMLModel)."""
import numpy as np

from inference.coreml_loader import FeatureSpec


class FakeSCRFDModel:
    """Emits SCRFD-style outputs (strides 8/16/32, 2 anchors) with one face at a known anchor."""

    def __init__(self, size=640, kind="multiarray", face_cx=320, face_cy=320, stride=16, half=40, as_nchw=False):
        self.size = size
        if kind == "image":
            self.inputs = [FeatureSpec("input.1", "image", width=size, height=size)]
        else:
            self.inputs = [FeatureSpec("input.1", "multiarray", shape=(1, 3, size, size), dtype=np.float32)]
        self.outputs = []
        self.face = (face_cx, face_cy, stride, half)
        self.as_nchw = as_nchw
        self.calls = 0

    def _one(self):
        outs = {}
        cx, cy, st, half = self.face
        for i, s in enumerate((8, 16, 32)):
            fh = fw = self.size // s
            n = fh * fw * 2
            score = np.zeros((n, 1), np.float32)
            bbox = np.zeros((n, 4), np.float32)
            kps = np.zeros((n, 10), np.float32)
            if s == st:
                idx = ((cy // s) * fw + (cx // s)) * 2
                score[idx] = 0.9
                bbox[idx] = half / s
                pts = np.array([[-15, -10], [15, -10], [0, 5], [-12, 20], [12, 20]], np.float32) / s
                kps[idx] = pts.reshape(-1)
            if self.as_nchw:
                def to4d(a, k):
                    return a.reshape(fh, fw, 2 * k).transpose(2, 0, 1)[None]
                outs[f"score_{s}"], outs[f"bbox_{s}"], outs[f"kps_{s}"] = to4d(score, 1), to4d(bbox, 4), to4d(kps, 10)
            else:
                outs[f"o{i}"], outs[f"o{i + 3}"], outs[f"o{i + 6}"] = score[None], bbox[None], kps[None]
        return outs

    def predict(self, feeds):
        self.calls += 1
        return self._one()

    def predict_batch(self, feeds_list):
        return [self.predict(f) for f in feeds_list]


class FakeArcFaceModel:
    """Embedding = deterministic projection of the mean colour -> similar crops give similar vectors."""

    def __init__(self, dim=512, native_batch=1):
        shape = (native_batch, 3, 112, 112)
        self.inputs = [FeatureSpec("data", "multiarray", shape=shape, dtype=np.float32)]
        self.outputs = [FeatureSpec("fc1", "multiarray", shape=(native_batch, dim))]
        rng = np.random.default_rng(42)
        self.proj = rng.standard_normal((3, dim)).astype(np.float32)
        self.native_batch = native_batch
        self.calls = 0

    def predict(self, feeds):
        self.calls += 1
        x = feeds["data"]  # (B, 3, 112, 112)
        means = x.mean(axis=(2, 3))  # (B, 3)
        return {"fc1": (means + 0.01) @ self.proj}

    def predict_batch(self, feeds_list):
        return [self.predict(f) for f in feeds_list]
