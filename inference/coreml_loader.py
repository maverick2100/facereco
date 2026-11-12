"""Core ML model loading with compiled-model caching and spec introspection."""
from __future__ import annotations

import hashlib
import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from inference.ane_config import resolve_compute_units

log = logging.getLogger(__name__)

# Core ML protobuf enum values (avoid importing the proto modules directly).
_COLOR_SPACES = {10: "GRAYSCALE", 20: "RGB", 30: "BGR", 40: "GRAYSCALE_FLOAT16"}
_ARRAY_DTYPES = {65568: np.float32, 65552: np.float16, 65600: np.float64, 131104: np.int32}


@dataclass
class FeatureSpec:
    name: str
    kind: str                      # "image" | "multiarray" | "other"
    shape: Tuple[int, ...] = ()
    dtype: Any = np.float32
    width: int = 0
    height: int = 0
    color_space: str = "RGB"

    def describe(self) -> str:
        if self.kind == "image":
            return f"{self.name}: Image {self.width}x{self.height} {self.color_space}"
        if self.kind == "multiarray":
            return f"{self.name}: MultiArray {list(self.shape)} {np.dtype(self.dtype).name}"
        return f"{self.name}: {self.kind}"


def parse_feature(feature: Any) -> FeatureSpec:
    t = feature.type
    which = t.WhichOneof("Type")
    if which == "imageType":
        it = t.imageType
        w, h = int(it.width), int(it.height)
        if (not w or not h) and len(it.enumeratedSizes.sizes):
            s = it.enumeratedSizes.sizes[0]
            w, h = int(s.width), int(s.height)
        if (not w or not h) and it.HasField("imageSizeRange"):
            w = int(it.imageSizeRange.widthRange.lowerBound)
            h = int(it.imageSizeRange.heightRange.lowerBound)
        return FeatureSpec(feature.name, "image", width=w, height=h,
                           color_space=_COLOR_SPACES.get(int(it.colorSpace), "RGB"))
    if which == "multiArrayType":
        ma = t.multiArrayType
        shape = tuple(int(x) for x in ma.shape)
        if not shape and len(ma.enumeratedShapes.shapes):
            shape = tuple(int(x) for x in ma.enumeratedShapes.shapes[0].shape)
        if not shape and len(ma.shapeRange.sizeRanges):
            shape = tuple(int(r.lowerBound) for r in ma.shapeRange.sizeRanges)
        return FeatureSpec(feature.name, "multiarray", shape=shape,
                           dtype=_ARRAY_DTYPES.get(int(ma.dataType), np.float32))
    return FeatureSpec(feature.name, which or "other")


def _fingerprint(path: Path, extra: str) -> str:
    """Cache key for a model package.

    coremltools rewrites Manifest.json (new mtime) every time it opens an .mlpackage, so
    small files are hashed by content; big weight files by size + mtime (never rewritten).
    """
    h = hashlib.sha1(extra.encode())
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for f in files:
        st = f.stat()
        h.update(str(f.relative_to(path.parent)).encode())
        if st.st_size <= 4 * 1024 * 1024:
            h.update(f.read_bytes())
        else:
            h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


class CoreMLModel:
    """Thin wrapper around coremltools' MLModel / CompiledMLModel."""

    def __init__(self, path: Path, compute_units: str = "CPU_AND_NE", cache_dir: Optional[Path] = None,
                 name: Optional[str] = None) -> None:
        import coremltools as ct

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Core ML model not found: {self.path}")
        self.name = name or self.path.stem
        self.compute_units = compute_units
        self._lock = threading.Lock()
        self._batch_supported = True

        t0 = time.perf_counter()
        self._model, spec = self._load(ct, resolve_compute_units(compute_units), cache_dir)
        self.load_time_s = time.perf_counter() - t0

        self.model_type = spec.WhichOneof("Type")
        self.spec_version = int(spec.specificationVersion)
        self.inputs: List[FeatureSpec] = [parse_feature(f) for f in spec.description.input]
        self.outputs: List[FeatureSpec] = [parse_feature(f) for f in spec.description.output]
        log.info("Loaded %s (%s, spec v%d) on %s in %.2fs | in: %s | out: %s",
                 self.name, self.model_type, self.spec_version, compute_units, self.load_time_s,
                 "; ".join(i.describe() for i in self.inputs),
                 "; ".join(o.describe() for o in self.outputs))

    # ------------------------------------------------------------------ loading
    def _load(self, ct: Any, cu: Any, cache_dir: Optional[Path]):
        compiled_cls = getattr(ct.models, "CompiledMLModel", None)
        cache_path = None
        if cache_dir is not None and compiled_cls is not None:
            cache_path = Path(cache_dir) / f"{self.path.stem}-{_fingerprint(self.path, ct.__version__)}.mlmodelc"
            if cache_path.exists():
                try:
                    spec = ct.utils.load_spec(str(self.path))
                    model = compiled_cls(str(cache_path), compute_units=cu)
                    log.debug("Using cached compiled model %s", cache_path)
                    return model, spec
                except Exception as exc:
                    log.warning("Cached compiled model unusable (%s); recompiling", exc)
                    shutil.rmtree(cache_path, ignore_errors=True)

        model = ct.models.MLModel(str(self.path), compute_units=cu)
        spec = model.get_spec()
        if cache_path is not None:
            try:
                src = Path(model.get_compiled_model_path())
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                for stale in cache_path.parent.glob(f"{self.path.stem}-*.mlmodelc"):
                    shutil.rmtree(stale, ignore_errors=True)
                shutil.copytree(src, cache_path)
                log.info("Cached compiled model at %s", cache_path)
            except Exception as exc:
                log.debug("Could not cache compiled model: %s", exc)
        return model, spec

    # ------------------------------------------------------------------ inference
    def predict(self, feeds: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            return self._model.predict(feeds)

    def predict_batch(self, feeds_list: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run several independent predictions in one Core ML batch call when supported."""
        n = len(feeds_list)
        if n == 0:
            return []
        if n == 1:
            return [self.predict(feeds_list[0])]
        if self._batch_supported:
            try:
                with self._lock:
                    out = self._model.predict(list(feeds_list))
                if isinstance(out, list) and len(out) == n:
                    return out
                self._batch_supported = False
            except Exception as exc:
                log.info("%s: batch predict unsupported (%s) - using sequential calls", self.name, exc)
                self._batch_supported = False
        with self._lock:
            return [self._model.predict(f) for f in feeds_list]

    def info(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "type": self.model_type,
            "compute_units": self.compute_units,
            "load_time_s": round(self.load_time_s, 2),
            "inputs": [i.describe() for i in self.inputs],
            "outputs": [o.describe() for o in self.outputs],
            "batch_api": self._batch_supported,
        }
