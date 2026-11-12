"""Application settings: environment (.env) + config/cameras.yaml."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from constants import (
    ARCFACE_INPUT_MEAN,
    ARCFACE_INPUT_STD,
    CONFIG_DIR,
    DEFAULT_ARCFACE_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_SCRFD_PATH,
    PROJECT_ROOT,
    SCRFD_INPUT_MEAN,
    SCRFD_INPUT_STD,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- env helpers
def _env(name: str, default: Any) -> Any:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    raw = raw.strip()
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def _env_path(name: str, default: Path) -> Path:
    p = Path(str(_env(name, str(default)))).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# --------------------------------------------------------------------------- dataclasses
@dataclass
class CameraConfig:
    camera_id: str
    name: str
    url: str
    enabled: bool = True
    location: str = ""
    codec: str = "h264"          # h264 | h265 | auto
    decoder: str = "vtdec"       # VideoToolbox hardware decoder (vtdec / vtdec_hw)
    width: int = 1280            # resolution delivered to Python
    height: int = 720
    fps: float = 4.0             # frames/s delivered to the pipeline (GStreamer drops the rest)
    latency_ms: int = 200
    transport: str = "tcp"       # tcp | udp
    priority: float = 1.0        # >1 = scheduled more aggressively
    roi: Optional[List[int]] = None  # [x1, y1, x2, y2] in delivered resolution

    @classmethod
    def from_dict(cls, data: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None) -> "CameraConfig":
        merged = dict(defaults or {})
        merged.update(data or {})
        known = {f.name for f in fields(cls)}
        unknown = set(merged) - known
        if unknown:
            log.warning("Camera %s: ignoring unknown keys %s", merged.get("camera_id"), sorted(unknown))
        kwargs = {k: v for k, v in merged.items() if k in known}
        if "camera_id" not in kwargs or "url" not in kwargs:
            raise ValueError(f"camera entry needs camera_id and url: {data}")
        kwargs["camera_id"] = str(kwargs["camera_id"])
        kwargs.setdefault("name", kwargs["camera_id"])
        cam = cls(**kwargs)
        if cam.roi is not None and len(cam.roi) != 4:
            raise ValueError(f"camera {cam.camera_id}: roi must be [x1, y1, x2, y2]")
        return cam


@dataclass
class ModelSettings:
    scrfd_path: Path = DEFAULT_SCRFD_PATH
    arcface_path: Path = DEFAULT_ARCFACE_PATH
    compute_units: str = "CPU_AND_NE"   # CPU_AND_NE keeps work on the Neural Engine (no GPU fallback)
    cache_compiled: bool = True
    scrfd_input_mean: float = SCRFD_INPUT_MEAN   # only used when model input is a MultiArray
    scrfd_input_std: float = SCRFD_INPUT_STD
    scrfd_apply_sigmoid: bool = False            # set true if your SCRFD outputs logits
    arcface_input_mean: float = ARCFACE_INPUT_MEAN
    arcface_input_std: float = ARCFACE_INPUT_STD
    warmup_iterations: int = 3


@dataclass
class DetectionSettings:
    det_threshold: float = 0.5
    nms_threshold: float = 0.4
    min_face_size: int = 40          # px (shorter bbox side) in delivered resolution
    max_faces_per_frame: int = 20
    batch_size: int = 4              # frames per SCRFD batch
    max_wait_ms: float = 8.0         # max time to wait to fill a batch
    max_frame_age_s: float = 1.0     # frames older than this are dropped (never process backlog)


@dataclass
class RecognitionSettings:
    match_threshold: float = 0.45    # cosine similarity
    match_margin: float = 0.05       # best - second best person must exceed this
    batch_size: int = 16             # crops per ArcFace batch
    max_wait_ms: float = 10.0
    queue_size: int = 256
    blur_threshold: float = 40.0     # Laplacian variance of 112x112 crop
    max_yaw_ratio: float = 0.45      # nose offset / eye distance
    min_quality: float = 0.25
    event_cooldown_s: float = 10.0   # same person, same camera
    record_unknown: bool = True
    unknown_cooldown_s: float = 30.0
    unknown_similarity: float = 0.5
    track_iou: float = 0.45          # skip ArcFace for a face that overlaps a recent identified face
    track_reverify_s: float = 2.0    # ...but re-embed it at least this often
    gallery_reload_s: float = 30.0


@dataclass
class IngestionSettings:
    backend: str = "auto"            # auto | gstreamer | opencv
    reconnect_delay_s: float = 2.0
    reconnect_max_delay_s: float = 60.0
    stale_timeout_s: float = 15.0
    queue_depth: int = 1             # frames kept per camera (latest-frame-wins)


@dataclass
class StorageSettings:
    db_path: Path = DEFAULT_DB_PATH
    save_snapshots: bool = False
    snapshot_quality: int = 85
    event_flush_interval_s: float = 1.0
    event_batch_size: int = 100
    event_queue_size: int = 5000
    retention_days: int = 90


@dataclass
class MonitoringSettings:
    health_host: str = "127.0.0.1"
    health_port: int = 8085          # 0 disables the HTTP endpoint
    health_interval_s: float = 30.0
    enable_powermetrics: bool = True  # only works when running as root (sudo)


@dataclass
class Settings:
    models: ModelSettings = field(default_factory=ModelSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    recognition: RecognitionSettings = field(default_factory=RecognitionSettings)
    ingestion: IngestionSettings = field(default_factory=IngestionSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    monitoring: MonitoringSettings = field(default_factory=MonitoringSettings)
    cameras: List[CameraConfig] = field(default_factory=list)
    log_level: str = "INFO"
    logging_config: Path = CONFIG_DIR / "logging.yaml"

    @property
    def enabled_cameras(self) -> List[CameraConfig]:
        return [c for c in self.cameras if c.enabled]


# --------------------------------------------------------------------------- loaders
def load_cameras(path: Path) -> List[CameraConfig]:
    if not path.exists():
        log.warning("Camera config %s not found", path)
        return []
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    defaults = raw.get("defaults") or {}
    cams = [CameraConfig.from_dict(c, defaults) for c in (raw.get("cameras") or [])]
    ids = [c.camera_id for c in cams]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate camera_id(s) in {path}: {sorted(dupes)}")
    return cams


def load_settings(env_file: Optional[Path] = None, cameras_file: Optional[Path] = None) -> Settings:
    env_file = env_file or (PROJECT_ROOT / ".env")
    if load_dotenv is not None and env_file.exists():
        load_dotenv(env_file, override=False)

    m = ModelSettings(
        scrfd_path=_env_path("FR_SCRFD_MODEL", DEFAULT_SCRFD_PATH),
        arcface_path=_env_path("FR_ARCFACE_MODEL", DEFAULT_ARCFACE_PATH),
        compute_units=_env("FR_COMPUTE_UNITS", "CPU_AND_NE"),
        cache_compiled=_env("FR_CACHE_COMPILED", True),
        scrfd_input_mean=_env("FR_SCRFD_MEAN", SCRFD_INPUT_MEAN),
        scrfd_input_std=_env("FR_SCRFD_STD", SCRFD_INPUT_STD),
        scrfd_apply_sigmoid=_env("FR_SCRFD_SIGMOID", False),
        arcface_input_mean=_env("FR_ARCFACE_MEAN", ARCFACE_INPUT_MEAN),
        arcface_input_std=_env("FR_ARCFACE_STD", ARCFACE_INPUT_STD),
        warmup_iterations=_env("FR_WARMUP_ITERS", 3),
    )
    d = DetectionSettings(
        det_threshold=_env("FR_DET_THRESHOLD", 0.5),
        nms_threshold=_env("FR_NMS_THRESHOLD", 0.4),
        min_face_size=_env("FR_MIN_FACE_SIZE", 40),
        max_faces_per_frame=_env("FR_MAX_FACES", 20),
        batch_size=_env("FR_DET_BATCH", 4),
        max_wait_ms=_env("FR_DET_MAX_WAIT_MS", 8.0),
        max_frame_age_s=_env("FR_MAX_FRAME_AGE_S", 1.0),
    )
    r = RecognitionSettings(
        match_threshold=_env("FR_MATCH_THRESHOLD", 0.45),
        match_margin=_env("FR_MATCH_MARGIN", 0.05),
        batch_size=_env("FR_REC_BATCH", 16),
        max_wait_ms=_env("FR_REC_MAX_WAIT_MS", 10.0),
        queue_size=_env("FR_REC_QUEUE", 256),
        blur_threshold=_env("FR_BLUR_THRESHOLD", 40.0),
        max_yaw_ratio=_env("FR_MAX_YAW", 0.45),
        min_quality=_env("FR_MIN_QUALITY", 0.25),
        event_cooldown_s=_env("FR_EVENT_COOLDOWN_S", 10.0),
        record_unknown=_env("FR_RECORD_UNKNOWN", True),
        unknown_cooldown_s=_env("FR_UNKNOWN_COOLDOWN_S", 30.0),
        unknown_similarity=_env("FR_UNKNOWN_SIMILARITY", 0.5),
        track_iou=_env("FR_TRACK_IOU", 0.45),
        track_reverify_s=_env("FR_TRACK_REVERIFY_S", 2.0),
        gallery_reload_s=_env("FR_GALLERY_RELOAD_S", 30.0),
    )
    i = IngestionSettings(
        backend=_env("FR_INGESTION_BACKEND", "auto"),
        reconnect_delay_s=_env("FR_RECONNECT_DELAY_S", 2.0),
        reconnect_max_delay_s=_env("FR_RECONNECT_MAX_DELAY_S", 60.0),
        stale_timeout_s=_env("FR_STALE_TIMEOUT_S", 15.0),
        queue_depth=_env("FR_QUEUE_DEPTH", 1),
    )
    s = StorageSettings(
        db_path=_env_path("FR_DB_PATH", DEFAULT_DB_PATH),
        save_snapshots=_env("FR_SAVE_SNAPSHOTS", False),
        snapshot_quality=_env("FR_SNAPSHOT_QUALITY", 85),
        event_flush_interval_s=_env("FR_EVENT_FLUSH_S", 1.0),
        event_batch_size=_env("FR_EVENT_BATCH", 100),
        event_queue_size=_env("FR_EVENT_QUEUE", 5000),
        retention_days=_env("FR_RETENTION_DAYS", 90),
    )
    mon = MonitoringSettings(
        health_host=_env("FR_HEALTH_HOST", "127.0.0.1"),
        health_port=_env("FR_HEALTH_PORT", 8085),
        health_interval_s=_env("FR_HEALTH_INTERVAL_S", 30.0),
        enable_powermetrics=_env("FR_POWERMETRICS", True),
    )
    cams_path = cameras_file or _env_path("FR_CAMERAS_FILE", CONFIG_DIR / "cameras.yaml")
    settings = Settings(
        models=m,
        detection=d,
        recognition=r,
        ingestion=i,
        storage=s,
        monitoring=mon,
        cameras=load_cameras(cams_path),
        log_level=str(_env("FR_LOG_LEVEL", "INFO")).upper(),
        logging_config=_env_path("FR_LOGGING_CONFIG", CONFIG_DIR / "logging.yaml"),
    )
    _validate(settings)
    return settings


def _validate(s: Settings) -> None:
    if s.ingestion.backend not in ("auto", "gstreamer", "opencv"):
        raise ValueError("FR_INGESTION_BACKEND must be auto, gstreamer or opencv")
    if not 0.0 < s.recognition.match_threshold < 1.0:
        raise ValueError("FR_MATCH_THRESHOLD must be in (0, 1)")
    if s.detection.batch_size < 1 or s.recognition.batch_size < 1:
        raise ValueError("batch sizes must be >= 1")
    for c in s.cameras:
        if c.fps <= 0:
            raise ValueError(f"camera {c.camera_id}: fps must be > 0")
