"""GStreamer lifecycle: init, GLib main loop thread, per-camera pipelines and a stale-stream watchdog."""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from config.settings import CameraConfig, IngestionSettings
from ingestion.camera_health import STALE, CameraHealthRegistry
from ingestion.camera_pipeline import GstCameraPipeline
from ingestion.frame_receiver import FrameReceiver

log = logging.getLogger(__name__)

_GST: Optional[Tuple[Any, Any]] = None
REQUIRED_ELEMENTS = ("rtspsrc", "rtph264depay", "h264parse", "vtdec", "videorate", "videoscale",
                     "videoconvert", "appsink")


def load_gstreamer() -> Tuple[Any, Any]:
    """Import and initialise GStreamer once; returns (Gst, GLib)."""
    global _GST
    if _GST is None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst  # type: ignore

        Gst.init(None)
        _GST = (Gst, GLib)
    return _GST


def gstreamer_available() -> bool:
    try:
        load_gstreamer()
        return True
    except Exception as exc:
        log.debug("GStreamer unavailable: %s", exc)
        return False


def missing_elements(names=REQUIRED_ELEMENTS) -> List[str]:
    Gst, _ = load_gstreamer()
    return [n for n in names if Gst.ElementFactory.find(n) is None]


class GstManager:
    WATCHDOG_INTERVAL_S = 5

    def __init__(self, cameras: List[CameraConfig], receivers: Dict[str, FrameReceiver],
                 health: CameraHealthRegistry, settings: IngestionSettings) -> None:
        self.cameras = cameras
        self.receivers = receivers
        self.health = health
        self.settings = settings
        self._pipelines: Dict[str, GstCameraPipeline] = {}
        self._loop: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None
        self._watchdog_id: Optional[int] = None

    def start(self) -> None:
        Gst, GLib = load_gstreamer()
        missing = missing_elements()
        if missing:
            log.warning("Missing GStreamer elements: %s (brew install gstreamer)", ", ".join(missing))
        log.info("GStreamer %s - starting %d camera pipelines", Gst.version_string(), len(self.cameras))

        self._loop = GLib.MainLoop()
        self._thread = threading.Thread(target=self._loop.run, name="glib-mainloop", daemon=True)
        self._thread.start()

        for cam in self.cameras:
            p = GstCameraPipeline(cam, self.receivers[cam.camera_id], self.health.get(cam.camera_id),
                                  self.settings, Gst)
            self._pipelines[cam.camera_id] = p
            p.start()
        self._watchdog_id = GLib.timeout_add_seconds(self.WATCHDOG_INTERVAL_S, self._watchdog)

    def _watchdog(self) -> bool:
        for cid, p in self._pipelines.items():
            h = self.health.get(cid)
            if not p.restarting and h.is_stale(self.settings.stale_timeout_s):
                log.warning("[%s] no frames for %.0fs - restarting", cid, self.settings.stale_timeout_s)
                h.set_status(STALE)
                p.schedule_restart("stale stream")
        return True  # keep the timer

    def stop(self) -> None:
        for p in self._pipelines.values():
            p.stop()
        if self._loop is not None:
            _, GLib = load_gstreamer()
            if self._watchdog_id is not None:
                GLib.source_remove(self._watchdog_id)
            self._loop.quit()
        if self._thread is not None:
            self._thread.join(timeout=5)
        log.info("GStreamer pipelines stopped")
