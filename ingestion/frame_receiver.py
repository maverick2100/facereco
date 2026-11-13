"""Frame receivers: turn decoded images into `Frame`s, plus an OpenCV fallback source.

The OpenCV source uses FFmpeg *software* decoding and therefore costs much more
CPU than the GStreamer/VideoToolbox path. Use it only for testing or if
GStreamer is not installed.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, List, Optional, Sequence, Union

import cv2
import numpy as np

from config.settings import CameraConfig, IngestionSettings
from ingestion.camera_health import STOPPED, CameraHealth, CameraHealthRegistry
from scheduler.camera_queue import CameraQueue, Frame

log = logging.getLogger(__name__)


class FrameReceiver:
    """Callable sink: image -> Frame -> CameraQueue(s) (+ health accounting).

    A camera running several tasks (face + person) is decoded once; the same read-only
    frame is handed to each task's queue.
    """

    def __init__(self, camera_id: str, queue: Union[CameraQueue, Sequence[CameraQueue]],
                 health: Optional[CameraHealth] = None) -> None:
        self.camera_id = camera_id
        self.queues: List[CameraQueue] = [queue] if isinstance(queue, CameraQueue) else list(queue)
        self.health = health
        self._frame_id = 0

    @property
    def queue(self) -> CameraQueue:
        return self.queues[0]

    def __call__(self, image: np.ndarray) -> None:
        self._frame_id += 1
        frame = Frame(self.camera_id, self._frame_id, image, time.time(), time.monotonic())
        for q in self.queues:
            q.put(frame)
        if self.health is not None:
            self.health.on_frame()


class OpenCVFrameSource:
    def __init__(self, camera: CameraConfig, receiver: FrameReceiver, health: CameraHealth,
                 settings: IngestionSettings) -> None:
        self.camera = camera
        self.receiver = receiver
        self.health = health
        self.settings = settings
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"cv-{self.camera.camera_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.health.set_status(STOPPED)

    def _open(self) -> "cv2.VideoCapture":
        url = self.camera.url
        if url.startswith("file://"):
            url = url[len("file://"):]
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self) -> None:
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"rtsp_transport;{self.camera.transport}")
        backoff = self.settings.reconnect_delay_s
        interval = 1.0 / self.camera.fps
        target = (self.camera.width, self.camera.height)
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                self.health.on_error("open failed")
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self.settings.reconnect_max_delay_s)
                self.health.on_reconnect("open failed")
                continue
            backoff = self.settings.reconnect_delay_s
            last_emit = 0.0
            while not self._stop.is_set():
                if not cap.grab():  # grab() decodes but skips colour conversion
                    self.health.on_error("read failed")
                    break
                now = time.monotonic()
                if now - last_emit < interval:
                    continue
                ok, img = cap.retrieve()
                if not ok or img is None:
                    continue
                last_emit = now
                if (img.shape[1], img.shape[0]) != target:
                    img = cv2.resize(img, target, interpolation=cv2.INTER_AREA)
                self.receiver(img)
            cap.release()
            if not self._stop.is_set():
                self.health.on_reconnect("stream ended")
                self._stop.wait(backoff)


class OpenCVManager:
    """Same interface as GstManager, using OpenCV sources."""

    def __init__(self, cameras: List[CameraConfig], receivers: Dict[str, FrameReceiver],
                 health: CameraHealthRegistry, settings: IngestionSettings) -> None:
        self._sources = [
            OpenCVFrameSource(c, receivers[c.camera_id], health.get(c.camera_id), settings) for c in cameras
        ]

    def start(self) -> None:
        log.warning("Using OpenCV/FFmpeg software decoding - expect much higher CPU than GStreamer+VideoToolbox")
        for s in self._sources:
            s.start()

    def stop(self) -> None:
        for s in self._sources:
            s.stop()
