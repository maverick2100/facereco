"""GStreamer appsink -> numpy BGR conversion."""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

import numpy as np

log = logging.getLogger(__name__)


def sample_to_bgr(sample: Any, Gst: Any) -> Optional[np.ndarray]:
    """Copy a BGR GstSample into a contiguous (H, W, 3) uint8 array (handles row padding)."""
    caps = sample.get_caps()
    st = caps.get_structure(0)
    width = st.get_value("width")
    height = st.get_value("height")
    buf = sample.get_buffer()
    ok, info = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None
    try:
        data = np.frombuffer(info.data, dtype=np.uint8)
        if height <= 0 or data.size < width * height * 3:
            return None
        stride = data.size // height
        rows = data[: stride * height].reshape(height, stride)
        img = np.ascontiguousarray(rows[:, : width * 3]).reshape(height, width, 3)
        if img.base is not None and np.shares_memory(img, data):
            img = img.copy()  # never keep a view into a GStreamer buffer
    finally:
        buf.unmap(info)
    return img


class AppSinkHandler:
    def __init__(self, camera_id: str, on_frame: Callable[[np.ndarray], None], Gst: Any) -> None:
        self.camera_id = camera_id
        self._on_frame = on_frame
        self._Gst = Gst
        self._last_err_log = 0.0

    def attach(self, appsink: Any) -> None:
        appsink.set_property("emit-signals", True)
        appsink.connect("new-sample", self._on_new_sample)

    def _on_new_sample(self, sink: Any) -> Any:
        Gst = self._Gst
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        try:
            img = sample_to_bgr(sample, Gst)
            if img is not None:
                self._on_frame(img)
        except Exception:  # never let an exception kill the streaming thread
            now = time.monotonic()
            if now - self._last_err_log > 10:
                self._last_err_log = now
                log.exception("[%s] failed to handle sample", self.camera_id)
        return Gst.FlowReturn.OK
