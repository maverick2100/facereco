import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.live_view import KNOWN, PENDING, FaceBox, LiveView  # noqa: E402


def test_mosaic_draws_boxes_and_keeps_labels():
    lv = LiveView({"c1": "Door", "c2": "Lobby", "c3": "Back"}, {"c1": [0, 0, 640, 360]}, window_width=1200)
    img = np.full((720, 1280, 3), 80, np.uint8)
    ts = time.time()
    lv.publish_faces("c1", img, ts, [FaceBox([100, 100, 300, 320], "Alice", KNOWN, 0.82)])
    lv.publish_persons("c2", img, ts, np.array([[50, 50, 400, 700, 0.9]], np.float32))
    lv.publish_faces("unknown-cam", img, ts, [])          # cameras not selected are ignored

    m = lv.mosaic({"c1": "streaming 4.0 fps"})
    assert m.shape == (2 * (600 * 9 // 16), 1200, 3)       # 3 cameras -> 2x2 grid of 600px tiles
    green = (m[:337, :600] == (0, 200, 0)).all(axis=2)     # known-face box on camera 1
    orange = (m[:337, 600:] == (230, 160, 0)).all(axis=2)  # person box on camera 2
    assert green.sum() > 100 and orange.sum() > 100

    # re-check frame: a pending face keeps its previous name
    lv.publish_faces("c1", img, ts + 0.25, [FaceBox([104, 102, 302, 322], "", PENDING)])
    assert lv._slots["c1"].faces[0].label == "Alice"


def test_run_window_stops_on_event(monkeypatch):
    import services.live_view as lvmod

    shown = []
    monkeypatch.setattr(lvmod.cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(lvmod.cv2, "imshow", lambda name, img: shown.append(img.shape))
    monkeypatch.setattr(lvmod.cv2, "waitKey", lambda d: -1)
    monkeypatch.setattr(lvmod.cv2, "getWindowProperty", lambda *a: 1.0)
    monkeypatch.setattr(lvmod.cv2, "destroyAllWindows", lambda: None)
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    LiveView({"c1": "Door"}).run_window(stop, fps=50)
    assert shown


def test_run_window_quits_on_q(monkeypatch):
    import services.live_view as lvmod

    monkeypatch.setattr(lvmod.cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(lvmod.cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(lvmod.cv2, "waitKey", lambda d: ord("q"))
    monkeypatch.setattr(lvmod.cv2, "destroyAllWindows", lambda: None)
    stop = threading.Event()
    LiveView({"c1": "Door"}).run_window(stop)
    assert stop.is_set()
