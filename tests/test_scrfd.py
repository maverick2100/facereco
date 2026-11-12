import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fakes import FakeSCRFDModel  # noqa: E402
from inference.scrfd_engine import SCRFDEngine  # noqa: E402
from processing.face_detection import filter_faces, faces_from_detections, nms  # noqa: E402
from processing.image_preprocessing import letterbox, to_model_input  # noqa: E402


def test_letterbox_scale_and_padding():
    img = np.full((720, 1280, 3), 200, np.uint8)
    canvas, scale = letterbox(img, (640, 640))
    assert canvas.shape == (640, 640, 3)
    assert scale == pytest.approx(0.5)
    assert canvas[359, 0, 0] == 200 and canvas[400, 0, 0] == 0


def test_multiarray_input_normalisation():
    model = FakeSCRFDModel()
    img = np.zeros((640, 640, 3), np.uint8)
    img[..., 2] = 255  # red in BGR
    x = to_model_input(img, model.inputs[0], 127.5, 128.0)
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32
    assert x[0, 0, 0, 0] == pytest.approx((255 - 127.5) / 128)  # channel 0 = R after swapRB


@pytest.mark.parametrize("as_nchw", [False, True])
def test_decode_single_face_on_1280x720(as_nchw):
    eng = SCRFDEngine(FakeSCRFDModel(as_nchw=as_nchw), det_threshold=0.5)
    frame = np.zeros((720, 1280, 3), np.uint8)
    dets, kps = eng.detect(frame)
    assert dets.shape == (1, 5) and kps.shape == (1, 5, 2)
    # anchor (320,320) in model space -> (640,640) in a frame scaled by 0.5; half-size 40 -> 80
    np.testing.assert_allclose(dets[0, :4], [560, 560, 720, 720], atol=1e-3)
    assert dets[0, 4] == pytest.approx(0.9)
    np.testing.assert_allclose(kps[0, 0], [610, 620], atol=1e-3)
    assert [lv.stride for lv in eng._levels] == [8, 16, 32] and eng.num_anchors == 2


def test_threshold_filters_everything():
    eng = SCRFDEngine(FakeSCRFDModel(), det_threshold=0.95)
    dets, kps = eng.detect(np.zeros((640, 640, 3), np.uint8))
    assert dets.shape == (0, 5) and kps.shape == (0, 5, 2)


def test_batch_detection():
    model = FakeSCRFDModel()
    eng = SCRFDEngine(model)
    res = eng.detect_batch([np.zeros((720, 1280, 3), np.uint8)] * 4)
    assert len(res) == 4 and all(r[0].shape[0] == 1 for r in res) and model.calls == 4


def test_nms_and_filters():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [300, 300, 340, 340]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = nms(boxes, scores, 0.4)
    assert keep.tolist() == [0, 2]
    dets = np.hstack([boxes, scores[:, None]])
    faces = faces_from_detections(dets, np.zeros((3, 5, 2), np.float32))
    assert len(filter_faces(faces, min_size=50)) == 2
    assert len(filter_faces(faces, roi=[250, 250, 400, 400])) == 1
    assert len(filter_faces(faces, max_faces=1)) == 1
