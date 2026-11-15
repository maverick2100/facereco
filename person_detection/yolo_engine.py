"""YOLOv7 person detector on Core ML.

Output layouts differ between ONNX -> Core ML conversions, so they are recognised
by shape (see `YOLOv7Engine.decode`):

  * decoded      (1, N, 5+nc) [cx, cy, w, h, obj, cls...]  - yolov7 export.py --grid
                 (1, 5+nc, N) transposed, or (1, N, 4+nc) without objectness
  * raw heads    (1, 3, H, W, 5+nc) per stride, or NCHW (1, 3*(5+nc), H, W)  - export without --grid
  * end2end NMS  (N, 7) [batch, x1, y1, x2, y2, cls, score]                  - export --end2end
  * Core ML NMS  "coordinates" (N, 4) normalised cx,cy,w,h + "confidence" (N, nc)
  * split        "boxes" (1, N, 4) x1,y1,x2,y2 + "scores" (1, N) best-class score x objectness
                 + "class_idx" (1, N) best class  - decode + argmax inside the graph, no NMS

Only the person class is kept. Boxes are returned in original-frame pixels.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from constants import COCO_PERSON_CLASS_ID, YOLO_DEFAULT_INPUT, YOLO_PAD_VALUE, YOLOV7_ANCHORS
from processing.face_detection import nms
from processing.image_preprocessing import input_size_of, to_model_input

log = logging.getLogger(__name__)

Letterbox = Tuple[float, float, float]  # scale, pad_x, pad_y


def letterbox_center(img: np.ndarray, size: Tuple[int, int], pad_value: int = YOLO_PAD_VALUE
                     ) -> Tuple[np.ndarray, Letterbox]:
    """YOLO-style letterbox: keep aspect ratio, centre on a grey (w, h) canvas.

    original = (model - pad) / scale
    """
    tw, th = size
    h, w = img.shape[:2]
    scale = min(tw / float(w), th / float(h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img if (nw, nh) == (w, h) else cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if (nw, nh) == (tw, th):
        return np.ascontiguousarray(resized), (scale, 0.0, 0.0)
    px, py = (tw - nw) // 2, (th - nh) // 2
    canvas = np.full((th, tw, 3), pad_value, dtype=np.uint8)
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, (scale, float(px), float(py))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _empty() -> np.ndarray:
    return np.zeros((0, 5), dtype=np.float32)


class YOLOv7Engine:
    def __init__(self, model: Any, conf_threshold: float = 0.4, nms_threshold: float = 0.45,
                 class_id: int = COCO_PERSON_CLASS_ID, num_classes: int = 80, input_std: float = 255.0,
                 max_detections: int = 300) -> None:
        self.model = model
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.class_id = class_id
        self.num_classes = num_classes
        self.input_std = input_std
        self.max_detections = max_detections
        image_inputs = [i for i in model.inputs if i.kind in ("image", "multiarray")]
        if not image_inputs:
            raise RuntimeError(f"YOLO model has no image/array input: {[i.name for i in model.inputs]}")
        # Pipeline models exported with Core ML NMS also take iouThreshold / confidenceThreshold.
        self.input_spec = next((i for i in image_inputs if i.kind == "image"), image_inputs[0])
        self._extra_inputs = {
            i.name: i for i in model.inputs
            if i.name != self.input_spec.name and i.name in ("iouThreshold", "confidenceThreshold")
        }
        self.input_w, self.input_h = input_size_of(self.input_spec, YOLO_DEFAULT_INPUT)
        self._grids: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._layout_logged = False
        log.info("YOLOv7 input %dx%d (%s), person class %d", self.input_w, self.input_h,
                 self.input_spec.kind, class_id)

    @property
    def input_size(self) -> Tuple[int, int]:
        return self.input_w, self.input_h

    # ------------------------------------------------------------------ public
    def preprocess(self, img_bgr: np.ndarray) -> Tuple[Dict[str, Any], Letterbox]:
        canvas, lb = letterbox_center(img_bgr, (self.input_w, self.input_h))
        feeds: Dict[str, Any] = {self.input_spec.name: to_model_input(canvas, self.input_spec, 0.0, self.input_std)}
        for name in self._extra_inputs:
            value = self.nms_threshold if name == "iouThreshold" else self.conf_threshold
            feeds[name] = np.array([value], dtype=np.float64)
        return feeds, lb

    def detect_batch(self, images: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Returns one (N, 5) [x1, y1, x2, y2, score] array per image, in image pixels."""
        if not images:
            return []
        prepared = [self.preprocess(img) for img in images]
        outputs = self.model.predict_batch([p[0] for p in prepared])
        return [self.postprocess(out, lb, img.shape[:2])
                for out, (_, lb), img in zip(outputs, prepared, images)]

    def detect(self, img_bgr: np.ndarray) -> np.ndarray:
        return self.detect_batch([img_bgr])[0]

    # ------------------------------------------------------------------ decoding
    def postprocess(self, outputs: Dict[str, Any], lb: Letterbox, image_hw: Tuple[int, int]) -> np.ndarray:
        boxes, scores, nms_done = self.decode(outputs)
        if boxes.shape[0] == 0:
            return _empty()
        if not nms_done:
            keep = nms(boxes, scores, self.nms_threshold)[: self.max_detections]
            boxes, scores = boxes[keep], scores[keep]
        scale, px, py = lb
        boxes = boxes.copy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / scale
        h, w = image_hw
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
        return np.hstack([boxes, scores[:, None]]).astype(np.float32)

    def decode(self, outputs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, bool]:
        """-> (boxes (N,4) x1y1x2y2 in model-input pixels, person scores (N,), nms_already_applied)."""
        arrays = {k: np.asarray(v, dtype=np.float32) for k, v in outputs.items()}
        no = 5 + self.num_classes

        if "coordinates" in arrays and "confidence" in arrays:
            self._log_layout("Core ML NMS (coordinates/confidence)")
            return self._decode_coreml_nms(arrays["coordinates"], arrays["confidence"])

        split = self._find_split(arrays)
        if split is not None:
            self._log_layout("split boxes(x1y1x2y2) / scores / class_idx")
            return self._decode_split(*split)

        heads = []
        for a in arrays.values():
            if a.ndim == 5 and a.shape[-1] == no:
                heads.append(a)
            elif a.ndim == 4 and a.shape[1] == 3 * no:
                _, _, gh, gw = a.shape
                heads.append(a.reshape(a.shape[0], 3, no, gh, gw).transpose(0, 1, 3, 4, 2))
        if heads:
            self._log_layout(f"raw heads x{len(heads)}")
            return self._decode_raw_heads(heads)

        flat = [a for a in arrays.values() if a.ndim >= 2]
        if not flat:
            raise RuntimeError(f"Unrecognised YOLO outputs: { {k: v.shape for k, v in arrays.items()} }")
        a = max(flat, key=lambda x: x.size)
        a = a.reshape(-1, a.shape[-2], a.shape[-1])[0] if a.ndim >= 3 else a
        if a.shape[-1] == 7 and a.shape[0] != 7:
            self._log_layout("end2end NMS (N,7)")
            return self._decode_end2end(a)
        if a.shape[0] in (no, no - 1) and a.shape[1] not in (no, no - 1):
            a = a.T  # (C, N) -> (N, C)
        if a.shape[1] not in (no, no - 1):
            raise RuntimeError(f"Unrecognised YOLO output shape {a.shape}: expected (N, {no}) for "
                               f"{self.num_classes} classes - set FR_PERSON_NUM_CLASSES")
        self._log_layout(f"decoded {a.shape}")
        return self._decode_flat(a, has_obj=a.shape[1] == no)

    def _log_layout(self, desc: str) -> None:
        if not self._layout_logged:
            log.info("YOLO output layout: %s", desc)
            self._layout_logged = True

    def _xywh_to_xyxy(self, xywh: np.ndarray, normalised: Optional[bool] = None) -> np.ndarray:
        if normalised is None:  # guess: some exports emit 0..1 coordinates
            normalised = bool(xywh.size) and float(np.abs(xywh).max()) <= 1.5
        if normalised:
            xywh = xywh * np.array([self.input_w, self.input_h, self.input_w, self.input_h], np.float32)
        cx, cy, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    def _decode_flat(self, a: np.ndarray, has_obj: bool) -> Tuple[np.ndarray, np.ndarray, bool]:
        cls_start = 5 if has_obj else 4
        scores = a[:, cls_start + self.class_id]
        if has_obj:
            scores = scores * a[:, 4]
        idx = np.nonzero(scores >= self.conf_threshold)[0]
        return self._xywh_to_xyxy(a[idx, :4]), scores[idx], False

    def _decode_raw_heads(self, heads: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, bool]:
        all_boxes, all_scores = [], []
        for h in heads:
            _, na, gh, gw, _ = h.shape
            stride = int(round(self.input_h / gh))
            anchors = np.asarray(YOLOV7_ANCHORS.get(stride, YOLOV7_ANCHORS[32])[:na], np.float32)
            y = _sigmoid(h[0])                                   # (na, gh, gw, no)
            scores = y[..., 4] * y[..., 5 + self.class_id]
            idx = np.nonzero(scores >= self.conf_threshold)
            if idx[0].size == 0:
                continue
            sel = y[idx]                                         # (k, no)
            grid = np.stack([idx[2], idx[1]], axis=1).astype(np.float32)  # (x, y)
            xy = (sel[:, :2] * 2.0 - 0.5 + grid) * stride
            wh = (sel[:, 2:4] * 2.0) ** 2 * anchors[idx[0]]
            all_boxes.append(np.concatenate([xy - wh / 2, xy + wh / 2], axis=1))
            all_scores.append(scores[idx])
        if not all_boxes:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), False
        return np.concatenate(all_boxes), np.concatenate(all_scores), False

    _CLASS_NAMES = ("class_idx", "classes", "class_ids", "labels", "class")

    def _find_split(self, arrays: Dict[str, np.ndarray]):
        """(boxes (N,4), scores (N,), classes (N,)) when the model emits them separately, else None."""
        if len(arrays) != 3:
            return None
        # boxes = the (.., N, 4) array; the other two hold N values each
        boxes = [k for k, v in arrays.items() if v.ndim in (2, 3) and v.shape[-1] == 4
                 and all(o.size * 4 == v.size for j, o in arrays.items() if j != k)]
        if len(boxes) != 1:
            return None
        vecs = [k for k in arrays if k != boxes[0]]
        b = arrays[boxes[0]].reshape(-1, 4)
        cls = next((k for k in vecs if k.lower() in self._CLASS_NAMES), None)
        if cls is None:  # names lost in conversion: the class array holds whole numbers
            whole = [k for k in vecs if np.all(arrays[k] == np.round(arrays[k]))]
            if len(whole) != 1:
                return None
            cls = whole[0]
        score = next(k for k in vecs if k != cls)
        return b, arrays[score].reshape(-1), arrays[cls].reshape(-1)

    def _decode_split(self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, bool]:
        idx = np.nonzero((np.round(classes) == self.class_id) & (scores >= self.conf_threshold))[0]
        return boxes[idx], scores[idx], False

    def _decode_end2end(self, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        mask = (a[:, 5].round() == self.class_id) & (a[:, 6] >= self.conf_threshold)
        return a[mask, 1:5], a[mask, 6], True

    def _decode_coreml_nms(self, coords: np.ndarray, conf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        coords = coords.reshape(-1, 4)
        conf = conf.reshape(coords.shape[0], -1) if coords.shape[0] else conf.reshape(0, 1)
        if coords.shape[0] == 0 or conf.shape[1] <= self.class_id:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), True
        scores = conf[:, self.class_id]
        idx = np.nonzero(scores >= self.conf_threshold)[0]
        return self._xywh_to_xyxy(coords[idx], normalised=True), scores[idx], True


def filter_persons(dets: np.ndarray, min_height: float = 0, roi: Optional[Sequence[int]] = None) -> np.ndarray:
    """Drop boxes shorter than `min_height` px or whose centre is outside `roi`."""
    if dets.shape[0] == 0:
        return dets
    keep = (dets[:, 3] - dets[:, 1]) >= min_height
    if roi is not None:
        cx = (dets[:, 0] + dets[:, 2]) / 2.0
        cy = (dets[:, 1] + dets[:, 3]) / 2.0
        keep &= (cx >= roi[0]) & (cx <= roi[2]) & (cy >= roi[1]) & (cy <= roi[3])
    return dets[keep]
