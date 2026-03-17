"""
Step 002 - person detection using YOLOv8 ONNX.

Public API:
- detect_persons(session, image_path, roi=None)
"""

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from config import CONFIDENCE_THRESHOLD, INPUT_SIZE, NMS_IOU_THRESHOLD
from step001_model_io import _letterbox_params, preprocess_image


Box = Tuple[int, int, int, int]


def detect_persons(
    session: ort.InferenceSession,
    image_path: str,
    roi: Optional[Sequence[int]] = None,
) -> Tuple[Optional[np.ndarray], List[Box]]:
    """
    Run YOLO on a frame (optionally cropped to ROI) and return:

    Returns:
        (image_bgr, boxes), where boxes is a list of (x1, y1, x2, y2) in full-image coordinates.
        If the image cannot be read, returns (None, []).
    """
    img_full = cv2.imread(image_path)
    if img_full is None:
        return None, []
    #-----------------------------------------ROI cropping-----------------------------------------
    if roi is not None:
        h_full, w_full = img_full.shape[:2]
        rx1, ry1, rx2, ry2 = roi
        rx1 = max(0, min(rx1, w_full - 1))
        rx2 = max(0, min(rx2, w_full))
        ry1 = max(0, min(ry1, h_full - 1))
        ry2 = max(0, min(ry2, h_full))
        if rx2 <= rx1 or ry2 <= ry1:
            roi = None
            img_for_yolo = img_full #Use the full image if the ROI is invalid.
        else:
            img_for_yolo = img_full[ry1:ry2, rx1:rx2] #Crop the image to the ROI.
    else:
        img_for_yolo = img_full #Use the full image if no ROI is provided.
        h_full, w_full = img_full.shape[:2] #Get the full image dimensions.

    # Preprocessed tensor (letterbox, BGR→RGB, normalize [0,1], NCHW) is passed to YOLO.
    preprocessed = preprocess_image(img_for_yolo, INPUT_SIZE)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: preprocessed})
    output = outputs[0]

    if len(output.shape) == 3:
        output = output[0]
    #-----------------------------------------Scale and padding-----------------------------------------
    #Letterbox - Computes scale and padding so the image fits in a square (input_size×input_size) without stretching.
    scale, pad_left, pad_top = _letterbox_params(img_for_yolo, INPUT_SIZE)
    # Parse YOLO output: boxes (cx,cy,w,h) and per-class scores. Class 0 = person.
    boxes_xywh = output[0:4, :].T
    scores = output[4:, :].transpose()
    person_scores = scores[:, 0]

    h_det, w_det = img_for_yolo.shape[:2]
    candidates_xyxy: List[Box] = []
    candidate_scores = []

    # Person or not: only keep detections with score > CONFIDENCE_THRESHOLD.
    for i, score in enumerate(person_scores):
        if score <= CONFIDENCE_THRESHOLD:
            continue

        cx, cy, w, h = boxes_xywh[i]
        orig_cx = (cx - pad_left) / scale
        orig_cy = (cy - pad_top) / scale
        orig_w = w / scale
        orig_h = h / scale

        x1_local = int(orig_cx - orig_w / 2)
        y1_local = int(orig_cy - orig_h / 2)
        x2_local = int(orig_cx + orig_w / 2)
        y2_local = int(orig_cy + orig_h / 2)

        x1_local = max(0, min(x1_local, w_det))
        y1_local = max(0, min(y1_local, h_det))
        x2_local = max(0, min(x2_local, w_det))
        y2_local = max(0, min(y2_local, h_det))

        if roi is not None:
            x1 = x1_local + rx1
            y1 = y1_local + ry1
            x2 = x2_local + rx1
            y2 = y2_local + ry1
            x1 = max(0, min(x1, w_full))
            y1 = max(0, min(y1, h_full))
            x2 = max(0, min(x2, w_full))
            y2 = max(0, min(y2, h_full))
        else:
            x1, y1, x2, y2 = x1_local, y1_local, x2_local, y2_local

        candidates_xyxy.append((x1, y1, x2, y2))
        candidate_scores.append(float(score))

    if not candidates_xyxy:
        return img_full, []

    boxes_xywh_for_nms = [
        [x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in candidates_xyxy
    ]

    # NMS removes overlapping boxes; remaining boxes are the final person detections.
    indices = cv2.dnn.NMSBoxes(
        boxes_xywh_for_nms,
        candidate_scores,
        CONFIDENCE_THRESHOLD,
        NMS_IOU_THRESHOLD,
    )
    if len(indices) > 0:
        indices = indices.flatten() if hasattr(indices, "flatten") else indices
    else:
        indices = []

    person_boxes = [candidates_xyxy[int(i)] for i in indices]
    return img_full, person_boxes

