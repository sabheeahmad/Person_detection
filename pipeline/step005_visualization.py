"""
Step 005 - visualization helpers for debugging and event snapshots.

Public API:
- draw_debug_annotations
- create_debug_image_pair
- create_crossing_image_pair
"""

import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


Box = Tuple[int, int, int, int]
Point = Tuple[int, int]


def draw_debug_annotations(
    image_path: str,
    boxes: List[Box],
    centroids: List[Point],
    line_config: dict,
    line_y: float,
    highlight_person_index: Optional[int] = None,
    img: Optional[np.ndarray] = None,
    roi: Optional[Sequence[int]] = None,
    person_ids: Optional[List[int]] = None,
    line_config_b: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """
    Draw bounding boxes, centroids, ROI, and line overlay on a single frame.
    If person_ids is provided, draw "ID: <id>" above each person box.

    If img is provided, it is used directly; otherwise image_path is loaded from disk.
    """
    if img is None:
        img = cv2.imread(image_path)
    if img is None:
        return None

    img_copy = img.copy()

    if roi is not None:
        rx1, ry1, rx2, ry2 = roi
        cv2.rectangle(img_copy, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

    cv2.line(
        img_copy,
        (int(line_config["x1"]), int(line_config["y1"])),
        (int(line_config["x2"]), int(line_config["y2"])),
        (0, 0, 255),
        3,
    )
    if line_config_b is not None:
        cv2.line(
            img_copy,
            (int(line_config_b["x1"]), int(line_config_b["y1"])),
            (int(line_config_b["x2"]), int(line_config_b["y2"])),
            (0, 255, 255),
            3,
        )

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        if highlight_person_index is not None and idx == highlight_person_index:
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 255), 4)
        else:
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # ---------- Giving IDs: draw person ID above the box ----------
        if person_ids is not None and idx < len(person_ids):
            id_text = f"ID:{person_ids[idx]}"
            (tw, th), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_x = x1
            text_y = max(y1 - 5, th + 5)
            cv2.rectangle(img_copy, (text_x, text_y - th - 2), (text_x + tw + 4, text_y + 2), (0, 0, 0), -1)
            cv2.putText(img_copy, id_text, (text_x + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for centroid in centroids:
        cx, cy = centroid
        cv2.circle(img_copy, (cx, cy), 8, (0, 0, 255), -1)
        label = f"({cx}, {cy})"
        text_x = cx - 30
        text_y = cy - 15
        if text_y < 20:
            text_y = cy + 25
        cv2.putText(
            img_copy,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    cv2.putText(
        img_copy,
        f"Persons: {len(boxes)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    line_label = f"Line A Y_avg: {line_y:.1f}"
    if line_config_b is not None:
        by = (line_config_b["y1"] + line_config_b["y2"]) / 2
        line_label += f" | B Y_avg: {by:.1f}"
    cv2.putText(
        img_copy,
        line_label,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    return img_copy


def _combine_pair(
    prev_annotated: np.ndarray,
    curr_annotated: np.ndarray,
) -> Tuple[np.ndarray, int, int, int]:
    """Stack two annotated frames horizontally with a separator."""
    h1, w1 = prev_annotated.shape[:2]
    h2, w2 = curr_annotated.shape[:2]
    max_height = max(h1, h2)
    total_width = w1 + w2 + 20

    if h1 != max_height:
        prev_annotated = cv2.resize(prev_annotated, (w1, max_height))
    if h2 != max_height:
        curr_annotated = cv2.resize(curr_annotated, (w2, max_height))

    combined = np.zeros((max_height, total_width, 3), dtype=np.uint8)
    combined[:, :w1] = prev_annotated
    cv2.line(combined, (w1 + 10, 0), (w1 + 10, max_height), (255, 255, 255), 2)
    combined[:, w1 + 20 :] = curr_annotated
    return combined, max_height, w1, total_width


def create_debug_image_pair(
    prev_image_path: str,
    curr_image_path: str,
    prev_boxes: List[Box],
    curr_boxes: List[Box],
    prev_centroids: List[Point],
    curr_centroids: List[Point],
    line_config: dict,
    line_y: float,
    roi: Optional[Sequence[int]] = None,
    prev_img: Optional[np.ndarray] = None,
    curr_img: Optional[np.ndarray] = None,
    prev_person_ids: Optional[List[int]] = None,
    curr_person_ids: Optional[List[int]] = None,
    line_config_b: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """
    Create a side-by-side debug image for a frame pair.
    If prev_person_ids/curr_person_ids are provided, each person is labeled with ID above the box.

    Useful for inspecting detections and centroids even when no crossing occurs.
    """
    prev_annotated = draw_debug_annotations(
        prev_image_path,
        prev_boxes,
        prev_centroids,
        line_config,
        line_y,
        None,
        prev_img,
        roi,
        person_ids=prev_person_ids,
        line_config_b=line_config_b,
    )
    curr_annotated = draw_debug_annotations(
        curr_image_path,
        curr_boxes,
        curr_centroids,
        line_config,
        line_y,
        None,
        curr_img,
        roi,
        person_ids=curr_person_ids,
        line_config_b=line_config_b,
    )
    if prev_annotated is None or curr_annotated is None:
        return None

    combined, max_height, w1, _total_width = _combine_pair(prev_annotated, curr_annotated)

    cv2.putText(
        combined,
        f"Prev: {os.path.basename(prev_image_path)}",
        (10, max_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        combined,
        f"Curr: {os.path.basename(curr_image_path)}",
        (w1 + 30, max_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return combined


def create_crossing_image_pair(
    prev_image_path: str,
    curr_image_path: str,
    prev_boxes: List[Box],
    curr_boxes: List[Box],
    prev_centroids: List[Point],
    curr_centroids: List[Point],
    line_config: dict,
    line_y: float,
    roi: Optional[Sequence[int]],
    event_type: str,
    prev_highlight_idx: int,
    curr_highlight_idx: int,
    prev_img: Optional[np.ndarray] = None,
    curr_img: Optional[np.ndarray] = None,
    prev_person_ids: Optional[List[int]] = None,
    curr_person_ids: Optional[List[int]] = None,
    line_config_b: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """
    Create a side-by-side image for a detected crossing with a clear label.
    If prev_person_ids/curr_person_ids are provided, each person is labeled with ID above the box.
    """
    prev_annotated = draw_debug_annotations(
        prev_image_path,
        prev_boxes,
        prev_centroids,
        line_config,
        line_y,
        prev_highlight_idx,
        prev_img,
        roi,
        person_ids=prev_person_ids,
        line_config_b=line_config_b,
    )
    curr_annotated = draw_debug_annotations(
        curr_image_path,
        curr_boxes,
        curr_centroids,
        line_config,
        line_y,
        curr_highlight_idx,
        curr_img,
        roi,
        person_ids=curr_person_ids,
        line_config_b=line_config_b,
    )
    if prev_annotated is None or curr_annotated is None:
        return None

    combined, max_height, w1, total_width = _combine_pair(prev_annotated, curr_annotated)

    if event_type == "ENTRY":
        direction_text = "OUTSIDE → INSIDE"
        event_color = (0, 255, 0)
    else:
        direction_text = "INSIDE → OUTSIDE"
        event_color = (0, 0, 255)

    event_text = f"{event_type} DETECTED"
    text_size = cv2.getTextSize(event_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    text_x = (total_width - text_size[0]) // 2
    cv2.putText(
        combined,
        event_text,
        (text_x, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        event_color,
        3,
    )

    dir_text_size = cv2.getTextSize(direction_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    dir_text_x = (total_width - dir_text_size[0]) // 2
    cv2.putText(
        combined,
        direction_text,
        (dir_text_x, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    arrow_start = (w1 - 20, max_height // 2)
    arrow_end = (w1 + 40, max_height // 2)
    cv2.arrowedLine(combined, arrow_start, arrow_end, (255, 255, 0), 3, tipLength=0.3)

    cv2.putText(
        combined,
        f"Prev: {os.path.basename(prev_image_path)}",
        (10, max_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        combined,
        f"Curr: {os.path.basename(curr_image_path)}",
        (w1 + 30, max_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return combined

