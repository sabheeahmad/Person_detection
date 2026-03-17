"""
Step 003 - geometric helpers, line configuration, and crossing logic.

Public API:
- calculate_centroid
- load_configs
- match_persons
- detect_crossing
"""

import json
import math
import os
from typing import Dict, List, Sequence, Tuple

from config import CENTROID_POSITION_RATIO, LINE_CONFIG_FILE, MATCHING_THRESHOLD


Point = Tuple[int, int]


def calculate_centroid(bounding_box: Sequence[int]) -> Point:
    """
    Compute a stable centroid for a person bounding box.

    The x-coordinate is the horizontal center.
    The y-coordinate is near the feet (controlled by CENTROID_POSITION_RATIO),
    which makes line-crossing detection more robust.
    """
    x1, y1, x2, y2 = bounding_box
    centroid_x = int((x1 + x2) / 2)
    centroid_y = int(y1 + CENTROID_POSITION_RATIO * (y2 - y1))
    return (centroid_x, centroid_y)


def load_configs() -> Dict[str, dict]:
    """
    Load line configuration from JSON file.

    Returns:
        Dict with key 'line' containing x1, y1, x2, y2 and pre-computed y_avg.
    """
    if not os.path.exists(LINE_CONFIG_FILE):
        raise FileNotFoundError(f"Line config not found: {LINE_CONFIG_FILE}")

    with open(LINE_CONFIG_FILE, "r") as f:
        line_config = json.load(f)

    line_y = (line_config["y1"] + line_config["y2"]) / 2
    return {
        "line": {
            "x1": line_config["x1"],
            "y1": line_config["y1"],
            "x2": line_config["x2"],
            "y2": line_config["y2"],
            "y_avg": line_y,
        }
    }


def match_persons(
    prev_centroids: List[Point],
    curr_centroids: List[Point],
) -> List[Tuple[int, int, float]]:
    """
    Match persons between frames using greedy closest-pair matching.

    Returns:
        List of tuples (prev_index, curr_index, distance_pixels).
    """
    if not prev_centroids or not curr_centroids:
        return []

    all_pairs = []
    for i, prev_cent in enumerate(prev_centroids):
        for j, curr_cent in enumerate(curr_centroids):
            distance = math.dist(prev_cent, curr_cent)
            if distance < MATCHING_THRESHOLD:
                all_pairs.append((distance, i, j))

    all_pairs.sort(key=lambda x: x[0])

    matches: List[Tuple[int, int, float]] = []
    used_prev = set()
    used_curr = set()

    for distance, prev_idx, curr_idx in all_pairs:
        if prev_idx in used_prev or curr_idx in used_curr:
            continue
        matches.append((prev_idx, curr_idx, distance))
        used_prev.add(prev_idx)
        used_curr.add(curr_idx)

    return matches


def detect_crossing(prev_centroid_y: int, curr_centroid_y: int, line_y: float):
    """
    Determine if a tracked person crossed the line.

    Returns:
        'ENTRY' if outside -> inside,
        'EXIT' if inside -> outside,
        None otherwise.

    Convention:
        - y < line_y: outside
        - y >= line_y: inside
    """
    prev_side = "outside" if prev_centroid_y < line_y else "inside"
    curr_side = "outside" if curr_centroid_y < line_y else "inside"

    if prev_side == curr_side:
        return None

    if prev_side == "outside" and curr_side == "inside":
        return "ENTRY"
    if prev_side == "inside" and curr_side == "outside":
        return "EXIT"
    return None

