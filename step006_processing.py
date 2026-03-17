"""
Step 006 - high-level processing for a single frame pair.

Public API:
- process_frame_pair
"""

import math
import os
from typing import List, Sequence, Tuple

import cv2
import onnxruntime as ort

from config import (
    DEBUG_IMAGE_EVERY_N,
    DEBUG_OUTPUT_DIR,
    DETECTION_OUTPUT_DIR,
    MATCHING_THRESHOLD,
    SAVE_CROSSING_IMAGES,
    SAVE_DEBUG_IMAGES,
)
from step002_detection import detect_persons
from step003_tracking import calculate_centroid, detect_crossing, match_persons
from step005_visualization import create_crossing_image_pair, create_debug_image_pair


def process_frame_pair(
    prev_image_path: str,
    curr_image_path: str,
    line_y: float,
    line_config: dict,
    session: ort.InferenceSession,
    roi: Sequence[int] | None,
    frame_pair_index: int | None = None,
    verbose: bool = True,
) -> Tuple[int, int, list[tuple[str, str, str]]]:
    """
    Process a single frame pair and detect entry/exit events.

    Returns:
        (entries, exits, event_list)
        where event_list is a list of (event_type, prev_image_path, curr_image_path).
    """
    if verbose:
        print(
            f"\nProcessing: {os.path.basename(prev_image_path)} → "
            f"{os.path.basename(curr_image_path)}"
        )
    #-----------------------------------------Detection-----------------------------------------
    prev_img, prev_boxes = detect_persons(session, prev_image_path, roi)
    curr_img, curr_boxes = detect_persons(session, curr_image_path, roi)

    if verbose:
        print(f"  Previous frame: {len(prev_boxes)} person(s) detected")
        print(f"  Current frame: {len(curr_boxes)} person(s) detected")

    prev_centroids = [calculate_centroid(box) for box in prev_boxes]
    curr_centroids = [calculate_centroid(box) for box in curr_boxes]

    # Save debug image for every frame pair to DEBUG_OUTPUT_DIR
    if SAVE_DEBUG_IMAGES:
        should_save_debug = True
        if DEBUG_IMAGE_EVERY_N > 1 and frame_pair_index is not None:
            should_save_debug = frame_pair_index % DEBUG_IMAGE_EVERY_N == 0
        if should_save_debug:
            debug_image = create_debug_image_pair(
                prev_image_path,
                curr_image_path,
                prev_boxes,
                curr_boxes,
                prev_centroids,
                curr_centroids,
                line_config,
                line_y,
                roi=roi,
                prev_img=prev_img,
                curr_img=curr_img,
            )
            if debug_image is not None:
                os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
                prev_name = os.path.splitext(os.path.basename(prev_image_path))[0]
                curr_name = os.path.splitext(os.path.basename(curr_image_path))[0]
                debug_filename = f"debug_{prev_name}_{curr_name}.jpg"
                debug_path = os.path.join(DEBUG_OUTPUT_DIR, debug_filename)
                cv2.imwrite(debug_path, debug_image)
                if verbose:
                    print(f"    Debug image saved: {debug_path}")

    if not prev_boxes and not curr_boxes:
        if verbose:
            print("  No persons detected in either frame.")
        return (0, 0, [])

    if not prev_centroids or not curr_centroids:
        if verbose:
            print("  Not enough persons detected to detect crossing.")
        return (0, 0, [])

    if verbose:
        print("  Previous frame centroids:")
        for idx, cent in enumerate(prev_centroids):
            print(f"    [{idx}] ({cent[0]}, {cent[1]})")
        print("  Current frame centroids:")
        for idx, cent in enumerate(curr_centroids):
            print(f"    [{idx}] ({cent[0]}, {cent[1]})")
    #-----------------------------------------Matching-----------------------------------------
    if verbose and prev_centroids and curr_centroids:
        print("  Distance matrix:")
        for i, prev_cent in enumerate(prev_centroids):
            for j, curr_cent in enumerate(curr_centroids):
                distance = math.dist(prev_cent, curr_cent)
                print(f"    prev[{i}] -> curr[{j}]: {distance:.1f} px")
        all_distances: List[float] = [
            math.dist(p, c) for p in prev_centroids for c in curr_centroids
        ]
        if all_distances:
            print(
                f"  Minimum distance: {min(all_distances):.1f} px "
                f"(threshold: {MATCHING_THRESHOLD} px)"
            )

    matches = match_persons(prev_centroids, curr_centroids)
    if verbose:
        print(f"  Matched {len(matches)} person(s) between frames")
        if not matches and prev_centroids and curr_centroids:
            print(
                "  WARNING: No matches found! "
                f"All distances may exceed threshold of {MATCHING_THRESHOLD} px"
            )

    entries = 0
    exits = 0
    event_list: list[tuple[str, str, str]] = []
    #-----------------------------------------Crossing detection-----------------------------------------
    for prev_idx, curr_idx, distance in matches:
        prev_centroid = prev_centroids[prev_idx]
        curr_centroid = curr_centroids[curr_idx]
        crossing = detect_crossing(prev_centroid[1], curr_centroid[1], line_y)
        if not crossing:
            continue

        if verbose:
            print(f"    Person {prev_idx} -> {curr_idx}: {crossing} detected!")
            print(
                f"      Previous: {prev_centroid} "
                f"(side: {'outside' if prev_centroid[1] < line_y else 'inside'})"
            )
            print(
                f"      Current: {curr_centroid} "
                f"(side: {'outside' if curr_centroid[1] < line_y else 'inside'})"
            )
    #-----------------------------------------Crossing detection-----------------------------------------
        if SAVE_CROSSING_IMAGES:
            crossing_image = create_crossing_image_pair(
                prev_image_path,
                curr_image_path,
                prev_boxes,
                curr_boxes,
                prev_centroids,
                curr_centroids,
                line_config,
                line_y,
                roi=roi,
                event_type=crossing,
                prev_highlight_idx=prev_idx,
                curr_highlight_idx=curr_idx,
                prev_img=prev_img,
                curr_img=curr_img,
            )
            if crossing_image is not None:
                os.makedirs(DETECTION_OUTPUT_DIR, exist_ok=True)
                prev_name = os.path.splitext(os.path.basename(prev_image_path))[0]
                curr_name = os.path.splitext(os.path.basename(curr_image_path))[0]
                crossing_filename = (
                    f"crossing_{crossing}_{prev_name}_{curr_name}.jpg"
                )
                crossing_path = os.path.join(DETECTION_OUTPUT_DIR, crossing_filename)
                cv2.imwrite(crossing_path, crossing_image)
                if verbose:
                    print(f"    Crossing image saved: {crossing_path}")

        #-----------------------------------------Event recording-----------------------------------------
        if crossing == "ENTRY":
            entries += 1
            event_list.append(("ENTRY", prev_image_path, curr_image_path))
        elif crossing == "EXIT":
            exits += 1
            event_list.append(("EXIT", prev_image_path, curr_image_path))

    return (entries, exits, event_list)

