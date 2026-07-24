"""
Step 006 - high-level processing for a single frame pair.

Public API:
- process_frame_pair
"""

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import onnxruntime as ort

from config import (
    DEBUG_IMAGE_EVERY_N,
    DEBUG_OUTPUT_DIR,
    DETECTION_OUTPUT_DIR,
    MATCHING_THRESHOLD,
    REID_DISTANCE_THRESHOLD,
    SAVE_CROSSING_IMAGES,
    SAVE_DEBUG_IMAGES,
)
from pipeline.step002_detection import detect_persons
from pipeline.step003_tracking import (
    PersonTwoLineState,
    assign_ids_by_centroid_matching,
    build_two_line_contexts,
    calculate_centroid,
    match_persons,
    point_is_inside,
    two_line_gate_event,
)
from pipeline.step005_visualization import create_crossing_image_pair, create_debug_image_pair

# (path, boxes, person_ids) for a frame so we can carry IDs to the next pair
FrameData = Tuple[str, List[Tuple[int, int, int, int]], List[int]]


def process_frame_pair(
    prev_image_path: str,
    curr_image_path: str,
    line_y: float,
    line_config: dict,
    line_config_b: dict,
    session: ort.InferenceSession,
    roi: Sequence[int] | None,
    frame_pair_index: int | None = None,
    verbose: bool = True,
    prev_frame_data: Optional[FrameData] = None,
    next_available_id: int = 0,
    reid_history: Optional[List[FrameData]] = None,
    detection_output_dir: str | None = None,
    debug_output_dir: str | None = None,
    crossing_image_folder_prefix: str | None = None,
    crossing_state: Optional[Dict[int, PersonTwoLineState]] = None,
) -> Tuple[int, int, list[tuple[str, str, str]], FrameData, int]:
    """
    Process a single frame pair and detect entry/exit events.

    prev_frame_data: from the last pair's curr frame; used to assign prev_person_ids by centroid matching.
    reid_history: list of (path, boxes, ids) for the last K frames; used to reuse an ID when a person
        was missed for a few frames (re-identification).

    Returns:
        (entries, exits, event_list, curr_frame_data, next_available_id)
        where event_list is (event_type, prev_image_path, curr_image_path),
        curr_frame_data is (curr_image_path, curr_boxes, curr_person_ids) for ID propagation.
    crossing_image_folder_prefix: prepended to crossing image filenames as
    ``{prefix}_crossing_{EVENT}_{prev}_{curr}.jpg`` so multiple sequences can share one output dir.
    line_config_b: line B segment (two-line gate with line_config as line A).
    crossing_state: per-person two-line FSM + per-line hysteresis; pass the same dict for a whole
        sequence (typically from run_sequence). If None, a fresh dict is used (no memory across calls).
    """
    if crossing_state is None:
        crossing_state = {}
    if verbose:
        print(
            f"\nProcessing: {os.path.basename(prev_image_path)} → "
            f"{os.path.basename(curr_image_path)}"
        )
    det_out = detection_output_dir if detection_output_dir is not None else DETECTION_OUTPUT_DIR
    dbg_out = debug_output_dir if debug_output_dir is not None else DEBUG_OUTPUT_DIR
    #-----------------------------------------Detection-----------------------------------------
    prev_img, prev_boxes = detect_persons(session, prev_image_path, roi)
    curr_img, curr_boxes = detect_persons(session, curr_image_path, roi)

    if verbose:
        print(f"  Previous frame: {len(prev_boxes)} person(s) detected")
        print(f"  Current frame: {len(curr_boxes)} person(s) detected")

    prev_centroids = [calculate_centroid(box) for box in prev_boxes]
    curr_centroids = [calculate_centroid(box) for box in curr_boxes]

    if not prev_boxes and not curr_boxes:
        if verbose:
            print("  No persons detected in either frame.")
        return (0, 0, [], (curr_image_path, curr_boxes, []), next_available_id)

    if not prev_centroids or not curr_centroids:
        if verbose:
            print("  Not enough persons detected to detect crossing.")
        return (0, 0, [], (curr_image_path, curr_boxes, []), next_available_id)

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

    # ---------- Giving IDs: assign persistent person IDs using Euclidean distance matching ----------
    # Prev frame IDs: either from last pair's curr (match by centroid) or fresh IDs for first time we see this frame.
    if prev_frame_data is None or prev_frame_data[0] != prev_image_path:
        prev_person_ids = list(range(next_available_id, next_available_id + len(prev_boxes)))
        next_available_id += len(prev_boxes)
    else:
        _last_path, last_boxes, last_ids = prev_frame_data
        last_centroids = [calculate_centroid(b) for b in last_boxes]
        prev_person_ids, next_available_id = assign_ids_by_centroid_matching(
            prev_centroids, last_centroids, last_ids, next_available_id
        )
    # Curr frame IDs: matched persons keep prev ID; unmatched get new IDs (we may overwrite with re-ID below).
    curr_person_ids: List[int] = [-1] * len(curr_boxes)
    matched_curr = set()
    for prev_idx, curr_idx, _dist in matches:
        curr_person_ids[curr_idx] = prev_person_ids[prev_idx]
        matched_curr.add(curr_idx)
    for j in range(len(curr_boxes)):
        if curr_person_ids[j] < 0:
            curr_person_ids[j] = next_available_id
            next_available_id += 1

    # ---------- Re-identification (re-ID): reuse an old ID if person was missed for a few frames ----------
    # When YOLO misses a person for 2–3 frames, they get a new ID when they reappear. Here we look at the last
    # few frames (reid_history): if an unmatched curr person is at roughly the same position as someone who
    # had an ID in that history, we reuse that ID instead of keeping the new one.
    if reid_history and curr_centroids:
        # Build a flat list of (centroid, person_id) from all persons in the history frames.
        history_centroids_and_ids: List[Tuple[Tuple[int, int], int]] = []
        for _path, boxes, ids in reid_history:
            for box, pid in zip(boxes, ids):
                history_centroids_and_ids.append((calculate_centroid(box), pid))
        # IDs already assigned in this frame (from matching); don’t give the same ID to two people.
        used_ids = set(curr_person_ids)
        unmatched_indices = [j for j in range(len(curr_boxes)) if j not in matched_curr]
        for j in unmatched_indices:
            best_id: Optional[int] = None
            best_dist = float("inf")
            for (hist_cent, hist_id) in history_centroids_and_ids:
                if hist_id in used_ids:
                    continue
                d = math.dist(curr_centroids[j], hist_cent)
                if d < REID_DISTANCE_THRESHOLD and d < best_dist:
                    best_dist = d
                    best_id = hist_id
            if best_id is not None:
                curr_person_ids[j] = best_id
                used_ids.add(best_id)
                if verbose:
                    print(f"  Re-ID: curr person {j} re-assigned to ID {best_id} (distance {best_dist:.1f} px)")
    # ---------- End re-ID ----------

    curr_frame_data: FrameData = (curr_image_path, curr_boxes, curr_person_ids)
    # ---------- End giving IDs ----------

    # Save debug image for every frame pair to DEBUG_OUTPUT_DIR (with IDs drawn above each person box)
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
                line_config_b=line_config_b,
                roi=roi,
                prev_img=prev_img,
                curr_img=curr_img,
                prev_person_ids=prev_person_ids,
                curr_person_ids=curr_person_ids,
            )
            if debug_image is not None:
                os.makedirs(dbg_out, exist_ok=True)
                prev_name = os.path.splitext(os.path.basename(prev_image_path))[0]
                curr_name = os.path.splitext(os.path.basename(curr_image_path))[0]
                debug_filename = f"debug_{prev_name}_{curr_name}.jpg"
                debug_path = os.path.join(dbg_out, debug_filename)
                cv2.imwrite(debug_path, debug_image)
                if verbose:
                    print(f"    Debug image saved: {debug_path}")

    entries = 0
    exits = 0
    event_list: list[tuple[str, str, str]] = []
    ctx_a, ctx_b = build_two_line_contexts(line_config, line_config_b)
    #-----------------------------------------Crossing detection-----------------------------------------
    for prev_idx, curr_idx, distance in matches:
        prev_centroid = prev_centroids[prev_idx]
        curr_centroid = curr_centroids[curr_idx]
        person_id = curr_person_ids[curr_idx]
        crossing = two_line_gate_event(
            crossing_state,
            person_id,
            prev_centroid,
            curr_centroid,
            ctx_a,
            ctx_b,
        )
        if not crossing:
            continue

        if verbose:
            pa = point_is_inside(float(prev_centroid[0]), float(prev_centroid[1]), ctx_a)
            pb = point_is_inside(float(prev_centroid[0]), float(prev_centroid[1]), ctx_b)
            ca = point_is_inside(float(curr_centroid[0]), float(curr_centroid[1]), ctx_a)
            cb = point_is_inside(float(curr_centroid[0]), float(curr_centroid[1]), ctx_b)
            print(f"    Person {prev_idx} -> {curr_idx} (id={person_id}): {crossing} detected!")
            print(
                f"      Previous: {prev_centroid} "
                f"(line A: {'in' if pa else 'out'}, line B: {'in' if pb else 'out'})"
            )
            print(
                f"      Current: {curr_centroid} "
                f"(line A: {'in' if ca else 'out'}, line B: {'in' if cb else 'out'})"
            )

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
                line_config_b=line_config_b,
                roi=roi,
                event_type=crossing,
                prev_highlight_idx=prev_idx,
                curr_highlight_idx=curr_idx,
                prev_img=prev_img,
                curr_img=curr_img,
                prev_person_ids=prev_person_ids,
                curr_person_ids=curr_person_ids,
            )
            if crossing_image is not None:
                os.makedirs(det_out, exist_ok=True)
                prev_name = os.path.splitext(os.path.basename(prev_image_path))[0]
                curr_name = os.path.splitext(os.path.basename(curr_image_path))[0]
                if crossing_image_folder_prefix:
                    crossing_filename = (
                        f"{crossing_image_folder_prefix}_crossing_{crossing}_"
                        f"{prev_name}_{curr_name}.jpg"
                    )
                else:
                    crossing_filename = (
                        f"crossing_{crossing}_{prev_name}_{curr_name}.jpg"
                    )
                crossing_path = os.path.join(det_out, crossing_filename)
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

    return (entries, exits, event_list, curr_frame_data, next_available_id)

