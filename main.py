"""
Step 2.1 Detection entrypoint.

Coordinates:
- model loading (GPU-only),
- ROI and line configuration,
- per-frame-pair processing with crossing detection,
- CSV-backed state for total entries / exits.

By default, every immediate subfolder of `frames/` that contains at least two
images is processed in sorted order. Crossing images are written flat under
`detection_output/` with filenames ``{source_folder}_crossing_{EVENT}_{prev}_{curr}.jpg``.
Debug images still use `debug_output/<subfolder>/`.

All ENTRY/EXIT events append to one universal CSV: `detection_output/counts.csv`
(see `COUNTS_CSV_FILE` in config).

Set env `P_CHECK_FRAMES_DIR` to a single folder (e.g. `frames/test1`) to
process only that path. Crossing images use the capture folder basename as the prefix.
CSV remains universal under `detection_output/counts.csv`.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

from config import (
    CAPTURED_FRAMES_DIR,
    COUNTS_CSV_FILE,
    CROSSING_HYSTERESIS_FRAMES,
    DEBUG_OUTPUT_DIR,
    DETECTION_OUTPUT_DIR,
    FRAME_STEP,
    FRAMES_ROOT,
    REID_HISTORY_FRAMES,
    SLEEP_BETWEEN_PAIRS,
)
from pipeline.step001_model_io import load_mask_roi, load_model
from pipeline.step006_processing import process_frame_pair
from pipeline.step004_state_io import append_event_to_csv, ensure_counts_csv, read_state_from_csv
from pipeline.step003_tracking import load_configs

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# Keep crossing filenames portable and under typical filesystem limits.
_CROSSING_PREFIX_MAX_LEN = 120
_CROSSING_PREFIX_BAD_CHARS = '<>:"/\\|?*\n\r\t'


def _crossing_filename_prefix(capture_dir: str, output_slug: Optional[str]) -> str:
    raw = (
        output_slug
        if output_slug is not None
        else os.path.basename(os.path.abspath(capture_dir.rstrip(os.sep)))
    )
    raw = (raw or "capture").strip()
    safe = "".join(
        c if c not in _CROSSING_PREFIX_BAD_CHARS else "_" for c in raw
    ).strip(" .") or "capture"
    if len(safe) > _CROSSING_PREFIX_MAX_LEN:
        safe = safe[:_CROSSING_PREFIX_MAX_LEN]
    return safe


def _list_frame_paths(capture_dir: str) -> List[str]:
    names = [
        f
        for f in os.listdir(capture_dir)
        if any(f.endswith(ext) for ext in IMAGE_EXTENSIONS)
    ]
    names.sort()
    return [os.path.join(capture_dir, f) for f in names]


def discover_frame_sequences(frames_root: str) -> List[Tuple[str, str]]:
    """
    Find capture directories to run.

    Returns (capture_dir, output_slug) where output_slug names subfolders under
    debug_output/ and is used as the crossing-image filename prefix. If images
    live directly in frames_root, slug is the basename of frames_root (e.g. "frames").
    """
    if not os.path.isdir(frames_root):
        return []

    subdirs = sorted(
        name
        for name in os.listdir(frames_root)
        if os.path.isdir(os.path.join(frames_root, name)) and not name.startswith(".")
    )
    sequences: List[Tuple[str, str]] = []
    for name in subdirs:
        path = os.path.join(frames_root, name)
        if len(_list_frame_paths(path)) >= 2:
            sequences.append((path, name))

    if sequences:
        return sequences

    if len(_list_frame_paths(frames_root)) >= 2:
        slug = os.path.basename(os.path.abspath(frames_root)) or "frames"
        return [(frames_root, slug)]

    return []


def _resolve_runs() -> List[Tuple[str, Optional[str]]]:
    """
    (capture_dir, output_slug or None).

    slug None means single-folder mode: use global DETECTION_OUTPUT_DIR and COUNTS_CSV_FILE.
    """
    if CAPTURED_FRAMES_DIR:
        cap = CAPTURED_FRAMES_DIR
        if not os.path.isdir(cap):
            return []
        if len(_list_frame_paths(cap)) < 2:
            return []
        return [(cap, None)]

    return discover_frame_sequences(FRAMES_ROOT)


def run_sequence(
    capture_dir: str,
    output_slug: Optional[str],
    session,
    roi,
    line_y: float,
    line_config: dict,
    line_config_b: dict,
    resume_state_from_csv: bool,
) -> Tuple[int, int, int]:
    """
    Process all frame pairs in capture_dir.

    Events are always appended to the universal COUNTS_CSV_FILE under
    DETECTION_OUTPUT_DIR. If resume_state_from_csv is True (single-sequence run),
    counters continue from the last row in that file; otherwise they start at zero
    for this sequence so separate clips do not share occupancy.

    Returns (total_entries, total_exits, final_occupancy).
    """
    if output_slug is None:
        dbg_dir = DEBUG_OUTPUT_DIR
    else:
        dbg_dir = os.path.join(DEBUG_OUTPUT_DIR, output_slug)

    os.makedirs(DETECTION_OUTPUT_DIR, exist_ok=True)
    os.makedirs(dbg_dir, exist_ok=True)
    crossing_prefix = _crossing_filename_prefix(capture_dir, output_slug)
    print(f"Crossing images: {DETECTION_OUTPUT_DIR}/ ({crossing_prefix}_crossing_…)")
    print(f"Debug output directory: {dbg_dir}")
    print(f"Universal state CSV: {COUNTS_CSV_FILE}")

    frame_paths = _list_frame_paths(capture_dir)
    if len(frame_paths) < 2:
        print(f"Skip (need >= 2 frames): {capture_dir}")
        return (0, 0, 0)

    print(f"\nFound {len(frame_paths)} frame(s) in {capture_dir}")
    print("=" * 60)
    print("Step 2.1 - Entry/Exit Detection (state in CSV)")
    print("=" * 60)

    if resume_state_from_csv:
        total_inside, cumulative_entered, cumulative_exit = read_state_from_csv(
            COUNTS_CSV_FILE
        )
    else:
        total_inside, cumulative_entered, cumulative_exit = 0, 0, 0
    occupancy = total_inside
    total_entries = cumulative_entered
    total_exits = cumulative_exit
    step = max(1, FRAME_STEP)

    last_frame_data = None
    next_available_id = 0
    frame_history: list = []
    crossing_state: dict = {}

    for i in range(0, len(frame_paths) - 1, step):
        prev_image = frame_paths[i]
        curr_index = min(i + step, len(frame_paths) - 1)
        curr_image = frame_paths[curr_index]

        entries, exits, event_list, curr_frame_data, next_available_id = process_frame_pair(
            prev_image,
            curr_image,
            line_y,
            line_config,
            line_config_b,
            session,
            roi,
            frame_pair_index=i,
            verbose=True,
            prev_frame_data=last_frame_data,
            next_available_id=next_available_id,
            reid_history=frame_history,
            detection_output_dir=DETECTION_OUTPUT_DIR,
            debug_output_dir=dbg_dir,
            crossing_image_folder_prefix=crossing_prefix,
            crossing_state=crossing_state,
        )
        last_frame_data = curr_frame_data
        frame_history.append(curr_frame_data)
        frame_history = frame_history[-REID_HISTORY_FRAMES:]

        total_entries += entries
        total_exits += exits

        for event_type, frame_prev, frame_curr in event_list:
            if event_type == "ENTRY":
                occupancy += 1
            elif event_type == "EXIT":
                occupancy = max(0, occupancy - 1)
            append_event_to_csv(
                event_type,
                frame_prev,
                frame_curr,
                occupancy,
                total_entries,
                total_exits,
                counts_csv_file=COUNTS_CSV_FILE,
            )

        time.sleep(SLEEP_BETWEEN_PAIRS)

    print("\n" + "=" * 60)
    print(f"Sequence summary: {capture_dir}")
    print("=" * 60)
    print(f"Total ENTRY: {total_entries}")
    print(f"Total EXIT: {total_exits}")
    print(f"Final Total entries (people inside): {occupancy}")
    print(f"Processed frame pairs (step={step})")
    print(f"Events appended to: {COUNTS_CSV_FILE}")
    print("=" * 60)

    return (total_entries, total_exits, occupancy)


def main() -> None:
    """
    Run entry/exit detection over captured frames and persist state in CSV.
    """
    print("Loading configurations...")
    try:
        configs = load_configs()
        line_config = configs["line"]
        line_config_b = configs["line_b"]
        line_y = line_config["y_avg"]
        print(f"Line A Y_avg (label): {line_y:.1f} | Line B: mask/line_config4.json")
        print(
            "Two-line gate: EXIT = cross line A then B (in/out half-planes); "
            "ENTRY = cross B then A; shared inside reference from line A midpoint + Y offset; "
            f"hysteresis: {CROSSING_HYSTERESIS_FRAMES} frame-pair(s) per line."
        )
    except Exception as exc:
        print(f"Error loading configs: {exc}")
        return

    runs = _resolve_runs()
    if not runs:
        if CAPTURED_FRAMES_DIR:
            print(
                f"Error: P_CHECK_FRAMES_DIR={CAPTURED_FRAMES_DIR!r} is missing, not a directory, "
                "or has fewer than 2 images."
            )
        elif not os.path.isdir(FRAMES_ROOT):
            print(f"Error: frames root not found: {FRAMES_ROOT}")
        else:
            print(
                f"Error: No sequence to run under {FRAMES_ROOT!r}. "
                "Add a subfolder with at least 2 images, or place >= 2 images directly in "
                f"{FRAMES_ROOT!r}, or set P_CHECK_FRAMES_DIR to a single capture folder."
            )
        return

    if CAPTURED_FRAMES_DIR:
        print(f"Single-folder mode (P_CHECK_FRAMES_DIR): {CAPTURED_FRAMES_DIR}")
    else:
        labels = [slug for _, slug in runs]
        print(f"Multi-folder mode: {len(runs)} sequence(s) — {', '.join(labels)}")

    os.makedirs(DETECTION_OUTPUT_DIR, exist_ok=True)
    ensure_counts_csv(COUNTS_CSV_FILE)
    print(f"Universal counts CSV: {COUNTS_CSV_FILE}")

    session = load_model()
    roi = load_mask_roi()
    if roi is None:
        print("Error: valid ROI is required in mask/mask_config.json")
        return

    grand_entries = 0
    grand_exits = 0
    single_run = len(runs) == 1

    for capture_dir, output_slug in runs:
        te, tx, _ = run_sequence(
            capture_dir,
            output_slug,
            session,
            roi,
            line_y,
            line_config,
            line_config_b,
            resume_state_from_csv=single_run,
        )
        grand_entries += te
        grand_exits += tx

    if len(runs) > 1:
        print("\n" + "=" * 60)
        print("All sequences — combined ENTRY/EXIT counts (not occupancy)")
        print("=" * 60)
        print(f"Combined Total ENTRY: {grand_entries}")
        print(f"Combined Total EXIT: {grand_exits}")
        print("=" * 60)


if __name__ == "__main__":
    main()
