"""
Step 2.1 Detection entrypoint.

Coordinates:
- model loading (GPU-only),
- ROI and line configuration,
- per-frame-pair processing with crossing detection,
- CSV-backed state for total entries / exits.
"""

import os
import time

from config import (
    CAPTURED_FRAMES_DIR,
    COUNTS_CSV_FILE,
    DEBUG_OUTPUT_DIR,
    DETECTION_OUTPUT_DIR,
    FRAME_STEP,
    SLEEP_BETWEEN_PAIRS,
)
from step001_model_io import load_mask_roi, load_model
from step006_processing import process_frame_pair
from step004_state_io import append_event_to_csv, ensure_counts_csv, read_state_from_csv
from step003_tracking import load_configs


def main() -> None:
    """
    Run entry/exit detection over captured frames and persist state in CSV.
    """
    os.makedirs(DETECTION_OUTPUT_DIR, exist_ok=True)
    print(f"Detection output directory: {DETECTION_OUTPUT_DIR}")

    ensure_counts_csv()
    print(f"State CSV: {COUNTS_CSV_FILE} (columns: Total entries, Entered, Exit)")

    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
    print(f"Debug output directory: {DEBUG_OUTPUT_DIR}")

    print("Loading configurations...")
    try:
        configs = load_configs()
        line_config = configs["line"]
        line_y = line_config["y_avg"]
        print(f"Line Y position: {line_y:.1f}")
        print("(Above line = inside store, Below line = outside store)")
    except Exception as exc:
        print(f"Error loading configs: {exc}")
        return

    if not os.path.exists(CAPTURED_FRAMES_DIR):
        print(f"Error: Captured frames directory not found: {CAPTURED_FRAMES_DIR}")
        return

    image_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    frame_files = [
        f
        for f in os.listdir(CAPTURED_FRAMES_DIR)
        if any(f.endswith(ext) for ext in image_extensions)
    ]
    if len(frame_files) < 2:
        print(
            "Error: Need at least 2 frames to detect crossing. "
            f"Found {len(frame_files)} frame(s)."
        )
        return

    frame_files.sort()
    frame_paths = [os.path.join(CAPTURED_FRAMES_DIR, f) for f in frame_files]

    print(f"\nFound {len(frame_files)} frame(s) to process")
    print("=" * 60)
    print("Step 2.1 - Entry/Exit Detection (state in CSV)")
    print("=" * 60)

    session = load_model()

    roi = load_mask_roi()
    if roi is None:
        print("Error: valid ROI is required in mask/mask_config.json")
        return

    # Resume state from CSV if present.
    total_inside, cumulative_entered, cumulative_exit = read_state_from_csv()
    occupancy = total_inside
    total_entries = cumulative_entered
    total_exits = cumulative_exit
    step = max(1, FRAME_STEP)

    for i in range(0, len(frame_paths) - 1, step):
        prev_image = frame_paths[i]
        curr_index = min(i + step, len(frame_paths) - 1)
        curr_image = frame_paths[curr_index]

        entries, exits, event_list = process_frame_pair(
            prev_image,
            curr_image,
            line_y,
            line_config,
            session,
            roi,
            frame_pair_index=i,
            verbose=True,
        )

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
            )

        time.sleep(SLEEP_BETWEEN_PAIRS)

    print("\\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)
    print(f"Total ENTRY: {total_entries}")
    print(f"Total EXIT: {total_exits}")
    print(f"Final Total entries (people inside): {occupancy}")
    print(f"Processed frame pairs (step={step})")
    print(
        f"State maintained in: {COUNTS_CSV_FILE} (Total entries, Entered, Exit)"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()