"""
Step 004 - CSV-based persistence of entry/exit state.

Public API:
- ensure_counts_csv
- read_state_from_csv
- append_event_to_csv
"""

import csv
import os
from datetime import datetime
from typing import Tuple

from config import COUNTS_CSV_FILE, COUNTS_CSV_HEADER, DETECTION_OUTPUT_DIR


def ensure_counts_csv() -> None:
    """
    Ensure the counts CSV exists with the correct header.

    Creates DETECTION_OUTPUT_DIR and the file if they do not exist.
    """
    os.makedirs(DETECTION_OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(COUNTS_CSV_FILE):
        with open(COUNTS_CSV_FILE, "w", newline="") as f:
            csv.writer(f).writerow(COUNTS_CSV_HEADER)


def read_state_from_csv() -> Tuple[int, int, int]:
    """
    Read the last known state from counts.csv.

    Returns:
        (total_inside, cumulative_entered, cumulative_exit)
        or (0, 0, 0) if the file is missing or empty.
    """
    if not os.path.exists(COUNTS_CSV_FILE):
        return (0, 0, 0)

    with open(COUNTS_CSV_FILE, "r", newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) < 2:
        return (0, 0, 0)

    last_row = rows[-1]
    if len(last_row) < len(COUNTS_CSV_HEADER):
        return (0, 0, 0)

    try:
        total_inside = int(last_row[4])
        cumulative_entered = int(last_row[5])
        cumulative_exit = int(last_row[6])
        return (total_inside, cumulative_entered, cumulative_exit)
    except (ValueError, IndexError):
        return (0, 0, 0)


def append_event_to_csv(
    event_type: str,
    frame_prev: str,
    frame_curr: str,
    total_inside: int,
    cumulative_entered: int,
    cumulative_exit: int,
) -> None:
    """
    Append a single ENTRY/EXIT event row to counts.csv.

    The row contains a timestamp, event type, frame names, and state values.
    """
    os.makedirs(DETECTION_OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [
        timestamp,
        event_type,
        os.path.basename(frame_prev),
        os.path.basename(frame_curr),
        total_inside,
        cumulative_entered,
        cumulative_exit,
    ]

    with open(COUNTS_CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)

