"""
Shared configuration for step 2.1 detection with CSV-backed state.

All other modules should import constants from here instead of hardcoding
paths or thresholds.
"""

import os

# --- Paths ---

MASK_DIR = "mask"
MODEL_DIR = "model"
# Root folder: `main.py` scans immediate subfolders (each with images) one by one.
# If there are no qualifying subfolders but this root itself has >= 2 images, that root is processed once.
FRAMES_ROOT = "frames"
# Optional single-folder override (e.g. env P_CHECK_FRAMES_DIR=frames/test1) skips multi-folder scan.
CAPTURED_FRAMES_DIR = os.getenv("P_CHECK_FRAMES_DIR", "").strip() or None
DETECTION_OUTPUT_DIR = "detection_output"
DEBUG_OUTPUT_DIR = "debug_output"

COUNTS_CSV_FILE = os.path.join(DETECTION_OUTPUT_DIR, "counts.csv")
COUNTS_CSV_HEADER = [
    "timestamp",
    "type",
    "frame_prev",
    "frame_curr",
    "Total entries",
    "Entered",
    "Exit",
]

# Line A / line B: two-line gate (EXIT = A then B; ENTRY = B then A in half-plane order).
LINE_CONFIG_FILE = os.path.join(MASK_DIR, "line_config4.json")
LINE_B_CONFIG_FILE = os.path.join(MASK_DIR, "line_config3.json")
MASK_CONFIG_FILE = os.path.join(MASK_DIR, "frames.json")

# --- Model / inference ---

#False Positive (FP): The result says "Yes," but the reality is "No."

MODEL_NAME = os.path.join(MODEL_DIR, "yolov8n_4k.onnx")
CONFIDENCE_THRESHOLD = 0.20 #Lower = more detections (more false positives); higher = fewer, stricter.
NMS_IOU_THRESHOLD = 0.45 #Lower = more overlap (more false positives); higher = fewer, stricter.
INPUT_SIZE = int(os.getenv("YOLO_INPUT_SIZE", "1024")) #Lower = less memory usage; higher = more memory usage.
ONNX_NUM_THREADS = int(os.getenv("ONNX_NUM_THREADS", "1")) #Lower = less CPU usage; higher = more CPU usage.

# --- Tracking / matching ---

CENTROID_POSITION_RATIO = 0.9
MATCHING_THRESHOLD = 80#Lower = more matching (more false positives); higher = fewer, stricter.
FRAME_STEP = int(os.getenv("FRAME_STEP", "1"))#Process every 2nd frame (e.g. pairs 0–2, 2–4, 4–6).

# --- Re-identification (re-ID): reuse IDs when a person was missed for a few frames ---
# How many past frames to look at when an "unmatched" person might be someone we saw earlier.
REID_HISTORY_FRAMES = 5
# Max distance (px) between current centroid and a person in history to reuse that ID. Can be slightly larger than MATCHING_THRESHOLD since the person may have moved over several frames.
REID_DISTANCE_THRESHOLD = 100

# --- Line crossing (2D half-plane + hysteresis) ---
# Reference point for "inside" = line midpoint shifted down in image Y by this many pixels
# (matches legacy convention: inside = larger y / below the average line height).
LINE_INSIDE_REFERENCE_OFFSET_Y = float(os.getenv("LINE_INSIDE_REFERENCE_OFFSET_Y", "80"))
# Require this many consecutive frame-pair observations on the new side before ENTRY/EXIT.
CROSSING_HYSTERESIS_FRAMES = max(1, int(os.getenv("CROSSING_HYSTERESIS_FRAMES", "3")))
# Treat |cross product| below this as on the line (ambiguous); fall back to y vs y_avg.
LINE_ON_CROSS_EPS = float(os.getenv("LINE_ON_CROSS_EPS", "50"))

# --- Debug / output flags ---

SAVE_DEBUG_IMAGES = False
SAVE_CROSSING_IMAGES = True
DEBUG_IMAGE_EVERY_N = 1

# --- Runtime pacing ---

SLEEP_BETWEEN_PAIRS = float(os.getenv("SLEEP_BETWEEN_PAIRS", "0"))

