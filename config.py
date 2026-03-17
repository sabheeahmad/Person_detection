"""
Shared configuration for step 2.1 detection with CSV-backed state.

All other modules should import constants from here instead of hardcoding
paths or thresholds.
"""

import os

# --- Paths ---

MASK_DIR = "mask"
CAPTURED_FRAMES_DIR = "test/test3"
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

LINE_CONFIG_FILE = os.path.join(MASK_DIR, "line_config2.json")
MASK_CONFIG_FILE = os.path.join(MASK_DIR, "mask_config.json")

# --- Model / inference ---

#False Positive (FP): The result says "Yes," but the reality is "No."

MODEL_NAME = "yolov8n.onnx"
CONFIDENCE_THRESHOLD = 0.25 #Lower = more detections (more false positives); higher = fewer, stricter.
NMS_IOU_THRESHOLD = 0.45 #Lower = more overlap (more false positives); higher = fewer, stricter.
INPUT_SIZE = int(os.getenv("YOLO_INPUT_SIZE", "640")) #Lower = less memory usage; higher = more memory usage.
ONNX_NUM_THREADS = int(os.getenv("ONNX_NUM_THREADS", "1")) #Lower = less CPU usage; higher = more CPU usage.

# --- Tracking / matching ---

CENTROID_POSITION_RATIO = 0.9
MATCHING_THRESHOLD = 80#Lower = more matching (more false positives); higher = fewer, stricter.
FRAME_STEP = int(os.getenv("FRAME_STEP", "2"))#Process every 2nd frame (e.g. pairs 0–2, 2–4, 4–6).

# --- Debug / output flags ---

SAVE_DEBUG_IMAGES = True
SAVE_CROSSING_IMAGES = True
DEBUG_IMAGE_EVERY_N = 1

# --- Runtime pacing ---

SLEEP_BETWEEN_PAIRS = float(os.getenv("SLEEP_BETWEEN_PAIRS", "2.0"))

