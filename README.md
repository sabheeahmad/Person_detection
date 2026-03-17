# P_check — Jetson Nano Entry/Exit Detection

Entry/exit detection using YOLOv8 and line crossing, tuned for **NVIDIA Jetson Nano**. The script loads a YOLOv8 ONNX model, processes frame pairs from `output/`, detects persons, matches them across frames, and records ENTRY/EXIT events when centroids cross a configured line. Counts are written to CSV; optional debug and crossing images are saved.

---

## Jetson Nano Specs

| Component | Spec |
|-----------|------|
| **GPU** | NVIDIA Maxwell, 128 CUDA cores @ 921 MHz, 512 GFLOPS (FP16) |
| **RAM** | 4 GB LPDDR4 (shared CPU/GPU) |
| **CPU** | Quad-core ARM Cortex-A57 @ 1.43 GHz |
| **Storage** | 16 GB eMMC 5.1 |
| **Power** | ~5–10 W @ 5 V |
| **Connectivity** | Gigabit Ethernet, HDMI 2.0, USB 3.0/2.0 |

---

## Prerequisites

- **JetPack** 4.6+ or 5.x (L4T, CUDA, cuDNN, OpenCV from JetPack).
- **Python** 3.6–3.10.
- **OpenCV**: Use JetPack’s `cv2`; avoid `pip install opencv-python` on Jetson unless advised.
- **NumPy**: `pip install numpy`.
- **ONNX Runtime**: Prefer `onnxruntime` (CPU) for reliability. PyPI `onnxruntime-gpu` is x86-oriented and often unusable on ARM; use Jetson-built ORT or TensorRT EP if you need GPU inference.

---

## Setup

**Project layout:** Ensure `mask/` (with `line_config2.json`), `output/` (captured frames), and `yolov8n.onnx` exist.

**Line config:** `mask/line_config2.json` must contain `x1`, `y1`, `x2`, `y2` (line endpoints). Example:

```json
{
    "x1": 245,
    "y1": 192,
    "x2": 384,
    "y2": 192
}
```

**Model:** Use `yolov8n.onnx` (640×640). If exporting elsewhere:

```bash
pip install ultralytics
python -c "from ultralytics import YOLO; m=YOLO('yolov8n.pt'); m.export(format='onnx', imgsz=640)"
```

**Install dependencies (Jetson):**

```bash
pip install numpy onnxruntime
```

Use JetPack’s OpenCV; do not install `opencv-python` unless needed. If you have a Jetson-built `onnxruntime-gpu` or TensorRT, the script will use GPU when available.

---

## Directory Structure

| Path | Description |
|------|-------------|
| `mask/` | Line config; `line_config2.json` required |
| `output/` | Input frames (`.jpg`, `.jpeg`, `.png`) |
| `yolov8n.onnx` | YOLOv8 nano ONNX model (640×640) |
| `detection_output2/` | Output; `counts.csv` + crossing images (created by script) |
| `debug_output/` | Debug side‑by‑side images (created by script) |

---

## Jetson-Compatible Script

Save the script below as `step2_jetson_detection.py` and run it from the project root (see [How to run](#how-to-run)).

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_INPUT_SIZE` | `640` | Model input size (match ONNX export) |
| `FRAME_STEP` | `2` | Process every Nth frame pair |
| `SLEEP_BETWEEN_PAIRS` | `2.0` | Seconds between frame pairs |
| `ONNX_NUM_THREADS` | `2` | ONNX Runtime intra-op threads |

```python
"""
Jetson-compatible entry/exit detection. GPU when available, else CPU.
FRAME_STEP=2, SLEEP_BETWEEN_PAIRS=2.0 for lower load on Jetson Nano.
"""
import os
import json
import math
import csv
import time
from datetime import datetime

import cv2
import numpy as np
import onnxruntime as ort


MASK_DIR = "mask"
CAPTURED_FRAMES_DIR = "output"
DETECTION_OUTPUT_DIR = "detection_output2"
COUNTS_CSV_FILE = os.path.join(DETECTION_OUTPUT_DIR, "counts.csv")
DEBUG_OUTPUT_DIR = "debug_output"
LINE_CONFIG_FILE = os.path.join(MASK_DIR, "line_config2.json")
MODEL_NAME = "yolov8n.onnx"
CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
INPUT_SIZE = int(os.getenv("YOLO_INPUT_SIZE", "640"))
ONNX_NUM_THREADS = int(os.getenv("ONNX_NUM_THREADS", "2"))
CENTROID_POSITION_RATIO = 0.9
MATCHING_THRESHOLD = 350
FRAME_STEP = int(os.getenv("FRAME_STEP", "2"))
SLEEP_BETWEEN_PAIRS = float(os.getenv("SLEEP_BETWEEN_PAIRS", "2.0"))
SAVE_DEBUG_IMAGES = True
SAVE_CROSSING_IMAGES = True
DEBUG_IMAGE_EVERY_N = 1


def _get_providers():
    """Prefer GPU when available, else CPU (Jetson-friendly)."""
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_model():
    """Load YOLOv8 ONNX model (GPU if available, else CPU)."""
    if not os.path.exists(MODEL_NAME):
        raise FileNotFoundError(
            f"Model file {MODEL_NAME} not found. Ensure it exists in the project root."
        )
    providers = _get_providers()
    print(f"Loading YOLO model: {MODEL_NAME} (providers: {providers})")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = ONNX_NUM_THREADS
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        MODEL_NAME,
        sess_options=sess_options,
        providers=providers,
    )
    print(
        "Model loaded successfully. "
        f"provider={session.get_providers()[0]}, "
        f"INPUT_SIZE={INPUT_SIZE}, FRAME_STEP={FRAME_STEP}, SLEEP={SLEEP_BETWEEN_PAIRS}s"
    )
    return session


def _letterbox_params(img, input_size):
    original_height, original_width = img.shape[:2]
    scale = min(input_size / original_width, input_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    pad_left = (input_size - new_width) // 2
    pad_top = (input_size - new_height) // 2
    return scale, pad_left, pad_top


def preprocess_image(img, input_size):
    """Preprocess image for YOLOv8 ONNX (letterbox + normalize)."""
    original_height, original_width = img.shape[:2]
    scale, pad_left, pad_top = _letterbox_params(img, input_size)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = resized
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    batched = np.expand_dims(chw, axis=0)
    return batched


def detect_persons(session, image_path):
    """Run YOLO on a frame; return (img, boxes) with boxes as (x1,y1,x2,y2)."""
    img = cv2.imread(image_path)
    if img is None:
        return None, []

    preprocessed = preprocess_image(img, INPUT_SIZE)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: preprocessed})
    output = outputs[0]

    if len(output.shape) == 3:
        output = output[0]

    scale, pad_left, pad_top = _letterbox_params(img, INPUT_SIZE)
    boxes_xywh = output[0:4, :].T
    scores = output[4:, :].transpose()
    person_scores = scores[:, 0]

    h_img, w_img = img.shape[:2]
    candidates_xyxy = []
    candidate_scores = []
    for i in range(len(person_scores)):
        if person_scores[i] <= CONFIDENCE_THRESHOLD:
            continue
        cx, cy, w, h = boxes_xywh[i]
        orig_cx = (cx - pad_left) / scale
        orig_cy = (cy - pad_top) / scale
        orig_w = w / scale
        orig_h = h / scale
        x1 = int(orig_cx - orig_w / 2)
        y1 = int(orig_cy - orig_h / 2)
        x2 = int(orig_cx + orig_w / 2)
        y2 = int(orig_cy + orig_h / 2)
        x1 = max(0, min(x1, w_img))
        y1 = max(0, min(y1, h_img))
        x2 = max(0, min(x2, w_img))
        y2 = max(0, min(y2, h_img))
        candidates_xyxy.append((x1, y1, x2, y2))
        candidate_scores.append(float(person_scores[i]))

    if not candidates_xyxy:
        return img, []

    boxes_xywh_for_nms = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in candidates_xyxy]
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
    return img, person_boxes


def calculate_centroid(bounding_box):
    x1, y1, x2, y2 = bounding_box
    centroid_x = int((x1 + x2) / 2)
    centroid_y = int(y1 + CENTROID_POSITION_RATIO * (y2 - y1))
    return (centroid_x, centroid_y)


def load_configs():
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


def match_persons(prev_centroids, curr_centroids):
    if len(prev_centroids) == 0 or len(curr_centroids) == 0:
        return []
    all_pairs = []
    for i, prev_cent in enumerate(prev_centroids):
        for j, curr_cent in enumerate(curr_centroids):
            d = math.sqrt((curr_cent[0] - prev_cent[0]) ** 2 + (curr_cent[1] - prev_cent[1]) ** 2)
            if d < MATCHING_THRESHOLD:
                all_pairs.append((d, i, j))
    all_pairs.sort(key=lambda x: x[0])
    matches = []
    used_prev, used_curr = set(), set()
    for d, pi, ci in all_pairs:
        if pi not in used_prev and ci not in used_curr:
            matches.append((pi, ci, d))
            used_prev.add(pi)
            used_curr.add(ci)
    return matches


def detect_crossing(prev_centroid_y, curr_centroid_y, line_y):
    prev_side = "outside" if prev_centroid_y < line_y else "inside"
    curr_side = "outside" if curr_centroid_y < line_y else "inside"
    if prev_side != curr_side:
        if prev_side == "outside" and curr_side == "inside":
            return "ENTRY"
        if prev_side == "inside" and curr_side == "outside":
            return "EXIT"
    return None


def initialize_csv():
    os.makedirs(DETECTION_OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(COUNTS_CSV_FILE):
        with open(COUNTS_CSV_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "type", "frame_prev", "frame_curr"])


def save_event_to_csv(event_type, frame_prev, frame_curr):
    os.makedirs(DETECTION_OUTPUT_DIR, exist_ok=True)
    with open(COUNTS_CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            event_type,
            os.path.basename(frame_prev),
            os.path.basename(frame_curr),
        ])


def draw_debug_annotations(
    image_path, boxes, centroids, line_config, line_y, highlight_person_index=None, img=None
):
    if img is None:
        img = cv2.imread(image_path)
    if img is None:
        return None
    img_copy = img.copy()
    cv2.line(
        img_copy,
        (line_config["x1"], line_config["y1"]),
        (line_config["x2"], line_config["y2"]),
        (0, 0, 255), 3,
    )
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        color = (0, 255, 255) if highlight_person_index == idx else (0, 255, 0)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 4 if highlight_person_index == idx else 2)
    for cx, cy in centroids:
        cv2.circle(img_copy, (cx, cy), 8, (0, 0, 255), -1)
    cv2.putText(img_copy, f"Persons: {len(boxes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img_copy, f"Line Y: {line_y:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return img_copy


def create_debug_image_pair(
    prev_image_path, curr_image_path,
    prev_boxes, curr_boxes, prev_centroids, curr_centroids,
    line_config, line_y, prev_img=None, curr_img=None,
):
    prev_a = draw_debug_annotations(prev_image_path, prev_boxes, prev_centroids, line_config, line_y, None, prev_img)
    curr_a = draw_debug_annotations(curr_image_path, curr_boxes, curr_centroids, line_config, line_y, None, curr_img)
    if prev_a is None or curr_a is None:
        return None
    h1, w1 = prev_a.shape[:2]
    h2, w2 = curr_a.shape[:2]
    mh = max(h1, h2)
    tw = w1 + w2 + 20
    out = np.zeros((mh, tw, 3), dtype=np.uint8)
    if h1 != mh:
        prev_a = cv2.resize(prev_a, (w1, mh))
    if h2 != mh:
        curr_a = cv2.resize(curr_a, (w2, mh))
    out[:, :w1] = prev_a
    out[:, w1 + 20:] = curr_a
    cv2.line(out, (w1 + 10, 0), (w1 + 10, mh), (255, 255, 255), 2)
    return out


def create_crossing_image_pair(
    prev_image_path, curr_image_path,
    prev_boxes, curr_boxes, prev_centroids, curr_centroids,
    line_config, line_y, event_type, prev_hi, curr_hi,
    prev_img=None, curr_img=None,
):
    prev_a = draw_debug_annotations(prev_image_path, prev_boxes, prev_centroids, line_config, line_y, prev_hi, prev_img)
    curr_a = draw_debug_annotations(curr_image_path, curr_boxes, curr_centroids, line_config, line_y, curr_hi, curr_img)
    if prev_a is None or curr_a is None:
        return None
    h1, w1 = prev_a.shape[:2]
    h2, w2 = curr_a.shape[:2]
    mh = max(h1, h2)
    tw = w1 + w2 + 20
    out = np.zeros((mh, tw, 3), dtype=np.uint8)
    if h1 != mh:
        prev_a = cv2.resize(prev_a, (w1, mh))
    if h2 != mh:
        curr_a = cv2.resize(curr_a, (w2, mh))
    out[:, :w1] = prev_a
    out[:, w1 + 20:] = curr_a
    cv2.line(out, (w1 + 10, 0), (w1 + 10, mh), (255, 255, 255), 2)
    if event_type == "ENTRY":
        txt, col = "OUTSIDE → INSIDE", (0, 255, 0)
    else:
        txt, col = "INSIDE → OUTSIDE", (0, 0, 255)
    ts = cv2.getTextSize(f"{event_type} DETECTED", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    cv2.putText(out, f"{event_type} DETECTED", (max(0, (tw - ts[0]) // 2), 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, col, 3)
    cv2.putText(out, txt, (max(0, (tw - 180) // 2), 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.arrowedLine(out, (w1 - 20, mh // 2), (w1 + 40, mh // 2), (255, 255, 0), 3, tipLength=0.3)
    return out


def process_frame_pair(prev_path, curr_path, line_y, line_config, session, frame_pair_index=None, verbose=True):
    if verbose:
        print(f"\nProcessing: {os.path.basename(prev_path)} → {os.path.basename(curr_path)}")
    prev_img, prev_boxes = detect_persons(session, prev_path)
    curr_img, curr_boxes = detect_persons(session, curr_path)
    if verbose:
        print(f"  Previous: {len(prev_boxes)} person(s), Current: {len(curr_boxes)} person(s)")
    if len(prev_boxes) == 0 and len(curr_boxes) == 0:
        return (0, 0)
    prev_centroids = [calculate_centroid(b) for b in prev_boxes]
    curr_centroids = [calculate_centroid(b) for b in curr_boxes]
    if len(prev_centroids) == 0 or len(curr_centroids) == 0:
        return (0, 0)

    save_debug = SAVE_DEBUG_IMAGES
    if save_debug and DEBUG_IMAGE_EVERY_N > 1 and frame_pair_index is not None:
        save_debug = (frame_pair_index % DEBUG_IMAGE_EVERY_N == 0)
    if save_debug:
        debug_img = create_debug_image_pair(
            prev_path, curr_path, prev_boxes, curr_boxes,
            prev_centroids, curr_centroids, line_config, line_y,
            prev_img=prev_img, curr_img=curr_img,
        )
        if debug_img is not None:
            os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
            name = f"debug_{os.path.splitext(os.path.basename(prev_path))[0]}_{os.path.splitext(os.path.basename(curr_path))[0]}.jpg"
            cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, name), debug_img)

    matches = match_persons(prev_centroids, curr_centroids)
    entries = exits = 0
    for pi, ci, _ in matches:
        prev_c, curr_c = prev_centroids[pi], curr_centroids[ci]
        crossing = detect_crossing(prev_c[1], curr_c[1], line_y)
        if not crossing:
            continue
        if verbose:
            print(f"    {crossing} detected (prev[{pi}] -> curr[{ci}])")
        if SAVE_CROSSING_IMAGES:
            cross_img = create_crossing_image_pair(
                prev_path, curr_path, prev_boxes, curr_boxes,
                prev_centroids, curr_centroids, line_config, line_y,
                crossing, pi, ci, prev_img=prev_img, curr_img=curr_img,
            )
            if cross_img is not None:
                os.makedirs(DETECTION_OUTPUT_DIR, exist_ok=True)
                name = f"crossing_{crossing}_{os.path.splitext(os.path.basename(prev_path))[0]}_{os.path.splitext(os.path.basename(curr_path))[0]}.jpg"
                cv2.imwrite(os.path.join(DETECTION_OUTPUT_DIR, name), cross_img)
        if crossing == "ENTRY":
            entries += 1
            save_event_to_csv("ENTRY", prev_path, curr_path)
        elif crossing == "EXIT":
            exits += 1
            save_event_to_csv("EXIT", prev_path, curr_path)
    return (entries, exits)


def main():
    os.makedirs(DETECTION_OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
    initialize_csv()
    print(f"Detection output: {DETECTION_OUTPUT_DIR}, CSV: {COUNTS_CSV_FILE}")

    try:
        configs = load_configs()
        line_config = configs["line"]
        line_y = line_config["y_avg"]
        print(f"Line Y: {line_y:.1f} (above=inside, below=outside)")
    except Exception as e:
        print(f"Error loading configs: {e}")
        return

    if not os.path.exists(CAPTURED_FRAMES_DIR):
        print(f"Error: {CAPTURED_FRAMES_DIR} not found.")
        return
    exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    frame_files = sorted([f for f in os.listdir(CAPTURED_FRAMES_DIR) if any(f.endswith(e) for e in exts)])
    if len(frame_files) < 2:
        print(f"Error: Need >= 2 frames. Found {len(frame_files)}.")
        return
    frame_paths = [os.path.join(CAPTURED_FRAMES_DIR, f) for f in frame_files]

    print(f"\nFound {len(frame_files)} frame(s). Jetson mode: FRAME_STEP={FRAME_STEP}, SLEEP={SLEEP_BETWEEN_PAIRS}s")
    print("=" * 60)
    session = load_model()
    total_entries = total_exits = 0
    step = max(1, FRAME_STEP)

    for i in range(0, len(frame_paths) - 1, step):
        curr_i = min(i + step, len(frame_paths) - 1)
        e, x = process_frame_pair(
            frame_paths[i], frame_paths[curr_i],
            line_y, line_config, session, frame_pair_index=i, verbose=True,
        )
        total_entries += e
        total_exits += x
        time.sleep(SLEEP_BETWEEN_PAIRS)

    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)
    print(f"Total ENTRY: {total_entries}, Total EXIT: {total_exits}")
    print(f"Events saved to: {COUNTS_CSV_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## How to run

Ensure `output/`, `mask/line_config2.json`, and `yolov8n.onnx` exist, then:

```bash
python step2_jetson_detection.py
```

Optional env overrides (e.g. heavier throttling):

```bash
FRAME_STEP=4 SLEEP_BETWEEN_PAIRS=3 python step2_jetson_detection.py
```

---

## Troubleshooting

- **OOM / process killed:** Reduce memory use: set `SAVE_DEBUG_IMAGES = False` or increase `DEBUG_IMAGE_EVERY_N`, increase `FRAME_STEP` / `SLEEP_BETWEEN_PAIRS`, or use a smaller `YOLO_INPUT_SIZE` (e.g. 416) with a re-exported ONNX.
- **`CUDAExecutionProvider` not available:** Normal on many Jetson setups with stock pip. The script falls back to CPU. For GPU inference, use a Jetson-built ONNX Runtime or TensorRT; see NVIDIA Jetson docs.
- **Monitor memory:** Run `tegrastats` (or similar) while the script runs to watch RAM/GPU usage.

---

## References

- **step2_detection.py** — Development version with CPU fallback and ~1 frame-pair/sec.
- **step2.0_detection.py** — GPU-only, lower-usage variant (SLEEP, FRAME_STEP).
- **requirements.txt** — Dependencies for non-Jetson (e.g. x86) development.

This README is the single Jetson-specific artifact; all runnable code is in the script block above.
