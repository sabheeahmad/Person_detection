"""
Step 001 - model and ROI utilities.

Responsibilities:
- Load the YOLOv8 ONNX model on GPU.
- Read the rectangular ROI from mask configuration.
- Preprocess images for YOLO (letterbox + normalization).
"""

import json
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from config import (
    INPUT_SIZE,
    MASK_CONFIG_FILE,
    MODEL_NAME,
    ONNX_NUM_THREADS,
)


def _get_providers() -> list[str]:
    """Return ONNX Runtime providers, enforcing CUDA-only inference."""
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider not available. This script runs on GPU only "
            "(no CPU). Install onnxruntime-gpu and ensure CUDA is available."
        )
    return ["CUDAExecutionProvider"]


def load_model() -> ort.InferenceSession:
    """
    Load YOLOv8 ONNX model on GPU only (no CPU fallback).

    Returns:
        An initialized onnxruntime InferenceSession.
    """
    if not os.path.exists(MODEL_NAME):
        raise FileNotFoundError(
            f"Model file {MODEL_NAME} not found. It should be downloaded during Docker build."
        )

    providers = _get_providers()
    print(f"Loading YOLO model: {MODEL_NAME} (GPU only, providers: {providers})")

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
        f"INPUT_SIZE={INPUT_SIZE}"
    )
    return session


def load_mask_roi(config_path: str = MASK_CONFIG_FILE) -> Optional[Tuple[int, int, int, int]]:
    """
    Load rectangular ROI from the mask configuration file.

    Returns:
        (x1, y1, x2, y2) in image coordinates, or None on failure.
    """
    if not os.path.exists(config_path):
        print(f"Error: Mask config not found: {config_path}")
        return None

    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        x = int(cfg["x"])
        y = int(cfg["y"])
        width = int(cfg["width"])
        height = int(cfg["height"])
    except Exception as exc:
        print(f"Error: Failed to read mask config '{config_path}': {exc}")
        return None

    if width <= 0 or height <= 0:
        print(f"Error: Invalid mask size in '{config_path}': width={width}, height={height}")
        return None

    x1, y1 = x, y
    x2, y2 = x + width, y + height
    print(f"Using ROI from mask_config: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")
    return (x1, y1, x2, y2)


def _letterbox_params(img, input_size: int) -> tuple[float, int, int]:
    """
    Compute letterbox parameters used to fit an image into a square canvas.

    Returns:
        (scale, pad_left, pad_top)
    """
    original_height, original_width = img.shape[:2]
    scale = min(input_size / original_width, input_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    pad_left = (input_size - new_width) // 2
    pad_top = (input_size - new_height) // 2
    return scale, pad_left, pad_top


def preprocess_image(img, input_size: int = INPUT_SIZE) -> np.ndarray:
    """
    Preprocess BGR image for YOLOv8 ONNX model.

    Steps:
        - Letterbox resize into a square canvas.
        - Convert BGR to RGB.
        - Normalize pixel values to [0, 1].
        - Reorder to NCHW tensor.
    """
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

