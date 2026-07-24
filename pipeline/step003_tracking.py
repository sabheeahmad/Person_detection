"""
Step 003 - geometric helpers, line configuration, crossing logic, and giving IDs.

Public API:
- calculate_centroid
- load_configs
- match_persons
- assign_ids_by_centroid_matching  (giving IDs: persistent person IDs via Euclidean distance)
- build_line_crossing_context_shared, point_is_inside, two_line_gate_event
"""

import json
import math
import os
from typing import Dict, List, Literal, Optional, Sequence, Tuple, TypedDict

from config import (
    CENTROID_POSITION_RATIO,
    CROSSING_HYSTERESIS_FRAMES,
    LINE_B_CONFIG_FILE,
    LINE_CONFIG_FILE,
    LINE_INSIDE_REFERENCE_OFFSET_Y,
    LINE_ON_CROSS_EPS,
    MATCHING_THRESHOLD,
)


Point = Tuple[int, int]


class LineCrossingContext(TypedDict):
    """Precomputed values for 2D line side tests (see build_line_crossing_context)."""

    x1: float
    y1: float
    x2: float
    y2: float
    y_avg: float
    cross_ref: float
    on_line_eps: float


class PersonCrossingState(TypedDict):
    """Per-person hysteresis for confirmed inside/outside transitions (one line)."""

    stable_inside: bool
    pending_inside: Optional[bool]
    pending_count: int


GatePhase = Literal[
    "idle_inside",
    "idle_outside",
    "exit_after_a",
    "entry_after_b",
    "corridor_ambiguous",
]


class PersonTwoLineState(TypedDict):
    """Per-person two-line gate: hysteresis per line + FSM phase."""

    line_a: PersonCrossingState
    line_b: PersonCrossingState
    gate_phase: GatePhase


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


def _normalize_line_dict(raw: dict) -> dict:
    y_avg = (raw["y1"] + raw["y2"]) / 2
    return {
        "x1": raw["x1"],
        "y1": raw["y1"],
        "x2": raw["x2"],
        "y2": raw["y2"],
        "y_avg": y_avg,
    }


def load_configs() -> Dict[str, dict]:
    """
    Load line A and line B (two-line gate) from JSON.

    Returns:
        ``line``: line A (same as before; used for line_y label),
        ``line_b``: line B segment,
        ``line_a``: alias of ``line`` for clarity.
    """
    if not os.path.exists(LINE_CONFIG_FILE):
        raise FileNotFoundError(f"Line config not found: {LINE_CONFIG_FILE}")
    if not os.path.exists(LINE_B_CONFIG_FILE):
        raise FileNotFoundError(f"Line B config not found: {LINE_B_CONFIG_FILE}")

    with open(LINE_CONFIG_FILE, "r") as f:
        line_a = _normalize_line_dict(json.load(f))
    with open(LINE_B_CONFIG_FILE, "r") as f:
        line_b = _normalize_line_dict(json.load(f))

    return {"line": line_a, "line_a": line_a, "line_b": line_b}


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


# ---------- Giving IDs: assign persistent IDs using Euclidean distance matching ----------
def assign_ids_by_centroid_matching(
    centroids_A: List[Point],
    centroids_B: List[Point],
    ids_B: List[int],
    next_available_id: int,
) -> Tuple[List[int], int]:
    """
    Assign IDs to persons in frame A by matching their centroids to frame B (same person = closest within threshold).
    Used to carry IDs across frames: B is the previous frame we already have IDs for; A is the current frame we are assigning IDs to.
    Unmatched persons in A get new IDs.

    Returns:
        (ids_A, next_available_id) where ids_A[i] is the ID for person i in frame A.
    """
    if not centroids_A:
        return [], next_available_id
    if not centroids_B or not ids_B:
        ids_A = list(range(next_available_id, next_available_id + len(centroids_A)))
        return ids_A, next_available_id + len(centroids_A)

    all_pairs = []
    for i, cent_a in enumerate(centroids_A):
        for j, cent_b in enumerate(centroids_B):
            dist = math.dist(cent_a, cent_b)
            if dist < MATCHING_THRESHOLD:
                all_pairs.append((dist, i, j))
    all_pairs.sort(key=lambda x: x[0])

    ids_A: List[int] = [-1] * len(centroids_A)
    used_B = set()

    for _dist, i_a, j_b in all_pairs:
        if ids_A[i_a] >= 0 or j_b in used_B:
            continue
        ids_A[i_a] = ids_B[j_b]
        used_B.add(j_b)

    for i in range(len(centroids_A)):
        if ids_A[i] < 0:
            ids_A[i] = next_available_id
            next_available_id += 1

    return (ids_A, next_available_id)


def line_cross_product(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Signed 2× cross of (P - P1) with (P2 - P1); sign = which side of the infinite line through P1–P2."""
    return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)


def build_line_crossing_context_shared(
    line_config: dict,
    ref_px: float,
    ref_py: float,
    on_line_eps: float = LINE_ON_CROSS_EPS,
) -> LineCrossingContext:
    """
    Half-plane inside/outside for this segment using a **shared** reference point (ref_px, ref_py)
    so line A and line B agree on which side is "inside" (store interior).
    """
    x1 = float(line_config["x1"])
    y1 = float(line_config["y1"])
    x2 = float(line_config["x2"])
    y2 = float(line_config["y2"])
    y_avg = float(line_config.get("y_avg", (y1 + y2) / 2))
    cross_ref = line_cross_product(ref_px, ref_py, x1, y1, x2, y2)
    seg_len = math.hypot(x2 - x1, y2 - y1)
    eps = max(on_line_eps, 0.005 * seg_len) if seg_len > 0 else on_line_eps
    if abs(cross_ref) < 1e-6:
        cross_ref = 1.0
    ctx: LineCrossingContext = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "y_avg": y_avg,
        "cross_ref": cross_ref,
        "on_line_eps": eps,
    }
    return ctx


def inside_reference_from_line_a(line_a: dict, inside_offset_y: float = LINE_INSIDE_REFERENCE_OFFSET_Y) -> Tuple[float, float]:
    """Reference point in image coords: line A midpoint shifted down by inside_offset_y."""
    x1, y1, x2, y2 = float(line_a["x1"]), float(line_a["y1"]), float(line_a["x2"]), float(line_a["y2"])
    return ((x1 + x2) / 2, (y1 + y2) / 2 + inside_offset_y)


def build_two_line_contexts(line_a: dict, line_b: dict) -> Tuple[LineCrossingContext, LineCrossingContext]:
    rx, ry = inside_reference_from_line_a(line_a)
    return (
        build_line_crossing_context_shared(line_a, rx, ry),
        build_line_crossing_context_shared(line_b, rx, ry),
    )


def point_is_inside(px: float, py: float, ctx: LineCrossingContext) -> bool:
    """
    True if point lies on the same side of the line as the inside reference (see build_line_crossing_context).
    Near-degenerate (on-line) uses legacy y >= y_avg.
    """
    c = line_cross_product(px, py, ctx["x1"], ctx["y1"], ctx["x2"], ctx["y2"])
    if abs(c) < ctx["on_line_eps"]:
        return py >= ctx["y_avg"]
    return c * ctx["cross_ref"] > 0


def _advance_inside_hysteresis(
    st: PersonCrossingState,
    observed_inside: bool,
    confirm_frames: int,
) -> None:
    """Update stable_inside toward observed_inside; require confirm_frames consecutive agreeing reads."""
    if observed_inside == st["stable_inside"]:
        st["pending_inside"] = None
        st["pending_count"] = 0
        return

    if st["pending_inside"] != observed_inside:
        st["pending_inside"] = observed_inside
        st["pending_count"] = 1
    else:
        st["pending_count"] += 1

    if st["pending_count"] < confirm_frames:
        return

    st["stable_inside"] = observed_inside
    st["pending_inside"] = None
    st["pending_count"] = 0


def _initial_gate_phase(inside_a: bool, inside_b: bool) -> GatePhase:
    if inside_a and inside_b:
        return "idle_inside"
    if not inside_a and not inside_b:
        return "idle_outside"
    return "corridor_ambiguous"


def _empty_line_state(stable_inside: bool) -> PersonCrossingState:
    return {
        "stable_inside": stable_inside,
        "pending_inside": None,
        "pending_count": 0,
    }


def two_line_gate_event(
    crossing_state: Dict[int, PersonTwoLineState],
    person_id: int,
    prev_centroid: Point,
    curr_centroid: Point,
    ctx_a: LineCrossingContext,
    ctx_b: LineCrossingContext,
    confirm_frames: int = CROSSING_HYSTERESIS_FRAMES,
) -> Optional[Literal["ENTRY", "EXIT"]]:
    """
    Two-line gate: EXIT = inside both half-planes → cross line A out (not inside A, still inside B) →
    outside both. ENTRY = outside both → cross B in (not A, inside B) → inside both.

    Per-line hysteresis reduces jitter; FSM enforces A-then-B for exit and B-then-A for entry.
    """
    pxp, pyp = float(prev_centroid[0]), float(prev_centroid[1])
    pxc, pyc = float(curr_centroid[0]), float(curr_centroid[1])
    prev_a = point_is_inside(pxp, pyp, ctx_a)
    prev_b = point_is_inside(pxp, pyp, ctx_b)
    curr_a = point_is_inside(pxc, pyc, ctx_a)
    curr_b = point_is_inside(pxc, pyc, ctx_b)

    if person_id not in crossing_state:
        crossing_state[person_id] = {
            "line_a": _empty_line_state(prev_a),
            "line_b": _empty_line_state(prev_b),
            "gate_phase": _initial_gate_phase(prev_a, prev_b),
        }

    st = crossing_state[person_id]
    _advance_inside_hysteresis(st["line_a"], curr_a, confirm_frames)
    _advance_inside_hysteresis(st["line_b"], curr_b, confirm_frames)

    a = st["line_a"]["stable_inside"]
    b = st["line_b"]["stable_inside"]
    phase: GatePhase = st["gate_phase"]
    event: Optional[Literal["ENTRY", "EXIT"]] = None

    if phase == "corridor_ambiguous":
        if a and b:
            event = "ENTRY"
            phase = "idle_inside"
        elif not a and not b:
            event = "EXIT"
            phase = "idle_outside"
    elif phase == "idle_inside":
        if not a and b:
            phase = "exit_after_a"
        elif not a and not b:
            event = "EXIT"
            phase = "idle_outside"
        elif a and b:
            pass
    elif phase == "exit_after_a":
        if not a and not b:
            event = "EXIT"
            phase = "idle_outside"
        elif a and b:
            phase = "idle_inside"
    elif phase == "idle_outside":
        if not a and b:
            phase = "entry_after_b"
        elif a and b:
            event = "ENTRY"
            phase = "idle_inside"
    elif phase == "entry_after_b":
        if a and b:
            event = "ENTRY"
            phase = "idle_inside"
        elif not a and not b:
            phase = "idle_outside"

    st["gate_phase"] = phase
    return event

