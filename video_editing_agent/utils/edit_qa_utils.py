"""Shared helpers for VLM QA retry prompts (keyframe inpaint + video propagation)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
EDITING_OPERATIONS_TO_AVOID_HEADER = (
    "EDITING OPERATIONS TO AVOID / 避免的编辑操作 (from prior QA — these are mistakes from the "
    "last attempt; do NOT repeat them on this single retry):"
)

_EDIT_RETRY_POSITIVE_HEADER = (
    "KEEP THESE CORRECT OPERATIONS (apply first — maintain correct edit actions):"
)
_EDIT_RETRY_AVOID_HEADER = (
    "AVOID THESE EDITING MISTAKES (apply second — do not repeat failed operations):"
)


def build_edit_retry_guidance_section(
    *,
    positive_prompt: str = "",
    avoid_operations: str = "",
    baseline_positive: str = "",
    baseline_avoid: str = "",
    retry_objective: str = "",
) -> str:
    """Assemble image-edit retry guidance with LLM-refined deduplication.

    Order (highest priority LAST, so the model's recency bias emphasizes it):
    1. KEEP (positive — what to maintain)
    2. MUST APPLY (missing edits / retry objective — what to do)
    3. AVOID (critical problem areas — what NOT to do, emphasized at the end)

    On the first attempt, only baseline guidance is included when provided.
    On retry, QA ``positive_prompt`` is appended after baseline positive; QA
    ``avoid_operations`` replaces the standalone baseline avoid (callers should
    merge state-lock into ``avoid_operations`` before passing).
    """
    is_retry = bool((positive_prompt or "").strip() or (avoid_operations or "").strip())

    positive_lines: List[str] = []
    avoid_lines: List[str] = []

    if (baseline_positive or "").strip():
        positive_lines.append(baseline_positive.strip())
    if (positive_prompt or "").strip():
        positive_lines.append(positive_prompt.strip())

    if is_retry:
        if (avoid_operations or "").strip():
            avoid_lines.append(avoid_operations.strip())
        elif (baseline_avoid or "").strip():
            avoid_lines.append(baseline_avoid.strip())
    elif (baseline_avoid or "").strip():
        avoid_lines.append(baseline_avoid.strip())

    parts: List[str] = []
    # 1. Positive guidance first (maintain what's correct)
    if positive_lines:
        parts.append(f"{_EDIT_RETRY_POSITIVE_HEADER}\n" + "\n".join(positive_lines))
    # 2. Missing edits / retry objective (what MUST be applied)
    if (retry_objective or "").strip():
        parts.append(
            "MUST APPLY THESE EDITS (apply before checking avoid list — these are the edits that were missing):\n"
            + retry_objective.strip()
        )
    # 3. AVOID section LAST — highest priority emphasis at the end of the prompt
    if avoid_lines:
        parts.append(
            "CRITICAL — AVOID THESE EDITING MISTAKES (HIGHEST PRIORITY — read carefully, do NOT repeat these failures):\n"
            + "\n".join(avoid_lines)
        )
    if not parts:
        return ""
    return "\n\n" + "\n\n".join(parts) + "\n"


def normalize_qa_error_list(raw: object) -> List[str]:
    """Normalize VLM ``edit_errors`` / ``failed_aspects`` to a string list."""
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [str(raw).strip()] if str(raw).strip() else []


def build_keyframe_qa_avoid_operations(qa: Dict[str, Any]) -> str:
    """Build avoid-operations text from VLM keyframe QA (retry_focus + edit_errors fallback)."""
    focus = str(qa.get("retry_focus_prompt", "") or "").strip()
    if focus:
        return focus

    errors = normalize_qa_error_list(qa.get("edit_errors"))
    if not errors:
        errors = normalize_qa_error_list(qa.get("failed_aspects"))

    lines: List[str] = []
    for err in errors:
        lowered = err.lower()
        if lowered.startswith("do not ") or lowered.startswith("avoid "):
            lines.append(f"- {err}")
        else:
            lines.append(f"- Do not repeat this mistake: {err}")

    feedback = str(qa.get("feedback", "") or "").strip()
    if feedback and not lines:
        lines.append(f"- Do not repeat this mistake: {feedback}")

    return "\n".join(lines)


def build_editing_operations_to_avoid_block(avoid_ops: str) -> str:
    """Format QA ``retry_focus_prompt`` as an explicit avoid block."""
    focus = (avoid_ops or "").strip()
    if not focus:
        return ""
    return f"{EDITING_OPERATIONS_TO_AVOID_HEADER}\n{focus}"


def append_editing_operations_to_avoid(base_prompt: str, avoid_ops: str) -> str:
    """Append avoid-operations block to a base edit/inpaint prompt."""
    block = build_editing_operations_to_avoid_block(avoid_ops)
    base = (base_prompt or "").strip()
    if not block:
        return base
    return f"{base}\n\n{block}"


def prepare_keyframe_qa_images(
    original_image_path: str,
    edited_image_path: str,
) -> Tuple[Image.Image, Image.Image, bool]:
    """Load keyframe QA pair; resize edited frame to original video dimensions.

    Returns:
        (edited_image, original_image, resized_for_qa)
    """
    orig_img = Image.open(original_image_path).convert("RGB")
    edit_img = Image.open(edited_image_path).convert("RGB")
    resized = False
    if edit_img.size != orig_img.size:
        logger.info(
            "Keyframe QA: resizing edited frame %s → original %s for VLM comparison",
            edit_img.size,
            orig_img.size,
        )
        edit_img = edit_img.resize(orig_img.size, Image.Resampling.LANCZOS)
        resized = True
    return edit_img, orig_img, resized


def assess_letterbox_structure_preserved(
    original_image: Image.Image,
    edited_image: Image.Image,
    *,
    darkness_threshold: float = 18.0,
    max_bar_brightness_delta: float = 10.0,
    min_bar_fraction: float = 0.025,
) -> Tuple[bool, str]:
    """Deterministically verify original black bars/margins are still black.

    VLM QA can miss removed letterboxing after the edited frame is resized for
    comparison. This check only fails when the original has measurable black
    bars and those exact regions become visibly non-black in the edited frame.
    """
    if edited_image.size != original_image.size:
        edited_image = edited_image.resize(original_image.size, Image.Resampling.LANCZOS)

    orig = np.asarray(original_image.convert("RGB"), dtype=np.float32)
    edit = np.asarray(edited_image.convert("RGB"), dtype=np.float32)
    height, width = orig.shape[:2]
    min_rows = max(2, int(round(height * min_bar_fraction)))
    min_cols = max(2, int(round(width * min_bar_fraction)))

    def _leading_bar(mask: np.ndarray) -> int:
        count = 0
        for value in mask:
            if bool(value):
                count += 1
            else:
                break
        return count

    row_dark = orig.mean(axis=(1, 2)) <= darkness_threshold
    col_dark = orig.mean(axis=(0, 2)) <= darkness_threshold
    bars = {
        "top": _leading_bar(row_dark),
        "bottom": _leading_bar(row_dark[::-1]),
        "left": _leading_bar(col_dark),
        "right": _leading_bar(col_dark[::-1]),
    }

    checked = False
    failures: List[str] = []
    orig_global = float(orig.mean())
    for side, thickness in bars.items():
        min_thickness = min_rows if side in {"top", "bottom"} else min_cols
        if thickness < min_thickness:
            continue
        checked = True
        if side == "top":
            orig_region = orig[:thickness, :, :]
            edit_region = edit[:thickness, :, :]
        elif side == "bottom":
            orig_region = orig[height - thickness :, :, :]
            edit_region = edit[height - thickness :, :, :]
        elif side == "left":
            orig_region = orig[:, :thickness, :]
            edit_region = edit[:, :thickness, :]
        else:
            orig_region = orig[:, width - thickness :, :]
            edit_region = edit[:, width - thickness :, :]

        edit_mean = float(edit_region.mean())
        orig_mean = float(orig_region.mean())
        if (
            edit_mean > darkness_threshold + max_bar_brightness_delta
            and edit_mean > orig_mean + max_bar_brightness_delta
            and edit_mean > orig_global * 0.35
        ):
            failures.append(
                f"{side} black bar changed from mean {orig_mean:.1f} to {edit_mean:.1f}"
            )

    if not checked:
        return True, "original frame has no measurable black bars"
    if failures:
        return False, "; ".join(failures)
    return True, "original black bars preserved"


def fuse_vlm_detection_into_edit_prompt(
    edit_prompt: str,
    *,
    location_prompt: str = "",
    success_criteria_prompt: str = "",
) -> str:
    """Merge VLM location + QA success-criteria text into the base edit prompt."""
    edit = (edit_prompt or "").strip()
    location = (location_prompt or "").strip()
    criteria = (success_criteria_prompt or "").strip()

    parts: List[str] = []
    if location and location not in edit:
        parts.append(f"Target in frame: {location}")
    if edit:
        parts.append(edit)
    elif location:
        parts.append(location)

    merged = " ".join(parts).strip()
    if criteria and criteria not in merged and criteria != edit:
        merged = f"{merged} | Success criteria: {criteria}" if merged else criteria
    return merged or edit or location or criteria


def merge_vlm_prompts_into_planned_edits(
    planned_edits: List[Dict[str, Any]],
    *,
    success_criteria_by_instruction: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Fuse VLM location + QA prompts into each planned edit's ``edit_prompt``."""
    merged: List[Dict[str, Any]] = []
    for item in planned_edits:
        iid = str(item.get("instruction_id", "")).strip()
        updated = dict(item)
        updated["edit_prompt"] = fuse_vlm_detection_into_edit_prompt(
            str(item.get("edit_prompt", "") or ""),
            location_prompt=str(
                item.get("location_edit_prompt")
                or item.get("location_prompt")
                or ""
            ),
            success_criteria_prompt=success_criteria_by_instruction.get(iid, ""),
        )
        merged.append(updated)
    return merged


def overwrite_planned_edits_sidecar(
    location_sidecar_path: str,
    planned_edits: List[Dict[str, Any]],
) -> None:
    """Overwrite ``planned_edits`` in a Module-3 location sidecar JSON file."""
    if not location_sidecar_path or not os.path.exists(location_sidecar_path):
        return
    with open(location_sidecar_path, encoding="utf-8") as fh:
        data = json.load(fh)
    data["planned_edits"] = planned_edits
    with open(location_sidecar_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


SINGLE_KEYFRAME_QA_CRITICAL_FLAGS = (
    "frame_structure_preserved",
    "edit_instruction_requirements_met",
    "edit_completed",
    "entity_identity_preserved",
    "unrelated_edit_changes_absent",
    "background_unedited_regions_preserved",
    "canonical_reference_alignment_ok",
)

SINGLE_KEYFRAME_QA_IMPORTANT_FLAGS = (
    "visibility_extent_preserved",
    "pose_expression_preserved",
    "entity_local_lighting_preserved",
    "environment_blend_ok",
)

# All flags (for logging / sidecar compatibility).
SINGLE_KEYFRAME_QA_REQUIRED_FLAGS = (
    *SINGLE_KEYFRAME_QA_CRITICAL_FLAGS,
    *SINGLE_KEYFRAME_QA_IMPORTANT_FLAGS,
)

_SINGLE_KEYFRAME_QA_MIN_SCORE = 0.6
_SINGLE_KEYFRAME_QA_MIN_IMPORTANT_PASS = len(SINGLE_KEYFRAME_QA_IMPORTANT_FLAGS)
_BACKGROUND_DRIFT_GRID_SIZE = 4
_BACKGROUND_DRIFT_CELL_CHANGE_THRESHOLD = 0.28
_BACKGROUND_DRIFT_PIXEL_DIFF_THRESHOLD = 15.0
_BACKGROUND_DRIFT_REEDIT_RATIO_THRESHOLD = 1.35
_BACKGROUND_DRIFT_REEDIT_ABS_MARGIN = 0.01
_BACKGROUND_DRIFT_REEDIT_MIN_VIOLATION_CELLS = 2
_BACKGROUND_DRIFT_REEDIT_SINGLE_CELL_THRESHOLD = 0.45

_NON_EDIT_REGION_CHANGE_CANONICAL = {
    "": "unknown",
    "unknown": "unknown",
    "none": "none",
    "no_change": "none",
    "trace": "trace",
    "tiny": "trace",
    "tiny_seam": "trace",
    "minor": "trace",
    "slight": "trace",
    "localized": "trace",
    "local": "trace",
    "moderate": "moderate",
    "material": "moderate",
    "noticeable": "moderate",
    "significant": "moderate",
    "major": "severe",
    "severe": "severe",
    "large": "severe",
    "broad": "severe",
}


def normalize_non_edit_region_change_severity(value: Any) -> str:
    """Map VLM/free-form severity text to a stable set of labels."""
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in _NON_EDIT_REGION_CHANGE_CANONICAL:
        return _NON_EDIT_REGION_CHANGE_CANONICAL[text]
    for token, canonical in _NON_EDIT_REGION_CHANGE_CANONICAL.items():
        if token and token in text:
            return canonical
    return "unknown"


def non_edit_region_change_requires_reedit(value: Any) -> bool:
    """Moderate-or-worse non-edit region drift should trigger a retry/fail."""
    return normalize_non_edit_region_change_severity(value) in {"moderate", "severe"}


def classify_background_drift_severity(drift_metrics: Dict[str, Any]) -> str:
    """Classify pixel drift outside edit zones so only material drift hard-fails."""
    if not isinstance(drift_metrics, dict) or not drift_metrics:
        return "unknown"
    try:
        outside_frac = float(
            drift_metrics.get("background_drift_outside_changed_fraction", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        outside_frac = 0.0
    try:
        outside_budget = float(
            drift_metrics.get("background_drift_outside_budget", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        outside_budget = 0.0
    violations = drift_metrics.get("background_drift_violation_cells") or []
    violation_count = len(violations) if isinstance(violations, list) else int(bool(violations))
    max_cell_fraction = 0.0
    if isinstance(violations, list):
        for item in violations:
            if not isinstance(item, dict):
                continue
            try:
                max_cell_fraction = max(
                    max_cell_fraction,
                    float(item.get("changed_fraction", 0.0) or 0.0),
                )
            except (TypeError, ValueError):
                continue

    if outside_frac <= outside_budget and violation_count == 0:
        return "none"

    effective_budget = max(outside_budget, 1e-6)
    ratio = outside_frac / effective_budget
    material_drift = (
        max_cell_fraction >= _BACKGROUND_DRIFT_REEDIT_SINGLE_CELL_THRESHOLD
        or violation_count >= _BACKGROUND_DRIFT_REEDIT_MIN_VIOLATION_CELLS
        or outside_frac > outside_budget + _BACKGROUND_DRIFT_REEDIT_ABS_MARGIN
        or ratio >= _BACKGROUND_DRIFT_REEDIT_RATIO_THRESHOLD
    )
    if material_drift:
        return "moderate"
    return "trace"


def _normalize_location_text(text: str) -> str:
    return str(text or "").strip().lower()


def _expand_grid_cells(
    cells: Set[Tuple[int, int]],
    grid_size: int,
    *,
    margin: int = 1,
) -> Set[Tuple[int, int]]:
    expanded: Set[Tuple[int, int]] = set()
    for row, col in cells:
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    expanded.add((nr, nc))
    return expanded


def _compute_core_location_grid_cells(
    location_description: str,
    grid_size: int,
) -> Set[Tuple[int, int]]:
    """Map a coarse location phrase to core grid cells (no margin expansion)."""
    text = _normalize_location_text(location_description)
    rows = list(range(grid_size))
    cols = list(range(grid_size))

    if any(token in text for token in ("upper", "top")) and "mid" not in text:
        rows = [0, 1]
    elif any(token in text for token in ("midground", "mid-ground", "middle")):
        rows = [1, 2]
    elif any(token in text for token in ("lower", "bottom", "foreground")):
        rows = [2, 3]

    if any(
        phrase in text
        for phrase in ("center-left", "left-center", "centre-left", "left-centre")
    ) or (
        "left" in text and any(token in text for token in ("center", "centre"))
    ):
        cols = [0, 1, 2]
    elif any(
        phrase in text
        for phrase in ("center-right", "right-center", "centre-right", "right-centre")
    ) or (
        "right" in text and any(token in text for token in ("center", "centre"))
    ):
        cols = [1, 2, 3]
    elif "far left" in text or "extreme left" in text:
        cols = [0]
    elif "left" in text:
        cols = [0, 1]
    elif "far right" in text or "extreme right" in text:
        cols = [grid_size - 1]
    elif "right" in text:
        cols = [2, 3]
    elif "center" in text or "centre" in text:
        cols = [1, 2]

    return {(row, col) for row in rows for col in cols}


def _location_grid_cells(location_description: str, grid_size: int) -> Set[Tuple[int, int]]:
    """Map a coarse location phrase to allowed edit grid cells."""
    return _expand_grid_cells(
        _compute_core_location_grid_cells(location_description, grid_size),
        grid_size,
        margin=1,
    )


def _cap_cells_to_budget(
    cells: Set[Tuple[int, int]],
    max_cells: int,
) -> Set[Tuple[int, int]]:
    if len(cells) <= max_cells:
        return set(cells)
    center_row = sum(row for row, _col in cells) / len(cells)
    center_col = sum(col for _row, col in cells) / len(cells)
    ranked = sorted(
        cells,
        key=lambda rc: (rc[0] - center_row) ** 2 + (rc[1] - center_col) ** 2,
    )
    return set(ranked[:max_cells])


def _allowed_edit_grid_cells(
    entity_location_records: Dict[str, Dict[str, Any]],
    *,
    grid_size: int,
) -> Set[Tuple[int, int]]:
    allowed: Set[Tuple[int, int]] = set()
    total_cells = grid_size * grid_size
    for record in entity_location_records.values():
        if not record.get("present"):
            continue
        location = str(record.get("location_description", "") or "").strip()
        if not location:
            continue
        core = _compute_core_location_grid_cells(location, grid_size)
        area_fraction = float(record.get("approximate_area_fraction", 0.1) or 0.1)
        area_fraction = max(0.05, min(area_fraction, 0.5))
        budget = max(1, min(len(core), round(area_fraction * total_cells * 2.5)))
        core = _cap_cells_to_budget(core, budget)
        allowed |= core
    if not allowed:
        center = grid_size // 2
        allowed = _expand_grid_cells({(center, center)}, grid_size, margin=1)
    return allowed


def measure_keyframe_background_drift(
    original_image_path: str,
    edited_image_path: str,
    entity_location_records: Dict[str, Dict[str, Any]],
    *,
    grid_size: int = _BACKGROUND_DRIFT_GRID_SIZE,
    cell_change_threshold: float = _BACKGROUND_DRIFT_CELL_CHANGE_THRESHOLD,
    pixel_diff_threshold: float = _BACKGROUND_DRIFT_PIXEL_DIFF_THRESHOLD,
) -> Dict[str, Any]:
    """Detect background edits outside coarse entity-location zones via pixel diff."""
    if not original_image_path or not edited_image_path:
        return {
            "background_unedited_regions_preserved": False,
            "background_drift_violation_cells": [],
            "background_drift_error": "missing image path",
        }
    if not os.path.exists(original_image_path) or not os.path.exists(edited_image_path):
        return {
            "background_unedited_regions_preserved": False,
            "background_drift_violation_cells": [],
            "background_drift_error": "image file missing",
        }

    orig = Image.open(original_image_path).convert("RGB")
    edit = Image.open(edited_image_path).convert("RGB")
    if edit.size != orig.size:
        edit = edit.resize(orig.size, Image.Resampling.LANCZOS)

    orig_arr = np.asarray(orig, dtype=np.float32)
    edit_arr = np.asarray(edit, dtype=np.float32)
    diff = np.abs(orig_arr - edit_arr).mean(axis=2)
    height, width = diff.shape
    allowed_cells = _allowed_edit_grid_cells(
        entity_location_records,
        grid_size=grid_size,
    )

    violations: List[Dict[str, Any]] = []
    outside_changed_pixels = 0
    total_pixels = height * width
    allowed_pixel_mask = np.zeros((height, width), dtype=bool)
    for row, col in allowed_cells:
        allowed_pixel_mask[
            row * height // grid_size:(row + 1) * height // grid_size,
            col * width // grid_size:(col + 1) * width // grid_size,
        ] = True

    changed_mask = diff > pixel_diff_threshold
    outside_changed_pixels = int((changed_mask & ~allowed_pixel_mask).sum())

    for row in range(grid_size):
        row_start = row * height // grid_size
        row_end = (row + 1) * height // grid_size
        for col in range(grid_size):
            col_start = col * width // grid_size
            col_end = (col + 1) * width // grid_size
            patch = diff[row_start:row_end, col_start:col_end]
            if patch.size == 0:
                continue
            changed_fraction = float((patch > pixel_diff_threshold).mean())
            if changed_fraction < cell_change_threshold:
                continue
            if (row, col) in allowed_cells:
                continue
            violations.append({
                "grid_cell": [row, col],
                "changed_fraction": round(changed_fraction, 3),
            })

    outside_changed_fraction = (
        float(outside_changed_pixels) / float(total_pixels) if total_pixels else 0.0
    )
    outside_budget = min(
        0.06,
        max(
            0.015,
            sum(
                float(record.get("approximate_area_fraction", 0.0) or 0.0)
                for record in entity_location_records.values()
                if record.get("present")
            )
            * 0.35,
        ),
    )
    outside_drift = outside_changed_fraction > outside_budget

    return {
        "background_unedited_regions_preserved": not violations and not outside_drift,
        "background_drift_violation_cells": violations,
        "background_drift_allowed_cells": sorted(allowed_cells),
        "background_drift_outside_changed_fraction": round(
            outside_changed_fraction, 4
        ),
        "background_drift_outside_budget": round(outside_budget, 4),
        "background_drift_grid_size": grid_size,
    }


def merge_background_drift_into_qa_result(
    qa: Dict[str, Any],
    drift_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach pixel-level background drift diagnostics and hard-fail material non-edit drift."""
    out = dict(qa)
    heuristic_ok = bool(
        drift_metrics.get("background_unedited_regions_preserved", False)
    )
    severity = classify_background_drift_severity(drift_metrics)
    requires_reedit = non_edit_region_change_requires_reedit(severity)
    out["background_drift_metrics"] = drift_metrics
    out["background_drift_heuristic_ok"] = heuristic_ok
    out["background_drift_severity"] = severity
    out["background_drift_requires_reedit"] = requires_reedit
    out["background_drift_hard_gate_disabled"] = not requires_reedit

    violations = drift_metrics.get("background_drift_violation_cells") or []
    cells = ", ".join(
        f"({v['grid_cell'][0]},{v['grid_cell'][1]}:{v['changed_fraction']:.2f})"
        for v in violations[:4]
        if isinstance(v, dict) and isinstance(v.get("grid_cell"), list) and len(v.get("grid_cell")) >= 2
    )
    outside_frac = drift_metrics.get("background_drift_outside_changed_fraction")
    outside_budget = drift_metrics.get("background_drift_outside_budget")
    note_parts = []
    if cells:
        note_parts.append(f"background drift outside edit zones at cells {cells}")
    if outside_frac is not None and outside_budget is not None:
        comparator = ">" if outside_frac > outside_budget else "<="
        note_parts.append(
            f"outside-zone pixel change {outside_frac:.3f} {comparator} budget {outside_budget:.3f}"
        )
    advisory = "; ".join(note_parts) or "background drift outside edit zones"
    if not heuristic_ok:
        out["background_drift_advisory"] = advisory

    if requires_reedit:
        out["passed"] = False
        out["background_unedited_regions_preserved"] = False
        failed_aspects = [
            str(item).strip()
            for item in (out.get("failed_aspects") or [])
            if str(item).strip()
        ]
        if "non-edit regions changed too much" not in failed_aspects:
            failed_aspects.append("non-edit regions changed too much")
        out["failed_aspects"] = failed_aspects
        base_feedback = str(out.get("feedback", "") or "").strip()
        gate_note = (
            "Non-edit regions changed beyond the allowed retry threshold; re-edit required. "
            + advisory
        )
        out["feedback"] = f"{base_feedback} {gate_note}".strip()
        retry_focus = str(out.get("retry_focus_prompt", "") or "").strip()
        retry_note = (
            "Preserve all non-edit regions outside the exact edit silhouette. If a tiny seam is needed, keep it "
            "strictly local; do not repaint unrelated walls, floor, ceiling, props, edge patches, or non-target people."
        )
        out["retry_focus_prompt"] = f"{retry_focus} {retry_note}".strip()

    return out


_CANONICAL_ALIGNMENT_EDIT_CUES = (
    "hat",
    "cap",
    "hair",
    "color",
    "colour",
    "shirt",
    "dress",
    "vest",
    "accessory",
    "glasses",
    "beard",
    "jacket",
    "coat",
)


def canonical_alignment_check_applicable(edit_instruction: str) -> bool:
    """Whether a focused canonical style check is meaningful for this edit."""
    text = str(edit_instruction or "").strip().lower()
    if not text:
        return False
    if any(token in text for token in ("remove", "delete", "erase", "inpaint out")):
        return False
    return any(cue in text for cue in _CANONICAL_ALIGNMENT_EDIT_CUES)


def instruction_id_from_canonical_ref_path(canonical_path: str) -> str:
    """Parse ``instr_XXX`` from ``{instruction_id}_ref_canonical.png``."""
    base = os.path.basename(str(canonical_path or "").strip())
    if base.endswith("_ref_canonical.png"):
        return base[: -len("_ref_canonical.png")]
    return ""


def _strip_bottom_caption_strip(
    img: Image.Image,
    *,
    min_caption_ratio: float = 0.08,
) -> Image.Image:
    """Remove bottom caption by detecting the last row with significant content."""
    arr = np.asarray(img.convert("RGB"))
    h, w, _ = arr.shape
    scan_start = max(1, int(h * (1 - min_caption_ratio)))
    content_bottom = h - 1
    for y in range(h - 1, scan_start - 1, -1):
        row = arr[y]
        if float(np.mean(row)) < 250.0 or float(np.std(row)) > 8.0:
            content_bottom = y
            break
    else:
        content_bottom = max(1, int(h * 0.88)) - 1
    return img.crop((0, 0, w, max(1, content_bottom + 1)))


def extract_canonical_target_panel(canonical_path: str) -> Image.Image | None:
    """Crop the RIGHT (edited-target) panel from a canonical reference card."""
    if not canonical_path or not os.path.exists(canonical_path):
        return None
    img = Image.open(canonical_path).convert("RGB")
    upper = _strip_bottom_caption_strip(img)
    width, height = upper.size
    if width < height * 1.15:
        return upper
    gap = 12
    panel_width = (width - gap) // 2
    if panel_width <= 0:
        return None
    return upper.crop((panel_width + gap, 0, width, height))


def parse_edit_instruction_for_instruction_id(
    canonical_edit_block: str,
    instruction_id: str,
) -> str:
    """Extract the edit instruction line for one instruction from the QA block."""
    prefix = f"- {instruction_id} /"
    for line in str(canonical_edit_block or "").splitlines():
        text = line.strip()
        if text.startswith(prefix):
            marker = "edit instruction = "
            if marker in text:
                return text.split(marker, 1)[1].strip()
            return text
    return ""


def merge_canonical_alignment_into_qa_result(
    qa: Dict[str, Any],
    alignment_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Override canonical_reference_alignment_ok when focused checks fail."""
    if not alignment_results:
        return qa
    out = dict(qa)
    failed = [entry for entry in alignment_results if not entry.get("alignment_ok", True)]
    out["canonical_alignment_checks"] = list(alignment_results)
    if not failed:
        out["canonical_reference_alignment_ok"] = True
        return out

    out["canonical_reference_alignment_ok"] = False
    notes = [
        str(entry.get("feedback", "") or "").strip()
        for entry in failed
        if str(entry.get("feedback", "") or "").strip()
    ]
    if notes:
        base_feedback = str(out.get("feedback", "") or "").strip()
        note = "canonical alignment failed: " + "; ".join(notes)
        out["feedback"] = f"{base_feedback} | {note}".strip(" |")
    failed_aspects = list(out.get("failed_aspects") or [])
    for entry in failed:
        for attr in entry.get("mismatched_attributes") or []:
            text = str(attr).strip()
            if text and text not in failed_aspects:
                failed_aspects.append(text)
    if "canonical_reference_alignment" not in failed_aspects:
        failed_aspects.append("canonical_reference_alignment")
    out["failed_aspects"] = failed_aspects

    retry_bits = [
        str(entry.get("retry_focus_prompt", "") or "").strip()
        for entry in failed
        if str(entry.get("retry_focus_prompt", "") or "").strip()
    ]
    if retry_bits and not str(out.get("retry_focus_prompt", "") or "").strip():
        out["retry_focus_prompt"] = " ".join(retry_bits)
    elif retry_bits:
        out["retry_focus_prompt"] = (
            f"{out.get('retry_focus_prompt', '')} {' '.join(retry_bits)}"
        ).strip()
    return out


def _qa_bool_from_vlm(data: Dict[str, Any], primary: str, *fallback_keys: str) -> bool:
    if primary in data:
        return bool(data.get(primary, False))
    for key in fallback_keys:
        if key in data:
            return bool(data.get(key, False))
    return False


def build_single_keyframe_qa_result_from_vlm(
    data: Dict[str, Any],
    *,
    resized_for_qa: bool = False,
    has_canonical_refs: bool = False,
) -> Dict[str, Any]:
    """Normalize VLM single-keyframe QA JSON into gate input."""
    passed = bool(data.get("passed", False))
    try:
        score = float(data.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0

    structure_ok = _qa_bool_from_vlm(data, "frame_structure_preserved")
    visibility_ok = _qa_bool_from_vlm(data, "visibility_extent_preserved")
    pose_ok = _qa_bool_from_vlm(data, "pose_expression_preserved")
    blend_ok = _qa_bool_from_vlm(data, "environment_blend_ok")
    lighting_ok = _qa_bool_from_vlm(
        data, "entity_local_lighting_preserved", "environment_blend_ok"
    )
    unrelated_ok = _qa_bool_from_vlm(data, "unrelated_edit_changes_absent")
    edit_ok = _qa_bool_from_vlm(data, "edit_completed")
    edit_requirements_ok = _qa_bool_from_vlm(
        data,
        "edit_instruction_requirements_met",
        "edit_completed",
    )
    identity_ok = _qa_bool_from_vlm(data, "entity_identity_preserved")
    background_ok = (
        bool(data.get("background_unedited_regions_preserved", False))
        if "background_unedited_regions_preserved" in data
        else False
    )
    if has_canonical_refs:
        canonical_ok = (
            bool(data.get("canonical_reference_alignment_ok", False))
            if "canonical_reference_alignment_ok" in data
            else False
        )
    else:
        canonical_ok = True

    failed_aspects = data.get("failed_aspects") or []
    if not isinstance(failed_aspects, list):
        failed_aspects = [str(failed_aspects)]

    feedback = str(data.get("feedback", "") or "").strip()
    if resized_for_qa:
        feedback = (
            f"{feedback} | edited frame resized to original dimensions for QA"
        ).strip(" |")

    return {
        "passed": passed,
        "score": score if score > 0.0 else (0.85 if passed else 0.0),
        "frame_structure_preserved": structure_ok,
        "visibility_extent_preserved": visibility_ok,
        "pose_expression_preserved": pose_ok,
        "environment_blend_ok": blend_ok,
        "entity_local_lighting_preserved": lighting_ok,
        "unrelated_edit_changes_absent": unrelated_ok,
        "edit_instruction_requirements_met": edit_requirements_ok,
        "edit_completed": edit_ok,
        "entity_identity_preserved": identity_ok,
        "background_unedited_regions_preserved": background_ok,
        "canonical_reference_alignment_ok": canonical_ok,
        "non_edit_region_change_severity": normalize_non_edit_region_change_severity(
            data.get("non_edit_region_change_severity")
        ),
        "non_edit_region_change_summary": str(
            data.get("non_edit_region_change_summary", "") or ""
        ).strip(),
        "failed_aspects": [str(x) for x in failed_aspects if str(x).strip()],
        "feedback": feedback,
        "retry_focus_prompt": str(data.get("retry_focus_prompt", "") or "").strip(),
        "positive_prompt": str(data.get("positive_prompt", "") or "").strip(),
    }


def apply_single_keyframe_edit_qa_gate(qa: Dict[str, Any]) -> Dict[str, Any]:
    """Strict QA: all critical checks + all important checks + min score."""
    out = dict(qa)
    critical_checks = {
        flag: bool(out.get(flag, False))
        for flag in SINGLE_KEYFRAME_QA_CRITICAL_FLAGS
    }
    if bool(out.get("qa_has_canonical_refs")):
        if not critical_checks.get("canonical_reference_alignment_ok", False):
            critical_checks["edit_completed"] = False
            critical_checks["edit_instruction_requirements_met"] = False
    if not critical_checks.get("edit_completed", False):
        critical_checks["edit_instruction_requirements_met"] = False
    important_checks = {
        flag: bool(out.get(flag, False))
        for flag in SINGLE_KEYFRAME_QA_IMPORTANT_FLAGS
    }
    non_edit_severity = normalize_non_edit_region_change_severity(
        out.get("non_edit_region_change_severity")
    )
    out["non_edit_region_change_severity"] = non_edit_severity
    if non_edit_region_change_requires_reedit(non_edit_severity):
        critical_checks["background_unedited_regions_preserved"] = False

    checks = {**critical_checks, **important_checks}
    try:
        score = float(out.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0

    reject_reasons: List[str] = [
        name for name, ok in critical_checks.items() if not ok
    ]
    if non_edit_region_change_requires_reedit(non_edit_severity):
        reject_reasons.append(
            f"non_edit_region_change_severity={non_edit_severity}"
        )
    important_pass_count = sum(1 for ok in important_checks.values() if ok)
    if important_pass_count < _SINGLE_KEYFRAME_QA_MIN_IMPORTANT_PASS:
        reject_reasons.append(
            "important_checks="
            f"{important_pass_count}/{len(SINGLE_KEYFRAME_QA_IMPORTANT_FLAGS)}"
            f"<{_SINGLE_KEYFRAME_QA_MIN_IMPORTANT_PASS}"
        )
    if score < _SINGLE_KEYFRAME_QA_MIN_SCORE:
        reject_reasons.append(f"score={score:.2f}<{_SINGLE_KEYFRAME_QA_MIN_SCORE:.1f}")

    all_ok = not reject_reasons
    out["passed"] = all_ok
    out["qa_check_flags"] = checks
    out["qa_critical_flags"] = critical_checks
    out["qa_important_flags"] = important_checks
    out["qa_important_pass_count"] = important_pass_count
    out["qa_reject_reasons"] = reject_reasons

    if not all_ok and bool(qa.get("passed", False)):
        base_feedback = str(out.get("feedback", "") or "").strip()
        gate_note = "QA gate override — failed: " + ", ".join(reject_reasons)
        out["feedback"] = f"{base_feedback} | {gate_note}".strip(" |")

    if not all_ok and not str(out.get("retry_focus_prompt", "") or "").strip():
        hints: List[str] = []
        if not checks.get("pose_expression_preserved"):
            hints.append(
                "Do not change pose, expression, head angle, gaze, or body posture outside the edit."
            )
        if not checks.get("environment_blend_ok"):
            hints.append(
                "Do not paste flat edits — match scene lighting, shadows, color temperature, and contact edges."
            )
        if not checks.get("entity_local_lighting_preserved"):
            hints.append(
                "Preserve the entity's original local lighting at its scene position — same light direction, "
                "shadows, highlights, rim light, and ambient color on the edited region."
            )
        if not checks.get("unrelated_edit_changes_absent"):
            hints.append(
                "Change ONLY attributes explicitly required by the edit instructions; do not alter non-target "
                "people/objects or non-requested target attributes such as expression, pose, action, gaze, "
                "clothing, hands/arms, visible extent, or lighting beyond the instructed edit."
            )
        if not checks.get("visibility_extent_preserved"):
            hints.append(
                "Do not reveal or hallucinate body/face regions that were occluded, cropped, or off-frame "
                "in the original."
            )
        if not checks.get("frame_structure_preserved"):
            hints.append("Preserve exact canvas, framing, depth layering, and scene layout.")
        if not checks.get("edit_instruction_requirements_met") or not checks.get("edit_completed"):
            hints.append(
                "Do not skip, weaken, or misapply any planned edit; complete every requested edit "
                "on the correct target entity/location before optimizing preservation."
            )
        if not checks.get("background_unedited_regions_preserved"):
            hints.append(
                "Do not repaint or alter background, walls, pillars, floor, ceiling, scenery, edge patches, "
                "or any non-edit region outside the edited entity silhouettes and minimal inpaint fill for removals."
            )
        if non_edit_region_change_requires_reedit(non_edit_severity):
            hints.append(
                "Treat any moderate or larger non-edit-region change as a hard failure: only a tiny local seam "
                "adjacent to the edit boundary is acceptable."
            )
        if not checks.get("canonical_reference_alignment_ok"):
            hints.append(
                "Match the canonical reference RIGHT panel exactly for instructed attributes — same hat/cap "
                "style and silhouette (pillbox vs fedora vs beret vs brim shape), hair color, and accessory form."
            )
        out["retry_focus_prompt"] = " ".join(hints)

    return out


def build_single_keyframe_qa_avoid_operations(qa: Dict[str, Any]) -> str:
    """Prefer retry_focus_prompt; fall back to gate-aware defaults."""
    focus = build_keyframe_qa_avoid_operations(qa)
    if focus:
        return focus
    return str(qa.get("retry_focus_prompt", "") or "").strip()
