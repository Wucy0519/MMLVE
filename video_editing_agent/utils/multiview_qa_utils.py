"""VLM QA helpers for front-view entity reference synthesis and edit.

Function names keep "multiview" for compatibility with existing call sites.
"""

from __future__ import annotations

from typing import Any, Dict, List

from video_editing_agent.utils.edit_qa_utils import (
    build_edit_retry_guidance_section,
    normalize_qa_error_list,
)

MULTIVIEW_SYNTHESIS_QA_CRITICAL_FLAGS = (
    "front_view_orientation_correct",
    "entity_identity_matches_reference",
    "source_appearance_matches_reference",
    "art_style_matches_source",
)

MULTIVIEW_EDIT_QA_CRITICAL_FLAGS = (
    *MULTIVIEW_SYNTHESIS_QA_CRITICAL_FLAGS,
    "edit_completed",
    "edit_attributes_match_instruction",
)

MULTIVIEW_QA_IMPORTANT_FLAGS = (
    "panel_structure_preserved",
    "neutral_background_ok",
)

_MULTIVIEW_QA_MIN_SCORE = 0.6
_MULTIVIEW_QA_MIN_IMPORTANT_PASS = 1


def build_multiview_qa_avoid_operations(qa: Dict[str, Any]) -> str:
    """Build avoid-operations text from front-view QA for the next image attempt."""
    focus = str(qa.get("retry_focus_prompt", "") or "").strip()
    if focus:
        return focus

    errors = normalize_qa_error_list(qa.get("failed_aspects"))
    if not errors:
        errors = normalize_qa_error_list(qa.get("edit_errors"))

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


def append_multiview_avoid_section(base_prompt: str, avoid_ops: str) -> str:
    """Append QA retry guidance to a front-view synthesis/edit prompt."""
    base = (base_prompt or "").strip()
    guidance = build_edit_retry_guidance_section(avoid_operations=avoid_ops)
    if not guidance:
        return base
    return f"{base}{guidance}"


def _qa_bool(data: Dict[str, Any], key: str) -> bool:
    if key in data:
        return bool(data.get(key, False))
    return False


def build_multiview_synthesis_qa_from_vlm(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize VLM front-view synthesis QA JSON."""
    try:
        score = float(data.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0

    checks = {
        flag: _qa_bool(data, flag)
        for flag in MULTIVIEW_SYNTHESIS_QA_CRITICAL_FLAGS
    }
    if "source_appearance_matches_reference" not in data:
        checks["source_appearance_matches_reference"] = False
    important = {
        flag: _qa_bool(data, flag)
        for flag in MULTIVIEW_QA_IMPORTANT_FLAGS
    }

    failed_aspects = data.get("failed_aspects") or []
    if not isinstance(failed_aspects, list):
        failed_aspects = [str(failed_aspects)]

    return {
        "passed": bool(data.get("passed", False)),
        "score": score if score > 0.0 else (0.85 if data.get("passed") else 0.0),
        **checks,
        **important,
        "failed_aspects": [str(x) for x in failed_aspects if str(x).strip()],
        "feedback": str(data.get("feedback", "") or "").strip(),
        "retry_focus_prompt": str(data.get("retry_focus_prompt", "") or "").strip(),
        "positive_prompt": str(data.get("positive_prompt", "") or "").strip(),
        "qa_task": "synthesis",
    }


def build_multiview_edit_qa_from_vlm(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize VLM front-view edit QA JSON."""
    out = build_multiview_synthesis_qa_from_vlm(data)
    out.update({
        flag: _qa_bool(data, flag)
        for flag in (
            "edit_completed",
            "edit_attributes_match_instruction",
        )
    })
    if "edit_attributes_match_instruction" not in data:
        out["edit_attributes_match_instruction"] = False
    out["qa_task"] = "edit"
    return out


def apply_multiview_qa_gate(
    qa: Dict[str, Any],
    *,
    task: str = "synthesis",
) -> Dict[str, Any]:
    """Apply moderately strict gate for front-view reference QA."""
    out = dict(qa)
    critical_flags = (
        MULTIVIEW_EDIT_QA_CRITICAL_FLAGS
        if task == "edit"
        else MULTIVIEW_SYNTHESIS_QA_CRITICAL_FLAGS
    )
    critical_checks = {flag: bool(out.get(flag, False)) for flag in critical_flags}
    important_checks = {
        flag: bool(out.get(flag, False))
        for flag in MULTIVIEW_QA_IMPORTANT_FLAGS
    }

    try:
        score = float(out.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0

    reject_reasons = [name for name, ok in critical_checks.items() if not ok]
    important_pass = sum(1 for ok in important_checks.values() if ok)
    if important_pass < _MULTIVIEW_QA_MIN_IMPORTANT_PASS:
        reject_reasons.append(
            f"important_checks={important_pass}/{len(MULTIVIEW_QA_IMPORTANT_FLAGS)}"
            f"<{_MULTIVIEW_QA_MIN_IMPORTANT_PASS}"
        )
    if score < _MULTIVIEW_QA_MIN_SCORE:
        reject_reasons.append(f"score={score:.2f}<{_MULTIVIEW_QA_MIN_SCORE:.1f}")

    all_ok = not reject_reasons
    out["passed"] = all_ok
    out["qa_critical_flags"] = critical_checks
    out["qa_important_flags"] = important_checks
    out["qa_important_pass_count"] = important_pass
    out["qa_reject_reasons"] = reject_reasons

    if not all_ok and bool(qa.get("passed", False)):
        base_feedback = str(out.get("feedback", "") or "").strip()
        gate_note = "QA gate override — failed: " + ", ".join(reject_reasons)
        out["feedback"] = f"{base_feedback} | {gate_note}".strip(" |")

    if not all_ok and not str(out.get("retry_focus_prompt", "") or "").strip():
        hints: List[str] = []
        if not critical_checks.get("front_view_orientation_correct"):
            hints.append("Generate a single front-facing or near-front entity reference, not a side/back view or 2x2 sheet.")
        if not critical_checks.get("entity_identity_matches_reference"):
            hints.append(
                "Preserve the exact same entity identity as the input keyframe grid — face, build, "
                "clothing, hair, and distinguishing marks."
            )
        if not critical_checks.get("source_appearance_matches_reference"):
            hints.append(
                "Match SOURCE appearance from the keyframe grid exactly — same hair STYLE (updo, curls, "
                "length, parting, bun), hair COLOR (pre-edit natural color), and clothing details. "
                "Do not invent a different hairstyle or generic hair."
            )
        if not critical_checks.get("art_style_matches_source"):
            hints.append(
                "Match the visual art style of the source keyframes exactly — if the source is photorealistic, "
                "keep photorealistic; if anime/cartoon/3D/stylized, keep that same style. "
                "Do NOT convert between art styles."
            )
        if task == "edit":
            if not critical_checks.get("edit_completed"):
                hints.append(
                    "Apply the edit to the single front-view entity reference while preserving identity and pose."
                )
            if not critical_checks.get("edit_attributes_match_instruction"):
                hints.append(
                    "Edited attributes must match the instruction exactly — correct hair COLOR, "
                    "preserved hair STYLE from the source sheet, and correct hat/accessory TYPE "
                    "(not merely similar color)."
                )
        out["retry_focus_prompt"] = " ".join(hints)

    return out


def merge_multiview_focused_qa_into_result(
    qa: Dict[str, Any],
    focused: Dict[str, Any],
    *,
    alignment_flag: str,
    cascade_flags: tuple[str, ...] = (),
) -> Dict[str, Any]:
    """Override a critical QA flag when a focused sub-check fails."""
    if not focused:
        return qa
    out = dict(qa)
    focused_checks = dict(out.get("focused_checks") or {})
    focused_checks[alignment_flag] = focused
    out["focused_checks"] = focused_checks
    out["focused_appearance_check"] = focused
    alignment_ok = bool(focused.get("alignment_ok", False))
    out[alignment_flag] = alignment_ok
    if alignment_ok:
        return out

    out["passed"] = False
    for flag in cascade_flags:
        out[flag] = False
    notes = [
        str(focused.get("feedback", "") or "").strip(),
    ]
    notes = [n for n in notes if n]
    if notes:
        base_feedback = str(out.get("feedback", "") or "").strip()
        note = f"focused check failed ({alignment_flag}): " + "; ".join(notes)
        out["feedback"] = f"{base_feedback} | {note}".strip(" |")

    failed_aspects = list(out.get("failed_aspects") or [])
    for attr in focused.get("mismatched_attributes") or []:
        text = str(attr).strip()
        if text and text not in failed_aspects:
            failed_aspects.append(text)
    if alignment_flag not in failed_aspects:
        failed_aspects.append(alignment_flag)
    out["failed_aspects"] = failed_aspects

    retry_bits = [
        str(focused.get("retry_focus_prompt", "") or "").strip(),
    ]
    retry_bits = [t for t in retry_bits if t]
    if retry_bits:
        existing = str(out.get("retry_focus_prompt", "") or "").strip()
        out["retry_focus_prompt"] = (
            f"{existing} {' '.join(retry_bits)}".strip() if existing else " ".join(retry_bits)
        )
    return out
