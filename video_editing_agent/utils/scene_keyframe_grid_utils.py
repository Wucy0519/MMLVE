"""Helpers for per-keyframe editing and scene keyframe strip assembly."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image

from video_editing_agent.schemas.instructions import EntityInstruction
from video_editing_agent.utils.ffmpeg_utils import probe_video_size
from video_editing_agent.utils.mask_utils import (
    _label_entity_panel,
    entity_ref_canonical_path,
    entity_ref_multiview_path,
    entity_ref_src_path,
)

KEYFRAME_ENTITY_PRESENCE_MIN_CONFIDENCE = 0.86
KEYFRAME_ENTITY_REMOVAL_MIN_CONFIDENCE = 0.82
KEYFRAME_ENTITY_NEIGHBOR_RECOVERY_MIN_CONFIDENCE = 0.90
KEYFRAME_ENTITY_SCENE_CONTINUITY_MIN_CONFIDENCE = 0.80
KEYFRAME_ENTITY_MIN_AREA_FRACTION = 0.03
KEYFRAME_ENTITY_REQUIRED_COMPLETENESS = "sufficient"
KEYFRAME_VIEWPOINT_TOLERANT_MIN_CONFIDENCE = 0.76
KEYFRAME_VIEWPOINT_TOLERANT_MIN_AREA_FRACTION = 0.025
KEYFRAME_WIDE_SHOT_MIN_CONFIDENCE = 0.76
KEYFRAME_WIDE_SHOT_MIN_AREA_FRACTION = 0.01
KEYFRAME_CLOSE_SHOT_MIN_CONFIDENCE = 0.78
KEYFRAME_STABLE_CUE_MIN_CONFIDENCE = 0.78
KEYFRAME_STABLE_CUE_MIN_AREA_FRACTION = 0.01
KEYFRAME_CHALLENGING_VISIBILITY_MIN_CONFIDENCE = 0.80
KEYFRAME_CHALLENGING_VISIBILITY_MIN_AREA_FRACTION = 0.01
# Existence-score discounts for entities that occupy only a tiny part of the
# frame. A small on-screen footprint usually means only a fragment of the body
# is visible (e.g. a single shoulder), so the VLM existence score is capped
# rather than treated as a confidently-present, fully framed subject.
KEYFRAME_SMALL_VISIBLE_FOOTPRINT_MAX = 0.18
KEYFRAME_SMALL_VISIBLE_FOOTPRINT_CAP = 60.0
KEYFRAME_TINY_VISIBLE_FOOTPRINT_MAX = 0.06
KEYFRAME_TINY_VISIBLE_FOOTPRINT_CAP = 30.0


def normalize_target_instance_scope(scope: str | None) -> str:
    """Normalize instruction target scope to ``single`` or ``multiple``."""
    text = str(scope or "single").strip().lower()
    return "multiple" if text == "multiple" else "single"


def format_target_instance_scope_line(scope: str | None) -> str:
    """Human-readable detection/edit constraint for one instruction."""
    if normalize_target_instance_scope(scope) == "multiple":
        return (
            "target_instance_scope: multiple — edit/detection applies to EVERY person/object in the "
            "frame that matches subject_features (all instances with the same distinguishing features)."
        )
    return (
        "target_instance_scope: single — match exactly ONE specific tracked person/object instance "
        "(the same individual in the reference), NOT every look-alike with similar clothing or features."
    )


_PARTIAL_LIMB_PARTS = frozenset({
    "arm",
    "hand",
    "wrist",
    "elbow",
    "shoulder",
    "leg",
    "foot",
    "ankle",
    "knee",
    "limb",
})
_IDENTITY_CORE_PARTS = frozenset({
    "face",
    "head",
    "hair",
    "full_body",
    "full_figure",
    "torso",
    "upper_body",
    "body",
    "back",
    "neck",
})
_REMOVAL_IDENTITY_CLOTHING_PARTS = frozenset({
    "vest",
    "cap",
    "hat",
    "torso",
    "face",
    "head",
    "body",
    "upper_body",
})
_EDGE_CROP_PHRASES = (
    "edge of the frame",
    "far left",
    "far right",
    "left edge",
    "right edge",
    "partially cropped",
    "cropped at",
    "cut off at",
    "only partially visible",
)
_LIMB_PHRASES = (
    "only an arm",
    "only the arm",
    "arm only",
    "only a hand",
    "hand only",
    "only the shoulder",
    "shoulder only",
    "partial arm",
    "partial limb",
)
def _visible_parts_list(record: Dict[str, Any]) -> List[str]:
    raw = record.get("visible_parts") or []
    if isinstance(raw, str):
        raw = [raw]
    return _normalize_visible_parts_list([
        str(part).strip().lower() for part in raw if str(part).strip()
    ])


def _normalize_visible_parts_list(parts: List[str]) -> List[str]:
    """Canonicalize VLM visible_parts tokens."""
    normalized: List[str] = []
    for raw in parts:
        part = str(raw).strip().lower()
        if not part:
            continue
        if part in {"upper body", "upper_body", "upper torso", "upper_torso", "chest", "dress"}:
            token = "torso"
        elif part in {"face", "head", "facial"}:
            token = "face"
        elif part in {"hair", "hairstyle", "hair style"}:
            token = "hair"
        elif "cap" in part or "hat" in part or "headwear" in part:
            token = "cap"
        elif part in {"back", "back of head", "neck"}:
            token = part
        elif "shoulder" in part:
            token = "shoulder"
        elif any(word in part for word in ("arm", "hand", "wrist", "elbow")):
            token = "arm"
        elif part in _IDENTITY_CORE_PARTS or part in _PARTIAL_LIMB_PARTS:
            token = part
        else:
            token = part
        if token not in normalized:
            normalized.append(token)
    return normalized


_VALID_VISIBILITY_QUALITIES = frozenset({
    "clear",
    "too_small",
    "blurry",
    "ambiguous",
    "occluded",
})


def _sanitize_vlm_location_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """Fix common VLM field mistakes before presence gating."""
    out = dict(record)
    visibility = str(out.get("visibility_quality", "") or "").strip().lower()
    clarity = str(out.get("localization_clarity", "") or "").strip().lower()
    completeness = str(
        out.get("entity_visibility_completeness", "") or ""
    ).strip().lower()

    if visibility in {"high", "good", "excellent", "strong"}:
        visibility = "clear"
    elif visibility in {"low", "poor", "none", "missing"}:
        visibility = "ambiguous"
    elif visibility and visibility not in _VALID_VISIBILITY_QUALITIES:
        visibility = "clear" if clarity == "high" else "ambiguous"

    if clarity in {"medium", "moderate", "fair", "ok"}:
        clarity = "medium"
    elif clarity in {"high", "clear", "strong", "good", "excellent"}:
        clarity = "high"
    elif clarity in {"", "none", "missing", "low", "poor"}:
        clarity = "low"

    if completeness in {"none", "missing", ""}:
        completeness = "fragment"
    elif completeness in {
        "high",
        "full",
        "fully visible",
        "fully_visible",
        "complete",
        "whole",
        "full_body",
    }:
        completeness = KEYFRAME_ENTITY_REQUIRED_COMPLETENESS

    out["visibility_quality"] = visibility
    out["localization_clarity"] = clarity
    out["entity_visibility_completeness"] = completeness
    out["visible_parts"] = _normalize_visible_parts_list(_visible_parts_list(out))
    return out


def _edit_targets_head_region(edit_prompt: str) -> bool:
    text = (edit_prompt or "").lower()
    return any(token in text for token in ("hair", "hat", "head", "facial", "cap"))


def _edit_targets_upper_body(edit_prompt: str) -> bool:
    """True when the edit targets shoulder/torso/chest (e.g. accessory placement)."""
    text = (edit_prompt or "").lower()
    return any(
        token in text
        for token in ("shoulder", "torso", "chest", "upper body", "upper_body", "suspender")
    )


def _edit_targets_removal(edit_prompt: str) -> bool:
    text = (edit_prompt or "").lower()
    return any(
        token in text
        for token in ("remove", "delete", "erase", "inpaint out", "inpainting")
    )


def _requested_attachment_point(edit_prompt: str) -> str:
    """Return a body-side placement point that must be visible to edit."""
    text = (edit_prompt or "").lower()
    if "shoulder" not in text:
        return ""
    if "left shoulder" in text:
        return "left_shoulder"
    if "right shoulder" in text:
        return "right_shoulder"
    return "shoulder"


def _attachment_visibility_text(record: Dict[str, Any]) -> str:
    return " ".join(
        str(record.get(field, "") or "")
        for field in (
            "attachment_visibility_reasoning",
            "reasoning",
            "initial_detection_reasoning",
            "location_description",
            "viewpoint",
        )
    ).lower()


def _attachment_visibility_from_text(record: Dict[str, Any], attachment: str) -> bool | None:
    text = _attachment_visibility_text(record)
    side = attachment.split("_", 1)[0] if "_" in attachment else ""
    if side:
        side_phrase = rf"(?:anatomical\s+)?{side}\s+shoulder"
        positive_patterns = (
            rf"{side_phrase}.{{0,120}}\b(visible|clearly visible|fully visible|unobstructed|available|accessible|exposed|editable)\b",
            rf"\b(visible|clearly visible|fully visible|unobstructed|available|accessible|exposed|editable)\b.{{0,120}}{side_phrase}",
            rf"\b(exposing|showing|revealing)\b.{{0,80}}{side_phrase}",
            rf"{side_phrase}.{{0,120}}\bfor\b.{{0,40}}\b(edit|placement)\b",
        )
        negative_patterns = (
            rf"{side_phrase}.{{0,60}}\b(not visible|is not visible|hidden|occluded|turned away|not available|ambiguous|cropped out)\b",
            rf"\b(not visible|hidden|occluded|turned away|not available|ambiguous|cropped out)\b.{{0,60}}{side_phrase}",
        )
    else:
        positive_patterns = (
            r"\bshoulder\b.{0,80}\b(visible|clearly visible|fully visible|unobstructed|available|accessible|exposed|editable)\b",
            r"\b(visible|clearly visible|fully visible|unobstructed|available|accessible|exposed|editable)\b.{0,80}\bshoulder\b",
        )
        negative_patterns = (
            r"\bshoulder\b.{0,40}\b(not visible|is not visible|hidden|occluded|turned away|not available|ambiguous|cropped out)\b",
            r"\b(not visible|hidden|occluded|turned away|not available|ambiguous|cropped out)\b.{0,40}\bshoulder\b",
        )
    if any(re.search(pattern, text) for pattern in negative_patterns):
        return False
    if any(re.search(pattern, text) for pattern in positive_patterns):
        return True
    return None


def _attachment_text_override_safe(record: Dict[str, Any], attachment: str) -> bool:
    """Allow correcting contradictory VLM structure only with strong local evidence."""
    if not attachment.endswith("_shoulder") and attachment != "shoulder":
        return False
    visible_parts = set(_visible_parts_list(record))
    if not (visible_parts & {"face", "head", "hair", "torso", "upper_body", "body", "back", "neck"}):
        return False
    if str(record.get("localization_clarity", "") or "").strip().lower() not in {"high", "medium"}:
        return False
    if not bool(record.get("identity_verifiable_from_visible_parts")):
        return False
    if (
        _specific_visual_identity_evidence_count(record) < 2
        and not _feature_audit_has_clear_match(record)
    ):
        return False
    if bool(record.get("cross_entity_conflict_suppressed")):
        return False
    return True


def _attachment_side_field_reliable(record: Dict[str, Any], attachment: str) -> bool:
    if "_" not in attachment:
        return True
    side = attachment.split("_", 1)[0]
    side_field = str(record.get(f"anatomical_{side}_screen_side", "") or "").strip().lower()
    if side_field and side_field not in {"unknown", "hidden", "occluded", "none", "n/a"}:
        return True
    # Relaxed: when side field is empty/unknown, trust strong entity identity
    # evidence rather than blocking the edit. The image model can determine
    # the correct shoulder from the edit prompt context.
    return _attachment_text_override_safe(record, attachment)


def _attachment_visible_from_record(record: Dict[str, Any], attachment: str) -> bool | None:
    """Interpret VLM attachment visibility fields.

    None means unknown; callers can decide whether to allow legacy records.
    """
    if not attachment:
        return None
    text_visible = _attachment_visibility_from_text(record, attachment)
    visibility = record.get("attachment_visibility")
    if isinstance(visibility, dict):
        if attachment in visibility:
            structured_visible = bool(visibility.get(attachment))
            if (
                text_visible is not False
                and not structured_visible
                and _attachment_text_override_safe(record, attachment)
            ):
                return True
            if text_visible is False and structured_visible:
                return False
            return structured_visible
        if attachment.endswith("_shoulder") and "shoulder" in visibility:
            structured_visible = bool(visibility.get("shoulder"))
            if (
                text_visible is not False
                and not structured_visible
                and _attachment_text_override_safe(record, attachment)
            ):
                return True
            if text_visible is False and structured_visible:
                return False
            return structured_visible

    key = f"{attachment}_visible"
    if key in record:
        structured_visible = bool(record.get(key))
        if text_visible is not False and not structured_visible and _attachment_text_override_safe(record, attachment):
            return True
        if text_visible is False and structured_visible:
            return False
        return structured_visible
    if attachment.endswith("_shoulder") and "target_shoulder_visible" in record:
        structured_visible = bool(record.get("target_shoulder_visible"))
        if text_visible is not False and not structured_visible and _attachment_text_override_safe(record, attachment):
            return True
        if text_visible is False and structured_visible:
            return False
        return structured_visible
    if "target_attachment_visible" in record:
        structured_visible = bool(record.get("target_attachment_visible"))
        if text_visible is not False and not structured_visible and _attachment_text_override_safe(record, attachment):
            return True
        if text_visible is False and structured_visible:
            return False
        return structured_visible

    return text_visible


def _attachment_strong_entity_bypass(record: Dict[str, Any], attachment: str) -> bool:
    """Allow physical placement edit even when attachment visibility is uncertain.

    This is a relaxed fallback: when the structured VLM field says the
    attachment point is not visible but the text doesn't explicitly say
    "hidden"/"not visible", and the entity is strongly identified (face
    + torso visible, high confidence, identity verifiable), allow the
    edit. The image model can infer the correct placement from context.
    """
    if not attachment.endswith("_shoulder") and attachment != "shoulder":
        return False
    visible_parts = set(_visible_parts_list(record))
    # Must have face/head/hair visible for reliable identity
    if not (visible_parts & {"face", "head", "hair"}):
        return False
    # Must have torso/shoulder/neck visible for placement context
    if not (visible_parts & {"torso", "upper_body", "body", "shoulder", "neck", "arm"}):
        return False
    if not bool(record.get("identity_verifiable_from_visible_parts")):
        return False
    if str(record.get("localization_clarity", "") or "").strip().lower() not in {"high", "medium"}:
        return False
    try:
        confidence = float(record.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        existence_score = float(record.get("existence_confidence_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        existence_score = 0.0
    if confidence < 0.8 or existence_score < 70.0:
        return False
    # Text must not explicitly say the attachment is hidden/not visible
    if _attachment_visibility_from_text(record, attachment) is False:
        return False
    return True


def _instruction_attachment_editable(record: Dict[str, Any], instr: EntityInstruction) -> bool:
    """False when a requested physical placement point is hidden or ambiguous.

    Physical placement edits are high-risk: if the model cannot reliably see
    the exact target side/attachment point in the current frame, it tends to
    place the object on a nearby person, the opposite shoulder, or copied
    scene-continuity location. Require current-frame attachment evidence rather
    than treating "unknown" as editable.
    """
    attachment = _requested_attachment_point(instr.edit_prompt)
    if not attachment:
        return True
    visible = _attachment_visible_from_record(record, attachment)
    if visible is not True:
        # Relaxed fallback: allow edit when entity is strongly identified
        # even if structured attachment visibility says False/None.
        if not _attachment_strong_entity_bypass(record, attachment):
            return False

    if attachment.endswith("_shoulder") and _edit_targets_physical_placement(instr.edit_prompt):
        visible_parts = _visible_parts_list(record)
        if not any(
            part in visible_parts
            for part in ("face", "head", "hair", "torso", "upper_body", "body", "back", "neck", "back of head")
        ):
            return False
        record_text = _normalize_detection_text(
            " ".join(
                str(record.get(key, "") or "")
                for key in (
                    "reasoning",
                    "initial_detection_reasoning",
                    "attachment_visibility_reasoning",
                    "location_description",
                )
            )
            + " "
            + " ".join(str(item) for item in (record.get("identity_cues") or []))
            + " "
            + " ".join(visible_parts)
        )
        subject_text = _normalize_detection_text(instr.subject_features or "")
        strong_subject_cues = [
            cue
            for cue in (
                "suspenders",
                "brown shirt",
                "middle-parted",
                "middle parted",
                "blond hair",
                "blonde hair",
            )
            if cue in subject_text
        ]
        if "suspenders" in subject_text:
            strong_identity_visible = "suspenders" in record_text
        else:
            strong_identity_visible = bool(strong_subject_cues) and any(
                cue in record_text for cue in strong_subject_cues
            )
        if "suspenders" in subject_text and not strong_identity_visible:
            if not bool(record.get("identity_verifiable_from_visible_parts")):
                return False
        if (
            not bool(record.get("identity_verifiable_from_visible_parts"))
            and not strong_identity_visible
        ):
            return False

        body_orientation = str(record.get("body_orientation", "") or "").lower()
        viewpoint = str(record.get("viewpoint", "") or "").lower()
        orientation_text = f"{body_orientation} {viewpoint}"
        if any(token in orientation_text for token in ("back", "rear", "turned away")):
            if not any(part in visible_parts for part in ("torso", "upper_body", "body", "back")):
                return False

        side = attachment.split("_", 1)[0]
        if not _attachment_side_field_reliable(record, attachment):
            return False

    return True


def _has_identity_core_visible(visible_parts: List[str]) -> bool:
    return any(part in _IDENTITY_CORE_PARTS for part in visible_parts)


def _has_key_visible_identity_region(
    record: Dict[str, Any],
    visible_parts: List[str],
    *,
    edit_prompt: str = "",
    subject_features: str = "",
) -> bool:
    """True when visible pixels include a region meaningful enough for identity matching."""
    parts = set(visible_parts)
    if parts & {"face", "head", "hair", "cap", "hat"}:
        return True
    if not parts or all(part in _PARTIAL_LIMB_PARTS for part in parts):
        return False

    substantial_body_parts = {
        "torso",
        "upper_body",
        "body",
        "back",
        "neck",
        "full_body",
        "full_figure",
    }
    if not (parts & substantial_body_parts):
        return False
    if bool(record.get("identity_verifiable_from_visible_parts")):
        return True
    return (
        _has_clothing_identity_cues(
            record,
            visible_parts,
            subject_features=subject_features,
        )
        or _has_removal_identity_cues(
            record,
            visible_parts,
            edit_prompt=edit_prompt,
            subject_features=subject_features,
        )
    )


def _calibrate_existence_confidence_score(
    data: Dict[str, Any],
    visible_parts: List[str],
    score: float,
) -> tuple[float, List[str]]:
    """Cap VLM existence scores when only weak identity regions are visible."""
    if score <= 0:
        return 0.0, []

    if _feature_audit_clear_mismatch_reasons(data) or _clear_identity_conflict_reasons(data):
        return 0.0, ["visible_feature_clear_mismatch"]

    parts = set(visible_parts)
    completeness = str(data.get("entity_visibility_completeness", "") or "").lower()
    visibility = str(data.get("visibility_quality", "") or "").lower()
    identity_verifiable = bool(data.get("identity_verifiable_from_visible_parts"))
    evidence_text_parts = [
        str(data.get("location_description", "") or ""),
        str(data.get("reasoning", "") or ""),
        str(data.get("attachment_visibility_reasoning", "") or ""),
        str(data.get("visibility_quality", "") or ""),
        str(data.get("entity_visibility_completeness", "") or ""),
    ]
    raw_candidates = data.get("candidate_evaluations") or []
    if isinstance(raw_candidates, list):
        for item in raw_candidates:
            if isinstance(item, dict):
                evidence_text_parts.extend([
                    str(item.get("candidate_location", "") or ""),
                    " ".join(str(part) for part in (item.get("visible_parts") or [])),
                    " ".join(str(match) for match in (item.get("identity_matches") or [])),
                    " ".join(str(conflict) for conflict in (item.get("identity_conflicts") or [])),
                ])
    evidence_text = " ".join(evidence_text_parts).lower()
    cap = 100.0
    reasons: List[str] = []
    # Lighting, expression, gaze, pose, and action are identity-matching
    # variations, not visibility failures. They intentionally do not cap score.

    core_face_head_hair = {"face", "head", "hair"}
    has_core_identity = bool(parts & core_face_head_hair)
    has_headwear = bool(parts & {"cap", "hat"})
    substantial_body = {
        "torso",
        "upper_body",
        "body",
        "back",
        "neck",
        "full_body",
        "full_figure",
    }
    area_fraction = float(data.get("approximate_area_fraction", 0.0) or 0.0)
    specific_count = _specific_visual_identity_evidence_count(data)
    candidate_conflicts = [
        conflict
        for conflict in _candidate_identity_conflicts(data)
        if not _conflict_is_uncertain_or_blurry(conflict)
    ]
    robust_identity = (
        not candidate_conflicts
        and (
            _feature_audit_has_clear_match(data)
            or _record_matches_face_or_hair_identity(data)
            or (
                specific_count >= 2
                and (
                    identity_verifiable
                    or _candidate_has_present_decision(data)
                    or _feature_audit_only_uncertain_non_matches(data)
                )
            )
        )
    )
    direct_face_identity = _direct_face_identity_ok(data)
    back_view_only = _back_view_only_identity(data, visible_parts)

    if not parts:
        cap = min(cap, 20.0)
        reasons.append("no_visible_identity_parts")
    elif all(part in _PARTIAL_LIMB_PARTS for part in parts):
        cap = min(cap, 35.0)
        reasons.append("limb_or_shoulder_only")
    elif not has_core_identity:
        if has_headwear and parts & substantial_body:
            cap = min(cap, 82.0 if robust_identity else 75.0)
            reasons.append("headwear_and_body_without_face_hair")
        elif parts & substantial_body:
            cap = min(cap, 78.0 if robust_identity else 65.0)
            reasons.append("body_or_clothing_without_face_head_hair")
        else:
            cap = min(cap, 55.0 if robust_identity else 45.0)
            reasons.append("non_core_identity_region_only")

    if completeness == "fragment":
        cap = min(cap, 55.0 if robust_identity else 40.0)
        reasons.append("fragment_visibility")
    elif completeness == "partial" and not has_core_identity:
        cap = min(cap, 78.0 if robust_identity else 65.0)
        reasons.append("partial_without_core_identity")
    elif completeness == "partial":
        cap = min(cap, 88.0 if robust_identity else 75.0)
        reasons.append("partial_body_visible")

    if visibility in {"too_small", "blurry", "ambiguous", "occluded"}:
        if robust_identity:
            relaxed_cap = {
                "too_small": 80.0,
                "blurry": 76.0,
                "ambiguous": 72.0,
                "occluded": 74.0,
            }.get(visibility, 80.0)
            if not direct_face_identity:
                relaxed_cap -= 6.0
            cap = min(cap, relaxed_cap)
            reasons.append(f"visibility_quality={visibility}_but_identity_specific")
        else:
            cap = min(cap, 68.0)
            reasons.append(f"visibility_quality={visibility}")

    # A small on-screen footprint means only a fragment of the body is visible
    # (e.g. a single shoulder poking into frame). Lower the existence score so a
    # sliver cannot be scored as confidently as a fully framed subject.
    # This penalty only applies when core identity parts (face / head / hair) are
    # NOT visible. A person who is small but has face+hair clearly visible (e.g. a
    # distant background figure) should not be penalized by area alone.
    if not has_core_identity:
        if area_fraction <= KEYFRAME_TINY_VISIBLE_FOOTPRINT_MAX:
            cap = min(cap, 48.0 if robust_identity else KEYFRAME_TINY_VISIBLE_FOOTPRINT_CAP)
            reasons.append(
                "tiny_visible_footprint_identity_supported"
                if robust_identity else "tiny_visible_footprint"
            )
        elif area_fraction <= KEYFRAME_SMALL_VISIBLE_FOOTPRINT_MAX:
            cap = min(cap, 60.0 if robust_identity else KEYFRAME_SMALL_VISIBLE_FOOTPRINT_CAP)
            reasons.append(
                "small_visible_footprint_identity_supported"
                if robust_identity else "small_visible_footprint"
            )

    if any(
        token in evidence_text
        for token in (
            "foreground blur",
            "out of focus",
            "out-of-focus",
            "defocus",
            "too blurred to verify",
            "too blurry to verify",
            "identity is blurred",
            "face is blurred",
            "head is blurred",
            "soft focus",
            "soft-focus",
            "shallow depth of field",
            "shallow depth-of-field",
        )
    ):
        cap = min(cap, 88.0 if robust_identity else 80.0)
        reasons.append("blur_but_identity_specific" if robust_identity else "identity_reducing_blur")

    if any(
        token in evidence_text
        for token in (
            "partially occluded",
            "heavily occluded",
            "occluded by",
            "obscured by",
            "blocked by",
            "hidden by",
            "covered by",
            "partially hidden",
            "partially obscured",
            "partially blocked",
            "foreground person",
            "foreground figure",
            "foreground object",
            "foreground blur",
            "hand in front",
            "partially overlapping",
            "railing in front",
            "pillar in front",
            "prop in front",
            "shadowed",
            "backlit",
            "back-lit",
        )
    ):
        cap = min(
            cap,
            90.0 if robust_identity and has_core_identity else
            80.0 if robust_identity else
            85.0 if has_core_identity else 70.0,
        )
        reasons.append("non_target_occlusion_or_obstruction")

    if back_view_only:
        cap = min(cap, 62.0 if direct_face_identity else 48.0)
        reasons.append(
            "back_view_without_direct_face_identity"
            if not direct_face_identity else "back_view_identity_limited"
        )

    if not identity_verifiable and not has_core_identity:
        cap = min(cap, 58.0 if robust_identity else 42.0)
        reasons.append(
            "identity_not_verifiable_but_supported_by_specific_cues"
            if robust_identity else "identity_not_verifiable_without_core_identity"
        )

    # Cross-check: when location_description describes an edge-crop / fragment
    # region but visible_parts claim face/hair, the VLM likely hallucinated
    # core identity at a position where it cannot exist. Penalize aggressively.
    location_text = str(data.get("location_description", "") or "").lower()
    edge_crop_tokens = (
        "edge", "corner", "cropped", "sliver", "cut off",
        "partially visible", "far left edge", "extreme left", "extreme right",
        "bottom corner", "top corner", "frame edge", "edge of the frame",
        "extreme far left", "extreme far right", "only visible at",
        "partially cut off",
    )
    if has_core_identity and any(token in location_text for token in edge_crop_tokens):
        cap = min(cap, 40.0)
        reasons.append("edge_crop_location_contradicts_core_identity")

    # Feature matching score boost: raise scores and caps when target features explicitly match
    audit_entries = _feature_audit_entries(data)
    num_matches = sum(1 for entry in audit_entries if entry.get("status") == "match")
    if num_matches > 0:
        # Give +5 points bonus for each matched feature in feature_audit
        bonus = float(num_matches) * 5.0
        score = min(100.0, score + bonus)
        # Also give the cap some breathing room to allow the bonus score to flow through
        cap = min(100.0, cap + bonus * 0.5)
        reasons.append(f"feature_match_bonus_plus_{int(bonus)}")

    if cap >= 100.0:
        cap = 95.0
        reasons.append("avoid_extreme_present_score")

    calibrated = min(score, cap)
    return calibrated, reasons if calibrated < score else []


_CLOTHING_IDENTITY_CUES = (
    "vest",
    "flat cap",
    "cap",
    "railing",
    "stubble",
    "suspenders",
    "brown shirt",
    "shirt",
    "dress",
    "hat",
    "hair",
    "color-blocked",
    "yellow-and-white",
)

_HEADWEAR_CUES = ("hat", "cap", "flat cap", "headwear", "fedora", "beret")
_HAIR_IDENTITY_CUES = ("hair", "blond", "blonde", "middle-parted", "middle parted")
_SPECIFIC_HAIR_IDENTITY_TOKENS = (
    "updo",
    "hairline",
    "hair line",
    "middle-part",
    "middle parted",
    "side-part",
    "side parted",
    "parted hair",
    "bun",
    "chignon",
    "braid",
    "braided",
    "bangs",
    "fringe",
    "curly hair",
    "wavy hair",
    "straight hair",
    "loose hair",
    "short hair",
    "long hair",
    "hairstyle",
    "hair style",
)
_CONCRETE_FACE_IDENTITY_TOKENS = (
    "matching face",
    "matches face",
    "matches the face",
    "face shape",
    "face profile",
    "profile shape",
    "eye shape",
    "eyes",
    "eyebrow",
    "jaw",
    "jawline",
    "nose",
    "mouth",
    "lips",
    "cheek",
    "chin",
    "forehead",
)
_STRONG_SUBJECT_FEATURE_CUES = (
    "suspenders",
    "vest",
    "flat cap",
    "cap",
    "hat",
    "dress",
    "shirt",
    "beard",
    "stubble",
    "color-blocked",
    "yellow-and-white",
    "middle-parted",
    "blond",
    "blonde",
)

_NON_FRONTAL_VIEWPOINT_CUES = (
    "side",
    "side_profile",
    "side profile",
    "profile",
    "three_quarter",
    "three quarter",
    "three-quarter",
    "3/4",
    "3/4 profile",
    "back",
    "back view",
    "rear",
    "rear view",
    "turned away",
    "turned sideways",
    "head turned",
    "not frontal",
)

_WIDE_SHOT_CUES = (
    "wide",
    "wide shot",
    "long shot",
    "long-shot",
    "full shot",
    "distant",
    "distant figure",
    "small",
    "small figure",
    "tiny",
    "far",
    "far away",
    "background",
    "background figure",
    "midground",
    "mid-ground",
    "full body",
    "full-body",
)

_CLOSE_SHOT_CUES = (
    "close",
    "close-up",
    "closeup",
    "extreme close-up",
    "tight close-up",
    "tight crop",
    "near camera",
    "foreground",
    "fills the frame",
    "fills most of the frame",
    "large face",
    "large head",
    "cropped face",
    "cropped head",
    "close cropped",
)

_CHALLENGING_VISIBILITY_CUES = (
    "soft focus",
    "soft-focus",
    "shallow depth of field",
    "shallow depth-of-field",
    "low light",
    "dim light",
    "dimly lit",
    "dark lighting",
    "shadow",
    "shadowed",
    "in shadow",
    "backlit",
    "back-lit",
    "silhouette",
    "silhouetted",
    "underexposed",
    "low exposure",
    "motion blur",
)


def _has_clothing_identity_cues(
    record: Dict[str, Any],
    visible_parts: List[str],
    *,
    subject_features: str = "",
) -> bool:
    """Distinctive clothing/build cues suffice for partial torso/shoulder edits."""
    identity_verifiable = bool(record.get("identity_verifiable_from_visible_parts", False))
    if any(part in _REMOVAL_IDENTITY_CLOTHING_PARTS for part in visible_parts):
        return identity_verifiable or _record_matches_required_subject_cues(
            record,
            subject_features=subject_features,
        ) >= 1
    text = " ".join(
        str(record.get(key, "") or "")
        for key in (
            "location_description",
            "vlm_location_description",
            "reasoning",
            "initial_detection_reasoning",
        )
    ).lower()
    matched_cues = sum(
        1
        for cue in _CLOTHING_IDENTITY_CUES
        if cue in (subject_features or "").lower() and cue in text
    )
    return identity_verifiable or matched_cues >= 2


def _identity_cues_list(record: Dict[str, Any]) -> List[str]:
    raw = record.get("identity_cues") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(cue).strip().lower() for cue in raw if str(cue).strip()]


def _record_identity_text(record: Dict[str, Any]) -> str:
    """Identity-related text returned by VLM, normalized for heuristic gates."""
    cues = " ".join(_identity_cues_list(record))
    parts = " ".join(_visible_parts_list(record))
    initial_parts = " ".join(
        _normalize_visible_parts_list([
            str(part)
            for part in (record.get("initial_detection_visible_parts") or [])
            if str(part).strip()
        ])
    )
    fields = " ".join(
        str(record.get(key, "") or "")
        for key in (
            "location_description",
            "vlm_location_description",
            "reasoning",
            "initial_detection_reasoning",
            "attachment_visibility_reasoning",
        )
    )
    return f"{cues} {parts} {initial_parts} {fields}".lower()


_WEAK_IDENTITY_EVIDENCE_TOKENS = (
    "scene role",
    "story role",
    "narrative",
    "prominent",
    "primary subject",
    "main subject",
    "central subject",
    "engagement in conversation",
    "conversation",
    "expression",
    "facial expression",
    "gaze",
    "looking away",
    "pose",
    "posture",
    "action",
    "gesture",
    "head angle",
    "head tilt",
    "smile",
    "smiling",
    "frown",
    "frowning",
    "laughing",
    "crying",
    "talking",
    "speaking",
    "walking",
    "running",
    "leaning",
    "bending",
    "crouching",
    "seated",
    "sitting",
    "standing",
)

_GENERIC_IDENTITY_CUES = {
    "facial structure",
    "face structure",
    "facial identity",
    "facial features",
    "body identity",
    "same identity",
    "matches identity",
    "same person",
    "same individual",
    "correct identity",
    "scene role",
    "story role",
    "expression",
    "facial expression",
    "gaze",
    "pose",
    "posture",
    "action",
    "gesture",
    "head angle",
    "head tilt",
}


def _candidate_identity_match_text(record: Dict[str, Any]) -> str:
    """Positive candidate-level identity evidence from VLM audits.

    Keep this limited to ``identity_matches``. Conflict/reject text often names
    the same visual categories ("wrong face", "hair conflict") and must not be
    counted as positive identity evidence by the presence gate.
    """
    matches: List[str] = []
    raw = record.get("candidate_evaluations") or []
    if not isinstance(raw, list):
        return ""
    for item in raw:
        if not isinstance(item, dict):
            continue
        value = item.get("identity_matches")
        if isinstance(value, list):
            matches.extend(str(v) for v in value if str(v).strip())
        elif value:
            matches.append(str(value))
    return " ".join(matches).lower()


def _candidate_identity_conflicts(record: Dict[str, Any]) -> List[str]:
    conflicts: List[str] = []
    raw = record.get("candidate_evaluations") or []
    if not isinstance(raw, list):
        return conflicts
    for item in raw:
        if not isinstance(item, dict):
            continue
        value = item.get("identity_conflicts")
        if isinstance(value, list):
            conflicts.extend(str(v).strip() for v in value if str(v).strip())
        elif value:
            text = str(value).strip()
            if text:
                conflicts.append(text)
    return conflicts


def _candidate_has_present_decision(record: Dict[str, Any]) -> bool:
    decisions = [
        str(item.get("decision", "") or "").lower()
        for item in (record.get("candidate_evaluations") or [])
        if isinstance(item, dict)
    ]
    return any(
        decision in {"present", "match", "matched", "same", "recover_present"}
        or "present" in decision
        for decision in decisions
    )


def _conflict_is_uncertain_or_blurry(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "uncertain",
            "unclear",
            "blurry",
            "blurred",
            "ambiguous",
            "occluded",
            "hidden",
            "not visible",
            "cannot verify",
            "cannot be verified",
            "not reliably visible",
            "too small",
            "off-frame",
            "off frame",
            "partially visible",
        )
    )


def _clear_identity_conflict_reasons(record: Dict[str, Any]) -> List[str]:
    """Reject only concrete visible contradictions; ignore blurry/uncertain conflicts."""
    reasons: List[str] = []
    for conflict in _candidate_identity_conflicts(record):
        lowered = conflict.lower()
        if _conflict_is_uncertain_or_blurry(lowered):
            continue
        if lowered.startswith("clear:"):
            reasons.append("clear_identity_conflict")
            continue
        if any(
            token in lowered
            for token in (
                "wrong ",
                "different ",
                "does not match",
                "do not match",
                "not match",
                "mismatch",
                "conflict",
                "contradict",
                "inconsistent",
                "not the same",
                "look-alike",
                "lookalike",
            )
        ):
            reasons.append("clear_identity_conflict")
    return list(dict.fromkeys(reasons))


_FEATURE_AUDIT_MATCH_STATUSES = frozenset(
    {"match", "matches", "matching", "matched", "same", "consistent"}
)
_FEATURE_AUDIT_CLEAR_MISMATCH_STATUSES = frozenset(
    {
        "clear_mismatch",
        "mismatch",
        "conflict",
        "contradiction",
        "wrong",
        "clear mismatch",
        "visible_mismatch",
        "does_not_match",
    }
)
_FEATURE_AUDIT_UNCERTAIN_STATUSES = frozenset(
    {
        "uncertain_blurry",
        "uncertain",
        "blurry",
        "not_visible",
        "not_verifiable",
        "occluded",
        "hidden",
        "off_frame",
        "cannot_verify",
        "cannot_be_verified",
        "not_reliably_visible",
    }
)


def _normalize_feature_audit_status(status: str) -> str:
    """Map VLM feature_audit status strings to canonical buckets."""
    normalized = (
        str(status or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if not normalized:
        return "unknown"
    if normalized in _FEATURE_AUDIT_MATCH_STATUSES:
        return "match"
    if normalized in _FEATURE_AUDIT_CLEAR_MISMATCH_STATUSES:
        return "clear_mismatch"
    if normalized in _FEATURE_AUDIT_UNCERTAIN_STATUSES:
        return "uncertain_blurry"
    if normalized in {"not_visible", "not_visible_in_frame", "absent", "off_frame"}:
        return "not_visible"
    if normalized.startswith("clear") and "mismatch" in normalized:
        return "clear_mismatch"
    if _conflict_is_uncertain_or_blurry(normalized):
        return "uncertain_blurry"
    if any(
        token in normalized
        for token in (
            "mismatch",
            "conflict",
            "contradict",
            "wrong",
            "different",
            "not_match",
            "does_not_match",
        )
    ):
        return "clear_mismatch"
    if any(token in normalized for token in ("match", "same", "consistent")):
        return "match"
    return normalized


def _feature_audit_entries(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = record.get("feature_audit")
    if not isinstance(raw, list):
        return []
    entries: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        feature = str(item.get("feature") or item.get("feature_name") or "").strip()
        if not feature:
            continue
        status = _normalize_feature_audit_status(
            str(item.get("status") or item.get("audit_status") or "")
        )
        note = str(item.get("note") or item.get("reasoning") or "").strip()
        entries.append(
            {
                "feature": feature,
                "status": status,
                "note": note,
            }
        )
    return entries


def _feature_audit_clear_mismatch_reasons(record: Dict[str, Any]) -> List[str]:
    """Reject when structured feature audit reports any clearly visible contradiction."""
    entries = _feature_audit_entries(record)
    if not entries:
        return []
    clear_mismatches = [
        entry
        for entry in entries
        if entry.get("status") == "clear_mismatch"
    ]
    if not clear_mismatches:
        return []
    return ["visible_feature_clear_mismatch"]


def _identity_contradiction_reasons(record: Dict[str, Any]) -> List[str]:
    """One clear visible feature mismatch rejects; blurry-only mismatches do not."""
    reasons = _feature_audit_clear_mismatch_reasons(record)
    if reasons:
        return reasons
    return _clear_identity_conflict_reasons(record)


def _feature_audit_has_clear_match(record: Dict[str, Any]) -> bool:
    """True when feature audit cites at least one clearly matching identity feature."""
    entries = _feature_audit_entries(record)
    if not entries:
        return False
    match_count = sum(1 for entry in entries if entry.get("status") == "match")
    if match_count >= 2:
        return True
    if match_count == 1:
        for entry in entries:
            if entry.get("status") != "match":
                continue
            feature = str(entry.get("feature", "") or "").lower()
            if any(
                token in feature
                for token in (
                    "face",
                    "hair",
                    "head",
                    "profile",
                    "jaw",
                    "nose",
                    "eye",
                    "beard",
                    "hairline",
                )
            ):
                return True
    return False


def _feature_audit_only_uncertain_non_matches(record: Dict[str, Any]) -> bool:
    """True when every non-match in feature_audit is blurry/uncertain/not visible."""
    entries = _feature_audit_entries(record)
    if not entries:
        return False
    for entry in entries:
        status = str(entry.get("status", "") or "")
        if status == "match":
            continue
        if status not in {"uncertain_blurry", "not_visible", "unknown"}:
            return False
    return any(entry.get("status") == "match" for entry in entries)


def _positive_identity_evidence_text(record: Dict[str, Any]) -> str:
    """Positive identity evidence only; excludes reasoning/reject text."""
    pieces: List[str] = []
    pieces.extend(_identity_cues_list(record))
    pieces.append(_candidate_identity_match_text(record))
    return " ".join(str(piece) for piece in pieces if str(piece).strip()).lower()


def _merge_unique_strings(*values: Any) -> List[str]:
    merged: List[str] = []
    for value in values:
        raw_items = value if isinstance(value, list) else [value]
        for item in raw_items:
            text = str(item).strip()
            if text and text not in merged:
                merged.append(text)
    return merged


def _merge_candidate_evaluations(*values: Any) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            key = json.dumps(item, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                merged.append(item)
    return merged


def _specific_visual_identity_evidence_count(record: Dict[str, Any]) -> int:
    """Count concrete visual identity categories, ignoring narrative-only cues."""
    cue_text = " ".join(
        cue
        for cue in _identity_cues_list(record)
        if cue not in _GENERIC_IDENTITY_CUES
    )
    candidate_text = _candidate_identity_match_text(record)
    text = f"{cue_text} {candidate_text}".lower()
    visible_parts = set(_visible_parts_list(record))
    categories = set()
    if any(
        token in text
        for token in (
            "hair silhouette",
            "matching hair silhouette",
            "hairline",
            "matching hairline",
            "hair style",
            "hair color",
            "hair colour",
            "hair color/style",
            "hair colour/style",
            "hairstyle",
            "hair part",
            "hair parting",
            "updo",
            "curly hair",
            "wavy hair",
            "straight hair",
            "middle-part",
            "middle parted",
            "middle-parted hair",
            "middle parted hair",
            "blond hair",
            "blonde hair",
            "auburn",
            "reddish",
            "red hair",
            "brown hair",
            "black hair",
            "grey hair",
            "gray hair",
            "white hair",
        )
    ):
        categories.add("hair")
    if any(
        token in text
        for token in (
            "face shape",
            "facial structure",
            "facial features",
            "matching face",
            "matches the face",
            "matches reference face",
            "matches the reference face",
            "match face",
            "matches face",
            "face identity",
            "face and hair",
            "hair and facial features",
            "hair and face",
            "face profile",
            "facial profile",
            "matching face profile",
            "matching profile",
            "profile matches",
            "reference profile",
            "side profile",
            "three-quarter profile",
            "profile shape",
            "jaw",
            "jawline",
            "nose",
            "eyes",
            "eye shape",
            "eyebrow",
            "mouth",
            "lips",
            "cheek",
            "chin",
            "forehead",
        )
    ):
        categories.add("face")
    elif (
        visible_parts & {"face", "head"}
        and any(token in candidate_text for token in ("facial structure", "facial features"))
    ):
        categories.add("face")
    elif re.search(r"\bmatch(?:es|ing)?\b.{0,48}\bface\b", text) or re.search(
        r"\bface\b.{0,48}\bmatch(?:es|ing)?\b", text
    ):
        categories.add("face")
    elif "facial structure" in text and any(token in text for token in ("match", "matches", "matching")):
        categories.add("face")
    elif "facial features" in text and any(token in text for token in ("match", "matches", "matching")):
        categories.add("face")
    elif (
        candidate_text
        and "facial structure" in candidate_text
        and any(token in candidate_text for token in ("hair", "updo", "hairstyle", "hair silhouette"))
    ):
        categories.add("face")
    if any(
        token in text
        for token in (
            "flat cap",
            "flat_cap",
            "newsboy cap",
            "newsboy_cap",
            "hat",
            "cap",
            "vest",
            "suspenders",
            "beard",
            "stubble",
            "mustache",
            "moustache",
            "dress",
            "yellow-and-white",
            "color-blocked",
            "distinctive accessory",
            "brooch",
            "necklace",
        )
    ):
        categories.add("accessory_or_clothing")
    if any(
        token in text
        for token in (
            "body build",
            "body shape",
            "upper-body shape",
            "head shape",
            "silhouette",
            "outline",
            "frame shape",
            "shoulder line",
        )
    ):
        categories.add("body")
    return len(categories)


def _physical_placement_identity_reasons(
    record: Dict[str, Any],
    *,
    edit_prompt: str = "",
    subject_features: str = "",
) -> List[str]:
    """Reject placement edits when identity is too weak to safely attach objects."""
    if not _edit_targets_physical_placement(edit_prompt):
        return []
    if not (bool(record.get("present")) or bool(record.get("vlm_present"))):
        return []

    visible_parts = set(_visible_parts_list(record))
    positive_text = _positive_identity_evidence_text(record)
    subject = (subject_features or "").lower()
    specific_count = _specific_visual_identity_evidence_count(record)
    face_or_hair_identity_ok = (
        _record_matches_face_or_hair_identity(record)
        or _state_change_robust_identity_ok(record)
        or _close_shot_identity_ok(record, list(visible_parts), subject_features=subject_features)
    )
    reasons: List[str] = []

    if "suspenders" in subject and "suspenders" not in positive_text and not face_or_hair_identity_ok:
        reasons.append("missing_required_suspenders_for_placement")

    has_face = "face" in visible_parts
    has_head_or_hair = bool(visible_parts & {"head", "hair"})
    if not has_face and has_head_or_hair and specific_count < 2:
        reasons.append("physical_placement_identity_not_strong_enough")

    attachment = _requested_attachment_point(edit_prompt)
    if attachment:
        visible = _attachment_visible_from_record(record, attachment)
        if (visible is not True or not _attachment_side_field_reliable(record, attachment)) and not _attachment_strong_entity_bypass(record, attachment):
            reasons.append("physical_placement_attachment_not_reliably_editable")

    identity_reasons = [
        reason
        for reason in reasons
        if reason != "physical_placement_attachment_not_reliably_editable"
    ]
    if identity_reasons and any(
        token in str(record.get("initial_detection_reasoning", "") or "").lower()
        for token in ("not visible", "not present", "foreground figure", "different", "wrong")
    ):
        reasons.append("physical_placement_conflicts_with_initial_identity_reject")

    return list(dict.fromkeys(reasons))


def _state_change_robust_identity_ok(record: Dict[str, Any]) -> bool:
    """True when stable visible identity evidence exists despite action/expression drift."""
    if not bool(record.get("identity_verifiable_from_visible_parts")):
        return False
    visible_parts = set(_visible_parts_list(record))
    if not (visible_parts & {"face", "head", "hair", "torso", "upper_body", "body", "back", "neck"}):
        return False
    if _feature_audit_has_clear_match(record):
        return True
    if _record_matches_face_or_hair_identity(record):
        return True
    candidate_text = _candidate_identity_match_text(record)
    return bool(candidate_text and _specific_visual_identity_evidence_count(record) >= 1)


def _weak_identity_evidence_reasons(record: Dict[str, Any]) -> List[str]:
    if not bool(record.get("present")) and not bool(record.get("vlm_present")):
        return []
    if bool(record.get("catalog_trusted")):
        return []
    if not bool(record.get("identity_verifiable_from_visible_parts")):
        return []
    if not (set(_visible_parts_list(record)) & {"face", "head", "hair"}):
        return []
    specific_count = _specific_visual_identity_evidence_count(record)
    if specific_count >= 2:
        return []
    if _state_change_robust_identity_ok(record):
        return []
    confidence = float(record.get("confidence", 0.0) or 0.0)
    candidate_text = _candidate_identity_match_text(record)
    candidate_decisions = [
        str(item.get("decision", "") or "").lower()
        for item in (record.get("candidate_evaluations") or [])
        if isinstance(item, dict)
    ]
    candidate_conflicts = [
        conflict
        for item in (record.get("candidate_evaluations") or [])
        if isinstance(item, dict)
        for conflict in (item.get("identity_conflicts") or [])
        if str(conflict).strip()
    ]
    visible_parts = set(_visible_parts_list(record))
    has_visible_face_or_hair = bool(visible_parts & {"face", "head", "hair"})
    has_present_candidate = any(
        decision in {"present", "match", "matched", "same"} or "present" in decision
        for decision in candidate_decisions
    )
    if (
        confidence >= 0.95
        and has_visible_face_or_hair
        and has_present_candidate
        and candidate_text
        and not candidate_conflicts
    ):
        return []
    text = _record_identity_text(record)
    cue_text = " ".join(_identity_cues_list(record))
    weak_story_only = any(token in text for token in _WEAK_IDENTITY_EVIDENCE_TOKENS)
    generic_only = any(cue in cue_text for cue in _GENERIC_IDENTITY_CUES)
    if weak_story_only or generic_only or specific_count == 0:
        return ["weak_identity_evidence_without_specific_visual_match"]
    return []


def _head_edit_identity_reasons(
    record: Dict[str, Any],
    *,
    edit_prompt: str = "",
) -> List[str]:
    """Extra identity strictness before editing hair/headwear on a detected person."""
    if not _edit_targets_head_region(edit_prompt):
        return []
    if _edit_targets_physical_placement(edit_prompt):
        return []
    if not (bool(record.get("present")) or bool(record.get("vlm_present"))):
        return []

    visible_parts = set(_visible_parts_list(record))
    initial_parts = set(_normalize_visible_parts_list([
        str(part)
        for part in (record.get("initial_detection_visible_parts") or [])
        if str(part).strip()
    ]))
    if not (visible_parts | initial_parts) & {"face", "head", "hair"}:
        return ["head_edit_target_head_or_face_not_visible"]
    if (
        bool(record.get("catalog_trusted"))
        and bool(record.get("identity_verifiable_from_visible_parts"))
        and (visible_parts | initial_parts) & {"face", "head", "hair"}
    ):
        return []

    positive_text = " ".join(
        str(piece)
        for piece in (
            _positive_identity_evidence_text(record),
            record.get("initial_detection_reasoning", ""),
            " ".join(str(part) for part in initial_parts),
        )
        if str(piece).strip()
    ).lower()
    candidate_text = _candidate_identity_match_text(record)
    has_specific_hair = any(token in positive_text for token in _SPECIFIC_HAIR_IDENTITY_TOKENS)
    has_concrete_face = any(token in positive_text for token in _CONCRETE_FACE_IDENTITY_TOKENS)
    has_named_face_match = bool(
        re.search(r"\bmatch(?:es|ing)?\s+[a-z][a-z' -]{1,32}\s+face\b", positive_text)
        or re.search(r"\bmatch(?:es|ing)?\s+[a-z][a-z' -]{1,32}'s\s+face\b", positive_text)
    )
    has_concrete_face = has_concrete_face or has_named_face_match

    has_generic_hair = any(
        token in positive_text
        for token in (
            "hair color",
            "hair colour",
            "blonde hair",
            "blond hair",
            "brown hair",
            "black hair",
            "red hair",
            "hair silhouette",
            "color/silhouette",
        )
    )
    has_generic_face = any(
        token in positive_text
        for token in ("facial structure", "face structure", "facial identity")
    )
    if has_specific_hair or has_concrete_face:
        return []
    if has_generic_hair or has_generic_face or candidate_text:
        confidence = float(record.get("confidence", 0.0) or 0.0)
        visibility = str(record.get("visibility_quality", "") or "").lower()
        clarity = str(record.get("localization_clarity", "") or "").lower()
        identity_verifiable = bool(record.get("identity_verifiable_from_visible_parts"))
        record_text = _record_identity_text(record)
        explicit_identity_conflict = any(
            token in record_text
            for token in (
                "wrong person",
                "different person",
                "different individual",
                "does not match",
                "not the same",
                "identity conflicts",
                "conflicts with",
            )
        )
        candidate_decisions = [
            str(item.get("decision", "") or "").lower()
            for item in (record.get("candidate_evaluations") or [])
            if isinstance(item, dict)
        ]
        has_present_candidate = any(
            decision in {"present", "match", "matched", "same"} or "present" in decision
            for decision in candidate_decisions
        )
        candidate_only_uncertain = bool(candidate_decisions) and not has_present_candidate

        very_weak_identity = not identity_verifiable and confidence < 0.75
        very_weak_localization = (
            confidence < 0.8
            and visibility in {"ambiguous", "poor", "missing"}
            and clarity in {"low", "ambiguous", "poor", "missing"}
        )
        if explicit_identity_conflict or very_weak_identity or very_weak_localization:
            return ["head_edit_identity_evidence_too_generic"]
        if candidate_only_uncertain and confidence < 0.9:
            return ["head_edit_identity_evidence_too_generic"]
        return []
    return ["head_edit_missing_concrete_face_or_hair_identity"]


def _all_negative_physical_recovery_reasons(
    record: Dict[str, Any],
    *,
    edit_prompt: str = "",
) -> List[str]:
    """Block high-risk all-negative recovery for placement edits on weak crops."""
    if not bool(record.get("all_negative_recovery_checked")):
        return []
    if not _edit_targets_physical_placement(edit_prompt):
        return []
    if not (bool(record.get("present")) or bool(record.get("vlm_present"))):
        return []

    visible_parts = set(_visible_parts_list(record))
    has_face_head_hair = bool(visible_parts & {"face", "head", "hair"})
    specific_count = _specific_visual_identity_evidence_count(record)

    reasons: List[str] = []
    if not (has_face_head_hair and specific_count >= 2):
        if not has_face_head_hair:
            reasons.append("all_negative_physical_placement_without_face_head_hair_identity")

        initial_text = " ".join(
            str(record.get(key, "") or "")
            for key in (
                "initial_detection_reasoning",
                "reasoning",
                "candidate_evaluations",
            )
        ).lower()
        if any(
            token in initial_text
            for token in (
                "another entity",
                "different entity",
                "wrong person",
                "not visible",
                "not present",
                "entity_",
                "instruction_id",
            )
        ) and specific_count < 3:
            reasons.append("all_negative_recovery_conflicts_with_prior_identity_reject")

    attachment = _requested_attachment_point(edit_prompt)
    if attachment:
        visible = _attachment_visible_from_record(record, attachment)
        if (visible is not True or not _attachment_side_field_reliable(record, attachment)) and not _attachment_strong_entity_bypass(record, attachment):
            reasons.append("all_negative_physical_placement_attachment_not_reliably_editable")
    return list(dict.fromkeys(reasons))


_PRESENCE_NEUTRAL_EDITABILITY_REASONS = {
    "physical_placement_attachment_not_reliably_editable",
    "all_negative_physical_placement_attachment_not_reliably_editable",
}


def _split_presence_and_editability_reasons(
    reasons: Sequence[str],
) -> tuple[List[str], List[str]]:
    """Separate true absence/identity failures from non-editable attachment failures."""
    presence_reasons: List[str] = []
    editability_reasons: List[str] = []
    for reason in reasons:
        if reason in _PRESENCE_NEUTRAL_EDITABILITY_REASONS:
            editability_reasons.append(reason)
        else:
            presence_reasons.append(reason)
    return presence_reasons, editability_reasons


def _record_matches_face_or_hair_identity(record: Dict[str, Any]) -> bool:
    """True when VLM says the visible face/hair identifies the tracked subject."""
    parts = set(_visible_parts_list(record))
    text = _record_identity_text(record)
    has_face_or_hair = bool(parts & {"face", "head", "hair"})
    if (
        bool(record.get("identity_verifiable_from_visible_parts"))
        and has_face_or_hair
        and _specific_visual_identity_evidence_count(record) >= 2
    ):
        return True
    return has_face_or_hair and any(
        phrase in text
        for phrase in (
            "face and hair",
            "hair and facial",
            "hair silhouette",
            "matching hair silhouette",
            "hairline",
            "matching hairline",
            "hair part",
            "hair parting",
            "middle-parted hair",
            "middle parted hair",
            "matches the face",
            "matches reference face",
            "matches the reference face",
            "matching face shape",
            "matching face profile",
            "matching profile",
            "profile matches",
            "side profile",
            "reference profile",
            "matching jaw",
            "matching jawline",
            "matching nose",
            "matching eyes",
            "matches the reference",
            "matches reference identity",
            "exact subject",
        )
    )


def _record_matches_required_subject_cues(
    record: Dict[str, Any],
    *,
    subject_features: str = "",
) -> int:
    """Count strong subject cues that are explicitly visible in VLM text."""
    subject = (subject_features or "").lower()
    text = _record_identity_text(record)
    count = 0
    for cue in _STRONG_SUBJECT_FEATURE_CUES:
        if cue in subject and cue in text:
            count += 1
    return count


def _direct_face_identity_ok(record: Dict[str, Any]) -> bool:
    """Require direct face / hair evidence instead of generic same-person wording."""
    return _feature_audit_has_clear_match(record) or _record_matches_face_or_hair_identity(record)


def _back_view_only_identity(
    record: Dict[str, Any],
    visible_parts: List[str],
) -> bool:
    """True when the candidate is mainly a back-view body without visible face/hair."""
    if any(part in {"face", "hair"} for part in visible_parts):
        return False
    orientation_text = " ".join(
        str(record.get(key, "") or "")
        for key in (
            "body_orientation",
            "viewpoint",
            "location_description",
            "vlm_location_description",
            "reasoning",
        )
    ).lower()
    return any(
        token in orientation_text
        for token in (
            "back",
            "back view",
            "rear",
            "rear view",
            "turned away",
            "from behind",
            "back-facing",
            "back facing",
        )
    )


def _strict_identity_evidence_ok(
    record: Dict[str, Any],
    visible_parts: List[str],
    *,
    subject_features: str = "",
    min_cues: int = 2,
    allow_back_view: bool = False,
) -> bool:
    """Tighten weak-visibility acceptance so small/blurry/back-view look-alikes stay rejected."""
    if not visible_parts or all(part in _PARTIAL_LIMB_PARTS for part in visible_parts):
        return False

    direct_face_identity = _direct_face_identity_ok(record)
    stable_cues = _stable_identity_cue_count(record, subject_features=subject_features)
    identity_verifiable = bool(record.get("identity_verifiable_from_visible_parts"))
    area_fraction = float(record.get("approximate_area_fraction", 0.0) or 0.0)
    back_view_only = _back_view_only_identity(record, visible_parts)

    if back_view_only and not allow_back_view:
        return False
    if back_view_only and not direct_face_identity:
        return (
            identity_verifiable
            and stable_cues >= max(min_cues + 1, 3)
            and area_fraction >= 0.04
        )
    if direct_face_identity:
        return True
    if identity_verifiable and stable_cues >= min_cues:
        return True
    return stable_cues >= max(min_cues + 1, 3)


def _challenging_visibility_identity_ok(
    record: Dict[str, Any],
    visible_parts: List[str],
    *,
    subject_features: str = "",
) -> bool:
    """Only keep difficult-visibility matches when identity evidence is still direct and specific."""
    confidence = float(record.get("confidence", 0.0) or 0.0)
    if confidence < KEYFRAME_CHALLENGING_VISIBILITY_MIN_CONFIDENCE:
        return False
    area_fraction = float(record.get("approximate_area_fraction", 0.0) or 0.0)
    if area_fraction < KEYFRAME_CHALLENGING_VISIBILITY_MIN_AREA_FRACTION:
        return False
    if not visible_parts or all(part in _PARTIAL_LIMB_PARTS for part in visible_parts):
        return False
    if not any(part in _IDENTITY_CORE_PARTS for part in visible_parts):
        return False

    visibility = str(record.get("visibility_quality", "") or "").strip().lower()
    text = " ".join(
        str(record.get(key, "") or "")
        for key in (
            "location_description",
            "vlm_location_description",
            "reasoning",
            "viewpoint",
            "visibility_quality",
        )
    ).lower()
    challenging_context = visibility in {"too_small", "blurry", "ambiguous", "occluded"} or any(
        cue in text for cue in _CHALLENGING_VISIBILITY_CUES
    )
    if not challenging_context:
        return False

    if _identity_contradiction_reasons(record):
        return False
    candidate_conflicts = [
        conflict
        for conflict in _candidate_identity_conflicts(record)
        if not _conflict_is_uncertain_or_blurry(conflict)
    ]
    if candidate_conflicts:
        return False

    direct_face_identity = _direct_face_identity_ok(record)
    if _back_view_only_identity(record, visible_parts) and not direct_face_identity:
        return False

    specific_count = _specific_visual_identity_evidence_count(record)
    if not _strict_identity_evidence_ok(
        record,
        visible_parts,
        subject_features=subject_features,
        min_cues=3,
    ):
        return False

    if visibility == "too_small":
        return (
            _wide_shot_identity_ok(record, visible_parts, subject_features=subject_features)
            and bool(record.get("entity_visibility_completeness") == "sufficient")
            and (direct_face_identity or specific_count >= 3)
        )

    if visibility in {"blurry", "ambiguous", "occluded"}:
        if not direct_face_identity:
            return False
        return (
            _close_shot_identity_ok(record, visible_parts, subject_features=subject_features)
            or _viewpoint_tolerant_identity_ok(record, visible_parts, subject_features=subject_features)
            or _stable_visual_identity_ok(record, visible_parts, subject_features=subject_features)
            or bool(record.get("identity_verifiable_from_visible_parts"))
        )

    return direct_face_identity or specific_count >= 3


def _strong_feature_mismatch_reasons(
    record: Dict[str, Any],
    *,
    subject_features: str = "",
    edit_prompt: str = "",
) -> List[str]:
    """Reject confident-looking matches that contradict hard identity features.

    VLMs often over-promote look-alikes when a nearby extra shares one cue. These
    checks only fire on concrete contradictions, while preserving tolerance for
    pose/expression/action drift.
    """
    subject = (subject_features or "").lower()
    text = _record_identity_text(record)
    reasons: List[str] = []
    face_or_hair_identity_ok = _record_matches_face_or_hair_identity(record)

    # When the VLM's reasoning text explicitly discusses face/hair/facial
    # features (even to claim a mismatch), it clearly DID see the face.
    # In that case, missing clothing features (e.g. suspenders cropped out
    # in a close-up) should be treated as a framing issue, not an identity
    # contradiction. This prevents the code from reinforcing a VLM false
    # negative where the VLM incorrectly claims "different facial features"
    # while the face is actually visible and matches the reference.
    vlm_discusses_face_or_hair = any(
        token in text
        for token in (
            "face", "facial", "hair", "hairline", "hair part",
            "hair silhouette", "jaw", "nose", "eye", "eyebrow",
            "cheek", "chin", "profile", "middle-parted", "middle parted",
        )
    )

    entity_id = str(record.get("entity_id", "") or "").lower()
    instruction_id = str(record.get("instruction_id", "") or "").lower()

    subject_has_headwear = any(cue in subject for cue in _HEADWEAR_CUES)
    subject_hair_is_identity = any(cue in subject for cue in _HAIR_IDENTITY_CUES)
    record_mentions_headwear = any(cue in text for cue in _HEADWEAR_CUES)

    if (
        subject_hair_is_identity
        and not subject_has_headwear
        and record_mentions_headwear
        and _record_matches_required_subject_cues(record, subject_features=subject_features) == 0
    ):
        reasons.append("unexpected_headwear_conflicts_with_hair_identity")

    if "suspenders" in subject and "suspenders" not in text:
        # Skip suspenders-based rejection when the VLM explicitly discussed
        # face/hair features — the face was visible, and missing suspenders
        # is a framing/crop issue, not an identity contradiction.
        suspenders_rejection_blocked = vlm_discusses_face_or_hair

        if _edit_targets_physical_placement(edit_prompt):
            if not face_or_hair_identity_ok and not suspenders_rejection_blocked:
                reasons.append("missing_required_suspenders_for_placement")
        if (
            not face_or_hair_identity_ok
            and not suspenders_rejection_blocked
            and any(cue in text for cue in ("vest", "flat cap", "cap", "hat"))
        ):
            reasons.append("missing_required_suspenders")

    if (
        "vest" in subject
        and "suspenders" in text
        and "vest" not in text
        and not face_or_hair_identity_ok
    ):
        reasons.append("suspenders_conflict_with_required_vest")

    subject_strong_cues = [cue for cue in _STRONG_SUBJECT_FEATURE_CUES if cue in subject]
    text_matches = sum(1 for cue in subject_strong_cues if cue in text)
    text_conflicts = any(
        (
            cue in text
            and cue not in subject
            and not (cue == "cap" and "flat cap" in subject)
        )
        for cue in ("suspenders", "vest", "flat cap", "cap", "hat")
    )
    if subject_strong_cues and text_conflicts and text_matches == 0 and not face_or_hair_identity_ok:
        reasons.append("strong_identity_features_do_not_match_subject")

    return list(dict.fromkeys(reasons))


def _viewpoint_tolerant_identity_ok(
    record: Dict[str, Any],
    visible_parts: List[str],
    *,
    subject_features: str = "",
) -> bool:
    """Accept non-frontal views only when stable identity evidence is still strong enough."""
    area_fraction = float(record.get("approximate_area_fraction", 0.0) or 0.0)
    if area_fraction < KEYFRAME_VIEWPOINT_TOLERANT_MIN_AREA_FRACTION:
        return False
    if not any(part in {"face", "head", "hair", "torso", "upper_body", "body", "back", "neck", "full_body", "full_figure"} for part in visible_parts):
        return False

    text = " ".join(
        str(record.get(key, "") or "")
        for key in (
            "viewpoint",
            "location_description",
            "vlm_location_description",
            "reasoning",
        )
    ).lower()
    non_frontal = any(cue in text for cue in _NON_FRONTAL_VIEWPOINT_CUES)
    if not non_frontal:
        return False

    return _strict_identity_evidence_ok(
        record,
        visible_parts,
        subject_features=subject_features,
        min_cues=2,
        allow_back_view=True,
    )


def _stable_identity_cue_count(record: Dict[str, Any], *, subject_features: str = "") -> int:
    cues = set(_identity_cues_list(record))
    text = " ".join(
        str(record.get(key, "") or "")
        for key in (
            "location_description",
            "vlm_location_description",
            "reasoning",
            "initial_detection_reasoning",
            "viewpoint",
        )
    ).lower()
    subject = (subject_features or "").lower()
    for cue in _CLOTHING_IDENTITY_CUES:
        if cue in subject and cue in text:
            cues.add(cue)
    for cue in _HAIR_IDENTITY_CUES:
        if cue in subject and cue in text:
            cues.add(cue)
    return len(cues)


def _stable_visual_identity_ok(
    record: Dict[str, Any],
    visible_parts: List[str],
    *,
    subject_features: str = "",
) -> bool:
    """Accept stable-cue matches only when the evidence is stronger than generic similarity."""
    confidence = float(record.get("confidence", 0.0) or 0.0)
    if confidence < KEYFRAME_STABLE_CUE_MIN_CONFIDENCE:
        return False
    area_fraction = float(record.get("approximate_area_fraction", 0.0) or 0.0)
    if area_fraction < KEYFRAME_STABLE_CUE_MIN_AREA_FRACTION:
        return False
    if not visible_parts or all(part in _PARTIAL_LIMB_PARTS for part in visible_parts):
        return False
    if not any(part in _IDENTITY_CORE_PARTS for part in visible_parts):
        return False
    return _strict_identity_evidence_ok(
        record,
        visible_parts,
        subject_features=subject_features,
        min_cues=2,
    )


def _wide_shot_identity_ok(
    record: Dict[str, Any],
    visible_parts: List[str],
    *,
    subject_features: str = "",
) -> bool:
    """Small wide-shot entities need stricter identity proof than large foreground subjects."""
    area_fraction = float(record.get("approximate_area_fraction", 0.0) or 0.0)
    if area_fraction < KEYFRAME_WIDE_SHOT_MIN_AREA_FRACTION:
        return False
    if all(part in _PARTIAL_LIMB_PARTS for part in visible_parts):
        return False

    completeness = str(
        record.get("entity_visibility_completeness", "") or ""
    ).strip().lower()
    has_core = any(
        part in {"face", "head", "hair", "torso", "upper_body", "body", "back", "neck", "full_body", "full_figure"}
        for part in visible_parts
    )
    has_enough_body = has_core or completeness == "sufficient"
    if not has_enough_body:
        return False

    text = " ".join(
        str(record.get(key, "") or "")
        for key in (
            "location_description",
            "vlm_location_description",
            "reasoning",
            "viewpoint",
            "visibility_quality",
        )
    ).lower()
    wide_context = any(cue in text for cue in _WIDE_SHOT_CUES) or area_fraction < KEYFRAME_ENTITY_MIN_AREA_FRACTION
    if not wide_context:
        return False

    return _strict_identity_evidence_ok(
        record,
        visible_parts,
        subject_features=subject_features,
        min_cues=3,
    )


def _close_shot_identity_ok(
    record: Dict[str, Any],
    visible_parts: List[str],
    *,
    subject_features: str = "",
) -> bool:
    """Close-up crops still need concrete identity, not just a plausible look-alike."""
    area_fraction = float(record.get("approximate_area_fraction", 0.0) or 0.0)
    if area_fraction < 0.12:
        return False
    if not any(part in {"face", "head", "hair", "torso", "upper_body", "body", "back", "neck"} for part in visible_parts):
        return False
    if all(part in _PARTIAL_LIMB_PARTS for part in visible_parts):
        return False

    text = " ".join(
        str(record.get(key, "") or "")
        for key in (
            "location_description",
            "vlm_location_description",
            "reasoning",
            "viewpoint",
            "visibility_quality",
        )
    ).lower()
    close_context = any(cue in text for cue in _CLOSE_SHOT_CUES) or area_fraction >= 0.25
    if not close_context:
        return False

    return _strict_identity_evidence_ok(
        record,
        visible_parts,
        subject_features=subject_features,
        min_cues=2,
        allow_back_view=True,
    )


def single_entity_best_attempt_recoverable(
    record: Dict[str, Any],
    *,
    edit_prompt: str = "",
    subject_features: str = "",
) -> bool:
    """Recover only truly strong borderline hits; do not rescue weak small/blurry/back-view guesses."""
    if not (bool(record.get("present")) or bool(record.get("vlm_present"))):
        return False

    visible_parts = _visible_parts_list(record)
    challenging_visibility_ok = _challenging_visibility_identity_ok(
        record,
        visible_parts,
        subject_features=subject_features,
    )
    confidence = float(record.get("confidence", 0.0) or 0.0)
    required_confidence = (
        KEYFRAME_CHALLENGING_VISIBILITY_MIN_CONFIDENCE
        if challenging_visibility_ok else 0.9
    )
    if confidence < required_confidence:
        return False

    if not visible_parts or all(part in _PARTIAL_LIMB_PARTS for part in visible_parts):
        return False
    if not any(
        part in {"face", "head", "hair", "torso", "upper_body", "body", "back", "neck", "shoulder"}
        for part in visible_parts
    ):
        return False

    direct_face_identity = _direct_face_identity_ok(record)
    specific_count = _specific_visual_identity_evidence_count(record)
    weak_visibility = str(record.get("visibility_quality", "") or "").strip().lower() in {
        "too_small",
        "blurry",
        "ambiguous",
        "occluded",
    }
    if _back_view_only_identity(record, visible_parts) and not direct_face_identity:
        return False
    if weak_visibility and not direct_face_identity:
        return False

    has_verifiable_identity = (
        direct_face_identity
        or (
            bool(record.get("identity_verifiable_from_visible_parts"))
            and specific_count >= 3
            and (
                _candidate_has_present_decision(record)
                or _feature_audit_only_uncertain_non_matches(record)
            )
        )
    )
    if not has_verifiable_identity:
        return False

    if _identity_contradiction_reasons(record):
        return False
    if _strong_feature_mismatch_reasons(
        record,
        subject_features=subject_features,
        edit_prompt=edit_prompt,
    ):
        return False

    candidate_conflicts = [
        conflict
        for conflict in _candidate_identity_conflicts(record)
        if not _conflict_is_uncertain_or_blurry(conflict)
    ]
    if candidate_conflicts:
        return False

    has_present_candidate = _candidate_has_present_decision(record)
    if not has_present_candidate and not direct_face_identity:
        return False

    has_strong_identity = (
        direct_face_identity
        or (not weak_visibility and _state_change_robust_identity_ok(record))
        or (not weak_visibility and specific_count >= 3)
        or challenging_visibility_ok
    )
    if not has_strong_identity:
        return False

    angle_or_crop_robust = (
        _viewpoint_tolerant_identity_ok(record, visible_parts, subject_features=subject_features)
        or _close_shot_identity_ok(record, visible_parts, subject_features=subject_features)
        or _wide_shot_identity_ok(record, visible_parts, subject_features=subject_features)
        or _stable_visual_identity_ok(record, visible_parts, subject_features=subject_features)
        or challenging_visibility_ok
    )
    return angle_or_crop_robust


def _has_removal_identity_cues(
    record: Dict[str, Any],
    visible_parts: List[str],
    *,
    edit_prompt: str,
    subject_features: str = "",
) -> bool:
    """Clothing/build cues are enough to inpaint-remove a partially visible person."""
    if not _edit_targets_removal(edit_prompt):
        return False
    if (
        any(part in {"face", "head", "hair", "torso", "upper_body", "body", "back", "cap"} for part in visible_parts)
        and _specific_visual_identity_evidence_count(record) >= 2
    ):
        return True
    return _has_clothing_identity_cues(
        record,
        visible_parts,
        subject_features=subject_features,
    )


def _removal_partial_detection_ok(
    record: Dict[str, Any],
    *,
    edit_prompt: str,
    visible_parts: List[str],
    clarity: str,
    area_fraction: float,
    min_area_fraction: float,
    subject_features: str = "",
) -> bool:
    if not _edit_targets_removal(edit_prompt):
        return False
    if clarity != "high":
        return False
    if area_fraction < min_area_fraction:
        return False
    return _has_removal_identity_cues(
        record,
        visible_parts,
        edit_prompt=edit_prompt,
        subject_features=subject_features,
    )


def _partial_edit_detection_ok(
    record: Dict[str, Any],
    *,
    edit_prompt: str,
    visible_parts: List[str],
    clarity: str,
    area_fraction: float,
    min_area_fraction: float,
    subject_features: str = "",
) -> bool:
    """Partial torso/shoulder visibility is enough for localized accessory edits."""
    if clarity != "high" or area_fraction < min_area_fraction:
        return False
    if _edit_targets_removal(edit_prompt):
        return _removal_partial_detection_ok(
            record,
            edit_prompt=edit_prompt,
            visible_parts=visible_parts,
            clarity=clarity,
            area_fraction=area_fraction,
            min_area_fraction=min_area_fraction,
            subject_features=subject_features,
        )
    has_face = any(part in {"face", "head", "hair"} for part in visible_parts)
    has_torso = any(
        part in {"torso", "upper_body", "shoulder", "body", "back", "neck"} for part in visible_parts
    )
    if has_face and (
        _edit_targets_head_region(edit_prompt) or area_fraction >= min_area_fraction
    ):
        return True
    if has_torso and _edit_targets_upper_body(edit_prompt):
        attachment = _requested_attachment_point(edit_prompt)
        attachment_visible = _attachment_visible_from_record(record, attachment)
        if (
            attachment
            and attachment_visible is True
            and area_fraction >= max(min_area_fraction, 0.08)
            and any(part in {"shoulder", "torso", "upper_body", "body", "back"} for part in visible_parts)
        ):
            return True
        return _has_clothing_identity_cues(
            record,
            visible_parts,
            subject_features=subject_features,
        )
    return False


def _infer_partial_visibility_risks(
    record: Dict[str, Any],
    *,
    edit_prompt: str = "",
    subject_features: str = "",
    catalog_trusted: bool = False,
) -> List[str]:
    """Heuristic safety net when VLM over-confidently marks partial limbs as clear."""
    if catalog_trusted or record.get("catalog_trusted"):
        return []

    risks: List[str] = []
    visible_parts = _visible_parts_list(record)
    has_identity_core = _has_identity_core_visible(visible_parts)
    removal_identity_ok = _has_removal_identity_cues(
        record,
        visible_parts,
        edit_prompt=edit_prompt,
        subject_features=subject_features,
    )
    partial_edit_ok = _partial_edit_detection_ok(
        record,
        edit_prompt=edit_prompt,
        visible_parts=visible_parts,
        clarity=str(record.get("localization_clarity", "") or "").strip().lower(),
        area_fraction=float(record.get("approximate_area_fraction", 0.0) or 0.0),
        min_area_fraction=KEYFRAME_ENTITY_MIN_AREA_FRACTION,
        subject_features=subject_features,
    )
    identity_cue_ok = removal_identity_ok or partial_edit_ok
    key_visible_identity_ok = _has_key_visible_identity_region(
        record,
        visible_parts,
        edit_prompt=edit_prompt,
        subject_features=subject_features,
    )
    clarity = str(record.get("localization_clarity", "") or "").strip().lower()

    if (
        record.get("identity_verifiable_from_visible_parts") is False
        and not has_identity_core
        and not identity_cue_ok
    ):
        risks.append("identity_not_verifiable_from_visible_parts")
    if visible_parts:
        if all(part in _PARTIAL_LIMB_PARTS for part in visible_parts):
            if not identity_cue_ok:
                risks.append(f"visible_parts=limb_only({','.join(visible_parts)})")
        elif (
            not any(part in _IDENTITY_CORE_PARTS for part in visible_parts)
            and not identity_cue_ok
        ):
            risks.append(f"visible_parts=no_core_identity({','.join(visible_parts)})")
        elif not key_visible_identity_ok and not identity_cue_ok:
            risks.append(f"visible_parts=no_key_identity_region({','.join(visible_parts)})")

    if identity_cue_ok and clarity == "high":
        return risks

    text = " ".join(
        str(record.get(key, "") or "")
        for key in ("location_description", "reasoning", "vlm_location_description")
    ).lower()
    if not text:
        return risks

    edge_cue = any(phrase in text for phrase in _EDGE_CROP_PHRASES)
    limb_cue = any(phrase in text for phrase in _LIMB_PHRASES) or any(
        word in text for word in _PARTIAL_LIMB_PARTS
    )
    face_cue = "face" in text or "facial" in text or "head" in text
    has_face_part = any(part in {"face", "head"} for part in visible_parts)

    if edge_cue and limb_cue and not face_cue and not has_face_part:
        risks.append("edge_cropped_partial_limb_without_face")
    if edge_cue and not face_cue and not has_face_part:
        risks.append("edge_cropped_without_face")
    if any(phrase in text for phrase in _LIMB_PHRASES):
        risks.append("partial_limb_description")

    return risks


def _detection_location_text(record: Dict[str, Any]) -> str:
    """Return the best available location text from a detection record."""
    for key in ("vlm_location_description", "location_description"):
        text = str(record.get(key, "") or "").strip()
        if text:
            return text
    return ""


def finalize_entity_detection_location(
    record: Dict[str, Any],
    *,
    catalog: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Ensure detected entities keep a non-empty location for downstream localization."""
    out = dict(record)
    loc = _detection_location_text(out)
    if not loc and catalog:
        loc = str(catalog.get("location_description", "") or "").strip()
    if not loc:
        return out

    out["vlm_location_description"] = loc
    if bool(out.get("vlm_present", False)) or bool(out.get("present", False)):
        out["location_description"] = loc
    return out


def apply_keyframe_entity_presence_gate(
    record: Dict[str, Any],
    *,
    min_confidence: float = KEYFRAME_ENTITY_PRESENCE_MIN_CONFIDENCE,
    min_area_fraction: float = KEYFRAME_ENTITY_MIN_AREA_FRACTION,
    edit_prompt: str = "",
    subject_features: str = "",
    catalog_trusted: bool = False,
) -> Dict[str, Any]:
    """Force present=false when match quality is below strict detection thresholds."""
    out = _sanitize_vlm_location_fields(record)
    if catalog_trusted:
        out["catalog_trusted"] = True
    vlm_present = bool(out.get("vlm_present", out.get("present", False)))
    vlm_location = _detection_location_text(out)
    confidence = float(out.get("confidence", 0.0) or 0.0)
    visibility = str(out.get("visibility_quality", "") or "").strip().lower()
    clarity = str(out.get("localization_clarity", "") or "").strip().lower()
    area_fraction = float(out.get("approximate_area_fraction", 0.0) or 0.0)
    completeness = str(
        out.get("entity_visibility_completeness", "") or ""
    ).strip().lower()
    visible_parts = _visible_parts_list(out)
    has_face = any(part in {"face", "head"} for part in visible_parts)
    has_identity_core = _has_identity_core_visible(visible_parts)
    identity_verifiable = bool(out.get("identity_verifiable_from_visible_parts", False))
    removal_partial_ok = _removal_partial_detection_ok(
        out,
        edit_prompt=edit_prompt,
        visible_parts=visible_parts,
        clarity=clarity,
        area_fraction=area_fraction,
        min_area_fraction=min_area_fraction,
        subject_features=subject_features,
    )
    partial_edit_ok = _partial_edit_detection_ok(
        out,
        edit_prompt=edit_prompt,
        visible_parts=visible_parts,
        clarity=clarity,
        area_fraction=area_fraction,
        min_area_fraction=min_area_fraction,
        subject_features=subject_features,
    )
    viewpoint_tolerant_ok = _viewpoint_tolerant_identity_ok(
        out,
        visible_parts,
        subject_features=subject_features,
    )
    wide_shot_ok = _wide_shot_identity_ok(
        out,
        visible_parts,
        subject_features=subject_features,
    )
    close_shot_ok = _close_shot_identity_ok(
        out,
        visible_parts,
        subject_features=subject_features,
    )
    stable_visual_ok = _stable_visual_identity_ok(
        out,
        visible_parts,
        subject_features=subject_features,
    )
    challenging_visibility_ok = _challenging_visibility_identity_ok(
        out,
        visible_parts,
        subject_features=subject_features,
    )
    direct_face_identity = _direct_face_identity_ok(out)
    back_view_only = _back_view_only_identity(out, visible_parts)
    relaxed_partial_ok = removal_partial_ok or partial_edit_ok

    reject_reasons: List[str] = []
    if vlm_present:
        reject_reasons.extend(_identity_contradiction_reasons(out))
    if vlm_present and confidence < min_confidence:
        threshold_candidates = [min_confidence]
        if stable_visual_ok:
            threshold_candidates.append(KEYFRAME_STABLE_CUE_MIN_CONFIDENCE)
        if relaxed_partial_ok:
            threshold_candidates.append(KEYFRAME_ENTITY_REMOVAL_MIN_CONFIDENCE)
        if viewpoint_tolerant_ok:
            threshold_candidates.append(KEYFRAME_VIEWPOINT_TOLERANT_MIN_CONFIDENCE)
        if wide_shot_ok:
            threshold_candidates.append(KEYFRAME_WIDE_SHOT_MIN_CONFIDENCE)
        if close_shot_ok:
            threshold_candidates.append(KEYFRAME_CLOSE_SHOT_MIN_CONFIDENCE)
        effective_min = min(threshold_candidates)
        if confidence < effective_min:
            reject_reasons.append(
                f"confidence {confidence:.3f} < required {effective_min:.2f}"
            )
    if vlm_present and visibility != "clear":
        allow_occluded = (
            visibility == "occluded"
            and has_face
            and clarity == "high"
            and area_fraction >= min_area_fraction
        )
        allow_occluded_removal = (
            visibility == "occluded" and removal_partial_ok
        )
        allow_occluded_partial = (
            visibility == "occluded" and partial_edit_ok
        )
        allow_blurry = (
            visibility == "blurry"
            and has_identity_core
            and identity_verifiable
            and direct_face_identity
            and clarity == "high"
            and area_fraction >= min_area_fraction
        )
        allow_non_frontal_ambiguous = (
            visibility in {"ambiguous", "occluded", "blurry"}
            and viewpoint_tolerant_ok
            and direct_face_identity
            and clarity in {"high", "medium"}
            and not back_view_only
        )
        allow_wide_shot_small = (
            visibility in {"too_small", "ambiguous", "blurry", "clear"}
            and wide_shot_ok
            and (direct_face_identity or (identity_verifiable and not back_view_only))
            and clarity in {"high", "medium"}
        )
        allow_close_shot_crop = (
            visibility in {"clear", "ambiguous", "occluded", "blurry"}
            and close_shot_ok
            and (direct_face_identity or (identity_verifiable and not back_view_only))
            and clarity in {"high", "medium"}
        )
        allow_stable_visual = (
            visibility in {"clear", "too_small", "ambiguous", "blurry", "occluded"}
            and stable_visual_ok
            and (direct_face_identity or (identity_verifiable and not back_view_only))
            and clarity in {"high", "medium"}
        )
        if (
            not allow_occluded
            and not allow_occluded_removal
            and not allow_occluded_partial
            and not allow_blurry
            and not allow_non_frontal_ambiguous
            and not allow_wide_shot_small
            and not allow_close_shot_crop
            and not allow_stable_visual
        ):
            reject_reasons.append(f"visibility_quality={visibility or 'missing'}")
    if vlm_present and clarity != "high" and not (
        clarity == "medium"
        and (stable_visual_ok or viewpoint_tolerant_ok or wide_shot_ok or close_shot_ok)
        and direct_face_identity
    ):
        reject_reasons.append(f"localization_clarity={clarity or 'missing'}")
    if vlm_present and area_fraction < min_area_fraction and not wide_shot_ok and not stable_visual_ok:
        reject_reasons.append(
            f"approximate_area_fraction {area_fraction:.3f} < required {min_area_fraction:.2f}"
        )
    if vlm_present and back_view_only and not direct_face_identity:
        reject_reasons.append("back_view_without_direct_face_identity")
    if vlm_present and completeness != KEYFRAME_ENTITY_REQUIRED_COMPLETENESS:
        allow_partial = (
            completeness == "partial"
            and clarity in {"high", "medium"}
            and (area_fraction >= min_area_fraction or stable_visual_ok or wide_shot_ok)
            and (
                (
                    has_face
                    and (
                        _edit_targets_head_region(edit_prompt)
                        or area_fraction >= min_area_fraction
                    )
                )
                or (
                    _edit_targets_removal(edit_prompt)
                    and _has_removal_identity_cues(
                        out,
                        visible_parts,
                        edit_prompt=edit_prompt,
                        subject_features=subject_features,
                    )
                )
                or (
                    _edit_targets_upper_body(edit_prompt)
                    and _has_clothing_identity_cues(
                        out,
                        visible_parts,
                        subject_features=subject_features,
                    )
                )
                or (viewpoint_tolerant_ok and direct_face_identity)
                or (wide_shot_ok and (direct_face_identity or identity_verifiable))
                or (close_shot_ok and (direct_face_identity or identity_verifiable))
                or (stable_visual_ok and (direct_face_identity or identity_verifiable))
                or challenging_visibility_ok
            )
            and not (back_view_only and not direct_face_identity)
        )
        if not allow_partial:
            reject_reasons.append(
                f"entity_visibility_completeness={completeness or 'missing'}"
            )
    reject_reasons.extend(
        risk
        for risk in _infer_partial_visibility_risks(
            out,
            edit_prompt=edit_prompt,
            subject_features=subject_features,
            catalog_trusted=catalog_trusted,
        )
        if risk not in reject_reasons
    )
    reject_reasons.extend(
        reason
        for reason in _weak_identity_evidence_reasons(out)
        if reason not in reject_reasons
    )
    reject_reasons.extend(
        reason
        for reason in _head_edit_identity_reasons(
            out,
            edit_prompt=edit_prompt,
        )
        if reason not in reject_reasons
    )
    reject_reasons.extend(
        reason
        for reason in _physical_placement_identity_reasons(
            out,
            edit_prompt=edit_prompt,
            subject_features=subject_features,
        )
        if reason not in reject_reasons
    )
    reject_reasons.extend(
        reason
        for reason in _all_negative_physical_recovery_reasons(
            out,
            edit_prompt=edit_prompt,
        )
        if reason not in reject_reasons
    )
    reject_reasons.extend(
        reason
        for reason in _strong_feature_mismatch_reasons(
            out,
            subject_features=subject_features,
            edit_prompt=edit_prompt,
        )
        if reason not in reject_reasons
    )

    presence_reject_reasons, editability_reject_reasons = (
        _split_presence_and_editability_reasons(reject_reasons)
    )
    # Presence gating removed: final entity existence is decided upstream by the
    # repeated-screening aggregate rule (all votes true OR score sum above the
    # threshold). Reject reasons are retained only for diagnostics and no longer
    # force present=False or zero the existence score.
    gated = False
    attachment = _requested_attachment_point(edit_prompt)
    if editability_reject_reasons and attachment:
        out["target_attachment_visible"] = False
        visibility_map = out.get("attachment_visibility")
        if not isinstance(visibility_map, dict):
            visibility_map = {}
        else:
            visibility_map = dict(visibility_map)
        visibility_map[attachment] = False
        if attachment.endswith("_shoulder"):
            visibility_map.setdefault("shoulder", False)
        out["attachment_visibility"] = visibility_map

    # Keep the VLM's own presence verdict; never override it via gating, and
    # never zero the existence score. The caller applies the final aggregate
    # validity rule across repeated screening attempts.
    out["present"] = bool(vlm_present)
    if vlm_present and vlm_location:
        out["vlm_location_description"] = vlm_location
    if not out["present"]:
        out["location_description"] = ""

    out["confidence"] = confidence
    out["visibility_quality"] = visibility
    out["localization_clarity"] = clarity
    out["approximate_area_fraction"] = area_fraction
    out["entity_visibility_completeness"] = completeness
    out["visible_parts"] = visible_parts
    out["vlm_present"] = vlm_present
    out["presence_gated"] = gated
    out["presence_reject_reasons"] = presence_reject_reasons
    out["editability_reject_reasons"] = editability_reject_reasons
    out["editability_gated"] = bool(editability_reject_reasons)

    # Soft score penalty: since hard gating is removed, each active presence
    # reject reason (such as missing suspenders for placement) must penalize the
    # calibrated existence score and confidence. This ensures the aggregate
    # vote/score system still filters out entities with critical presence flaws.
    if presence_reject_reasons:
        penalty = float(len(presence_reject_reasons)) * 30.0
        calibrated_score = float(out.get("existence_confidence_score", 0.0) or 0.0)
        out["existence_confidence_score"] = max(0.0, calibrated_score - penalty)
        out["confidence"] = max(0.0, out["confidence"] - (penalty / 100.0))

    if editability_reject_reasons:
        base = str(out.get("reasoning", "") or "").strip()
        gate_note = "editability gated: " + "; ".join(editability_reject_reasons)
        out["reasoning"] = f"{base} | {gate_note}".strip(" |")

    return out


def absent_entity_location_record(
    instruction_id: str,
    entity_id: str,
    *,
    reasoning: str = "entity not detected",
    visibility_quality: str = "missing",
) -> Dict[str, Any]:
    """Default absent record for an instruction with no detection."""
    return {
        "instruction_id": instruction_id,
        "entity_id": entity_id,
        "present": False,
        "location_description": "",
        "confidence": 0.0,
        "existence_confidence_score": 0.0,
        "visibility_quality": visibility_quality,
        "approximate_area_fraction": 0.0,
        "localization_clarity": "low",
        "entity_visibility_completeness": "fragment",
        "visible_parts": [],
        "identity_verifiable_from_visible_parts": False,
        "reasoning": reasoning,
        "vlm_present": False,
        "presence_gated": False,
        "presence_reject_reasons": [],
    }


def normalize_vlm_entity_location_record(
    data: Dict[str, Any],
    *,
    instruction_id: str,
    entity_id: str,
    edit_prompt: str = "",
    subject_features: str = "",
) -> Dict[str, Any]:
    """Parse one entity entry from VLM JSON and apply presence gate."""
    visible_parts_raw = data.get("visible_parts") or []
    if isinstance(visible_parts_raw, str):
        visible_parts_raw = [visible_parts_raw]
    visible_parts = [
        str(part).strip().lower()
        for part in visible_parts_raw
        if str(part).strip()
    ]
    identity_verifiable = data.get("identity_verifiable_from_visible_parts")
    if identity_verifiable is None:
        identity_verifiable = bool(data.get("identity_verifiable"))
    contradiction_reasons = _identity_contradiction_reasons(data)
    if contradiction_reasons and bool(data.get("present", False)):
        data["present"] = False
    existence_score_raw = data.get("existence_confidence_score")
    if existence_score_raw is None:
        existence_score_raw = data.get("entity_presence_confidence_score")
    if existence_score_raw is None:
        existence_score_raw = data.get("presence_confidence_score")
    try:
        existence_score = max(0.0, min(100.0, float(existence_score_raw)))
    except (TypeError, ValueError):
        try:
            raw_confidence = float(data.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            raw_confidence = 0.0
        existence_score = raw_confidence if raw_confidence > 1.0 else raw_confidence * 100.0
    raw_existence_score = existence_score
    if not bool(data.get("present", False)):
        existence_score = 0.0
        score_calibration_reasons: List[str] = []
    else:
        existence_score, score_calibration_reasons = _calibrate_existence_confidence_score(
            data,
            visible_parts,
            existence_score,
        )
    try:
        raw_confidence = float(data.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        raw_confidence = 0.0
    normalized_confidence = raw_confidence / 100.0 if raw_confidence > 1.0 else raw_confidence
    if bool(data.get("present", False)) and existence_score > 0:
        normalized_confidence = max(
            normalized_confidence,
            existence_score / 100.0,
            KEYFRAME_ENTITY_PRESENCE_MIN_CONFIDENCE,
        )

    record = _sanitize_vlm_location_fields({
        "instruction_id": instruction_id,
        "entity_id": entity_id,
        "present": bool(data.get("present", False)),
        "location_description": str(
            data.get("location_description", "") or ""
        ).strip(),
        "confidence": normalized_confidence,
        "existence_confidence_score": existence_score,
        "visibility_quality": str(
            data.get("visibility_quality", "") or ""
        ).strip().lower(),
        "approximate_area_fraction": float(
            data.get("approximate_area_fraction", 0.0) or 0.0
        ),
        "localization_clarity": str(
            data.get("localization_clarity", "") or ""
        ).strip().lower(),
        "entity_visibility_completeness": str(
            data.get("entity_visibility_completeness", "") or ""
        ).strip().lower(),
        "visible_parts": visible_parts,
        "viewpoint": str(data.get("viewpoint", "") or "").strip().lower(),
        "identity_verifiable_from_visible_parts": bool(identity_verifiable),
        "attachment_visibility": data.get("attachment_visibility") or {},
        "target_attachment_visible": data.get("target_attachment_visible"),
        "target_attachment_point": str(data.get("target_attachment_point", "") or "").strip(),
        "body_orientation": str(data.get("body_orientation", "") or "").strip(),
        "anatomical_left_screen_side": str(
            data.get("anatomical_left_screen_side", "") or ""
        ).strip(),
        "anatomical_right_screen_side": str(
            data.get("anatomical_right_screen_side", "") or ""
        ).strip(),
        "attachment_visibility_reasoning": str(
            data.get("attachment_visibility_reasoning", "") or ""
        ).strip(),
        "candidate_evaluations": data.get("candidate_evaluations") or [],
        "feature_audit": _feature_audit_entries(data) or data.get("feature_audit") or [],
        "reasoning": str(data.get("reasoning", "") or "").strip(),
    })
    if contradiction_reasons:
        record["identity_contradiction_reasons"] = contradiction_reasons
    if score_calibration_reasons:
        record["existence_confidence_score_raw"] = raw_existence_score
        record["existence_confidence_score_calibrated"] = True
        record["existence_confidence_score_calibration_reasons"] = score_calibration_reasons
    return apply_keyframe_entity_presence_gate(
        record,
        edit_prompt=edit_prompt,
        subject_features=subject_features,
    )


def format_batch_entity_detection_catalog(
    entity_specs: Sequence[Dict[str, Any]],
) -> str:
    """Format entity list + reference image indices for batch VLM detection."""
    lines: List[str] = []
    for spec in entity_specs:
        iid = str(spec.get("instruction_id", "")).strip()
        eid = str(spec.get("entity_id", "")).strip()
        ref_idx = int(spec.get("ref_image_index", 0) or 0)
        subject = str(spec.get("subject_features", "") or "").strip() or "N/A"
        edit = str(spec.get("edit_prompt", "") or "").strip() or "N/A"
        scope_line = format_target_instance_scope_line(
            spec.get("target_instance_scope", "single")
        )
        prior_block = format_prior_detection_block(spec.get("prior_detection"))
        lines.append(
            f"- {iid} / {eid} → reference image {ref_idx}\n"
            f"  subject_features: {subject}\n"
            f"  edit_prompt: {edit}\n"
            f"  {scope_line}"
        )
        if "appears in the first" in subject.lower() or "first few frames" in subject.lower():
            lines.append(
                "  NOTE: first-frame outfit/time wording is an initial identity anchor, not a permanent "
                "hard clothing requirement. In later scenes, match the same tracked person by face, hair, "
                "body identity, and story continuity even if the current outfit/expression/viewpoint differs."
            )
        if prior_block:
            lines.append(f"  {prior_block.replace(chr(10), chr(10) + '  ')}")
        reference_context = str(spec.get("reference_identity_context", "") or "").strip()
        if reference_context:
            lines.append(
                "  REFERENCE IDENTITY CONTEXT (multi-view/history; use for side profile, expression, "
                "and wardrobe-change matching):\n"
                f"  {reference_context.replace(chr(10), chr(10) + '  ')}"
            )
        catalog = spec.get("catalog_appearance")
        if isinstance(catalog, dict) and catalog.get("present"):
            view_angle = str(catalog.get("view_angle", "") or "").strip() or "unknown"
            visibility = str(catalog.get("visibility_state", "") or "").strip()
            location = str(catalog.get("location_description", "") or "").strip()
            lines.append(
                "  CATALOG APPEARANCE (verified for this exact keyframe during grounding):\n"
                f"  - view_angle: {view_angle}\n"
                f"  - visibility: {visibility or 'N/A'}\n"
                f"  - location: {location or 'N/A'}\n"
                "  The target may appear as side/back profile or soft-focus — still mark present=true "
                "when image 1 matches these cues and the front-view reference identity."
            )
    return "\n".join(lines) if lines else "(no entities)"


KEYFRAME_ENTITY_DETECT_MIN_CONFIDENCE = 0.5


def _entity_record_from_vlm_entry(
    entry: Dict[str, Any],
    *,
    instruction_id: str,
    entity_id: str,
) -> Dict[str, Any]:
    """Normalize one entity dict from VLM detect/verify JSON."""
    visible_parts_raw = entry.get("visible_parts") or []
    if isinstance(visible_parts_raw, str):
        visible_parts_raw = [visible_parts_raw]
    visible_parts = [
        str(part).strip().lower()
        for part in visible_parts_raw
        if str(part).strip()
    ]
    identity_cues_raw = entry.get("identity_cues") or []
    if isinstance(identity_cues_raw, str):
        identity_cues_raw = [identity_cues_raw]
    identity_cues = [
        str(cue).strip()
        for cue in identity_cues_raw
        if str(cue).strip()
    ]
    confidence = float(entry.get("confidence", 0.0) or 0.0)
    present = bool(entry.get("present", False)) and confidence >= KEYFRAME_ENTITY_DETECT_MIN_CONFIDENCE
    location = str(entry.get("location_description", "") or "").strip()
    if not present:
        location = ""
    existence_score_raw = entry.get("existence_confidence_score")
    if existence_score_raw is None:
        existence_score_raw = entry.get("entity_presence_confidence_score")
    if existence_score_raw is None:
        existence_score_raw = entry.get("presence_confidence_score")
    try:
        existence_score = max(0.0, min(100.0, float(existence_score_raw)))
    except (TypeError, ValueError):
        existence_score = confidence if confidence > 1.0 else confidence * 100.0
    if not present:
        existence_score = 0.0
    identity_verifiable = entry.get("identity_verifiable_from_visible_parts")
    if identity_verifiable is None:
        identity_verifiable = entry.get("identity_verifiable")
    return {
        "instruction_id": instruction_id,
        "entity_id": entity_id,
        "present": present,
        "confidence": confidence,
        "existence_confidence_score": existence_score,
        "location_description": location,
        "visibility_quality": str(entry.get("visibility_quality", "") or "").strip().lower(),
        "approximate_area_fraction": float(entry.get("approximate_area_fraction", 0.0) or 0.0),
        "visible_parts": visible_parts,
        "viewpoint": str(entry.get("viewpoint", "") or "").strip().lower(),
        "identity_cues": identity_cues,
        "identity_verifiable_from_visible_parts": bool(identity_verifiable),
        "localization_clarity": str(entry.get("localization_clarity", "") or "").strip().lower(),
        "entity_visibility_completeness": str(
            entry.get("entity_visibility_completeness", "") or ""
        ).strip().lower(),
        "attachment_visibility": entry.get("attachment_visibility") or {},
        "target_attachment_visible": entry.get("target_attachment_visible"),
        "target_attachment_point": str(entry.get("target_attachment_point", "") or "").strip(),
        "body_orientation": str(entry.get("body_orientation", "") or "").strip(),
        "anatomical_left_screen_side": str(
            entry.get("anatomical_left_screen_side", "") or ""
        ).strip(),
        "anatomical_right_screen_side": str(
            entry.get("anatomical_right_screen_side", "") or ""
        ).strip(),
        "attachment_visibility_reasoning": str(
            entry.get("attachment_visibility_reasoning", "") or ""
        ).strip(),
        "reasoning": str(entry.get("reasoning", "") or "").strip(),
        "candidate_evaluations": entry.get("candidate_evaluations") or [],
        "vlm_present": bool(entry.get("present", False)),
        "location_corrected": bool(entry.get("location_corrected", False)),
    }


def parse_entity_detect_response(
    data: Dict[str, Any],
    entity_specs: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Map step-1 VLM detection JSON to instruction_id → location record."""
    specs_by_id = {
        str(spec.get("instruction_id", "")).strip(): spec
        for spec in entity_specs
        if str(spec.get("instruction_id", "")).strip()
    }
    raw_entities = data.get("entities") or data.get("entity_locations") or []
    if not isinstance(raw_entities, list):
        raw_entities = []

    results: Dict[str, Dict[str, Any]] = {}
    for entry in raw_entities:
        if not isinstance(entry, dict):
            continue
        iid = str(entry.get("instruction_id", "")).strip()
        if not iid or iid not in specs_by_id:
            continue
        spec = specs_by_id[iid]
        eid = str(spec.get("entity_id", "") or entry.get("entity_id", "")).strip()
        record = _entity_record_from_vlm_entry(
            entry,
            instruction_id=iid,
            entity_id=eid,
        )
        results[iid] = apply_keyframe_entity_presence_gate(
            record,
            edit_prompt=str(spec.get("edit_prompt", "") or ""),
            subject_features=str(spec.get("subject_features", "") or ""),
        )

    for iid, spec in specs_by_id.items():
        if iid not in results:
            results[iid] = absent_entity_location_record(
                iid,
                str(spec.get("entity_id", "")).strip(),
                reasoning="missing from detection VLM response",
            )
    return results


def parse_entity_verify_response(
    data: Dict[str, Any],
    entity_specs: Sequence[Dict[str, Any]],
    detection_records: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Map step-2 VLM verification JSON to instruction_id → verified location record."""
    specs_by_id = {
        str(spec.get("instruction_id", "")).strip(): spec
        for spec in entity_specs
        if str(spec.get("instruction_id", "")).strip()
    }
    raw_entities = data.get("entities") or []
    if not isinstance(raw_entities, list):
        raw_entities = []
    detection_records = detection_records or {}

    results: Dict[str, Dict[str, Any]] = {}
    for entry in raw_entities:
        if not isinstance(entry, dict):
            continue
        iid = str(entry.get("instruction_id", "")).strip()
        if not iid or iid not in specs_by_id:
            continue
        spec = specs_by_id[iid]
        eid = str(spec.get("entity_id", "") or entry.get("entity_id", "")).strip()
        record = _entity_record_from_vlm_entry(
            entry,
            instruction_id=iid,
            entity_id=eid,
        )
        has_verify_existence_score = any(
            entry.get(key) not in (None, "")
            for key in (
                "existence_confidence_score",
                "entity_presence_confidence_score",
                "presence_confidence_score",
            )
        )
        if not has_verify_existence_score:
            detected = detection_records.get(iid) or {}
            if "existence_confidence_score" in detected:
                record["existence_confidence_score"] = detected.get("existence_confidence_score")
                record["existence_confidence_score_inherited_from_detection"] = True
        record = apply_keyframe_entity_presence_gate(
            record,
            edit_prompt=str(spec.get("edit_prompt", "") or ""),
            subject_features=str(spec.get("subject_features", "") or ""),
        )
        record["verified"] = True
        results[iid] = record

    for iid, spec in specs_by_id.items():
        if iid not in results:
            results[iid] = absent_entity_location_record(
                iid,
                str(spec.get("entity_id", "")).strip(),
                reasoning="missing from location verification VLM response",
            )
    return results


def merge_scene_consistency_location_record(
    original: Dict[str, Any],
    corrected: Dict[str, Any],
    *,
    edit_prompt: str = "",
    subject_features: str = "",
) -> Dict[str, Any]:
    """Merge scene-consistency VLM output and re-apply presence gate on the result."""
    merged = dict(original)
    pre_present = bool(original.get("present"))
    pre_location = str(original.get("location_description", "") or "")
    pre_reject_reasons = [
        str(reason).strip()
        for reason in (original.get("presence_reject_reasons") or [])
        if str(reason).strip()
    ]
    corrected_has_identity_evidence = bool(
        corrected.get("identity_verifiable_from_visible_parts")
        or corrected.get("identity_verifiable")
    )
    if (
        not pre_present
        and bool(corrected.get("present"))
        and _initial_identity_rejected(original)
        and not _strong_current_identity_override(corrected)
    ):
        blocked = dict(original)
        blocked["scene_consistency_checked"] = True
        blocked["scene_consistency_promotion_blocked"] = True
        blocked["scene_consistency_block_reason"] = (
            "initial detection explicitly rejected the candidate as a different identity; "
            "scene consistency did not provide strong current-frame identity override"
        )
        blocked["pre_consistency_gate_reject_reasons"] = pre_reject_reasons
        return blocked
    if (
        not pre_present
        and bool(corrected.get("present"))
        and original.get("presence_gated")
        and _prior_override_blocked(pre_reject_reasons)
        and not corrected_has_identity_evidence
    ):
        blocked = dict(original)
        blocked["scene_consistency_checked"] = True
        blocked["scene_consistency_promotion_blocked"] = True
        blocked["scene_consistency_block_reason"] = (
            "initial detection had a strong identity/visibility gate; "
            "scene consistency did not provide current-frame verifiable identity evidence"
        )
        blocked["pre_consistency_gate_reject_reasons"] = pre_reject_reasons
        return blocked

    original_identity_cues = original.get("identity_cues") or []
    corrected_identity_cues = corrected.get("identity_cues") or []
    original_candidates = original.get("candidate_evaluations") or []
    corrected_candidates = corrected.get("candidate_evaluations") or []
    original_visible_parts = original.get("visible_parts") or []
    corrected_visible_parts = corrected.get("visible_parts") or []

    merged.update(corrected)
    merged["identity_cues"] = _merge_unique_strings(
        corrected_identity_cues,
        original_identity_cues,
    )
    merged["candidate_evaluations"] = _merge_candidate_evaluations(
        corrected_candidates,
        original_candidates,
    )
    merged["visible_parts"] = _merge_unique_strings(
        corrected_visible_parts,
        original_visible_parts,
    )
    for key in (
        "viewpoint",
        "body_orientation",
        "anatomical_left_screen_side",
        "anatomical_right_screen_side",
        "attachment_visibility_reasoning",
    ):
        if not merged.get(key) and original.get(key):
            merged[key] = original.get(key)
    
    # If the corrected record was gated by the VLM prompt (e.g. missing cues) but the original had them,
    # and the VLM actually confirmed the entity is visible (vlm_present=True), restore present=True
    # so the gate can re-evaluate with the full merged features.
    if bool(merged.get("vlm_present")) and not bool(merged.get("present")):
        merged["present"] = True
        
    if (
        not bool(merged.get("identity_verifiable_from_visible_parts"))
        and bool(original.get("identity_verifiable_from_visible_parts"))
        and bool(merged.get("present"))
    ):
        merged["identity_verifiable_from_visible_parts"] = True
    if (
        not pre_present
        and bool(corrected.get("present"))
        and original.get("presence_gated")
    ):
        merged["pre_consistency_gate_reject_reasons"] = pre_reject_reasons
    merged = apply_keyframe_entity_presence_gate(
        merged,
        edit_prompt=edit_prompt,
        subject_features=subject_features,
    )
    merged["scene_consistency_checked"] = True
    if (
        pre_present != bool(merged.get("present"))
        or pre_location != str(merged.get("location_description", "") or "")
    ):
        merged["scene_consistency_corrected"] = True
        merged["pre_consistency_present"] = pre_present
        merged["pre_consistency_location"] = pre_location
    if not pre_present and bool(corrected.get("present")) and not merged.get("present"):
        merged["scene_consistency_promotion_blocked"] = True
    return merged


def build_cross_keyframe_edit_continuity_prompt(
    *,
    prior_refs: Sequence[Dict[str, Any]],
    present_instruction_ids: Sequence[str],
    instructions_by_id: Dict[str, EntityInstruction],
) -> str:
    """Positive guidance so later keyframes apply the same edits as earlier ones."""
    present_set = {str(iid).strip() for iid in present_instruction_ids if str(iid).strip()}
    lines: List[str] = []
    for ref in prior_refs:
        if not isinstance(ref, dict):
            continue
        keyframe_id = str(ref.get("keyframe_id", "") or "previous keyframe").strip()
        shared_ids = [
            str(iid).strip()
            for iid in (ref.get("instruction_ids") or [])
            if str(iid).strip() in present_set
        ]
        for iid in shared_ids:
            instr = instructions_by_id.get(iid)
            if instr is None:
                continue
            edit_text = (instr.edit_prompt or "").strip()
            if not edit_text:
                continue
            lines.append(
                f"- {keyframe_id}: for {iid} / {instr.entity_id}, apply the same edit here: "
                f"{edit_text}"
            )
    if not lines:
        return ""
    return (
        "CROSS-KEYFRAME EDIT CONTINUITY (same scene — replicate prior keyframe outcomes on "
        "the located targets in this frame):\n"
        + "\n".join(lines)
    )


def parse_scene_keyframe_presence_consistency_response(
    data: Dict[str, Any],
    entity_specs: Sequence[Dict[str, Any]],
    keyframe_ids: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Map scene consistency VLM JSON to keyframe_id → instruction_id → record."""
    specs_by_id = {
        str(spec.get("instruction_id", "")).strip(): spec
        for spec in entity_specs
        if str(spec.get("instruction_id", "")).strip()
    }
    allowed_keyframes = {str(keyframe_id).strip() for keyframe_id in keyframe_ids}
    raw_keyframes = data.get("keyframes") or []
    if not isinstance(raw_keyframes, list):
        raw_keyframes = []

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for keyframe_entry in raw_keyframes:
        if not isinstance(keyframe_entry, dict):
            continue
        keyframe_id = str(
            keyframe_entry.get("keyframe_id")
            or keyframe_entry.get("keyframe_stem")
            or ""
        ).strip()
        if not keyframe_id or keyframe_id not in allowed_keyframes:
            continue

        raw_entities = keyframe_entry.get("entities") or []
        if not isinstance(raw_entities, list):
            raw_entities = []

        keyframe_results: Dict[str, Dict[str, Any]] = {}
        for entry in raw_entities:
            if not isinstance(entry, dict):
                continue
            iid = str(entry.get("instruction_id", "")).strip()
            if not iid or iid not in specs_by_id:
                continue
            spec = specs_by_id[iid]
            eid = str(spec.get("entity_id", "") or entry.get("entity_id", "")).strip()
            record = _entity_record_from_vlm_entry(
                entry,
                instruction_id=iid,
                entity_id=eid,
            )
            record["localization_clarity"] = str(
                entry.get("localization_clarity", "") or ""
            ).strip().lower() or (
                "high" if record.get("present") else "low"
            )
            record["entity_visibility_completeness"] = str(
                entry.get("entity_visibility_completeness", "") or ""
            ).strip().lower() or (
                "partial" if record.get("present") else "fragment"
            )
            record["identity_verifiable_from_visible_parts"] = bool(
                entry.get("identity_verifiable_from_visible_parts")
                or entry.get("identity_verifiable")
            )
            record = apply_keyframe_entity_presence_gate(
                record,
                edit_prompt=str(spec.get("edit_prompt", "") or ""),
                subject_features=str(spec.get("subject_features", "") or ""),
            )
            record["verified"] = True
            record["scene_consistency_checked"] = True
            keyframe_results[iid] = record
        if keyframe_results:
            results[keyframe_id] = keyframe_results
    return results


def format_detection_results_block(
    location_records: Dict[str, Dict[str, Any]],
) -> str:
    """Format step-1 detection records for step-2 verification prompt.

    Note: the ``reasoning`` text produced by step-1 is deliberately omitted
    from the verification prompt. Feeding prior free-text reasoning into the
    verify VLM biases it toward the same conclusion and can propagate
    contradictory narrative (e.g. referencing wrong keyframe numbers). The
    verifier should rely on the structured fields (present / confidence /
    location / visible_parts) plus the actual keyframe pixels and reference
    image, not on step-1's prose justification.
    """
    lines: List[str] = []
    for iid, record in location_records.items():
        present = bool(record.get("present"))
        conf = float(record.get("confidence", 0.0) or 0.0)
        loc = str(record.get("location_description", "") or "").strip() or "(empty)"
        lines.append(
            f"- {iid}: present={present}, confidence={conf:.2f}, "
            f"location={loc}"
        )
    return "\n".join(lines) if lines else "(no detection results)"


def format_observed_edits_block(observed_data: Dict[str, Any]) -> str:
    """Format step-4 edit comparison output for step-5 QA prompt."""
    ops = observed_data.get("observed_edit_operations") or []
    if not isinstance(ops, list) or not ops:
        summary = str(observed_data.get("summary", "") or "").strip()
        return summary or "(no visible edit operations detected)"

    lines: List[str] = []
    summary = str(observed_data.get("summary", "") or "").strip()
    if summary:
        lines.append(f"Summary: {summary}")
    for idx, op in enumerate(ops, start=1):
        if not isinstance(op, dict):
            continue
        lines.append(
            f"{idx}. region={op.get('region', 'N/A')} | target={op.get('target', 'N/A')} | "
            f"operation={op.get('operation', 'N/A')} | change={op.get('change_description', 'N/A')} | "
            f"confidence={float(op.get('confidence', 0.0) or 0.0):.2f}"
        )
    return "\n".join(lines)


def parse_keyframe_edit_comparison_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize step-4 VLM edit comparison JSON."""
    ops = data.get("observed_edit_operations") or []
    if not isinstance(ops, list):
        ops = []
    normalized_ops: List[Dict[str, Any]] = []
    for op in ops:
        if not isinstance(op, dict):
            continue
        normalized_ops.append({
            "region": str(op.get("region", "") or "").strip(),
            "target": str(op.get("target", "") or "").strip(),
            "operation": str(op.get("operation", "") or "").strip(),
            "change_description": str(op.get("change_description", "") or "").strip(),
            "confidence": float(op.get("confidence", 0.0) or 0.0),
        })
    return {
        "observed_edit_operations": normalized_ops,
        "summary": str(data.get("summary", "") or "").strip(),
    }


def parse_keyframe_edit_completion_qa_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize step-5 VLM completion QA JSON."""
    return {
        "passed": bool(data.get("passed", False)),
        "score": float(data.get("score", 0.0) or 0.0),
        "frame_structure_preserved": bool(data.get("frame_structure_preserved", False)),
        "edit_instruction_requirements_met": bool(
            data.get(
                "edit_instruction_requirements_met",
                data.get("edit_completed", False),
            )
        ),
        "edit_completed": bool(data.get("edit_completed", False)),
        "canonical_reference_alignment_ok": bool(
            data.get("canonical_reference_alignment_ok", False)
        ),
        "original_entity_state_preserved": bool(
            data.get("original_entity_state_preserved", False)
        ),
        "photorealistic_scene_integration_ok": bool(
            data.get("photorealistic_scene_integration_ok", False)
        ),
        "unrelated_edit_changes_absent": bool(
            data.get("unrelated_edit_changes_absent", False)
        ),
        "background_unedited_regions_preserved": bool(
            data.get("background_unedited_regions_preserved", False)
        ),
        "failed_aspects": [
            str(x).strip()
            for x in (data.get("failed_aspects") or [])
            if str(x).strip()
        ],
        "feedback": str(data.get("feedback", "") or "").strip(),
        "retry_focus_prompt": str(data.get("retry_focus_prompt", "") or "").strip(),
        "positive_prompt": str(data.get("positive_prompt", "") or "").strip(),
    }


def parse_batch_entity_location_response(
    data: Dict[str, Any],
    entity_specs: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Map batch VLM response to instruction_id → gated location record."""
    specs_by_id = {
        str(spec.get("instruction_id", "")).strip(): spec
        for spec in entity_specs
        if str(spec.get("instruction_id", "")).strip()
    }
    raw_entities = data.get("entities") or data.get("entity_locations") or []
    if not isinstance(raw_entities, list):
        raw_entities = []

    parsed_by_id: Dict[str, Dict[str, Any]] = {}
    for entry in raw_entities:
        if not isinstance(entry, dict):
            continue
        iid = str(entry.get("instruction_id", "")).strip()
        if not iid or iid not in specs_by_id:
            continue
        spec = specs_by_id[iid]
        eid = str(spec.get("entity_id", "") or entry.get("entity_id", "")).strip()
        edit_prompt = str(spec.get("edit_prompt", "") or "")
        parsed_by_id[iid] = normalize_vlm_entity_location_record(
            entry,
            instruction_id=iid,
            entity_id=eid,
            edit_prompt=edit_prompt,
            subject_features=str(spec.get("subject_features", "") or ""),
        )

    results: Dict[str, Dict[str, Any]] = {}
    for iid, spec in specs_by_id.items():
        if iid in parsed_by_id:
            results[iid] = parsed_by_id[iid]
        else:
            results[iid] = absent_entity_location_record(
                iid,
                str(spec.get("entity_id", "")).strip(),
                reasoning="missing from batch VLM response",
            )
    return results


PRIOR_KEYFRAME_DETECTION_MIN_CONFIDENCE = 0.9
PRIOR_KEYFRAME_DETECTION_MIN_QUALITY = 80.0
CATALOG_APPEARANCE_MIN_CONFIDENCE = 0.9
CATALOG_APPEARANCE_MIN_QUALITY = 75.0

_PRIOR_OVERRIDE_BLOCKERS = (
    "edge_cropped",
    "visible_parts=",
    "identity_not_verifiable",
    "partial_limb",
    "approximate_area_fraction",
)


def _prior_override_blocked(reject_reasons: Sequence[str]) -> bool:
    for reason in reject_reasons:
        text = str(reason).strip().lower()
        if any(token in text for token in _PRIOR_OVERRIDE_BLOCKERS):
            return True
    return False


def load_prior_keyframe_detection(
    ref_dir: str,
    instruction_id: str,
    keyframe_path: str,
) -> Dict[str, Any] | None:
    """Load entity detection record for this exact keyframe from entity ref meta."""
    meta_path = os.path.join(ref_dir, f"{instruction_id}_ref.meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        data = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    manifest = data.get("input_manifest") or {}
    target = os.path.normpath(os.path.abspath(keyframe_path))
    for entry in manifest.get("keyframes") or []:
        if not isinstance(entry, dict):
            continue
        entry_path = str(entry.get("keyframe_path") or "").strip()
        if not entry_path:
            continue
        if os.path.normpath(os.path.abspath(entry_path)) == target:
            return entry
    return None


def _shorten_reference_text(value: Any, *, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


_REFERENCE_ALIAS_STOPWORDS = {
    "A",
    "An",
    "The",
    "This",
    "That",
    "There",
    "Her",
    "His",
    "She",
    "He",
    "They",
    "Their",
    "It",
    "Its",
    "Image",
    "Keyframe",
    "Scene",
    "Reference",
    "Entity",
    "Subject",
}


def _extract_reference_aliases(text: str, *, limit: int = 6) -> List[str]:
    """Extract likely character/name aliases without binding to any specific story."""
    subject_like_counts: Dict[str, int] = {}
    sentence_pattern = re.compile(r"(?<=[.!?])\s+")
    for sentence in sentence_pattern.split(text or ""):
        stripped = sentence.strip()
        match = re.match(
            r"^([A-Z][a-z]{2,})\s+"
            r"(continues|looks|stands|remains|faces|holds|maintains|appears|moves|turns|gazes|walks|sits)\b",
            stripped,
        )
        if not match:
            continue
        token = match.group(1).strip()
        if token in _REFERENCE_ALIAS_STOPWORDS:
            continue
        subject_like_counts[token] = subject_like_counts.get(token, 0) + 1

    aliases: List[str] = []
    for token, count in sorted(subject_like_counts.items(), key=lambda item: item[1], reverse=True):
        if count < 2:
            continue
        if token not in aliases:
            aliases.append(token)
        if len(aliases) >= limit:
            break
    return aliases


def load_reference_identity_context(
    ref_dir: str,
    instruction_id: str,
    *,
    max_entries: int = 6,
    max_chars: int = 1800,
) -> str:
    """Summarize multi-view reference appearances for robust identity matching."""
    meta_path = os.path.join(ref_dir, f"{instruction_id}_ref.meta.json")
    if not os.path.exists(meta_path):
        return ""
    try:
        data = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return ""

    manifest = data.get("input_manifest") or {}
    appearances = manifest.get("appearances_catalog") or manifest.get("keyframes") or []
    if not isinstance(appearances, list):
        return ""

    selected_indices = manifest.get("selected_indices") or []
    ordered: List[Dict[str, Any]] = []
    if isinstance(selected_indices, list):
        by_index = {
            int(item.get("appearance_index")): item
            for item in appearances
            if isinstance(item, dict) and str(item.get("appearance_index", "")).strip().lstrip("-").isdigit()
        }
        for raw_idx in selected_indices:
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
            if idx in by_index:
                ordered.append(by_index[idx])
    if not ordered:
        ordered = [item for item in appearances if isinstance(item, dict)]

    all_reference_text = " ".join(
        str(entry.get(key, "") or "")
        for entry in ordered
        if isinstance(entry, dict)
        for key in (
            "keyframe_description",
            "scene_moment_description",
            "visibility_state",
            "pose_and_action",
            "reasoning",
        )
    )
    aliases = _extract_reference_aliases(all_reference_text)
    lines: List[str] = []
    if aliases:
        lines.append("identity_aliases: " + ", ".join(aliases))
    lines.append(
        "identity_signature: match the same tracked person/object by face structure, hair color, "
        "hair silhouette/style, head/upper-body shape, recurring role/name cues, and multi-view continuity."
    )
    lines.append(
        "mutable_appearance: clothing, scene, pose, expression, gaze, camera angle, distance, and partial "
        "profile/frontal differences can change and must not be treated as identity conflicts by themselves."
    )
    for entry in ordered[:max_entries]:
        view = str(entry.get("view_angle", "") or entry.get("orientation_note", "") or "unknown").strip()
        description = _shorten_reference_text(
            entry.get("keyframe_description")
            or entry.get("scene_moment_description")
            or entry.get("reasoning")
        )
        visibility = _shorten_reference_text(entry.get("visibility_state"), limit=180)
        pose = _shorten_reference_text(entry.get("pose_and_action"), limit=160)
        parts = [f"view={view}"]
        if description:
            parts.append(f"description={description}")
        if visibility:
            parts.append(f"visibility={visibility}")
        if pose:
            parts.append(f"pose={pose}")
        lines.append("- " + "; ".join(parts))

    context = "\n".join(lines)
    if len(context) > max_chars:
        context = context[: max(0, max_chars - 3)].rstrip() + "..."
    return context


def load_catalog_keyframe_appearance(
    appearances_path: str,
    instruction_id: str,
    keyframe_path: str,
) -> Dict[str, Any] | None:
    """Load verified keyframe appearance from entity_keyframe_appearances.json."""
    if not appearances_path or not os.path.exists(appearances_path):
        return None
    try:
        data = json.loads(Path(appearances_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    target = os.path.normpath(os.path.abspath(keyframe_path))
    for entity in data.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        if str(entity.get("instruction_id", "")).strip() != instruction_id:
            continue
        for appearance in entity.get("appearances") or []:
            if not isinstance(appearance, dict):
                continue
            entry_path = str(appearance.get("keyframe_path") or "").strip()
            if not entry_path:
                continue
            if os.path.normpath(os.path.abspath(entry_path)) == target:
                return appearance
    return None


def _visible_parts_from_prior(prior: Dict[str, Any]) -> List[str]:
    text = " ".join(
        str(prior.get(key, "") or "")
        for key in ("visibility_state", "keyframe_description", "pose_and_action")
    ).lower()
    parts: List[str] = []
    if "face" in text or "facial" in text or "head" in text:
        parts.append("face")
    if "torso" in text or "upper body" in text or "chest" in text or "shoulder" in text:
        parts.append("torso")
    if "full body" in text or "head to toe" in text:
        parts.extend(["face", "torso", "full_body"])
    return parts or ["face", "torso"]


def reconcile_keyframe_presence_with_catalog(
    record: Dict[str, Any],
    catalog: Dict[str, Any] | None,
    *,
    edit_prompt: str = "",
    subject_features: str = "",
) -> Dict[str, Any]:
    """Keep catalog as diagnostics only; VLM vote is the source of truth."""
    if not catalog or not catalog.get("present"):
        return record
    out = dict(record)
    out["catalog_appearance_available"] = True
    return out


def reconcile_keyframe_presence_with_prior(
    record: Dict[str, Any],
    prior: Dict[str, Any] | None,
    *,
    edit_prompt: str = "",
) -> Dict[str, Any]:
    """Restore only gated false-negatives where VLM saw the entity but gate rejected."""
    if not prior or not prior.get("present"):
        return record
    if record.get("present"):
        return record

    vlm_present = bool(record.get("vlm_present", False))
    if not vlm_present:
        return record
    if not record.get("presence_gated", False):
        return record

    reject_reasons = [
        str(reason).strip()
        for reason in (record.get("presence_reject_reasons") or [])
        if str(reason).strip()
    ]
    if _prior_override_blocked(reject_reasons):
        return record

    prior_conf = float(prior.get("confidence", 0.0) or 0.0)
    prior_quality = float(prior.get("quality_score", 0.0) or 0.0)
    if prior_conf < PRIOR_KEYFRAME_DETECTION_MIN_CONFIDENCE:
        return record
    if prior_quality < PRIOR_KEYFRAME_DETECTION_MIN_QUALITY:
        return record

    prior_visibility = str(prior.get("visibility_state", "") or "").lower()
    merged = dict(record)
    merged.update({
        "present": True,
        "vlm_present": True,
        "location_description": (
            str(prior.get("location_description", "") or "").strip()
            or str(record.get("location_description", "") or "").strip()
        ),
        "confidence": max(float(record.get("confidence", 0.0) or 0.0), prior_conf),
        "visibility_quality": "clear",
        "localization_clarity": "high",
        "entity_visibility_completeness": (
            "partial"
            if any(
                token in prior_visibility
                for token in ("partial", "cropped", "edge", "fragment", "occluded")
            )
            else "sufficient"
        ),
        "visible_parts": _visible_parts_from_prior(prior),
        "identity_verifiable_from_visible_parts": any(
            token in prior_visibility for token in ("face", "head", "facial")
        ),
        "approximate_area_fraction": max(
            float(record.get("approximate_area_fraction", 0.0) or 0.0),
            0.12,
        ),
        "reasoning": (
            f"prior catalog hint for same keyframe (conf={prior_conf:.2f}, "
            f"quality={prior_quality:.0f}); "
            f"{str(record.get('reasoning', '') or '').strip()}"
        ).strip(),
    })
    gated = apply_keyframe_entity_presence_gate(merged, edit_prompt=edit_prompt)
    if not gated.get("present"):
        return record
    gated["prior_detection_used"] = True
    return gated


_SCENE_PRIOR_CONTINUITY_FALSE_NEGATIVE_CUES = (
    "out of focus",
    "soft focus",
    "blurred",
    "blur",
    "side profile",
    "side view",
    "profile",
    "does not match",
    "look-alike",
    "different individual",
    "facial features",
    "not match",
    "cannot verify",
    "insufficient identity",
    "no sufficient identity",
)


def _batch_false_negative_reason(text: str) -> bool:
    """True when batch VLM likely rejected due to viewpoint/blur, not true absence."""
    normalized = _normalize_detection_text(text)
    return any(cue in normalized for cue in _SCENE_PRIOR_CONTINUITY_FALSE_NEGATIVE_CUES)


def reconcile_keyframe_presence_with_scene_prior(
    record: Dict[str, Any],
    scene_prior_records: Sequence[Dict[str, Any]] | None,
    *,
    edit_prompt: str = "",
) -> Dict[str, Any]:
    """Restore gated false-negatives when the entity was present in earlier same-scene keyframes."""
    if record.get("present"):
        return record

    priors = [
        entry for entry in (scene_prior_records or []) if isinstance(entry, dict)
    ]
    priors_present = [entry for entry in priors if entry.get("present")]
    if not priors_present:
        return record

    latest_present = priors_present[-1]

    if not record.get("vlm_present"):
        reasoning = str(record.get("reasoning", "") or "").strip()
        if not _batch_false_negative_reason(reasoning):
            return record
        merged = dict(record)
        merged.update({
            "present": True,
            "vlm_present": True,
            "location_description": (
                str(record.get("location_description", "") or "").strip()
                or str(latest_present.get("location_description", "") or "").strip()
            ),
            "confidence": max(
                float(record.get("confidence", 0.0) or 0.0),
                float(latest_present.get("confidence", 0.0) or 0.0),
                0.92,
            ),
            "visibility_quality": (
                str(record.get("visibility_quality", "") or "blurry").strip().lower()
                or "blurry"
            ),
            "localization_clarity": "high",
            "entity_visibility_completeness": "partial",
            "visible_parts": latest_present.get("visible_parts")
            or _visible_parts_from_prior(latest_present)
            or ["face", "torso"],
            "identity_verifiable_from_visible_parts": True,
            "approximate_area_fraction": max(
                float(record.get("approximate_area_fraction", 0.0) or 0.0),
                float(latest_present.get("approximate_area_fraction", 0.0) or 0.0),
                0.08,
            ),
            "reasoning": (
                f"scene continuity override (profile/blur false-negative) from "
                f"{str(latest_present.get('keyframe_stem', '') or 'prior').strip()}; "
                f"{reasoning}"
            ).strip("; "),
        })
        gated = apply_keyframe_entity_presence_gate(
            merged,
            edit_prompt=edit_prompt,
            min_confidence=PRIOR_KEYFRAME_DETECTION_MIN_CONFIDENCE,
        )
        if not gated.get("present"):
            return record
        gated["scene_prior_detection_used"] = True
        gated["scene_continuity_override"] = True
        return gated

    if not record.get("presence_gated"):
        return record

    reject_reasons = [
        str(reason).strip()
        for reason in (record.get("presence_reject_reasons") or [])
        if str(reason).strip()
    ]
    if _prior_override_blocked(reject_reasons):
        return record

    merged = dict(record)
    merged["present"] = True
    merged["location_description"] = (
        str(record.get("location_description", "") or "").strip()
        or str(latest_present.get("location_description", "") or "").strip()
    )
    merged["confidence"] = max(
        float(record.get("confidence", 0.0) or 0.0),
        float(latest_present.get("confidence", 0.0) or 0.0),
    )
    merged["reasoning"] = (
        f"scene continuity from {str(latest_present.get('keyframe_stem', '') or 'prior').strip()}; "
        f"{str(record.get('reasoning', '') or '').strip()}"
    ).strip("; ")

    gated = apply_keyframe_entity_presence_gate(merged, edit_prompt=edit_prompt)
    if not gated.get("present"):
        return record
    gated["scene_prior_detection_used"] = True
    return gated


def format_prior_detection_block(prior: Dict[str, Any] | None) -> str:
    if not prior or not prior.get("present"):
        return ""
    return (
        "PRIOR VERIFIED DETECTION (this exact keyframe was confirmed during reference building):\n"
        f"- confidence: {float(prior.get('confidence', 0.0) or 0.0):.2f}\n"
        f"- quality_score: {float(prior.get('quality_score', 0.0) or 0.0):.0f}\n"
        f"- identification_clarity_score: {float(prior.get('identification_clarity_score', 0.0) or 0.0):.0f}\n"
        f"- view_angle: {str(prior.get('view_angle', '') or '').strip()}\n"
        f"- visibility: {str(prior.get('visibility_state', '') or '').strip()}\n"
        f"- location: {str(prior.get('location_description', '') or '').strip()}\n"
        "Use this prior as strong same-keyframe evidence when handling viewpoint, scale, crop, expression, "
        "or wardrobe differences between image 1 and the reference. If image 1 contains a candidate at this "
        "location whose visible key regions (face/head/hair/profile/upper body) match the prior and the "
        "reference, mark present=true even if the reference image is a distant full-body view and image 1 is "
        "a close-up or three-quarter view. Still apply strict visibility and identity rules: do not mark "
        "present if only a partial edge limb, half shoulder, or generic clothing fragment is visible. Do not "
        "use the prior alone to assign a 90-100 existence_confidence_score; if the current frame only shows "
        "clothing, torso, shoulder, arm, or an edge crop, keep the score in the weak/partial range."
    )


_SCENE_PRIOR_ABSENT_STREAK_MIN = 2
_SCENE_PRIOR_REJECTION_CUES = (
    "does not match",
    "do not match",
    "not match",
    "not the entity",
    "not present",
    "no individual matching",
    "look-alike",
    "doesn't match",
)
_SCENE_PRIOR_SPATIAL_CUES = (
    "left",
    "right",
    "center",
    "centre",
    "foreground",
    "background",
    "stair",
    "railing",
    "rail",
    "mid-ground",
    "midground",
    "far left",
    "left edge",
    "left side",
    "extreme left",
    "vertical support",
)
_SCENE_ENTITY_DESCRIPTOR_ANCHORS = (
    "flat cap",
    "brown vest",
    "wooden railing",
    "railing",
    "rail",
    "leaning",
    "stubble",
    "suspenders",
    "far left",
    "left edge",
    "left side",
    "vertical support",
    "extreme left",
    "light-colored shirt",
    "support pole",
    "pole",
    "pillar",
)
_SCENE_LOCATION_STRUCTURE_ANCHORS = frozenset({
    "railing",
    "rail",
    "pole",
    "support",
    "pillar",
    "stair",
})
_SCENE_DESCRIPTOR_TOKEN_STOP = frozenset({
    "man",
    "woman",
    "wearing",
    "with",
    "the",
    "and",
    "has",
    "while",
    "appears",
    "throughout",
    "scene",
    "leaning",
    "present",
    "keyframe",
    "specific",
    "this",
    "that",
    "visible",
    "frame",
    "matches",
    "match",
    "entity",
})
_SCENE_PRIOR_ABSENT_CONFLICT_MIN_ANCHORS = 2


def _normalize_detection_text(text: str) -> str:
    return str(text or "").strip().lower()


_INITIAL_IDENTITY_REJECTION_TOKENS = (
    "does not match",
    "do not match",
    "not match",
    "not the same",
    "wrong person",
    "different person",
    "different individual",
    "different identity",
    "different character",
    "look-alike",
    "separate identity",
    "separate identity entity",
    "separate entity",
    "separate person",
    "not the target entity",
    "not the requested entity",
    "not the intended entity",
    "identity_not_verifiable",
    "strong_identity_features_do_not_match_subject",
    "unexpected_headwear_conflicts_with_hair_identity",
    "missing_required_suspenders",
)


def _initial_identity_rejection_text(record: Dict[str, Any]) -> str:
    pieces: List[str] = []
    for key in (
        "initial_detection_reasoning",
        "verification_absence_reasoning",
        "pre_consistency_gate_reject_reasons",
        "reasoning",
    ):
        value = record.get(key)
        if isinstance(value, (list, tuple)):
            pieces.extend(str(item) for item in value if str(item).strip())
        elif value is not None:
            pieces.append(str(value))
    return _normalize_detection_text(" ".join(pieces))


def _initial_identity_rejected(record: Dict[str, Any]) -> bool:
    """True when initial detection explicitly saw/reasoned a different identity."""
    initial_present = record.get("initial_detection_vlm_present")
    if initial_present is None:
        initial_present = record.get("vlm_present")
    if initial_present is not False:
        return False
    text = _initial_identity_rejection_text(record)
    if not text:
        return False
    if any(token in text for token in _INITIAL_IDENTITY_REJECTION_TOKENS):
        return True

    identifiers = [
        str(record.get("instruction_id", "") or "").strip().lower(),
        str(record.get("entity_id", "") or "").strip().lower(),
    ]
    for identifier in identifiers:
        if not identifier:
            continue
        escaped = re.escape(identifier)
        if re.search(rf"\bnot\s+(?:being\s+)?['\"]?{escaped}\b", text):
            return True
        if re.search(rf"\bis\s+not\s+['\"]?{escaped}\b", text):
            return True
    return False


def _reference_face_hair_identity_match(record: Dict[str, Any]) -> bool:
    """True for strong current-frame identity evidence tied to reference face/hair."""
    text = _normalize_detection_text(
        " ".join(
            str(record.get(key, "") or "")
            for key in ("reasoning", "identity_cues", "candidate_evaluations")
        )
    )
    visible_parts = set(_visible_parts_list(record))
    has_face_or_hair = bool(visible_parts & {"face", "head", "hair"})
    if not has_face_or_hair:
        return False

    reference_anchor = any(
        token in text
        for token in (
            "reference facial features",
            "reference face",
            "reference identity",
            "reference image",
            "reference images",
            "from reference",
            "from the reference",
            "matches the reference",
            "matching the reference",
        )
    )
    facial_or_hair_match = any(
        token in text
        for token in (
            "facial features",
            "facial structure",
            "face structure",
            "face shape",
            "facial proportions",
            "hair style",
            "hairstyle",
            "hair silhouette",
            "hair color",
            "auburn hair",
            "reddish hair",
            "red curly hair",
            "updo",
        )
    )
    direct_identifiable_match = any(
        token in text
        for token in (
            "clearly identifiable as",
            "clearly identified as",
            "same tracked person",
            "same person",
            "same identity",
        )
    )
    return (
        bool(record.get("identity_verifiable_from_visible_parts"))
        and _specific_visual_identity_evidence_count(record) >= 2
        and ((reference_anchor and facial_or_hair_match) or direct_identifiable_match)
    )


def _strong_current_identity_override(record: Dict[str, Any]) -> bool:
    """Require concrete current-frame evidence before overriding a prior rejection."""
    concrete_count = _specific_visual_identity_evidence_count(record)
    confidence = float(record.get("confidence", 0.0) or 0.0)
    current_text = _normalize_detection_text(
        " ".join(
            str(record.get(key, "") or "")
            for key in ("reasoning", "identity_cues", "candidate_evaluations")
        )
    )
    strong_named_match = any(
        token in current_text
        for token in (
            "matches reference face",
            "matches the reference face",
            "matches reference identity",
            "matches the reference identity",
            "clearly matches the reference identity",
            "matching face",
            "same face",
            "exact subject",
            "updo",
            "suspenders",
            "flat cap",
            "stubble",
            "vest",
        )
    )
    current_reference_identity_match = _reference_face_hair_identity_match(record)
    verifier_corrected_false_negative = (
        any(
            token in current_text
            for token in (
                "initial assessment incorrectly",
                "initial detection incorrectly",
                "incorrectly identified",
                "incorrectly marked",
                "incorrectly rejected",
            )
        )
        and any(
            token in current_text
            for token in (
                "reference image",
                "reference images",
                "reference identity",
                "from reference",
                "matches face structure",
                "matches facial structure",
                "matches face shape",
                "red curly hair",
                "reddish hair",
            )
        )
    )
    return (
        confidence >= 0.98
        and concrete_count >= 3
        and strong_named_match
    ) or (
        confidence >= 0.95
        and concrete_count >= 2
        and verifier_corrected_false_negative
    ) or (
        confidence >= 0.98
        and concrete_count >= 2
        and current_reference_identity_match
    )


def _extract_spatial_cues(text: str) -> set[str]:
    normalized = _normalize_detection_text(text)
    return {cue for cue in _SCENE_PRIOR_SPATIAL_CUES if cue in normalized}


def _prior_rejected_candidate_at_region(text: str) -> bool:
    normalized = _normalize_detection_text(text)
    return any(cue in normalized for cue in _SCENE_PRIOR_REJECTION_CUES)


def _prior_absent_is_definitive_region_rejection(prior: Dict[str, Any]) -> bool:
    """Gate-only or catalog false-negatives should not block later keyframes."""
    if prior.get("present"):
        return False
    if prior.get("presence_gated"):
        return False
    if prior.get("catalog_appearance_used"):
        return False
    prior_text = _normalize_detection_text(
        f"{prior.get('location_description', '')} {prior.get('reasoning', '')}"
    )
    return bool(prior_text) and _prior_rejected_candidate_at_region(prior_text)


def _spatial_cues_overlap(left: set[str], right: set[str]) -> bool:
    if not left or not right:
        return False
    return bool(left & right)


def _extract_entity_anchors(text: str) -> set[str]:
    normalized = _normalize_detection_text(text)
    anchors = {cue for cue in _SCENE_ENTITY_DESCRIPTOR_ANCHORS if cue in normalized}
    for phrase in ("flat cap", "brown vest", "wooden railing", "light-colored shirt"):
        if phrase in normalized:
            anchors.add(phrase.replace(" ", "_"))
    for word in re.findall(r"[a-z]{4,}", normalized):
        if word not in _SCENE_DESCRIPTOR_TOKEN_STOP:
            anchors.add(word)
    return anchors


def _prior_absent_conflicts_with_present(
    prior: Dict[str, Any],
    current_text: str,
    *,
    subject_features: str = "",
) -> bool:
    """True when an earlier absent record denied this same entity appearance."""
    if not _prior_absent_is_definitive_region_rejection(prior):
        return False

    prior_text = _normalize_detection_text(
        f"{prior.get('location_description', '')} {prior.get('reasoning', '')}"
    )
    prior_anchors = _extract_entity_anchors(f"{prior_text} {subject_features}")
    current_anchors = _extract_entity_anchors(f"{current_text} {subject_features}")
    if not prior_anchors or not current_anchors:
        return False
    return (
        len(prior_anchors & current_anchors)
        >= _SCENE_PRIOR_ABSENT_CONFLICT_MIN_ANCHORS
    )


def _is_high_certainty_temporal_detection(record: Dict[str, Any]) -> bool:
    """True when detection is strong enough to allow first appearance mid-scene."""
    if record.get("presence_gated"):
        return False
    confidence = float(record.get("confidence", 0.0) or 0.0)
    if confidence < KEYFRAME_ENTITY_PRESENCE_MIN_CONFIDENCE:
        return False
    visibility = str(record.get("visibility_quality", "") or "").strip().lower()
    if visibility != "clear":
        return False
    clarity = str(record.get("localization_clarity", "") or "").strip().lower()
    if clarity != "high":
        return False
    completeness = str(
        record.get("entity_visibility_completeness", "") or ""
    ).strip().lower()
    if completeness not in {"sufficient", "partial"}:
        return False
    if not record.get("identity_verifiable_from_visible_parts"):
        return False

    visible_parts = _visible_parts_list(record)
    has_face = any(part in {"face", "head"} for part in visible_parts)
    area_fraction = float(record.get("approximate_area_fraction", 0.0) or 0.0)
    if area_fraction < KEYFRAME_ENTITY_MIN_AREA_FRACTION and not has_face:
        return False

    if completeness == "sufficient" and (has_face or area_fraction >= 0.08):
        return True
    return has_face and area_fraction >= 0.08


def _location_descriptions_conflict(
    prior_location: str,
    current_location: str,
    *,
    subject_features: str = "",
) -> bool:
    prior_text = _normalize_detection_text(prior_location)
    current_text = _normalize_detection_text(current_location)
    if not prior_text or not current_text:
        return False

    prior_spatial = _extract_spatial_cues(prior_text)
    current_spatial = _extract_spatial_cues(current_text)
    if (
        prior_spatial
        and current_spatial
        and not _spatial_cues_overlap(prior_spatial, current_spatial)
    ):
        return True

    prior_struct = _extract_entity_anchors(prior_text) & _SCENE_LOCATION_STRUCTURE_ANCHORS
    current_struct = _extract_entity_anchors(current_text) & _SCENE_LOCATION_STRUCTURE_ANCHORS
    if prior_struct and current_struct and not (prior_struct & current_struct):
        return True

    subject_text = _normalize_detection_text(subject_features)
    if "railing" in subject_text and "railing" in prior_text and "pole" in current_text:
        return True
    return False


def _preferred_scene_location_record(
    priors_present: Sequence[Dict[str, Any]],
    *,
    subject_features: str = "",
) -> Dict[str, Any]:
    subject_text = _normalize_detection_text(subject_features)
    if "railing" in subject_text:
        for prior in priors_present:
            prior_loc = _normalize_detection_text(
                str(prior.get("location_description", "") or "")
            )
            if "railing" in prior_loc or "rail" in prior_loc:
                return prior
    return priors_present[0]


def stabilize_scene_entity_location(
    record: Dict[str, Any],
    scene_prior_records: Sequence[Dict[str, Any]] | None,
    *,
    subject_features: str = "",
) -> Dict[str, Any]:
    """Keep entity location stable across keyframes in the same scene when drift is implausible."""
    if not record.get("present"):
        return record

    priors = [
        entry for entry in (scene_prior_records or []) if isinstance(entry, dict)
    ]
    priors_present = [entry for entry in priors if entry.get("present")]
    if not priors_present:
        return record

    current_loc = str(record.get("location_description", "") or "").strip()
    preferred = _preferred_scene_location_record(
        priors_present,
        subject_features=subject_features,
    )
    preferred_loc = str(preferred.get("location_description", "") or "").strip()
    if not preferred_loc or not current_loc:
        return record
    if not _location_descriptions_conflict(
        preferred_loc,
        current_loc,
        subject_features=subject_features,
    ):
        return record

    out = dict(record)
    out["location_description"] = preferred_loc
    out["location_stabilized_from"] = str(
        preferred.get("keyframe_stem", "") or "prior"
    ).strip()
    prior_reason = str(record.get("reasoning", "") or "").strip()
    out["reasoning"] = (
        f"location stabilized from {out['location_stabilized_from']}; {prior_reason}"
        if prior_reason
        else f"location stabilized from {out['location_stabilized_from']}"
    ).strip()
    return out


def format_scene_prior_detection_block(
    entity_specs: Sequence[Dict[str, Any]],
    scene_prior_by_instruction: Dict[str, Sequence[Dict[str, Any]]] | None,
) -> str:
    """Format same-scene earlier keyframe detections for batch VLM context."""
    prior_map = scene_prior_by_instruction or {}
    lines: List[str] = []
    for spec in entity_specs:
        iid = str(spec.get("instruction_id", "")).strip()
        eid = str(spec.get("entity_id", "")).strip()
        priors = [
            entry
            for entry in (prior_map.get(iid) or [])
            if isinstance(entry, dict)
        ]
        if not priors:
            lines.append(
                f"- {iid} / {eid}: no earlier keyframe detections in this scene yet."
            )
            continue
        lines.append(f"- {iid} / {eid} — earlier keyframes in this scene (time order):")
        for prior in priors:
            stem = str(prior.get("keyframe_stem", "") or "unknown").strip()
            present = bool(prior.get("present"))
            location = (
                str(prior.get("location_description", "") or "").strip()
                or str(prior.get("vlm_location_description", "") or "").strip()
                or "N/A"
            )
            # reasoning text is intentionally excluded from the VLM prompt to
            # avoid biasing the detector with prior free-text justifications.
            lines.append(
                f"  * {stem}: present={present} | location={location}"
            )
    return "\n".join(lines) if lines else "(no entities)"


def apply_scene_temporal_presence_gate(
    record: Dict[str, Any],
    scene_prior_records: Sequence[Dict[str, Any]] | None,
    *,
    subject_features: str = "",
) -> Dict[str, Any]:
    """Reject present=true that contradicts earlier keyframe detections in the same scene."""
    priors = [
        entry for entry in (scene_prior_records or []) if isinstance(entry, dict)
    ]
    if not priors or not record.get("present"):
        return record

    priors_present = [entry for entry in priors if entry.get("present")]
    priors_absent = [entry for entry in priors if not entry.get("present")]
    reject_reasons: List[str] = []

    current_text = _normalize_detection_text(
        f"{record.get('location_description', '')} {record.get('reasoning', '')}"
    )
    current_spatial = _extract_spatial_cues(current_text)

    for prior in priors_absent:
        if _prior_absent_conflicts_with_present(
            prior,
            current_text,
            subject_features=subject_features,
        ):
            stem = str(prior.get("keyframe_stem", "") or "prior").strip()
            reject_reasons.append(
                f"prior_absent_entity_descriptor_conflict_in_{stem}"
            )
            break

    if not reject_reasons:
        for prior in priors_absent:
            if not _prior_absent_is_definitive_region_rejection(prior):
                continue
            prior_text = _normalize_detection_text(
                f"{prior.get('location_description', '')} {prior.get('reasoning', '')}"
            )
            prior_spatial = _extract_spatial_cues(prior_text)
            if _spatial_cues_overlap(prior_spatial, current_spatial):
                stem = str(prior.get("keyframe_stem", "") or "prior").strip()
                reject_reasons.append(
                    f"present_at_region_previously_rejected_in_{stem}"
                )
                break

    has_absent_descriptor_conflict = any(
        reason.startswith("prior_absent_entity_descriptor_conflict_in_")
        for reason in reject_reasons
    )
    high_certainty = _is_high_certainty_temporal_detection(record)
    if (
        not priors_present
        and len(priors_absent) >= _SCENE_PRIOR_ABSENT_STREAK_MIN
        and not high_certainty
        and not has_absent_descriptor_conflict
    ):
        reject_reasons.append(
            f"sudden_present_after_{len(priors_absent)}_prior_absent_keyframes"
        )

    if not reject_reasons:
        return record

    out = dict(record)
    if out.get("vlm_present") is None:
        out["vlm_present"] = bool(out.get("present"))
    preserved_loc = _detection_location_text(out)
    out["present"] = False
    if bool(out.get("vlm_present")) and preserved_loc:
        out["location_description"] = preserved_loc
        out["vlm_location_description"] = preserved_loc
    else:
        out["location_description"] = ""
    out["temporal_gated"] = True
    out["temporal_reject_reasons"] = list(dict.fromkeys(reject_reasons))
    prior_reason = str(record.get("reasoning", "") or "").strip()
    gate_note = "temporal gated: " + "; ".join(reject_reasons)
    out["reasoning"] = (
        f"{prior_reason} | {gate_note}" if prior_reason else gate_note
    ).strip()
    return out


def _has_visible_candidate_at_neighbor_region(
    absent_record: Dict[str, Any],
    neighbor_record: Dict[str, Any],
) -> bool:
    """True when an absent record still describes a visible candidate near a neighbor hit."""
    absent_text = _normalize_detection_text(
        f"{absent_record.get('location_description', '')} "
        f"{absent_record.get('reasoning', '')} "
        f"{absent_record.get('initial_detection_location_description', '')} "
        f"{absent_record.get('initial_detection_reasoning', '')}"
    )
    neighbor_text = _normalize_detection_text(
        f"{neighbor_record.get('location_description', '')} {neighbor_record.get('reasoning', '')}"
    )
    if not absent_text or not neighbor_text:
        return False
    if not any(token in absent_text for token in ("partial", "partially", "obscured", "cropped", "visible", "person", "individual", "left", "right", "foreground")):
        return False
    absent_spatial = _extract_spatial_cues(absent_text)
    neighbor_spatial = _extract_spatial_cues(neighbor_text)
    return _spatial_cues_overlap(absent_spatial, neighbor_spatial)


def _definitive_absence_blocks_neighbor_recovery(
    absent_record: Dict[str, Any],
    *,
    subject_features: str = "",
) -> bool:
    """True when VLM explicitly saw the area and rejected the target identity."""
    text = _normalize_detection_text(
        f"{absent_record.get('reasoning', '')} "
        f"{absent_record.get('initial_detection_reasoning', '')} "
        f"{absent_record.get('location_description', '')} "
        f"{absent_record.get('initial_detection_location_description', '')}"
    )
    if not text:
        return False

    explicit_absence = any(
        phrase in text
        for phrase in (
            "not present",
            "not visible",
            "not in the frame",
            "no man",
            "no person matching",
            "no character matches",
            "only women",
            "other figures are women",
            "does not match",
            "wrong person",
            "look-alike",
        )
    )
    if not explicit_absence:
        return False

    subject = (subject_features or "").lower()
    if "flat cap" in subject and any(
        phrase in text
        for phrase in (
            "flat cap",
            "brown vest",
            "other figures are women",
            "only women",
            "women",
        )
    ):
        return True
    if "suspenders" in subject and any(
        phrase in text for phrase in ("no individual", "not visible", "not present", "cap")
    ):
        return True
    return "identity_not_verifiable_from_visible_parts" in text or "does not match" in text


def _neighbor_edge_foreground_deletion_continuity_ok(
    neighbor_record: Dict[str, Any],
) -> bool:
    """Allow conservative recovery for delete targets at frame edges/foreground.

    Some VLM passes miss an edge-cropped foreground delete target completely in
    one keyframe, while the adjacent keyframe strongly detects the same target
    at the same border. This fallback intentionally applies only to strong,
    substantial neighbor detections in edge/foreground regions.
    """
    if not neighbor_record.get("present"):
        return False
    confidence = float(neighbor_record.get("confidence", 0.0) or 0.0)
    if confidence < KEYFRAME_ENTITY_REMOVAL_MIN_CONFIDENCE:
        return False

    text = _normalize_detection_text(
        f"{neighbor_record.get('location_description', '')} "
        f"{neighbor_record.get('reasoning', '')}"
    )
    if not any(
        cue in text
        for cue in (
            "left edge",
            "right edge",
            "extreme left",
            "extreme right",
            "far left",
            "far right",
            "left side",
            "right side",
            "foreground",
        )
    ):
        return False

    visible_parts = _visible_parts_list(neighbor_record)
    if not visible_parts or all(part in _PARTIAL_LIMB_PARTS for part in visible_parts):
        return False
    if not any(
        part in {"face", "head", "torso", "upper_body", "body", "full_body", "full_figure"}
        for part in visible_parts
    ):
        return False

    area_fraction = float(neighbor_record.get("approximate_area_fraction", 0.0) or 0.0)
    return area_fraction >= 0.10


def _neighbor_strong_deletion_continuity_ok(
    neighbor_record: Dict[str, Any],
) -> bool:
    """True when an adjacent keyframe strongly locates the same delete target."""
    if not neighbor_record.get("present"):
        return False
    confidence = float(neighbor_record.get("confidence", 0.0) or 0.0)
    if confidence < KEYFRAME_ENTITY_NEIGHBOR_RECOVERY_MIN_CONFIDENCE:
        return False
    visible_parts = _visible_parts_list(neighbor_record)
    if not any(
        part in {"face", "head", "hair", "torso", "upper_body", "body", "full_body", "full_figure", "back"}
        for part in visible_parts
    ):
        return False
    area_fraction = float(neighbor_record.get("approximate_area_fraction", 0.0) or 0.0)
    return area_fraction >= KEYFRAME_ENTITY_MIN_AREA_FRACTION


def _neighbor_strong_entity_continuity_ok(
    neighbor_record: Dict[str, Any],
) -> bool:
    """True when an adjacent keyframe strongly locates the same tracked entity."""
    if not neighbor_record.get("present"):
        return False
    confidence = float(neighbor_record.get("confidence", 0.0) or 0.0)
    if confidence < KEYFRAME_ENTITY_NEIGHBOR_RECOVERY_MIN_CONFIDENCE:
        return False
    visible_parts = _visible_parts_list(neighbor_record)
    if not any(part in _IDENTITY_CORE_PARTS for part in visible_parts):
        return False
    if all(part in _PARTIAL_LIMB_PARTS for part in visible_parts):
        return False
    area_fraction = float(neighbor_record.get("approximate_area_fraction", 0.0) or 0.0)
    return area_fraction >= KEYFRAME_WIDE_SHOT_MIN_AREA_FRACTION


def recover_scene_presence_from_neighbor_keyframes(
    location_records_by_keyframe: Dict[str, Dict[str, Dict[str, Any]]],
    keyframe_order: Sequence[str],
    instruction_by_id: Dict[str, EntityInstruction],
) -> int:
    """Recover same-scene false negatives using adjacent keyframe strong detections.

    This repairs single-keyframe false negatives without using prior edited
    images as references. For non-removal edits, recovery requires visible
    current-frame evidence (for example a consistency-pass location that was
    gated by confidence) plus a strong adjacent detection of the same
    instruction_id.
    """
    corrected_count = 0
    ordered = [stem for stem in keyframe_order if stem in location_records_by_keyframe]
    for idx, keyframe_stem in enumerate(ordered):
        records = location_records_by_keyframe.get(keyframe_stem) or {}
        for iid, record in list(records.items()):
            if record.get("present"):
                continue
            instr = instruction_by_id.get(iid)
            if instr is None:
                continue
            targets_removal = _edit_targets_removal(instr.edit_prompt)

            neighbor_candidates: List[Tuple[str, Dict[str, Any]]] = []
            for neighbor_idx in (idx - 1, idx + 1):
                if neighbor_idx < 0 or neighbor_idx >= len(ordered):
                    continue
                neighbor_stem = ordered[neighbor_idx]
                neighbor = (
                    location_records_by_keyframe.get(neighbor_stem, {}).get(iid) or {}
                )
                if neighbor.get("present"):
                    neighbor_candidates.append((neighbor_stem, neighbor))

            for neighbor_stem, neighbor in neighbor_candidates:
                if float(neighbor.get("confidence", 0.0) or 0.0) < KEYFRAME_ENTITY_REMOVAL_MIN_CONFIDENCE:
                    continue
                has_local_candidate = _has_visible_candidate_at_neighbor_region(
                    record,
                    neighbor,
                )
                definitive_absence = _definitive_absence_blocks_neighbor_recovery(
                    record,
                    subject_features=instr.subject_features,
                )
                has_edge_foreground_continuity = (
                    targets_removal
                    and _neighbor_edge_foreground_deletion_continuity_ok(neighbor)
                    and not definitive_absence
                )
                has_strong_neighbor_continuity = (
                    (
                        _neighbor_strong_deletion_continuity_ok(neighbor)
                        if targets_removal
                        else _neighbor_strong_entity_continuity_ok(neighbor)
                    )
                    and not definitive_absence
                )
                has_current_frame_recovery_evidence = (
                    has_local_candidate
                    or bool(str(record.get("vlm_location_description", "") or "").strip())
                    or bool(str(record.get("initial_detection_location_description", "") or "").strip())
                )
                if not targets_removal and not has_current_frame_recovery_evidence:
                    continue
                if (
                    not has_local_candidate
                    and not has_edge_foreground_continuity
                    and not has_strong_neighbor_continuity
                ):
                    continue

                recovered = dict(record)
                neighbor_visible = _visible_parts_list(neighbor)
                recovery_reason = (
                    "neighbor edge/foreground continuity"
                    if has_edge_foreground_continuity and not has_local_candidate
                    else (
                        "adjacent strong delete-target continuity"
                        if targets_removal
                        else "adjacent strong entity-track continuity"
                    )
                    if has_strong_neighbor_continuity and not has_local_candidate
                    else "neighbor-frame visible candidate continuity"
                )
                recovered.update({
                    "present": True,
                    "vlm_present": True,
                    "confidence": max(
                        float(record.get("confidence", 0.0) or 0.0),
                        KEYFRAME_ENTITY_REMOVAL_MIN_CONFIDENCE
                        if targets_removal
                        else KEYFRAME_ENTITY_SCENE_CONTINUITY_MIN_CONFIDENCE,
                    ),
                    "location_description": (
                        str(record.get("location_description", "") or "").strip()
                        or f"same visible candidate region as {neighbor_stem}: "
                        f"{str(neighbor.get('location_description', '') or '').strip()}"
                    ),
                    "visibility_quality": "clear",
                    "localization_clarity": "high",
                    "entity_visibility_completeness": "partial",
                    "visible_parts": neighbor_visible or ["torso"],
                    "identity_verifiable_from_visible_parts": True,
                    "approximate_area_fraction": max(
                        float(record.get("approximate_area_fraction", 0.0) or 0.0),
                        min(float(neighbor.get("approximate_area_fraction", 0.0) or 0.0), 0.15),
                        KEYFRAME_ENTITY_MIN_AREA_FRACTION,
                    ),
                    "neighbor_presence_recovered": True,
                    "neighbor_presence_source_keyframe": neighbor_stem,
                    "pre_neighbor_recovery_present": bool(record.get("present")),
                    "pre_neighbor_recovery_reasoning": str(record.get("reasoning", "") or ""),
                    "reasoning": (
                        f"{recovery_reason} recovery from {neighbor_stem}; "
                        f"{str(record.get('reasoning', '') or '').strip()}"
                    ).strip("; "),
                })
                gated = apply_keyframe_entity_presence_gate(
                    recovered,
                    edit_prompt=instr.edit_prompt,
                    subject_features=instr.subject_features,
                    min_confidence=(
                        KEYFRAME_ENTITY_REMOVAL_MIN_CONFIDENCE
                        if targets_removal
                        else KEYFRAME_ENTITY_SCENE_CONTINUITY_MIN_CONFIDENCE
                    ),
                )
                if not gated.get("present"):
                    continue
                gated["neighbor_presence_recovered"] = True
                gated["neighbor_presence_source_keyframe"] = neighbor_stem
                gated["neighbor_presence_recovery_reason"] = recovery_reason
                gated["pre_neighbor_recovery_present"] = bool(record.get("present"))
                gated["pre_neighbor_recovery_reasoning"] = str(
                    record.get("reasoning", "") or ""
                )
                records[iid] = gated
                corrected_count += 1
                break
    return corrected_count


def keyframe_strip_layout(
    video_width: int,
    video_height: int,
    num_keyframes: int,
) -> Tuple[int, int]:
    """Return (cols, rows): landscape → 1×N strip, portrait → N×1 strip."""
    count = max(1, num_keyframes)
    if video_width >= video_height:
        return count, 1
    return 1, count


def infer_strip_layout_from_video(
    video_path: str,
    num_keyframes: int,
) -> Tuple[int, int]:
    try:
        width, height = probe_video_size(video_path)
    except Exception:
        width, height = 1920, 1080
    return keyframe_strip_layout(width, height, num_keyframes)


def keyframe_stem_from_entry(entry: Dict[str, Any], *, fallback_index: int) -> str:
    filename = str(entry.get("filename") or "").strip()
    if filename:
        return os.path.splitext(filename)[0]
    path = str(entry.get("path") or "").strip()
    if path:
        return os.path.splitext(os.path.basename(path))[0]
    return f"keyframe_{fallback_index:04d}"


def scene_keyframe_grid_dir(keyframes_dir: str, scene_id: str) -> str:
    return os.path.join(keyframes_dir, scene_id)


def single_keyframe_work_dir(keyframes_dir: str, scene_id: str, keyframe_stem: str) -> str:
    return os.path.join(scene_keyframe_grid_dir(keyframes_dir, scene_id), keyframe_stem)


def single_keyframe_edited_path(keyframes_dir: str, scene_id: str, keyframe_stem: str) -> str:
    return os.path.join(single_keyframe_work_dir(keyframes_dir, scene_id, keyframe_stem), "edited.png")


def single_keyframe_locations_path(keyframes_dir: str, scene_id: str, keyframe_stem: str) -> str:
    return os.path.join(
        single_keyframe_work_dir(keyframes_dir, scene_id, keyframe_stem),
        "entity_locations.json",
    )


def single_instruction_location_path(
    keyframes_dir: str,
    scene_id: str,
    keyframe_stem: str,
    instruction_id: str,
) -> str:
    return os.path.join(
        single_keyframe_work_dir(keyframes_dir, scene_id, keyframe_stem),
        f"{instruction_id}.location.json",
    )


def original_keyframe_grid_path(keyframes_dir: str, scene_id: str) -> str:
    return os.path.join(scene_keyframe_grid_dir(keyframes_dir, scene_id), "original_keyframe_grid.png")


def edited_keyframe_grid_path(keyframes_dir: str, scene_id: str) -> str:
    return os.path.join(scene_keyframe_grid_dir(keyframes_dir, scene_id), "edited_keyframe_grid.png")


def grid_edit_manifest_path(keyframes_dir: str, scene_id: str) -> str:
    return os.path.join(scene_keyframe_grid_dir(keyframes_dir, scene_id), "grid_edit.json")


def grid_edit_prompt_path(keyframes_dir: str, scene_id: str) -> str:
    return os.path.join(scene_keyframe_grid_dir(keyframes_dir, scene_id), "grid_edit_prompt.txt")


def grid_edit_qa_path(keyframes_dir: str, scene_id: str) -> str:
    return os.path.join(scene_keyframe_grid_dir(keyframes_dir, scene_id), "grid_edit.qa.json")


def single_keyframe_qa_path(keyframes_dir: str, scene_id: str, keyframe_stem: str) -> str:
    return os.path.join(single_keyframe_work_dir(keyframes_dir, scene_id, keyframe_stem), "edit.qa.json")


def single_keyframe_detection_path(keyframes_dir: str, scene_id: str, keyframe_stem: str) -> str:
    return os.path.join(
        single_keyframe_work_dir(keyframes_dir, scene_id, keyframe_stem),
        "entity_detection.json",
    )


def single_keyframe_location_verify_path(keyframes_dir: str, scene_id: str, keyframe_stem: str) -> str:
    return os.path.join(
        single_keyframe_work_dir(keyframes_dir, scene_id, keyframe_stem),
        "location_verify.json",
    )


def single_keyframe_edit_comparison_path(
    keyframes_dir: str,
    scene_id: str,
    keyframe_stem: str,
    *,
    attempt: int,
) -> str:
    suffix = "" if attempt <= 1 else f".attempt{attempt}"
    return os.path.join(
        single_keyframe_work_dir(keyframes_dir, scene_id, keyframe_stem),
        f"edit_comparison{suffix}.json",
    )


def build_entity_specs_for_keyframe(
    instructions: Sequence[EntityInstruction],
    ref_dir: str,
) -> Tuple[List[Dict[str, Any]], List[EntityInstruction]]:
    """Build VLM entity specs using each instruction's source identity reference."""
    entity_specs: List[Dict[str, Any]] = []
    missing: List[EntityInstruction] = []
    for instr in instructions:
        src_ref = entity_ref_src_path(ref_dir, instr.instruction_id)
        if not os.path.exists(src_ref):
            missing.append(instr)
            continue
        entity_specs.append({
            "instruction_id": instr.instruction_id,
            "entity_id": instr.entity_id,
            "identity_ref_path": src_ref,
            # Historical key name kept for compatibility with older helpers.
            "multiview_ref_path": src_ref,
            "reference_identity_context": load_reference_identity_context(
                ref_dir,
                instr.instruction_id,
            ),
            "subject_features": instr.subject_features or "",
            "edit_prompt": instr.edit_prompt or "",
            "target_instance_scope": instr.target_instance_scope or "single",
        })
    return entity_specs, missing


def build_labeled_keyframe_strip(
    image_paths: Sequence[str],
    *,
    cols: int,
    rows: int,
    cell_size: int = 512,
    label_prefix: str = "Keyframe",
) -> Tuple[Image.Image, List[str]]:
    paths = [p for p in image_paths if p and os.path.exists(p)]
    label_strip_h = 28
    panel_h = cell_size + label_strip_h
    if not paths:
        canvas = Image.new(
            "RGB",
            (cell_size * max(cols, 1), panel_h * max(rows, 1)),
            (255, 255, 255),
        )
        return canvas, []

    grid_w = cols * cell_size
    grid_h = rows * panel_h
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    labels: List[str] = []

    for idx, path in enumerate(paths):
        row, col = divmod(idx, cols)
        if row >= rows:
            break
        label = f"{label_prefix} {idx + 1}"
        labels.append(label)

        img = Image.open(path).convert("RGB")
        img.thumbnail((cell_size, cell_size), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (cell_size, cell_size), (255, 255, 255))
        ox = (cell_size - img.width) // 2
        oy = (cell_size - img.height) // 2
        panel.paste(img, (ox, oy))
        labeled = _label_entity_panel(panel, label)

        px = col * cell_size
        py = row * panel_h
        canvas.paste(labeled, (px, py))

    return canvas, labels


def save_labeled_keyframe_strip(
    image_paths: Sequence[str],
    output_path: str,
    *,
    cols: int,
    rows: int,
    cell_size: int = 512,
) -> Tuple[str, List[str]]:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    grid, labels = build_labeled_keyframe_strip(
        image_paths,
        cols=cols,
        rows=rows,
        cell_size=cell_size,
    )
    grid.save(output_path)
    return output_path, labels


def format_edit_instructions_block(instructions: Sequence[Any]) -> str:
    lines: List[str] = []
    for instr in instructions:
        iid = getattr(instr, "instruction_id", "") or instr.get("instruction_id", "")
        eid = getattr(instr, "entity_id", "") or instr.get("entity_id", "")
        action = getattr(getattr(instr, "action", None), "value", None) or instr.get("action", "")
        edit = (getattr(instr, "edit_prompt", None) or instr.get("edit_prompt") or "").strip()
        subject = (getattr(instr, "subject_features", None) or instr.get("subject_features") or "").strip()
        scope = getattr(instr, "target_instance_scope", None) or instr.get("target_instance_scope", "single")
        lines.append(
            f"- {iid} / {eid} ({action}): {edit}"
            + (f" | subject: {subject}" if subject else "")
            + f" | {format_target_instance_scope_line(scope)}"
        )
    return "\n".join(lines) if lines else "(no instructions)"


def _protected_attributes_for_edit_prompt(edit_prompt: str) -> str:
    """Describe attributes that must stay locked for a localized edit."""
    text = (edit_prompt or "").lower()
    protected: List[str] = [
        "identity",
        "pose/action/body posture",
        "face/expression/gaze/head angle",
        "local lighting/shadows",
        "occlusion and visible extent",
    ]
    if not any(token in text for token in ("cloth", "dress", "shirt", "vest", "coat", "pants", "skirt", "outfit", "uniform")):
        protected.append("all clothing/outfit colors, patterns, shape, and texture")
    if not any(token in text for token in ("face", "expression", "smile", "mouth", "eye", "gaze")):
        protected.append("facial features and emotional expression")
    if not any(token in text for token in ("body", "pose", "action", "gesture", "arm", "hand", "leg")):
        protected.append("body/hand/arm/leg positions and gestures")
    if not any(token in text for token in ("hair", "hat", "cap", "headwear")):
        protected.append("hair/headwear appearance")
    return "; ".join(dict.fromkeys(protected))


def _infer_original_entity_state_from_record(record: Dict[str, Any]) -> str:
    """Summarize pose/expression/action cues available from detection metadata."""
    bits: List[str] = []
    for key in ("viewpoint", "body_orientation", "visibility_quality"):
        text = str(record.get(key, "") or "").strip()
        if text:
            bits.append(f"{key}={text}")

    parts_raw = record.get("visible_parts") or []
    if isinstance(parts_raw, str):
        parts_raw = [parts_raw]
    parts = [str(part).strip() for part in parts_raw if str(part).strip()]
    if parts:
        bits.append(f"visible_parts={', '.join(parts)}")

    loc = str(record.get("location_description", "") or "").strip()
    if loc:
        bits.append(f"location={loc}")

    cues_raw = record.get("identity_cues") or []
    if isinstance(cues_raw, str):
        cues_raw = [cues_raw]
    cues = [str(cue).strip() for cue in cues_raw if str(cue).strip()]
    if cues:
        bits.append(f"identity_cues={', '.join(cues)}")

    for key in (
        "attachment_visibility_reasoning",
        "reasoning",
        "initial_detection_reasoning",
        "pose_and_action",
        "pose_expression",
        "visibility_state",
        "boundary_notes",
    ):
        text = str(record.get(key, "") or "").strip()
        if text and text not in bits:
            bits.append(text)

    return " | ".join(bits)


def build_default_keyframe_state_preservation_prompts(
    canonical_edit_block: str = "",
) -> tuple[str, str]:
    """Default positive/avoid guidance injected on every keyframe edit attempt."""
    positive = (
        "Preserve each target entity's original pose, body posture, action, facial expression, "
        "gaze/eye direction, mouth state, head orientation, hand/arm positions, clothing, "
        "occlusion, visible extent, local lighting, shadows, blur, and scene interaction "
        "exactly as they appear in image 1. Change ONLY the attributes explicitly named in "
        "EDIT INSTRUCTIONS."
    )
    avoid = (
        "Do not copy pose, expression, gaze, head angle, body action, clothing, or lighting from "
        "canonical RIGHT panels or prior edited keyframes. Do not redraw, replace, beautify, or "
        "relit the face/head/body. Do not transplant the reference-card person's state into "
        "the video keyframe."
    )
    block = (canonical_edit_block or "").lower()
    if any(token in block for token in ("hair", "hat", "headwear", "cap")):
        avoid += (
            " For hair/headwear edits: change ONLY the requested hair color/style and hat; "
            "keep every face pixel, expression, gaze, head turn, neck angle, and clothing unchanged."
        )
    if any(token in block for token in ("remove", "delete", "erase", "inpaint")):
        avoid += (
            " For removal edits: inpaint only the located target silhouette; do not alter any other "
            "person's pose, expression, clothing, or action."
        )
    if any(token in block for token in ("place", "put", "shoulder")):
        avoid += (
            " For placement edits: keep the target person's original body pose, expression, clothing, "
            "and lighting unchanged; add only the requested object."
        )
    return positive, avoid


def build_keyframe_retry_edit_reinforcement(
    canonical_edit_block: str,
    entity_locations_block: str,
) -> str:
    """Restate mandatory edits at the end of a retry prompt.

    QA retry guidance can be dominated by preservation/avoid language. This
    block makes the original planned edits non-negotiable on every retry.
    """
    edits = (canonical_edit_block or "").strip() or "(no edit instructions)"
    locations = (entity_locations_block or "").strip() or "(no entity locations)"
    return (
        "Re-apply EVERY planned edit below to the listed current-keyframe target locations. "
        "The QA positive/avoid notes only describe what to keep or avoid; they must never cancel, "
        "weaken, or replace these planned edits. Do NOT output the unedited original frame. "
        "Do NOT drop an edit that was completed in an earlier attempt while fixing another issue. "
        "If any retry note says to preserve, keep, or avoid changing an attribute that is explicitly "
        "the target of a planned edit below, ignore that note for the target attribute and perform "
        "the planned edit; preservation applies only to non-requested attributes and non-target regions.\n\n"
        "MANDATORY PLANNED EDITS:\n"
        f"{edits}\n\n"
        "CURRENT KEYFRAME TARGET LOCATIONS:\n"
        f"{locations}"
    )


def merge_keyframe_state_preservation_avoid(
    avoid_operations: str,
    canonical_edit_block: str = "",
) -> str:
    """Ensure retry avoid prompts always restate locked original-state constraints."""
    _, default_avoid = build_default_keyframe_state_preservation_prompts(
        canonical_edit_block
    )
    custom = (avoid_operations or "").strip()
    if not custom:
        return default_avoid
    lowered = custom.lower()
    if "do not copy pose" in lowered or "keep every face pixel" in lowered:
        return custom
    return f"{custom}\n{default_avoid}".strip()


def _pre_edit_locked_region_reasoning(edit_prompt: str) -> str:
    """LLM-facing reasoning about what must not change for this edit."""
    text = (edit_prompt or "").lower()
    locked: List[str] = [
        "Before editing, infer the original head/body orientation, facial expression, gaze, pose, clothing, and local lighting from image 1.",
        "These inferred original states are locked unless explicitly named in the edit instruction.",
    ]
    if any(token in text for token in ("hair", "hat", "cap", "headwear")):
        locked.extend([
            "Hair/headwear edit: ONLY hair color/style named by the instruction and the added/changed headwear may change.",
            "Do NOT change face shape, facial expression, eye direction, mouth state, head turn, head tilt, neck angle, clothing color/pattern/shape, body pose, hands/arms, or the subject's local lighting/shadows.",
        ])
    if _edit_targets_removal(edit_prompt):
        locked.extend([
            "Removal edit: ONLY the located target's original visible silhouette may be removed/inpainted.",
            "If the target is partially hidden by hands, foreground people, props, railings, or blur, preserve those occluding non-target pixels and remove only target-owned visible pixels behind/around them.",
            "Do NOT remove, blur, repaint, relight, or alter any second similar person, adjacent person, clothing, face, occluding object/person, or background outside the target silhouette.",
        ])
    if _edit_targets_physical_placement(edit_prompt):
        locked.extend([
            "Physical placement edit: keep the target person's original body, clothing, pose, and lighting unchanged; add only the requested object at the visible attachment point.",
        ])
    return " ".join(dict.fromkeys(locked))


def format_canonical_edit_block(instructions: Sequence[EntityInstruction]) -> str:
    lines: List[str] = []
    for instr in instructions:
        edit = (instr.edit_prompt or "").strip()
        protected = _protected_attributes_for_edit_prompt(edit)
        locked_reasoning = _pre_edit_locked_region_reasoning(edit)
        removal_note = ""
        if _edit_targets_removal(edit):
            removal_note = (
                "For removal/delete, remove the target entity's entire visible person/object silhouette "
                "in this keyframe, including all visible head, face, hair, hat/cap/headwear, clothing, "
                "torso, limbs, and accessories that belong to the target; deleting only a hat, cap, "
                "headwear, clothing patch, or other subpart is incomplete. If hands, foreground people, "
                "props, railings, or blur occlude the target, preserve those non-target occluders and "
                "remove only the target-owned visible pixels; never reveal or invent hidden target body parts. "
            )
        lines.append(
            f"- {instr.instruction_id} / {instr.entity_id}: "
            f"canonical card from entity_refs/{instr.instruction_id}_ref_canonical.png shows "
            f"LEFT=source/original entity identity to find in the keyframe, "
            f"RIGHT=edited-attribute reference for this instruction; "
            f"subject_features = {(instr.subject_features or '').strip() or 'N/A'}; "
            f"edit instruction = {edit or 'N/A'}; "
            f"{removal_note}"
            f"ONLY attributes explicitly named in edit instruction may change; "
            f"protected attributes that must remain from the original keyframe = {protected}; "
            f"pre-edit locked-region reasoning = {locked_reasoning}; "
            f"ignore any unrelated differences in the RIGHT panel; "
            f"{format_target_instance_scope_line(instr.target_instance_scope)}"
        )
    return "\n".join(lines) if lines else "(no instructions)"


def instructions_present_in_keyframe(
    instructions: Sequence[EntityInstruction],
    location_records: Dict[str, Dict[str, Any]],
) -> List[EntityInstruction]:
    """Return instructions whose entity was detected (present=true) in this keyframe."""
    return [
        instr
        for instr in instructions
        if location_records.get(instr.instruction_id, {}).get("present")
    ]


def _instruction_allows_detected_scene_promotion(instr: EntityInstruction) -> bool:
    """True when a high-confidence detection may repair missed time binding."""
    condition = instr.time_condition
    if str(condition.condition_type or "").strip().lower() != "event":
        return False
    if _edit_targets_removal(instr.edit_prompt):
        return False
    if _edit_targets_physical_placement(instr.edit_prompt):
        return False
    return True


def _edit_targets_physical_placement(edit_prompt: str) -> bool:
    """True for edits that add/place an object onto a target entity."""
    text = (edit_prompt or "").lower()
    if any(token in text for token in ("hair", "hat", "cap", "headwear")):
        return False
    return any(token in text for token in ("place", "put")) and any(
        token in text for token in ("shoulder", "hand", "arm", "torso", "body")
    )


def _subject_strong_feature_cues(subject_features: str) -> List[str]:
    subject = (subject_features or "").lower()
    return [cue for cue in _STRONG_SUBJECT_FEATURE_CUES if cue in subject]


def _unbound_promotion_subject_cues_ok(
    record: Dict[str, Any],
    instr: EntityInstruction,
) -> bool:
    """Do not promote an unbound event edit on generic identity evidence alone."""
    if _strong_current_identity_override(record):
        return True

    initial_parts = set(_normalize_visible_parts_list([
        str(part)
        for part in (record.get("initial_detection_visible_parts") or [])
        if str(part).strip()
    ]))
    current_parts = set(_visible_parts_list(record))
    has_identity_core = bool((initial_parts | current_parts) & _IDENTITY_CORE_PARTS)
    if (
        bool(record.get("initial_detection_vlm_present"))
        and has_identity_core
        and _specific_visual_identity_evidence_count(record) >= 2
        and not _initial_identity_rejected(record)
    ):
        return True

    subject_cues = _subject_strong_feature_cues(instr.subject_features)
    if not subject_cues:
        return True
    matched = _record_matches_required_subject_cues(
        record,
        subject_features=instr.subject_features,
    )
    required = 2 if len(set(subject_cues)) >= 2 else 1
    return matched >= required


def _location_record_safe_for_scene_promotion(
    record: Dict[str, Any],
    instr: EntityInstruction,
) -> bool:
    """Conservative gate for editing detected-but-unbound event instructions."""
    if not record.get("present"):
        return False
    if float(record.get("confidence", 0.0) or 0.0) < 0.90:
        return False
    if str(record.get("localization_clarity", "") or "").strip().lower() != "high":
        return False
    if float(record.get("approximate_area_fraction", 0.0) or 0.0) < 0.08:
        return False
    if all(part in _PARTIAL_LIMB_PARTS for part in _visible_parts_list(record)):
        return False

    if _edit_targets_physical_placement(instr.edit_prompt):
        attachment = _requested_attachment_point(instr.edit_prompt)
        attachment_visible = _attachment_visible_from_record(record, attachment)
        if float(record.get("confidence", 0.0) or 0.0) < 0.97:
            return False
        if not bool(record.get("identity_verifiable_from_visible_parts")):
            return False
        if attachment and attachment_visible is not True:
            return False

    initial_parts = _normalize_visible_parts_list([
        str(part)
        for part in (record.get("initial_detection_visible_parts") or [])
        if str(part).strip()
    ])
    initial_has_core = any(
        part in {"face", "head", "torso", "upper_body", "body", "full_body", "full_figure"}
        for part in initial_parts
    )
    initial_identity_ok = (
        bool(record.get("initial_detection_vlm_present"))
        and initial_has_core
        and not all(part in _PARTIAL_LIMB_PARTS for part in initial_parts)
    )
    stable_cues_ok = _stable_identity_cue_count(
        {
            **record,
            "reasoning": " ".join(
                str(record.get(key, "") or "")
                for key in ("reasoning", "initial_detection_reasoning")
            ),
        },
        subject_features=instr.subject_features,
    ) >= 2
    face_or_hair_identity_ok = _record_matches_face_or_hair_identity(record)
    stable_visual_ok = _stable_visual_identity_ok(
        record,
        _visible_parts_list(record),
        subject_features=instr.subject_features,
    )
    strong_current_override = _strong_current_identity_override(record)
    consistency_corrected_ok = (
        bool(record.get("scene_consistency_corrected"))
        and bool(record.get("vlm_present"))
        and float(record.get("confidence", 0.0) or 0.0) >= KEYFRAME_ENTITY_SCENE_CONTINUITY_MIN_CONFIDENCE
        and not record.get("presence_gated")
        and not record.get("presence_reject_reasons")
        and any(part in _IDENTITY_CORE_PARTS for part in _visible_parts_list(record))
    )
    initial_rejected_identity = _initial_identity_rejected(record)
    if initial_rejected_identity:
        # Scene consistency is useful inside already-bound scenes, but it should
        # not silently bind a previously rejected event instruction to a new
        # scene unless the current frame has very concrete identity evidence.
        if not strong_current_override:
            return False
    if not _unbound_promotion_subject_cues_ok(record, instr):
        return False

    return (
        strong_current_override
        or initial_identity_ok
        or (
            bool(record.get("identity_verifiable_from_visible_parts"))
            and _specific_visual_identity_evidence_count(record) >= 3
        )
        or stable_cues_ok
        or face_or_hair_identity_ok
        or stable_visual_ok
        or consistency_corrected_ok
    )


def _location_record_editable_for_instruction(
    record: Dict[str, Any],
    instr: EntityInstruction,
) -> bool:
    """True only when the final record is present and actionable for real editing."""
    if not record.get("present"):
        return False
    if record.get("single_entity_detection_vote_passed", True) is False:
        return False

    editability_reasons = {
        str(reason).strip()
        for reason in (record.get("editability_reject_reasons") or [])
        if str(reason).strip()
    }
    attachment_only_editability_gate = (
        _edit_targets_physical_placement(instr.edit_prompt)
        and bool(editability_reasons)
        and editability_reasons <= _PRESENCE_NEUTRAL_EDITABILITY_REASONS
    )
    if bool(record.get("editability_gated")) and not attachment_only_editability_gate:
        return False

    location = str(record.get("location_description", "") or "").strip()
    if not location:
        return False

    try:
        existence_score = float(record.get("existence_confidence_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        existence_score = 0.0
    if existence_score <= 0.0:
        return False

    visible_parts = set(_visible_parts_list(record))
    initial_parts = set(_normalize_visible_parts_list([
        str(part)
        for part in (record.get("initial_detection_visible_parts") or [])
        if str(part).strip()
    ]))
    combined_parts = visible_parts | initial_parts
    if combined_parts and all(part in _PARTIAL_LIMB_PARTS for part in combined_parts):
        return False

    if _edit_targets_physical_placement(instr.edit_prompt):
        if not _instruction_attachment_editable(record, instr):
            return False

    if _edit_targets_head_region(instr.edit_prompt) and not _edit_targets_physical_placement(instr.edit_prompt):
        reject_reasons = {
            str(reason).strip()
            for reason in (record.get("presence_reject_reasons") or [])
            if str(reason).strip()
        }
        if "head_edit_target_head_or_face_not_visible" in reject_reasons:
            return False
        if not combined_parts & {"face", "head", "hair"}:
            return False

    return True


def select_present_instructions_for_keyframe(
    bound_instructions: Sequence[EntityInstruction],
    candidate_instructions: Sequence[EntityInstruction],
    location_records: Dict[str, Dict[str, Any]],
) -> List[EntityInstruction]:
    """Select only truly editable instructions from final VLM keyframe records."""
    selected: List[EntityInstruction] = []
    seen: set[str] = set()
    for instr in bound_instructions:
        record = location_records.get(instr.instruction_id, {})
        if _location_record_editable_for_instruction(record, instr):
            selected.append(instr)
            seen.add(instr.instruction_id)

    for instr in candidate_instructions:
        if instr.instruction_id in seen:
            continue
        record = location_records.get(instr.instruction_id, {})
        if not _location_record_editable_for_instruction(record, instr):
            continue
        selected.append(instr)
        seen.add(instr.instruction_id)
    return selected


_SCREEN_REGION_TOKENS = {
    "left",
    "right",
    "center",
    "central",
    "foreground",
    "mid-ground",
    "midground",
    "middle-ground",
    "background",
    "upper",
    "lower",
    "edge",
}


def _record_location_text(record: Dict[str, Any]) -> str:
    texts: List[str] = [
        str(record.get("location_description", "") or ""),
        str(record.get("vlm_location_description", "") or ""),
    ]
    raw_candidates = record.get("candidate_evaluations") or []
    if isinstance(raw_candidates, list):
        for item in raw_candidates:
            if isinstance(item, dict):
                texts.append(str(item.get("candidate_location", "") or ""))
    return " ".join(texts).lower().replace("_", "-")


def _record_region_tokens(record: Dict[str, Any]) -> set[str]:
    text = _record_location_text(record)
    # Directional phrases describe pose/gaze, not candidate location.
    text = re.sub(r"\b(facing|towards?|toward|looking|oriented)\s+(?:the\s+)?(?:screen-)?(?:left|right|center)\b", " ", text)
    text = re.sub(r"\b(?:left|right)\s+of\s+the\s+frame\b", " ", text)
    tokens = {token for token in _SCREEN_REGION_TOKENS if token in text}
    if "central" in tokens:
        tokens.add("center")
    if "middle-ground" in tokens or "midground" in tokens:
        tokens.add("mid-ground")
    return tokens


def _record_primary_location_text(record: Dict[str, Any]) -> str:
    """Return only the target candidate's own location, excluding scene context."""
    raw_candidates = record.get("candidate_evaluations") or []
    if isinstance(raw_candidates, list):
        for item in raw_candidates:
            if isinstance(item, dict):
                loc = str(item.get("candidate_location", "") or "").strip()
                if loc:
                    text = loc.lower().replace("_", "-")
                    return re.split(
                        r"\b(?:against|with|while|nearby|next to|beside|in front of|behind)\b",
                        text,
                        maxsplit=1,
                    )[0]
    text = str(record.get("location_description", "") or "").strip().lower()
    if not text:
        text = str(record.get("vlm_location_description", "") or "").strip().lower()
    text = text.replace("_", "-")
    text = re.split(
        r"\b(?:against|with|while|nearby|next to|beside|in front of|behind)\b",
        text,
        maxsplit=1,
    )[0]
    return text.split(".", 1)[0]


def _region_tokens_from_text(text: str) -> set[str]:
    text = text.lower().replace("_", "-")
    text = re.sub(r"\b(facing|towards?|toward|looking|oriented)\s+(?:the\s+)?(?:screen-)?(?:left|right|center)\b", " ", text)
    text = re.sub(r"\b(?:left|right)\s+of\s+the\s+frame\b", " ", text)
    tokens = {token for token in _SCREEN_REGION_TOKENS if token in text}
    if "central" in tokens:
        tokens.add("center")
    if "middle-ground" in tokens or "midground" in tokens:
        tokens.add("mid-ground")
    return tokens


def _subject_gender_bucket(instr: EntityInstruction | None) -> str:
    if instr is None:
        return ""
    text = f"{instr.entity_id} {instr.subject_features}".lower()
    if any(token in text for token in ("woman", "girl", "female")):
        return "female"
    if any(token in text for token in ("man", "boy", "male")):
        return "male"
    return ""


def _record_has_front_back_contradiction(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    a_text = _record_location_text(a)
    b_text = _record_location_text(b)
    return (
        ("in front" in a_text and "behind" in b_text)
        or ("behind" in a_text and "in front" in b_text)
    )


def _records_likely_same_candidate(
    a: Dict[str, Any],
    b: Dict[str, Any],
    a_instr: EntityInstruction | None = None,
    b_instr: EntityInstruction | None = None,
) -> bool:
    a_gender = _subject_gender_bucket(a_instr)
    b_gender = _subject_gender_bucket(b_instr)
    if a_gender and b_gender and a_gender != b_gender:
        return False
    if _record_has_front_back_contradiction(a, b):
        return False

    a_primary_tokens = _region_tokens_from_text(_record_primary_location_text(a))
    b_primary_tokens = _region_tokens_from_text(_record_primary_location_text(b))
    horizontal_tokens = {"left", "right", "center", "edge"}
    depth_tokens = {"foreground", "mid-ground", "background", "edge"}
    a_primary_horizontal = a_primary_tokens & horizontal_tokens
    b_primary_horizontal = b_primary_tokens & horizontal_tokens
    if (
        a_primary_horizontal
        and b_primary_horizontal
        and not (a_primary_horizontal & b_primary_horizontal)
    ):
        return False
    a_primary_depth = a_primary_tokens & depth_tokens
    b_primary_depth = b_primary_tokens & depth_tokens
    if a_primary_depth and b_primary_depth and not (a_primary_depth & b_primary_depth):
        return False

    a_tokens = _record_region_tokens(a)
    b_tokens = _record_region_tokens(b)
    if not a_tokens or not b_tokens:
        return False
    shared = a_tokens & b_tokens
    a_depth = a_tokens & depth_tokens
    b_depth = b_tokens & depth_tokens
    if a_depth and b_depth and not (a_depth & b_depth):
        return False
    if (
        a_primary_horizontal
        and b_primary_horizontal
        and (a_primary_horizontal & b_primary_horizontal)
        and a_primary_depth
        and b_primary_depth
        and (a_primary_depth & b_primary_depth)
    ):
        return True
    if len(shared) >= 2 and (shared & {"left", "right", "center", "edge"}):
        return True
    a_loc = str(a.get("location_description", "") or "").strip().lower()
    b_loc = str(b.get("location_description", "") or "").strip().lower()
    return bool(a_loc and b_loc and a_loc == b_loc)


def _record_assignment_strength(record: Dict[str, Any], instr: EntityInstruction | None) -> float:
    score = float(record.get("confidence", 0.0) or 0.0)
    score += 0.2 * _specific_visual_identity_evidence_count(record)
    visible_parts = set(_visible_parts_list(record))
    if "face" in visible_parts:
        score += 0.2
    if visible_parts & {"head", "hair"}:
        score += 0.1
    if record.get("all_negative_recovery_checked"):
        score -= 0.25
    if instr is not None and _edit_targets_physical_placement(instr.edit_prompt):
        reasons = _physical_placement_identity_reasons(
            record,
            edit_prompt=instr.edit_prompt,
            subject_features=instr.subject_features,
        )
        score -= 0.4 * len(reasons)
    return score


def resolve_cross_entity_candidate_conflicts(
    location_records: Dict[str, Dict[str, Any]],
    *,
    instructions_by_id: Dict[str, EntityInstruction],
) -> int:
    """Ensure one visible candidate is not bound to multiple single-scope entities."""
    present_ids = [
        iid
        for iid, record in location_records.items()
        if record.get("present") or record.get("vlm_present")
    ]
    corrected = 0
    for idx, left_iid in enumerate(present_ids):
        left = location_records.get(left_iid) or {}
        left_instr = instructions_by_id.get(left_iid)
        if left_instr is not None and normalize_target_instance_scope(left_instr.target_instance_scope) != "single":
            continue
        if not left.get("present"):
            continue
        for right_iid in present_ids[idx + 1:]:
            right = location_records.get(right_iid) or {}
            right_instr = instructions_by_id.get(right_iid)
            if right_instr is not None and normalize_target_instance_scope(right_instr.target_instance_scope) != "single":
                continue
            if not right.get("present"):
                continue
            if not _records_likely_same_candidate(left, right, left_instr, right_instr):
                continue
            left_score = _record_assignment_strength(left, left_instr)
            right_score = _record_assignment_strength(right, right_instr)
            loser_iid, winner_iid = (
                (right_iid, left_iid) if left_score >= right_score else (left_iid, right_iid)
            )
            loser = location_records.get(loser_iid) or {}
            loser["present"] = False
            loser["location_description"] = ""
            loser["cross_entity_conflict_suppressed"] = True
            loser["cross_entity_conflict_winner"] = winner_iid
            reasons = list(loser.get("presence_reject_reasons") or [])
            reason = f"cross_entity_candidate_conflict_with_{winner_iid}"
            if reason not in reasons:
                reasons.append(reason)
            loser["presence_reject_reasons"] = reasons
            loser["presence_gated"] = True
            base = str(loser.get("reasoning", "") or "").strip()
            loser["reasoning"] = f"{base} | presence gated: {reason}".strip(" |")
            corrected += 1
    return corrected


def collect_canonical_ref_paths(
    ref_dir: str,
    instructions: Sequence[EntityInstruction],
) -> List[str]:
    paths: List[str] = []
    for instr in instructions:
        path = entity_ref_canonical_path(ref_dir, instr.instruction_id)
        if os.path.exists(path) and path not in paths:
            paths.append(path)
    return paths


def format_present_keyframe_locations_block(
    location_records: Dict[str, Dict[str, Any]],
    *,
    instructions_by_id: Dict[str, EntityInstruction] | None = None,
) -> str:
    """Location block for image edit — only entities selected for this keyframe."""
    lines: List[str] = []
    allowed_ids = set(instructions_by_id) if instructions_by_id is not None else None
    for iid, record in location_records.items():
        if allowed_ids is not None and iid not in allowed_ids:
            continue
        if not record.get("present"):
            continue
        loc = str(record.get("location_description", "")).strip() or "present (unspecified)"
        confidence = float(record.get("confidence", 0.0) or 0.0)
        quality = str(record.get("visibility_quality", "") or "").strip()
        instr = (instructions_by_id or {}).get(iid)
        scope_note = ""
        if instr is not None:
            scope_note = f"; {format_target_instance_scope_line(instr.target_instance_scope)}"
            if normalize_target_instance_scope(instr.target_instance_scope) == "single":
                scope_note += (
                    " Edit only this described screen instance; if multiple look-alikes exist, "
                    "all non-target look-alikes are locked/uneditable."
                )
        lines.append(
            f"- {iid}: PRESENT — confidence={confidence:.2f}; "
            f"visibility_quality={quality or 'unknown'}; location={loc}{scope_note}"
        )
    return "\n".join(lines) if lines else "(no entities to edit in this keyframe)"


def format_visibility_constraints_block(
    location_records: Dict[str, Dict[str, Any]],
    *,
    instruction_ids: Sequence[str] | None = None,
    instructions_by_id: Dict[str, EntityInstruction] | None = None,
) -> str:
    """Per-entity original visibility limits for edit + QA."""
    allowed = set(instruction_ids) if instruction_ids else None
    lines: List[str] = []
    for iid, record in location_records.items():
        if allowed is not None and iid not in allowed:
            continue
        if not record.get("present"):
            continue
        parts_raw = record.get("visible_parts") or []
        if isinstance(parts_raw, str):
            parts_raw = [parts_raw]
        parts = [str(p).strip() for p in parts_raw if str(p).strip()]
        completeness = str(record.get("entity_visibility_completeness", "") or "").strip()
        loc = str(record.get("location_description", "") or "").strip()
        state = _infer_original_entity_state_from_record(record)
        if not state:
            state_bits: List[str] = []
            for key in (
                "pose_and_action",
                "pose_expression",
                "visibility_state",
                "reasoning",
                "boundary_notes",
            ):
                text = str(record.get(key, "") or "").strip()
                if text:
                    state_bits.append(text)
            state = " | ".join(state_bits)
        attachment_note = ""
        removal_visibility_note = ""
        instr = (instructions_by_id or {}).get(iid)
        if instr is not None:
            locked_reasoning = _pre_edit_locked_region_reasoning(instr.edit_prompt)
            if _edit_targets_removal(instr.edit_prompt):
                removal_visibility_note = (
                    " Removal visibility rule: delete/inpaint only target-owned pixels that are visible in this original keyframe. "
                    "Hands, foreground people, props, railings, blur, and any other occluding non-target pixels are locked and must remain unchanged; "
                    "do not reveal, reconstruct, or invent target body parts hidden behind them. "
                )
            attachment = _requested_attachment_point(instr.edit_prompt)
            if attachment:
                visible = _attachment_visible_from_record(record, attachment)
                reasoning = str(
                    record.get("attachment_visibility_reasoning", "") or ""
                ).strip()
                visible_text = (
                    "unknown"
                    if visible is None
                    else "visible"
                    if visible
                    else "NOT visible / occluded"
                )
                body_orientation = str(record.get("body_orientation", "") or "").strip()
                left_screen = str(
                    record.get("anatomical_left_screen_side", "") or ""
                ).strip()
                right_screen = str(
                    record.get("anatomical_right_screen_side", "") or ""
                ).strip()
                side_note = (
                    f" body_orientation={body_orientation or 'unknown'}; "
                    f"anatomical left appears on screen side={left_screen or 'unknown'}; "
                    f"anatomical right appears on screen side={right_screen or 'unknown'}; "
                )
                attachment_note = (
                    f" Requested physical placement point = {attachment}; "
                    f"visibility in original keyframe = {visible_text}; "
                    f"{side_note}"
                    f"{reasoning}. "
                    f"If the requested point is not visible, the object must be hidden/absent, not drawn on the visible opposite side. "
                    f"If visible, place it on the anatomical side above, not the viewer-side label guessed from image coordinates."
                )
        lines.append(
            f"- {iid}: in the ORIGINAL keyframe, visible parts = "
            f"{', '.join(parts) if parts else 'unspecified'}; "
            f"completeness = {completeness or 'unknown'}; "
            f"location = {loc or 'unspecified'}. "
            f"{attachment_note}"
            f"{removal_visibility_note}"
            f"Pre-edit locked-region reasoning = {locked_reasoning if instr is not None else 'infer and preserve all non-requested original state from image 1'}. "
            f"Original state to preserve = {state or 'use image 1 exactly for pose, expression, action, occlusion, and lighting'}. "
            f"Edits must stay inside this visible extent — never reveal, complete, or repaint "
            f"occluded/cropped/off-frame body regions. Preserve the original pose, expression, gaze, action, "
            f"local lighting, shadows, blur, and scene interaction unless explicitly changed by the edit instruction."
        )
    return "\n".join(lines) if lines else "(no visibility constraints)"


def format_single_keyframe_locations_block(location_records: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    for iid, record in location_records.items():
        if record.get("present"):
            loc = str(record.get("location_description", "")).strip() or "present (unspecified)"
            lines.append(f"- {iid}: PRESENT — {loc}")
        else:
            lines.append(f"- {iid}: NOT PRESENT")
    return "\n".join(lines) if lines else "(no entities located)"


def load_scene_keyframe_grid_edit_inputs(
    keyframes_dir: str,
    scene_id: str,
    *,
    scenes_dir: str = "",
) -> Tuple[str, str]:
    manifest_path = grid_edit_manifest_path(keyframes_dir, scene_id)
    prompt_path = grid_edit_prompt_path(keyframes_dir, scene_id)
    edited_grid = edited_keyframe_grid_path(keyframes_dir, scene_id)

    grid_edit_prompt = ""
    if os.path.exists(prompt_path):
        try:
            grid_edit_prompt = Path(prompt_path).read_text(encoding="utf-8").strip()
        except OSError:
            pass

    if os.path.exists(manifest_path):
        try:
            data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            if not grid_edit_prompt:
                grid_edit_prompt = str(data.get("grid_edit_prompt", "") or "").strip()
            recorded_grid = str(data.get("edited_keyframe_grid", "") or "").strip()
            if recorded_grid and os.path.exists(recorded_grid):
                edited_grid = recorded_grid
        except (OSError, json.JSONDecodeError):
            pass

    if scenes_dir and (not os.path.exists(edited_grid) or not grid_edit_prompt):
        legacy_dir = os.path.join(scenes_dir, scene_id, "keyframe_grid")
        legacy_manifest = os.path.join(legacy_dir, "grid_edit.json")
        legacy_grid = os.path.join(legacy_dir, "edited_keyframe_grid.png")
        if os.path.exists(legacy_manifest):
            try:
                data = json.loads(Path(legacy_manifest).read_text(encoding="utf-8"))
                if not grid_edit_prompt:
                    grid_edit_prompt = str(data.get("grid_edit_prompt", "") or "").strip()
                if not os.path.exists(edited_grid) and os.path.exists(legacy_grid):
                    edited_grid = legacy_grid
            except (OSError, json.JSONDecodeError):
                pass

    if not os.path.exists(edited_grid):
        raise FileNotFoundError(f"Missing edited keyframe grid for {scene_id}: {edited_grid}")
    if not grid_edit_prompt:
        raise RuntimeError(f"Missing grid_edit_prompt for {scene_id} under {keyframes_dir}")
    return edited_grid, grid_edit_prompt
