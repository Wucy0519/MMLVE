"""
Module 3 — per-keyframe editing before video propagation.

Per-keyframe pipeline (5 steps):
1. VLM entity screening — for each keyframe, ask once per entity with
   keyframe + entity_refs/instr_00N_ref_src.png → presence and detailed location
2. Deterministic per-keyframe conflict handling
3. Image edit — canonical refs + locations + edit instructions
4. VLM completion QA — original + edited + refs + instructions → validate edits; retry image edit up to 3 times on failure

Detection runs for every entity in ``entity_instru``; image edit applies only to
instructions bound to the scene via ``time_instru`` that are present in the frame.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any, Dict, List

from video_editing_agent.clients.base import ModelApiClientBase, ModelApiError
from video_editing_agent.config import AgentConfig
from video_editing_agent.schemas.instructions import EntityInstruction, EntityInstructionSet
from video_editing_agent.schemas.scenes import TimeInstructionSet
from video_editing_agent.utils.keyframe_manifest_utils import load_scene_keyframe_entries
from video_editing_agent.utils.edit_qa_utils import build_keyframe_qa_avoid_operations
from video_editing_agent.utils.scene_keyframe_grid_utils import (
    absent_entity_location_record,
    build_entity_specs_for_keyframe,
    collect_canonical_ref_paths,
    edited_keyframe_grid_path,
    format_canonical_edit_block,
    format_edit_instructions_block,
    format_present_keyframe_locations_block,
    format_visibility_constraints_block,
    load_catalog_keyframe_appearance,
    merge_keyframe_state_preservation_avoid,
    merge_scene_consistency_location_record,
    grid_edit_manifest_path,
    grid_edit_prompt_path,
    grid_edit_qa_path,
    infer_strip_layout_from_video,
    keyframe_stem_from_entry,
    original_keyframe_grid_path,
    recover_scene_presence_from_neighbor_keyframes,
    save_labeled_keyframe_strip,
    scene_keyframe_grid_dir,
    select_present_instructions_for_keyframe,
    single_entity_best_attempt_recoverable,
    single_instruction_location_path,
    single_keyframe_detection_path,
    single_keyframe_edited_path,
    single_keyframe_location_verify_path,
    single_keyframe_locations_path,
    single_keyframe_qa_path,
    single_keyframe_work_dir,
)
from video_editing_agent.utils.workspace_checkpoints import (
    load_module3_keyframe_grid_checkpoint,
    persist_module3_keyframe_grid_skip,
    save_module3_keyframe_grid_manifest,
)

logger = logging.getLogger(__name__)

MAX_KEYFRAME_EDIT_RETRIES = 3
SINGLE_ENTITY_DETECTION_ATTEMPTS = 3
SINGLE_ENTITY_DETECTION_SCORE_THRESHOLD = 220.0
KEYFRAME_QA_REGRESSION_SCORE_MARGIN = 0.05
KEYFRAME_VERIFICATION_DROP_PRESERVE_MIN_CONFIDENCE = 0.75

_KEYFRAME_QA_REGRESSION_CRITICAL_FLAGS = (
    "frame_structure_preserved",
    "edit_completed",
    "canonical_reference_alignment_ok",
    "original_entity_state_preserved",
    "unrelated_edit_changes_absent",
    "background_unedited_regions_preserved",
)


def _qa_flag_pass_count(qa: Dict, flags: tuple[str, ...]) -> int:
    return sum(1 for flag in flags if bool(qa.get(flag)))


def _qa_failed_aspect_count(qa: Dict) -> int:
    raw = qa.get("failed_aspects") or []
    if isinstance(raw, list):
        return len([item for item in raw if str(item).strip()])
    return 1 if str(raw).strip() else 0


def _qa_background_drift_fraction(qa: Dict) -> float:
    metrics = qa.get("background_drift_metrics") or {}
    if not isinstance(metrics, dict):
        return 1.0
    try:
        return float(metrics.get("background_drift_outside_changed_fraction", 1.0) or 0.0)
    except (TypeError, ValueError):
        return 1.0


def _qa_background_violation_count(qa: Dict) -> int:
    metrics = qa.get("background_drift_metrics") or {}
    if not isinstance(metrics, dict):
        return 0
    violations = metrics.get("background_drift_violation_cells") or []
    if isinstance(violations, list):
        return len(violations)
    return 1 if violations else 0


def _verification_absence_has_identity_contradiction(record: Dict[str, Any]) -> bool:
    """Return true when verification gives concrete evidence step-1 picked the wrong subject."""
    fields = [
        str(record.get("reasoning", "") or ""),
        str(record.get("location_description", "") or ""),
        str(record.get("vlm_location_description", "") or ""),
        " ".join(str(item) for item in (record.get("presence_reject_reasons") or [])),
    ]
    text = " ".join(fields).lower()
    return any(
        phrase in text
        for phrase in (
            "wrong person",
            "different person",
            "different individual",
            "look-alike",
            "does not match",
            "not the same",
            "identity conflicts",
            "conflicts with",
            "blonde_candidate_conflicts",
            "strong_identity_features_do_not_match_subject",
            "narrative_context_without_reference_identity_match",
            "not present",
            "does not appear",
            "absent from the frame",
            "not in the frame",
            "no visible candidate",
            "no individual matching",
            "only woman",
            "only a woman",
            "only the woman",
            "only girl",
            "only a girl",
            "only the girl",
            "no man",
        )
    )


def _verification_has_strong_face_hair_evidence(verified: Dict[str, Any]) -> bool:
    """Check whether step-2 verification provides strong, specific face/hair identity evidence.

    This is used to decide whether verification is allowed to override a
    step-1 absent verdict. The VLM step-1 sometimes incorrectly rejects an
    entity by invoking character names, actor names, or narrative roles
    (e.g. "this is Jack/Leonardo DiCaprio, not the target"), even when the
    face and hair clearly match the reference. When step-2 verify corrects
    such a false rejection with concrete face/hair identity evidence, the
    correction should be respected.

    Criteria (ALL must hold):
    1. candidate_evaluations contains at least one "present" decision
    2. The present candidate's visible_parts include face, head, or hair
    3. The candidate's identity_matches cite face/hair/facial structure
    4. The candidate has no concrete identity_conflicts (uncertain ones allowed)
    5. The approximate_area_fraction is significant (> 0.04) — not a tiny fragment
    """
    candidates = verified.get("candidate_evaluations") or []
    if not isinstance(candidates, list):
        return False

    face_hair_tokens = (
        "face", "hair", "facial", "hairline", "hair part", "hair silhouette",
        "jaw", "nose", "eye", "eyebrow", "cheek", "chin", "profile",
        "middle-parted", "middle parted", "hair color", "hair style",
    )

    area_fraction = float(verified.get("approximate_area_fraction", 0.0) or 0.0)
    if area_fraction < 0.04:
        return False

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        if str(cand.get("decision", "") or "").lower() != "present":
            continue

        cand_parts = set(str(p) for p in (cand.get("visible_parts") or []))
        if not (cand_parts & {"face", "head", "hair"}):
            continue

        identity_matches = " ".join(
            str(m) for m in (cand.get("identity_matches") or [])
        ).lower()
        if not any(token in identity_matches for token in face_hair_tokens):
            continue

        conflicts = [
            str(c) for c in (cand.get("identity_conflicts") or [])
            if str(c).strip() and not str(c).strip().lower().startswith("uncertain")
        ]
        if conflicts:
            continue

        return True

    return False


def _preserve_detected_presence_when_verification_drops(
    detected: Dict[str, Any],
    verified: Dict[str, Any],
) -> Dict[str, Any]:
    """Keep strong step-1 detections when step-2 merely omits or gates them."""
    if verified.get("present") or not (detected.get("present") or detected.get("vlm_present")):
        return verified
    if _verification_absence_has_identity_contradiction(verified):
        return verified

    detected_conf = float(detected.get("confidence", 0.0) or 0.0)
    if detected_conf < KEYFRAME_VERIFICATION_DROP_PRESERVE_MIN_CONFIDENCE:
        return verified
    detected_location = str(
        detected.get("location_description")
        or detected.get("vlm_location_description")
        or ""
    ).strip()
    if not detected_location:
        return verified

    merged = dict(detected)
    merged.update({
        "verified": False,
        "verification_drop_preserved_step1_detection": True,
        "verification_absence_reasoning": str(verified.get("reasoning", "") or ""),
        "pre_preserve_verified_present": bool(verified.get("present")),
        "initial_detection_reasoning": str(detected.get("reasoning", "") or ""),
        "initial_detection_location_description": detected_location,
        "initial_detection_visible_parts": detected.get("visible_parts") or [],
        "initial_detection_vlm_present": bool(
            detected.get("vlm_present", detected.get("present"))
        ),
    })
    return merged


def _json_file_loads(path: str) -> Dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _nonempty_file(path: str) -> bool:
    return bool(path and os.path.exists(path) and os.path.getsize(path) > 0)


def _keyframe_qa_quality_regressed(current: Dict, previous: Dict | None) -> bool:
    """Return true when a failed retry is clearly worse than the prior result."""
    if not previous or current.get("passed"):
        return False
    if previous.get("passed") and not current.get("passed"):
        return True

    current_critical = _qa_flag_pass_count(
        current,
        _KEYFRAME_QA_REGRESSION_CRITICAL_FLAGS,
    )
    previous_critical = _qa_flag_pass_count(
        previous,
        _KEYFRAME_QA_REGRESSION_CRITICAL_FLAGS,
    )
    if current_critical < previous_critical:
        return True

    for flag in _KEYFRAME_QA_REGRESSION_CRITICAL_FLAGS:
        if previous.get(flag) is True and current.get(flag) is False:
            return True

    current_bg_drift = _qa_background_drift_fraction(current)
    previous_bg_drift = _qa_background_drift_fraction(previous)
    if current_bg_drift > previous_bg_drift + 0.03:
        return True
    if _qa_background_violation_count(current) > _qa_background_violation_count(previous):
        return True

    current_score = float(current.get("score", 0.0) or 0.0)
    previous_score = float(previous.get("score", 0.0) or 0.0)
    if current_score + KEYFRAME_QA_REGRESSION_SCORE_MARGIN < previous_score:
        return True

    return _qa_failed_aspect_count(current) > _qa_failed_aspect_count(previous) + 1


def _qa_text(qa: Dict) -> str:
    fields = [
        str(qa.get("feedback", "") or ""),
        str(qa.get("retry_focus_prompt", "") or ""),
        str(qa.get("positive_prompt", "") or ""),
    ]
    failed = qa.get("failed_aspects") or []
    if isinstance(failed, list):
        fields.extend(str(item) for item in failed)
    else:
        fields.append(str(failed))
    comparison = qa.get("edit_comparison") or {}
    if isinstance(comparison, dict):
        fields.append(str(comparison.get("summary", "") or ""))
        for op in comparison.get("observed_edit_operations") or []:
            if isinstance(op, dict):
                fields.append(str(op.get("operation", "") or ""))
                fields.append(str(op.get("change_description", "") or ""))
    return " ".join(fields).lower()


def _qa_has_planned_removal_missing(qa: Dict) -> bool:
    text = _qa_text(qa)
    return "planned removal missing" in text or "removal of the man" in text and "ignored" in text


def _qa_has_new_entity_hallucination(qa: Dict) -> bool:
    text = _qa_text(qa)
    return any(
        phrase in text
        for phrase in (
            "new character",
            "new characters",
            "new person",
            "new woman",
            "new man",
            "new entity",
            "different person",
            "another person",
            "replaced with",
            "replaced by",
            "replacement person",
            "substitute",
            "inserted a completely different",
            "added to the scene",
            "pasted person",
            "pasted full body",
            "pasted full-body",
            "reference-card person",
            "canonical target person",
            "full-body replacement",
            "full body replacement",
            "look-alike",
            "leonardo",
            "introduced into the background",
        )
    )


def _qa_has_physical_placement_failure(qa: Dict) -> bool:
    text = _qa_text(qa)
    return any(
        phrase in text
        for phrase in (
            "wrong shoulder",
            "right shoulder instead of",
            "left shoulder instead of",
            "wrong relative size",
            "incorrect relative size",
            "too large",
            "too small",
            "oversized",
            "floating",
            "floats",
            "sticker-like",
            "pasted/sticker",
            "pasted-in asset",
            "pasted flat",
            "missing contact shadow",
            "perspective inconsistent",
            "wrong orientation",
            "incorrect orientation",
        )
    )


def _qa_has_protected_state_drift(qa: Dict) -> bool:
    """True when QA reports changes to locked pose/face/expression/gaze/state."""
    text = _qa_text(qa)
    return any(
        phrase in text
        for phrase in (
            "protected original state drifted",
            "original_entity_state_preserved",
            "face changed",
            "face appears altered",
            "face completely altered",
            "facial expression",
            "expression changed",
            "expression drift",
            "gaze changed",
            "gaze drift",
            "head orientation",
            "head turned",
            "head turn",
            "head tilt",
            "identity drift",
            "altered/replaced",
            "replaced by a different",
            "redrawn face",
            "pasted face",
            "pasted head",
            "pasted body",
            "pasted full body",
            "pasted full-body",
            "reference-card-like",
            "reference card",
            "full-body replacement",
            "full body replacement",
            "pose changed",
            "pose drift",
            "action changed",
            "lighting changed",
            "lighting drift",
            "relit",
            "re-lit",
        )
    )


def _qa_has_background_drift(qa: Dict) -> bool:
    text = _qa_text(qa)
    return any(
        phrase in text
        for phrase in (
            "background changed",
            "background was changed",
            "background_unedited_regions_preserved",
            "background object changed",
            "background object added",
            "added chair",
            "new chair",
            "extra chair",
            "chair appeared",
            "added furniture",
            "new furniture",
            "extra furniture",
            "added bench",
            "new bench",
            "added prop",
            "new prop",
            "wall patch",
            "repainted wall",
            "background texture",
            "texture shift",
            "inpaint bleed",
            "outside the target silhouette",
            "unrelated background",
            "far-right woman",
            "far right woman",
            "rightmost woman",
            "right edge woman",
            "non-target person",
            "non-target woman",
        )
    )


def _keyframe_qa_selection_rank(qa: Dict) -> tuple:
    """Rank failed attempts so the final artifact is the least-bad result.

    Background preservation is weighted early because a completed edit that
    repaints unrelated regions is usually worse than a slightly weaker edit
    whose original scene remains intact.
    """
    score = float(qa.get("score", 0.0) or 0.0)
    return (
        bool(qa.get("passed")),
        bool(qa.get("frame_structure_preserved")),
        bool(qa.get("background_unedited_regions_preserved")),
        not _qa_has_background_drift(qa),
        -_qa_background_violation_count(qa),
        -_qa_background_drift_fraction(qa),
        bool(qa.get("unrelated_edit_changes_absent")),
        bool(qa.get("edit_completed")),
        bool(qa.get("canonical_reference_alignment_ok")),
        not _qa_has_planned_removal_missing(qa),
        not _qa_has_physical_placement_failure(qa),
        not _qa_has_new_entity_hallucination(qa),
        bool(qa.get("original_entity_state_preserved")),
        not _qa_has_protected_state_drift(qa),
        bool(qa.get("photorealistic_scene_integration_ok")),
        _qa_flag_pass_count(qa, _KEYFRAME_QA_REGRESSION_CRITICAL_FLAGS),
        score,
        -_qa_failed_aspect_count(qa),
        -int(qa.get("physical_attempt", qa.get("attempt", 0)) or 0),
    )


def _edit_block_targets_removal(canonical_block: str) -> bool:
    text = (canonical_block or "").lower()
    return any(token in text for token in ("remove", "delete", "erase", "inpaint"))


def _sanitize_positive_prompt_for_removal(positive_prompt: str, canonical_block: str) -> str:
    """Drop positive feedback that conflicts with mandatory removal edits."""
    text = (positive_prompt or "").strip()
    if not text or not _edit_block_targets_removal(canonical_block):
        return text
    lowered = text.lower()
    conflict_phrases = (
        "presence of the man",
        "keep the man",
        "maintain the man",
        "leave the man",
        "do not remove the target man",
        "do not remove the man",
    )
    if any(phrase in lowered for phrase in conflict_phrases):
        return (
            "Preserve frame structure, lighting, non-target people, and background outside "
            "the removal silhouette. Keep any other correctly completed non-removal edits, "
            "but still remove every located removal target."
        )
    return text


def _sanitize_positive_prompt_for_retry(positive_prompt: str, canonical_block: str) -> str:
    """Keep safe positive retry guidance without cancelling mandatory edits."""
    text = (positive_prompt or "").strip()
    if not text:
        return ""
    if _edit_block_targets_removal(canonical_block):
        return _sanitize_positive_prompt_for_removal(text, canonical_block)
    lowered = text.lower()
    conflict_phrases = (
        "do not apply",
        "do not edit",
        "not applicable",
        "omit if not found",
        "skip the edit",
        "leave the target unchanged",
        "keep the target unchanged",
        "maintain the target unchanged",
        "preserve the target unchanged",
    )
    if any(phrase in lowered for phrase in conflict_phrases):
        return (
            "Keep the original frame structure, target identity/state, non-target people, "
            "and background outside the exact edit regions. Keep any correctly completed "
            "planned edit attributes, but still perform every mandatory planned edit."
        )
    return text


def _sanitize_avoid_prompt_for_removal(avoid_prompt: str, canonical_block: str) -> str:
    """Remove ambiguous 'do not remove target' wording from retry guidance."""
    text = (avoid_prompt or "").strip()
    if not text or not _edit_block_targets_removal(canonical_block):
        return text
    lowered = text.lower()
    ambiguous_phrases = (
        "do not remove the target man from the frame and replace",
        "do not remove the man from the frame and replace",
        "do not remove the target man and replace",
        "do not remove the man and replace",
    )
    if not any(phrase in lowered for phrase in ambiguous_phrases):
        return text
    return (
        "Remove every located removal target by inpainting only its original silhouette. "
        "Do not replace the removed target with other people, new characters, or unrelated background. "
        "Preserve all non-target people and background outside the removal silhouette."
    )


class SceneKeyframeGridEditor:
    """Edit scene keyframes one-by-one, then assemble strip for Module 4."""

    def __init__(self, config: AgentConfig, api_client: ModelApiClientBase) -> None:
        self.config = config
        self.api_client = api_client
        self._shots_analysis_index: Dict[str, Dict[str, Any]] | None = None

    def _load_shots_analysis_index(self) -> Dict[str, Dict[str, Any]]:
        """Return shot/scene id -> story analysis, cached for entity reasoning."""
        if self._shots_analysis_index is not None:
            return self._shots_analysis_index

        index: Dict[str, Dict[str, Any]] = {}
        path = self.config.shots_analysis_path
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                for shot in data.get("shots") or []:
                    if not isinstance(shot, dict):
                        continue
                    keys = {
                        str(shot.get("shot_id", "") or "").strip(),
                        str(shot.get("scene_id", "") or "").strip(),
                    }
                    for key in list(keys):
                        if key.startswith("shot_"):
                            keys.add("scene_" + key.removeprefix("shot_"))
                        if key.startswith("scene_"):
                            keys.add("shot_" + key.removeprefix("scene_"))
                    for key in keys:
                        if key:
                            index[key] = shot
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to load shots analysis context: %s", exc)

        self._shots_analysis_index = index
        return index

    def _scene_story_context(
        self,
        scene_id: str,
        scene: Any,
        current_keyframe_stem: str = "",
    ) -> str:
        """Compact narrative context used to disambiguate partial entity views.

        When ``current_keyframe_stem`` is provided, only the plot description
        and the matching keyframe's description are included. Descriptions of
        OTHER keyframes in the same scene are deliberately excluded to prevent
        the verification VLM from cross-referencing (and sometimes copy-pasting
        reasoning from) neighboring keyframes.
        """
        shot = self._load_shots_analysis_index().get(scene_id)
        if shot is None and scene_id.startswith("scene_"):
            shot = self._load_shots_analysis_index().get(
                "shot_" + scene_id.removeprefix("scene_")
            )

        parts: List[str] = []
        if shot:
            plot = str(shot.get("plot_description", "") or "").strip()
            if plot:
                parts.append(f"plot: {plot}")
            keyframes = shot.get("keyframes") or []
            for idx, item in enumerate(keyframes[:6], start=1):
                if not isinstance(item, dict):
                    continue
                # When verifying a specific keyframe, skip descriptions of
                # other keyframes to avoid cross-keyframe context leakage.
                if current_keyframe_stem:
                    candidate_stem = f"keyframe_{idx:04d}"
                    if candidate_stem != current_keyframe_stem:
                        continue
                desc = str(item.get("description", "") or "").strip()
                role = str(item.get("role", "") or "").strip()
                if desc:
                    parts.append(
                        f"keyframe_{idx:04d}"
                        f"{f' ({role})' if role else ''}: {desc}"
                    )

        if not parts:
            scene_desc = str(getattr(scene, "description", "") or "").strip()
            if scene_desc:
                parts.append(f"scene description: {scene_desc}")

        if not parts:
            return "(no scene story context available)"

        text = "\n".join(f"- {part}" for part in parts)
        return text[:4500]

    @staticmethod
    def _single_entity_detection_score(record: Dict[str, Any]) -> float:
        """Rank repeated VLM detections by existence score (gating removed)."""
        score = record.get("existence_confidence_score")
        if score is None:
            score = float(record.get("confidence", 0.0) or 0.0) * 100.0
        return max(0.0, min(100.0, float(score or 0.0)))

    @staticmethod
    def _summarize_single_entity_attempt(
        record: Dict[str, Any],
        attempt: int,
    ) -> Dict[str, Any]:
        """Keep repeated detection diagnostics compact in sidecar JSON."""
        raw_score = float(record.get("existence_confidence_score", 0.0) or 0.0)
        effective_score = raw_score if record.get("present") else 0.0
        return {
            "attempt": attempt,
            "present": bool(record.get("present")),
            "vlm_present": bool(record.get("vlm_present", record.get("present"))),
            "confidence": float(record.get("confidence", 0.0) or 0.0),
            "existence_confidence_score": raw_score,
            "existence_confidence_score_raw": float(
                record.get("existence_confidence_score_raw", raw_score) or 0.0
            ),
            "existence_confidence_score_calibrated": bool(
                record.get("existence_confidence_score_calibrated")
            ),
            "existence_confidence_score_calibration_reasons": list(
                record.get("existence_confidence_score_calibration_reasons") or []
            ),
            "effective_existence_confidence_score": effective_score,
            "presence_gated": bool(record.get("presence_gated")),
            "presence_reject_reasons": list(record.get("presence_reject_reasons") or []),
            "location_description": str(record.get("location_description", "") or ""),
            "visible_parts": list(record.get("visible_parts") or []),
            "reasoning": str(record.get("reasoning", "") or ""),
        }

    @staticmethod
    def _single_entity_effective_existence_score(record: Dict[str, Any]) -> float:
        """Return the 0-100 score that counts toward the three-run threshold.

        Gating removed: the score reflects the VLM's existence confidence
        regardless of the (now informational) present flag.
        """
        score = record.get("existence_confidence_score")
        if score is None:
            score = float(record.get("confidence", 0.0) or 0.0) * 100.0
        return max(0.0, min(100.0, float(score or 0.0)))

    async def _locate_entity_on_single_keyframe_with_vote(
        self,
        *,
        keyframe_path: str,
        entity_multiview_ref_path: str,
        instruction_id: str,
        entity_id: str,
        subject_features: str,
        edit_prompt: str,
        scene_id: str,
        keyframe_stem: str,
        prior_detection: Dict[str, Any] | None = None,
        catalog_appearance: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Run repeated single-entity VLM screening and accept all-true votes OR score-sum > threshold."""
        attempts: List[Dict[str, Any]] = []
        detection_prior = prior_detection or catalog_appearance
        for attempt_idx in range(1, SINGLE_ENTITY_DETECTION_ATTEMPTS + 1):
            try:
                record = await self.api_client.locate_entity_on_single_keyframe(
                    keyframe_path=keyframe_path,
                    entity_multiview_ref_path=entity_multiview_ref_path,
                    instruction_id=instruction_id,
                    entity_id=entity_id,
                    subject_features=subject_features,
                    edit_prompt=edit_prompt,
                    prior_detection=detection_prior,
                )
            except Exception as exc:
                logger.warning(
                    "%s/%s: single-entity screening attempt %d/%d failed for %s: %s",
                    scene_id,
                    keyframe_stem,
                    attempt_idx,
                    SINGLE_ENTITY_DETECTION_ATTEMPTS,
                    instruction_id,
                    exc,
                )
                record = absent_entity_location_record(
                    instruction_id,
                    entity_id,
                    reasoning=f"single-entity screening attempt {attempt_idx} failed: {exc}",
                )
                record["single_entity_screening_attempt_error"] = str(exc)
            record["single_entity_screening_attempt"] = attempt_idx
            attempts.append(record)

        present_votes = sum(1 for record in attempts if bool(record.get("present")))
        existence_score_sum = sum(
            self._single_entity_effective_existence_score(record)
            for record in attempts
        )
        all_votes_true_passed = present_votes == SINGLE_ENTITY_DETECTION_ATTEMPTS
        score_sum_passed = existence_score_sum > SINGLE_ENTITY_DETECTION_SCORE_THRESHOLD
        best = max(attempts, key=self._single_entity_detection_score) if attempts else {}
        best_attempt_recovered = bool(best) and single_entity_best_attempt_recoverable(
            best,
            edit_prompt=edit_prompt,
            subject_features=subject_features,
        )
        # Detection is valid ONLY when the vote threshold (all 3/3 true) OR
        # the score-sum threshold (> SINGLE_ENTITY_DETECTION_SCORE_THRESHOLD)
        # is met. The best_attempt_recovered path is NOT a valid bypass —
        # it was previously used to rescue borderline single-attempt hits,
        # but that defeated the purpose of having two hard gates.
        detection_valid = all_votes_true_passed or score_sum_passed
        # If all three votes are true, trust the strongest present attempt even if
        # calibration kept the summed score below threshold. Otherwise fall back to
        # score-bearing attempts for the score-sum rule.
        present_records = [
            record for record in attempts if bool(record.get("present"))
        ]
        scored_records = [
            record
            for record in attempts
            if self._single_entity_effective_existence_score(record) > 0.0
        ]
        if detection_valid:
            candidate_records = present_records if all_votes_true_passed else scored_records
            final = dict(max(candidate_records or attempts, key=self._single_entity_detection_score))
            final["present"] = True
        else:
            final = absent_entity_location_record(
                instruction_id,
                entity_id,
                reasoning=(
                    "single-entity screening absent: requires either "
                    f"{SINGLE_ENTITY_DETECTION_ATTEMPTS}/{SINGLE_ENTITY_DETECTION_ATTEMPTS} true votes "
                    f"or score sum > {SINGLE_ENTITY_DETECTION_SCORE_THRESHOLD:.0f}; got "
                    f"votes={present_votes}/{SINGLE_ENTITY_DETECTION_ATTEMPTS}, "
                    f"score_sum={existence_score_sum:.1f}"
                ),
            )
            if best:
                final["score_vote_best_attempt"] = self._summarize_single_entity_attempt(
                    best,
                    int(best.get("single_entity_screening_attempt", 0) or 0),
                )

        final["single_entity_detection_attempts"] = SINGLE_ENTITY_DETECTION_ATTEMPTS
        final["single_entity_detection_present_votes"] = present_votes
        final["single_entity_detection_all_votes_true_passed"] = all_votes_true_passed
        final["single_entity_detection_existence_score_sum"] = existence_score_sum
        final["single_entity_detection_score_threshold"] = (
            SINGLE_ENTITY_DETECTION_SCORE_THRESHOLD
        )
        final["single_entity_detection_score_sum_passed"] = score_sum_passed
        final["single_entity_detection_vote_passed"] = detection_valid
        final["single_entity_detection_attempt_records"] = [
            self._summarize_single_entity_attempt(record, idx)
            for idx, record in enumerate(attempts, start=1)
        ]
        final["single_entity_detection_catalog_reconciled"] = False
        final["single_entity_detection_catalog_available"] = bool(
            catalog_appearance and catalog_appearance.get("present")
        )
        return final

    async def run(
        self,
        time_instru: TimeInstructionSet,
        entity_instru: EntityInstructionSet,
    ) -> Dict[str, str]:
        """Return scene_id → edited keyframe grid image path."""
        logger.info("SceneKeyframeGridEditor.run started (per-keyframe mode)")
        instr_by_id = {i.instruction_id: i for i in entity_instru.instructions}
        # time_instru binding is no longer used to filter which entities are
        # detected/edited. Every entity from entity_instru is a candidate on
        # every keyframe; the VLM detection + editability gate decides
        # whether an edit is actually applied.
        all_instructions = list(entity_instru.instructions)
        ref_dir = os.path.join(self.config.workspace_dir, "entity_refs")

        if self.config.resume_from_checkpoints:
            scene_grids, skipped = load_module3_keyframe_grid_checkpoint(self.config)
        else:
            scene_grids, skipped = {}, {}

        for scene in time_instru.scenes:
            scene_id = scene.scene_id

            # Use ALL entities for every scene — time_instru binding is no
            # longer used to filter which entities are detected/edited.
            instructions = all_instructions

            if (
                self.config.resume_from_checkpoints
                and self._scene_keyframe_grid_checkpoint_ready(
                    scene_id=scene_id,
                    instructions=instructions,
                    detection_instructions=instructions,
                )
            ):
                out_path = edited_keyframe_grid_path(self.config.keyframes_dir, scene_id)
                logger.info(
                    "%s: completed keyframe-grid checkpoint found — skip image edit",
                    scene_id,
                )
                scene_grids[scene_id] = out_path
                skipped.pop(scene_id, None)
                continue

            # Detect and edit ALL entities on every scene. The VLM detection
            # + editability gate decides whether an edit is actually applied.
            try:
                out_path = await self._edit_scene_keyframes(
                    scene=scene,
                    scene_id=scene_id,
                    instructions=instructions,
                    detection_instructions=instructions,
                    ref_dir=ref_dir,
                )
                scene_grids[scene_id] = out_path
                skipped.pop(scene_id, None)
            except Exception as exc:
                logger.error(
                    "Scene %s per-keyframe edit failed: %s",
                    scene_id,
                    exc,
                    exc_info=True,
                )
                persist_module3_keyframe_grid_skip(
                    self.config,
                    scene_id,
                    str(exc),
                    skipped,
                )
                continue

            save_module3_keyframe_grid_manifest(
                self.config,
                scene_grids,
                skipped_scenes=skipped,
                time_instru=time_instru,
            )

        logger.info(
            "SceneKeyframeGridEditor.run done — %d scene grids",
            len(scene_grids),
        )
        return scene_grids

    def _scene_keyframe_grid_checkpoint_ready(
        self,
        *,
        scene_id: str,
        instructions: List[EntityInstruction],
        detection_instructions: List[EntityInstruction],
    ) -> bool:
        """True when a scene's per-keyframe edit artifacts are complete enough to reuse."""
        manifest_path = grid_edit_manifest_path(self.config.keyframes_dir, scene_id)
        qa_path = grid_edit_qa_path(self.config.keyframes_dir, scene_id)
        edited_grid = edited_keyframe_grid_path(self.config.keyframes_dir, scene_id)
        manifest = _json_file_loads(manifest_path)
        qa_manifest = _json_file_loads(qa_path)
        if not manifest or not qa_manifest or not _nonempty_file(edited_grid):
            return False
        if manifest.get("mode") != "per_keyframe_v2":
            return False
        if str(manifest.get("edited_keyframe_grid", "") or "") and not _nonempty_file(
            str(manifest.get("edited_keyframe_grid"))
        ):
            return False

        work_dirs = manifest.get("keyframe_work_dirs") or []
        if not isinstance(work_dirs, list) or not work_dirs:
            return False
        expected_count = int(manifest.get("keyframe_count", 0) or 0)
        if expected_count and len(work_dirs) != expected_count:
            return False

        qa_by_keyframe = {
            str(item.get("keyframe_stem", "") or ""): item
            for item in (qa_manifest.get("per_keyframe_qa") or [])
            if isinstance(item, dict)
        }
        qa_enabled = bool(qa_manifest.get("qa_enabled", self.config.keyframe_edit_qa))
        for work_dir_raw in work_dirs:
            work_dir = str(work_dir_raw or "")
            if not work_dir:
                return False
            keyframe_stem = os.path.basename(work_dir.rstrip(os.sep))
            edited_path = single_keyframe_edited_path(
                self.config.keyframes_dir,
                scene_id,
                keyframe_stem,
            )
            if not _nonempty_file(edited_path):
                return False
            locations = _json_file_loads(
                single_keyframe_locations_path(
                    self.config.keyframes_dir,
                    scene_id,
                    keyframe_stem,
                )
            )
            if locations is None:
                return False
            present_instructions = select_present_instructions_for_keyframe(
                instructions,
                detection_instructions,
                locations,
            )
            if present_instructions:
                qa_sidecar = single_keyframe_qa_path(
                    self.config.keyframes_dir,
                    scene_id,
                    keyframe_stem,
                )
                if qa_enabled and (_json_file_loads(qa_sidecar) is None):
                    return False
                if qa_enabled and keyframe_stem not in qa_by_keyframe:
                    return False

        return True

    async def _edit_scene_keyframes(
        self,
        *,
        scene,
        scene_id: str,
        instructions: List[EntityInstruction],
        detection_instructions: List[EntityInstruction],
        ref_dir: str,
    ) -> str:
        entries = load_scene_keyframe_entries(scene)
        keyframe_entries = [
            e for e in entries if e.get("path") and os.path.exists(str(e.get("path")))
        ]
        if not keyframe_entries:
            raise RuntimeError(f"No keyframes for {scene_id}")

        source_clip = os.path.join(self.config.scenes_dir, scene_id, f"{scene_id}.mp4")
        cols, rows = infer_strip_layout_from_video(source_clip, len(keyframe_entries))

        work_dir = scene_keyframe_grid_dir(self.config.keyframes_dir, scene_id)
        os.makedirs(work_dir, exist_ok=True)

        original_paths = [str(e["path"]) for e in keyframe_entries]
        orig_grid = original_keyframe_grid_path(self.config.keyframes_dir, scene_id)
        edited_grid = edited_keyframe_grid_path(self.config.keyframes_dir, scene_id)

        save_labeled_keyframe_strip(original_paths, orig_grid, cols=cols, rows=rows)

        canonical_paths = collect_canonical_ref_paths(ref_dir, instructions)
        if instructions and not canonical_paths:
            raise RuntimeError(
                f"No canonical refs for scene {scene_id} — run entity ref builder first"
            )

        qa_enabled = self.config.keyframe_edit_qa


        edited_paths: List[str] = []
        per_keyframe_qa: List[dict] = []
        keyframe_items: List[Dict[str, str]] = []
        location_records_by_keyframe: Dict[str, Dict[str, Dict]] = {}
        selected_instructions_by_id: Dict[str, EntityInstruction] = {}

        # Run all detections first. A later keyframe can expose a same-scene
        # false negative in an earlier keyframe, so editing must wait until the
        # scene-level consistency pass finishes.
        for idx, entry in enumerate(keyframe_entries):
            keyframe_path = str(entry["path"])
            keyframe_stem = keyframe_stem_from_entry(entry, fallback_index=idx + 1)
            kf_work = single_keyframe_work_dir(
                self.config.keyframes_dir, scene_id, keyframe_stem
            )
            os.makedirs(kf_work, exist_ok=True)

            # Build per-keyframe story context that only includes the current
            # keyframe's description. Including other keyframes' descriptions
            # caused the VLM to cross-reference (and sometimes copy-paste
            # reasoning from) neighboring keyframes.
            scene_story_context = self._scene_story_context(
                scene_id, scene, current_keyframe_stem=keyframe_stem
            )

            location_records = await self._detect_and_verify_entities_on_keyframe(
                keyframe_path=keyframe_path,
                scene_id=scene_id,
                keyframe_stem=keyframe_stem,
                instructions=detection_instructions,
                ref_dir=ref_dir,
                scene_story_context=scene_story_context,
            )
            keyframe_items.append({
                "keyframe_id": keyframe_stem,
                "path": keyframe_path,
                "scene_story_context": scene_story_context,
            })
            location_records_by_keyframe[keyframe_stem] = location_records

        # The revised pre-edit identification flow is intentionally per-keyframe
        # and per-entity. Do not run the older scene-level VLM consistency or
        # neighbor recovery passes here, because they batch entities/keyframes
        # after the dedicated single-entity screening and can re-promote absent
        # targets. QA/retry behavior after editing remains unchanged.

        for item in keyframe_items:
            keyframe_stem = item["keyframe_id"]
            location_records = location_records_by_keyframe[keyframe_stem]
            locations_sidecar = single_keyframe_locations_path(
                self.config.keyframes_dir, scene_id, keyframe_stem
            )
            with open(locations_sidecar, "w", encoding="utf-8") as fh:
                json.dump(location_records, fh, indent=2, ensure_ascii=False)
            for iid, record in location_records.items():
                sidecar = single_instruction_location_path(
                    self.config.keyframes_dir,
                    scene_id,
                    keyframe_stem,
                    iid,
                )
                with open(sidecar, "w", encoding="utf-8") as fh:
                    json.dump(record, fh, indent=2, ensure_ascii=False)

        for item in keyframe_items:
            keyframe_path = item["path"]
            keyframe_stem = item["keyframe_id"]
            location_records = location_records_by_keyframe[keyframe_stem]
            present_instructions = select_present_instructions_for_keyframe(
                instructions,
                detection_instructions,
                location_records,
            )
            for instr in present_instructions:
                selected_instructions_by_id.setdefault(instr.instruction_id, instr)
            present_location_block = format_present_keyframe_locations_block(
                location_records,
                instructions_by_id={
                    i.instruction_id: i for i in present_instructions
                },
            )
            edited_path = single_keyframe_edited_path(
                self.config.keyframes_dir, scene_id, keyframe_stem
            )

            if not present_instructions:
                logger.info(
                    "%s/%s: no present entities — skip image edit, keep original",
                    scene_id,
                    keyframe_stem,
                )
                os.makedirs(os.path.dirname(edited_path) or ".", exist_ok=True)
                shutil.copy2(keyframe_path, edited_path)
                edited_paths.append(edited_path)
                continue

            present_canonical_paths = collect_canonical_ref_paths(
                ref_dir, present_instructions
            )
            if not present_canonical_paths:
                logger.warning(
                    "%s/%s: present entities but missing canonical refs — keep original",
                    scene_id,
                    keyframe_stem,
                )
                os.makedirs(os.path.dirname(edited_path) or ".", exist_ok=True)
                shutil.copy2(keyframe_path, edited_path)
                edited_paths.append(edited_path)
                continue

            present_canonical_block = format_canonical_edit_block(present_instructions)
            visibility_constraints_block = format_visibility_constraints_block(
                location_records,
                instruction_ids=[i.instruction_id for i in present_instructions],
                instructions_by_id={
                    i.instruction_id: i for i in present_instructions
                },
            )
            present_instruction_ids = [i.instruction_id for i in present_instructions]

            edited_path = await self._edit_single_keyframe_with_qa(
                scene_id=scene_id,
                keyframe_stem=keyframe_stem,
                keyframe_path=keyframe_path,
                canonical_paths=present_canonical_paths,
                location_block=present_location_block,
                canonical_block=present_canonical_block,
                visibility_constraints_block=visibility_constraints_block,
                scene_story_context=item.get("scene_story_context", ""),
                instruction_ids=present_instruction_ids,
                entity_location_records=location_records,
                qa_enabled=qa_enabled,
            )
            edited_paths.append(edited_path)

            qa_sidecar = single_keyframe_qa_path(
                self.config.keyframes_dir, scene_id, keyframe_stem
            )
            if os.path.exists(qa_sidecar):
                try:
                    with open(qa_sidecar, encoding="utf-8") as fh:
                        qa_data = json.load(fh)
                    per_keyframe_qa.append(qa_data)
                except (OSError, json.JSONDecodeError):
                    pass

        save_labeled_keyframe_strip(edited_paths, edited_grid, cols=cols, rows=rows)

        edit_block = format_edit_instructions_block(
            list(selected_instructions_by_id.values())
        )
        prompt_path = grid_edit_prompt_path(self.config.keyframes_dir, scene_id)
        with open(prompt_path, "w", encoding="utf-8") as fh:
            fh.write(edit_block)

        qa_sidecar = grid_edit_qa_path(self.config.keyframes_dir, scene_id)
        with open(qa_sidecar, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "scene_id": scene_id,
                    "mode": "per_keyframe_v2",
                    "pipeline_steps": [
                        "detect",
                        "verify_location",
                        "image_edit",
                        "completion_qa",
                    ],
                    "original_grid": orig_grid,
                    "edited_grid": edited_grid,
                    "layout": {"cols": cols, "rows": rows},
                    "keyframe_count": len(keyframe_entries),
                    "per_keyframe_qa": per_keyframe_qa,
                    "qa_enabled": qa_enabled,
                    "max_edit_retries": MAX_KEYFRAME_EDIT_RETRIES,
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )

        manifest = grid_edit_manifest_path(self.config.keyframes_dir, scene_id)
        with open(manifest, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "scene_id": scene_id,
                    "mode": "per_keyframe_v2",
                    "original_keyframe_grid": orig_grid,
                    "edited_keyframe_grid": edited_grid,
                    "grid_edit_prompt": edit_block,
                    "grid_edit_prompt_path": prompt_path,
                    "cols": cols,
                    "rows": rows,
                    "keyframe_count": len(keyframe_entries),
                    "instruction_ids": list(selected_instructions_by_id),
                    "detection_instruction_ids": [
                        i.instruction_id for i in detection_instructions
                    ],
                    "keyframe_work_dirs": [
                        single_keyframe_work_dir(
                            self.config.keyframes_dir,
                            scene_id,
                            keyframe_stem_from_entry(e, fallback_index=i + 1),
                        )
                        for i, e in enumerate(keyframe_entries)
                    ],
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )

        logger.info("%s per-keyframe edit → %s", scene_id, edited_grid)
        return edited_grid

    async def _detect_and_verify_entities_on_keyframe(
        self,
        *,
        keyframe_path: str,
        scene_id: str,
        keyframe_stem: str,
        instructions: List[EntityInstruction],
        ref_dir: str,
        scene_story_context: str = "",
    ) -> Dict[str, Dict]:
        """Steps 1–2: VLM entity detection then single-pass location verification."""
        entity_specs, missing_ref_instructions = build_entity_specs_for_keyframe(
            instructions,
            ref_dir,
        )
        for spec in entity_specs:
            iid = str(spec.get("instruction_id", "") or "").strip()
            if not iid:
                continue
            catalog = load_catalog_keyframe_appearance(
                self.config.entity_keyframe_appearances_path,
                iid,
                keyframe_path,
            )
            if catalog:
                spec["catalog_appearance"] = catalog
        location_records: Dict[str, Dict] = {}

        for instr in missing_ref_instructions:
            location_records[instr.instruction_id] = absent_entity_location_record(
                instr.instruction_id,
                instr.entity_id,
                reasoning="missing source identity reference",
            )

        if not entity_specs:
            return location_records

        # Step 1: screen entities on this keyframe one by one. Each VLM call sees
        # only the current keyframe and this instruction's source identity reference
        # (entity_refs/instr_00N_ref_src.png), avoiding batch attention competition
        # between easy and difficult entities.
        detection_records: Dict[str, Dict[str, Any]] = {}
        for spec in entity_specs:
            iid = str(spec.get("instruction_id", "") or "").strip()
            eid = str(spec.get("entity_id", "") or "").strip()
            ref_path = str(
                spec.get("identity_ref_path")
                or spec.get("multiview_ref_path")
                or ""
            ).strip()
            if not iid:
                continue
            record = await self._locate_entity_on_single_keyframe_with_vote(
                keyframe_path=keyframe_path,
                entity_multiview_ref_path=ref_path,
                instruction_id=iid,
                entity_id=eid,
                subject_features=str(spec.get("subject_features", "") or ""),
                edit_prompt=str(spec.get("edit_prompt", "") or ""),
                scene_id=scene_id,
                keyframe_stem=keyframe_stem,
                prior_detection=spec.get("prior_detection"),
                catalog_appearance=spec.get("catalog_appearance"),
            )
            record["single_entity_screening"] = True
            record["identity_ref_path"] = ref_path
            detection_records[iid] = record
        detection_sidecar = single_keyframe_detection_path(
            self.config.keyframes_dir, scene_id, keyframe_stem
        )
        with open(detection_sidecar, "w", encoding="utf-8") as fh:
            json.dump(detection_records, fh, indent=2, ensure_ascii=False)

        logger.info(
            "%s/%s: step-1 detection — %d present / %d total (valid when %d/%d votes are true or score sum > %.0f)",
            scene_id,
            keyframe_stem,
            sum(1 for r in detection_records.values() if r.get("present")),
            len(detection_records),
            SINGLE_ENTITY_DETECTION_ATTEMPTS,
            SINGLE_ENTITY_DETECTION_ATTEMPTS,
            SINGLE_ENTITY_DETECTION_SCORE_THRESHOLD,
        )

        # Step 2: run a same-keyframe verification pass across all entities.
        # This catches step-1 hallucinations / wrong-person bindings that can
        # still survive repeated single-entity screening when each instruction
        # is evaluated in isolation.
        verify_error = ""
        raw_verified_records: Dict[str, Dict[str, Any]] = {}
        try:
            raw_verified_records = await self.api_client.verify_entity_locations_on_keyframe(
                keyframe_path=keyframe_path,
                entity_specs=entity_specs,
                detection_records=detection_records,
                scene_story_context=scene_story_context,
            )
        except Exception as exc:
            verify_error = str(exc)
            logger.warning(
                "%s/%s: step-2 location verification failed, preserving step-1 detections: %s",
                scene_id,
                keyframe_stem,
                exc,
            )

        if raw_verified_records:
            verified_records: Dict[str, Dict[str, Any]] = {}
            for spec in entity_specs:
                iid = str(spec.get("instruction_id", "") or "").strip()
                if not iid:
                    continue
                detected = dict(detection_records.get(iid) or {})
                verified = dict(raw_verified_records.get(iid) or {})
                merged_verified = dict(detected)
                merged_verified.update(verified)
                merged_verified["verified"] = bool(verified.get("verified", True))
                merged_verified["verification_mode"] = "single_pass_verify"

                # Step-2 verification can only DOWNGRADE present→absent, never
                # upgrade absent→present — UNLESS verification provides strong,
                # specific face/hair identity evidence. This exception catches
                # cases where step-1 incorrectly rejected the entity by invoking
                # character names / actor names / narrative roles (e.g. "this is
                # Jack / Leonardo DiCaprio, not the target"), even though the
                # face and hair clearly match the reference image. When step-2
                # corrects such a false rejection with concrete face/hair
                # identity matches and no conflicts, the correction is respected.
                detected_present = bool(detected.get("present"))
                verified_present = bool(merged_verified.get("present"))
                if not detected_present and verified_present:
                    if _verification_has_strong_face_hair_evidence(merged_verified):
                        # Allow the absent→present correction: step-2 provided
                        # strong face/hair identity evidence that overrides the
                        # false step-1 rejection.
                        merged_verified["verification_corrected_absent_to_present"] = True
                        merged_verified["verification_correction_reason"] = (
                            "strong_face_hair_identity_evidence_from_verify_pass"
                        )
                        # Set a reasonable existence score from the verify
                        # confidence so downstream editability checks don't
                        # reject on existence_score <= 0.
                        verify_conf = float(
                            merged_verified.get("confidence", 0.0) or 0.0
                        )
                        if verify_conf > 0.0:
                            merged_verified["existence_confidence_score"] = max(
                                verify_conf * 100.0, 65.0
                            )
                    else:
                        merged_verified["present"] = False
                        merged_verified["vlm_present"] = False
                        merged_verified["confidence"] = float(
                            detected.get("confidence", 0.0) or 0.0
                        )
                        merged_verified["location_description"] = str(
                            detected.get("location_description", "") or ""
                        )
                        merged_verified["visible_parts"] = list(
                            detected.get("visible_parts") or []
                        )
                        merged_verified["existence_confidence_score"] = float(
                            detected.get("existence_confidence_score", 0.0) or 0.0
                        )
                        merged_verified["verification_blocked_absent_to_present"] = True
                        merged_verified["verification_absence_reasoning"] = str(
                            verified.get("reasoning", "") or ""
                        )

                verified_records[iid] = _preserve_detected_presence_when_verification_drops(
                    detected,
                    merged_verified,
                )
        else:
            verified_records = {
                iid: {**dict(record), "verified": True, "verification_mode": "single_entity_screening"}
                for iid, record in detection_records.items()
            }
            raw_verified_records = {
                iid: dict(record) for iid, record in verified_records.items()
            }

        corrected = sum(
            1 for iid, record in verified_records.items()
            if record.get("location_corrected")
            or (
                bool(record.get("present"))
                != bool(detection_records.get(iid, {}).get("present"))
            )
            or bool(record.get("verification_drop_preserved_step1_detection"))
            or bool(record.get("catalog_appearance_used"))
        )
        if corrected:
            logger.info(
                "%s/%s: single-entity screening finalized %d entity record(s)",
                scene_id,
                keyframe_stem,
                corrected,
            )

        verify_sidecar = single_keyframe_location_verify_path(
            self.config.keyframes_dir, scene_id, keyframe_stem
        )
        with open(verify_sidecar, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "detection": detection_records,
                    "raw_verified": raw_verified_records,
                    "verified": verified_records,
                    "verification_fallback_used": bool(verify_error),
                    "verification_error": verify_error,
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )

        location_records.update(verified_records)

        for iid, record in location_records.items():
            sidecar = single_instruction_location_path(
                self.config.keyframes_dir,
                scene_id,
                keyframe_stem,
                iid,
            )
            with open(sidecar, "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=2, ensure_ascii=False)

        return location_records

    async def _edit_single_keyframe_with_qa(
        self,
        *,
        scene_id: str,
        keyframe_stem: str,
        keyframe_path: str,
        canonical_paths: List[str],
        location_block: str,
        canonical_block: str,
        visibility_constraints_block: str,
        instruction_ids: List[str],
        entity_location_records: Dict[str, Dict[str, Any]],
        qa_enabled: bool,
        scene_story_context: str = "",
    ) -> str:
        """Steps 3–5: image edit, edit comparison, completion QA with retries."""
        edited_path = single_keyframe_edited_path(
            self.config.keyframes_dir, scene_id, keyframe_stem
        )
        qa_history: List[dict] = []
        avoid_ops = ""
        positive_ops = ""
        max_attempts = MAX_KEYFRAME_EDIT_RETRIES + 1
        current_path = edited_path
        qa_passed = False
        paid_attempt = 1
        physical_attempt = 0
        previous_qa: Dict | None = None
        free_regression_retry_used = False
        reuse_same_prompt = False
        best_qa: Dict | None = None
        best_path = edited_path

        while paid_attempt <= max_attempts:
            physical_attempt += 1
            is_free_regression_retry = reuse_same_prompt
            attempt_path = (
                edited_path
                if physical_attempt == 1
                else f"{edited_path}.attempt{physical_attempt}.png"
            )

            # Step 3: image edit
            try:
                await self.api_client.edit_single_keyframe_with_canonical_refs(
                    keyframe_path=keyframe_path,
                    canonical_ref_paths=canonical_paths,
                    entity_locations_block=location_block,
                    canonical_edit_block=canonical_block,
                    visibility_constraints_block=visibility_constraints_block,
                    scene_story_context=scene_story_context,
                    output_path=attempt_path,
                    avoid_operations=avoid_ops if paid_attempt > 1 else "",
                    positive_prompt=positive_ops if paid_attempt > 1 else "",
                )
            except ModelApiError as exc:
                logger.warning(
                    "%s/%s keyframe image generation failed on "
                    "attempt %d/%d (physical attempt %d): %s",
                    scene_id,
                    keyframe_stem,
                    paid_attempt,
                    max_attempts,
                    physical_attempt,
                    exc,
                )
                if paid_attempt >= max_attempts:
                    logger.error(
                        "%s/%s keyframe image generation exhausted all %d attempts",
                        scene_id,
                        keyframe_stem,
                        max_attempts,
                    )
                    raise
                paid_attempt += 1
                continue
            current_path = attempt_path

            if not qa_enabled:
                break

            # Step 4: validate edit completion directly from original/edit/ref images.
            qa = await self.api_client.validate_keyframe_edit_completion(
                edited_keyframe_path=current_path,
                original_keyframe_path=keyframe_path,
                canonical_ref_paths=canonical_paths,
                entity_locations_block=location_block,
                canonical_edit_block=canonical_block,
                visibility_constraints_block=visibility_constraints_block,
                scene_story_context=scene_story_context,
                entity_location_records=entity_location_records,
            )
            qa["attempt"] = paid_attempt
            qa["physical_attempt"] = physical_attempt
            qa["free_regression_retry"] = is_free_regression_retry
            qa["edit_comparison"] = {
                "skipped": True,
                "reason": "completion QA directly compares original and edited images",
            }
            qa["output_path"] = current_path
            qa_history.append(qa)
            if best_qa is None or _keyframe_qa_selection_rank(qa) > _keyframe_qa_selection_rank(best_qa):
                best_qa = qa
                best_path = current_path

            if qa.get("passed"):
                qa_passed = True
                logger.info(
                    "%s/%s keyframe QA passed on attempt %d/%d (physical attempt %d)",
                    scene_id,
                    keyframe_stem,
                    paid_attempt,
                    max_attempts,
                    physical_attempt,
                )
                break

            if (
                not free_regression_retry_used
                and _keyframe_qa_quality_regressed(qa, previous_qa)
            ):
                free_regression_retry_used = True
                reuse_same_prompt = True
                qa["quality_regression_free_retry_scheduled"] = True
                qa["quality_regression_compared_to_physical_attempt"] = (
                    previous_qa.get("physical_attempt") if previous_qa else None
                )
                logger.warning(
                    "%s/%s keyframe QA regressed on paid attempt %d "
                    "(physical attempt %d) — retrying once for free with the same prompt: %s",
                    scene_id,
                    keyframe_stem,
                    paid_attempt,
                    physical_attempt,
                    qa.get("feedback", ""),
                )
                previous_qa = qa
                continue

            reuse_same_prompt = False

            if paid_attempt >= max_attempts:
                logger.warning(
                    "%s/%s — QA failed on attempt %d/%d (physical attempts=%d); "
                    "keeping best physical attempt %s: %s",
                    scene_id,
                    keyframe_stem,
                    paid_attempt,
                    max_attempts,
                    physical_attempt,
                    (best_qa or {}).get("physical_attempt", physical_attempt),
                    qa.get("feedback", ""),
                )
                break

            avoid_ops = merge_keyframe_state_preservation_avoid(
                _sanitize_avoid_prompt_for_removal(
                    build_keyframe_qa_avoid_operations(qa),
                    canonical_block,
                ),
                canonical_block,
            )
            positive_ops = _sanitize_positive_prompt_for_retry(
                str(qa.get("positive_prompt", "")).strip(),
                canonical_block,
            )
            logger.warning(
                "%s/%s keyframe QA failed on attempt %d/%d "
                "(physical attempt %d) — re-editing: %s",
                scene_id,
                keyframe_stem,
                paid_attempt,
                max_attempts,
                physical_attempt,
                qa.get("feedback", ""),
            )
            previous_qa = qa
            paid_attempt += 1

        selected_path = best_path if qa_enabled and best_qa else current_path
        selected_attempt_reason = (
            "best_attempt_by_qa_rank_passed"
            if qa_passed
            else "best_failed_attempt_by_completion_and_preservation"
        )
        state_preservation_all_attempts_failed = False
        if (
            qa_enabled
            and not qa_passed
            and best_qa
            and _qa_has_protected_state_drift(best_qa)
            and not any(
                bool(item.get("original_entity_state_preserved"))
                and not _qa_has_protected_state_drift(item)
                for item in qa_history
            )
        ):
            state_preservation_all_attempts_failed = True
            selected_attempt_reason = "best_failed_attempt_despite_protected_state_drift"
            logger.warning(
                "%s/%s — all keyframe edit attempts drifted protected state; "
                "keeping the best generated attempt rather than silently outputting the unedited original",
                scene_id,
                keyframe_stem,
            )
        if selected_path != edited_path and os.path.exists(selected_path):
            os.replace(selected_path, edited_path)

        qa_sidecar = single_keyframe_qa_path(
            self.config.keyframes_dir, scene_id, keyframe_stem
        )
        with open(qa_sidecar, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "scene_id": scene_id,
                    "keyframe_stem": keyframe_stem,
                    "original_keyframe": keyframe_path,
                    "edited_keyframe": edited_path,
                    "instruction_ids_edited": instruction_ids,
                    "pipeline_mode": "per_keyframe_v2",
                    "attempts": qa_history,
                    "re_edited": len(qa_history) > 1,
                    "qa_passed": qa_passed,
                    "max_attempts": max_attempts,
                    "physical_attempts": physical_attempt,
                    "selected_physical_attempt": (
                        (best_qa or {}).get("physical_attempt")
                        if qa_enabled and best_qa
                        else physical_attempt
                    ),
                    "selected_attempt_reason": selected_attempt_reason,
                    "state_preservation_fallback_used": False,
                    "state_preservation_all_attempts_failed": state_preservation_all_attempts_failed,
                    "free_regression_retry_used": free_regression_retry_used,
                    "final_avoid_operations": avoid_ops,
                    "final_positive_operations": positive_ops,
                    "qa_enabled": qa_enabled,
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )

        return edited_path
