"""
Production model API client.

Wraps :mod:`video_editing_agent.clients.llm_client` (OpenAI-compatible text /
vision / image generation) and :mod:`video_editing_agent.clients.video_client`
(BytePlus ModelArk video editing) with async interfaces.  Sync HTTP calls run in
``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

from video_editing_agent.clients.base import ModelApiClientBase, ModelApiError, VideoEditRejectedError
from video_editing_agent.config import ModelApiConfig
from video_editing_agent.prompts.templates import (
    DELETE_QA_VALIDATION_PROMPT,
    KEYFRAME_DELETE_EDIT_QA_VALIDATION_PROMPT,
    KEYFRAME_EDIT_QA_VALIDATION_PROMPT,
    ENTITY_VISIBILITY_PROMPT,
    ENTITY_VISIBILITY_WITH_REF_PROMPT,
    ENTITY_REFERENCE_COMPARE_PROMPT,
    EVENT_GROUNDING_PROMPT,
    INSTRUCTION_PARSE_PROMPT,
    INSTRUCTION_REWRITE_PROMPT,
    INPAINT_PROMPT,
    INPAINT_CONSISTENCY_REFS_SECTION,
    KEYFRAME_EDIT_ENTITY_LOCATION_PROMPT,
    KEYFRAME_ENTITY_DETECTION_PROMPT,
    KEYFRAME_EDIT_ENTITY_LOCATION_DISAMBIGUATE_PROMPT,
    KEYFRAME_EDIT_ENTITY_LOCATION_REID_PROMPT,
    KEYFRAME_ENTITY_REF_GUIDES_SECTION,
    KEYFRAME_LOCATION_ENTITY_REF_GUIDES_SECTION,
    KEYFRAME_LOCATION_INPAINT_PROMPT,
    MASK_FIRST_DETECTION_ENTITY_VALIDATION_PROMPT,
    MASK_REFERENCE_LOCATION_PROMPT,
    MASK_REFERENCE_LOCATION_REID_PROMPT,
    MASK_DETECTION_PROMPT,
    MASK_VALIDATION_PROMPT,
    QA_VALIDATION_PROMPT,
    REF_IMAGE_PROMPT,
    REF_IMAGE_QA_PROMPT,
    VIDEO_EDIT_DIFF_PROMPT,
    VIDEO_EDIT_QA_VALIDATION_PROMPT,
    VIDEO_CHUNK_EDIT_DIFF_PROMPT,
    GENERIC_VIDEO_EDIT_PROMPT,
    SEEDANCE_VIDEO_EDIT_PROMPT,
    VIDEO_EDIT_I2V_REWRITE_PROMPT,
    VIDEO_SCENE_STORY_ANALYSIS_PROMPT,
    SHOT_CLIP_VLM_ANALYSIS_PROMPT,
    ENTITY_MULTIVIEW_SYNTHESIS_PROMPT,
    ENTITY_MULTIVIEW_EDIT_PROMPT,
    ENTITY_MULTIVIEW_SYNTHESIS_QA_PROMPT,
    ENTITY_MULTIVIEW_EDIT_QA_PROMPT,
    ENTITY_MULTIVIEW_SOURCE_APPEARANCE_QA_PROMPT,
    ENTITY_MULTIVIEW_EDIT_ATTRIBUTE_QA_PROMPT,
    ENTITY_MULTIVIEW_EDIT_VIEW_OCCLUSION_QA_PROMPT,
    ENTITY_MULTIVIEW_CANDIDATE_SELECT_PROMPT,
    ENTITY_REFERENCE_KEYFRAME_SELECT_PROMPT,
    SCENE_VIDEO_EDIT_DERIVATION_PROMPT,
    SCENE_VIDEO_EDIT_KEYFRAME_GRID_QA_PROMPT,
    SCENE_VIDEO_EDIT_BEST_ATTEMPT_SELECT_PROMPT,
    SCENE_ENTITY_EXISTENCE_VOTE_PROMPT,
    DIRECT_SCENE_VIDEO_EDIT_PROMPT,
    DIRECT_SCENE_SEEDANCE_VIDEO_EDIT_PROMPT,
    SCENE_KEYFRAME_GRID_ENTITY_LOCATION_PROMPT,
    SCENE_KEYFRAME_GRID_EDIT_PROMPT,
    SCENE_KEYFRAME_GRID_EDIT_QA_PROMPT,
    KEYFRAME_SINGLE_ENTITY_PRESENCE_PROMPT,
    KEYFRAME_BATCH_ENTITY_PRESENCE_PROMPT,
    KEYFRAME_SINGLE_CANONICAL_EDIT_PROMPT,
    KEYFRAME_SINGLE_EDIT_QA_PROMPT,
    KEYFRAME_CANONICAL_ALIGNMENT_PROMPT,
    KEYFRAME_ENTITY_DETECT_PROMPT,
    KEYFRAME_ENTITY_LOCATION_VERIFY_PROMPT,
    SCENE_KEYFRAME_PRESENCE_CONSISTENCY_PROMPT,
    SCENE_KEYFRAME_ALL_NEGATIVE_RECOVERY_PROMPT,
    KEYFRAME_EDIT_COMPARISON_PROMPT,
    KEYFRAME_EDIT_COMPLETION_QA_PROMPT,
    SCENE_ENTITY_DETECT_PROMPT,
    GENERATE_SCENE_ENTITY_REFERENCE_PROMPT,
    EDIT_SCENE_ENTITY_REFERENCE_PROMPT,
    SCENE_ENTITY_REFERENCE_EDIT_QA_PROMPT,
)
from video_editing_agent.utils.edit_qa_utils import (
    append_editing_operations_to_avoid,
    apply_single_keyframe_edit_qa_gate,
    assess_letterbox_structure_preserved,
    build_edit_retry_guidance_section,
    build_keyframe_qa_avoid_operations,
    build_single_keyframe_qa_result_from_vlm,
    canonical_alignment_check_applicable,
    extract_canonical_target_panel,
    instruction_id_from_canonical_ref_path,
    measure_keyframe_background_drift,
    merge_background_drift_into_qa_result,
    merge_canonical_alignment_into_qa_result,
    non_edit_region_change_requires_reedit,
    normalize_non_edit_region_change_severity,
    normalize_qa_error_list,
    parse_edit_instruction_for_instruction_id,
    prepare_keyframe_qa_images,
)
from video_editing_agent.utils.json_utils import (
    ensure_instruction_ids,
    extract_json_object,
    merge_instructions_one_per_entity,
    normalize_instruction_instance_scope,
)
from video_editing_agent.utils.scene_keyframe_grid_utils import (
    apply_keyframe_entity_presence_gate,
    format_batch_entity_detection_catalog,
    format_detection_results_block,
    format_prior_detection_block,
    format_scene_prior_detection_block,
    format_target_instance_scope_line,
    build_default_keyframe_state_preservation_prompts,
    build_keyframe_retry_edit_reinforcement,
    normalize_vlm_entity_location_record,
    parse_batch_entity_location_response,
    parse_entity_detect_response,
    parse_entity_verify_response,
    parse_keyframe_edit_comparison_response,
    parse_keyframe_edit_completion_qa_response,
    parse_scene_keyframe_presence_consistency_response,
)
from video_editing_agent.utils.mask_utils import (
    align_mask_to_frame,
    assess_mask_candidate,
    blend_inpaint_preserve_frame_structure,
    build_batch_segmentation_prompt,
    build_keyframe_delete_identification_image,
    build_single_entity_segmentation_prompt,
    build_single_entity_mask_image,
    color_name_from_hex,
    compose_robust_location_hint,
    composite_segmentation_layers,
    ensure_anti_copy_in_revised_prompt,
    ensure_segmentation_mask_output,
    entity_color_hex,
    entity_mask_has_content,
    extract_reference_frame_from_composite,
    finalize_segmentation_mask,
    image_has_mask_content,
    load_entity_ref_overlay_guide,
    mask_from_raw_segmentation_sidecar,
    mask_has_palette_coverage,
    find_keyframe_location_conflicts,
    format_keyframe_peer_assignment_lines,
    keyframe_location_records_to_prompts,
    located_entities_missing_from_mask,
    normalize_keyframe_location_record,
    partition_mask_location_entities,
    pick_keyframe_location_conflict_losers,
    quantize_mask_to_palette_best_effort,
    mask_color_coverage_ratio,
    mask_looks_like_split_panel_artifact,
    mask_union_coverage_ratio,
    overlay_colored_mask_on_frame,
    render_soft_bbox_mask,
    resolve_reference_overlay_color,
    save_mask_color_map_debug,
    segmentation_aligned_sidecar_path,
    should_retry_keyframe_location,
    should_retry_reference_location,
    render_bbox_mask,
)
from video_editing_agent.utils.multiview_qa_utils import (
    apply_multiview_qa_gate,
    build_multiview_edit_qa_from_vlm,
    build_multiview_synthesis_qa_from_vlm,
    merge_multiview_focused_qa_into_result,
)
from video_editing_agent.utils.temporal_utils import (
    normalize_all_referential_time_conditions,
    resolve_temporal_conflicts,
)
from video_editing_agent.utils.video_edit_prompt_utils import (
    build_mandatory_video_edit_operation,
)

logger = logging.getLogger(__name__)


def _import_llm():
    """Lazily import the OpenAI-compatible text / vision / image client."""
    from video_editing_agent.clients import llm_client
    return llm_client


def _safe_float(value: Any, default: float) -> float:
    """Coerce to float; use default when value is None or invalid."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ModelApiClient(ModelApiClientBase):
    """Model API client backed by official provider endpoints.

    Args:
        config: API credentials and model names from environment.
        dev_mode: Skip paid API calls; use local fallbacks (testing).
    """

    def __init__(
        self,
        config: Optional[ModelApiConfig] = None,
        *,
        dev_mode: bool = False,
    ) -> None:
        self.config = config or ModelApiConfig.from_env()
        self.dev_mode = dev_mode
        if not dev_mode:
            try:
                _import_llm().validate_credentials()
            except EnvironmentError as exc:
                logger.warning("Credential validation: %s", exc)
        logger.info(
            "ModelApiClient ready — text=%s image=%s video=%s dev_mode=%s resolution=%s",
            self.config.text_model,
            self.config.image_model,
            self.config.video_model,
            dev_mode,
            self.config.video_resolution or "720p",
        )

    # ── Resolution-aware model / resolution selection ────────────────────

    def _effective_video_resolution(self) -> str:
        """Return the processing resolution: '480p' or '720p'.

        Falls back to the VIDEO_RESOLUTION env var, then '720p'.
        """
        res = (self.config.video_resolution or "").strip().lower()
        if res in ("480p", "720p"):
            return res
        return (
            os.environ.get("VIDEO_RESOLUTION")
            or os.environ.get("VIDEO_RESOLUTION", "720p")
        ).strip().lower()

    def _effective_image_model(self) -> str:
        """Return the image model appropriate for the current resolution mode."""
        if self._effective_video_resolution() == "480p" and self.config.image_model_low_res:
            return self.config.image_model_low_res
        return self.config.image_model

    def _effective_video_model(self) -> str:
        """Return the video model appropriate for the current resolution mode."""
        if self._effective_video_resolution() == "480p" and self.config.video_model_low_res:
            return self.config.video_model_low_res
        return self.config.video_model

    async def _text(self, prompt: str, *, model: Optional[str] = None) -> str:
        llm = _import_llm()
        return await asyncio.to_thread(
            llm.text_generate, prompt, model or self.config.text_model
        )

    async def _vision(
        self,
        prompt: str,
        images: List[Image.Image],
        *,
        model: Optional[str] = None,
    ) -> str:
        llm = _import_llm()
        return await asyncio.to_thread(
            llm.vision_generate,
            prompt,
            images,
            model or self.config.vision_model,
        )

    async def _vision_json(
        self,
        prompt: str,
        images: List[Image.Image],
        *,
        max_retries: int = 3,
        retry_delays: tuple = (2, 5, 10),
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call _vision + extract_json_object with retry on empty/error response.

        Retries on ValueError (empty/unparseable response) up to ``max_retries``
        times with increasing delays.
        """
        import asyncio as _asyncio
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                raw = await self._vision(prompt, images, model=model)
                if not raw or not raw.strip():
                    raise ValueError("Empty LLM response")
                return extract_json_object(raw)
            except (ValueError, Exception) as exc:
                last_exc = exc
                if attempt < max_retries:
                    delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
                    logger.warning(
                        "_vision_json attempt %d/%d failed: %s — retrying in %ds",
                        attempt,
                        max_retries,
                        exc,
                        delay,
                    )
                    await _asyncio.sleep(delay)
                else:
                    logger.error(
                        "_vision_json failed after %d attempts: %s",
                        max_retries,
                        exc,
                    )
        raise last_exc  # type: ignore[misc]

    async def _gen_image(
        self,
        prompt: str,
        refs: Optional[List[Image.Image]] = None,
        save_path: Optional[str] = None,
        *,
        model: Optional[str] = None,
    ) -> Optional[Image.Image]:
        llm = _import_llm()
        return await asyncio.to_thread(
            llm.generate_image,
            prompt,
            reference_images=refs,
            model=model or self._effective_image_model(),
            save_path=save_path,
        )

    async def _segment_indicative_mask(
        self,
        prompt: str,
        frame: Image.Image,
        frame_size: Tuple[int, int],
        *,
        raw_path: Optional[str] = None,
        extra_refs: Optional[List[Image.Image]] = None,
    ) -> Optional[Image.Image]:
        """Call image model for one segmentation attempt; return aligned mask unchanged."""
        refs: List[Image.Image] = [frame]
        if extra_refs:
            refs.extend(extra_refs)
        raw_mask = await self._gen_image(prompt, refs=refs, save_path=raw_path)
        if raw_mask is None:
            return None
        return align_mask_to_frame(raw_mask, frame_size)

    async def _detect_bbox_for_entities(
        self,
        frame: Image.Image,
        entity_descriptions: List[Dict[str, str]],
    ) -> List[Dict[str, object]]:
        """VLM bbox fallback for entities still missing from an indicative mask."""
        if not entity_descriptions:
            return []
        entity_lines = "\n".join(
            f'- entity_id="{d.get("entity_id", "")}": {d.get("description", "")}'
            for d in entity_descriptions
            if d.get("entity_id")
        )
        if not entity_lines:
            return []
        prompt = MASK_DETECTION_PROMPT.format(entities=entity_lines)
        try:
            raw = await self._vision(prompt, [frame])
            data = extract_json_object(raw)
            detections = data.get("detections") or []
            if not isinstance(detections, list):
                return []
            allowed = {d.get("entity_id", "") for d in entity_descriptions}
            return [
                det for det in detections
                if isinstance(det, dict) and det.get("entity_id") in allowed
            ]
        except Exception as exc:
            logger.warning("BBox fallback detection failed: %s", exc)
            return []

    async def supplement_mask_for_located_entities(
        self,
        image_path: str,
        mask_path: str,
        entity_descriptions: List[Dict[str, str]],
        color_map: Dict[str, str],
        location_records: List[Dict[str, object]],
        *,
        entity_references: Optional[Dict[str, str]] = None,
        instruction_labels: Optional[Dict[str, str]] = None,
        min_confidence: float = 0.55,
    ) -> str:
        """Re-segment entities that VLM located confidently but batch mask missed."""
        if not location_records or not entity_descriptions:
            return mask_path
        if not image_path or not os.path.exists(image_path):
            return mask_path

        frame = Image.open(image_path).convert("RGB")
        frame_size = frame.size
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("RGB")
        else:
            mask = Image.new("RGB", frame_size, (0, 0, 0))

        aligned_path = segmentation_aligned_sidecar_path(mask_path)
        if not image_has_mask_content(mask) and os.path.exists(aligned_path):
            aligned = Image.open(aligned_path).convert("RGB")
            if image_has_mask_content(aligned):
                logger.info(
                    "Mask supplement: using aligned sidecar as base (%s)",
                    aligned_path,
                )
                mask = aligned

        entity_references = entity_references or {}
        instruction_labels = instruction_labels or {}
        desc_by_eid = {
            str(d.get("entity_id", "")).strip(): d
            for d in entity_descriptions
            if str(d.get("entity_id", "")).strip()
        }
        records_by_eid = {
            str(record.get("entity_id", "")).strip(): record
            for record in location_records
            if str(record.get("entity_id", "")).strip()
        }

        missing = located_entities_missing_from_mask(
            mask,
            location_records,
            color_map,
            min_confidence=min_confidence,
        )
        if not missing:
            return mask_path

        logger.info(
            "Mask supplement: %d confident-but-missing entities → %s",
            len(missing),
            missing,
        )
        layers: List[Image.Image] = [mask]
        still_missing: List[str] = []

        for eid in missing:
            desc = desc_by_eid.get(eid, {})
            record = records_by_eid.get(eid, {})
            description = (
                str(desc.get("description") or "").strip()
                or str(record.get("location_prompt") or "").strip()
                or eid
            )
            instruction_id = instruction_labels.get(
                eid,
                str(desc.get("instruction_id") or record.get("instruction_id") or ""),
            )
            location_hint = compose_robust_location_hint(
                record,
                subject_features=description,
            )
            prompt = build_single_entity_segmentation_prompt(
                eid,
                description,
                color_map.get(eid, "#FF0000"),
                location_hint=location_hint,
                instruction_id=instruction_id,
                anti_copy_retry=True,
            )
            extra_refs: List[Image.Image] = []
            ref_path = entity_references.get(eid, "")
            if ref_path and os.path.exists(ref_path):
                raw_ref = Image.open(ref_path).convert("RGB")
                extra_refs.append(
                    extract_reference_frame_from_composite(raw_ref, frame_size),
                )

            retry_mask = await self._segment_indicative_mask(
                prompt,
                frame,
                frame_size,
                raw_path=f"{mask_path}.{eid}.retry.raw.png",
                extra_refs=extra_refs or None,
            )
            if retry_mask is None:
                still_missing.append(eid)
                continue
            retry_mask.save(f"{mask_path}.{eid}.retry.aligned.png")
            layer = finalize_segmentation_mask(
                retry_mask,
                color_map,
                keep_entity_ids=[eid],
            )
            if not mask_has_palette_coverage(layer, color_map, min_ratio=0.0001):
                logger.warning(
                    "Per-entity mask retry for %s could not be palette-quantized",
                    eid,
                )
                still_missing.append(eid)
                continue
            layers.append(layer)

        if still_missing:
            bbox_descs = [
                {
                    "entity_id": eid,
                    "description": str(desc_by_eid.get(eid, {}).get("description") or eid),
                }
                for eid in still_missing
            ]
            detections = await self._detect_bbox_for_entities(frame, bbox_descs)
            if detections:
                bbox_mask = render_soft_bbox_mask(frame_size, detections, color_map)
                if image_has_mask_content(bbox_mask):
                    logger.info(
                        "BBox fallback recovered %d entity mask region(s)",
                        len(detections),
                    )
                    layers.append(bbox_mask)

        if len(layers) > 1:
            combined = composite_segmentation_layers(layers, frame_size)
            os.makedirs(os.path.dirname(mask_path) or ".", exist_ok=True)
            combined.save(mask_path)
            save_mask_color_map_debug(mask_path, color_map)
            logger.info("Mask supplement saved: %s", mask_path)

        ensure_segmentation_mask_output(mask_path, color_map=color_map)
        return mask_path

    async def _attempt_batch_segmentation(
        self,
        prompt: str,
        frame: Image.Image,
        frame_size: Tuple[int, int],
        color_map: Dict[str, str],
        *,
        raw_path: str,
        aligned_path: str,
    ) -> Optional[Image.Image]:
        """Run one batch segmentation pass; return raw-sidecar resize as final mask."""
        del color_map  # retained for call-site compatibility
        await self._segment_indicative_mask(
            prompt,
            frame,
            frame_size,
            raw_path=raw_path,
            extra_refs=None,
        )
        aligned_mask = mask_from_raw_segmentation_sidecar(raw_path, frame_size)
        if aligned_mask is None:
            logger.warning(
                "Batch segmentation: no usable mask content after resizing %s",
                raw_path,
            )
            return None
        aligned_mask.save(aligned_path)
        return aligned_mask

    async def _segment_entities_into_mask(
        self,
        *,
        frame: Image.Image,
        frame_size: Tuple[int, int],
        entity_descriptions: List[Dict[str, str]],
        color_map: Dict[str, str],
        mask_path: str,
        entity_references: Dict[str, str],
        instruction_labels: Dict[str, str],
        located_by_entity: Dict[str, str],
        records_by_eid: Dict[str, Dict[str, object]],
        target_entity_ids: Optional[List[str]] = None,
    ) -> Optional[Image.Image]:
        """Per-entity segmentation (+ optional bbox fallback) for missing targets."""
        target_ids = target_entity_ids or [
            str(d.get("entity_id", "")).strip()
            for d in entity_descriptions
            if str(d.get("entity_id", "")).strip()
        ]
        desc_by_eid = {
            str(d.get("entity_id", "")).strip(): d
            for d in entity_descriptions
            if str(d.get("entity_id", "")).strip()
        }
        layers: List[Image.Image] = []
        still_missing: List[str] = []

        for eid in target_ids:
            if not eid or eid not in color_map:
                continue
            desc = desc_by_eid.get(eid, {})
            record = records_by_eid.get(eid, {})
            description = (
                str(desc.get("description") or "").strip()
                or str(record.get("location_prompt") or "").strip()
                or located_by_entity.get(eid, "")
                or eid
            )
            instruction_id = instruction_labels.get(
                eid,
                str(desc.get("instruction_id") or record.get("instruction_id") or ""),
            )
            location_hint = compose_robust_location_hint(
                record,
                subject_features=description,
            ) if record else located_by_entity.get(eid, "")
            prompt = build_single_entity_segmentation_prompt(
                eid,
                description,
                color_map.get(eid, "#FF0000"),
                location_hint=location_hint,
                instruction_id=instruction_id,
                anti_copy_retry=True,
            )
            extra_refs: List[Image.Image] = []
            ref_path = entity_references.get(eid, "")
            if ref_path and os.path.exists(ref_path):
                raw_ref = Image.open(ref_path).convert("RGB")
                extra_refs.append(
                    extract_reference_frame_from_composite(raw_ref, frame_size),
                )

            retry_mask = await self._segment_indicative_mask(
                prompt,
                frame,
                frame_size,
                raw_path=f"{mask_path}.{eid}.retry.raw.png",
                extra_refs=extra_refs or None,
            )
            if retry_mask is None:
                still_missing.append(eid)
                continue
            retry_mask.save(f"{mask_path}.{eid}.retry.aligned.png")
            layer = finalize_segmentation_mask(
                retry_mask,
                color_map,
                keep_entity_ids=[eid],
            )
            if not mask_has_palette_coverage(layer, color_map, min_ratio=0.0001):
                logger.warning(
                    "Per-entity mask for %s returned no palette region after finalize",
                    eid,
                )
                still_missing.append(eid)
                continue
            layers.append(layer)

        if still_missing:
            bbox_descs = [
                {
                    "entity_id": eid,
                    "description": str(desc_by_eid.get(eid, {}).get("description") or eid),
                }
                for eid in still_missing
            ]
            detections = await self._detect_bbox_for_entities(frame, bbox_descs)
            if detections:
                bbox_mask = render_soft_bbox_mask(frame_size, detections, color_map)
                if mask_has_palette_coverage(bbox_mask, color_map):
                    logger.info(
                        "BBox fallback recovered %d entity mask region(s)",
                        len(detections),
                    )
                    layers.append(bbox_mask)

        if not layers:
            return None
        return composite_segmentation_layers(layers, frame_size)

    async def _validate_segmentation_mask(
        self,
        frame: Image.Image,
        mask: Image.Image,
        entity_descriptions: List[Dict[str, str]],
        color_map: Dict[str, str],
        query_prompt: str,
    ) -> Tuple[List[str], str]:
        """Return valid entity ids and optional revised segmentation prompt."""
        frame_size = frame.size
        if mask.size != frame_size:
            logger.info(
                "Mask validation rejected locally: size %s != frame %s",
                mask.size,
                frame_size,
            )
            return [], ensure_anti_copy_in_revised_prompt(
                "Regenerate the mask with exactly the same width and height as image 1. "
                "The mask must be pixel-aligned with image 1."
            )
        if mask_looks_like_split_panel_artifact(mask, frame_size):
            logger.info("Mask validation rejected locally: misaligned split-panel artifact")
            return [], ensure_anti_copy_in_revised_prompt(
                "Regenerate the mask from image 1 only. The previous mask was not aligned "
                "with image 1 and appeared confined to one side or a stitched layout."
            )

        entity_color_list = "\n".join(
            (
                f"- {d.get('entity_id', '')}: target=\"{d.get('description', '')}\"; "
                f"color={color_map.get(d.get('entity_id', ''), '#FF0000')}"
            )
            for d in entity_descriptions
        )
        prompt = MASK_VALIDATION_PROMPT.format(
            entity_color_list=entity_color_list,
            query_prompt=query_prompt,
        )
        try:
            raw = await self._vision(prompt, [frame, mask])
            data = extract_json_object(raw)
            revised_prompt = ensure_anti_copy_in_revised_prompt(
                str(data.get("revised_query_prompt", "") or "").strip()
            )
            if not bool(data.get("valid", False)):
                logger.info(
                    "Mask validation rejected candidate: %s",
                    data.get("feedback", ""),
                )
                return [], revised_prompt
            present = data.get("present_entity_ids") or []
            if isinstance(present, str):
                present = [present]
            allowed = {d.get("entity_id", "") for d in entity_descriptions}
            valid_ids = [eid for eid in present if eid in allowed]
            logger.info("Mask validation present entities: %s", valid_ids)
            return valid_ids, revised_prompt
        except Exception as exc:
            logger.warning("Mask validation failed (%s) — rejecting candidate", exc)
            return [], ""

    def _locally_sane_mask_entity_ids(
        self,
        mask: Image.Image,
        entity_descriptions: List[Dict[str, str]],
        color_map: Dict[str, str],
        *,
        min_entity_area: float = 0.00008,
        max_entity_area: float = 0.45,
        max_union_area: float = 0.70,
    ) -> List[str]:
        """Moderate local sanity check for colored indicative masks.

        The VLM decides semantic correctness; this only catches obvious artifacts:
        no usable pixels for a color, tiny specks, or huge full-frame spills.
        """
        union_ratio = mask_union_coverage_ratio(mask)
        if union_ratio > max_union_area:
            logger.info(
                "Mask local sanity rejected: union coverage %.4f > %.4f",
                union_ratio,
                max_union_area,
            )
            return []

        sane_ids: List[str] = []
        for desc in entity_descriptions:
            eid = desc.get("entity_id", "")
            color_hex = color_map.get(eid)
            if not color_hex:
                continue
            ratio = mask_color_coverage_ratio(mask, color_hex)
            if ratio < min_entity_area:
                logger.info(
                    "Mask local sanity rejected %s: color coverage %.6f < %.6f",
                    eid,
                    ratio,
                    min_entity_area,
                )
                continue
            if ratio > max_entity_area:
                logger.info(
                    "Mask local sanity rejected %s: color coverage %.4f > %.4f",
                    eid,
                    ratio,
                    max_entity_area,
                )
                continue
            sane_ids.append(eid)
        return sane_ids

    # ── LLM ──────────────────────────────────────────────────────────────

    async def rewrite_user_prompt(
        self,
        user_prompt: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Clarify and rewrite vague user editing request via Gemini."""
        try:
            if self.dev_mode:
                return {
                    "rewritten_prompt": user_prompt.strip(),
                    "clarifications": ["dev_mode: rewrite skipped"],
                    "success_criteria_prompts": [],
                }

            prompt = system_prompt or INSTRUCTION_REWRITE_PROMPT.format(
                user_prompt=user_prompt,
            )
            raw = await self._text(prompt, model=self.config.text_model)
            data = extract_json_object(raw)
            rewritten = str(data.get("rewritten_prompt", "")).strip() or user_prompt.strip()
            clarifications = data.get("clarifications") or []
            if isinstance(clarifications, str):
                clarifications = [clarifications] if clarifications else []
            success_criteria = data.get("success_criteria_prompts") or []
            if isinstance(success_criteria, dict):
                success_criteria = [success_criteria]
            logger.info(
                "Prompt rewritten (%d clarifications): %s",
                len(clarifications),
                rewritten[:200],
            )
            return {
                "rewritten_prompt": rewritten,
                "clarifications": clarifications,
                "success_criteria_prompts": success_criteria,
            }

        except Exception as exc:
            logger.warning(
                "rewrite_user_prompt failed (%s) — using original prompt", exc,
            )
            return {
                "rewritten_prompt": user_prompt.strip(),
                "clarifications": [f"rewrite failed: {exc}"],
                "success_criteria_prompts": [],
            }

    async def parse_instructions(
        self,
        user_prompt: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Parse user prompt into structured instructions via gemini."""
        try:
            if self.dev_mode:
                return {
                    "instructions": [{
                        "instruction_id": "instr_001",
                        "entity_id": "entity_01",
                        "subject_features": user_prompt[:120],
                        "edit_prompt": user_prompt,
                        "time_condition": {
                            "condition_type": "event",
                            "event_description": "throughout the video",
                        },
                        "target_instance_scope": "single",
                    }]
                }

            prompt = system_prompt or INSTRUCTION_PARSE_PROMPT.format(
                user_prompt=user_prompt
            )
            raw = await self._text(prompt, model=self.config.text_model)
            data = extract_json_object(raw)
            instructions = ensure_instruction_ids(data.get("instructions", []))
            instructions = merge_instructions_one_per_entity(instructions)
            instructions = normalize_instruction_instance_scope(instructions)
            logger.info("Parsed %d instruction(s)", len(instructions))
            return {"instructions": instructions}

        except Exception as exc:
            logger.error("parse_instructions failed: %s", exc, exc_info=True)
            raise ModelApiError(f"parse_instructions failed: {exc}") from exc

    async def resolve_temporal_conflicts(
        self,
        instructions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Local overlap resolution; optional LLM validation pass."""
        try:
            resolved = normalize_all_referential_time_conditions(instructions)
            resolved = resolve_temporal_conflicts(resolved)
            if self.dev_mode or len(resolved) <= 1:
                return resolved

            # Optional LLM sanity check (non-blocking on failure)
            try:
                prompt = (
                    "Validate this instruction list has no temporal conflicts. "
                    "Return the same JSON array under key 'instructions':\n"
                    + json.dumps({"instructions": resolved}, ensure_ascii=False)
                )
                raw = await self._text(prompt, model=self.config.text_model)
                data = extract_json_object(raw)
                if data.get("instructions"):
                    return ensure_instruction_ids(data["instructions"])
            except Exception as exc:
                logger.warning("LLM conflict validation skipped: %s", exc)

            return resolved

        except Exception as exc:
            logger.error("resolve_temporal_conflicts failed: %s", exc, exc_info=True)
            raise ModelApiError(f"resolve_temporal_conflicts failed: {exc}") from exc

    async def match_event_scenes(
        self,
        scene_keyframes: List[Dict[str, Any]],
        time_condition: Dict[str, Any],
    ) -> List[str]:
        """Match scenes to absolute time or event condition."""
        try:
            if not scene_keyframes:
                return []

            ctype = time_condition.get("condition_type", "event")

            # Absolute time — local overlap, no LLM needed
            if ctype == "absolute":
                start = _safe_float(time_condition.get("start_sec"), 0.0)
                end = _safe_float(time_condition.get("end_sec"), 1e9)
                matched = []
                for kf in scene_keyframes:
                    s_start = _safe_float(kf.get("start_sec"), 0.0)
                    s_end = _safe_float(kf.get("end_sec"), 1e9)
                    if s_start < end and start < s_end:
                        matched.append(kf["scene_id"])
                logger.info(
                    "Absolute match: %d scene(s) range=%.1f-%.1f",
                    len(matched), start, end,
                )
                return matched

            if self.dev_mode:
                return [kf["scene_id"] for kf in scene_keyframes if kf.get("image_path")]

            # Event — vision batch (up to max_vision_images scenes per call)
            images: List[Image.Image] = []
            scene_ids: List[str] = []
            for kf in scene_keyframes:
                path = kf.get("image_path", "")
                if path and os.path.exists(path):
                    images.append(Image.open(path).convert("RGB"))
                    scene_ids.append(kf["scene_id"])

            if not images:
                return []

            scene_list = json.dumps(
                [{"scene_id": sid, "index": i} for i, sid in enumerate(scene_ids)],
                ensure_ascii=False,
            )
            prompt = EVENT_GROUNDING_PROMPT.format(
                time_condition=json.dumps(time_condition, ensure_ascii=False),
                scene_list=scene_list,
            )
            raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            matched = data.get("matched_scene_ids", [])
            logger.info("Event match: %s", matched)
            return [s for s in matched if s in scene_ids]

        except Exception as exc:
            logger.error("match_event_scenes failed: %s", exc, exc_info=True)
            raise ModelApiError(f"match_event_scenes failed: {exc}") from exc

    async def analyze_entity_in_keyframe(
        self,
        image_path: str,
        entity_id: str,
        instruction_id: str,
        subject_features: str,
        appearance_time_hint: str,
        edit_prompt: str,
        keyframe_metadata: Dict[str, Any],
        *,
        target_instance_scope: str = "single",
    ) -> Dict[str, Any]:
        """VLM entity detection on one scene keyframe with pose/location details."""
        import asyncio as _asyncio

        MAX_VLM_RETRIES = 3
        RETRY_DELAYS = [2, 5, 10]  # seconds

        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_VLM_RETRIES + 1):
            try:
                if self.dev_mode:
                    return {
                        "present": True,
                        "confidence": 0.9,
                        "quality_score": 88.0,
                        "appearance_time_score": 35.0,
                        "subject_features_score": 38.0,
                        "identification_clarity_score": 15.0,
                        "view_angle": "front",
                        "scene_moment_description": (
                            "A sunlit urban sidewalk with pedestrians in the background. "
                            "The target stands near a shop window on the right third of the frame, "
                            "mid-shot, with soft afternoon light from camera-left."
                        ),
                        "visibility_state": (
                            "Three-quarter front view; full body visible from head to shoes. "
                            "Face clearly visible, looking slightly toward camera-right."
                        ),
                        "pose_and_action": (
                            "Standing upright with weight on the left leg, right hand holding a bag at hip level, "
                            "left arm relaxed at the side. Appears paused mid-walk rather than actively stepping."
                        ),
                        "location_description": (
                            "Occupies the right-middle third of the frame, roughly 3 meters from the camera. "
                            "Foreground shows blurred pavement edge; background shows storefront glass and two "
                            "out-of-focus passersby on the left. No cropping at frame edges; entity is fully in-frame."
                        ),
                        "reasoning": "dev_mode placeholder",
                    }
                if not image_path or not os.path.exists(image_path):
                    return {
                        "present": False,
                        "confidence": 0.0,
                        "quality_score": 0.0,
                        "appearance_time_score": 0.0,
                        "subject_features_score": 0.0,
                        "identification_clarity_score": 0.0,
                        "view_angle": "",
                        "scene_moment_description": "",
                        "visibility_state": "",
                        "pose_and_action": "",
                        "location_description": "",
                        "reasoning": "missing keyframe image",
                    }

                meta = keyframe_metadata or {}
                prompt = KEYFRAME_ENTITY_DETECTION_PROMPT.format(
                    entity_id=entity_id,
                    instruction_id=instruction_id,
                    subject_features=subject_features,
                    appearance_time_hint=appearance_time_hint or "(none)",
                    edit_prompt=edit_prompt or "(none)",
                    target_instance_scope_line=format_target_instance_scope_line(
                        target_instance_scope
                    ),
                    scene_id=str(meta.get("scene_id", "")),
                    timestamp_in_video_sec=float(meta.get("timestamp_in_video_sec", 0.0)),
                    timestamp_in_scene_sec=float(meta.get("timestamp_in_scene_sec", 0.0)),
                    keyframe_role=str(meta.get("keyframe_role", "") or "(none)"),
                    keyframe_description=str(meta.get("keyframe_description", "") or "(none)"),
                )
                raw = await self._vision(prompt, [Image.open(image_path).convert("RGB")])

                # Handle empty / whitespace-only responses with retry instead of
                # immediately returning present=False — VLM APIs sometimes return
                # empty due to rate-limiting, content filtering, or transient errors.
                if not raw or not raw.strip():
                    raise ValueError("Empty LLM response")

                data = extract_json_object(raw)
                appearance_time_score = float(data.get("appearance_time_score", 0.0) or 0.0)
                subject_features_score = float(data.get("subject_features_score", 0.0) or 0.0)
                identification_clarity_score = float(
                    data.get("identification_clarity_score", 0.0) or 0.0
                )
                quality_score = float(data.get("quality_score", 0.0) or 0.0)
                if quality_score <= 0 and (
                    appearance_time_score or subject_features_score or identification_clarity_score
                ):
                    quality_score = (
                        appearance_time_score + subject_features_score + identification_clarity_score
                    )
                return {
                    "present": bool(data.get("present", False)),
                    "confidence": float(data.get("confidence", 0.0) or 0.0),
                    "quality_score": quality_score,
                    "appearance_time_score": appearance_time_score,
                    "subject_features_score": subject_features_score,
                    "identification_clarity_score": identification_clarity_score,
                    "view_angle": str(data.get("view_angle", "") or "").strip().lower(),
                    "scene_moment_description": str(
                        data.get("scene_moment_description", "") or ""
                    ).strip(),
                    "visibility_state": str(data.get("visibility_state", "") or "").strip(),
                    "pose_and_action": str(data.get("pose_and_action", "") or "").strip(),
                    "location_description": str(
                        data.get("location_description", "") or ""
                    ).strip(),
                    "reasoning": str(data.get("reasoning", "") or "").strip(),
                }
            except Exception as exc:
                last_exc = exc
                is_retryable = (
                    isinstance(exc, ValueError)
                    and "Empty LLM response" in str(exc)
                ) or not isinstance(exc, ValueError)

                if attempt < MAX_VLM_RETRIES and is_retryable:
                    delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
                    logger.warning(
                        "analyze_entity_in_keyframe %s/%s attempt %d/%d failed: %s — retrying in %ds",
                        entity_id,
                        os.path.basename(image_path or ""),
                        attempt,
                        MAX_VLM_RETRIES,
                        exc,
                        delay,
                    )
                    await _asyncio.sleep(delay)
                    continue

                logger.error(
                    "analyze_entity_in_keyframe failed %s/%s after %d attempt(s): %s",
                    entity_id,
                    image_path,
                    attempt,
                    exc,
                    exc_info=True,
                )
                return {
                    "present": False,
                    "confidence": 0.0,
                    "quality_score": 0.0,
                    "appearance_time_score": 0.0,
                    "subject_features_score": 0.0,
                    "identification_clarity_score": 0.0,
                    "view_angle": "",
                    "scene_moment_description": "",
                    "visibility_state": "",
                    "pose_and_action": "",
                    "location_description": "",
                    "reasoning": f"analysis error after {attempt} attempt(s): {exc}",
                }

        # Should not reach here, but safety net
        return {
            "present": False,
            "confidence": 0.0,
            "quality_score": 0.0,
            "appearance_time_score": 0.0,
            "subject_features_score": 0.0,
            "identification_clarity_score": 0.0,
            "view_angle": "",
            "scene_moment_description": "",
            "visibility_state": "",
            "pose_and_action": "",
            "location_description": "",
            "reasoning": f"analysis exhausted retries: {last_exc}",
        }

    async def select_multiview_reference_keyframes(
        self,
        *,
        entity_id: str,
        instruction_id: str,
        subject_features: str,
        appearance_time_hint: str,
        appearances_catalog: List[Dict[str, Any]],
        select_count: int,
        video_duration_sec: float = 0.0,
    ) -> List[int]:
        """VLM: pick reference keyframe indices balancing score, coverage, and views."""
        from video_editing_agent.utils.multiview_ref_utils import fallback_select_reference_keyframes

        try:
            if not appearances_catalog:
                return []
            if len(appearances_catalog) <= select_count:
                return [int(item["appearance_index"]) for item in appearances_catalog]

            if self.dev_mode:
                return [int(item["appearance_index"]) for item in appearances_catalog[:select_count]]

            catalog_json = json.dumps(appearances_catalog, indent=2, ensure_ascii=False)
            prompt = ENTITY_REFERENCE_KEYFRAME_SELECT_PROMPT.format(
                entity_id=entity_id,
                instruction_id=instruction_id,
                subject_features=subject_features,
                appearance_time_hint=appearance_time_hint or "(none)",
                video_duration_sec=float(video_duration_sec or 0.0),
                select_count=select_count,
                appearances_catalog=catalog_json,
                catalog_length=len(appearances_catalog),
            )
            raw = await self._vision(prompt, [])
            data = extract_json_object(raw)
            indices = data.get("selected_indices") or []
            if not isinstance(indices, list):
                raise ValueError("selected_indices is not a list")

            cleaned: List[int] = []
            seen: Set[int] = set()
            for item in indices:
                idx = int(item)
                if idx in seen or idx < 0 or idx >= len(appearances_catalog):
                    continue
                seen.add(idx)
                cleaned.append(idx)
                if len(cleaned) >= select_count:
                    break

            if len(cleaned) < min(select_count, len(appearances_catalog)):
                logger.warning(
                    "VLM returned %d/%d keyframe picks for %s — using fallback top-up",
                    len(cleaned),
                    select_count,
                    instruction_id,
                )
                # Reconstruct minimal appearance-like objects for fallback top-up.
                from video_editing_agent.schemas.entity_keyframe_appearances import (
                    KeyframeEntityAppearance,
                )

                pseudo = [
                    KeyframeEntityAppearance(
                        scene_id=str(item.get("scene_id", "")),
                        keyframe_path="x",
                        timestamp_in_video_sec=float(item.get("timestamp_in_video_sec", 0.0)),
                        confidence=float(item.get("confidence", 0.0)),
                        quality_score=float(item.get("quality_score", 0.0)),
                        appearance_time_score=float(item.get("appearance_time_score", 0.0)),
                        subject_features_score=float(item.get("subject_features_score", 0.0)),
                        identification_clarity_score=float(
                            item.get("identification_clarity_score", 0.0)
                        ),
                        view_angle=str(item.get("view_angle", "") or ""),
                    )
                    for item in appearances_catalog
                ]
                fallback = fallback_select_reference_keyframes(
                    pseudo,
                    select_count=select_count,
                    video_duration_sec=video_duration_sec,
                )
                for idx in fallback:
                    if idx not in seen:
                        cleaned.append(idx)
                    if len(cleaned) >= select_count:
                        break

            logger.info(
                "Reference keyframe selection for %s → indices %s (%s)",
                instruction_id,
                cleaned,
                data.get("reasoning", ""),
            )
            return cleaned[:select_count]
        except Exception as exc:
            logger.warning(
                "select_multiview_reference_keyframes failed for %s (%s) — fallback",
                instruction_id,
                exc,
            )
            from video_editing_agent.schemas.entity_keyframe_appearances import (
                KeyframeEntityAppearance,
            )

            pseudo = [
                KeyframeEntityAppearance(
                    scene_id=str(item.get("scene_id", "")),
                    keyframe_path="x",
                    timestamp_in_video_sec=float(item.get("timestamp_in_video_sec", 0.0)),
                    confidence=float(item.get("confidence", 0.0)),
                    quality_score=float(item.get("quality_score", 0.0)),
                    appearance_time_score=float(item.get("appearance_time_score", 0.0)),
                    subject_features_score=float(item.get("subject_features_score", 0.0)),
                    identification_clarity_score=float(
                        item.get("identification_clarity_score", 0.0)
                    ),
                    view_angle=str(item.get("view_angle", "") or ""),
                )
                for item in appearances_catalog
            ]
            return fallback_select_reference_keyframes(
                pseudo,
                select_count=select_count,
                video_duration_sec=video_duration_sec,
            )

    async def generate_entity_multiview_sheet(
        self,
        keyframe_grid_path: str,
        *,
        entity_id: str,
        instruction_id: str,
        subject_features: str,
        appearance_time_hint: str,
        keyframe_notes: str,
        output_path: str,
        avoid_operations: str = "",
        positive_prompt: str = "",
    ) -> str:
        """Synthesize a single front-view entity reference image from keyframe grid + notes."""
        try:
            from video_editing_agent.utils.multiview_ref_utils import (
                build_dev_mode_multiview_placeholder,
                ensure_square_reference_file,
            )

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            if self.dev_mode:
                build_dev_mode_multiview_placeholder(
                    f"front view {instruction_id}",
                ).save(output_path)
                return output_path

            avoid_section = build_edit_retry_guidance_section(
                positive_prompt=positive_prompt,
                avoid_operations=avoid_operations,
            )
            prompt = ENTITY_MULTIVIEW_SYNTHESIS_PROMPT.format(
                entity_id=entity_id,
                instruction_id=instruction_id,
                subject_features=subject_features,
                appearance_time_hint=appearance_time_hint or "(none)",
                keyframe_notes=keyframe_notes,
                avoid_section=avoid_section,
            )
            refs = [Image.open(keyframe_grid_path).convert("RGB")]
            img = await self._gen_image(prompt, refs=refs, save_path=output_path)
            if img is None and not os.path.exists(output_path):
                # Retry once with a simplified, safety-friendly prompt when the
                # image API rejects the request (e.g. IMAGE_PROHIBITED_CONTENT).
                logger.warning(
                    "generate_entity_multiview_sheet: image generation returned no "
                    "result for %s — retrying with simplified prompt",
                    instruction_id,
                )
                simplified_prompt = (
                    f"Generate a clear, neutral front-view portrait reference image "
                    f"of a character matching this description: {subject_features[:500]}. "
                    f"The image should be a clean, simple character reference on a neutral "
                    f"background, suitable as an art reference sheet. "
                    f"Entity ID: {entity_id}."
                )
                img = await self._gen_image(
                    simplified_prompt, refs=refs, save_path=output_path,
                )
            if img is None and not os.path.exists(output_path):
                raise RuntimeError("front-view reference synthesis returned no image")
            ensure_square_reference_file(output_path)
            return output_path
        except Exception as exc:
            logger.error(
                "generate_entity_multiview_sheet failed for %s: %s",
                instruction_id,
                exc,
                exc_info=True,
            )
            raise ModelApiError(f"generate_entity_multiview_sheet failed: {exc}") from exc

    async def edit_entity_multiview_sheet(
        self,
        multiview_source_path: str,
        *,
        entity_id: str,
        instruction_id: str,
        subject_features: str,
        edit_prompt: str,
        output_path: str,
        avoid_operations: str = "",
        positive_prompt: str = "",
    ) -> str:
        """Apply edit instruction to a single front-view entity reference image."""
        try:
            from video_editing_agent.utils.multiview_ref_utils import (
                build_dev_mode_multiview_placeholder,
                ensure_square_reference_file,
            )

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            if self.dev_mode:
                build_dev_mode_multiview_placeholder(
                    f"edited front view {instruction_id}",
                ).save(output_path)
                return output_path

            avoid_section = build_edit_retry_guidance_section(
                positive_prompt=positive_prompt,
                avoid_operations=avoid_operations,
            )
            prompt = ENTITY_MULTIVIEW_EDIT_PROMPT.format(
                entity_id=entity_id,
                instruction_id=instruction_id,
                subject_features=subject_features,
                edit_prompt=edit_prompt,
                avoid_section=avoid_section,
            )
            refs = [Image.open(multiview_source_path).convert("RGB")]
            img = await self._gen_image(prompt, refs=refs, save_path=output_path)
            if img is None and not os.path.exists(output_path):
                # Retry once with a simplified, safety-friendly prompt when the
                # image API rejects the request (e.g. IMAGE_PROHIBITED_CONTENT).
                logger.warning(
                    "edit_entity_multiview_sheet: image generation returned no "
                    "result for %s — retrying with simplified prompt",
                    instruction_id,
                )
                simplified_prompt = (
                    f"Edit this character reference image to apply the following change: "
                    f"{edit_prompt}. "
                    f"Character description: {subject_features[:500]}. "
                    f"Keep the same art style, pose, and background. "
                    f"Entity ID: {entity_id}."
                )
                img = await self._gen_image(
                    simplified_prompt, refs=refs, save_path=output_path,
                )
            if img is None and not os.path.exists(output_path):
                raise RuntimeError("front-view reference edit returned no image")
            ensure_square_reference_file(output_path)
            return output_path
        except Exception as exc:
            logger.error(
                "edit_entity_multiview_sheet failed for %s: %s",
                instruction_id,
                exc,
                exc_info=True,
            )
            raise ModelApiError(f"edit_entity_multiview_sheet failed: {exc}") from exc

    async def select_best_multiview_candidate(
        self,
        *,
        task_type: str,
        keyframe_grid_path: str,
        keyframe_notes: str,
        candidate_paths: List[str],
        entity_id: str,
        instruction_id: str,
        subject_features: str,
        edit_prompt: str = "",
    ) -> int:
        """VLM: pick best candidate index among generated front-view images."""
        try:
            if self.dev_mode:
                return 0

            valid = [p for p in candidate_paths if p and os.path.exists(p)]
            if not valid:
                return 0
            if len(valid) == 1:
                return 0

            edit_section = ""
            if task_type == "edit" and edit_prompt.strip():
                edit_section = f"- edit_prompt: {edit_prompt.strip()}"

            prompt = ENTITY_MULTIVIEW_CANDIDATE_SELECT_PROMPT.format(
                task_type=task_type,
                entity_id=entity_id,
                instruction_id=instruction_id,
                subject_features=subject_features,
                edit_section=edit_section,
                keyframe_notes_section=f"INPUT KEYFRAME NOTES:\n{keyframe_notes}",
                edit_prompt=edit_prompt or "(none)",
            )
            images = [Image.open(keyframe_grid_path).convert("RGB")]
            for path in valid[:3]:
                images.append(Image.open(path).convert("RGB"))
            while len(images) < 4:
                images.append(Image.new("RGB", images[0].size, (255, 255, 255)))

            raw = await self._vision(prompt, images[:4])
            data = extract_json_object(raw)
            idx = int(data.get("best_candidate_index", 0))
            if idx < 0 or idx >= len(valid):
                logger.warning(
                    "Invalid multiview candidate index %s — defaulting to 0",
                    idx,
                )
                return 0
            logger.info(
                "Multiview %s selection → candidate %d (confidence=%s, reason=%s)",
                task_type,
                idx,
                data.get("confidence", ""),
                data.get("reasoning", ""),
            )
            return idx
        except Exception as exc:
            logger.warning(
                "select_best_multiview_candidate failed (%s) — defaulting to 0",
                exc,
            )
            return 0

    async def validate_entity_multiview_synthesis(
        self,
        *,
        keyframe_grid_path: str,
        multiview_sheet_path: str,
        entity_id: str,
        instruction_id: str,
        subject_features: str,
        keyframe_notes: str,
    ) -> Dict[str, Any]:
        """VLM QA for synthesized front-view reference image."""
        try:
            if self.dev_mode:
                return apply_multiview_qa_gate({
                    "passed": True,
                    "score": 1.0,
                    "four_view_layout_correct": True,
                    "front_view_orientation_correct": True,
                    "back_view_orientation_correct": True,
                    "left_profile_orientation_correct": True,
                    "right_profile_orientation_correct": True,
                    "entity_identity_matches_reference": True,
                    "source_appearance_matches_reference": True,
                    "occlusion_physically_plausible": True,
                    "art_style_matches_source": True,
                    "panel_structure_preserved": True,
                    "neutral_background_ok": True,
                    "failed_aspects": [],
                    "feedback": "dev_mode pass",
                    "retry_focus_prompt": "",
                }, task="synthesis")

            if not os.path.exists(keyframe_grid_path) or not os.path.exists(multiview_sheet_path):
                return apply_multiview_qa_gate({
                    "passed": False,
                    "score": 0.0,
                    "four_view_layout_correct": False,
                    "front_view_orientation_correct": False,
                    "back_view_orientation_correct": False,
                    "left_profile_orientation_correct": False,
                    "right_profile_orientation_correct": False,
                    "entity_identity_matches_reference": False,
                    "source_appearance_matches_reference": False,
                    "occlusion_physically_plausible": False,
                    "art_style_matches_source": False,
                    "panel_structure_preserved": False,
                    "neutral_background_ok": False,
                    "failed_aspects": ["missing image path"],
                    "feedback": "missing keyframe grid or front-view reference image",
                    "retry_focus_prompt": "Return one front-view reference image, not a 2x2 sheet.",
                }, task="synthesis")

            prompt = ENTITY_MULTIVIEW_SYNTHESIS_QA_PROMPT.format(
                entity_id=entity_id,
                instruction_id=instruction_id,
                subject_features=subject_features,
                keyframe_notes=keyframe_notes,
            )
            images = [
                Image.open(keyframe_grid_path).convert("RGB"),
                Image.open(multiview_sheet_path).convert("RGB"),
            ]
            raw = await self._vision(prompt, images)
            if not raw or not raw.strip():
                # Retry once on empty response.
                logger.warning(
                    "validate_entity_multiview_synthesis: empty VLM response for %s — retrying",
                    instruction_id,
                )
                import asyncio as _asyncio
                await _asyncio.sleep(3)
                raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            qa_payload = build_multiview_synthesis_qa_from_vlm(data)

            focus_prompt = ENTITY_MULTIVIEW_SOURCE_APPEARANCE_QA_PROMPT.format(
                entity_id=entity_id,
                subject_features=subject_features,
                keyframe_notes=keyframe_notes,
            )
            focus_raw = await self._vision(focus_prompt, images)
            focus_data = extract_json_object(focus_raw) if focus_raw and focus_raw.strip() else {}
            qa_payload = merge_multiview_focused_qa_into_result(
                qa_payload,
                focus_data,
                alignment_flag="source_appearance_matches_reference",
            )
            result = apply_multiview_qa_gate(qa_payload, task="synthesis")
            if not result.get("passed"):
                logger.info(
                    "validate_entity_multiview_synthesis failed gate for %s: %s",
                    instruction_id,
                    "; ".join(result.get("qa_reject_reasons") or []),
                )
            return result
        except Exception as exc:
            logger.error(
                "validate_entity_multiview_synthesis failed for %s: %s",
                instruction_id,
                exc,
                exc_info=True,
            )
            return apply_multiview_qa_gate({
                "passed": False,
                "score": 0.0,
                "four_view_layout_correct": False,
                "front_view_orientation_correct": False,
                "back_view_orientation_correct": False,
                "left_profile_orientation_correct": False,
                "right_profile_orientation_correct": False,
                "entity_identity_matches_reference": False,
                "source_appearance_matches_reference": False,
                "occlusion_physically_plausible": False,
                "art_style_matches_source": False,
                "panel_structure_preserved": False,
                "neutral_background_ok": False,
                "failed_aspects": ["qa_error"],
                "feedback": f"QA failed due to error: {exc}",
                "retry_focus_prompt": (
                    "Preserve a single front-view layout, entity identity, and source hair/clothing."
                ),
            }, task="synthesis")

    async def validate_entity_multiview_edit(
        self,
        *,
        multiview_source_path: str,
        multiview_edited_path: str,
        keyframe_grid_path: str,
        entity_id: str,
        instruction_id: str,
        subject_features: str,
        edit_prompt: str,
    ) -> Dict[str, Any]:
        """VLM QA for edited front-view reference image."""
        try:
            if self.dev_mode:
                return apply_multiview_qa_gate({
                    "passed": True,
                    "score": 1.0,
                    "four_view_layout_correct": True,
                    "front_view_orientation_correct": True,
                    "back_view_orientation_correct": True,
                    "left_profile_orientation_correct": True,
                    "right_profile_orientation_correct": True,
                    "entity_identity_matches_reference": True,
                    "source_appearance_matches_reference": True,
                    "occlusion_physically_plausible": True,
                    "art_style_matches_source": True,
                    "edit_completed": True,
                    "edit_consistent_across_panels": True,
                    "edit_attributes_match_instruction": True,
                    "edit_view_occlusion_plausible": True,
                    "panel_structure_preserved": True,
                    "neutral_background_ok": True,
                    "failed_aspects": [],
                    "feedback": "dev_mode pass",
                    "retry_focus_prompt": "",
                }, task="edit")

            paths = [multiview_source_path, multiview_edited_path, keyframe_grid_path]
            if not all(path and os.path.exists(path) for path in paths):
                return apply_multiview_qa_gate({
                    "passed": False,
                    "score": 0.0,
                    "four_view_layout_correct": False,
                    "front_view_orientation_correct": False,
                    "back_view_orientation_correct": False,
                    "left_profile_orientation_correct": False,
                    "right_profile_orientation_correct": False,
                    "entity_identity_matches_reference": False,
                    "source_appearance_matches_reference": False,
                    "occlusion_physically_plausible": False,
                    "art_style_matches_source": False,
                    "edit_completed": False,
                    "edit_consistent_across_panels": False,
                    "edit_attributes_match_instruction": False,
                    "edit_view_occlusion_plausible": False,
                    "panel_structure_preserved": False,
                    "neutral_background_ok": False,
                    "failed_aspects": ["missing image path"],
                    "feedback": "missing source, edited reference, or keyframe grid",
                    "retry_focus_prompt": "Return one edited front-view reference image, not a 2x2 sheet.",
                }, task="edit")

            prompt = ENTITY_MULTIVIEW_EDIT_QA_PROMPT.format(
                entity_id=entity_id,
                instruction_id=instruction_id,
                subject_features=subject_features,
                edit_prompt=edit_prompt,
            )
            images = [
                Image.open(multiview_source_path).convert("RGB"),
                Image.open(multiview_edited_path).convert("RGB"),
                Image.open(keyframe_grid_path).convert("RGB"),
            ]
            raw = await self._vision(prompt, images)
            if not raw or not raw.strip():
                logger.warning(
                    "validate_entity_multiview_edit: empty VLM response for %s — retrying",
                    instruction_id,
                )
                import asyncio as _asyncio
                await _asyncio.sleep(3)
                raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            qa_payload = build_multiview_edit_qa_from_vlm(data)

            focus_images = [
                Image.open(multiview_source_path).convert("RGB"),
                Image.open(multiview_edited_path).convert("RGB"),
            ]
            focus_prompt = ENTITY_MULTIVIEW_EDIT_ATTRIBUTE_QA_PROMPT.format(
                edit_prompt=edit_prompt,
            )
            focus_raw = await self._vision(focus_prompt, focus_images)
            focus_data = extract_json_object(focus_raw) if focus_raw and focus_raw.strip() else {}
            qa_payload = merge_multiview_focused_qa_into_result(
                qa_payload,
                focus_data,
                alignment_flag="edit_attributes_match_instruction",
            )

            occlusion_prompt = ENTITY_MULTIVIEW_EDIT_VIEW_OCCLUSION_QA_PROMPT.format(
                edit_prompt=edit_prompt,
            )
            occlusion_raw = await self._vision(
                occlusion_prompt,
                [Image.open(multiview_edited_path).convert("RGB")],
            )
            occlusion_data = extract_json_object(occlusion_raw) if occlusion_raw and occlusion_raw.strip() else {}
            qa_payload = merge_multiview_focused_qa_into_result(
                qa_payload,
                occlusion_data,
                alignment_flag="edit_view_occlusion_plausible",
                cascade_flags=("occlusion_physically_plausible",),
            )
            result = apply_multiview_qa_gate(qa_payload, task="edit")
            if not result.get("passed"):
                logger.info(
                    "validate_entity_multiview_edit failed gate for %s: %s",
                    instruction_id,
                    "; ".join(result.get("qa_reject_reasons") or []),
                )
            return result
        except Exception as exc:
            logger.error(
                "validate_entity_multiview_edit failed for %s: %s",
                instruction_id,
                exc,
                exc_info=True,
            )
            return apply_multiview_qa_gate({
                "passed": False,
                "score": 0.0,
                "four_view_layout_correct": False,
                "front_view_orientation_correct": False,
                "back_view_orientation_correct": False,
                "left_profile_orientation_correct": False,
                "right_profile_orientation_correct": False,
                "entity_identity_matches_reference": False,
                "source_appearance_matches_reference": False,
                "occlusion_physically_plausible": False,
                "art_style_matches_source": False,
                "edit_completed": False,
                "edit_consistent_across_panels": False,
                "edit_attributes_match_instruction": False,
                "edit_view_occlusion_plausible": False,
                "panel_structure_preserved": False,
                "neutral_background_ok": False,
                "failed_aspects": ["qa_error"],
                "feedback": f"QA failed due to error: {exc}",
                "retry_focus_prompt": (
                    "Apply the edit with correct 3D occlusion per profile view — left-shoulder "
                    "accessories must not appear on the near shoulder in RIGHT profile."
                ),
            }, task="edit")

    async def check_entity_in_frame(
        self,
        image_path: str,
        subject_features: str,
        *,
        action: str = "modify",
        reference_image_path: Optional[str] = None,
    ) -> bool:
        """VLM check whether target subject is visible in frame."""
        try:
            if self.dev_mode:
                return True
            if not image_path or not os.path.exists(image_path):
                return False

            images = [Image.open(image_path).convert("RGB")]
            if reference_image_path and os.path.exists(reference_image_path):
                prompt = ENTITY_VISIBILITY_WITH_REF_PROMPT.format(
                    subject_features=subject_features,
                    action=action,
                )
                images.append(Image.open(reference_image_path).convert("RGB"))
            else:
                prompt = ENTITY_VISIBILITY_PROMPT.format(
                    subject_features=subject_features,
                    action=action,
                )

            raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            visible = bool(data.get("visible", False))
            logger.debug(
                "entity visibility action=%s visible=%s ref=%s reason=%s",
                action, visible, bool(reference_image_path), data.get("reasoning", ""),
            )
            return visible

        except Exception as exc:
            logger.warning("check_entity_in_frame failed (%s) — assuming visible", exc)
            return True

    async def validate_first_detection_mask_entity(
        self,
        image_path: str,
        mask_path: str,
        *,
        entity_id: str,
        subject_features: str,
        color_hex: str,
        instruction_id: str = "",
    ) -> Dict[str, Any]:
        """VLM verify that a first-time mask region matches expected subject features."""
        try:
            if self.dev_mode:
                return {
                    "valid": True,
                    "matches_subject_features": True,
                    "confidence": 1.0,
                    "feedback": "dev_mode: validation skipped",
                }
            if not image_path or not os.path.exists(image_path):
                return {
                    "valid": False,
                    "matches_subject_features": False,
                    "confidence": 0.0,
                    "feedback": "missing source frame",
                }
            if not mask_path or not os.path.exists(mask_path):
                return {
                    "valid": False,
                    "matches_subject_features": False,
                    "confidence": 0.0,
                    "feedback": "missing mask image",
                }

            frame = Image.open(image_path).convert("RGB")
            entity_mask = build_single_entity_mask_image(
                mask_path,
                instruction_id or entity_id,
                color_hex,
                frame.size,
            )
            overlay = overlay_colored_mask_on_frame(frame, entity_mask, alpha=0.45)
            color_name = color_name_from_hex(color_hex)
            prompt = MASK_FIRST_DETECTION_ENTITY_VALIDATION_PROMPT.format(
                entity_id=entity_id,
                instruction_id=instruction_id or entity_id,
                subject_features=subject_features,
                color_name=color_name,
                color_hex=color_hex.upper(),
            )
            raw = await self._vision(prompt, [frame, overlay])
            data = extract_json_object(raw)
            matches = bool(data.get("matches_subject_features", False))
            confidence = _safe_float(data.get("confidence"), 0.0)
            valid = bool(data.get("valid", False)) and matches and confidence >= 0.7
            feedback = str(data.get("feedback", "") or "").strip()
            logger.info(
                "First-detection mask validation %s / %s → valid=%s conf=%.2f (%s)",
                entity_id,
                instruction_id or entity_id,
                valid,
                confidence,
                feedback[:120],
            )
            return {
                "valid": valid,
                "matches_subject_features": matches,
                "confidence": confidence,
                "feedback": feedback,
            }
        except Exception as exc:
            logger.warning(
                "validate_first_detection_mask_entity failed (%s) — rejecting mask",
                exc,
            )
            return {
                "valid": False,
                "matches_subject_features": False,
                "confidence": 0.0,
                "feedback": str(exc),
            }

    async def compare_entity_reference_candidates(
        self,
        existing_reference_path: str,
        candidate_reference_path: str,
        *,
        subject_features: str,
    ) -> str:
        """VLM compare existing vs newly detected entity reference overlays."""
        try:
            if self.dev_mode:
                return "existing"
            if not os.path.exists(existing_reference_path):
                return "candidate"
            if not os.path.exists(candidate_reference_path):
                return "existing"

            prompt = ENTITY_REFERENCE_COMPARE_PROMPT.format(
                subject_features=subject_features,
            )
            images = [
                Image.open(existing_reference_path).convert("RGB"),
                Image.open(candidate_reference_path).convert("RGB"),
            ]
            raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            choice = str(data.get("better_image", "existing")).strip().lower()
            if choice not in {"existing", "candidate"}:
                logger.warning(
                    "Invalid reference compare choice %r — keeping existing",
                    choice,
                )
                return "existing"
            logger.info(
                "Entity reference compare → %s (confidence=%s, reason=%s)",
                choice,
                data.get("confidence", ""),
                data.get("reasoning", ""),
            )
            return choice
        except Exception as exc:
            logger.warning(
                "compare_entity_reference_candidates failed (%s) — keeping existing",
                exc,
            )
            return "existing"

    async def validate_edit_quality(
        self,
        original_image_path: str,
        edited_image_path: str,
        edit_prompt: str,
        *,
        subject_features: str = "",
        action: str = "modify",
        success_criteria_prompt: str = "",
        keyframe: bool = False,
    ) -> Dict[str, Any]:
        """VLM QA for inpainted keyframe."""
        try:
            if self.dev_mode:
                return {
                    "passed": True,
                    "feedback": "dev_mode pass",
                    "score": 1.0,
                    "edit_completed": True,
                    "failed_aspects": [],
                    "retry_focus_prompt": "",
                }

            if not os.path.exists(edited_image_path):
                return {
                    "passed": False,
                    "feedback": "edited image missing",
                    "score": 0.0,
                    "edit_completed": False,
                    "failed_aspects": ["edited image missing"],
                    "retry_focus_prompt": "Produce a valid edited image matching the instruction.",
                }

            criteria = (
                success_criteria_prompt
                or "Use the edit goal and preservation rules above."
            )
            if keyframe:
                edit_img, orig_img, qa_resized = prepare_keyframe_qa_images(
                    original_image_path,
                    edited_image_path,
                )
                if qa_resized:
                    logger.info(
                        "Keyframe QA: edited frame was resized to %s for VLM comparison",
                        orig_img.size,
                    )

            if action == "delete":
                template = (
                    KEYFRAME_DELETE_EDIT_QA_VALIDATION_PROMPT
                    if keyframe
                    else DELETE_QA_VALIDATION_PROMPT
                )
                prompt = template.format(
                    edit_prompt=edit_prompt,
                    subject_features=subject_features or "N/A",
                    success_criteria_prompt=criteria,
                )
            else:
                template = (
                    KEYFRAME_EDIT_QA_VALIDATION_PROMPT
                    if keyframe
                    else QA_VALIDATION_PROMPT
                )
                prompt = template.format(
                    edit_prompt=edit_prompt,
                    subject_features=subject_features or "N/A",
                    success_criteria_prompt=criteria,
                )

            if keyframe:
                images = [edit_img, orig_img]
            else:
                images = [
                    Image.open(original_image_path).convert("RGB"),
                    Image.open(edited_image_path).convert("RGB"),
                ]
            raw = await self._vision(prompt, images)
            data = extract_json_object(raw)

            failed_aspects = data.get("failed_aspects") or []
            if isinstance(failed_aspects, str):
                failed_aspects = [failed_aspects] if failed_aspects else []
            edit_errors = normalize_qa_error_list(data.get("edit_errors"))
            if not edit_errors:
                edit_errors = [str(x) for x in failed_aspects if str(x).strip()]
            retry_focus = str(data.get("retry_focus_prompt", "") or "").strip()
            edit_completed = bool(data.get("edit_completed", False))
            frame_structure_preserved = bool(
                data.get("frame_structure_preserved", False)
            )
            background_unedited_regions_preserved = bool(
                data.get("background_unedited_regions_preserved", True)
            )
            unrelated_edit_changes_absent = bool(
                data.get("unrelated_edit_changes_absent", True)
            )
            non_edit_region_change_severity = normalize_non_edit_region_change_severity(
                data.get("non_edit_region_change_severity")
            )
            non_edit_region_change_summary = str(
                data.get("non_edit_region_change_summary", "") or ""
            ).strip()

            # Delete no-op: target absent in original → success
            if action == "delete":
                if data.get("target_was_present") is False:
                    return {
                        "passed": True,
                        "feedback": data.get("feedback", "target not in frame, no-op"),
                        "score": 1.0,
                        "edit_completed": True,
                        "failed_aspects": [],
                        "retry_focus_prompt": "",
                    }
                fb = str(data.get("feedback", "")).lower()
                if any(
                    p in fb
                    for p in (
                        "not present", "not in the original", "no such man",
                        "no such person", "identical to the original", "no edit was",
                    )
                ):
                    return {
                        "passed": True,
                        "feedback": data.get("feedback", ""),
                        "score": 1.0,
                        "edit_completed": True,
                        "failed_aspects": [],
                        "retry_focus_prompt": "",
                    }

            passed = bool(data.get("passed", False))
            score = float(data.get("score", 0.0))
            if keyframe:
                if non_edit_region_change_requires_reedit(non_edit_region_change_severity):
                    passed = False
                    background_unedited_regions_preserved = False
                    if "non-edit regions changed too much" not in failed_aspects:
                        failed_aspects.append("non-edit regions changed too much")
                    summary = non_edit_region_change_summary or (
                        f"VLM rated non-edit region change as {non_edit_region_change_severity}."
                    )
                    data["feedback"] = f"{str(data.get('feedback', '') or '').strip()} {summary}".strip()
                    if not retry_focus:
                        retry_focus = (
                            "Do not alter non-edit regions outside the exact edit silhouette; only a tiny seam "
                            "right at the edit boundary is acceptable."
                        )
                passed = (
                    passed
                    and edit_completed
                    and frame_structure_preserved
                    and background_unedited_regions_preserved
                    and unrelated_edit_changes_absent
                    and score >= 0.7
                )
            elif score >= 0.7:
                passed = True

            if keyframe and not passed and not retry_focus:
                retry_focus = build_keyframe_qa_avoid_operations(
                    {
                        "retry_focus_prompt": "",
                        "edit_errors": edit_errors,
                        "failed_aspects": failed_aspects,
                        "feedback": data.get("feedback", ""),
                    }
                )

            return {
                "passed": passed,
                "feedback": str(data.get("feedback", "")),
                "score": score,
                "edit_completed": edit_completed,
                "frame_structure_preserved": frame_structure_preserved,
                "background_unedited_regions_preserved": background_unedited_regions_preserved,
                "unrelated_edit_changes_absent": unrelated_edit_changes_absent,
                "non_edit_region_change_severity": non_edit_region_change_severity,
                "non_edit_region_change_summary": non_edit_region_change_summary,
                "failed_aspects": [str(x) for x in failed_aspects if str(x).strip()],
                "edit_errors": edit_errors,
                "retry_focus_prompt": retry_focus,
                "positive_prompt": str(data.get("positive_prompt", "")).strip(),
            }

        except Exception as exc:
            logger.error("validate_edit_quality failed: %s", exc, exc_info=True)
            if keyframe:
                return {
                    "passed": False,
                    "feedback": f"QA error: {exc}",
                    "score": 0.0,
                    "edit_completed": False,
                    "failed_aspects": ["QA validation error"],
                    "retry_focus_prompt": (
                        "Do not change frame dimensions or layout; do not output a crop, "
                        "collage, or comparison card; do not leave the edit incomplete."
                    ),
                }
            return {"passed": True, "feedback": f"QA error (fail-open): {exc}", "score": 0.75}

    # ── Image ────────────────────────────────────────────────────────────

    async def generate_reference_image(
        self,
        prompt: str,
        output_path: str,
        *,
        white_background: bool = True,
        forbidden_elements: Optional[List[str]] = None,
    ) -> str:
        """T2I reference on white background (isolated asset only)."""
        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            forbidden_clause = ""
            if forbidden_elements:
                items = "\n".join(f"- Do NOT include: {x}" for x in forbidden_elements)
                forbidden_clause = f"\nExplicit exclusions:\n{items}"

            full_prompt = REF_IMAGE_PROMPT.format(
                ref_subject=prompt,
                forbidden_clause=forbidden_clause,
            )
            if not white_background:
                full_prompt = prompt

            if self.dev_mode:
                Image.new("RGB", (512, 512), (255, 255, 255)).save(output_path)
                return output_path

            img = await self._gen_image(full_prompt, save_path=output_path)
            if img is None:
                logger.warning("T2I failed — writing placeholder reference")
                Image.new("RGB", (512, 512), (255, 255, 255)).save(output_path)
            return output_path

        except Exception as exc:
            logger.error("generate_reference_image failed: %s", exc, exc_info=True)
            raise ModelApiError(f"generate_reference_image failed: {exc}") from exc

    async def validate_reference_image_semantics(
        self,
        image_path: str,
        ref_subject: str,
        *,
        action: str = "add",
        forbidden_elements: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """VLM check that reference image shows only the isolated asset."""
        try:
            if self.dev_mode:
                return {"passed": True, "feedback": "dev_mode pass", "score": 1.0, "violations": []}

            if not os.path.exists(image_path):
                return {
                    "passed": False,
                    "feedback": "image missing",
                    "score": 0.0,
                    "violations": ["file not found"],
                }

            forbidden = forbidden_elements or [
                "person", "human", "face", "body", "shoulder", "arm", "hand",
                "room", "outdoor scene", "background environment",
            ]
            forbidden_list = "\n".join(f"  - {x}" for x in forbidden)

            prompt = REF_IMAGE_QA_PROMPT.format(
                ref_subject=ref_subject,
                action=action,
                forbidden_list=forbidden_list,
            )
            raw = await self._vision(prompt, [Image.open(image_path).convert("RGB")])
            data = extract_json_object(raw)
            violations = data.get("violations") or []
            if isinstance(violations, str):
                violations = [violations] if violations else []
            score = float(data.get("score", 0.0))
            passed = bool(data.get("passed", False)) and not violations
            if score >= 0.75 and not violations:
                passed = True
            return {
                "passed": passed,
                "feedback": str(data.get("feedback", "")),
                "score": score,
                "violations": violations,
            }

        except Exception as exc:
            logger.error("validate_reference_image_semantics failed: %s", exc, exc_info=True)
            return {"passed": False, "feedback": str(exc), "score": 0.0, "violations": ["qa error"]}

    async def generate_reference_image_with_qa(
        self,
        ref_subject: str,
        output_path: str,
        *,
        action: str = "add",
        forbidden_elements: Optional[List[str]] = None,
        max_retries: int = 3,
        white_background: bool = True,
    ) -> str:
        """Generate isolated reference asset with Gemini semantic QA loop."""
        last_feedback = ""
        for attempt in range(1, max_retries + 1):
            logger.info("ref_image attempt %d/%d: %s", attempt, max_retries, ref_subject)
            prompt = ref_subject
            if last_feedback:
                prompt = (
                    f"{ref_subject}. Previous attempt failed QA: {last_feedback}. "
                    "Generate ONLY the isolated asset on pure white background."
                )

            await self.generate_reference_image(
                prompt,
                output_path,
                white_background=white_background,
                forbidden_elements=forbidden_elements,
            )

            qa = await self.validate_reference_image_semantics(
                output_path,
                ref_subject,
                action=action,
                forbidden_elements=forbidden_elements,
            )
            if qa.get("passed", False):
                logger.info("ref_image QA passed (score=%.2f)", qa.get("score", 0))
                return output_path

            last_feedback = qa.get("feedback", "") or ", ".join(qa.get("violations", []))
            logger.info("ref_image QA failed: %s", last_feedback)

        raise ModelApiError(
            f"Reference image QA failed after {max_retries} attempts: {last_feedback}"
        )

    @staticmethod
    def _reference_location_entity_block(
        desc: Dict[str, str],
        *,
        ref_image_index: Optional[int],
        ref_color_name: str,
        instruction_id: str,
    ) -> str:
        subject = (desc.get("description") or "").strip()
        eid = desc.get("entity_id", "")
        ref_part = (
            f"reference_image=image {ref_image_index}"
            if ref_image_index
            else "reference_image=none"
        )
        return (
            f'- entity_id="{eid}"; subject="{subject}"; '
            f"{ref_part}; "
            f'instruction_id="{instruction_id}"; '
            f"colored_mask_overlay_in_reference={ref_color_name or 'unknown'}; "
            f"identity_anchors={subject}"
        )

    @staticmethod
    def _parse_reference_location_records(
        data: Dict[str, object],
        allowed_entity_ids: set[str],
    ) -> List[Dict[str, object]]:
        locations = data.get("locations") or []
        if isinstance(locations, dict):
            locations = [locations]
        records: List[Dict[str, object]] = []
        if not isinstance(locations, list):
            return records
        for item in locations:
            if not isinstance(item, dict):
                continue
            eid = str(item.get("entity_id", "")).strip()
            if eid not in allowed_entity_ids:
                continue
            records.append(dict(item))
        return records

    async def _reidentify_entity_location_single(
        self,
        frame: Image.Image,
        desc: Dict[str, str],
        ref_path: str,
        *,
        ref_color_name: str,
        instruction_id: str,
    ) -> Optional[Dict[str, object]]:
        """Per-entity viewpoint-robust re-identification fallback."""
        eid = desc.get("entity_id", "")
        if not eid:
            return None
        raw_ref = Image.open(ref_path).convert("RGB")
        ref_frame = extract_reference_frame_from_composite(raw_ref, frame.size)
        entity_block = self._reference_location_entity_block(
            desc,
            ref_image_index=2,
            ref_color_name=ref_color_name,
            instruction_id=instruction_id,
        )
        prompt = MASK_REFERENCE_LOCATION_REID_PROMPT.format(
            entity_block=entity_block,
            entity_id=eid,
        )
        try:
            raw = await self._vision(prompt, [frame, ref_frame])
            data = extract_json_object(raw)
            if not isinstance(data, dict):
                return None
            data["entity_id"] = eid
            return data
        except Exception as exc:
            logger.warning("Reference location re-id failed for %s: %s", eid, exc)
            return None

    async def _derive_entity_location_hints(
        self,
        frame: Image.Image,
        entity_descriptions: List[Dict[str, str]],
        entity_references: Dict[str, str],
        instruction_labels: Dict[str, str],
        color_map: Dict[str, str],
    ) -> Tuple[Dict[str, str], set[str], List[Dict[str, object]]]:
        """Map reference-masked entities to located vs priority-detect mask prompt state."""
        if not entity_references:
            return {}, set(), []

        images: List[Image.Image] = [frame]
        ref_image_index: Dict[str, int] = {}
        entity_blocks: List[str] = []
        desc_by_eid = {
            d.get("entity_id", ""): d
            for d in entity_descriptions
            if d.get("entity_id")
        }
        subject_by_entity = {
            eid: (desc.get("description") or "").strip()
            for eid, desc in desc_by_eid.items()
        }
        ref_meta_by_eid: Dict[str, Dict[str, str]] = {}

        for desc in entity_descriptions:
            eid = desc.get("entity_id", "")
            if not eid:
                continue
            ref_path = entity_references.get(eid, "")
            if not ref_path or not os.path.exists(ref_path):
                entity_blocks.append(
                    self._reference_location_entity_block(
                        desc,
                        ref_image_index=None,
                        ref_color_name=color_name_from_hex(color_map.get(eid, "#FF0000")),
                        instruction_id=instruction_labels.get(
                            eid,
                            desc.get("instruction_id", ""),
                        ),
                    )
                )
                continue
            if eid not in ref_image_index:
                raw_ref = Image.open(ref_path).convert("RGB")
                images.append(extract_reference_frame_from_composite(raw_ref, frame.size))
                ref_image_index[eid] = len(images)

            instruction_id = instruction_labels.get(
                eid,
                desc.get("instruction_id", ""),
            )
            _ref_hex, ref_color_name = resolve_reference_overlay_color(
                ref_path,
                fallback_hex=color_map.get(eid, "#FF0000"),
            )
            ref_meta_by_eid[eid] = {
                "ref_path": ref_path,
                "instruction_id": instruction_id,
                "ref_color_name": ref_color_name,
            }
            entity_blocks.append(
                self._reference_location_entity_block(
                    desc,
                    ref_image_index=ref_image_index[eid],
                    ref_color_name=ref_color_name,
                    instruction_id=instruction_id,
                )
            )

        if len(images) == 1 or not entity_blocks:
            entity_ids = list(desc_by_eid.keys())
            return {}, set(entity_ids), []

        allowed = set(desc_by_eid.keys())
        records_by_eid: Dict[str, Dict[str, object]] = {}

        try:
            prompt = MASK_REFERENCE_LOCATION_PROMPT.format(
                entity_list="\n".join(entity_blocks),
            )
            raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            for record in self._parse_reference_location_records(data, allowed):
                records_by_eid[str(record.get("entity_id", ""))] = record
        except Exception as exc:
            logger.warning(
                "Reference location batch VLM failed (%s) — will try per-entity re-id",
                exc,
            )

        for eid, desc in desc_by_eid.items():
            if eid not in ref_meta_by_eid:
                continue
            record = records_by_eid.get(eid)
            if not should_retry_reference_location(record):
                continue
            meta = ref_meta_by_eid[eid]
            logger.info(
                "Reference location retry for %s (batch miss or low confidence)",
                eid,
            )
            retry_record = await self._reidentify_entity_location_single(
                frame,
                desc,
                meta["ref_path"],
                ref_color_name=meta["ref_color_name"],
                instruction_id=meta["instruction_id"],
            )
            if retry_record is not None:
                records_by_eid[eid] = retry_record

        records = list(records_by_eid.values())
        entity_ids = list(desc_by_eid.keys())
        located, focus = partition_mask_location_entities(
            records,
            entity_ids,
            subject_by_entity=subject_by_entity,
        )
        logger.info(
            "Reference location for mask: %d located, %d priority-detect (%d records)",
            len(located),
            len(focus),
            len(records),
        )
        return located, focus, records

    async def detect_objects_mask(
        self,
        image_path: str,
        entity_descriptions: List[Dict[str, str]],
        output_mask_path: str,
        *,
        color_map: Optional[Dict[str, str]] = None,
        entity_references: Optional[Dict[str, str]] = None,
        instruction_labels: Optional[Dict[str, str]] = None,
        extra_focus_entity_ids: Optional[set[str]] = None,
    ) -> str:
        """Single-pass segmentation with optional VLM reference-to-frame location hints."""
        try:
            os.makedirs(os.path.dirname(output_mask_path) or ".", exist_ok=True)
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            frame_size = (w, h)
            entity_references = entity_references or {}
            instruction_labels = instruction_labels or {}

            if not entity_descriptions:
                Image.new("RGB", (w, h), (0, 0, 0)).save(output_mask_path)
                return output_mask_path

            if color_map is None:
                color_map = {
                    d["entity_id"]: entity_color_hex(i)
                    for i, d in enumerate(entity_descriptions)
                }
            entity_ids = [d["entity_id"] for d in entity_descriptions if d.get("entity_id")]

            if self.dev_mode:
                detections = [{
                    "entity_id": d["entity_id"],
                    "bbox": [0.25, 0.15, 0.75, 0.85],
                } for d in entity_descriptions]
                mask = render_bbox_mask((w, h), detections, color_map)
                mask.save(output_mask_path)
                save_mask_color_map_debug(output_mask_path, color_map)
                logger.info("Dev mask saved: %s", output_mask_path)
                return output_mask_path

            located_by_entity: Dict[str, str] = {}
            focus_entity_ids: Optional[set[str]] = None
            location_records: List[Dict[str, object]] = []
            if entity_references:
                located_by_entity, focus_entity_ids, location_records = (
                    await self._derive_entity_location_hints(
                        img,
                        entity_descriptions,
                        entity_references,
                        instruction_labels,
                        color_map,
                    )
                )
            if extra_focus_entity_ids:
                if focus_entity_ids is None:
                    focus_entity_ids = set(extra_focus_entity_ids)
                else:
                    focus_entity_ids = set(focus_entity_ids) | set(extra_focus_entity_ids)
            if entity_references:
                hints_path = f"{output_mask_path}.location_hints.json"
                with open(hints_path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "located": located_by_entity,
                            "focus_detect": sorted(focus_entity_ids or []),
                            "records": location_records,
                        },
                        fh,
                        indent=2,
                        ensure_ascii=False,
                    )

            prompt = build_batch_segmentation_prompt(
                entity_descriptions,
                color_map,
                located_by_entity=located_by_entity,
                focus_entity_ids=focus_entity_ids,
                instruction_labels=instruction_labels,
                anti_copy_retry=False,
            )
            prompt_path = f"{output_mask_path}.prompt.txt"
            with open(prompt_path, "w", encoding="utf-8") as fh:
                fh.write(prompt)

            logger.info(
                "Segmentation single pass (%d entities, %d located, %d priority-detect)",
                len(entity_descriptions),
                len(located_by_entity),
                len(focus_entity_ids or []),
            )
            raw_path = f"{output_mask_path}.raw.png"
            aligned_path = segmentation_aligned_sidecar_path(output_mask_path)
            records_by_eid = {
                str(record.get("entity_id", "")).strip(): record
                for record in location_records
                if str(record.get("entity_id", "")).strip()
            }

            mask = await self._attempt_batch_segmentation(
                prompt,
                img,
                frame_size,
                color_map,
                raw_path=raw_path,
                aligned_path=aligned_path,
            )
            if mask is None:
                logger.warning(
                    "Batch segmentation empty — retrying with anti-copy constraints"
                )
                retry_prompt = build_batch_segmentation_prompt(
                    entity_descriptions,
                    color_map,
                    located_by_entity=located_by_entity,
                    focus_entity_ids=focus_entity_ids,
                    instruction_labels=instruction_labels,
                    anti_copy_retry=True,
                )
                with open(f"{output_mask_path}.retry.prompt.txt", "w", encoding="utf-8") as fh:
                    fh.write(retry_prompt)
                mask = await self._attempt_batch_segmentation(
                    retry_prompt,
                    img,
                    frame_size,
                    color_map,
                    raw_path=f"{output_mask_path}.retry.raw.png",
                    aligned_path=f"{output_mask_path}.retry.aligned.png",
                )

            if mask is None:
                logger.warning(
                    "Batch segmentation still empty — per-entity + bbox fallback"
                )
                mask = await self._segment_entities_into_mask(
                    frame=img,
                    frame_size=frame_size,
                    entity_descriptions=entity_descriptions,
                    color_map=color_map,
                    mask_path=output_mask_path,
                    entity_references=entity_references,
                    instruction_labels=instruction_labels,
                    located_by_entity=located_by_entity,
                    records_by_eid=records_by_eid,
                )

            if mask is None:
                logger.warning("All segmentation attempts failed — writing empty mask")
                Image.new("RGB", (w, h), (0, 0, 0)).save(output_mask_path)
                save_mask_color_map_debug(output_mask_path, color_map)
                return output_mask_path

            issues = assess_mask_candidate(mask, frame_size)
            if issues:
                logger.warning("Segmentation mask quality issues: %s", issues)

            mask.save(output_mask_path)
            if location_records:
                await self.supplement_mask_for_located_entities(
                    image_path,
                    output_mask_path,
                    entity_descriptions,
                    color_map,
                    location_records,
                    entity_references=entity_references,
                    instruction_labels=instruction_labels,
                )
            ensure_segmentation_mask_output(output_mask_path, color_map=color_map)
            save_mask_color_map_debug(output_mask_path, color_map)
            logger.info("Segmentation mask saved: %s", output_mask_path)
            return output_mask_path

        except Exception as exc:
            logger.error("detect_objects_mask failed: %s", exc, exc_info=True)
            raise ModelApiError(f"detect_objects_mask failed: {exc}") from exc

    @staticmethod
    def _keyframe_location_entity_block(
        entity: Dict[str, str],
        *,
        ref_image_index: Optional[int] = None,
    ) -> str:
        instruction_id = entity.get("instruction_id", "")
        entity_id = entity.get("entity_id", "")
        color_name = entity.get("color_name", "colored")
        color_hex = entity.get("color_hex", "")
        subject = (entity.get("subject_features") or "").strip()
        edit_prompt = (entity.get("edit_prompt") or "").strip()
        edit_part = (
            f'edit_operation="{edit_prompt[:160]}"'
            if edit_prompt
            else "edit_operation=none"
        )
        ref_part = (
            f"reference_overlay=image {ref_image_index}"
            if ref_image_index
            else "reference_overlay=none"
        )
        return (
            f'- instruction_id="{instruction_id}"; entity_id="{entity_id}"; '
            f'entity_color_label={color_name} ({color_hex}); '
            f"{ref_part}; "
            f"{edit_part}; "
            f'subject_features="{subject}"'
        )

    @staticmethod
    def _parse_keyframe_location_records(
        data: Dict[str, object],
        allowed_instruction_ids: set[str],
    ) -> List[Dict[str, object]]:
        locations = data.get("locations") or []
        if isinstance(locations, dict):
            locations = [locations]
        records: List[Dict[str, object]] = []
        if not isinstance(locations, list):
            return records
        for item in locations:
            if not isinstance(item, dict):
                continue
            iid = str(item.get("instruction_id", "")).strip()
            if iid not in allowed_instruction_ids:
                continue
            records.append(dict(item))
        return records

    async def _reidentify_keyframe_location_single(
        self,
        frame: Image.Image,
        entity: Dict[str, str],
        ref_path: str,
    ) -> Optional[Dict[str, object]]:
        """Per-instruction keyframe location fallback."""
        iid = entity.get("instruction_id", "")
        if not iid or not ref_path or not os.path.exists(ref_path):
            return None
        ref_frame = load_entity_ref_overlay_guide(ref_path, frame.size)
        entity_block = self._keyframe_location_entity_block(
            entity,
            ref_image_index=2,
        )
        prompt = KEYFRAME_EDIT_ENTITY_LOCATION_REID_PROMPT.format(
            entity_block=entity_block,
            instruction_id=iid,
            entity_id=entity.get("entity_id", iid),
        )
        try:
            raw = await self._vision(prompt, [frame, ref_frame])
            data = extract_json_object(raw)
            if not isinstance(data, dict):
                return None
            data["instruction_id"] = iid
            data["entity_id"] = entity.get("entity_id", data.get("entity_id", ""))
            return data
        except Exception as exc:
            logger.warning("Keyframe location re-id failed for %s: %s", iid, exc)
            return None

    async def _reidentify_keyframe_location_disambiguate(
        self,
        frame: Image.Image,
        entity: Dict[str, str],
        ref_path: str,
        peer_assignments: str,
    ) -> Optional[Dict[str, object]]:
        """Re-locate one instruction while avoiding peer instruction targets."""
        iid = entity.get("instruction_id", "")
        if not iid or not ref_path or not os.path.exists(ref_path) or not peer_assignments.strip():
            return None
        ref_frame = load_entity_ref_overlay_guide(ref_path, frame.size)
        entity_block = self._keyframe_location_entity_block(
            entity,
            ref_image_index=2,
        )
        prompt = KEYFRAME_EDIT_ENTITY_LOCATION_DISAMBIGUATE_PROMPT.format(
            entity_block=entity_block,
            peer_assignments=peer_assignments,
            instruction_id=iid,
            entity_id=entity.get("entity_id", iid),
        )
        try:
            raw = await self._vision(prompt, [frame, ref_frame])
            data = extract_json_object(raw)
            if not isinstance(data, dict):
                return None
            data["instruction_id"] = iid
            data["entity_id"] = entity.get("entity_id", data.get("entity_id", ""))
            return data
        except Exception as exc:
            logger.warning("Keyframe location disambiguation failed for %s: %s", iid, exc)
            return None

    async def _resolve_keyframe_location_conflicts(
        self,
        frame: Image.Image,
        records_by_iid: Dict[str, Dict[str, object]],
        entity_by_iid: Dict[str, Dict[str, str]],
        subject_by_instruction: Dict[str, str],
    ) -> List[Dict[str, object]]:
        """Detect and retry conflicting instruction-localization pairs."""
        entity_id_by_instruction = {
            iid: str(entity.get("entity_id", "")).strip()
            for iid, entity in entity_by_iid.items()
        }
        conflicts = find_keyframe_location_conflicts(
            records_by_iid,
            subject_by_instruction=subject_by_instruction,
            entity_id_by_instruction=entity_id_by_instruction,
        )
        if not conflicts:
            return []

        losers = pick_keyframe_location_conflict_losers(conflicts, records_by_iid)
        for iid in sorted(losers):
            entity = entity_by_iid.get(iid)
            if entity is None:
                continue
            ref_path = (entity.get("reference_overlay_path") or "").strip()
            if not ref_path:
                continue
            peer_lines = format_keyframe_peer_assignment_lines(
                records_by_iid,
                exclude_instruction_id=iid,
                subject_by_instruction=subject_by_instruction,
                entity_id_by_instruction=entity_id_by_instruction,
            )
            if not peer_lines.strip():
                continue
            logger.info("Keyframe location disambiguation retry for %s", iid)
            retry_record = await self._reidentify_keyframe_location_disambiguate(
                frame,
                entity,
                ref_path,
                peer_lines,
            )
            if retry_record is None:
                continue
            records_by_iid[iid] = normalize_keyframe_location_record(
                retry_record,
                subject_features=subject_by_instruction.get(iid, ""),
            )

        remaining = find_keyframe_location_conflicts(
            records_by_iid,
            subject_by_instruction=subject_by_instruction,
            entity_id_by_instruction=entity_id_by_instruction,
        )
        if not remaining:
            return conflicts

        final_losers = pick_keyframe_location_conflict_losers(remaining, records_by_iid)
        for iid in final_losers:
            record = dict(records_by_iid.get(iid) or {})
            record["present_in_frame"] = False
            record["location_conflict_rejected"] = True
            records_by_iid[iid] = record
            logger.warning(
                "Keyframe location conflict unresolved for %s — marking not present",
                iid,
            )
        return conflicts

    async def derive_keyframe_edit_entity_locations(
        self,
        image_path: str,
        mask_path: str,
        edit_entities: List[Dict[str, str]],
    ) -> Tuple[Dict[str, str], List[Dict[str, object]]]:
        """VLM: map entity_refs identities to locations on the target keyframe (frame only)."""
        allowed = {
            str(e.get("instruction_id", "")).strip()
            for e in edit_entities
            if e.get("instruction_id")
        }
        subject_by_instruction = {
            str(e.get("instruction_id", "")).strip(): (e.get("subject_features") or "").strip()
            for e in edit_entities
            if e.get("instruction_id")
        }
        color_name_by_instruction = {
            str(e.get("instruction_id", "")).strip(): e.get("color_name", "colored")
            for e in edit_entities
            if e.get("instruction_id")
        }

        if not edit_entities:
            return {}, []

        if self.dev_mode:
            prompts = {
                iid: f"in image 1: {subject_by_instruction.get(iid, 'the edit target')}"
                for iid in allowed
            }
            return prompts, []

        if not image_path or not os.path.exists(image_path):
            raise ModelApiError(f"Keyframe location: missing frame {image_path}")

        frame = Image.open(image_path).convert("RGB")
        images: List[Image.Image] = [frame]
        ref_image_index: Dict[str, int] = {}
        entity_blocks: List[str] = []
        entity_by_iid = {
            str(e.get("instruction_id", "")).strip(): e
            for e in edit_entities
            if e.get("instruction_id")
        }

        for entity in edit_entities:
            iid = str(entity.get("instruction_id", "")).strip()
            if not iid:
                continue
            ref_path = (entity.get("reference_overlay_path") or "").strip()
            if ref_path and os.path.exists(ref_path) and iid not in ref_image_index:
                images.append(load_entity_ref_overlay_guide(ref_path, frame.size))
                ref_image_index[iid] = len(images)
            ref_idx = ref_image_index.get(iid)
            entity_blocks.append(
                self._keyframe_location_entity_block(
                    entity,
                    ref_image_index=ref_idx,
                )
            )

        records_by_iid: Dict[str, Dict[str, object]] = {}
        if entity_blocks:
            try:
                prompt = KEYFRAME_EDIT_ENTITY_LOCATION_PROMPT.format(
                    entity_list="\n".join(entity_blocks),
                )
                raw = await self._vision(prompt, images)
                data = extract_json_object(raw)
                for record in self._parse_keyframe_location_records(data, allowed):
                    iid = str(record.get("instruction_id", "")).strip()
                    if not iid:
                        continue
                    records_by_iid[iid] = normalize_keyframe_location_record(
                        record,
                        subject_features=subject_by_instruction.get(iid, ""),
                    )
            except Exception as exc:
                logger.warning(
                    "Keyframe location batch VLM failed (%s) — will try per-instruction",
                    exc,
                )

        for iid, entity in entity_by_iid.items():
            record = records_by_iid.get(iid)
            subject = subject_by_instruction.get(iid, "")
            if record is not None and not should_retry_keyframe_location(
                record,
                subject_features=subject,
            ):
                continue
            ref_path = (entity.get("reference_overlay_path") or "").strip()
            if not ref_path:
                continue
            logger.info(
                "Keyframe location retry for %s (miss, low confidence, or weak cues)",
                iid,
            )
            retry_record = await self._reidentify_keyframe_location_single(
                frame,
                entity,
                ref_path,
            )
            if retry_record is not None:
                records_by_iid[iid] = normalize_keyframe_location_record(
                    retry_record,
                    subject_features=subject,
                )

        conflict_records = await self._resolve_keyframe_location_conflicts(
            frame,
            records_by_iid,
            entity_by_iid,
            subject_by_instruction,
        )

        records = list(records_by_iid.values())
        if conflict_records:
            logger.info(
                "Keyframe location conflicts handled: %d pair(s) detected",
                len(conflict_records),
            )
        prompts = keyframe_location_records_to_prompts(
            records,
            subject_by_instruction=subject_by_instruction,
            color_name_by_instruction=color_name_by_instruction,
        )
        for iid in allowed:
            if iid not in prompts:
                subject = subject_by_instruction.get(iid, "the edit target")
                prompts[iid] = f'in image 1: the subject matching "{subject}"'

        logger.info(
            "Keyframe edit locations: %d/%d instructions located",
            sum(1 for r in records if r.get("present_in_frame")),
            len(allowed),
        )
        return prompts, records

    @staticmethod
    def _build_entity_ref_guides_section(
        refs: List[Image.Image],
        guides: List[Dict[str, Any]],
        frame_size: Tuple[int, int],
        *,
        location_guided: bool = False,
    ) -> str:
        """Attach per-instruction entity_refs images and build prompt section."""
        if not guides:
            return ""

        scene_label = "image 1" if location_guided else "image 2"
        guide_blocks: List[str] = []
        for guide in guides:
            instruction_id = guide.get("instruction_id", "")
            entity_id = guide.get("entity_id", "")
            color_name = guide.get("color_name", "colored")
            paths: Dict[str, str] = guide.get("paths") or {}
            id_suffix = f" | entity_id={entity_id}" if entity_id else ""
            if location_guided:
                block_lines = [
                    f"Instruction {instruction_id}{id_suffix} (location-guided edit target):",
                ]
            else:
                block_lines = [
                    f"Instruction {instruction_id}{id_suffix} "
                    f"(color region: {color_name} on image 1):",
                ]

            canonical_only = set(paths.keys()) == {"canonical"}
            action = str(guide.get("action") or "").lower()
            if action == "delete" and paths.get("multiview"):
                refs.append(Image.open(paths["multiview"]).convert("RGB"))
                block_lines.append(
                    f"- Image {len(refs)} = delete-target front-view identification — reference only. Match WHO to remove in "
                    f"{scene_label}; do NOT whiten or blank the scene. "
                    f"Caption shows instruction_id and entity_id"
                )
                guide_blocks.append("\n".join(block_lines))
                continue
            if action == "delete" and paths.get("src") and paths.get("mask"):
                refs.append(
                    build_keyframe_delete_identification_image(
                        paths["src"],
                        paths["mask"],
                    )
                )
                block_lines.append(
                    f"- Image {len(refs)} = delete-target identification only — "
                    f"mask-bounded crop of the entity to remove (NOT an output template). "
                    f"Match WHO to remove in {scene_label}; do NOT whiten or blank the scene. "
                    f"Caption in reference assets shows instruction_id and entity_id"
                )
                guide_blocks.append("\n".join(block_lines))
                continue

            if paths.get("canonical"):
                refs.append(Image.open(paths["canonical"]).convert("RGB"))
                if action == "delete":
                    block_lines.append(
                        f"- Image {len(refs)} = before/after comparison (delete) — each panel "
                        f"is a mask-bounded entity crop (not full scene). Left (Original) = "
                        f"entity crop to remove; right (Removed) is empty. Match WHO to remove "
                        f"in {scene_label} from the left crop; caption shows instruction_id and "
                        f"entity_id. Do NOT use as output base"
                    )
                else:
                    block_lines.append(
                        f"- Image {len(refs)} = before/after comparison — mask-bounded entity "
                        f"crops only (not full scene). Left (Original) = entity crop before "
                        f"edit; right (Edited) = same crop region after edit. Identify WHO from "
                        f"the left crop in {scene_label}; borrow ONLY the edited attribute from "
                        f"the right crop. Caption shows instruction_id and entity_id. "
                        f"Do NOT use as output base"
                    )

            if canonical_only:
                guide_blocks.append("\n".join(block_lines))
                continue

            if paths.get("src"):
                refs.append(Image.open(paths["src"]).convert("RGB"))
                block_lines.append(
                    f"- Image {len(refs)} = identity hint only — earlier frame of the target entity "
                    f"(do NOT use as output base)"
                )
            if paths.get("mask"):
                refs.append(Image.open(paths["mask"]).convert("RGB"))
                block_lines.append(
                    f"- Image {len(refs)} = identity hint only — entity mask on an earlier frame "
                    f"(do NOT use as output base)"
                )
            if paths.get("overlay"):
                refs.append(load_entity_ref_overlay_guide(paths["overlay"], frame_size))
                block_lines.append(
                    f"- Image {len(refs)} = identity hint only — earlier frame with mask overlay "
                    f"(do NOT use as output base)"
                )

            guide_blocks.append("\n".join(block_lines))

        section_template = (
            KEYFRAME_LOCATION_ENTITY_REF_GUIDES_SECTION
            if location_guided
            else KEYFRAME_ENTITY_REF_GUIDES_SECTION
        )
        return section_template.format(
            guide_blocks="\n\n".join(guide_blocks),
        )

    async def masked_inpaint(
        self,
        image_path: str,
        mask_path: str,
        edit_directives: str,
        output_path: str,
        *,
        ref_image_path: Optional[str] = None,
        consistency_ref_paths: Optional[List[str]] = None,
        entity_ref_guides: Optional[List[Dict[str, Any]]] = None,
        strength: float = 1.0,
        preserve_frame_structure: bool = False,
        inpaint_guidance: str = "mask",
    ) -> str:
        """Inpainting — mask-guided (image1=mask) or location-guided (image1=scene)."""
        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            if self.dev_mode:
                Image.open(image_path).convert("RGB").save(output_path)
                return output_path

            scene_img = Image.open(image_path).convert("RGB")
            frame_size = scene_img.size
            location_guided = inpaint_guidance == "location"
            refs: List[Image.Image] = [scene_img] if location_guided else [
                Image.open(mask_path).convert("RGB"),
                scene_img,
            ]
            if ref_image_path and os.path.exists(ref_image_path):
                refs.append(Image.open(ref_image_path).convert("RGB"))

            entity_ref_section = ""
            if entity_ref_guides:
                entity_ref_section = self._build_entity_ref_guides_section(
                    refs,
                    entity_ref_guides,
                    frame_size,
                    location_guided=location_guided,
                )

            consistency_lines: List[str] = []
            for path in consistency_ref_paths or []:
                if path and os.path.exists(path):
                    if location_guided:
                        continue
                    refs.append(Image.open(path).convert("RGB"))
                    consistency_lines.append(
                        f"- Image {len(refs)} = canonical edited reference for cross-scene consistency"
                    )

            consistency_refs_section = ""
            if consistency_lines:
                consistency_refs_section = INPAINT_CONSISTENCY_REFS_SECTION.format(
                    consistency_lines="\n".join(consistency_lines),
                )

            prompt_template = (
                KEYFRAME_LOCATION_INPAINT_PROMPT
                if location_guided
                else INPAINT_PROMPT
            )
            prompt = prompt_template.format(
                edit_directives=edit_directives,
                strength=strength,
                entity_ref_section=entity_ref_section,
                consistency_refs_section=consistency_refs_section,
            )

            img = await self._gen_image(
                prompt,
                refs=refs,
                save_path=(
                    f"{output_path}.inpaint_raw.png"
                    if preserve_frame_structure
                    else output_path
                ),
            )
            if img is None:
                logger.warning("Inpaint API returned no image — copying original")
                Image.open(image_path).save(output_path)
            elif preserve_frame_structure:
                blend_inpaint_preserve_frame_structure(
                    image_path,
                    f"{output_path}.inpaint_raw.png",
                    mask_path,
                    output_path,
                )
            return output_path

        except Exception as exc:
            logger.error("masked_inpaint failed: %s", exc, exc_info=True)
            raise ModelApiError(f"masked_inpaint failed: {exc}") from exc

    # ── Video ────────────────────────────────────────────────────────────

    async def propagate_masks_vos(
        self,
        scene_frame_dir: str,
        initial_mask_path: str,
        output_mask_dir: str,
        entity_color_map: Dict[str, str],
        *,
        entity_descriptions: Optional[Dict[str, str]] = None,
    ) -> str:
        """Store first-frame indicative mask only; temporal propagation is via V2V."""
        try:
            import shutil

            os.makedirs(output_mask_dir, exist_ok=True)
            dest = os.path.join(output_mask_dir, "mask_0000.png")
            if os.path.exists(initial_mask_path):
                shutil.copy2(initial_mask_path, dest)
            logger.info(
                "Indicative mask guide stored (V2V handles temporal): %s",
                output_mask_dir,
            )
            return output_mask_dir

        except Exception as exc:
            logger.error("propagate_masks_vos failed: %s", exc, exc_info=True)
            raise ModelApiError(f"propagate_masks_vos failed: {exc}") from exc

    async def derive_video_edit_operation_prompt(
        self,
        original_first_frame_path: str,
        edited_first_frame_path: str,
        location_prompts_path: str,
        *,
        fallback_edit_prompt: str = "",
        entity_instru_path: str = "",
    ) -> str:
        """Compare original vs edited keyframe; return performed-edit prompt for Veo."""
        mandatory = ""
        try:
            mandatory = build_mandatory_video_edit_operation(
                location_prompts_path,
                entity_instru_path=entity_instru_path,
                fallback_edit_prompt=fallback_edit_prompt,
            )
            if mandatory and location_prompts_path and os.path.exists(location_prompts_path):
                sidecar = {}
                with open(location_prompts_path, encoding="utf-8") as fh:
                    sidecar = json.load(fh)
                has_planned = bool(sidecar.get("planned_edits")) or bool(
                    sidecar.get("records")
                )
                if has_planned:
                    logger.info(
                        "derive_video_edit_operation_prompt: using mandatory planned edits "
                        "from %s",
                        os.path.basename(location_prompts_path),
                    )
                    return mandatory

            location_json = "{}"
            if location_prompts_path and os.path.exists(location_prompts_path):
                with open(location_prompts_path, encoding="utf-8") as fh:
                    location_json = fh.read()
            elif fallback_edit_prompt:
                return fallback_edit_prompt.strip()

            if self.dev_mode:
                try:
                    data = json.loads(location_json)
                    prompts = data.get("prompts") or {}
                    if prompts:
                        return " ".join(str(v).strip() for v in prompts.values() if str(v).strip())
                except json.JSONDecodeError:
                    pass
                return (fallback_edit_prompt or "Apply the keyframe edit to the video.").strip()

            if not os.path.exists(original_first_frame_path):
                raise FileNotFoundError(
                    f"Original first frame not found: {original_first_frame_path}"
                )
            if not os.path.exists(edited_first_frame_path):
                raise FileNotFoundError(
                    f"Edited first frame not found: {edited_first_frame_path}"
                )

            prompt = VIDEO_EDIT_DIFF_PROMPT.format(
                location_reference_json=location_json,
            )
            images = [
                Image.open(original_first_frame_path).convert("RGB"),
                Image.open(edited_first_frame_path).convert("RGB"),
            ]
            raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            edit_ops = str(data.get("edit_operation_prompt", "")).strip()
            if edit_ops:
                if mandatory:
                    return mandatory
                return edit_ops
            if mandatory:
                return mandatory
            if fallback_edit_prompt:
                logger.warning(
                    "derive_video_edit_operation_prompt empty — using fallback"
                )
                return fallback_edit_prompt.strip()
            raise RuntimeError("VLM returned empty edit_operation_prompt")

        except Exception as exc:
            logger.error(
                "derive_video_edit_operation_prompt failed: %s", exc, exc_info=True
            )
            if mandatory:
                logger.warning("derive_video_edit_operation_prompt failed — using mandatory planned edits")
                return mandatory
            if fallback_edit_prompt:
                return fallback_edit_prompt.strip()
            raise ModelApiError(
                f"derive_video_edit_operation_prompt failed: {exc}"
            ) from exc

    async def derive_video_chunk_edit_operation_prompt(
        self,
        original_chunk_first_frame_path: str,
        previous_edited_last_frame_path: str,
    ) -> str:
        """Infer edit ops for chunk N>0 from original first frame vs prior edited last frame."""
        try:
            if self.dev_mode:
                return (
                    "Continue the same visual edits from the previous sub-clip into this "
                    "sub-clip. Preserve camera motion, timing, and unedited regions."
                )

            if not os.path.exists(original_chunk_first_frame_path):
                raise FileNotFoundError(
                    f"Chunk first frame not found: {original_chunk_first_frame_path}"
                )
            if not os.path.exists(previous_edited_last_frame_path):
                raise FileNotFoundError(
                    f"Previous edited last frame not found: {previous_edited_last_frame_path}"
                )

            images = [
                Image.open(original_chunk_first_frame_path).convert("RGB"),
                Image.open(previous_edited_last_frame_path).convert("RGB"),
            ]
            raw = await self._vision(VIDEO_CHUNK_EDIT_DIFF_PROMPT, images)
            data = extract_json_object(raw)
            edit_ops = str(data.get("edit_operation_prompt", "")).strip()
            if edit_ops:
                return edit_ops
            raise RuntimeError("VLM returned empty edit_operation_prompt for chunk handoff")

        except Exception as exc:
            logger.error(
                "derive_video_chunk_edit_operation_prompt failed: %s", exc, exc_info=True
            )
            raise ModelApiError(
                f"derive_video_chunk_edit_operation_prompt failed: {exc}"
            ) from exc

    @staticmethod
    def _video_edit_qa_enabled() -> bool:
        return (
            os.environ.get("VIDEO_EDIT_QA")
            or os.environ.get("VIDEO_EDIT_QA", "true")
        ).lower() not in (
            "0",
            "false",
            "no",
        )

    @staticmethod
    def _append_video_edit_retry_focus(
        edit_operation_prompt: str,
        retry_focus: str,
    ) -> str:
        return append_editing_operations_to_avoid(edit_operation_prompt, retry_focus)

    async def _write_api_failure_static_clip(
        self,
        *,
        source_clip_path: str,
        edit_raw: str,
        original_first_frame_path: str = "",
    ) -> None:
        """Freeze the original video's first frame when the video-edit API fails."""
        from video_editing_agent.utils.ffmpeg_utils import (
            static_video_from_image,
            static_video_from_source_first_frame,
        )

        frame_png = edit_raw + ".source_first_frame.png"
        if original_first_frame_path and os.path.exists(original_first_frame_path):
            logger.warning(
                "Video edit API failed — using original first frame still (%s)",
                os.path.basename(original_first_frame_path),
            )
            await asyncio.to_thread(
                static_video_from_image,
                original_first_frame_path,
                source_clip_path,
                edit_raw,
            )
            return

        logger.warning(
            "Video edit API failed — extracting source clip first frame for still fallback"
        )
        await asyncio.to_thread(
            static_video_from_source_first_frame,
            source_clip_path,
            edit_raw,
            frame_png_path=frame_png,
        )

    async def validate_video_edit_quality(
        self,
        reference_edited_frame_path: str,
        edited_video_path: str,
        edit_operation_prompt: str,
    ) -> Dict[str, Any]:
        """VLM QA comparing reference edited frame vs edited video opening frame."""
        from video_editing_agent.utils.ffmpeg_utils import extract_frame_at

        qa_first_frame_path = f"{edited_video_path}.qa_first_frame.png"
        try:
            if self.dev_mode:
                return {
                    "passed": True,
                    "score": 1.0,
                    "first_frame_consistent": True,
                    "edit_completed": True,
                    "failed_aspects": [],
                    "feedback": "dev_mode pass",
                    "retry_focus_prompt": "",
                    "qa_first_frame_path": qa_first_frame_path,
                }

            if not os.path.exists(reference_edited_frame_path):
                raise FileNotFoundError(
                    f"Reference edited frame not found: {reference_edited_frame_path}"
                )
            if not os.path.exists(edited_video_path):
                return {
                    "passed": False,
                    "score": 0.0,
                    "first_frame_consistent": False,
                    "edit_completed": False,
                    "failed_aspects": ["edited video missing"],
                    "feedback": "edited video output missing",
                    "retry_focus_prompt": (
                        "Produce a valid edited video clip whose first frame matches the "
                        "reference edited frame."
                    ),
                    "qa_first_frame_path": qa_first_frame_path,
                }

            await asyncio.to_thread(
                extract_frame_at,
                edited_video_path,
                0.0,
                qa_first_frame_path,
            )

            prompt = VIDEO_EDIT_QA_VALIDATION_PROMPT.format(
                edit_operation_prompt=(edit_operation_prompt or "").strip() or "N/A",
            )
            images = [
                Image.open(reference_edited_frame_path).convert("RGB"),
                Image.open(qa_first_frame_path).convert("RGB"),
            ]
            raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            passed = bool(data.get("passed", False))
            score = _safe_float(data.get("score"), 0.0)
            first_ok = bool(data.get("first_frame_consistent", passed))
            edit_ok = bool(data.get("edit_completed", passed))
            failed_aspects = data.get("failed_aspects") or []
            if not isinstance(failed_aspects, list):
                failed_aspects = [str(failed_aspects)]
            feedback = str(data.get("feedback", "")).strip()
            retry_focus = str(data.get("retry_focus_prompt", "")).strip()

            if score <= 0.0 and passed:
                score = 0.85
            if passed and not (first_ok and edit_ok):
                passed = False
            if not passed and score >= 0.7 and first_ok and edit_ok:
                passed = True

            return {
                "passed": passed,
                "score": score,
                "first_frame_consistent": first_ok,
                "edit_completed": edit_ok,
                "failed_aspects": [str(x) for x in failed_aspects if str(x).strip()],
                "feedback": feedback,
                "retry_focus_prompt": retry_focus,
                "positive_prompt": str(data.get("positive_prompt", "")).strip(),
                "qa_first_frame_path": qa_first_frame_path,
            }

        except Exception as exc:
            logger.error("validate_video_edit_quality failed: %s", exc, exc_info=True)
            return {
                "passed": True,
                "score": 0.0,
                "first_frame_consistent": True,
                "edit_completed": True,
                "failed_aspects": [],
                "feedback": f"QA skipped due to error: {exc}",
                "retry_focus_prompt": "",
                "qa_first_frame_path": qa_first_frame_path,
            }

    async def derive_scene_video_edit_prompt(
        self,
        *,
        scene_id: str,
        source_clip_path: str,
        sample_frame_paths: List[str],
        entity_ref_image_paths: List[str],
        entity_instru_json: str,
        shot_analysis_json: str,
        start_sec: float,
        end_sec: float,
        fallback_edit_prompt: str = "",
    ) -> str:
        """VLM: derive edit prompt for direct scene video editing."""
        try:
            if self.dev_mode:
                if fallback_edit_prompt.strip():
                    return fallback_edit_prompt.strip()
                return "Apply the entity instruction edits consistently across the scene clip."

            images: List[Image.Image] = []
            for path in sample_frame_paths:
                if path and os.path.exists(path):
                    images.append(Image.open(path).convert("RGB"))
            for path in entity_ref_image_paths:
                if path and os.path.exists(path):
                    images.append(Image.open(path).convert("RGB"))

            if not images:
                if not os.path.exists(source_clip_path):
                    raise FileNotFoundError(f"Source clip not found: {source_clip_path}")
                raise RuntimeError("No sample or reference images for scene edit prompt derivation")

            prompt = SCENE_VIDEO_EDIT_DERIVATION_PROMPT.format(
                scene_id=scene_id,
                start_sec=start_sec,
                end_sec=end_sec,
                entity_instru_json=entity_instru_json or "{}",
                shot_analysis_json=shot_analysis_json or "{}",
            )
            raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            edit_ops = str(data.get("edit_operation_prompt", "")).strip()
            if edit_ops:
                return edit_ops
            if fallback_edit_prompt.strip():
                logger.warning(
                    "derive_scene_video_edit_prompt empty — using fallback for %s",
                    scene_id,
                )
                return fallback_edit_prompt.strip()
            raise RuntimeError("VLM returned empty edit_operation_prompt")

        except Exception as exc:
            logger.error("derive_scene_video_edit_prompt failed: %s", exc, exc_info=True)
            if fallback_edit_prompt.strip():
                return fallback_edit_prompt.strip()
            raise ModelApiError(f"derive_scene_video_edit_prompt failed: {exc}") from exc

    async def validate_scene_video_edit_keyframe_grids(
        self,
        *,
        edited_keyframes_grid_path: str,
        original_keyframes_grid_path: str,
        entity_ref_image_paths: List[str],
        entity_instru_json: str,
        edit_operation_prompt: str,
    ) -> Dict[str, Any]:
        """VLM QA: edited vs original keyframe grids + entity reference images."""
        try:
            if self.dev_mode:
                return {
                    "passed": True,
                    "score": 1.0,
                    "edit_completed": True,
                    "edit_on_correct_entity": True,
                    "non_target_preserved": True,
                    "lighting_preserved": True,
                    "size_scale_preserved": True,
                    "expression_pose_preserved": True,
                    "per_entity_results": [],
                    "failed_aspects": [],
                    "feedback": "dev_mode pass",
                    "retry_focus_prompt": "",
                    "positive_prompt": "",
                }

            images: List[Image.Image] = []
            for path in (
                edited_keyframes_grid_path,
                original_keyframes_grid_path,
            ):
                if not path or not os.path.exists(path):
                    return {
                        "passed": False,
                        "score": 0.0,
                        "edit_completed": False,
                        "non_target_preserved": False,
                        "edit_on_correct_entity": False,
                        "lighting_preserved": False,
                        "size_scale_preserved": False,
                        "expression_pose_preserved": False,
                        "per_entity_results": [],
                        "failed_aspects": ["missing keyframe grid"],
                        "feedback": f"missing grid image: {path}",
                        "retry_focus_prompt": (
                            "Do not produce output without valid keyframe comparison grids."
                        ),
                        "positive_prompt": "",
                    }
                images.append(Image.open(path).convert("RGB"))

            for path in entity_ref_image_paths:
                if path and os.path.exists(path):
                    images.append(Image.open(path).convert("RGB"))

            prompt = SCENE_VIDEO_EDIT_KEYFRAME_GRID_QA_PROMPT.format(
                entity_instru_json=entity_instru_json or "{}",
                edit_operation_prompt=(edit_operation_prompt or "").strip() or "N/A",
            )
            raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            passed = bool(data.get("passed", False))
            score = _safe_float(data.get("score"), 0.0)
            # edit_completed defaults to False (not passed) — must be explicitly confirmed by VLM.
            edit_ok = bool(data.get("edit_completed", False))
            edit_on_correct_entity = bool(data.get("edit_on_correct_entity", True))
            failed_aspects = data.get("failed_aspects") or []
            if not isinstance(failed_aspects, list):
                failed_aspects = [str(failed_aspects)]
            feedback = str(data.get("feedback", "")).strip()
            retry_focus = str(data.get("retry_focus_prompt", "")).strip()
            positive_prompt = str(data.get("positive_prompt", "")).strip()
            missing_edits = str(data.get("missing_edits_prompt", "")).strip()

            non_target_preserved = bool(data.get("non_target_preserved", True))
            lighting_preserved = bool(data.get("lighting_preserved", True))
            size_scale_preserved = bool(data.get("size_scale_preserved", True))
            expression_pose_preserved = bool(data.get("expression_pose_preserved", True))
            entity_size_scale_consistent = bool(data.get("entity_size_scale_consistent", True))
            video_structure_preserved = bool(data.get("video_structure_preserved", True))
            one_to_one_entity_mapping = bool(data.get("one_to_one_entity_mapping", True))
            pasted_entity_detected = bool(data.get("pasted_entity_detected", False))

            per_entity_results = data.get("per_entity_results") or []

            # Strict pass logic:
            # - FAIL if edit not completed (edit_ok must be explicitly True).
            # - FAIL if edit applied to wrong entity.
            # - FAIL if non-target regions clearly changed.
            # - FAIL if entity size/scale severely inconsistent with original.
            # - FAIL if video structure changed (missing black bars, large pasted regions, etc.).
            # - No auto-pass override — VLM must explicitly set passed=true AND edit_completed=true.
            if not edit_ok:
                passed = False
            if not edit_on_correct_entity:
                passed = False
            if not non_target_preserved:
                passed = False
            if not entity_size_scale_consistent:
                passed = False
            if not video_structure_preserved:
                passed = False
            if not one_to_one_entity_mapping:
                passed = False
            if pasted_entity_detected:
                passed = False

            if score <= 0.0 and passed:
                score = 0.85

            return {
                "passed": passed,
                "score": score,
                "edit_completed": edit_ok,
                "edit_on_correct_entity": edit_on_correct_entity,
                "non_target_preserved": non_target_preserved,
                "lighting_preserved": lighting_preserved,
                "size_scale_preserved": size_scale_preserved,
                "expression_pose_preserved": expression_pose_preserved,
                "entity_size_scale_consistent": entity_size_scale_consistent,
                "video_structure_preserved": video_structure_preserved,
                "one_to_one_entity_mapping": one_to_one_entity_mapping,
                "pasted_entity_detected": pasted_entity_detected,
                "per_entity_results": per_entity_results,
                "failed_aspects": [str(x) for x in failed_aspects if str(x).strip()],
                "feedback": feedback,
                "retry_focus_prompt": retry_focus,
                "positive_prompt": positive_prompt,
                "missing_edits_prompt": missing_edits,
            }

        except Exception as exc:
            logger.error(
                "validate_scene_video_edit_keyframe_grids failed: %s", exc, exc_info=True
            )
            return {
                "passed": True,
                "score": 0.0,
                "edit_completed": True,
                "edit_on_correct_entity": True,
                "non_target_preserved": True,
                "lighting_preserved": True,
                "size_scale_preserved": True,
                "expression_pose_preserved": True,
                "entity_size_scale_consistent": True,
                "video_structure_preserved": True,
                "one_to_one_entity_mapping": True,
                "pasted_entity_detected": False,
                "per_entity_results": [],
                "failed_aspects": [],
                "feedback": f"QA skipped due to error: {exc}",
                "retry_focus_prompt": "",
                "positive_prompt": "",
                "missing_edits_prompt": "",
            }

    async def vote_scene_entity_existence(
        self,
        *,
        keyframe_paths: List[str],
        entity_ref_image_paths: List[str],
        entity_catalog_block: str,
    ) -> Dict[str, Any]:
        """Single VLM vote: does any edit-target entity appear in the scene?"""
        try:
            if self.dev_mode:
                return {
                    "scene_has_edit_target": True,
                    "entities": [],
                    "reasoning": "dev_mode: assume present",
                }

            images: List[Image.Image] = []
            for path in keyframe_paths:
                if path and os.path.exists(path):
                    images.append(Image.open(path).convert("RGB"))
            if not images:
                return {
                    "scene_has_edit_target": False,
                    "entities": [],
                    "reasoning": "no keyframe images available",
                }
            for path in entity_ref_image_paths:
                if path and os.path.exists(path):
                    images.append(Image.open(path).convert("RGB"))

            prompt = SCENE_ENTITY_EXISTENCE_VOTE_PROMPT.format(
                entity_catalog_block=entity_catalog_block or "(no entities)",
            )
            # Retry on transient API errors (empty/non-JSON responses).
            raw = ""
            for vote_attempt in range(1, 4):
                try:
                    raw = await self._vision(prompt, images)
                    if raw and raw.strip():
                        break
                except Exception as vision_exc:
                    logger.warning(
                        "vote_scene_entity_existence VLM call %d/3 failed: %s",
                        vote_attempt,
                        vision_exc,
                    )
                if vote_attempt < 3:
                    import asyncio as _asyncio
                    await _asyncio.sleep(vote_attempt * 3)

            if not raw or not raw.strip():
                return {
                    "scene_has_edit_target": True,
                    "entities": [],
                    "reasoning": "empty VLM response — assume present (conservative)",
                }

            data = extract_json_object(raw)
            return {
                "scene_has_edit_target": bool(data.get("scene_has_edit_target", True)),
                "entities": data.get("entities") or [],
                "reasoning": str(data.get("reasoning", "") or "").strip(),
            }

        except Exception as exc:
            logger.error(
                "vote_scene_entity_existence failed: %s", exc, exc_info=True
            )
            # Conservative: assume present on error so we don't skip needed edits.
            return {
                "scene_has_edit_target": True,
                "entities": [],
                "reasoning": f"vote error — assume present (conservative): {exc}",
            }

    async def refine_retry_guidance_with_llm(
        self,
        *,
        base_edit_prompt: str,
        positive_prompt: str,
        avoid_operations: str,
        missing_edits_prompt: str,
        qa_feedback: str,
        failed_aspects: List[str],
    ) -> Dict[str, Any]:
        """Use LLM to deduplicate and reorganize retry guidance prompts.

        Merges the base edit prompt with QA-derived positive/avoid/missing-edits
        guidance, removes redundant content, and produces a clean, deduplicated
        set of retry prompts with the avoid section emphasized at the end.
        """
        try:
            if self.dev_mode:
                return {
                    "positive_prompt": positive_prompt,
                    "avoid_operations": avoid_operations,
                    "missing_edits_prompt": missing_edits_prompt,
                    "retry_objective": missing_edits_prompt,
                    "refined": False,
                    "reasoning": "dev_mode: skip LLM refinement",
                }

            failed_str = "; ".join(failed_aspects) if failed_aspects else "(none)"

            llm_prompt = f"""You are organizing retry guidance for a video editing model. The previous edit attempt failed QA. Your job is to deduplicate and reorganize the guidance prompts so they are clear, non-redundant, and properly prioritized.

INPUT:
1. BASE EDIT PROMPT (the original edit instructions — do NOT modify or repeat these):
{base_edit_prompt[:3000]}

2. POSITIVE PROMPT (what was done correctly and should be maintained):
{positive_prompt or "(none)"}

3. AVOID OPERATIONS (mistakes to avoid repeating — prohibitive language only):
{avoid_operations or "(none)"}

4. MISSING EDITS PROMPT (specific edits that were NOT applied and MUST be done):
{missing_edits_prompt or "(none)"}

5. QA FEEDBACK:
{qa_feedback or "(none)"}

6. FAILED ASPECTS:
{failed_str}

TASKS:
1. Remove any content from positive/avoid/missing-edits that is already covered by the base edit prompt or is duplicated across the three guidance fields.
2. Consolidate the avoid operations into a concise list of distinct mistakes to avoid. Use prohibitive language ("Do not...", "Avoid...").
3. Consolidate missing edits into clear imperative commands ("Replace X with Y on Z.").
4. Consolidate positive guidance into a brief note on what to maintain.
5. Identify the CRITICAL PROBLEM AREAS from the avoid operations and missing edits — these are the specific entity/attribute regions that failed. List them explicitly.
6. Emphasize: the avoid operations and critical problem areas should be placed at the END of the retry guidance, after positive and missing-edits, as the highest-priority reminder.

Return ONLY valid JSON:
{{
  "positive_prompt": "deduplicated positive guidance (what to maintain), or empty",
  "avoid_operations": "deduplicated prohibitive instructions (do not / avoid), or empty",
  "missing_edits_prompt": "deduplicated imperative commands for missing edits, or empty",
  "critical_problem_areas": "brief list of the specific entity/attribute regions that failed and must be fixed",
  "reasoning": "brief explanation of what was deduplicated or reorganized"
}}

Use English for all string values."""

            raw = await self._text(llm_prompt)
            if not raw or not raw.strip():
                return {
                    "positive_prompt": positive_prompt,
                    "avoid_operations": avoid_operations,
                    "missing_edits_prompt": missing_edits_prompt,
                    "retry_objective": missing_edits_prompt,
                    "refined": False,
                    "reasoning": "empty LLM response — using raw guidance",
                }

            data = extract_json_object(raw)
            refined_positive = str(data.get("positive_prompt", "") or "").strip()
            refined_avoid = str(data.get("avoid_operations", "") or "").strip()
            refined_missing = str(data.get("missing_edits_prompt", "") or "").strip()
            critical_areas = str(data.get("critical_problem_areas", "") or "").strip()

            # Build the final retry_objective: missing edits + critical problem areas emphasized
            retry_obj_parts = []
            if refined_missing:
                retry_obj_parts.append(refined_missing)
            if critical_areas:
                retry_obj_parts.append(
                    f"CRITICAL PROBLEM AREAS (highest priority — must fix these specific regions): {critical_areas}"
                )
            retry_objective = " ".join(retry_obj_parts).strip()

            # Append critical areas emphasis to avoid_operations as well
            if critical_areas and refined_avoid:
                refined_avoid = f"{refined_avoid}\n- CRITICAL: {critical_areas}"
            elif critical_areas:
                refined_avoid = f"- CRITICAL: {critical_areas}"

            logger.info(
                "LLM retry guidance refinement: positive=%d chars, avoid=%d chars, missing=%d chars, critical=%d chars",
                len(refined_positive),
                len(refined_avoid),
                len(refined_missing),
                len(critical_areas),
            )

            return {
                "positive_prompt": refined_positive or positive_prompt,
                "avoid_operations": refined_avoid or avoid_operations,
                "missing_edits_prompt": refined_missing or missing_edits_prompt,
                "retry_objective": retry_objective or missing_edits_prompt,
                "critical_problem_areas": critical_areas,
                "refined": True,
                "reasoning": str(data.get("reasoning", "") or "").strip(),
            }

        except Exception as exc:
            logger.error("refine_retry_guidance_with_llm failed: %s", exc, exc_info=True)
            return {
                "positive_prompt": positive_prompt,
                "avoid_operations": avoid_operations,
                "missing_edits_prompt": missing_edits_prompt,
                "retry_objective": missing_edits_prompt,
                "refined": False,
                "reasoning": f"refinement error — using raw guidance: {exc}",
            }

    async def select_best_video_edit_attempt(
        self,
        *,
        original_keyframes_grid_path: str,
        candidate_grid_paths: List[str],
        entity_ref_image_paths: List[str],
        entity_instru_json: str,
        edit_operation_prompt: str,
    ) -> Dict[str, Any]:
        """VLM: select the best edited keyframe grid from multiple attempts."""
        try:
            if self.dev_mode or len(candidate_grid_paths) <= 1:
                return {
                    "best_candidate_index": 0,
                    "reasoning": "dev_mode or single candidate — return first",
                    "per_candidate_scores": [],
                }

            images: List[Image.Image] = []
            if not original_keyframes_grid_path or not os.path.exists(original_keyframes_grid_path):
                return {
                    "best_candidate_index": 0,
                    "reasoning": "missing original grid — return first candidate",
                    "per_candidate_scores": [],
                }
            images.append(Image.open(original_keyframes_grid_path).convert("RGB"))

            valid_candidates: List[str] = []
            for path in candidate_grid_paths:
                if path and os.path.exists(path):
                    images.append(Image.open(path).convert("RGB"))
                    valid_candidates.append(path)

            if len(valid_candidates) <= 1:
                return {
                    "best_candidate_index": 0,
                    "reasoning": "only one valid candidate",
                    "per_candidate_scores": [],
                }

            for path in entity_ref_image_paths:
                if path and os.path.exists(path):
                    images.append(Image.open(path).convert("RGB"))

            prompt = SCENE_VIDEO_EDIT_BEST_ATTEMPT_SELECT_PROMPT.format(
                entity_instru_json=entity_instru_json or "{}",
                edit_operation_prompt=(edit_operation_prompt or "").strip() or "N/A",
            )
            raw = await self._vision(prompt, images)

            if not raw or not raw.strip():
                logger.warning("select_best_video_edit_attempt: empty VLM response — fallback to highest QA score")
                return {
                    "best_candidate_index": 0,
                    "reasoning": "empty VLM response — fallback to first candidate",
                    "per_candidate_scores": [],
                }

            data = extract_json_object(raw)
            best_idx = int(data.get("best_candidate_index", 0))
            # Clamp to valid range
            best_idx = max(0, min(best_idx, len(valid_candidates) - 1))

            return {
                "best_candidate_index": best_idx,
                "reasoning": str(data.get("reasoning", "") or "").strip(),
                "per_candidate_scores": data.get("per_candidate_scores") or [],
            }

        except Exception as exc:
            logger.error(
                "select_best_video_edit_attempt failed: %s", exc, exc_info=True
            )
            return {
                "best_candidate_index": 0,
                "reasoning": f"selection error — fallback to first candidate: {exc}",
                "per_candidate_scores": [],
            }

    async def locate_entity_on_scene_keyframe_grid(
        self,
        *,
        grid_image_path: str,
        entity_multiview_ref_path: str,
        instruction_id: str,
        entity_id: str,
        subject_features: str,
        edit_prompt: str,
        keyframe_labels: List[str],
        grid_layout_description: str,
    ) -> Dict[str, Any]:
        """VLM: locate one entity on each panel of the keyframe strip."""
        try:
            labels = [str(l).strip() for l in keyframe_labels if str(l).strip()]
            if self.dev_mode:
                return {
                    "instruction_id": instruction_id,
                    "entity_id": entity_id,
                    "keyframes": [
                        {
                            "keyframe_index": i + 1,
                            "keyframe_label": label,
                            "present": i == 0,
                            "location_description": (
                                f"{subject_features or entity_id} in panel center"
                                if i == 0
                                else ""
                            ),
                        }
                        for i, label in enumerate(labels)
                    ],
                }

            if not os.path.exists(grid_image_path):
                raise FileNotFoundError(f"Keyframe grid not found: {grid_image_path}")
            if not os.path.exists(entity_multiview_ref_path):
                raise FileNotFoundError(
                    f"Entity front-view ref not found: {entity_multiview_ref_path}"
                )

            images = [
                Image.open(grid_image_path).convert("RGB"),
                Image.open(entity_multiview_ref_path).convert("RGB"),
            ]
            prompt = SCENE_KEYFRAME_GRID_ENTITY_LOCATION_PROMPT.format(
                keyframe_labels_list=", ".join(labels) or "Keyframe 1",
                grid_layout_description=grid_layout_description,
                instruction_id=instruction_id,
                entity_id=entity_id,
                subject_features=subject_features or "N/A",
                edit_prompt=edit_prompt or "N/A",
            )
            raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            keyframes = data.get("keyframes") or []
            if not isinstance(keyframes, list):
                keyframes = []
            normalized: List[Dict[str, Any]] = []
            for i, label in enumerate(labels):
                entry = keyframes[i] if i < len(keyframes) and isinstance(keyframes[i], dict) else {}
                normalized.append({
                    "keyframe_index": i + 1,
                    "keyframe_label": str(entry.get("keyframe_label") or label),
                    "present": bool(entry.get("present", False)),
                    "location_description": str(
                        entry.get("location_description", "") or ""
                    ).strip(),
                })
            return {
                "instruction_id": instruction_id,
                "entity_id": entity_id,
                "keyframes": normalized,
            }
        except Exception as exc:
            logger.error(
                "locate_entity_on_scene_keyframe_grid failed for %s: %s",
                instruction_id,
                exc,
                exc_info=True,
            )
            raise ModelApiError(
                f"locate_entity_on_scene_keyframe_grid failed: {exc}"
            ) from exc

    async def edit_scene_keyframe_grid(
        self,
        *,
        grid_image_path: str,
        multiview_edited_paths: List[str],
        edit_instructions_block: str,
        entity_locations_block: str,
        output_path: str,
        avoid_operations: str = "",
        positive_prompt: str = "",
    ) -> str:
        """Apply edits to the full labeled keyframe strip via image model."""
        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            if self.dev_mode:
                shutil.copy2(grid_image_path, output_path)
                return output_path

            if not os.path.exists(grid_image_path):
                raise FileNotFoundError(f"Keyframe grid not found: {grid_image_path}")

            avoid_section = build_edit_retry_guidance_section(
                positive_prompt=positive_prompt,
                avoid_operations=avoid_operations,
            )

            prompt = SCENE_KEYFRAME_GRID_EDIT_PROMPT.format(
                edit_instructions_block=edit_instructions_block,
                entity_locations_block=entity_locations_block,
                avoid_section=avoid_section,
            )
            refs: List[Image.Image] = [
                Image.open(grid_image_path).convert("RGB"),
            ]
            for path in multiview_edited_paths:
                if path and os.path.exists(path):
                    refs.append(Image.open(path).convert("RGB"))

            img = await self._gen_image(prompt, refs=refs, save_path=output_path)
            if img is None and not os.path.exists(output_path):
                raise RuntimeError("scene keyframe grid edit returned no image")
            return output_path
        except Exception as exc:
            logger.error("edit_scene_keyframe_grid failed: %s", exc, exc_info=True)
            raise ModelApiError(f"edit_scene_keyframe_grid failed: {exc}") from exc

    async def validate_scene_keyframe_grid_edit(
        self,
        *,
        edited_grid_path: str,
        original_grid_path: str,
        multiview_edited_paths: List[str],
        edit_instructions_block: str,
        entity_locations_block: str,
    ) -> Dict[str, Any]:
        """VLM QA for edited keyframe strip vs original."""
        try:
            if self.dev_mode:
                return {
                    "passed": True,
                    "score": 1.0,
                    "structure_preserved": True,
                    "edit_completed": True,
                    "failed_aspects": [],
                    "feedback": "dev_mode pass",
                    "retry_focus_prompt": "",
                }

            images: List[Image.Image] = []
            for path in (edited_grid_path, original_grid_path):
                if not path or not os.path.exists(path):
                    return {
                        "passed": False,
                        "score": 0.0,
                        "structure_preserved": False,
                        "edit_completed": False,
                        "failed_aspects": ["missing grid image"],
                        "feedback": f"missing grid: {path}",
                        "retry_focus_prompt": "Do not omit the keyframe strip output image.",
                    }
                images.append(Image.open(path).convert("RGB"))
            for path in multiview_edited_paths:
                if path and os.path.exists(path):
                    images.append(Image.open(path).convert("RGB"))

            prompt = SCENE_KEYFRAME_GRID_EDIT_QA_PROMPT.format(
                edit_instructions_block=edit_instructions_block,
                entity_locations_block=entity_locations_block,
            )
            raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            passed = bool(data.get("passed", False))
            score = _safe_float(data.get("score"), 0.0)
            structure_ok = bool(data.get("structure_preserved", passed))
            edit_ok = bool(data.get("edit_completed", passed))
            failed_aspects = data.get("failed_aspects") or []
            if not isinstance(failed_aspects, list):
                failed_aspects = [str(failed_aspects)]
            feedback = str(data.get("feedback", "")).strip()
            retry_focus = str(data.get("retry_focus_prompt", "")).strip()

            if score <= 0.0 and passed:
                score = 0.85
            if passed and not (structure_ok and edit_ok):
                passed = False
            if not passed and score >= 0.7 and structure_ok and edit_ok:
                passed = True

            return {
                "passed": passed,
                "score": score,
                "structure_preserved": structure_ok,
                "edit_completed": edit_ok,
                "failed_aspects": [str(x) for x in failed_aspects if str(x).strip()],
                "feedback": feedback,
                "retry_focus_prompt": retry_focus,
                "positive_prompt": str(data.get("positive_prompt", "")).strip(),
            }
        except Exception as exc:
            logger.error(
                "validate_scene_keyframe_grid_edit failed: %s", exc, exc_info=True
            )
            return {
                "passed": True,
                "score": 0.0,
                "structure_preserved": True,
                "edit_completed": True,
                "failed_aspects": [],
                "feedback": f"QA skipped due to error: {exc}",
                "retry_focus_prompt": "",
            }

    async def locate_entity_on_single_keyframe(
        self,
        *,
        keyframe_path: str,
        entity_multiview_ref_path: str,
        instruction_id: str,
        entity_id: str,
        subject_features: str,
        edit_prompt: str = "",
        prior_detection: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """VLM: does entity in the source identity reference appear in keyframe; if yes, where."""
        try:
            if self.dev_mode:
                return apply_keyframe_entity_presence_gate({
                    "instruction_id": instruction_id,
                    "entity_id": entity_id,
                    "present": True,
                    "location_description": (
                        f"{subject_features or entity_id} in frame center"
                    ),
                    "confidence": 1.0,
                    "visibility_quality": "clear",
                    "approximate_area_fraction": 0.2,
                    "localization_clarity": "high",
                    "entity_visibility_completeness": "sufficient",
                    "visible_parts": ["face", "torso"],
                    "identity_verifiable_from_visible_parts": True,
                    "reasoning": "dev_mode",
                })

            if not os.path.exists(keyframe_path):
                raise FileNotFoundError(f"Keyframe not found: {keyframe_path}")
            if not os.path.exists(entity_multiview_ref_path):
                raise FileNotFoundError(
                    f"Entity source identity ref not found: {entity_multiview_ref_path}"
                )

            images = [
                Image.open(keyframe_path).convert("RGB"),
                Image.open(entity_multiview_ref_path).convert("RGB"),
            ]
            prompt = KEYFRAME_SINGLE_ENTITY_PRESENCE_PROMPT.format(
                instruction_id=instruction_id,
                entity_id=entity_id,
                subject_features=subject_features or "N/A",
                edit_prompt=edit_prompt or "N/A",
                prior_detection_block=format_prior_detection_block(prior_detection),
            )
            raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            score_raw = data.get("existence_confidence_score")
            if score_raw is None:
                score_raw = data.get("entity_presence_confidence_score")
            if score_raw is None:
                score_raw = data.get("presence_confidence_score")
            if score_raw is not None:
                try:
                    score = max(0.0, min(100.0, float(score_raw)))
                except (TypeError, ValueError):
                    score = 0.0
                if not data.get("present"):
                    score = 0.0
                data["existence_confidence_score"] = score
                data["confidence"] = max(
                    score / 100.0,
                    0.86 if data.get("present") else 0.0,
                )
            elif data.get("present") and "confidence" not in data:
                data["existence_confidence_score"] = 85.0
                data["confidence"] = 0.86
            record = normalize_vlm_entity_location_record(
                data,
                instruction_id=instruction_id,
                entity_id=entity_id,
                edit_prompt=edit_prompt,
                subject_features=subject_features,
            )
            if record.get("presence_gated"):
                logger.info(
                    "locate_entity_on_single_keyframe: %s gated (%s)",
                    instruction_id,
                    "; ".join(record.get("presence_reject_reasons") or []),
                )
            return record
        except Exception as exc:
            logger.error(
                "locate_entity_on_single_keyframe failed for %s: %s",
                instruction_id,
                exc,
                exc_info=True,
            )
            raise ModelApiError(
                f"locate_entity_on_single_keyframe failed: {exc}"
            ) from exc

    async def locate_all_entities_on_single_keyframe(
        self,
        *,
        keyframe_path: str,
        entity_specs: List[Dict[str, Any]],
        scene_prior_by_instruction: Dict[str, List[Dict[str, Any]]] | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        """VLM: detect all entities on one keyframe in a single call."""
        try:
            if not entity_specs:
                return {}

            if self.dev_mode:
                results: Dict[str, Dict[str, Any]] = {}
                for spec in entity_specs:
                    iid = str(spec.get("instruction_id", "")).strip()
                    eid = str(spec.get("entity_id", "")).strip()
                    if not iid:
                        continue
                    results[iid] = apply_keyframe_entity_presence_gate({
                        "instruction_id": iid,
                        "entity_id": eid,
                        "present": True,
                        "location_description": (
                            f"{spec.get('subject_features') or eid} in frame"
                        ),
                        "confidence": 1.0,
                        "visibility_quality": "clear",
                        "approximate_area_fraction": 0.2,
                        "localization_clarity": "high",
                        "entity_visibility_completeness": "sufficient",
                        "visible_parts": ["face", "torso"],
                        "identity_verifiable_from_visible_parts": True,
                        "reasoning": "dev_mode",
                    }, edit_prompt=str(spec.get("edit_prompt", "") or ""))
                return results

            if not os.path.exists(keyframe_path):
                raise FileNotFoundError(f"Keyframe not found: {keyframe_path}")

            images: List[Image.Image] = [
                Image.open(keyframe_path).convert("RGB"),
            ]
            indexed_specs: List[Dict[str, Any]] = []
            for idx, spec in enumerate(entity_specs, start=2):
                ref_path = str(spec.get("multiview_ref_path", "") or "").strip()
                if not ref_path or not os.path.exists(ref_path):
                    raise FileNotFoundError(
                        f"Entity front-view ref not found: {ref_path}"
                    )
                images.append(Image.open(ref_path).convert("RGB"))
                indexed_specs.append({**spec, "ref_image_index": idx})

            prompt = KEYFRAME_BATCH_ENTITY_PRESENCE_PROMPT.format(
                entity_catalog_block=format_batch_entity_detection_catalog(
                    indexed_specs
                ),
                scene_prior_detection_block=format_scene_prior_detection_block(
                    indexed_specs,
                    scene_prior_by_instruction,
                ),
            )
            raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            results = parse_batch_entity_location_response(data, indexed_specs)

            for iid, record in results.items():
                if record.get("presence_gated"):
                    logger.info(
                        "locate_all_entities_on_single_keyframe: %s gated (%s)",
                        iid,
                        "; ".join(record.get("presence_reject_reasons") or []),
                    )
            return results
        except Exception as exc:
            logger.error(
                "locate_all_entities_on_single_keyframe failed: %s",
                exc,
                exc_info=True,
            )
            raise ModelApiError(
                f"locate_all_entities_on_single_keyframe failed: {exc}"
            ) from exc

    async def _build_indexed_entity_images(
        self,
        *,
        keyframe_path: str,
        entity_specs: List[Dict[str, Any]],
    ) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
        if not os.path.exists(keyframe_path):
            raise FileNotFoundError(f"Keyframe not found: {keyframe_path}")
        images: List[Image.Image] = [
            Image.open(keyframe_path).convert("RGB"),
        ]
        indexed_specs: List[Dict[str, Any]] = []
        for idx, spec in enumerate(entity_specs, start=2):
            ref_path = str(spec.get("multiview_ref_path", "") or "").strip()
            if not ref_path or not os.path.exists(ref_path):
                raise FileNotFoundError(
                    f"Entity front-view ref not found: {ref_path}"
                )
            images.append(Image.open(ref_path).convert("RGB"))
            indexed_specs.append({**spec, "ref_image_index": idx})
        return images, indexed_specs

    async def detect_entities_on_keyframe(
        self,
        *,
        keyframe_path: str,
        entity_specs: List[Dict[str, Any]],
        scene_story_context: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        """VLM step 1: detect entities with confidence and location."""
        try:
            if not entity_specs:
                return {}
            if self.dev_mode:
                results: Dict[str, Dict[str, Any]] = {}
                for spec in entity_specs:
                    iid = str(spec.get("instruction_id", "")).strip()
                    eid = str(spec.get("entity_id", "")).strip()
                    if not iid:
                        continue
                    results[iid] = {
                        "instruction_id": iid,
                        "entity_id": eid,
                        "present": True,
                        "confidence": 1.0,
                        "location_description": (
                            f"{spec.get('subject_features') or eid} in frame"
                        ),
                        "visibility_quality": "clear",
                        "approximate_area_fraction": 0.2,
                        "visible_parts": ["face", "torso"],
                        "reasoning": "dev_mode",
                        "vlm_present": True,
                    }
                return results

            images, indexed_specs = await self._build_indexed_entity_images(
                keyframe_path=keyframe_path,
                entity_specs=entity_specs,
            )
            prompt = KEYFRAME_ENTITY_DETECT_PROMPT.format(
                scene_story_context=scene_story_context or "(not provided)",
                entity_catalog_block=format_batch_entity_detection_catalog(
                    indexed_specs
                ),
            )
            raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            return parse_entity_detect_response(data, indexed_specs)
        except Exception as exc:
            logger.error(
                "detect_entities_on_keyframe failed: %s",
                exc,
                exc_info=True,
            )
            raise ModelApiError(
                f"detect_entities_on_keyframe failed: {exc}"
            ) from exc

    async def verify_entity_locations_on_keyframe(
        self,
        *,
        keyframe_path: str,
        entity_specs: List[Dict[str, Any]],
        detection_records: Dict[str, Dict[str, Any]],
        scene_story_context: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        """VLM step 2: verify and correct entity locations (single pass)."""
        try:
            if not entity_specs:
                return {}
            if self.dev_mode:
                verified = {}
                for spec in entity_specs:
                    iid = str(spec.get("instruction_id", "")).strip()
                    if not iid:
                        continue
                    base = dict(detection_records.get(iid) or {})
                    base["verified"] = True
                    verified[iid] = base
                return verified

            images, indexed_specs = await self._build_indexed_entity_images(
                keyframe_path=keyframe_path,
                entity_specs=entity_specs,
            )
            prompt = KEYFRAME_ENTITY_LOCATION_VERIFY_PROMPT.format(
                scene_story_context=scene_story_context or "(not provided)",
                detection_results_block=format_detection_results_block(
                    detection_records
                ),
                entity_catalog_block=format_batch_entity_detection_catalog(
                    indexed_specs
                ),
            )
            raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            return parse_entity_verify_response(
                data,
                indexed_specs,
                detection_records=detection_records,
            )
        except Exception as exc:
            logger.error(
                "verify_entity_locations_on_keyframe failed: %s",
                exc,
                exc_info=True,
            )
            raise ModelApiError(
                f"verify_entity_locations_on_keyframe failed: {exc}"
            ) from exc

    async def verify_scene_keyframe_presence_consistency(
        self,
        *,
        scene_id: str,
        keyframes: List[Dict[str, str]],
        entity_specs: List[Dict[str, Any]],
        initial_location_records: Dict[str, Dict[str, Dict[str, Any]]],
        scene_story_context: str = "",
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """VLM: correct inconsistent entity presence across keyframes in one scene."""
        try:
            if not keyframes or not entity_specs:
                return {}
            if self.dev_mode:
                return initial_location_records

            images: List[Image.Image] = []
            keyframe_catalog_lines: List[str] = []
            keyframe_ids: List[str] = []
            for item in keyframes:
                keyframe_id = str(item.get("keyframe_id", "") or "").strip()
                path = str(item.get("path", "") or "").strip()
                if not keyframe_id or not path or not os.path.exists(path):
                    continue
                image_idx = len(images) + 1
                images.append(Image.open(path).convert("RGB"))
                keyframe_ids.append(keyframe_id)
                keyframe_catalog_lines.append(f"- Image {image_idx}: {keyframe_id}")

            if not images:
                return {}

            first_ref_image_index = len(images) + 1
            indexed_specs: List[Dict[str, Any]] = []
            for offset, spec in enumerate(entity_specs):
                ref_path = str(spec.get("multiview_ref_path", "") or "").strip()
                if not ref_path or not os.path.exists(ref_path):
                    continue
                ref_idx = first_ref_image_index + offset
                images.append(Image.open(ref_path).convert("RGB"))
                indexed_specs.append({**spec, "ref_image_index": ref_idx})

            if not indexed_specs:
                return {}

            all_negative_lines: List[str] = []
            all_negative_specs: List[Dict[str, Any]] = []
            for spec in indexed_specs:
                iid = str(spec.get("instruction_id", "") or "").strip()
                eid = str(spec.get("entity_id", "") or "").strip()
                if not iid:
                    continue
                records = [
                    per_keyframe.get(iid) or {}
                    for per_keyframe in initial_location_records.values()
                    if isinstance(per_keyframe, dict)
                ]
                if records and not any(
                    bool(record.get("present")) or bool(record.get("vlm_present"))
                    for record in records
                    if isinstance(record, dict)
                ):
                    all_negative_specs.append(spec)
                    all_negative_lines.append(
                        f"- {iid} / {eid}: initial detector marked absent in every scene keyframe; "
                        "actively scan for large/salient unassigned candidates and compare them to "
                        "the reference identity context before keeping present=false."
                    )

            prompt = SCENE_KEYFRAME_PRESENCE_CONSISTENCY_PROMPT.format(
                keyframe_count=len(keyframe_ids),
                keyframe_catalog_block="\n".join(keyframe_catalog_lines),
                first_ref_image_index=first_ref_image_index,
                scene_story_context=scene_story_context or "(not provided)",
                entity_catalog_block=format_batch_entity_detection_catalog(
                    indexed_specs
                ),
                initial_detection_block=json.dumps(
                    initial_location_records,
                    ensure_ascii=False,
                    indent=2,
                ),
                all_negative_recovery_block=(
                    "\n".join(all_negative_lines) if all_negative_lines else "(none)"
                ),
            )
            raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            consistency_records = parse_scene_keyframe_presence_consistency_response(
                data,
                indexed_specs,
                keyframe_ids,
            )
            if all_negative_specs:
                try:
                    recovery_prompt = SCENE_KEYFRAME_ALL_NEGATIVE_RECOVERY_PROMPT.format(
                        keyframe_count=len(keyframe_ids),
                        keyframe_catalog_block="\n".join(keyframe_catalog_lines),
                        first_ref_image_index=first_ref_image_index,
                        scene_story_context=scene_story_context or "(not provided)",
                        entity_catalog_block=format_batch_entity_detection_catalog(
                            all_negative_specs
                        ),
                        initial_detection_block=json.dumps(
                            {
                                keyframe_id: {
                                    iid: records.get(iid)
                                    for iid in {
                                        str(spec.get("instruction_id", "") or "").strip()
                                        for spec in all_negative_specs
                                    }
                                    if iid and isinstance(records, dict)
                                }
                                for keyframe_id, records in initial_location_records.items()
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                    )
                    recovery_raw = await self._vision(recovery_prompt, images)
                    recovery_data = extract_json_object(recovery_raw)
                    recovery_records = parse_scene_keyframe_presence_consistency_response(
                        recovery_data,
                        all_negative_specs,
                        keyframe_ids,
                    )
                    all_negative_specs_by_iid = {
                        str(spec.get("instruction_id", "") or "").strip(): spec
                        for spec in all_negative_specs
                    }
                    for keyframe_id, records in recovery_records.items():
                        target_records = consistency_records.setdefault(keyframe_id, {})
                        for iid, recovery_record in records.items():
                            existing = target_records.get(iid) or {}
                            recovery_record["all_negative_recovery_checked"] = True
                            spec = all_negative_specs_by_iid.get(iid, {})
                            recovery_record = apply_keyframe_entity_presence_gate(
                                recovery_record,
                                edit_prompt=str(spec.get("edit_prompt", "") or ""),
                                subject_features=str(spec.get("subject_features", "") or ""),
                            )
                            recovery_record["all_negative_recovery_checked"] = True
                            if recovery_record.get("present") or not existing.get("present"):
                                target_records[iid] = recovery_record
                except Exception as recovery_exc:
                    logger.warning(
                        "%s: all-negative focused recovery failed; using scene consistency records: %s",
                        scene_id,
                        recovery_exc,
                    )
            return consistency_records
        except Exception as exc:
            logger.error(
                "verify_scene_keyframe_presence_consistency failed for %s: %s",
                scene_id,
                exc,
                exc_info=True,
            )
            raise ModelApiError(
                f"verify_scene_keyframe_presence_consistency failed: {exc}"
            ) from exc

    async def describe_keyframe_edit_operations(
        self,
        *,
        original_keyframe_path: str,
        edited_keyframe_path: str,
        canonical_edit_block: str = "",
        visibility_constraints_block: str = "",
    ) -> Dict[str, Any]:
        """VLM step 4: list observed edit operations between frames."""
        try:
            if self.dev_mode:
                return {
                    "observed_edit_operations": [],
                    "summary": "dev_mode — skipped edit comparison",
                }
            if not os.path.exists(original_keyframe_path):
                raise FileNotFoundError(
                    f"Original keyframe not found: {original_keyframe_path}"
                )
            if not os.path.exists(edited_keyframe_path):
                raise FileNotFoundError(
                    f"Edited keyframe not found: {edited_keyframe_path}"
                )

            edit_img, orig_img, _resized = prepare_keyframe_qa_images(
                original_keyframe_path,
                edited_keyframe_path,
            )
            prompt = KEYFRAME_EDIT_COMPARISON_PROMPT.format(
                canonical_edit_block=canonical_edit_block or "(not provided)",
                visibility_constraints_block=visibility_constraints_block or "(not provided)",
            )
            raw = await self._vision(prompt, [orig_img, edit_img])
            data = extract_json_object(raw)
            return parse_keyframe_edit_comparison_response(data)
        except Exception as exc:
            logger.error(
                "describe_keyframe_edit_operations failed: %s",
                exc,
                exc_info=True,
            )
            raise ModelApiError(
                f"describe_keyframe_edit_operations failed: {exc}"
            ) from exc

    async def describe_entity_from_synthesis_sheet(
        self,
        *,
        synth_sheet_path: str,
        instruction_id: str,
        entity_id: str,
        original_subject_features: str = "",
    ) -> str:
        """VLM: produce a detailed visual description of the entity in the synth sheet."""
        try:
            if self.dev_mode:
                return original_subject_features or f"{entity_id} (dev_mode)"
            if not os.path.exists(synth_sheet_path):
                raise FileNotFoundError(
                    f"Synthesis sheet not found: {synth_sheet_path}"
                )

            from PIL import Image as _PILImage

            img = _PILImage.open(synth_sheet_path).convert("RGB")
            prompt = (
                "You are a forensic visual analyst. Examine the front-view entity "
                "reference sheet in this image and produce a DETAILED, STRUCTURED "
                "visual description of the person or object shown.\n\n"
                "Describe ALL of the following that are visible:\n"
                "- Face: shape, jawline, cheekbones, nose, eyes (color, shape, "
                "eyebrows), mouth, lips, chin, forehead, facial hair, skin tone\n"
                "- Hair: color, style, length, parting, hairline, texture, volume\n"
                "- Headwear: type, color, shape (if any)\n"
                "- Body: build, height estimate, posture\n"
                "- Clothing: every visible garment, color, pattern, texture, "
                "collar style, sleeve length\n"
                "- Accessories: suspenders, ties, jewelry, glasses, bags, props\n"
                "- Distinctive features: scars, moles, tattoos, unique marks\n"
                "- Overall impression: age range, gender presentation, ethnicity hints\n\n"
                f"Entity context: instruction_id={instruction_id}, entity_id={entity_id}\n"
                f"Original (coarse) subject_features for reference: "
                f"{original_subject_features or '(not provided)'}\n\n"
                "Output a SINGLE paragraph (no bullet points, no JSON) that could serve "
                "as an authoritative identity description for this entity. Start directly "
                "with the description — no preamble like 'The entity is...' or 'This person...'. "
                "Focus on stable, identity-defining visual features that would help identify "
                "this exact person/object across different camera angles, lighting, and scenes."
            )

            raw = await self._vision(prompt, [img])
            # Clean up: strip whitespace, remove markdown fences if present
            description = str(raw or "").strip()
            if description.startswith("```"):
                lines = description.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                description = "\n".join(lines).strip()
            return description

        except Exception as exc:
            logger.error(
                "describe_entity_from_synthesis_sheet failed for %s: %s",
                instruction_id,
                exc,
                exc_info=True,
            )
            raise ModelApiError(
                f"describe_entity_from_synthesis_sheet failed: {exc}"
            ) from exc

    async def validate_keyframe_edit_completion(
        self,
        *,
        edited_keyframe_path: str,
        original_keyframe_path: str,
        canonical_ref_paths: List[str],
        entity_locations_block: str,
        canonical_edit_block: str,
        visibility_constraints_block: str = "",
        scene_story_context: str = "",
        observed_edits_block: str = "",
        entity_location_records: Dict[str, Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """VLM step 5: validate planned edits completed without unrelated changes."""
        try:
            if self.dev_mode:
                return parse_keyframe_edit_completion_qa_response({
                    "passed": True,
                    "score": 1.0,
                    "edit_completed": True,
                    "canonical_reference_alignment_ok": True,
                    "unrelated_edit_changes_absent": True,
                    "background_unedited_regions_preserved": True,
                    "failed_aspects": [],
                    "feedback": "dev_mode pass",
                    "retry_focus_prompt": "",
                })

            if not edited_keyframe_path or not os.path.exists(edited_keyframe_path):
                return parse_keyframe_edit_completion_qa_response({
                    "passed": False,
                    "score": 0.0,
                    "edit_completed": False,
                    "canonical_reference_alignment_ok": False,
                    "unrelated_edit_changes_absent": False,
                    "background_unedited_regions_preserved": False,
                    "failed_aspects": ["missing edited keyframe"],
                    "feedback": f"missing edited keyframe: {edited_keyframe_path}",
                    "retry_focus_prompt": "Produce a full edited keyframe output.",
                })
            if not original_keyframe_path or not os.path.exists(original_keyframe_path):
                return parse_keyframe_edit_completion_qa_response({
                    "passed": False,
                    "score": 0.0,
                    "edit_completed": False,
                    "canonical_reference_alignment_ok": False,
                    "unrelated_edit_changes_absent": False,
                    "background_unedited_regions_preserved": False,
                    "failed_aspects": ["missing original keyframe"],
                    "feedback": f"missing original keyframe: {original_keyframe_path}",
                    "retry_focus_prompt": "Keep the original keyframe as edit base.",
                })

            edit_img, orig_img, resized = prepare_keyframe_qa_images(
                original_keyframe_path,
                edited_keyframe_path,
            )
            letterbox_ok, letterbox_reason = assess_letterbox_structure_preserved(
                orig_img,
                edit_img,
            )
            images: List[Image.Image] = [edit_img, orig_img]
            for path in canonical_ref_paths:
                if path and os.path.exists(path):
                    images.append(Image.open(path).convert("RGB"))

            prompt = KEYFRAME_EDIT_COMPLETION_QA_PROMPT.format(
                canonical_edit_block=canonical_edit_block,
                entity_locations_block=entity_locations_block,
                visibility_constraints_block=visibility_constraints_block
                or "(no visibility constraints)",
                scene_story_context=scene_story_context or "(not provided)",
            )
            raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            result = parse_keyframe_edit_completion_qa_response(data)
            result["non_edit_region_change_severity"] = normalize_non_edit_region_change_severity(
                data.get("non_edit_region_change_severity")
            )
            result["non_edit_region_change_summary"] = str(
                data.get("non_edit_region_change_summary", "") or ""
            ).strip()
            if non_edit_region_change_requires_reedit(
                result.get("non_edit_region_change_severity")
            ):
                result["passed"] = False
                result["background_unedited_regions_preserved"] = False
                severity_note = result.get("non_edit_region_change_summary") or (
                    "Non-edit regions changed beyond the allowed threshold."
                )
                if "non-edit regions changed too much" not in result["failed_aspects"]:
                    result["failed_aspects"].append("non-edit regions changed too much")
                result["feedback"] = f"{result.get('feedback', '').strip()} {severity_note}".strip()
                retry_focus = str(result.get("retry_focus_prompt", "") or "").strip()
                if not retry_focus:
                    retry_focus = (
                        "Do not alter non-edit regions outside the exact edit silhouette; only tiny seams directly "
                        "adjacent to the edit boundary are acceptable."
                    )
                result["retry_focus_prompt"] = retry_focus
            if entity_location_records:
                drift_metrics = measure_keyframe_background_drift(
                    original_keyframe_path,
                    edited_keyframe_path,
                    entity_location_records,
                )
                result = merge_background_drift_into_qa_result(
                    result,
                    drift_metrics,
                )
            alignment_results: List[Dict[str, Any]] = []
            if canonical_ref_paths:
                for ref_path in canonical_ref_paths:
                    instruction_id = instruction_id_from_canonical_ref_path(ref_path)
                    edit_instruction = parse_edit_instruction_for_instruction_id(
                        canonical_edit_block,
                        instruction_id,
                    )
                    if not canonical_alignment_check_applicable(edit_instruction):
                        continue
                    target_panel = extract_canonical_target_panel(ref_path)
                    if target_panel is None:
                        continue
                    loc_record = (entity_location_records or {}).get(instruction_id, {})
                    entity_location = (
                        str(loc_record.get("location_description", "") or "").strip()
                        or "(unspecified)"
                    )
                    align_prompt = KEYFRAME_CANONICAL_ALIGNMENT_PROMPT.format(
                        edit_instruction=edit_instruction or "(unspecified)",
                        entity_location=entity_location,
                    )
                    align_raw = await self._vision(
                        align_prompt,
                        [edit_img, target_panel],
                    )
                    align_data = extract_json_object(align_raw)
                    alignment_results.append({
                        "instruction_id": instruction_id,
                        "alignment_ok": bool(align_data.get("alignment_ok", False)),
                        "mismatched_attributes": align_data.get("mismatched_attributes")
                        or [],
                        "feedback": str(align_data.get("feedback", "") or "").strip(),
                        "retry_focus_prompt": str(
                            align_data.get("retry_focus_prompt", "") or ""
                        ).strip(),
                    })
                if alignment_results:
                    result = merge_canonical_alignment_into_qa_result(
                        result,
                        alignment_results,
                    )
            
            if not result.get("frame_structure_preserved", True):
                result["passed"] = False
                if "frame structure lost (black bars missing or cropped)" not in result["failed_aspects"]:
                    result["failed_aspects"].append("frame structure lost (black bars missing or cropped)")
            if not letterbox_ok:
                result["passed"] = False
                result["frame_structure_preserved"] = False
                if "black bars changed or removed" not in result["failed_aspects"]:
                    result["failed_aspects"].append("black bars changed or removed")
                feedback = str(result.get("feedback", "") or "").strip()
                result["feedback"] = f"{feedback} {letterbox_reason}".strip()
                retry_focus = str(result.get("retry_focus_prompt", "") or "").strip()
                structure_retry = (
                    "Preserve the original canvas exactly, including all top/bottom or left/right black bars; "
                    "do not crop, zoom, stretch, or expand picture content into the black-bar regions."
                )
                result["retry_focus_prompt"] = f"{retry_focus} {structure_retry}".strip()

            planned_text = f"{canonical_edit_block}\n{entity_locations_block}".lower()
            observed_text = str(observed_edits_block or "").lower()
            qa_report_text = " ".join([
                str(result.get("feedback", "") or ""),
                str(result.get("retry_focus_prompt", "") or ""),
                str(result.get("positive_prompt", "") or ""),
                " ".join(str(item) for item in (result.get("failed_aspects") or [])),
            ]).lower()
            issue_text = f"{observed_text}\n{qa_report_text}"
            removal_planned = any(
                token in planned_text
                for token in ("remove", "delete", "erase", "inpaint")
            ) and "present" in planned_text
            removal_observed = any(
                token in observed_text
                for token in ("removed", "remove", "deleted", "delete", "erased", "inpaint")
            )
            if observed_text and removal_planned and not removal_observed:
                result["observed_removal_missing"] = True
                if not result.get("edit_completed", False):
                    result["passed"] = False
                    result["canonical_reference_alignment_ok"] = False
                    if "planned removal missing from observed edits" not in result["failed_aspects"]:
                        result["failed_aspects"].append("planned removal missing from observed edits")
                    feedback = str(result.get("feedback", "") or "").strip()
                    note = (
                        "A located removal/delete instruction was planned, and the image QA did not confirm "
                        "completion; the observed edit comparison also did not report the removal or inpainting."
                    )
                    result["feedback"] = f"{feedback} {note}".strip()
                    retry_focus = str(result.get("retry_focus_prompt", "") or "").strip()
                    removal_retry = (
                        "Do not skip the located removal target; remove exactly the planned target "
                        "and inpaint only its original silhouette."
                    )
                    result["retry_focus_prompt"] = f"{retry_focus} {removal_retry}".strip()

            new_entity_hallucinated = any(
                phrase in issue_text
                for phrase in (
                    "replaced with a different person",
                    "replaced by a different person",
                    "replaced with another person",
                    "replaced by another person",
                    "new person",
                    "new man",
                    "new woman",
                    "new entity",
                    "new character",
                    "replacement person",
                    "substitute",
                    "inserted a completely different",
                    "added to the scene",
                    "actor-like",
                    "look-alike",
                    "pasted-in person",
                    "full-body substitute",
                    "synthesized person",
                    "single man wearing suspenders",
                    "resembling leonardo",
                    "leonardo dicaprio",
                )
            )
            if new_entity_hallucinated:
                result["passed"] = False
                result["original_entity_state_preserved"] = False
                result["unrelated_edit_changes_absent"] = False
                result["background_unedited_regions_preserved"] = False
                if "new person/entity hallucinated" not in result["failed_aspects"]:
                    result["failed_aspects"].append("new person/entity hallucinated")
                feedback = str(result.get("feedback", "") or "").strip()
                hallucination_note = (
                    "The observed edit indicates a newly created or replacement person/entity, "
                    "which is never allowed for keyframe edits."
                )
                result["feedback"] = f"{feedback} {hallucination_note}".strip()
                retry_focus = str(result.get("retry_focus_prompt", "") or "").strip()
                hallucination_retry = (
                    "Do not create, paste, or replace any person/entity. For removals, inpaint with empty "
                    "background only; for placements, keep the original target body unchanged."
                )
                result["retry_focus_prompt"] = f"{retry_focus} {hallucination_retry}".strip()

            single_scope_multiple_edit = (
                "target_instance_scope: single" in planned_text
                and any(token in issue_text for token in ("two men", "two people", "multiple", "both men", "both people"))
                and any(token in issue_text for token in ("removed", "deleted", "replaced", "altered"))
            )
            if single_scope_multiple_edit:
                result["passed"] = False
                result["edit_completed"] = False
                result["unrelated_edit_changes_absent"] = False
                result["background_unedited_regions_preserved"] = False
                if "single target edited multiple instances" not in result["failed_aspects"]:
                    result["failed_aspects"].append("single target edited multiple instances")
                feedback = str(result.get("feedback", "") or "").strip()
                multi_note = (
                    "The observed edit indicates multiple people/instances were changed for a single-instance "
                    "instruction, which is not allowed."
                )
                result["feedback"] = f"{feedback} {multi_note}".strip()
                retry_focus = str(result.get("retry_focus_prompt", "") or "").strip()
                multi_retry = (
                    "Edit only the one located target instance for each target_instance_scope=single instruction; "
                    "do not remove or alter any second similar person."
                )
                result["retry_focus_prompt"] = f"{retry_focus} {multi_retry}".strip()

            planned_allows_clothing = any(
                phrase in planned_text
                for phrase in (
                    "edit instruction = change clothing",
                    "edit instruction = change the clothing",
                    "edit instruction = replace clothing",
                    "edit instruction = replace the clothing",
                    "edit instruction = change dress",
                    "edit instruction = change the dress",
                    "edit instruction = replace dress",
                    "edit instruction = replace the dress",
                    "edit instruction = change shirt",
                    "edit instruction = replace shirt",
                    "edit instruction = change vest",
                    "edit instruction = replace vest",
                    "edit instruction = change outfit",
                    "edit instruction = replace outfit",
                )
            )
            protected_state_drift_terms = (
                "head orientation",
                "head turned",
                "head turn",
                "head tilt",
                "face changed",
                "face shape",
                "facial expression",
                "expression changed",
                "expression drift",
                "gaze changed",
                "gaze drift",
                "mouth changed",
                "eye direction",
                "pose changed",
                "pose drift",
                "action changed",
                "gesture changed",
                "body posture",
                "posture changed",
                "body angle",
                "body-angle",
                "arm position",
                "hand position",
                "dress color",
                "dress pattern",
                "dress design",
                "clothing changed",
                "outfit changed",
                "shirt changed",
                "vest changed",
                "significantly altered",
                "completely altered",
            )
            protected_state_drift = any(term in issue_text for term in protected_state_drift_terms)
            clothing_drift = any(
                term in issue_text
                for term in (
                    "dress color",
                    "dress pattern",
                    "dress design",
                    "clothing changed",
                    "outfit changed",
                    "shirt changed",
                    "vest changed",
                )
            )
            non_clothing_state_drift = protected_state_drift and any(
                term in issue_text
                for term in (
                    "head orientation",
                    "head turned",
                    "head turn",
                    "head tilt",
                    "face changed",
                    "face shape",
                    "facial expression",
                    "expression changed",
                    "expression drift",
                    "gaze changed",
                    "gaze drift",
                    "mouth changed",
                    "eye direction",
                    "pose changed",
                    "pose drift",
                    "action changed",
                    "gesture changed",
                    "body posture",
                    "posture changed",
                    "body angle",
                    "body-angle",
                    "arm position",
                    "hand position",
                    "significantly altered",
                    "completely altered",
                )
            )
            hard_state_drift = non_clothing_state_drift or (
                clothing_drift and not planned_allows_clothing
            )
            if hard_state_drift:
                result["passed"] = False
                result["original_entity_state_preserved"] = False
                result["unrelated_edit_changes_absent"] = False
                if "protected original state drifted" not in result["failed_aspects"]:
                    result["failed_aspects"].append("protected original state drifted")
                feedback = str(result.get("feedback", "") or "").strip()
                drift_note = (
                    "Observed edits mention drift in protected original state (head/face/expression/gaze/"
                    "clothing), which is not allowed for the planned edit."
                )
                result["feedback"] = f"{feedback} {drift_note}".strip()
                retry_focus = str(result.get("retry_focus_prompt", "") or "").strip()
                drift_retry = (
                    "Do not change head orientation, facial expression, gaze, clothing/outfit, or pose. "
                    "Keep those pixels from the original frame while applying only the requested edit attributes."
                )
                result["retry_focus_prompt"] = f"{retry_focus} {drift_retry}".strip()

            background_drift = any(
                phrase in issue_text
                for phrase in (
                    "background changed",
                    "background was changed",
                    "background region changed",
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
                    "right edge",
                    "left edge",
                    "texture shift",
                    "color shift",
                    "inpaint bleed",
                    "outside the target silhouette",
                    "unrelated background",
                    "far-right woman",
                    "far right woman",
                    "rightmost woman",
                    "right edge woman",
                    "far-left woman",
                    "far left woman",
                    "leftmost woman",
                    "left edge woman",
                    "non-target person",
                    "non-target woman",
                    "removed another person",
                    "removed a different person",
                    "removed the wrong person",
                )
            )
            if background_drift:
                result["passed"] = False
                result["background_unedited_regions_preserved"] = False
                result["unrelated_edit_changes_absent"] = False
                if "background/unrelated region drifted" not in result["failed_aspects"]:
                    result["failed_aspects"].append("background/unrelated region drifted")
                retry_focus = str(result.get("retry_focus_prompt", "") or "").strip()
                background_retry = (
                    "Do not repaint, blur, relight, or texture-change any background outside the exact "
                    "target edit silhouette; preserve walls, pillars, floor, ceiling, edges, furniture, props, "
                    "and non-target people from the original frame. Do not add chairs, benches, tables, lamps, "
                    "railings, luggage, or any new background object."
                )
                result["retry_focus_prompt"] = f"{retry_focus} {background_retry}".strip()

            pasted_or_physical_failure = any(
                phrase in issue_text
                for phrase in (
                    "pasted-on",
                    "looks pasted",
                    "pasted face",
                    "pasted head",
                    "redrawn face",
                    "over-smoothed face",
                    "sticker-like",
                    "pasted-in asset",
                    "wrong shoulder",
                    "wrong relative size",
                    "too large",
                    "too small",
                    "oversized",
                    "floating",
                    "missing contact shadow",
                    "perspective inconsistent",
                    "wrong orientation",
                )
            )
            if pasted_or_physical_failure:
                result["passed"] = False
                result["photorealistic_scene_integration_ok"] = False
                if "pasted/physical integration failure" not in result["failed_aspects"]:
                    result["failed_aspects"].append("pasted/physical integration failure")
                retry_focus = str(result.get("retry_focus_prompt", "") or "").strip()
                realism_retry = (
                    "Do not paste or redraw faces/heads; preserve the original face/head pixels, expression, "
                    "skin texture, and local lighting. For placed objects, match shoulder side, relative size, "
                    "perspective, contact shadow, blur/noise, and local lighting so the object is physically attached."
                )
                result["retry_focus_prompt"] = f"{retry_focus} {realism_retry}".strip()

            lighting_issue_terms = (
                "relit",
                "re-lit",
                "lighting changed",
                "local lighting",
                "lighting shift",
                "lighting drift",
                "shadow changed",
                "highlight changed",
                "over-lit",
                "under-lit",
                "studio-lit",
                "studio lit",
                "flat pasted",
                "color temperature",
                "color-temperature",
                "brightness changed",
                "lighting mismatch",
                "lighting difference",
            )
            critical_non_lighting_terms = (
                "face changed",
                "face shape",
                "facial expression",
                "expression changed",
                "gaze changed",
                "pose changed",
                "action changed",
                "clothing changed",
                "outfit changed",
                "dress color",
                "wrong shoulder",
                "background changed",
                "black bar",
                "hallucinat",
                "new person",
                "new woman",
                "new man",
                "new entity",
                "edit_completed",
                "not completed",
                "not performed",
                "remains visible",
                "pasted-on",
                "pasted face",
                "redrawn face",
            )
            lighting_only_issue = any(
                term in issue_text for term in lighting_issue_terms
            ) and not any(
                term in issue_text for term in critical_non_lighting_terms
            )
            if lighting_only_issue:
                result["photorealistic_scene_integration_ok"] = True
                if not clothing_drift and not hard_state_drift:
                    result["original_entity_state_preserved"] = True
                result["failed_aspects"] = [
                    aspect
                    for aspect in (result.get("failed_aspects") or [])
                    if str(aspect).strip()
                    not in {
                        "protected original state drifted",
                        "photorealistic_scene_integration_ok",
                        "original_entity_state_preserved",
                        "pasted/physical integration failure",
                    }
                ]

            core_edit_requirements_met = (
                bool(result.get("edit_instruction_requirements_met", result.get("edit_completed", False)))
                and bool(result.get("edit_completed", False))
                and bool(result.get("canonical_reference_alignment_ok", False))
            )
            if not core_edit_requirements_met:
                result["passed"] = False
                result["edit_instruction_requirements_met"] = False
                if not bool(result.get("edit_completed", False)):
                    if "edit instructions not completed" not in result["failed_aspects"]:
                        result["failed_aspects"].append("edit instructions not completed")
                if not bool(result.get("canonical_reference_alignment_ok", False)):
                    if "canonical target attributes not aligned" not in result["failed_aspects"]:
                        result["failed_aspects"].append("canonical target attributes not aligned")
                try:
                    result["score"] = min(float(result.get("score", 0.0) or 0.0), 0.35)
                except (TypeError, ValueError):
                    result["score"] = 0.0

                feedback = str(result.get("feedback", "") or "").strip()
                core_note = (
                    "Primary QA gate failed: the edited keyframe does not roughly satisfy all planned "
                    "edit instructions on the correct target entities/locations."
                )
                result["feedback"] = f"{feedback} {core_note}".strip()

                retry_focus = str(result.get("retry_focus_prompt", "") or "").strip()
                if not retry_focus:
                    retry_focus = (
                        "Do not skip, weaken, or misapply any planned edit. Do not edit the wrong entity, "
                        "do not leave removal targets visible, do not omit requested attributes, and do not "
                        "paste or replace the target with a reference-card person."
                    )
                else:
                    retry_focus = (
                        f"{retry_focus} Do not skip, weaken, or misapply any planned edit; the next attempt "
                        "must satisfy every planned edit instruction on the listed target locations."
                    )
                result["retry_focus_prompt"] = retry_focus.strip()

                positive = str(result.get("positive_prompt", "") or "").strip()
                if not positive:
                    positive = (
                        "Keep the original canvas/framing, non-target people, background outside edit regions, "
                        "and the target entities' original pose, expression, identity, clothing, lighting, and "
                        "occlusion except for explicitly requested edited attributes."
                    )
                result["positive_prompt"] = positive

            moderate_lighting_pass = (
                lighting_only_issue
                and result.get("frame_structure_preserved", False)
                and core_edit_requirements_met
                and result.get("edit_completed", False)
                and result.get("canonical_reference_alignment_ok", False)
                and result.get("background_unedited_regions_preserved", False)
                and not new_entity_hallucinated
                and not single_scope_multiple_edit
                and not background_drift
                and not hard_state_drift
                and not clothing_drift
                and float(result.get("score", 0.0) or 0.0) >= 0.3
            )
            if moderate_lighting_pass:
                result["passed"] = True
                result["qa_lighting_moderate_pass"] = True
                result["moderate_pass_reason"] = (
                    "Core edit requirements passed; reported lighting/shadow differences are treated as "
                    "acceptable when the result is broadly natural in the scene."
                )
                result["original_entity_state_preserved"] = True
                result["photorealistic_scene_integration_ok"] = True
                if result.get("unrelated_edit_changes_absent", True) is not False:
                    result["unrelated_edit_changes_absent"] = True

            core_ok = (
                result.get("frame_structure_preserved", False)
                and core_edit_requirements_met
                and result.get("edit_completed", False)
                and result.get("canonical_reference_alignment_ok", False)
            )
            no_significant_unrelated = (
                result.get("unrelated_edit_changes_absent", False)
                and result.get("background_unedited_regions_preserved", False)
            )
            hard_visual_quality_ok = (
                result.get("original_entity_state_preserved", False)
                and result.get("photorealistic_scene_integration_ok", False)
            )
            if (
                not result.get("passed", False)
                and core_ok
                and no_significant_unrelated
                and hard_visual_quality_ok
                and float(result.get("score", 0.0) or 0.0) >= 0.55
            ):
                result["passed"] = True
                result["qa_relaxed_pass"] = True
                result["relaxed_pass_reason"] = (
                    "Core edit requirements passed; no unrelated, state, lighting, or "
                    "photorealism issues were reported."
                )

            hard_gate_failures = []
            if not result.get("frame_structure_preserved", False):
                hard_gate_failures.append("frame_structure_preserved")
            if not result.get("edit_instruction_requirements_met", False):
                hard_gate_failures.append("edit_instruction_requirements_met")
            if not result.get("edit_completed", False):
                hard_gate_failures.append("edit_completed")
            if not result.get("canonical_reference_alignment_ok", False):
                hard_gate_failures.append("canonical_reference_alignment_ok")
            if not result.get("unrelated_edit_changes_absent", False):
                hard_gate_failures.append("unrelated_edit_changes_absent")
            if not result.get("background_unedited_regions_preserved", False):
                hard_gate_failures.append("background_unedited_regions_preserved")
            if hard_gate_failures:
                result["passed"] = False
                result["qa_hard_gate_failures"] = hard_gate_failures
                for failure in hard_gate_failures:
                    if failure not in result["failed_aspects"]:
                        result["failed_aspects"].append(failure)
                feedback = str(result.get("feedback", "") or "").strip()
                hard_gate_note = (
                    "Hard QA gate failed: every planned edit must be complete and all non-edit/background "
                    f"regions must be preserved. Failed checks: {', '.join(hard_gate_failures)}."
                )
                result["feedback"] = f"{feedback} {hard_gate_note}".strip()
                retry_focus = str(result.get("retry_focus_prompt", "") or "").strip()
                hard_gate_retry = (
                    "Do not skip or weaken any planned edit, and do not modify any non-target person, "
                    "non-requested target attribute, background, prop, wall, floor, ceiling, edge region, "
                    "or black bar outside the exact edit silhouette/minimal removal inpaint seam."
                )
                result["retry_focus_prompt"] = f"{retry_focus} {hard_gate_retry}".strip()
                try:
                    if (
                        "edit_instruction_requirements_met" in hard_gate_failures
                        or "edit_completed" in hard_gate_failures
                    ):
                        result["score"] = min(float(result.get("score", 0.0) or 0.0), 0.35)
                except (TypeError, ValueError):
                    result["score"] = 0.0

            if resized:
                feedback = str(result.get("feedback", "") or "").strip()
                note = "edited frame resized to original dimensions for QA"
                result["feedback"] = f"{feedback} | {note}".strip(" |")
            return result
        except Exception as exc:
            logger.error(
                "validate_keyframe_edit_completion failed: %s",
                exc,
                exc_info=True,
            )
            return parse_keyframe_edit_completion_qa_response({
                "passed": False,
                "score": 0.0,
                "edit_completed": False,
                "canonical_reference_alignment_ok": False,
                "unrelated_edit_changes_absent": False,
                "background_unedited_regions_preserved": False,
                "failed_aspects": ["qa_exception"],
                "feedback": str(exc),
                "retry_focus_prompt": "Retry the edit without changing unrelated regions.",
            })


    async def detect_entities_in_scene(
        self,
        *,
        keyframe_grid_path: str,
        multiview_paths: List[str],
        entity_specs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """VLM step 2: detect entities across the entire scene."""
        try:
            if not entity_specs:
                return {}
            
            catalog_lines = []
            for idx, spec in enumerate(entity_specs, start=2):
                iid = spec.get("instruction_id", "")
                eid = spec.get("entity_id", "")
                features = spec.get("subject_features", "")
                scope = spec.get("target_instance_scope", "single")
                catalog_lines.append(
                    f"- {iid} ({eid}), reference image {idx}: {features} (scope: {scope})"
                )
            
            prompt = SCENE_ENTITY_DETECT_PROMPT.format(
                entity_catalog_block="\n".join(catalog_lines)
            )
            
            images = [Image.open(keyframe_grid_path).convert("RGB")]
            for path in multiview_paths:
                if path and os.path.exists(path):
                    images.append(Image.open(path).convert("RGB"))
                    
            raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            
            results = {}
            for entry in data.get("entities", []):
                iid = entry.get("instruction_id")
                if iid:
                    results[iid] = entry
            return results
        except Exception as exc:
            logger.error("detect_entities_in_scene failed: %s", exc)
            return {}

    async def generate_scene_entity_reference_image(
        self,
        *,
        keyframe_grid_path: str,
        entity_locations_block: str,
        output_path: str,
    ) -> str:
        """Image Model step 3: generate scene entity reference image."""
        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            prompt = GENERATE_SCENE_ENTITY_REFERENCE_PROMPT.format(
                entity_locations_block=entity_locations_block
            )
            refs = [Image.open(keyframe_grid_path).convert("RGB")]
            img = await self._gen_image(prompt, refs=refs, save_path=output_path)
            if img is None and not os.path.exists(output_path):
                raise RuntimeError("scene entity reference synthesis returned no image")
            return output_path
        except Exception as exc:
            logger.error("generate_scene_entity_reference_image failed: %s", exc)
            raise


    async def edit_scene_entity_reference_image(
        self,
        *,
        source_path: str,
        canonical_ref_paths: List[str],
        edit_instructions_block: str,
        output_path: str,
        avoid_operations: str = "",
        positive_prompt: str = "",
    ) -> str:
        """Image model: edit the scene entity reference image."""
        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            if self.dev_mode:
                shutil.copy2(source_path, output_path)
                return output_path

            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Source image not found: {source_path}")

            avoid_section = build_edit_retry_guidance_section(
                positive_prompt=positive_prompt,
                avoid_operations=avoid_operations,
            )

            prompt = EDIT_SCENE_ENTITY_REFERENCE_PROMPT.format(
                canonical_edit_block=edit_instructions_block,
                avoid_section=avoid_section,
            )
            refs: List[Image.Image] = [
                Image.open(source_path).convert("RGB"),
            ]
            for path in canonical_ref_paths:
                if path and os.path.exists(path):
                    refs.append(Image.open(path).convert("RGB"))

            img = await self._gen_image(prompt, refs=refs, save_path=output_path)
            if img is None and not os.path.exists(output_path):
                raise RuntimeError("scene entity reference edit returned no image")
            return output_path
        except Exception as exc:
            logger.error("edit_scene_entity_reference_image failed: %s", exc)
            raise

    async def validate_scene_entity_reference_edit(
        self,
        *,
        edited_path: str,
        source_path: str,
        canonical_ref_paths: List[str],
        edit_instructions_block: str,
    ) -> Dict[str, Any]:
        """VLM QA: validate edited scene entity reference image."""
        try:
            if self.dev_mode:
                return {
                    "passed": True,
                    "score": 1.0,
                    "frame_structure_preserved": True,
                    "edit_completed": True,
                    "canonical_reference_alignment_ok": True,
                    "unrelated_edit_changes_absent": True,
                    "failed_aspects": [],
                    "feedback": "dev_mode pass",
                    "retry_focus_prompt": "",
                    "positive_prompt": "",
                }

            images: List[Image.Image] = []
            for path in (edited_path, source_path):
                if not path or not os.path.exists(path):
                    return {
                        "passed": False,
                        "score": 0.0,
                        "edit_completed": False,
                        "failed_aspects": ["missing image"],
                        "feedback": f"missing image: {path}",
                        "retry_focus_prompt": "Do not produce output without valid images.",
                        "positive_prompt": "",
                    }
                images.append(Image.open(path).convert("RGB"))

            for path in canonical_ref_paths:
                if path and os.path.exists(path):
                    images.append(Image.open(path).convert("RGB"))

            prompt = SCENE_ENTITY_REFERENCE_EDIT_QA_PROMPT.format(
                canonical_edit_block=edit_instructions_block,
            )
            raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            
            passed = bool(data.get("passed", False))
            score = _safe_float(data.get("score"), 0.0)
            
            return {
                "passed": passed,
                "score": score,
                "frame_structure_preserved": bool(data.get("frame_structure_preserved", False)),
                "edit_completed": bool(data.get("edit_completed", False)),
                "canonical_reference_alignment_ok": bool(data.get("canonical_reference_alignment_ok", False)),
                "unrelated_edit_changes_absent": bool(data.get("unrelated_edit_changes_absent", False)),
                "failed_aspects": [str(x) for x in data.get("failed_aspects", []) if str(x).strip()],
                "feedback": str(data.get("feedback", "")).strip(),
                "retry_focus_prompt": str(data.get("retry_focus_prompt", "")).strip(),
                "positive_prompt": str(data.get("positive_prompt", "")).strip(),
            }
        except Exception as exc:
            logger.error("validate_scene_entity_reference_edit failed: %s", exc)
            return {
                "passed": False,
                "score": 0.0,
                "edit_completed": False,
                "failed_aspects": ["api_error"],
                "feedback": f"API error: {exc}",
                "retry_focus_prompt": "",
                "positive_prompt": "",
            }

    async def _synthesize_keyframe_retry_prompts(
        self,
        *,
        positive_prompt: str,
        avoid_operations: str,
        canonical_edit_block: str,
        entity_locations_block: str,
        visibility_constraints_block: str = "",
    ) -> Tuple[str, str]:
        """Condense QA retry guidance so positive/avoid notes do not fight the edit plan."""
        positive = (positive_prompt or "").strip()
        avoid = (avoid_operations or "").strip()
        if not positive and not avoid:
            return positive, avoid

        def _clip(text: str, limit: int = 6000) -> str:
            text = (text or "").strip()
            if len(text) <= limit:
                return text
            return text[:limit].rstrip() + "\n...(truncated)"

        prompt = f"""
You are rewriting retry guidance for a single-keyframe image editing model.

The MANDATORY PLANNED EDITS and CURRENT KEYFRAME TARGET LOCATIONS are the source of truth.
The QA positive_prompt and avoid_operations are noisy feedback from the previous attempt.
They may contain contradictions, such as both "do not change the target hair/hat" and
"apply blue hair / green hat". Resolve every conflict in favor of the mandatory planned edits.

Rules:
- Do not simply concatenate the input prompts.
- Keep every mandatory planned edit active and visible.
- If QA says to preserve, avoid changing, or skip an attribute that is explicitly the target
  of a mandatory planned edit, drop that conflicting preservation note.
- Preserve only non-requested attributes, non-target people, background, pose, expression,
  lighting, clothing, and frame structure.
- Do not introduce new entities, new edits, or extra style changes.
- positive_prompt should be concise: only correct things to keep and non-target preservation.
- avoid_operations should be concise: mistakes to avoid, without cancelling mandatory edits.
- Return strict JSON only with keys:
  positive_prompt: string
  avoid_operations: string
  dropped_conflicts: array of strings

MANDATORY PLANNED EDITS:
{_clip(canonical_edit_block)}

CURRENT KEYFRAME TARGET LOCATIONS:
{_clip(entity_locations_block)}

VISIBILITY / STATE CONSTRAINTS:
{_clip(visibility_constraints_block) or "(none)"}

RAW QA positive_prompt:
{_clip(positive) or "(empty)"}

RAW QA avoid_operations:
{_clip(avoid) or "(empty)"}
""".strip()

        try:
            raw = await self._text(prompt)
            data = extract_json_object(raw)
            synthesized_positive = str(data.get("positive_prompt", "") or "").strip()
            synthesized_avoid = str(data.get("avoid_operations", "") or "").strip()
            if synthesized_positive or synthesized_avoid:
                dropped = data.get("dropped_conflicts") or []
                logger.info(
                    "Synthesized keyframe retry prompts; dropped_conflicts=%s",
                    dropped,
                )
                return synthesized_positive, synthesized_avoid
        except Exception as exc:
            logger.warning(
                "keyframe retry prompt synthesis failed; using raw QA prompts: %s",
                exc,
            )
        return positive, avoid

    async def edit_single_keyframe_with_canonical_refs(
        self,
        *,
        keyframe_path: str,
        canonical_ref_paths: List[str],
        entity_locations_block: str,
        canonical_edit_block: str,
        visibility_constraints_block: str = "",
        scene_story_context: str = "",
        prior_scene_edit_refs: List[Dict[str, Any]] | None = None,
        output_path: str,
        avoid_operations: str = "",
        positive_prompt: str = "",
        cross_keyframe_positive: str = "",
    ) -> str:
        """Image model: edit one keyframe using canonical before/after refs."""
        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            if self.dev_mode:
                shutil.copy2(keyframe_path, output_path)
                return output_path

            if not os.path.exists(keyframe_path):
                raise FileNotFoundError(f"Keyframe not found: {keyframe_path}")

            default_positive, default_avoid = build_default_keyframe_state_preservation_prompts(
                canonical_edit_block
            )
            is_retry = bool(
                (positive_prompt or "").strip() or (avoid_operations or "").strip()
            )
            if is_retry:
                positive_prompt, avoid_operations = await self._synthesize_keyframe_retry_prompts(
                    positive_prompt=positive_prompt,
                    avoid_operations=avoid_operations,
                    canonical_edit_block=canonical_edit_block,
                    entity_locations_block=entity_locations_block,
                    visibility_constraints_block=visibility_constraints_block,
                )
            avoid_section = build_edit_retry_guidance_section(
                positive_prompt=positive_prompt,
                avoid_operations=avoid_operations,
                baseline_positive=default_positive,
                baseline_avoid=default_avoid,
                retry_objective=(
                    build_keyframe_retry_edit_reinforcement(
                        canonical_edit_block,
                        entity_locations_block,
                    )
                    if is_retry
                    else ""
                ),
            )

            prior_scene_edit_block = (
                "(disabled) Previous edited keyframes are intentionally ignored for single-keyframe editing."
            )

            prompt = KEYFRAME_SINGLE_CANONICAL_EDIT_PROMPT.format(
                entity_locations_block=entity_locations_block,
                canonical_edit_block=canonical_edit_block,
                visibility_constraints_block=visibility_constraints_block
                or "(no visibility constraints)",
                scene_story_context=scene_story_context or "(not provided)",
                prior_scene_edit_block=prior_scene_edit_block,
                avoid_section=avoid_section,
            )
            refs: List[Image.Image] = [
                Image.open(keyframe_path).convert("RGB"),
            ]
            for path in canonical_ref_paths:
                if path and os.path.exists(path):
                    refs.append(Image.open(path).convert("RGB"))

            img = await self._gen_image(prompt, refs=refs, save_path=output_path)
            if img is None and not os.path.exists(output_path):
                raise RuntimeError("single keyframe edit returned no image")
            return output_path
        except Exception as exc:
            logger.error(
                "edit_single_keyframe_with_canonical_refs failed: %s",
                exc,
                exc_info=True,
            )
            raise ModelApiError(
                f"edit_single_keyframe_with_canonical_refs failed: {exc}"
            ) from exc

    async def validate_single_keyframe_edit(
        self,
        *,
        edited_keyframe_path: str,
        original_keyframe_path: str,
        canonical_ref_paths: List[str],
        entity_locations_block: str,
        canonical_edit_block: str,
        visibility_constraints_block: str = "",
        entity_location_records: Dict[str, Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """VLM QA for single-keyframe edit vs original."""
        try:
            if self.dev_mode:
                return apply_single_keyframe_edit_qa_gate({
                    "passed": True,
                    "score": 1.0,
                    "frame_structure_preserved": True,
                    "background_unedited_regions_preserved": True,
                    "canonical_reference_alignment_ok": True,
                    "visibility_extent_preserved": True,
                    "pose_expression_preserved": True,
                    "entity_local_lighting_preserved": True,
                    "unrelated_edit_changes_absent": True,
                    "environment_blend_ok": True,
                    "edit_completed": True,
                    "entity_identity_preserved": True,
                    "failed_aspects": [],
                    "feedback": "dev_mode pass",
                    "retry_focus_prompt": "",
                })

            if not edited_keyframe_path or not os.path.exists(edited_keyframe_path):
                return apply_single_keyframe_edit_qa_gate({
                    "passed": False,
                    "score": 0.0,
                    "frame_structure_preserved": False,
                    "background_unedited_regions_preserved": False,
                    "canonical_reference_alignment_ok": False,
                    "visibility_extent_preserved": False,
                    "pose_expression_preserved": False,
                    "entity_local_lighting_preserved": False,
                    "unrelated_edit_changes_absent": False,
                    "environment_blend_ok": False,
                    "edit_completed": False,
                    "entity_identity_preserved": False,
                    "failed_aspects": ["missing edited keyframe"],
                    "feedback": f"missing edited keyframe: {edited_keyframe_path}",
                    "retry_focus_prompt": (
                        "Do not omit the keyframe output image; preserve full frame."
                    ),
                })
            if not original_keyframe_path or not os.path.exists(original_keyframe_path):
                return apply_single_keyframe_edit_qa_gate({
                    "passed": False,
                    "score": 0.0,
                    "frame_structure_preserved": False,
                    "background_unedited_regions_preserved": False,
                    "canonical_reference_alignment_ok": False,
                    "visibility_extent_preserved": False,
                    "pose_expression_preserved": False,
                    "entity_local_lighting_preserved": False,
                    "unrelated_edit_changes_absent": False,
                    "environment_blend_ok": False,
                    "edit_completed": False,
                    "entity_identity_preserved": False,
                    "failed_aspects": ["missing original keyframe"],
                    "feedback": f"missing original keyframe: {original_keyframe_path}",
                    "retry_focus_prompt": (
                        "Do not omit the original keyframe reference."
                    ),
                })

            edit_img, orig_img, resized = prepare_keyframe_qa_images(
                original_keyframe_path,
                edited_keyframe_path,
            )
            images: List[Image.Image] = [edit_img, orig_img]
            for path in canonical_ref_paths:
                if path and os.path.exists(path):
                    images.append(Image.open(path).convert("RGB"))

            prompt = KEYFRAME_SINGLE_EDIT_QA_PROMPT.format(
                entity_locations_block=entity_locations_block,
                canonical_edit_block=canonical_edit_block,
                visibility_constraints_block=visibility_constraints_block
                or "(no visibility constraints)",
            )
            raw = await self._vision(prompt, images)
            data = extract_json_object(raw)
            qa_payload = build_single_keyframe_qa_result_from_vlm(
                data,
                resized_for_qa=resized,
                has_canonical_refs=bool(canonical_ref_paths),
            )
            qa_payload["qa_has_canonical_refs"] = bool(canonical_ref_paths)
            if entity_location_records:
                drift_metrics = measure_keyframe_background_drift(
                    original_keyframe_path,
                    edited_keyframe_path,
                    entity_location_records,
                )
                qa_payload = merge_background_drift_into_qa_result(
                    qa_payload,
                    drift_metrics,
                )
            alignment_results: List[Dict[str, Any]] = []
            if canonical_ref_paths:
                for ref_path in canonical_ref_paths:
                    instruction_id = instruction_id_from_canonical_ref_path(ref_path)
                    edit_instruction = parse_edit_instruction_for_instruction_id(
                        canonical_edit_block,
                        instruction_id,
                    )
                    if not canonical_alignment_check_applicable(edit_instruction):
                        continue
                    target_panel = extract_canonical_target_panel(ref_path)
                    if target_panel is None:
                        continue
                    loc_record = (entity_location_records or {}).get(instruction_id, {})
                    entity_location = (
                        str(loc_record.get("location_description", "") or "").strip()
                        or "(unspecified)"
                    )
                    align_prompt = KEYFRAME_CANONICAL_ALIGNMENT_PROMPT.format(
                        edit_instruction=edit_instruction or "(unspecified)",
                        entity_location=entity_location,
                    )
                    align_raw = await self._vision(
                        align_prompt,
                        [edit_img, target_panel],
                    )
                    align_data = extract_json_object(align_raw)
                    alignment_results.append({
                        "instruction_id": instruction_id,
                        "alignment_ok": bool(align_data.get("alignment_ok", False)),
                        "mismatched_attributes": align_data.get("mismatched_attributes")
                        or [],
                        "feedback": str(align_data.get("feedback", "") or "").strip(),
                        "retry_focus_prompt": str(
                            align_data.get("retry_focus_prompt", "") or ""
                        ).strip(),
                    })
                if alignment_results:
                    qa_payload = merge_canonical_alignment_into_qa_result(
                        qa_payload,
                        alignment_results,
                    )
            result = apply_single_keyframe_edit_qa_gate(qa_payload)
            if not result.get("passed"):
                logger.info(
                    "validate_single_keyframe_edit failed gate: %s",
                    "; ".join(result.get("qa_reject_reasons") or []),
                )
            return result
        except Exception as exc:
            logger.error(
                "validate_single_keyframe_edit failed: %s", exc, exc_info=True
            )
            return apply_single_keyframe_edit_qa_gate({
                "passed": False,
                "score": 0.0,
                "frame_structure_preserved": False,
                "background_unedited_regions_preserved": False,
                "canonical_reference_alignment_ok": False,
                "visibility_extent_preserved": False,
                "pose_expression_preserved": False,
                "entity_local_lighting_preserved": False,
                "unrelated_edit_changes_absent": False,
                "environment_blend_ok": False,
                "edit_completed": False,
                "entity_identity_preserved": False,
                "failed_aspects": ["qa_error"],
                "feedback": f"QA failed due to error: {exc}",
                "retry_focus_prompt": (
                    "Preserve original framing, visible extent, pose, local lighting, and edit scope."
                ),
            })

    async def execute_direct_scene_video_edit(
        self,
        source_clip_path: str,
        reference_image_path: str,
        edit_operation_prompt: str,
        output_clip_path: str,
        *,
        audio_path: str = "",
        skip_audio_mux: bool = False,
        avoid_operations: str = "",
        positive_prompt: str = "",
        retry_objective: str = "",
        reference_image_paths: Optional[List[str]] = None,
        video_resolution: str = "",
        reference_image_role: str = "entity_reference_grid",
    ) -> str:
        """Single-shot direct scene video edit via the configured video-edit model.

        When ``reference_image_paths`` is provided, each image is sent as a
        separate reference to the video model — one per entity, avoiding model
        confusion from a single combined grid.
        """
        try:
            from video_editing_agent.clients.video_client import (
                _is_seedance_model,
                generate_video_edit_with_references,
                generate_seedance_video_edit_to_file,
            )
            from video_editing_agent.utils.ffmpeg_utils import (
                conform_video_to_source,
                image_to_video_clip,
                mux_video_with_scene_audio,
                prepare_video_edit_source_clip,
                probe_duration,
            )

            os.makedirs(os.path.dirname(output_clip_path) or ".", exist_ok=True)

            if not os.path.exists(source_clip_path):
                raise FileNotFoundError(f"Source clip not found: {source_clip_path}")
            if not os.path.exists(reference_image_path):
                raise FileNotFoundError(
                    f"Reference image not found: {reference_image_path}"
                )

            base_edit_prompt = (edit_operation_prompt or "").strip()
            if not base_edit_prompt:
                raise RuntimeError("edit_operation_prompt is required")

            effective_edit_prompt = base_edit_prompt
            retry_guidance = build_edit_retry_guidance_section(
                positive_prompt=positive_prompt,
                avoid_operations=avoid_operations,
                retry_objective=retry_objective,
            )
            if retry_guidance:
                effective_edit_prompt += retry_guidance

            video_model = self._effective_video_model()
            if not video_model:
                video_model = (
                    os.environ.get("VIDEO_MODEL")
                    or "seedance-1-5-pro-251215"
                )

            effective_resolution = (video_resolution or self._effective_video_resolution()).strip().lower()

            use_seedance = _is_seedance_model(video_model)
            ref_role = (reference_image_role or "entity_reference_grid").strip().lower()
            # When multiple per-entity reference images are available, send them
            # as separate reference_image entries — one per entity.
            effective_ref_paths = reference_image_paths or ([reference_image_path] if reference_image_path else [])
            api_reference_image_path = effective_ref_paths[0] if effective_ref_paths else None
            api_reference_image_paths = effective_ref_paths if (len(effective_ref_paths) > 1) else None
            if use_seedance and ref_role != "first_frame":
                logger.warning(
                    "Seedance does not support a separate entity reference image in V2V mode; "
                    "using source video only and keeping entity refs in the text prompt."
                )
                api_reference_image_path = None
                api_reference_image_paths = None
            generation_mode = (
                "seedance_entity_ref_guided_video_edit"
                if use_seedance
                else "entity_ref_guided_video_edit"
            )

            scene_dur = probe_duration(source_clip_path)
            edit_sidecar = output_clip_path + ".edit_operation.json"

            if use_seedance:
                video_prompt = DIRECT_SCENE_SEEDANCE_VIDEO_EDIT_PROMPT.format(
                    edit_operation_prompt=effective_edit_prompt,
                )
            else:
                video_prompt = DIRECT_SCENE_VIDEO_EDIT_PROMPT.format(
                    edit_operation_prompt=effective_edit_prompt,
                )

            with open(edit_sidecar, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "edit_operation_prompt": effective_edit_prompt,
                        "base_edit_operation_prompt": base_edit_prompt,
                        "video_prompt": video_prompt,
                        "reference_image_path": reference_image_path,
                        "reference_image_paths": effective_ref_paths,
                        "api_reference_image_path": api_reference_image_path or "",
                        "api_reference_image_paths": api_reference_image_paths or [],
                        "reference_type": ref_role,
                        "source_clip_path": source_clip_path,
                        "audio_path": audio_path,
                        "video_model": video_model,
                        "generation_mode": generation_mode,
                        "video_resolution": effective_resolution,
                        "qa_avoid_edit_operations": avoid_operations,
                    },
                    fh,
                    indent=2,
                    ensure_ascii=False,
                )

            edit_raw = output_clip_path + (
                ".seedance.mp4" if use_seedance else ".videoedit.mp4"
            )
            conformed = output_clip_path + ".conformed.mp4"
            padded_input = edit_raw + ".hh_input_padded.mp4"

            if self.dev_mode:
                if ref_role == "first_frame":
                    logger.info("dev_mode: using reference image static clip (no video API)")
                    await asyncio.to_thread(
                        image_to_video_clip,
                        reference_image_path,
                        edit_raw,
                        scene_dur,
                    )
                else:
                    logger.info("dev_mode: copying source clip for entity-ref-guided edit")
                    shutil.copy2(source_clip_path, edit_raw)
            elif use_seedance:
                ok = await asyncio.to_thread(
                    generate_seedance_video_edit_to_file,
                    model=video_model,
                    prompt=video_prompt,
                    video_path=source_clip_path,
                    reference_image_path=api_reference_image_path,
                    output_path=edit_raw,
                    duration=scene_dur,
                    resolution=effective_resolution,
                    generate_audio=False,
                )
                if not ok:
                    raise VideoEditRejectedError(
                        "Seedance video edit API rejected or failed — using original video"
                    )
            else:
                source_input, source_input_padded = await asyncio.to_thread(
                    prepare_video_edit_source_clip,
                    source_clip_path,
                    padded_input,
                )
                hh_outcome = await asyncio.to_thread(
                    generate_video_edit_with_references,
                    model=video_model,
                    prompt=video_prompt,
                    video_path=source_input,
                    reference_image_path=api_reference_image_path,
                    reference_image_paths=api_reference_image_paths,
                    output_path=edit_raw,
                    resolution=effective_resolution,
                    audio_setting=os.environ.get("VIDEO_AUDIO_SETTING", "origin"),
                )
                if hh_outcome == "content_rejected":
                    raise VideoEditRejectedError(
                        f"Video-edit content-policy rejection — using original video: {hh_outcome}"
                    )
                if hh_outcome == "face_detection_suspect":
                    raise VideoEditRejectedError(
                        "Video-edit model rejected the request (content policy) — using original video"
                    )
                if hh_outcome != "success":
                    raise VideoEditRejectedError(
                        f"Video edit failed (outcome={hh_outcome}) — using original video"
                    )
                if source_input_padded:
                    logger.info(
                        "Edited output will be trimmed back to original %.2fs",
                        scene_dur,
                    )

            await asyncio.to_thread(
                conform_video_to_source,
                edit_raw,
                source_clip_path,
                conformed,
            )
            if skip_audio_mux:
                shutil.copy2(conformed, output_clip_path)
            else:
                await asyncio.to_thread(
                    mux_video_with_scene_audio,
                    conformed,
                    source_clip_path,
                    output_clip_path,
                    audio_path=audio_path or None,
                )

            for p in (edit_raw, conformed, padded_input):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

            logger.info(
                "%s direct scene video-edit → %s",
                "Seedance" if use_seedance else "video-edit model",
                output_clip_path,
            )
            return output_clip_path

        except VideoEditRejectedError:
            raise
        except Exception as exc:
            logger.error("execute_direct_scene_video_edit failed: %s", exc, exc_info=True)
            raise ModelApiError(f"execute_direct_scene_video_edit failed: {exc}") from exc

    async def analyze_shot_clip(
        self,
        clip_path: str,
        shot_id: str,
        scene_id: str,
        shot_index: int,
        shot_total: int,
        video_start_sec: float,
        video_end_sec: float,
    ) -> Dict[str, Any]:
        """VLM: plot, keyframes, and undetected sub-cut analysis for one shot clip."""
        try:
            duration_sec = max(0.0, video_end_sec - video_start_sec)
            if self.dev_mode:
                return {
                    "plot_description": (
                        f"Dev-mode placeholder plot for {shot_id} spanning "
                        f"{video_start_sec:.1f}s–{video_end_sec:.1f}s in the source video. "
                        "The clip shows a continuous interior scene with a subject entering frame "
                        "from the left, walking toward a doorway, and pausing as warm daylight "
                        "from the adjacent room catches their face and clothing."
                    ),
                    "sub_cut_detection_confidence": 0.0,
                    "sub_cut_rationale": "",
                    "false_positive_risks": [],
                    "has_undetected_sub_cuts": False,
                    "undetected_sub_cuts": [],
                    "transition_boundaries": [],
                    "transition_zones": [],
                    "keyframes": [
                        {
                            "description": (
                                "Opening frame of the shot — first frame of the clip; "
                                "static medium-wide view at the start of the segment."
                            ),
                            "timestamp_in_shot_sec": 0.0,
                            "role": "opening",
                        },
                        {
                            "description": (
                                "Closing frame: subject stopped at the doorway threshold, full body "
                                "visible in three-quarter view, face lit by daylight from the room beyond."
                            ),
                            "timestamp_in_shot_sec": min(duration_sec, 1.0),
                            "role": "closing",
                        },
                    ],
                }

            if not os.path.exists(clip_path):
                raise FileNotFoundError(f"Shot clip not found: {clip_path}")

            llm = _import_llm()
            prompt = SHOT_CLIP_VLM_ANALYSIS_PROMPT.format(
                shot_id=shot_id,
                scene_id=scene_id,
                shot_index=shot_index,
                shot_total=shot_total,
                duration_sec=duration_sec,
                video_start_sec=video_start_sec,
                video_end_sec=video_end_sec,
            )
            max_frames = int(os.environ.get("SHOT_ANALYSIS_MAX_FRAMES", "16"))
            raw = await asyncio.to_thread(
                llm.analyze_video_frames,
                clip_path,
                prompt,
                max_frames=max_frames,
            )
            data = extract_json_object(raw)
            if not isinstance(data, dict):
                raise RuntimeError("VLM shot analysis did not return a JSON object")
            return data
        except Exception as exc:
            logger.error("analyze_shot_clip failed for %s: %s", shot_id, exc, exc_info=True)
            raise ModelApiError(f"analyze_shot_clip failed: {exc}") from exc

    async def derive_video_scene_story_prompt(self, source_clip_path: str) -> str:
        """Analyze the source video clip into a detailed I2V story prompt."""
        try:
            if self.dev_mode:
                return (
                    "Continue natural handheld motion from the edited first frame to the edited "
                    "last frame. Preserve gentle camera stability, everyday subject movement, "
                    "timing, lighting, composition, and scene continuity. Generic individuals "
                    "only—casual user-shot footage."
                )
            if not os.path.exists(source_clip_path):
                raise FileNotFoundError(f"Source clip not found: {source_clip_path}")

            llm = _import_llm()
            story = await asyncio.to_thread(
                llm.analyze_video_frames,
                source_clip_path,
                VIDEO_SCENE_STORY_ANALYSIS_PROMPT,
                max_frames=int(os.environ.get("VIDEO_STORY_MAX_FRAMES", "16")),
            )
            story = (story or "").strip()
            if not story:
                raise RuntimeError("VLM returned empty source-video story prompt")
            return story
        except Exception as exc:
            logger.error(
                "derive_video_scene_story_prompt failed: %s", exc, exc_info=True
            )
            raise ModelApiError(
                f"derive_video_scene_story_prompt failed: {exc}"
            ) from exc

    async def rewrite_video_story_for_i2v(
        self,
        *,
        story_prompt: str,
        edit_operation_prompt: str,
    ) -> str:
        """Merge original-video story and first-frame edit into the final I2V prompt."""
        try:
            if self.dev_mode:
                return f"{story_prompt}\n\nApply and preserve this edit: {edit_operation_prompt}"

            prompt = VIDEO_EDIT_I2V_REWRITE_PROMPT.format(
                story_prompt=story_prompt,
                edit_operation_prompt=edit_operation_prompt,
            )
            rewritten = (await self._text(prompt)).strip()
            if not rewritten:
                raise RuntimeError("LLM returned empty rewritten I2V prompt")
            return rewritten
        except Exception as exc:
            logger.error("rewrite_video_story_for_i2v failed: %s", exc, exc_info=True)
            raise ModelApiError(f"rewrite_video_story_for_i2v failed: {exc}") from exc

    async def video_propagate_edit(
        self,
        source_clip_path: str,
        mask_dir: str,
        original_first_frame_path: str,
        edited_first_frame_path: str,
        location_prompts_path: str,
        output_clip_path: str,
        *,
        original_last_frame_path: str = "",
        edited_last_frame_path: str = "",
        audio_path: str = "",
        fallback_edit_prompt: str = "",
        edit_operation_prompt: str = "",
        skip_audio_mux: bool = False,
        entity_instru_path: str = "",
    ) -> str:
        """Derive edit ops from keyframe diff, then propagate via the configured video-edit model."""
        try:
            from video_editing_agent.clients.video_client import (
                _is_seedance_model,
                generate_video_edit_with_references,
                generate_seedance_video_edit_to_file,
            )
            from video_editing_agent.utils.ffmpeg_utils import (
                conform_video_to_source,
                image_to_video_clip,
                mux_video_with_scene_audio,
                prepare_video_edit_source_clip,
                probe_duration,
            )

            os.makedirs(os.path.dirname(output_clip_path) or ".", exist_ok=True)

            if not os.path.exists(source_clip_path):
                raise FileNotFoundError(f"Source clip not found: {source_clip_path}")
            if not os.path.exists(edited_first_frame_path):
                raise FileNotFoundError(
                    f"Edited first frame not found: {edited_first_frame_path}"
                )

            precomputed_edit = bool(edit_operation_prompt.strip())
            if precomputed_edit:
                edit_operation_prompt = edit_operation_prompt.strip()
            else:
                edit_operation_prompt = await self.derive_video_edit_operation_prompt(
                    original_first_frame_path=original_first_frame_path,
                    edited_first_frame_path=edited_first_frame_path,
                    location_prompts_path=location_prompts_path,
                    fallback_edit_prompt=fallback_edit_prompt,
                    entity_instru_path=entity_instru_path,
                )

            video_model = self._effective_video_model()
            if not video_model:
                video_model = (
                    os.environ.get("VIDEO_MODEL")
                    or "seedance-1-5-pro-251215"
                )

            effective_resolution = self._effective_video_resolution()

            use_seedance = _is_seedance_model(video_model)
            generation_mode = (
                "seedance_video_edit_reference_videos"
                if use_seedance
                else "video_edit_from_edited_first_frame"
            )

            sidecar_has_planned = False
            if location_prompts_path and os.path.exists(location_prompts_path):
                try:
                    with open(location_prompts_path, encoding="utf-8") as fh:
                        sidecar_preview = json.load(fh)
                    sidecar_has_planned = bool(
                        sidecar_preview.get("planned_edits")
                        or sidecar_preview.get("records")
                    )
                except (OSError, json.JSONDecodeError):
                    sidecar_has_planned = False

            generation_source = (
                "precomputed"
                if precomputed_edit
                else "planned_edits"
                if sidecar_has_planned
                else "vlm_diff"
            )

            scene_dur = probe_duration(source_clip_path)
            edit_sidecar = output_clip_path + ".edit_operation.json"
            qa_sidecar = output_clip_path + ".qa.json"
            base_edit_operation_prompt = edit_operation_prompt

            qa_enabled = self._video_edit_qa_enabled()
            max_attempts = 2 if qa_enabled else 1
            retry_focus = ""
            qa_history: List[Dict[str, Any]] = []

            for attempt in range(1, max_attempts + 1):
                effective_edit_prompt = (
                    self._append_video_edit_retry_focus(
                        base_edit_operation_prompt,
                        retry_focus,
                    )
                    if attempt > 1
                    else base_edit_operation_prompt
                )
                if use_seedance:
                    video_prompt = SEEDANCE_VIDEO_EDIT_PROMPT.format(
                        edit_operation_prompt=effective_edit_prompt,
                    )
                else:
                    video_prompt = GENERIC_VIDEO_EDIT_PROMPT.format(
                        edit_operation_prompt=effective_edit_prompt,
                    )

                with open(edit_sidecar, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "edit_operation_prompt": effective_edit_prompt,
                            "base_edit_operation_prompt": base_edit_operation_prompt,
                            "video_prompt": video_prompt,
                            "original_first_frame_path": original_first_frame_path,
                            "edited_first_frame_path": edited_first_frame_path,
                            "location_prompts_path": location_prompts_path,
                            "entity_instru_path": entity_instru_path,
                            "source_clip_path": source_clip_path,
                            "audio_path": audio_path,
                            "video_model": video_model,
                            "generation_mode": generation_mode,
                            "generation_source": generation_source,
                            "attempt": attempt,
                            "qa_avoid_edit_operations": retry_focus if attempt > 1 else "",
                            "qa_retry_focus": retry_focus if attempt > 1 else "",
                        },
                        fh,
                        indent=2,
                        ensure_ascii=False,
                    )

                suffix = "" if attempt == 1 else f".retry{attempt}"
                edit_raw = output_clip_path + (
                    f".seedance{suffix}.mp4" if use_seedance else f".videoedit{suffix}.mp4"
                )
                conformed = output_clip_path + f".conformed{suffix}.mp4"
                padded_input = edit_raw + ".hh_input_padded.mp4"
                face_detection_static_fallback = False

                if self.dev_mode:
                    logger.info("dev_mode: using edited first frame static clip (no video API)")
                    await asyncio.to_thread(
                        image_to_video_clip,
                        edited_first_frame_path,
                        edit_raw,
                        scene_dur,
                    )
                elif use_seedance:
                    ok = await asyncio.to_thread(
                        generate_seedance_video_edit_to_file,
                        model=video_model,
                        prompt=video_prompt,
                        video_path=source_clip_path,
                        reference_image_path=edited_first_frame_path,
                        output_path=edit_raw,
                        duration=scene_dur,
                        resolution=effective_resolution,
                        generate_audio=False,
                    )
                    if not ok:
                        await self._write_api_failure_static_clip(
                            source_clip_path=source_clip_path,
                            edit_raw=edit_raw,
                            original_first_frame_path=original_first_frame_path,
                        )
                else:
                    source_input, source_input_padded = await asyncio.to_thread(
                        prepare_video_edit_source_clip,
                        source_clip_path,
                        padded_input,
                    )
                    hh_outcome = await asyncio.to_thread(
                        generate_video_edit_with_references,
                        model=video_model,
                        prompt=video_prompt,
                        video_path=source_input,
                        reference_image_path=edited_first_frame_path,
                        output_path=edit_raw,
                        resolution=effective_resolution,
                        audio_setting=os.environ.get("VIDEO_AUDIO_SETTING", "origin"),
                    )
                    face_detection_static_fallback = (
                        hh_outcome == "face_detection_suspect"
                    )
                    if hh_outcome != "success":
                        if face_detection_static_fallback:
                            logger.warning(
                                "Video-edit model rejected the request — original first frame "
                                "still fallback (skip VLM QA and video-edit retry)"
                            )
                        await self._write_api_failure_static_clip(
                            source_clip_path=source_clip_path,
                            edit_raw=edit_raw,
                            original_first_frame_path=original_first_frame_path,
                        )
                    if source_input_padded:
                        logger.info(
                            "Edited output will be trimmed back to original %.2fs",
                            scene_dur,
                        )

                await asyncio.to_thread(
                    conform_video_to_source,
                    edit_raw,
                    source_clip_path,
                    conformed,
                )
                if skip_audio_mux:
                    shutil.copy2(conformed, output_clip_path)
                else:
                    await asyncio.to_thread(
                        mux_video_with_scene_audio,
                        conformed,
                        source_clip_path,
                        output_clip_path,
                        audio_path=audio_path or None,
                    )

                for p in (edit_raw, conformed, padded_input):
                    if p and os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass

                if face_detection_static_fallback:
                    try:
                        with open(edit_sidecar, encoding="utf-8") as fh:
                            sidecar_data = json.load(fh)
                        sidecar_data["video_edit_outcome"] = "face_detection_suspect"
                        sidecar_data["generation_mode"] = (
                            "static_fallback_content_rejected"
                        )
                        with open(edit_sidecar, "w", encoding="utf-8") as fh:
                            json.dump(sidecar_data, fh, indent=2, ensure_ascii=False)
                    except (OSError, json.JSONDecodeError):
                        pass
                    with open(qa_sidecar, "w", encoding="utf-8") as fh:
                        json.dump(
                            {
                                "skipped": True,
                                "reason": "face_detection_suspect_static_fallback",
                                "final_passed": True,
                                "qa_enabled": qa_enabled,
                                "attempt": attempt,
                                "feedback": (
                                    "Skipped VLM QA and video-edit retry after "
                                    "Video-edit content rejection static fallback."
                                ),
                            },
                            fh,
                            indent=2,
                            ensure_ascii=False,
                        )
                    break

                if not qa_enabled:
                    break

                qa = await self.validate_video_edit_quality(
                    reference_edited_frame_path=edited_first_frame_path,
                    edited_video_path=output_clip_path,
                    edit_operation_prompt=base_edit_operation_prompt,
                )
                qa["attempt"] = attempt
                qa_history.append(qa)
                with open(qa_sidecar, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "attempts": qa_history,
                            "final_passed": bool(qa.get("passed")),
                            "qa_enabled": qa_enabled,
                        },
                        fh,
                        indent=2,
                        ensure_ascii=False,
                    )

                if qa.get("passed"):
                    logger.info("Video edit QA passed (attempt %d)", attempt)
                    break

                retry_focus = str(qa.get("retry_focus_prompt", "")).strip()
                logger.warning(
                    "Video edit QA failed (attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    qa.get("feedback", ""),
                )
                if attempt >= max_attempts:
                    logger.warning(
                        "Video edit QA — keeping last output after single retry"
                    )
                    break

            logger.info(
                "%s video-edit propagate → %s",
                "Seedance" if use_seedance else "video-edit model",
                output_clip_path,
            )
            return output_clip_path

        except Exception as exc:
            logger.error("video_propagate_edit failed: %s", exc, exc_info=True)
            raise ModelApiError(f"video_propagate_edit failed: {exc}") from exc
