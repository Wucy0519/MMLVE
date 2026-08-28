"""
Abstract model API client interface.

All LLM, vision, image, and video model calls in the pipeline must go
through this interface so concrete HTTP implementations can be swapped
without touching pipeline modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class ModelApiClientBase(ABC):
    """Abstract client for the unified model APIs.

  Models referenced by the pipeline (configured via ``env.example.sh``):

  - ``gemini-3.1-pro``: structured JSON parsing, event grounding, VLM QA
  - ``gemini-3.1-flash-image``: T2I reference, solid indicative segmentation masks, inpainting
  - ``seedance-1-5-pro``: video-edit model (source video + reference_image)
    """

    # ── LLM (gemini-3.1-pro) ─────────────────────────────────────────────

    @abstractmethod
    async def rewrite_user_prompt(
        self,
        user_prompt: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Rewrite vague user input into a clear editing brief via Gemini.

        Returns:
            Dict with ``rewritten_prompt`` (str) and ``clarifications`` (list[str]).
        """
        ...

    @abstractmethod
    async def parse_instructions(
        self,
        user_prompt: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Parse natural-language prompt into structured instruction JSON.

        Args:
            user_prompt: Raw user editing request.
            system_prompt: Optional override for the structuring system prompt.

        Returns:
            Dict matching the ``EntityInstruction`` list schema (pre-conflict).

        Raises:
            ModelApiError: On HTTP or model failure.
        """
        ...

    @abstractmethod
    async def resolve_temporal_conflicts(
        self,
        instructions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Detect and resolve overlapping temporal conflicts per entity.

        Rules:
            - Validate timeline ordering first.
            - Only when two instructions for the same ``entity_id`` overlap in
              time, keep the **last** instruction and drop earlier ones.

        Args:
            instructions: Raw instruction dicts from :meth:`parse_instructions`.

        Returns:
            Conflict-resolved instruction dict list.
        """
        ...

    @abstractmethod
    async def match_event_scenes(
        self,
        scene_keyframes: List[Dict[str, Any]],
        time_condition: Dict[str, Any],
    ) -> List[str]:
        """Event grounding — decide which scenes satisfy a time condition.

        Args:
            scene_keyframes: List of ``{"scene_id": str, "image_path": str}``.
            time_condition: Serialized :class:`~video_editing_agent.schemas.instructions.TimeCondition`.

        Returns:
            Matched ``scene_id`` values.
        """
        ...

    @abstractmethod
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
        """VLM analysis for one PySceneDetect shot clip.

        Returns:
            Dict with ``plot_description``, ``keyframes``, ``has_undetected_sub_cuts``,
            and ``undetected_sub_cuts`` (timestamps relative to the shot clip).
        """
        ...

    @abstractmethod
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
        """VLM: detect edit-target entity on one scene keyframe.

        Returns:
            Dict with ``present``, ``confidence``, ``scene_moment_description``,
            ``visibility_state``, ``pose_and_action``, ``location_description``, ``reasoning``,
            ``quality_score``, ``appearance_time_score``, ``subject_features_score``,
            ``identification_clarity_score``, ``view_angle``.
        """
        ...

    @abstractmethod
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
        """VLM: pick reference keyframe indices for front-view synthesis."""
        ...

    @abstractmethod
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
        """Image model: synthesize one front-view entity reference from keyframe grid."""
        ...

    @abstractmethod
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
        """Image model: edit a front-view entity reference per instruction."""
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        """VLM: return best candidate index (0-based) among front-view outputs."""
        ...

    @abstractmethod
    async def check_entity_in_frame(
        self,
        image_path: str,
        subject_features: str,
        *,
        action: str = "modify",
        reference_image_path: Optional[str] = None,
    ) -> bool:
        """Return True if the target subject is visible in the frame."""
        ...

    @abstractmethod
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
        """VLM check that a first-time mask region matches expected subject features.

        Returns:
            Dict with ``valid``, ``matches_subject_features``, ``confidence``, ``feedback``.
        """
        ...

    @abstractmethod
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
        """VLM self-correction check for keyframe inpainting quality.

        Args:
            original_image_path: Unedited anchor frame (image 2 in keyframe QA).
            edited_image_path: Inpainted result frame (image 1 in keyframe QA).
            edit_prompt: Intended edit instruction text (not location directives).
            subject_features: Expected subject appearance.
            success_criteria_prompt: Rewrite-stage QA checklist for this edit.
            keyframe: Use stricter Module-3 QA; returns ``edit_errors`` and
            ``retry_focus_prompt`` (editing operations to avoid on retry).

        Returns:
            Dict with ``passed``, ``score``, ``feedback``; keyframe mode also returns
            ``edit_completed``, ``frame_structure_preserved``, ``failed_aspects``,
            ``edit_errors``, ``retry_focus_prompt``.
        """
        ...

    # ── Image (gemini-3.1-flash-image) ───────────────────────────────────

    @abstractmethod
    async def generate_reference_image(
        self,
        prompt: str,
        output_path: str,
        *,
        white_background: bool = True,
        forbidden_elements: Optional[List[str]] = None,
    ) -> str:
        """T2I isolated asset reference on white background."""
        ...

    @abstractmethod
    async def validate_reference_image_semantics(
        self,
        image_path: str,
        ref_subject: str,
        *,
        action: str = "add",
        forbidden_elements: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """VLM QA — reference must show only the isolated asset."""
        ...

    @abstractmethod
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
        """Generate reference image with semantic QA retry loop."""
        ...

    @abstractmethod
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
        """Image-model solid indicative segmentation on first frame (not bbox).

        Args:
            image_path: Scene first-frame image.
            entity_descriptions: ``[{"entity_id": str, "description": str}, ...]``.
            output_mask_path: Where to write the multi-color guide mask PNG.
            color_map: Stable ``entity_id`` → palette hex mapping (from workspace registry).
            entity_references: Optional ``entity_id`` → first-match reference frame path.
            instruction_labels: Optional ``entity_id`` → ``instruction_id`` for prompts.

        Note:
            When entity reference overlays exist, a VLM first compares each reference with
            the detection frame to produce per-entity location hints. One image-model call
            then segments image 1 using those hints (reference images are not attached).

        Returns:
            Absolute path to the indicative mask image.
        """
        ...

    @abstractmethod
    async def compare_entity_reference_candidates(
        self,
        existing_reference_path: str,
        candidate_reference_path: str,
        *,
        subject_features: str,
    ) -> str:
        """Pick which reference image better matches the target subject features.

        Returns:
            ``"existing"`` or ``"candidate"``.
        """
        ...

    @abstractmethod
    async def derive_keyframe_edit_entity_locations(
        self,
        image_path: str,
        mask_path: str,
        edit_entities: List[Dict[str, str]],
    ) -> Tuple[Dict[str, str], List[Dict[str, object]]]:
        """VLM: locate entities on the target keyframe using frame + entity_refs only.

        Args:
            image_path: Target edit frame.
            mask_path: Unused for location VLM (retained for call-site compatibility).
            edit_entities: Per-instruction dicts with instruction_id, entity_id,
                color_name, color_hex, subject_features, reference_overlay_path.

        Returns:
            ``(location_prompt_by_instruction_id, structured_records)``.
        """
        ...

    @abstractmethod
    async def masked_inpaint(
        self,
        image_path: str,
        mask_path: str,
        edit_directives: str,
        output_path: str,
        *,
        ref_image_path: Optional[str] = None,
        consistency_ref_paths: Optional[List[str]] = None,
        entity_ref_guides: Optional[List[Dict]] = None,
        strength: float = 1.0,
        preserve_frame_structure: bool = False,
        inpaint_guidance: str = "mask",
    ) -> str:
        """Inpainting with optional reference conditioning.

        Args:
            image_path: Original scene frame (image 2 for mask mode, image 1 for location mode).
            mask_path: Shared multi-color indicative mask (image 1 for mask mode; unused for location mode).
            edit_directives: Region- or location-specific edit instructions.
            output_path: Destination for the edited image.
            ref_image_path: Optional reference for add-asset appearance.
            consistency_ref_paths: Optional canonical edited references for cross-scene consistency.
            entity_ref_guides: Optional per-instruction entity_refs bundles (src/mask/overlay/canonical).
            strength: Inpainting strength knob for retries.
            preserve_frame_structure: When True, hard-composite unmasked pixels and letterbox
                bands from the original onto the model output. When False, keep the raw model output.
            inpaint_guidance: ``"mask"`` uses mask+scene images; ``"location"`` uses scene-only
                with location text in edit_directives (Module 3 keyframe editing).

        Returns:
            Absolute path to the edited image.
        """
        ...

    # ── Video (veo-3.1-fast) ─────────────────────────────────────────────

    @abstractmethod
    async def propagate_masks_vos(
        self,
        scene_frame_dir: str,
        initial_mask_path: str,
        output_mask_dir: str,
        entity_color_map: Dict[str, str],
        *,
        entity_descriptions: Optional[Dict[str, str]] = None,
    ) -> str:
        """Store first-frame indicative mask; V2V handles temporal propagation.

        Args:
            scene_frame_dir: Directory of ordered scene frames (unused; kept for API compat).
            initial_mask_path: First-frame indicative mask.
            output_mask_dir: Directory for mask guide output.
            entity_color_map: ``entity_id`` → color hex mapping.
            entity_descriptions: Unused; kept for API compatibility.

        Returns:
            Absolute path to ``output_mask_dir``.
        """
        ...

    @abstractmethod
    async def derive_video_edit_operation_prompt(
        self,
        original_first_frame_path: str,
        edited_first_frame_path: str,
        location_prompts_path: str,
        *,
        fallback_edit_prompt: str = "",
        entity_instru_path: str = "",
    ) -> str:
        """VLM: compare original vs edited frame using location sidecar; return edit ops prompt."""
        ...

    @abstractmethod
    async def derive_video_chunk_edit_operation_prompt(
        self,
        original_chunk_first_frame_path: str,
        previous_edited_last_frame_path: str,
    ) -> str:
        """VLM: infer edit ops from original chunk first frame vs prior edited last frame."""
        ...

    @abstractmethod
    async def validate_video_edit_quality(
        self,
        reference_edited_frame_path: str,
        edited_video_path: str,
        edit_operation_prompt: str,
    ) -> Dict[str, Any]:
        """VLM QA — reference edited frame vs edited video first frame.

        Returns:
            Dict with ``passed``, ``score``, ``first_frame_consistent``, ``edit_completed``,
            ``failed_aspects``, ``feedback``, ``retry_focus_prompt``.
        """
        ...

    @abstractmethod
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
        """VLM: derive direct scene video edit prompt from clip samples + refs + instructions."""
        ...

    @abstractmethod
    async def validate_scene_video_edit_keyframe_grids(
        self,
        *,
        edited_keyframes_grid_path: str,
        original_keyframes_grid_path: str,
        entity_ref_image_paths: List[str],
        entity_instru_json: str,
        edit_operation_prompt: str,
    ) -> Dict[str, Any]:
        """VLM QA comparing edited vs original keyframe grids + entity refs."""
        ...

    @abstractmethod
    async def vote_scene_entity_existence(
        self,
        *,
        keyframe_paths: List[str],
        entity_ref_image_paths: List[str],
        entity_catalog_block: str,
    ) -> Dict[str, Any]:
        """VLM single-vote: does any edit-target entity appear in the scene keyframes?

        Returns a dict with ``scene_has_edit_target`` (bool), per-entity results,
        and ``reasoning``.
        """
        ...

    @abstractmethod
    async def select_best_video_edit_attempt(
        self,
        *,
        original_keyframes_grid_path: str,
        candidate_grid_paths: List[str],
        entity_ref_image_paths: List[str],
        entity_instru_json: str,
        edit_operation_prompt: str,
    ) -> Dict[str, Any]:
        """VLM: select the best edited keyframe grid from multiple attempts.

        Returns a dict with ``best_candidate_index`` (0-based int),
        ``reasoning``, and ``per_candidate_scores``.
        """
        ...

    @abstractmethod
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
        """Run one video-edit generation without internal QA retry.

        When ``reference_image_paths`` is provided, each image is sent as a
        separate reference — one per entity — to avoid model confusion.
        """
        ...

    @abstractmethod
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

        Returns dict with keys: positive_prompt, avoid_operations, missing_edits_prompt,
        retry_objective.
        """
        ...

    @abstractmethod
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
        """VLM: per-instruction entity presence/location on each keyframe panel."""
        ...

    @abstractmethod
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
        """Image model: edit labeled keyframe strip using entity edit references."""
        ...

    @abstractmethod
    async def validate_scene_keyframe_grid_edit(
        self,
        *,
        edited_grid_path: str,
        original_grid_path: str,
        multiview_edited_paths: List[str],
        edit_instructions_block: str,
        entity_locations_block: str,
    ) -> Dict[str, Any]:
        """VLM QA for edited vs original keyframe strip."""
        ...

    @abstractmethod
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
        """VLM: does entity in front-view ref appear in keyframe; if yes, where."""
        ...

    @abstractmethod
    async def locate_all_entities_on_single_keyframe(
        self,
        *,
        keyframe_path: str,
        entity_specs: List[Dict[str, Any]],
        scene_prior_by_instruction: Dict[str, List[Dict[str, Any]]] | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        """VLM: detect all entities on one keyframe in a single call."""
        ...

    @abstractmethod
    async def detect_entities_on_keyframe(
        self,
        *,
        keyframe_path: str,
        entity_specs: List[Dict[str, Any]],
        scene_story_context: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        """VLM step 1: detect entities with confidence and location on one keyframe."""
        ...

    @abstractmethod
    async def verify_entity_locations_on_keyframe(
        self,
        *,
        keyframe_path: str,
        entity_specs: List[Dict[str, Any]],
        detection_records: Dict[str, Dict[str, Any]],
        scene_story_context: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        """VLM step 2: verify and correct entity locations (single pass)."""
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    async def describe_keyframe_edit_operations(
        self,
        *,
        original_keyframe_path: str,
        edited_keyframe_path: str,
        canonical_edit_block: str = "",
        visibility_constraints_block: str = "",
    ) -> Dict[str, Any]:
        """VLM step 4: describe observed edit operations between original and edited frames."""
        ...

    @abstractmethod
    async def describe_entity_from_synthesis_sheet(
        self,
        *,
        synth_sheet_path: str,
        instruction_id: str,
        entity_id: str,
        original_subject_features: str = "",
    ) -> str:
        """VLM: produce a detailed visual description of the entity shown in the synthesis sheet.

        The synthesis sheet (entity_refs/instr_00N_front_work/synth_sheet.png)
        is a front-view reference generated from multiple keyframe sightings.
        This method asks the VLM to describe the entity in detail so the
        description can replace the original (often coarse) subject_features
        as the authoritative identity description.
        """
        ...

    @abstractmethod
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
        """VLM step 5: validate edit completion without unrelated changes."""
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        """Video-edit propagation from source clip + edited first frame.

        Args:
            source_clip_path: Original scene video segment.
            mask_dir: Indicative mask guide dir (reference only, not composited).
            original_first_frame_path: Scene first frame before Module 3 edit.
            edited_first_frame_path: Module-3 edited first anchor frame.
            location_prompts_path: Module-3 ``.location_prompts.json`` sidecar.
            output_clip_path: Output edited scene clip path.
            original_last_frame_path: Unused (retained for API compatibility).
            edited_last_frame_path: Unused (retained for API compatibility).
            audio_path: Optional extracted scene audio (AAC).
            fallback_edit_prompt: Used when VLM diff or sidecar is unavailable.
            edit_operation_prompt: When set, skip VLM diff and use this prompt directly.
            skip_audio_mux: When True, write conformed video-only output (chunk sub-edits).

        Returns:
            Absolute path to the edited scene video.
        """
        ...


class ModelApiError(RuntimeError):
    """Raised when a model API call fails."""

    def __init__(self, message: str, *, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(message)


class VideoEditRejectedError(RuntimeError):
    """Raised when the video edit model refuses to edit (content policy, moderation, etc.).

    When this is raised, the caller should stop retrying and use the original
    unedited video as the output.
    """
