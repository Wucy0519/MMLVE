"""
Runtime configuration loaded from environment variables.

Credentials and model names are defined in ``env.example.sh`` at the project
root.  Source that file before running the agent::

    source env.example.sh

The pipeline talks to two officially-supported backends:

* An OpenAI-compatible ``chat/completions`` endpoint for text / vision / image
  generation (``OPENAI_API_KEY`` + ``OPENAI_BASE_URL``).
* BytePlus ModelArk (Seedance) for scene video editing (``BYTEPLUS_API_KEY``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _first_env(*names: str, default: str = "") -> str:
    """Return the first non-empty environment variable among ``names``."""
    for name in names:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return default


@dataclass
class ModelApiConfig:
    """Unified API configuration for LLM / vision / image / video backends.

    Attributes:
        api_token: Bearer API key for the OpenAI-compatible chat endpoint.
        api_url: Chat/completions endpoint for LLM and image models.
        text_model: Model for structured JSON parsing and VLM QA.
        vision_model: Model for image understanding tasks.
        image_model: Model for T2I reference, solid segmentation masks, and inpainting.
        video_model: Model for V2V edit propagation (BytePlus Seedance).
        request_timeout: HTTP timeout in seconds.
        max_vision_images: Max images per vision request.
        video_poll_sec: Polling interval for async video jobs.
    """

    api_token: str = ""
    api_url: str = "https://api.openai.com/v1/chat/completions"
    text_model: str = "gpt-4o-mini"
    vision_model: str = "gpt-4o-mini"
    image_model: str = "gpt-image-1"
    video_model: str = "seedance-1-5-pro-251215"
    request_timeout: int = 300
    max_vision_images: int = 8
    video_poll_sec: int = 30
    # Resolution mode determined from source video: "480p" or "720p".
    # Set by AgentConfig after probing the source video.
    video_resolution: str = ""
    # Low-resolution model overrides (used when video_resolution == "480p").
    image_model_low_res: str = ""
    video_model_low_res: str = ""

    @classmethod
    def from_env(cls) -> ModelApiConfig:
        """Build config from process environment (see ``env.example.sh``)."""
        return cls(
            api_token=_first_env("OPENAI_API_KEY", "LLM_API_KEY"),
            api_url=_first_env("OPENAI_BASE_URL", "LLM_BASE_URL", default=cls.api_url),
            text_model=_first_env("LLM_TEXT_MODEL", default=cls.text_model),
            vision_model=_first_env("LLM_VISION_MODEL", default=cls.vision_model),
            image_model=_first_env("LLM_IMAGE_MODEL", default=cls.image_model),
            video_model=_first_env("VIDEO_MODEL", default=cls.video_model),
            request_timeout=int(_first_env("LLM_REQUEST_TIMEOUT", default="300")),
            max_vision_images=int(_first_env("LLM_MAX_VISION_IMAGES", default="8")),
            video_poll_sec=int(_first_env("VIDEO_POLL_SEC", default="30")),
            image_model_low_res=_first_env("LLM_IMAGE_MODEL_480P"),
            video_model_low_res=_first_env("VIDEO_MODEL_480P"),
        )


@dataclass
class AgentConfig:
    """Top-level agent workspace and pipeline knobs.

    Attributes:
        workspace_dir: Root directory for intermediate artifacts and outputs.
        source_video_path: Absolute path to the input long video.
        scene_detect_threshold: PySceneDetect ContentDetector threshold (lower = more cuts).
        scene_detect_mode: ``content``, ``adaptive``, ``hybrid`` (content+adaptive),
            or ``enhanced`` (content+adaptive+hash+histogram — recommended).
        scene_detect_min_scene_len: Min frames between cuts (lower catches rapid cuts).
        scene_adaptive_threshold: AdaptiveDetector ratio threshold.
        scene_min_content_val: AdaptiveDetector absolute content floor.
        max_inpaint_retries: Max self-correction attempts in keyframe editing.
        mask_visibility_prefilter: If True, skip mask queries for entities not visible in the frame.
        mask_first_detection_validate: If True, run VLM identity check before persisting a new entity_refs asset.
        keyframe_entity_min_confidence: Minimum VLM identity confidence (0–1) to run Module-3 inpaint per instruction.
        keyframe_edit_qa: If True, run one VLM quality check after the first keyframe inpaint;
            on failure retry once with QA ``retry_focus_prompt`` appended as
            ``EDITING OPERATIONS TO AVOID`` (same mechanism as Module-4 video edit QA).
        resume_from_checkpoints: If True, skip modules whose workspace result files already exist.
        enable_shot_vlm_analysis: If True, run per-shot VLM analysis after PySceneDetect.
        api: Model API credentials and model names.
    """

    workspace_dir: str
    source_video_path: str
    scene_detect_threshold: float = 22.0
    scene_detect_mode: str = "enhanced"
    scene_detect_min_scene_len: int = 5
    scene_adaptive_threshold: float = 2.5
    scene_min_content_val: float = 8.0
    max_inpaint_retries: int = 3
    max_ref_image_retries: int = 3
    extract_fps: float = 8.0
    max_propagation_concurrency: int = 2
    dev_mode: bool = False
    mask_visibility_prefilter: bool = True
    mask_first_detection_validate: bool = True
    keyframe_entity_min_confidence: float = 0.6
    keyframe_edit_qa: bool = True
    resume_from_checkpoints: bool = True
    enable_shot_vlm_analysis: bool = True
    scene_transition_trim_half_sec: float = 0.12
    scene_transition_max_zone_sec: float = 0.6
    scene_sub_cut_min_confidence: float = 0.75
    scene_sub_cut_min_duration_sec: float = 0.8
    entity_keyframe_min_confidence: float = 0.5
    entity_multiview_top_k: int = 6
    entity_multiview_num_candidates: int = 3
    video_chunk_max_sec: float = 10.0
    video_chunk_min_tail_sec: float = 4.0
    api: ModelApiConfig = field(default_factory=ModelApiConfig.from_env)

    def __post_init__(self) -> None:
        env_chunk = os.environ.get("VIDEO_CHUNK_MAX_SEC")
        if env_chunk:
            try:
                self.video_chunk_max_sec = float(env_chunk)
            except ValueError:
                pass
        env_tail = os.environ.get("VIDEO_CHUNK_MIN_TAIL_SEC")
        if env_tail:
            try:
                self.video_chunk_min_tail_sec = float(env_tail)
            except ValueError:
                pass
        env_scene_mode = os.environ.get("SCENE_DETECT_MODE")
        if env_scene_mode:
            self.scene_detect_mode = env_scene_mode.strip().lower()
        env_scene_thresh = os.environ.get("SCENE_DETECT_THRESHOLD")
        if env_scene_thresh:
            try:
                self.scene_detect_threshold = float(env_scene_thresh)
            except ValueError:
                pass
        env_scene_min_len = os.environ.get("SCENE_MIN_SCENE_LEN")
        if env_scene_min_len:
            try:
                self.scene_detect_min_scene_len = int(env_scene_min_len)
            except ValueError:
                pass
        env_adaptive_thresh = os.environ.get("SCENE_ADAPTIVE_THRESHOLD")
        if env_adaptive_thresh:
            try:
                self.scene_adaptive_threshold = float(env_adaptive_thresh)
            except ValueError:
                pass
        env_min_content = os.environ.get("SCENE_MIN_CONTENT_VAL")
        if env_min_content:
            try:
                self.scene_min_content_val = float(env_min_content)
            except ValueError:
                pass
        env_trim = os.environ.get("SCENE_TRANSITION_TRIM_HALF_SEC")
        if env_trim:
            try:
                self.scene_transition_trim_half_sec = float(env_trim)
            except ValueError:
                pass
        env_max_zone = os.environ.get("SCENE_TRANSITION_MAX_ZONE_SEC")
        if env_max_zone:
            try:
                self.scene_transition_max_zone_sec = float(env_max_zone)
            except ValueError:
                pass
        env_sub_conf = os.environ.get("SCENE_SUB_CUT_MIN_CONFIDENCE")
        if env_sub_conf:
            try:
                self.scene_sub_cut_min_confidence = float(env_sub_conf)
            except ValueError:
                pass
        env_sub_dur = os.environ.get("SCENE_SUB_CUT_MIN_DURATION_SEC")
        if env_sub_dur:
            try:
                self.scene_sub_cut_min_duration_sec = float(env_sub_dur)
            except ValueError:
                pass
        env_kf_conf = os.environ.get("ENTITY_KEYFRAME_MIN_CONFIDENCE")
        if env_kf_conf:
            try:
                self.entity_keyframe_min_confidence = float(env_kf_conf)
            except ValueError:
                pass
        env_mv_top_k = os.environ.get("ENTITY_MULTIVIEW_TOP_K")
        if env_mv_top_k:
            try:
                self.entity_multiview_top_k = int(env_mv_top_k)
            except ValueError:
                pass
        env_mv_cands = os.environ.get("ENTITY_MULTIVIEW_NUM_CANDIDATES")
        if env_mv_cands:
            try:
                self.entity_multiview_num_candidates = int(env_mv_cands)
            except ValueError:
                pass

        # ── Resolution mode auto-detection ──────────────────────────────
        # Probe the source video height and select "480p" or "720p" mode.
        # Explicit VIDEO_RESOLUTION env var always overrides auto-detection.
        env_resolution = _first_env("VIDEO_RESOLUTION").strip()
        if env_resolution:
            self.api.video_resolution = env_resolution.lower()
        elif self.source_video_path and os.path.exists(self.source_video_path):
            try:
                from video_editing_agent.utils.ffmpeg_utils import determine_resolution_mode
                self.api.video_resolution = determine_resolution_mode(
                    self.source_video_path,
                )
            except Exception:
                self.api.video_resolution = "720p"
        else:
            self.api.video_resolution = "720p"

    @property
    def entity_instru_path(self) -> str:
        """Path to module-1 output JSON."""
        return os.path.join(self.workspace_dir, "entity_instru.json")

    @property
    def time_instru_path(self) -> str:
        """Path to module-2 output JSON."""
        return os.path.join(self.workspace_dir, "time_instru.json")

    @property
    def scenes_dir(self) -> str:
        """Directory for per-scene frame and mask sequences."""
        return os.path.join(self.workspace_dir, "scenes")

    @property
    def shots_dir(self) -> str:
        """Directory for flat PySceneDetect shot clips (``shot_NN.mp4``)."""
        return os.path.join(self.workspace_dir, "shots")

    @property
    def shots_analysis_path(self) -> str:
        """Path to per-shot VLM analysis JSON."""
        return os.path.join(self.workspace_dir, "shots_analysis.json")

    @property
    def entity_keyframe_appearances_path(self) -> str:
        """Path to per-entity keyframe sighting records from Module 2."""
        return os.path.join(self.workspace_dir, "entity_keyframe_appearances.json")

    @property
    def keyframes_dir(self) -> str:
        """Directory for module-3 keyframe grid edits and related artifacts."""
        return os.path.join(self.workspace_dir, "keyframes")

    @property
    def edited_clips_dir(self) -> str:
        """Directory for module-4 per-scene edited video clips."""
        return os.path.join(self.workspace_dir, "edited_clips")

    @property
    def final_output_path(self) -> str:
        """Path to module-5 assembled final video."""
        return os.path.join(self.workspace_dir, "final_output.mp4")

    @property
    def ref_images_dir(self) -> str:
        """Directory for T2I reference images generated in module 1."""
        return os.path.join(self.workspace_dir, "ref_images")
