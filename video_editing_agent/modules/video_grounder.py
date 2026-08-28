"""
Module 2 — Video structuring and spatio-temporal grounding.

Answers *when* (which scene) edits apply and records entity sightings on keyframes.
"""

from __future__ import annotations

import logging
import os
from typing import List

from video_editing_agent.utils.scene_detection import detect_scene_segments

from video_editing_agent.clients.base import ModelApiClientBase
from video_editing_agent.config import AgentConfig
from video_editing_agent.modules.entity_keyframe_grounder import (
    EntityKeyframeGrounder,
    build_time_bindings_from_appearances,
)
from video_editing_agent.modules.scene_resegmenter import (
    build_fallback_shot_analysis,
    build_scenes_from_pyscenedetect,
)
from video_editing_agent.modules.shot_analyzer import ShotAnalyzer
from video_editing_agent.schemas.instructions import EntityInstructionSet
from video_editing_agent.schemas.scenes import SceneClip, TimeInstructionSet
from video_editing_agent.schemas.shots import ShotAnalysisSet
from video_editing_agent.utils.workspace_checkpoints import (
    is_shot_analysis_complete,
    load_shot_analysis_result,
)
from video_editing_agent.utils.ffmpeg_utils import (
    cut_video_segment,
    prune_scene_frames_keep_first,
)

logger = logging.getLogger(__name__)


class VideoGrounder:
    """Video Grounding Agent (Module 2).

    Pipeline steps:
        1. Physical shot segmentation via PySceneDetect → ``shots/shot_NN.mp4``.
        2. Per-shot VLM analysis → ``shots_analysis.json``.
        3. One scene per PySceneDetect shot → ``scenes/scene_NN/`` + ``keyframes/``.
        4. Per-keyframe VLM entity detection → ``entity_keyframe_appearances.json``.
    """

    def __init__(
        self,
        config: AgentConfig,
        api_client: ModelApiClientBase,
    ) -> None:
        self.config = config
        self.api_client = api_client
        self._shot_analyzer = ShotAnalyzer(config, api_client)
        self._entity_keyframe_grounder = EntityKeyframeGrounder(config, api_client)

    async def run(
        self,
        entity_instru: EntityInstructionSet,
    ) -> TimeInstructionSet:
        """Execute video grounding pipeline."""
        try:
            logger.info("VideoGrounder.run started")

            physical_shots = self._detect_physical_shots()
            analysis = await self._analyze_shots_if_needed(physical_shots)
            scenes = self._build_scenes_from_pyscenedetect(physical_shots, analysis)

            if not entity_instru.instructions:
                logger.warning("No instructions — scenes only")
                result = TimeInstructionSet(
                    source_video_path=self.config.source_video_path,
                    scenes=scenes,
                    time_instructions=[],
                )
                self._prune_scene_frames(scenes)
                result.save(self.config.time_instru_path)
                return result

            appearance_set = await self._entity_keyframe_grounder.run(
                scenes,
                entity_instru,
            )
            bindings = build_time_bindings_from_appearances(
                scenes,
                appearance_set,
                min_confidence=self.config.entity_keyframe_min_confidence,
            )
            time_instru = TimeInstructionSet(time_instructions=bindings)
            time_instru.source_video_path = self.config.source_video_path
            time_instru.scenes = scenes
            self._prune_scene_frames(scenes)
            time_instru.save(self.config.time_instru_path)

            logger.info(
                "VideoGrounder.run done — %d scenes, %d bindings → %s",
                len(scenes),
                len(time_instru.time_instructions),
                self.config.time_instru_path,
            )
            return time_instru

        except Exception as exc:
            logger.error("VideoGrounder.run failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Video grounding failed: {exc}") from exc

    def _detect_physical_shots(self) -> List[SceneClip]:
        """PySceneDetect hard cuts → ``workspace/shots/shot_NN.mp4`` (no scenes/ yet)."""
        video_path = self.config.source_video_path
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Source video not found: {video_path}")

        logger.info(
            "PySceneDetect: %s mode=%s threshold=%.1f min_scene_len=%d",
            video_path,
            self.config.scene_detect_mode,
            self.config.scene_detect_threshold,
            self.config.scene_detect_min_scene_len,
        )

        os.makedirs(self.config.shots_dir, exist_ok=True)

        segments = detect_scene_segments(
            video_path,
            mode=self.config.scene_detect_mode,
            content_threshold=self.config.scene_detect_threshold,
            adaptive_threshold=self.config.scene_adaptive_threshold,
            min_content_val=self.config.scene_min_content_val,
            min_scene_len=self.config.scene_detect_min_scene_len,
        )

        if not segments:
            from video_editing_agent.utils.ffmpeg_utils import probe_duration
            dur = probe_duration(video_path)
            return [self._cut_physical_shot(1, 0.0, dur, video_path)]

        return [
            self._cut_physical_shot(idx, start_sec, end_sec, video_path)
            for idx, (start_sec, end_sec) in enumerate(segments, start=1)
        ]

    def _cut_physical_shot(
        self,
        idx: int,
        start_sec: float,
        end_sec: float,
        video_path: str,
    ) -> SceneClip:
        """Cut one PySceneDetect segment to ``shots/shot_NN.mp4``."""
        shot_id = f"shot_{idx:02d}"
        shot_path = os.path.join(self.config.shots_dir, f"{shot_id}.mp4")
        cut_video_segment(video_path, start_sec, end_sec, shot_path, reencode=True)
        logger.info(
            "Physical shot %s: %.2fs–%.2fs → %s",
            shot_id,
            start_sec,
            end_sec,
            shot_path,
        )
        return SceneClip(
            scene_id=shot_id,
            start_sec=start_sec,
            end_sec=end_sec,
            shot_id=shot_id,
            shot_clip_path=os.path.abspath(shot_path),
        )

    def _build_scenes_from_pyscenedetect(
        self,
        physical_shots: List[SceneClip],
        analysis: ShotAnalysisSet | None,
    ) -> List[SceneClip]:
        """Build one ``scenes/scene_NN`` per PySceneDetect shot (no sub-cut splitting)."""
        if analysis is None:
            analysis = build_fallback_shot_analysis(
                physical_shots,
                source_video_path=self.config.source_video_path,
            )
        return build_scenes_from_pyscenedetect(self.config, analysis)

    def _detect_scenes(self) -> List[SceneClip]:
        """Legacy alias — physical shots only."""
        return self._detect_physical_shots()

    async def _analyze_shots_if_needed(
        self,
        scenes: List[SceneClip],
    ) -> ShotAnalysisSet | None:
        """Run per-shot VLM analysis unless disabled or checkpointed."""
        if not self.config.enable_shot_vlm_analysis:
            logger.info("Shot VLM analysis disabled — skipping")
            return None
        if is_shot_analysis_complete(self.config, len(scenes)):
            logger.info(
                "Checkpoint: skip shot VLM analysis — %s",
                self.config.shots_analysis_path,
            )
            return load_shot_analysis_result(self.config)
        return await self._shot_analyzer.run(scenes)

    def _prune_scene_frames(self, scenes: List[SceneClip]) -> None:
        """Keep only legacy single-frame anchors; preserve multi-keyframe scenes."""
        total_removed = 0
        for scene in scenes:
            if scene.keyframes_dir or (scene.keyframe_paths and len(scene.keyframe_paths) > 1):
                continue
            if not scene.frame_dir or not scene.first_frame_path:
                continue
            removed = prune_scene_frames_keep_first(
                scene.frame_dir,
                scene.first_frame_path,
            )
            total_removed += removed
        if total_removed:
            logger.info(
                "Pruned %d non-anchor scene frame(s) under %s",
                total_removed,
                self.config.scenes_dir,
            )
