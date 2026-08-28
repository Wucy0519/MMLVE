"""
Per-shot VLM analysis after PySceneDetect physical segmentation.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import List

from video_editing_agent.clients.base import ModelApiClientBase
from video_editing_agent.config import AgentConfig
from video_editing_agent.schemas.scenes import SceneClip
from video_editing_agent.schemas.shots import ShotAnalysis, ShotAnalysisSet
from video_editing_agent.utils.shot_analysis_utils import (
    normalize_shot_vlm_payload,
    trim_settings_from_config,
)

logger = logging.getLogger(__name__)


class ShotAnalyzer:
    """Run VLM analysis on each ``workspace/shots/shot_NN.mp4`` clip."""

    def __init__(self, config: AgentConfig, api_client: ModelApiClientBase) -> None:
        self.config = config
        self.api_client = api_client

    async def run(self, scenes: List[SceneClip]) -> ShotAnalysisSet:
        """Analyze every shot clip and persist ``shots_analysis.json``."""
        analyses: List[ShotAnalysis] = []
        total = len(scenes)

        for idx, scene in enumerate(scenes, start=1):
            shot_id = scene.shot_id or f"shot_{idx:02d}"
            clip_path = scene.shot_clip_path or os.path.join(
                self.config.shots_dir, f"{shot_id}.mp4",
            )
            if not os.path.exists(clip_path):
                scene_clip = os.path.join(
                    self.config.scenes_dir,
                    scene.scene_id,
                    f"{scene.scene_id}.mp4",
                )
                if os.path.exists(scene_clip):
                    os.makedirs(self.config.shots_dir, exist_ok=True)
                    shutil.copy2(scene_clip, clip_path)
                else:
                    raise FileNotFoundError(f"Shot clip not found: {clip_path}")

            logger.info(
                "ShotAnalyzer: VLM analysis %s (%s) %.2fs–%.2fs",
                shot_id,
                scene.scene_id,
                scene.start_sec,
                scene.end_sec,
            )
            raw = await self.api_client.analyze_shot_clip(
                clip_path=clip_path,
                shot_id=shot_id,
                scene_id=shot_id,
                shot_index=idx,
                shot_total=total,
                video_start_sec=scene.start_sec,
                video_end_sec=scene.end_sec,
            )
            analysis = normalize_shot_vlm_payload(
                raw,
                shot_id=shot_id,
                scene_id=shot_id,
                clip_path=os.path.abspath(clip_path),
                shot_start_sec=scene.start_sec,
                shot_end_sec=scene.end_sec,
                trim_settings=trim_settings_from_config(self.config),
            )
            analyses.append(analysis)

        result = ShotAnalysisSet(
            source_video_path=self.config.source_video_path,
            shots=analyses,
        )
        result.save(self.config.shots_analysis_path)
        logger.info(
            "ShotAnalyzer done — %d shots → %s",
            len(analyses),
            self.config.shots_analysis_path,
        )
        return result
