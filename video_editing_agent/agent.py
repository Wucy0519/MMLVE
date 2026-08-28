"""
Main orchestrator — wires pipeline modules sequentially.

Usage::

    agent = VideoEditingAgent(config)
    final_path = await agent.run(user_prompt="Change the red shirt to blue")
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from video_editing_agent.clients.base import ModelApiClientBase
from video_editing_agent.clients.model_client import ModelApiClient
from video_editing_agent.config import AgentConfig
from video_editing_agent.schemas.instructions import EntityInstructionSet
from video_editing_agent.schemas.scenes import TimeInstructionSet
from video_editing_agent.utils.workspace_checkpoints import (
    is_module1_complete,
    is_module2_complete,
    is_module4_complete,
    is_module5_complete,
    is_shot_analysis_complete,
    load_module1_result,
    load_module2_result,
    load_module4_result,
    load_physical_shots_from_workspace,
    save_module4_manifest,
)
from video_editing_agent.modules.entity_multiview_reference_builder import EntityMultiviewReferenceBuilder
from video_editing_agent.modules.instruction_parser import InstructionParser
from video_editing_agent.schemas.entity_keyframe_appearances import EntityKeyframeAppearanceSet
from video_editing_agent.modules.video_assembler import VideoAssembler

from video_editing_agent.modules.shot_analyzer import ShotAnalyzer
from video_editing_agent.modules.video_grounder import VideoGrounder
from video_editing_agent.modules.video_propagator import VideoPropagator

logger = logging.getLogger(__name__)


class VideoEditingAgent:
    """Multi-shot real-scene long video editing agent.

    Orchestrates modules in order:

        1. :class:`InstructionParser` — NL → ``entity_instru.json``
        2. :class:`VideoGrounder` — scenes + ``time_instru.json`` + shot analysis
        2.25. :class:`EntityMultiviewReferenceBuilder` — entity_refs edit examples
        3. :class:`VideoPropagator` — entity-ref-guided direct video editing + keyframe-grid QA
        4. :class:`VideoAssembler` — ffmpeg stitch → ``final_output.mp4``

    """

    def __init__(
        self,
        config: AgentConfig,
        api_client: Optional[ModelApiClientBase] = None,
    ) -> None:
        self.config = config
        self.api_client = api_client or ModelApiClient(
            config.api,
            dev_mode=config.dev_mode,
        )

        self._instruction_parser = InstructionParser(config, self.api_client)
        self._video_grounder = VideoGrounder(config, self.api_client)
        self._entity_multiview_reference_builder = EntityMultiviewReferenceBuilder(
            config, self.api_client,
        )
        self._video_propagator = VideoPropagator(

            config, self.api_client, max_concurrency=config.max_propagation_concurrency,
        )
        self._video_assembler = VideoAssembler(config)

    async def run(self, user_prompt: str) -> str:
        """Execute the full editing pipeline end-to-end."""
        try:
            self._ensure_workspace()
            logger.info("VideoEditingAgent.run started — workspace=%s", self.config.workspace_dir)

            # If final_output.mp4 already exists, skip the entire pipeline.
            final_path = self.config.final_output_path
            if (
                self.config.resume_from_checkpoints
                and os.path.exists(final_path)
                and os.path.getsize(final_path) > 0
            ):
                logger.info(
                    "Checkpoint: final_output.mp4 already exists (%s) — "
                    "skipping entire pipeline",
                    final_path,
                )
                return final_path

            if is_module1_complete(self.config, user_prompt):
                entity_instru = load_module1_result(self.config)
                logger.info(
                    "Checkpoint: skip module 1 — loaded %s (%d instructions)",
                    self.config.entity_instru_path,
                    len(entity_instru.instructions),
                )
            else:
                entity_instru = await self._instruction_parser.run(user_prompt)

            if is_module2_complete(self.config):
                time_instru = load_module2_result(self.config)
                logger.info(
                    "Checkpoint: skip module 2 — loaded %s (%d scenes)",
                    self.config.time_instru_path,
                    len(time_instru.scenes),
                )
                if (
                    self.config.enable_shot_vlm_analysis
                    and not is_shot_analysis_complete(
                        self.config,
                        len(load_physical_shots_from_workspace(self.config)),
                    )
                ):
                    physical_shots = load_physical_shots_from_workspace(self.config)
                    if physical_shots:
                        await ShotAnalyzer(self.config, self.api_client).run(physical_shots)
            else:
                time_instru = await self._video_grounder.run(entity_instru)

            appearance_set = self._load_entity_keyframe_appearances()
            await self._entity_multiview_reference_builder.run(
                entity_instru,
                appearance_set,
            )

            # Direct scene video editing — skip keyframe editing entirely.

            if is_module4_complete(self.config, time_instru):
                edited_clips = load_module4_result(self.config)
                logger.info(
                    "Checkpoint: skip module 4 — loaded %d edited clips",
                    len(edited_clips),
                )
            else:
                edited_clips = await self._video_propagator.run(
                    time_instru, entity_instru,
                )
                save_module4_manifest(
                    self.config,
                    edited_clips,
                    time_instru=time_instru,
                )

            if is_module5_complete(self.config):
                final_path = self.config.final_output_path
                logger.info("Checkpoint: skip module 5 — %s already exists", final_path)
            else:
                final_path = await asyncio.to_thread(
                    self._video_assembler.run,
                    time_instru,
                    edited_clips,
                )

            logger.info("VideoEditingAgent.run completed → %s", final_path)
            return final_path

        except Exception as exc:
            logger.error("VideoEditingAgent.run failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Video editing pipeline failed: {exc}") from exc

    def _ensure_workspace(self) -> None:
        for d in (
            self.config.workspace_dir,
            self.config.scenes_dir,
            self.config.shots_dir,
            self.config.keyframes_dir,
            self.config.edited_clips_dir,
            os.path.join(self.config.workspace_dir, "entity_refs"),
        ):
            os.makedirs(d, exist_ok=True)
        logger.info("Workspace ready: %s", self.config.workspace_dir)

    def _load_entity_keyframe_appearances(self) -> EntityKeyframeAppearanceSet:
        path = self.config.entity_keyframe_appearances_path
        if os.path.exists(path):
            return EntityKeyframeAppearanceSet.load(path)
        return EntityKeyframeAppearanceSet(
            source_video_path=self.config.source_video_path,
        )
