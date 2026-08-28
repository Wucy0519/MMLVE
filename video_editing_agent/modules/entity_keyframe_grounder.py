"""
Per-scene keyframe entity grounding via VLM (replaces mask segmentation in Module 2).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from video_editing_agent.clients.base import ModelApiClientBase
from video_editing_agent.config import AgentConfig
from video_editing_agent.schemas.entity_keyframe_appearances import (
    EntityKeyframeAppearanceSet,
    EntityKeyframeRecord,
    KeyframeEntityAppearance,
)
from video_editing_agent.schemas.instructions import EntityInstruction, EntityInstructionSet
from video_editing_agent.schemas.scenes import SceneClip, TimeInstruction
from video_editing_agent.utils.keyframe_manifest_utils import load_scene_keyframe_entries
from video_editing_agent.utils.workspace_checkpoints import is_entity_keyframe_grounding_complete

logger = logging.getLogger(__name__)


class EntityKeyframeGrounder:
    """Detect edit-target entities on every scene keyframe; record per-entity sightings."""

    def __init__(self, config: AgentConfig, api_client: ModelApiClientBase) -> None:
        self.config = config
        self.api_client = api_client

    async def run(
        self,
        scenes: List[SceneClip],
        entity_instru: EntityInstructionSet,
    ) -> EntityKeyframeAppearanceSet:
        """Analyze all scene keyframes and write ``entity_keyframe_appearances.json``."""
        if is_entity_keyframe_grounding_complete(self.config, len(entity_instru.instructions)):
            logger.info(
                "Checkpoint: skip entity keyframe grounding — %s",
                self.config.entity_keyframe_appearances_path,
            )
            return EntityKeyframeAppearanceSet.load(
                self.config.entity_keyframe_appearances_path,
            )

        records: List[EntityKeyframeRecord] = []
        for instr in entity_instru.instructions:
            record = await self._analyze_entity(scenes, instr)
            records.append(record)

        result = EntityKeyframeAppearanceSet(
            source_video_path=self.config.source_video_path,
            entities=records,
        )
        result.save(self.config.entity_keyframe_appearances_path)
        logger.info(
            "EntityKeyframeGrounder done — %d entities → %s",
            len(records),
            self.config.entity_keyframe_appearances_path,
        )
        return result

    async def _analyze_entity(
        self,
        scenes: List[SceneClip],
        instr: EntityInstruction,
    ) -> EntityKeyframeRecord:
        appearances: List[KeyframeEntityAppearance] = []
        sorted_scenes = sorted(scenes, key=lambda s: s.start_sec)

        for scene in sorted_scenes:
            keyframe_entries = load_scene_keyframe_entries(scene)
            if not keyframe_entries:
                logger.warning(
                    "Scene %s has no keyframes — skip entity %s",
                    scene.scene_id,
                    instr.entity_id,
                )
                continue

            for entry in keyframe_entries:
                image_path = str(entry.get("path", "") or "")
                if not image_path or not os.path.exists(image_path):
                    continue

                meta = self._keyframe_metadata(entry, scene)
                raw = await self.api_client.analyze_entity_in_keyframe(
                    image_path=image_path,
                    entity_id=instr.entity_id,
                    instruction_id=instr.instruction_id,
                    subject_features=instr.subject_features,
                    appearance_time_hint=instr.appearance_time_hint or "",
                    edit_prompt=instr.edit_prompt,
                    keyframe_metadata=meta,
                    target_instance_scope=instr.target_instance_scope,
                )
                present = bool(raw.get("present", False))
                logger.info(
                    "Entity %s @ %s %s — present=%s conf=%.2f quality=%.0f view=%s",
                    instr.entity_id,
                    scene.scene_id,
                    os.path.basename(image_path),
                    present,
                    float(raw.get("confidence", 0.0) or 0.0),
                    float(raw.get("quality_score", 0.0) or 0.0),
                    str(raw.get("view_angle", "") or ""),
                )
                if not present:
                    continue

                appearances.append(self._to_appearance(scene.scene_id, image_path, meta, raw))

        return EntityKeyframeRecord(
            entity_id=instr.entity_id,
            instruction_id=instr.instruction_id,
            subject_features=instr.subject_features,
            appearance_time_hint=instr.appearance_time_hint or "",
            edit_prompt=instr.edit_prompt,
            appearances=appearances,
        )

    @staticmethod
    def _keyframe_metadata(entry: Dict[str, Any], scene: SceneClip) -> Dict[str, Any]:
        return {
            "scene_id": scene.scene_id,
            "timestamp_in_video_sec": float(
                entry.get("timestamp_in_video_sec", scene.start_sec) or scene.start_sec
            ),
            "timestamp_in_scene_sec": float(entry.get("timestamp_in_scene_sec", 0.0) or 0.0),
            "timestamp_in_shot_sec": float(entry.get("timestamp_in_shot_sec", 0.0) or 0.0),
            "keyframe_role": str(entry.get("role", "") or ""),
            "keyframe_description": str(entry.get("description", "") or ""),
        }

    @staticmethod
    def _to_appearance(
        scene_id: str,
        image_path: str,
        meta: Dict[str, Any],
        raw: Dict[str, Any],
    ) -> KeyframeEntityAppearance:
        return KeyframeEntityAppearance(
            scene_id=scene_id,
            keyframe_path=os.path.abspath(image_path),
            timestamp_in_video_sec=float(meta["timestamp_in_video_sec"]),
            timestamp_in_scene_sec=float(meta["timestamp_in_scene_sec"]),
            timestamp_in_shot_sec=float(meta.get("timestamp_in_shot_sec", 0.0)),
            keyframe_role=str(meta.get("keyframe_role", "")),
            keyframe_description=str(meta.get("keyframe_description", "")),
            scene_moment_description=str(raw.get("scene_moment_description", "") or "").strip(),
            present=bool(raw.get("present", False)),
            confidence=float(raw.get("confidence", 0.0) or 0.0),
            quality_score=float(raw.get("quality_score", 0.0) or 0.0),
            appearance_time_score=float(raw.get("appearance_time_score", 0.0) or 0.0),
            subject_features_score=float(raw.get("subject_features_score", 0.0) or 0.0),
            identification_clarity_score=float(
                raw.get("identification_clarity_score", 0.0) or 0.0
            ),
            view_angle=str(raw.get("view_angle", "") or "").strip(),
            visibility_state=str(raw.get("visibility_state", "") or "").strip(),
            pose_and_action=str(raw.get("pose_and_action", "") or "").strip(),
            location_description=str(raw.get("location_description", "") or "").strip(),
            reasoning=str(raw.get("reasoning", "") or "").strip(),
        )


def build_time_bindings_from_appearances(
    scenes: List[SceneClip],
    appearance_set: EntityKeyframeAppearanceSet,
    *,
    min_confidence: float,
) -> List[TimeInstruction]:
    """Bind instructions to scenes where the entity is confidently seen on a keyframe."""
    bindings: List[TimeInstruction] = []
    for scene in sorted(scenes, key=lambda s: s.start_sec):
        iids = appearance_set.instruction_ids_in_scene(
            scene.scene_id,
            min_confidence=min_confidence,
        )
        bindings.append(
            TimeInstruction(
                scene_id=scene.scene_id,
                start_sec=scene.start_sec,
                end_sec=scene.end_sec,
                instruction_ids=iids,
                requires_edit=bool(iids),
            )
        )
    return bindings
