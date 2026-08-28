"""Per-entity keyframe appearance records from Module 2 VLM grounding."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class KeyframeEntityAppearance:
    """One confirmed entity sighting on a single scene keyframe (present=true only)."""

    scene_id: str
    keyframe_path: str
    timestamp_in_video_sec: float
    timestamp_in_scene_sec: float = 0.0
    timestamp_in_shot_sec: float = 0.0
    keyframe_role: str = ""
    keyframe_description: str = ""
    scene_moment_description: str = ""
    present: bool = False
    confidence: float = 0.0
    quality_score: float = 0.0
    appearance_time_score: float = 0.0
    subject_features_score: float = 0.0
    identification_clarity_score: float = 0.0
    view_angle: str = ""
    visibility_state: str = ""
    pose_and_action: str = ""
    location_description: str = ""
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KeyframeEntityAppearance:
        allowed = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class EntityKeyframeRecord:
    """All keyframe sightings for one edit target entity — one row per entity."""

    entity_id: str
    instruction_id: str
    subject_features: str
    appearance_time_hint: str = ""
    edit_prompt: str = ""
    appearances: List[KeyframeEntityAppearance] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "instruction_id": self.instruction_id,
            "subject_features": self.subject_features,
            "appearance_time_hint": self.appearance_time_hint,
            "edit_prompt": self.edit_prompt,
            "appearances": [a.to_dict() for a in self.appearances],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EntityKeyframeRecord:
        return cls(
            entity_id=data["entity_id"],
            instruction_id=data["instruction_id"],
            subject_features=data.get("subject_features", ""),
            appearance_time_hint=data.get("appearance_time_hint", ""),
            edit_prompt=data.get("edit_prompt", ""),
            appearances=[
                KeyframeEntityAppearance.from_dict(a) for a in data.get("appearances", [])
            ],
        )


@dataclass
class EntityKeyframeAppearanceSet:
    """Workspace output — ``entity_keyframe_appearances.json``."""

    version: str = "1.0"
    source_video_path: str = ""
    entities: List[EntityKeyframeRecord] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        payload = {
            "version": self.version,
            "source_video_path": self.source_video_path,
            "entities": [e.to_dict() for e in self.entities],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> EntityKeyframeAppearanceSet:
        data = json.loads(Path(path).read_text())
        return cls(
            version=data.get("version", "1.0"),
            source_video_path=data.get("source_video_path", ""),
            entities=[EntityKeyframeRecord.from_dict(e) for e in data.get("entities", [])],
        )

    def appearances_for_scene(self, scene_id: str) -> List[KeyframeEntityAppearance]:
        out: List[KeyframeEntityAppearance] = []
        for entity in self.entities:
            for app in entity.appearances:
                if app.scene_id == scene_id:
                    out.append(app)
        return out

    def instruction_ids_in_scene(
        self,
        scene_id: str,
        *,
        min_confidence: float = 0.5,
    ) -> List[str]:
        """Instruction ids with a confident present sighting in ``scene_id``."""
        found: List[str] = []
        for entity in self.entities:
            for app in entity.appearances:
                if app.scene_id != scene_id:
                    continue
                if app.present and app.confidence >= min_confidence:
                    if entity.instruction_id not in found:
                        found.append(entity.instruction_id)
                    break
        return found

    def record_for_instruction(self, instruction_id: str) -> Optional[EntityKeyframeRecord]:
        """Return the entity record for one instruction id, if any."""
        for entity in self.entities:
            if entity.instruction_id == instruction_id:
                return entity
        return None
