"""
Module 1 output schemas — entity-level edit instructions.

All serialized JSON files use English keys and enum values.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class EditAction(str, Enum):
    """Semantic edit operation on a tracked entity."""

    ADD = "add"
    DELETE = "delete"
    MODIFY = "modify"


@dataclass
class TimeCondition:
    """When an edit should apply — absolute time window or event-based scope.

    Attributes:
        condition_type: ``"absolute"`` (explicit edit time window only) or ``"event"``.
        start_sec: Inclusive start time in seconds (absolute mode only).
        end_sec: Exclusive end time in seconds (absolute mode only).
        event_description: Natural-language scope, e.g. ``"wherever the entity appears"``.
            Referential timestamps in the user prompt (``"appears around 30s"``) belong in
            ``subject_features`` for identification, not here as absolute ``start_sec``.
    """

    condition_type: str  # "absolute" | "event"
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    event_description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TimeCondition:
        """Deserialize from a dict."""
        return cls(**data)


@dataclass
class EntityInstruction:
    """Single structured edit command for one entity.

    Attributes:
        instruction_id: Unique id, e.g. ``"instr_001"``.
        entity_id: Stable entity identifier across shots, e.g. ``"entity_01"``.
        action: Legacy internal tag; defaults to ``modify``. Not written to ``entity_instru.json``.
        subject_features: Visual/semantic description of the target subject (include
            referential appearance-time cues when the user uses them to identify WHO).
        appearance_time_hint: Natural-language appearance-time identification cue
            (e.g. "appears in the first few frames", "appears around second 30") — NOT an edit window.
        edit_prompt: Natural-language inpainting / generation prompt.
        success_criteria_prompt: VLM QA prompt/checklist for keyframe success.
        time_condition: When the edit applies.
        target_instance_scope: ``"single"`` (default) — one specific tracked instance;
            ``"multiple"`` only when the user explicitly requests all matching instances.
        needs_ref_image: Whether to synthesize an isolated T2I reference asset.
        ref_subject: Isolated asset description for T2I (no scene/people context).
        ref_image_path: Absolute local path to reference image (may be generated).
    """

    instruction_id: str
    entity_id: str
    subject_features: str
    edit_prompt: str
    time_condition: TimeCondition
    appearance_time_hint: str = ""
    action: EditAction = EditAction.MODIFY
    success_criteria_prompt: str = ""
    target_instance_scope: str = "single"
    needs_ref_image: bool = False
    ref_subject: Optional[str] = None
    ref_image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict (one entry per entity)."""
        d = asdict(self)
        d.pop("action", None)
        d.pop("needs_ref_image", None)
        d.pop("ref_subject", None)
        d.pop("ref_image_path", None)
        if not (d.get("appearance_time_hint") or "").strip():
            d.pop("appearance_time_hint", None)
        d.pop("appearance_time_sec", None)
        scope = str(d.get("target_instance_scope", "single") or "single").strip().lower()
        d["target_instance_scope"] = "multiple" if scope == "multiple" else "single"
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EntityInstruction:
        """Deserialize from a dict."""
        data = dict(data)
        raw_action = data.pop("action", None)
        if raw_action:
            data["action"] = EditAction(raw_action)
        data["time_condition"] = TimeCondition.from_dict(data["time_condition"])
        data.setdefault("success_criteria_prompt", "")
        data.setdefault("needs_ref_image", False)
        data.setdefault("ref_subject", None)
        data.setdefault("appearance_time_hint", "")
        scope = str(data.pop("target_instance_scope", "single") or "single").strip().lower()
        data["target_instance_scope"] = "multiple" if scope == "multiple" else "single"
        data.pop("appearance_time_sec", None)
        return cls(**data)


@dataclass
class EntityInstructionSet:
    """Module 1 output — full instruction collection with metadata.

    Attributes:
        version: Schema version string.
        source_prompt: Original user natural-language prompt.
        rewritten_prompt: Gemini-clarified editing brief used for structured parsing.
        clarifications: Notes on ambiguities resolved during rewrite.
        instructions: Resolved, conflict-free instruction list.
    """

    version: str = "1.0"
    source_prompt: str = ""
    rewritten_prompt: str = ""
    clarifications: List[str] = field(default_factory=list)
    instructions: List[EntityInstruction] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        """Persist to ``entity_instru.json``."""
        payload = {
            "version": self.version,
            "source_prompt": self.source_prompt,
            "rewritten_prompt": self.rewritten_prompt,
            "clarifications": self.clarifications,
            "instructions": [i.to_dict() for i in self.instructions],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> EntityInstructionSet:
        """Load from ``entity_instru.json``."""
        data = json.loads(Path(path).read_text())
        return cls(
            version=data.get("version", "1.0"),
            source_prompt=data.get("source_prompt", ""),
            rewritten_prompt=data.get("rewritten_prompt", ""),
            clarifications=data.get("clarifications", []),
            instructions=[EntityInstruction.from_dict(i) for i in data["instructions"]],
        )
