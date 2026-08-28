"""
Module 2 output schemas — scene structure and time-grounded instructions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SceneClip:
    """A physically detected shot from PySceneDetect.

    Attributes:
        scene_id: Identifier, e.g. ``"scene_01"``.
        start_sec: Inclusive start timestamp in seconds.
        end_sec: Exclusive end timestamp in seconds.
        frame_dir: Directory containing extracted frames for this scene.
        first_frame_path: Absolute path to the scene's first frame image.
        last_frame_path: Legacy optional field (no longer extracted or edited).
        audio_path: Absolute path to extracted scene audio (AAC).
        shot_id: Flat shot identifier, e.g. ``"shot_01"`` (same index as ``scene_id``).
        shot_clip_path: Absolute path to ``workspace/shots/shot_NN.mp4``.
        parent_shot_id: Source PySceneDetect shot when this scene is a sub-cut.
        keyframes_dir: ``scenes/scene_NN/keyframes`` with VLM-selected frames.
        keyframe_paths: Ordered absolute paths to extracted keyframe PNGs.
        keyframes_manifest_path: ``keyframes.json`` metadata sidecar.
        plot_description: VLM plot summary for this scene segment.
    """

    scene_id: str
    start_sec: float
    end_sec: float
    frame_dir: Optional[str] = None
    first_frame_path: Optional[str] = None
    last_frame_path: Optional[str] = None
    audio_path: Optional[str] = None
    shot_id: Optional[str] = None
    shot_clip_path: Optional[str] = None
    parent_shot_id: Optional[str] = None
    keyframes_dir: Optional[str] = None
    keyframe_paths: List[str] = field(default_factory=list)
    keyframes_manifest_path: Optional[str] = None
    plot_description: str = ""

    @property
    def duration_sec(self) -> float:
        """Scene duration in seconds."""
        return self.end_sec - self.start_sec

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SceneClip:
        """Deserialize from a dict."""
        data = dict(data)
        data.setdefault("shot_id", None)
        data.setdefault("shot_clip_path", None)
        data.setdefault("parent_shot_id", None)
        data.setdefault("keyframes_dir", None)
        data.setdefault("keyframe_paths", [])
        data.setdefault("keyframes_manifest_path", None)
        data.setdefault("plot_description", "")
        return cls(**data)


@dataclass
class SceneList:
    """Collection of all physically segmented scenes."""

    source_video_path: str
    scenes: List[SceneClip] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        """Persist scene list JSON."""
        payload = {
            "source_video_path": self.source_video_path,
            "scenes": [s.to_dict() for s in self.scenes],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> SceneList:
        """Load scene list JSON."""
        data = json.loads(Path(path).read_text())
        return cls(
            source_video_path=data["source_video_path"],
            scenes=[SceneClip.from_dict(s) for s in data["scenes"]],
        )


@dataclass
class SpatialMaskAsset:
    """Multi-channel mask sequence for one scene.

    Attributes:
        scene_id: Associated scene identifier.
        mask_dir: Directory of per-frame multi-color mask PNGs.
        entity_color_map: Maps ``entity_id`` to mask color hex (internal).
        entity_color_name_map: Maps ``entity_id`` to color name for prompts (e.g. ``"red"``).
    """

    scene_id: str
    mask_dir: str
    entity_color_map: Dict[str, str] = field(default_factory=dict)
    entity_color_name_map: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SpatialMaskAsset:
        """Deserialize from a dict."""
        return cls(**data)


@dataclass
class TimeInstruction:
    """Binds entity instructions to a specific scene and time range.

    Attributes:
        scene_id: Matched scene identifier.
        start_sec: Effective edit start within the source video.
        end_sec: Effective edit end within the source video.
        instruction_ids: List of ``instruction_id`` values bound to this scene.
        mask_asset: Optional spatial mask metadata for this scene.
        requires_edit: Whether this scene needs keyframe + propagation editing.
    """

    scene_id: str
    start_sec: float
    end_sec: float
    instruction_ids: List[str] = field(default_factory=list)
    mask_asset: Optional[SpatialMaskAsset] = None
    requires_edit: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d = asdict(self)
        if self.mask_asset is not None:
            d["mask_asset"] = self.mask_asset.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TimeInstruction:
        """Deserialize from a dict."""
        data = dict(data)
        mask = data.pop("mask_asset", None)
        obj = cls(**data)
        if mask is not None:
            obj.mask_asset = SpatialMaskAsset.from_dict(mask)
        return obj


@dataclass
class TimeInstructionSet:
    """Module 2 output — ``time_instru.json``."""

    version: str = "1.0"
    source_video_path: str = ""
    scenes: List[SceneClip] = field(default_factory=list)
    time_instructions: List[TimeInstruction] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        """Persist to ``time_instru.json``."""
        payload = {
            "version": self.version,
            "source_video_path": self.source_video_path,
            "scenes": [s.to_dict() for s in self.scenes],
            "time_instructions": [t.to_dict() for t in self.time_instructions],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> TimeInstructionSet:
        """Load from ``time_instru.json``."""
        data = json.loads(Path(path).read_text())
        return cls(
            version=data.get("version", "1.0"),
            source_video_path=data.get("source_video_path", ""),
            scenes=[SceneClip.from_dict(s) for s in data.get("scenes", [])],
            time_instructions=[
                TimeInstruction.from_dict(t) for t in data.get("time_instructions", [])
            ],
        )
