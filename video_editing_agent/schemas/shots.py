"""
Shot-level VLM analysis output — one entry per PySceneDetect physical segment.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ShotKeyframe:
    """A narratively important moment inside a shot clip."""

    description: str
    timestamp_in_shot_sec: float
    timestamp_in_video_sec: float
    role: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ShotKeyframe:
        return cls(**data)


@dataclass
class ShotTimeRange:
    """A time span in the full source video."""

    start_sec: float
    end_sec: float
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ShotTimeRange:
        return cls(**data)


@dataclass
class TransitionZone:
    """Gradual transition region between sub-shots — excluded from scene clips."""

    start_sec_in_shot: float
    end_sec_in_shot: float
    start_sec_in_video: float
    end_sec_in_video: float
    transition_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TransitionZone:
        return cls(**data)


@dataclass
class UndetectedSubCut:
    """A sub-shot boundary PySceneDetect missed (e.g. gradual transition)."""

    start_sec_in_shot: float
    end_sec_in_shot: float
    start_sec_in_video: float
    end_sec_in_video: float
    transition_type: str = ""
    sub_plot_description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UndetectedSubCut:
        return cls(**data)


@dataclass
class ShotAnalysis:
    """VLM analysis for one physical shot clip."""

    shot_id: str
    scene_id: str
    clip_path: str
    pyscenedetect_start_sec: float
    pyscenedetect_end_sec: float
    plot_description: str
    keyframes: List[ShotKeyframe] = field(default_factory=list)
    has_undetected_sub_cuts: bool = False
    undetected_sub_cuts: List[UndetectedSubCut] = field(default_factory=list)
    transition_zones: List[TransitionZone] = field(default_factory=list)
    effective_time_ranges_in_video: List[ShotTimeRange] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "scene_id": self.scene_id,
            "clip_path": self.clip_path,
            "pyscenedetect_start_sec": self.pyscenedetect_start_sec,
            "pyscenedetect_end_sec": self.pyscenedetect_end_sec,
            "plot_description": self.plot_description,
            "keyframes": [k.to_dict() for k in self.keyframes],
            "has_undetected_sub_cuts": self.has_undetected_sub_cuts,
            "undetected_sub_cuts": [u.to_dict() for u in self.undetected_sub_cuts],
            "transition_zones": [t.to_dict() for t in self.transition_zones],
            "effective_time_ranges_in_video": [
                r.to_dict() for r in self.effective_time_ranges_in_video
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ShotAnalysis:
        return cls(
            shot_id=data["shot_id"],
            scene_id=data["scene_id"],
            clip_path=data["clip_path"],
            pyscenedetect_start_sec=float(data["pyscenedetect_start_sec"]),
            pyscenedetect_end_sec=float(data["pyscenedetect_end_sec"]),
            plot_description=data.get("plot_description", ""),
            keyframes=[ShotKeyframe.from_dict(k) for k in data.get("keyframes", [])],
            has_undetected_sub_cuts=bool(data.get("has_undetected_sub_cuts", False)),
            undetected_sub_cuts=[
                UndetectedSubCut.from_dict(u) for u in data.get("undetected_sub_cuts", [])
            ],
            transition_zones=[
                TransitionZone.from_dict(t) for t in data.get("transition_zones", [])
            ],
            effective_time_ranges_in_video=[
                ShotTimeRange.from_dict(r)
                for r in data.get("effective_time_ranges_in_video", [])
            ],
        )


@dataclass
class ShotAnalysisSet:
    """All shot analyses for a source video — ``shots_analysis.json``."""

    version: str = "1.0"
    source_video_path: str = ""
    shots: List[ShotAnalysis] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        payload = {
            "version": self.version,
            "source_video_path": self.source_video_path,
            "shots": [s.to_dict() for s in self.shots],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> ShotAnalysisSet:
        data = json.loads(Path(path).read_text())
        return cls(
            version=data.get("version", "1.0"),
            source_video_path=data.get("source_video_path", ""),
            shots=[ShotAnalysis.from_dict(s) for s in data.get("shots", [])],
        )
