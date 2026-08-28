"""Helpers for direct per-scene video editing (no keyframe inpaint / mask path)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from video_editing_agent.config import AgentConfig
from video_editing_agent.schemas.instructions import EntityInstruction
from video_editing_agent.schemas.scenes import SceneClip
from video_editing_agent.schemas.shots import ShotAnalysis, ShotAnalysisSet
from video_editing_agent.utils.ffmpeg_utils import extract_frame_at, probe_duration
from video_editing_agent.utils.keyframe_manifest_utils import load_scene_keyframe_entries
from video_editing_agent.utils.mask_utils import (
    collect_keyframe_entity_ref_paths,
    entity_ref_canonical_path,
    entity_ref_multiview_edited_path,
    entity_ref_multiview_path,
    entity_ref_overlay_path,
)
from video_editing_agent.utils.multiview_ref_utils import (
    REFERENCE_GRID_COLS,
    build_keyframe_grid_image,
    save_keyframe_grid,
)


def load_shots_analysis(config: AgentConfig) -> Optional[ShotAnalysisSet]:
    path = config.shots_analysis_path
    if not os.path.exists(path):
        return None
    try:
        return ShotAnalysisSet.load(path)
    except Exception:
        return None


def shot_analysis_for_scene(
    analysis: Optional[ShotAnalysisSet],
    scene: SceneClip,
) -> Optional[ShotAnalysis]:
    if analysis is None:
        return None
    shot_id = scene.shot_id or scene.parent_shot_id or scene.scene_id
    for shot in analysis.shots:
        if shot.shot_id == shot_id or shot.scene_id == shot_id:
            return shot
    return None


def shot_analysis_context_dict(shot: Optional[ShotAnalysis]) -> Dict[str, Any]:
    if shot is None:
        return {}
    return {
        "shot_id": shot.shot_id,
        "plot_description": shot.plot_description,
        "pyscenedetect_start_sec": shot.pyscenedetect_start_sec,
        "pyscenedetect_end_sec": shot.pyscenedetect_end_sec,
        "keyframes": [
            {
                "description": kf.description,
                "role": kf.role,
                "timestamp_in_shot_sec": kf.timestamp_in_shot_sec,
                "timestamp_in_video_sec": kf.timestamp_in_video_sec,
            }
            for kf in shot.keyframes
        ],
    }


def entity_ref_dir(config: AgentConfig) -> str:
    return os.path.join(config.workspace_dir, "entity_refs")


def collect_entity_ref_assets(
    config: AgentConfig,
    instructions: List[EntityInstruction],
) -> Tuple[List[str], Dict[str, str]]:
    """Return (image_paths for VLM/video model, instruction_id → primary ref path)."""
    ref_dir = entity_ref_dir(config)
    image_paths: List[str] = []
    by_instruction: Dict[str, str] = {}

    for instr in instructions:
        paths = collect_keyframe_entity_ref_paths(
            ref_dir,
            instr.instruction_id,
            action=instr.action.value,
        )
        primary = (
            paths.get("multiview_edited")
            or paths.get("canonical")
            or paths.get("multiview")
            or paths.get("overlay")
            or entity_ref_multiview_edited_path(ref_dir, instr.instruction_id)
        )
        if not os.path.exists(primary):
            for candidate in (
                paths.get("multiview_edited"),
                paths.get("canonical"),
                paths.get("multiview"),
                paths.get("overlay"),
                entity_ref_canonical_path(ref_dir, instr.instruction_id),
                entity_ref_multiview_path(ref_dir, instr.instruction_id),
                entity_ref_overlay_path(ref_dir, instr.instruction_id),
            ):
                if candidate and os.path.exists(candidate):
                    primary = candidate
                    break
        if primary and os.path.exists(primary):
            by_instruction[instr.instruction_id] = primary
            if primary not in image_paths:
                image_paths.append(primary)
    return image_paths, by_instruction


def load_entity_instru_text(config: AgentConfig) -> str:
    path = config.entity_instru_path
    if not os.path.exists(path):
        return "{}"
    return Path(path).read_text(encoding="utf-8")


def sample_video_frames(
    video_path: str,
    output_dir: str,
    *,
    max_frames: int = 8,
) -> List[str]:
    """Extract evenly spaced frames from a clip for VLM context."""
    os.makedirs(output_dir, exist_ok=True)
    duration = max(probe_duration(video_path), 0.01)
    count = max(1, min(max_frames, int(duration * 2) or 1))
    if count == 1:
        times = [0.0]
    else:
        times = [duration * i / (count - 1) for i in range(count)]
        times = [min(t, max(0.0, duration - 0.05)) for t in times]

    paths: List[str] = []
    for idx, ts in enumerate(times, start=1):
        out = os.path.join(output_dir, f"sample_{idx:04d}.png")
        extract_frame_at(video_path, ts, out)
        paths.append(out)
    return paths


def original_keyframe_paths(scene: SceneClip) -> List[str]:
    entries = load_scene_keyframe_entries(scene)
    paths = [str(e.get("path", "")) for e in entries if e.get("path")]
    return [p for p in paths if os.path.exists(p)]


def extract_edited_keyframes_from_manifest(
    edited_video_path: str,
    scene: SceneClip,
    output_dir: str,
) -> List[str]:
    """Extract keyframes from edited video at scene-manifest timestamps."""
    os.makedirs(output_dir, exist_ok=True)
    entries = load_scene_keyframe_entries(scene)
    paths: List[str] = []
    for idx, entry in enumerate(entries, start=1):
        ts = float(entry.get("timestamp_in_scene_sec", 0.0) or 0.0)
        out = os.path.join(output_dir, f"edited_keyframe_{idx:04d}.png")
        extract_frame_at(edited_video_path, ts, out)
        paths.append(out)
    return paths


def build_keyframe_comparison_grids(
    original_paths: List[str],
    edited_paths: List[str],
    work_dir: str,
) -> Tuple[str, str]:
    """Save grid B (original) and grid A (edited); return (path_a, path_b)."""
    os.makedirs(work_dir, exist_ok=True)
    path_b = os.path.join(work_dir, "original_keyframes_grid.png")
    path_a = os.path.join(work_dir, "edited_keyframes_grid.png")
    save_keyframe_grid(original_paths, path_b, cols=REFERENCE_GRID_COLS)
    save_keyframe_grid(edited_paths, path_a, cols=REFERENCE_GRID_COLS)
    return path_a, path_b
