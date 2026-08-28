"""Build scene clips from PySceneDetect shots (no secondary re-segmentation)."""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any, Dict, List

from video_editing_agent.config import AgentConfig
from video_editing_agent.schemas.scenes import SceneClip
from video_editing_agent.schemas.shots import (
    ShotAnalysis,
    ShotAnalysisSet,
    ShotKeyframe,
    ShotTimeRange,
)
from video_editing_agent.utils.ffmpeg_utils import (
    cut_video_segment,
    extract_audio,
    extract_frame_at,
    has_audio_stream,
    probe_duration,
)
from video_editing_agent.utils.video_chunk_utils import extract_last_frame
from video_editing_agent.utils.shot_analysis_utils import (
    ensure_closing_keyframe,
    ensure_opening_keyframe,
    keyframe_in_transition_zone,
    trim_settings_from_config,
)

logger = logging.getLogger(__name__)

KEYFRAMES_MANIFEST = "keyframes.json"
_CLOSING_KEYFRAME_ROLES = frozenset({"closing", "end", "ending", "final", "outro"})
_SCENE_END_TOLERANCE_SEC = 0.35


def _is_scene_end_keyframe(
    kf: ShotKeyframe,
    *,
    scene_start_sec: float,
    scene_end_sec: float,
    scene_start_in_shot: float,
    scene_end_in_shot: float,
) -> bool:
    """True when the keyframe is intended for the sub-scene ending."""
    role = (kf.role or "").strip().lower()
    if role in _CLOSING_KEYFRAME_ROLES:
        return True
    if kf.timestamp_in_video_sec >= scene_end_sec - 0.05:
        return True
    if kf.timestamp_in_shot_sec >= scene_end_in_shot - 0.05:
        return True
    if kf.timestamp_in_video_sec >= scene_start_sec and role == "transition":
        return kf.timestamp_in_video_sec >= scene_end_sec - _SCENE_END_TOLERANCE_SEC
    return False


def _should_use_shot_last_frame(
    kf: ShotKeyframe,
    *,
    ts_in_shot: float,
    shot_media_duration: float,
    scene_start_sec: float,
    scene_end_sec: float,
    scene_start_in_shot: float,
    scene_end_in_shot: float,
) -> bool:
    """Use shot tail frame when an end keyframe timestamp exceeds clip duration."""
    if shot_media_duration <= 0:
        return False
    if not _is_scene_end_keyframe(
        kf,
        scene_start_sec=scene_start_sec,
        scene_end_sec=scene_end_sec,
        scene_start_in_shot=scene_start_in_shot,
        scene_end_in_shot=scene_end_in_shot,
    ):
        return False
    return (
        kf.timestamp_in_shot_sec > shot_media_duration - 0.01
        or kf.timestamp_in_video_sec > scene_end_sec + 0.01
        or ts_in_shot >= shot_media_duration - 0.01
    )


def build_fallback_shot_analysis(
    physical_shots: List[SceneClip],
    *,
    source_video_path: str,
) -> ShotAnalysisSet:
    """One effective range per physical shot when VLM analysis is disabled."""
    shots: List[ShotAnalysis] = []
    for shot in physical_shots:
        shot_id = shot.shot_id or shot.scene_id
        clip_path = shot.shot_clip_path or ""
        start = float(shot.start_sec)
        end = float(shot.end_sec)
        anchor = 0.0
        shots.append(
            ShotAnalysis(
                shot_id=shot_id,
                scene_id=shot_id,
                clip_path=clip_path,
                pyscenedetect_start_sec=start,
                pyscenedetect_end_sec=end,
                plot_description="",
                keyframes=[
                    ShotKeyframe(
                        description="Opening frame of the shot — first frame of the clip.",
                        timestamp_in_shot_sec=anchor,
                        timestamp_in_video_sec=start + anchor,
                        role="opening",
                    ),
                    ShotKeyframe(
                        description="Closing frame of the shot — last frame of the clip.",
                        timestamp_in_shot_sec=max(0.0, end - start - 0.001),
                        timestamp_in_video_sec=end,
                        role="closing",
                    ),
                ],
                has_undetected_sub_cuts=False,
                effective_time_ranges_in_video=[
                    ShotTimeRange(start_sec=start, end_sec=end),
                ],
            )
        )
    return ShotAnalysisSet(source_video_path=source_video_path, shots=shots)


def _keyframes_for_range(
    shot: ShotAnalysis,
    start_sec: float,
    end_sec: float,
) -> List[ShotKeyframe]:
    """Pick shot keyframes inside the trimmed scene range, excluding transitions."""
    shot_start = float(shot.pyscenedetect_start_sec)
    scene_start_in_shot = start_sec - shot_start
    scene_end_in_shot = end_sec - shot_start

    picked: List[ShotKeyframe] = []
    seen: set[tuple[str, float]] = set()

    def _add(kf: ShotKeyframe) -> None:
        key = (kf.description, round(kf.timestamp_in_shot_sec, 3))
        if key in seen:
            return
        seen.add(key)
        picked.append(kf)

    for kf in shot.keyframes:
        if keyframe_in_transition_zone(kf.timestamp_in_shot_sec, shot.transition_zones):
            continue
        if start_sec <= kf.timestamp_in_video_sec < end_sec:
            _add(kf)
            continue
        if kf.timestamp_in_video_sec < start_sec:
            continue
        if _is_scene_end_keyframe(
            kf,
            scene_start_sec=start_sec,
            scene_end_sec=end_sec,
            scene_start_in_shot=scene_start_in_shot,
            scene_end_in_shot=scene_end_in_shot,
        ) and (
            kf.timestamp_in_video_sec >= end_sec - _SCENE_END_TOLERANCE_SEC
            or kf.timestamp_in_shot_sec >= scene_end_in_shot - _SCENE_END_TOLERANCE_SEC
        ):
            _add(kf)

    picked.sort(key=lambda k: k.timestamp_in_shot_sec)
    picked = ensure_opening_keyframe(
        picked,
        shot_start_sec=shot_start,
        plot_description=shot.plot_description,
    )
    picked = ensure_closing_keyframe(
        picked,
        shot_start_sec=shot_start,
        shot_end_sec=end_sec,
        plot_description=shot.plot_description,
    )
    return picked


def _resolve_shot_clip_path(config: AgentConfig, shot: ShotAnalysis) -> str:
    """Return the physical shot MP4 used for keyframe extraction."""
    if shot.clip_path and os.path.exists(shot.clip_path):
        return shot.clip_path
    fallback = os.path.join(config.shots_dir, f"{shot.shot_id}.mp4")
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(
        f"Shot clip not found for {shot.shot_id}: {shot.clip_path!r} or {fallback}"
    )


def _write_keyframes_manifest(
    keyframes_dir: str,
    entries: List[Dict[str, Any]],
) -> str:
    path = os.path.join(keyframes_dir, KEYFRAMES_MANIFEST)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"keyframes": entries}, fh, indent=2, ensure_ascii=False)
    return path


def build_scenes_from_pyscenedetect(
    config: AgentConfig,
    analysis: ShotAnalysisSet,
    *,
    replace_existing: bool = True,
) -> List[SceneClip]:
    """Cut one scene per PySceneDetect shot; extract keyframes from shot clips."""
    return _build_scenes_from_analysis(
        config,
        analysis,
        replace_existing=replace_existing,
        pyscenedetect_only=True,
    )


def resegment_scenes_from_analysis(
    config: AgentConfig,
    analysis: ShotAnalysisSet,
    *,
    replace_existing: bool = True,
) -> List[SceneClip]:
    """Legacy alias — scenes follow PySceneDetect only (no sub-cut splitting)."""
    return build_scenes_from_pyscenedetect(
        config,
        analysis,
        replace_existing=replace_existing,
    )


def _build_scenes_from_analysis(
    config: AgentConfig,
    analysis: ShotAnalysisSet,
    *,
    replace_existing: bool = True,
    pyscenedetect_only: bool = True,
) -> List[SceneClip]:
    """Cut the source video into scenes; extract keyframes from shot clips."""
    video_path = config.source_video_path
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Source video not found: {video_path}")

    if replace_existing and os.path.isdir(config.scenes_dir):
        shutil.rmtree(config.scenes_dir)
    os.makedirs(config.scenes_dir, exist_ok=True)

    scenes: List[SceneClip] = []
    scene_idx = 0

    for shot in analysis.shots:
        if pyscenedetect_only:
            ranges = [
                ShotTimeRange(
                    start_sec=float(shot.pyscenedetect_start_sec),
                    end_sec=float(shot.pyscenedetect_end_sec),
                    description=shot.plot_description,
                )
            ]
        else:
            from video_editing_agent.utils.shot_analysis_utils import resolve_shot_scene_ranges

            ranges = resolve_shot_scene_ranges(
                shot,
                trim_settings=trim_settings_from_config(config),
            )
            if not ranges:
                ranges = [
                    ShotTimeRange(
                        start_sec=shot.pyscenedetect_start_sec,
                        end_sec=shot.pyscenedetect_end_sec,
                        description=shot.plot_description,
                    )
                ]

        for time_range in ranges:
            scene_idx += 1
            scene_id = f"scene_{scene_idx:02d}"
            start_sec = float(time_range.start_sec)
            end_sec = float(time_range.end_sec)
            if end_sec <= start_sec:
                logger.warning(
                    "Skipping invalid scene range %s %.2f–%.2f",
                    scene_id,
                    start_sec,
                    end_sec,
                )
                continue

            scene_dir = os.path.join(config.scenes_dir, scene_id)
            keyframes_dir = os.path.join(scene_dir, "keyframes")
            clip_path = os.path.join(scene_dir, f"{scene_id}.mp4")
            audio_path = os.path.join(scene_dir, f"{scene_id}.aac")
            os.makedirs(keyframes_dir, exist_ok=True)

            cut_video_segment(video_path, start_sec, end_sec, clip_path, reencode=True)
            if has_audio_stream(clip_path):
                extract_audio(clip_path, audio_path)

            shot_clip_path = _resolve_shot_clip_path(config, shot)
            shot_start = float(shot.pyscenedetect_start_sec)
            shot_duration = max(
                0.0,
                float(shot.pyscenedetect_end_sec) - shot_start,
            )
            try:
                shot_media_duration = probe_duration(shot_clip_path)
            except Exception:
                shot_media_duration = shot_duration
            shot_media_duration = max(shot_media_duration, shot_duration, 0.01)
            scene_start_in_shot = start_sec - shot_start
            scene_end_in_shot = end_sec - shot_start
            scene_duration = max(0.0, end_sec - start_sec)

            shot_keyframes = _keyframes_for_range(shot, start_sec, end_sec)
            manifest_entries: List[Dict[str, Any]] = []
            keyframe_paths: List[str] = []

            for kf_idx, kf in enumerate(shot_keyframes, start=1):
                frame_path = os.path.join(keyframes_dir, f"keyframe_{kf_idx:04d}.png")
                ts_in_shot = _clamp_timestamp_in_shot(
                    kf.timestamp_in_shot_sec,
                    scene_start_in_shot,
                    scene_end_in_shot,
                    shot_media_duration,
                )
                use_last_frame = _should_use_shot_last_frame(
                    kf,
                    ts_in_shot=ts_in_shot,
                    shot_media_duration=shot_media_duration,
                    scene_start_sec=start_sec,
                    scene_end_sec=end_sec,
                    scene_start_in_shot=scene_start_in_shot,
                    scene_end_in_shot=scene_end_in_shot,
                )
                if use_last_frame:
                    extract_last_frame(shot_clip_path, frame_path)
                    ts_in_shot = max(0.0, shot_media_duration - 0.001)
                    logger.info(
                        "Scene %s keyframe %d: timestamp past shot end — using last frame",
                        scene_id,
                        kf_idx,
                    )
                else:
                    extract_frame_at(shot_clip_path, ts_in_shot, frame_path)
                ts_in_video = min(shot_start + ts_in_shot, end_sec - 0.001)
                rel_in_scene = min(max(0.0, ts_in_video - start_sec), max(0.0, scene_duration - 0.001))
                abs_path = os.path.abspath(frame_path)
                keyframe_paths.append(abs_path)
                manifest_entries.append(
                    {
                        "path": abs_path,
                        "filename": os.path.basename(frame_path),
                        "description": kf.description,
                        "role": kf.role,
                        "timestamp_in_shot_sec": round(ts_in_shot, 3),
                        "timestamp_in_scene_sec": round(rel_in_scene, 3),
                        "timestamp_in_video_sec": round(ts_in_video, 3),
                        "used_last_frame": use_last_frame,
                        "source_shot_id": shot.shot_id,
                        "source_shot_clip": os.path.abspath(shot_clip_path),
                    }
                )

            manifest_path = _write_keyframes_manifest(keyframes_dir, manifest_entries)
            first_frame_path = keyframe_paths[0]
            plot = (time_range.description or shot.plot_description or "").strip()

            scenes.append(
                SceneClip(
                    scene_id=scene_id,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    frame_dir=os.path.abspath(keyframes_dir),
                    first_frame_path=first_frame_path,
                    audio_path=os.path.abspath(audio_path)
                    if os.path.exists(audio_path)
                    else None,
                    shot_id=shot.shot_id,
                    shot_clip_path=shot.clip_path,
                    parent_shot_id=shot.shot_id,
                    keyframes_dir=os.path.abspath(keyframes_dir),
                    keyframe_paths=keyframe_paths,
                    keyframes_manifest_path=os.path.abspath(manifest_path),
                    plot_description=plot,
                )
            )
            logger.info(
                "Scene %s from PySceneDetect shot %s: %.2fs–%.2fs, %d keyframe(s)",
                scene_id,
                shot.shot_id,
                start_sec,
                end_sec,
                len(keyframe_paths),
            )

    return scenes


def _clamp_timestamp_in_shot(
    ts_in_shot: float,
    scene_start_in_shot: float,
    scene_end_in_shot: float,
    shot_duration: float,
) -> float:
    """Clamp a shot-relative timestamp to the scene sub-range inside the shot."""
    lower = max(0.0, scene_start_in_shot)
    upper = min(shot_duration, scene_end_in_shot) if shot_duration > 0 else scene_end_in_shot
    upper = max(lower, upper - 0.01)
    return max(lower, min(ts_in_shot, upper))
