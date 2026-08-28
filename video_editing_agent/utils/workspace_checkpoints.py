"""Detect completed pipeline modules from workspace artifacts for resume."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from video_editing_agent.config import AgentConfig
from video_editing_agent.schemas.instructions import EntityInstruction, EntityInstructionSet
from video_editing_agent.schemas.scenes import SceneClip, TimeInstruction, TimeInstructionSet
from video_editing_agent.schemas.shots import ShotAnalysisSet
from video_editing_agent.utils.mask_utils import (
    entity_mask_has_content,
    entity_ref_canonical_path,
    entity_ref_mask_path,
    entity_ref_multiview_path,
    entity_ref_src_path,
    mask_has_content,
)

logger = logging.getLogger(__name__)

CANONICAL_EDITS_FILENAME = "canonical_edits.json"
EDITED_KEYFRAMES_FILENAME = "edited_keyframes.json"
SCENE_KEYFRAME_GRID_EDITS_FILENAME = "scene_keyframe_grid_edits.json"
EDITED_CLIPS_MANIFEST_FILENAME = "edited_clips.json"


def canonical_edits_path(config: AgentConfig) -> str:
    """Path to Module 2.5 completion manifest."""
    return os.path.join(config.workspace_dir, "entity_refs", CANONICAL_EDITS_FILENAME)


def edited_keyframes_path(config: AgentConfig) -> str:
    """Path to Module 3 completion manifest."""
    return os.path.join(config.keyframes_dir, EDITED_KEYFRAMES_FILENAME)


def scene_keyframe_grid_edits_path(config: AgentConfig) -> str:
    """Path to Module 3 keyframe-strip grid edit manifest."""
    return os.path.join(config.keyframes_dir, SCENE_KEYFRAME_GRID_EDITS_FILENAME)


def edited_clips_manifest_path(config: AgentConfig) -> str:
    """Path to Module 4 completion manifest."""
    return os.path.join(config.edited_clips_dir, EDITED_CLIPS_MANIFEST_FILENAME)


def edited_clip_path(config: AgentConfig, scene_id: str) -> str:
    """Standard output path for a propagated edited scene clip."""
    return os.path.join(config.edited_clips_dir, f"{scene_id}_edited.mp4")


def edited_clip_is_valid(path: str) -> bool:
    """Return True when an edited clip file exists and is non-empty."""
    return bool(path and os.path.exists(path) and os.path.getsize(path) > 0)


def module4_scene_is_done(scene_id: str, edited_clips: Dict[str, str]) -> bool:
    """True when a scene already has a valid edited clip on disk."""
    return edited_clip_is_valid(edited_clips.get(scene_id, ""))


def load_module4_checkpoint(config: AgentConfig) -> Dict[str, str]:
    """Load per-scene edited clips from manifest and edited_clips directory."""
    clips: Dict[str, str] = {}
    data = _load_json(edited_clips_manifest_path(config)) or {}
    for scene_id, path in (data.get("clips") or {}).items():
        path_str = str(path)
        if edited_clip_is_valid(path_str):
            clips[str(scene_id)] = path_str

    edited_dir = config.edited_clips_dir
    if os.path.isdir(edited_dir):
        for name in os.listdir(edited_dir):
            if not name.endswith("_edited.mp4"):
                continue
            scene_id = name[: -len("_edited.mp4")]
            path = os.path.join(edited_dir, name)
            if edited_clip_is_valid(path):
                clips.setdefault(scene_id, path)
    return clips


def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Invalid checkpoint file %s: %s", path, exc)
        return None


def _manifest_completed(path: str) -> bool:
    data = _load_json(path)
    return bool(data and data.get("completed"))


def _video_path_matches(config: AgentConfig, recorded_path: str) -> bool:
    if not recorded_path:
        return True
    return os.path.abspath(recorded_path) == os.path.abspath(config.source_video_path)


KeyframeAnchors = Dict[str, str]


def _normalize_keyframe_entry(value: object) -> KeyframeAnchors:
    """Parse manifest entry (legacy string or {begin,end} dict)."""
    if isinstance(value, dict):
        return {
            str(k): str(v)
            for k, v in value.items()
            if v and str(k) in ("begin", "end")
        }
    if isinstance(value, str) and value:
        return {"begin": value}
    return {}


def load_module3_checkpoint(
    config: AgentConfig,
) -> tuple[Dict[str, KeyframeAnchors], Dict[str, Dict[str, object]]]:
    """Load module 3 keyframes and recorded skipped scenes from workspace."""
    data = _load_json(edited_keyframes_path(config)) or {}
    keyframes_raw = data.get("keyframes") or {}
    skipped_raw = data.get("skipped_scenes") or {}
    keyframes = {
        str(scene_id): _normalize_keyframe_entry(entry)
        for scene_id, entry in keyframes_raw.items()
        if _normalize_keyframe_entry(entry)
    }
    skipped: Dict[str, Dict[str, object]] = {}
    for scene_id, entry in skipped_raw.items():
        if isinstance(entry, dict):
            skipped[str(scene_id)] = dict(entry)
        elif entry:
            skipped[str(scene_id)] = {"reason": str(entry)}
    return keyframes, skipped


def module3_scene_is_done(
    scene_id: str,
    keyframes: Dict[str, KeyframeAnchors],
    skipped_scenes: Dict[str, Dict[str, object]],
) -> bool:
    """True when a scene has a begin keyframe or a recorded skip."""
    if scene_id in skipped_scenes:
        return True
    anchors = keyframes.get(scene_id) or {}
    begin = anchors.get("begin", "")
    return bool(begin and os.path.exists(begin))


def reconcile_binding_requires_edit(
    binding: TimeInstruction,
    instr_by_id: Dict[str, EntityInstruction],
    scenes_dir: str,
) -> bool:
    """Re-evaluate requires_edit from bound instruction_ids (mask optional).

    Module 3 runs VLM entity detection before keyframe edit; scenes without
    ``mask_0000.png`` are not excluded here.
    """
    del scenes_dir  # retained for call-site compatibility
    has_edit = False
    for iid in binding.instruction_ids:
        if iid in instr_by_id:
            has_edit = True
            break
    changed = binding.requires_edit != has_edit
    binding.requires_edit = has_edit
    return changed


def reconcile_time_instruction_requires_edit(
    time_instru: TimeInstructionSet,
    entity_instru: EntityInstructionSet,
    scenes_dir: str,
) -> bool:
    """Refresh requires_edit for all scene bindings from current masks."""
    instr_by_id = {i.instruction_id: i for i in entity_instru.instructions}
    changed = False
    for binding in time_instru.time_instructions:
        if reconcile_binding_requires_edit(binding, instr_by_id, scenes_dir):
            changed = True
    return changed


def persist_module3_scene_skip(
    config: AgentConfig,
    time_instru: TimeInstructionSet,
    scene_id: str,
    reason: str,
    skipped_scenes: Dict[str, Dict[str, object]],
) -> None:
    """Record a skipped scene in the module-3 manifest (does not disable future retries)."""
    skipped_scenes[scene_id] = {
        "reason": reason,
    }
    logger.info(
        "Scene %s marked skipped for module 3 (%s)",
        scene_id,
        reason,
    )


def load_module3_keyframe_grid_checkpoint(
    config: AgentConfig,
) -> tuple[Dict[str, str], Dict[str, Dict[str, object]]]:
    """Load scene_id → edited keyframe grid path and skipped scenes."""
    from video_editing_agent.utils.scene_keyframe_grid_utils import (
        edited_keyframe_grid_path,
    )

    data = _load_json(scene_keyframe_grid_edits_path(config)) or {}
    grids_raw = data.get("scene_grids") or {}
    skipped_raw = data.get("skipped_scenes") or {}
    grids: Dict[str, str] = {}
    for scene_id, path in grids_raw.items():
        sid = str(scene_id)
        path_str = str(path)
        if path_str and os.path.exists(path_str):
            grids[sid] = path_str
            continue
        fallback = edited_keyframe_grid_path(config.keyframes_dir, sid)
        if os.path.exists(fallback):
            grids[sid] = fallback
    skipped: Dict[str, Dict[str, object]] = {}
    for scene_id, entry in skipped_raw.items():
        if isinstance(entry, dict):
            skipped[str(scene_id)] = dict(entry)
        elif entry:
            skipped[str(scene_id)] = {"reason": str(entry)}
    return grids, skipped


def module3_keyframe_grid_scene_is_done(
    scene_id: str,
    scene_grids: Dict[str, str],
    skipped_scenes: Dict[str, Dict[str, object]],
) -> bool:
    del skipped_scenes
    path = scene_grids.get(scene_id, "")
    return bool(path and os.path.exists(path))


def module3_keyframe_grid_scene_is_done_with_config(
    config: AgentConfig,
    scene_id: str,
    scene_grids: Dict[str, str],
    skipped_scenes: Dict[str, Dict[str, object]],
) -> bool:
    """Like ``module3_keyframe_grid_scene_is_done`` but falls back to ``keyframes/`` layout."""
    if module3_keyframe_grid_scene_is_done(scene_id, scene_grids, skipped_scenes):
        return True
    from video_editing_agent.utils.scene_keyframe_grid_utils import (
        edited_keyframe_grid_path,
    )

    return os.path.exists(edited_keyframe_grid_path(config.keyframes_dir, scene_id))


def persist_module3_keyframe_grid_skip(
    config: AgentConfig,
    scene_id: str,
    reason: str,
    skipped_scenes: Dict[str, Dict[str, object]],
) -> None:
    del config
    skipped_scenes[scene_id] = {"reason": reason}
    logger.info("Scene %s skipped for keyframe grid edit (%s)", scene_id, reason)


def is_module3_keyframe_grid_complete(
    config: AgentConfig,
    time_instru: TimeInstructionSet,
) -> bool:
    """Module 3 done when every physical scene has an edited keyframe grid."""
    if not config.resume_from_checkpoints:
        return False
    grids, skipped = load_module3_keyframe_grid_checkpoint(config)
    pending: list[str] = []
    for scene in time_instru.scenes:
        if module3_keyframe_grid_scene_is_done_with_config(
            config, scene.scene_id, grids, skipped
        ):
            continue
        pending.append(scene.scene_id)
    if pending:
        logger.info(
            "Module 3 keyframe grid incomplete — pending: %s",
            ", ".join(pending),
        )
        return False
    return True


def load_module3_keyframe_grid_result(config: AgentConfig) -> Dict[str, str]:
    grids, skipped = load_module3_keyframe_grid_checkpoint(config)
    return {
        scene_id: path
        for scene_id, path in grids.items()
        if module3_keyframe_grid_scene_is_done_with_config(
            config, scene_id, grids, skipped
        )
    }


def save_module3_keyframe_grid_manifest(
    config: AgentConfig,
    scene_grids: Dict[str, str],
    skipped_scenes: Optional[Dict[str, Dict[str, object]]] = None,
    *,
    time_instru: Optional[TimeInstructionSet] = None,
) -> None:
    path = scene_keyframe_grid_edits_path(config)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    existing_grids, existing_skipped = load_module3_keyframe_grid_checkpoint(config)
    merged_grids = {**existing_grids, **scene_grids}
    merged_skipped = {**existing_skipped, **(skipped_scenes or {})}

    completed = True
    if time_instru is not None:
        for scene in time_instru.scenes:
            if not module3_keyframe_grid_scene_is_done_with_config(
                config,
                scene.scene_id,
                merged_grids,
                merged_skipped,
            ):
                completed = False
                break

    payload = {
        "version": "1.0",
        "completed": completed,
        "scene_grids": merged_grids,
        "skipped_scenes": merged_skipped,
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def is_module1_complete(config: AgentConfig, user_prompt: str) -> bool:
    """Module 1 finished when entity_instru.json exists for the same user prompt."""
    if not config.resume_from_checkpoints:
        return False
    path = config.entity_instru_path
    if not os.path.exists(path):
        return False
    try:
        entity_instru = EntityInstructionSet.load(path)
    except Exception as exc:
        logger.warning("Cannot load %s: %s", path, exc)
        return False
    if entity_instru.source_prompt.strip() != user_prompt.strip():
        logger.info("User prompt changed — re-running module 1")
        return False
    return True


def load_module1_result(config: AgentConfig) -> EntityInstructionSet:
    """Load module 1 output from workspace."""
    return EntityInstructionSet.load(config.entity_instru_path)


def is_module2_complete(config: AgentConfig) -> bool:
    """Module 2 finished when time_instru.json and scene assets are present."""
    if not config.resume_from_checkpoints:
        return False
    path = config.time_instru_path
    if not os.path.exists(path):
        return False
    try:
        time_instru = TimeInstructionSet.load(path)
    except Exception as exc:
        logger.warning("Cannot load %s: %s", path, exc)
        return False
    if not _video_path_matches(config, time_instru.source_video_path):
        logger.info("Source video changed — re-running module 2")
        return False
    if not time_instru.scenes:
        return False
    for scene in time_instru.scenes:
        clip_path = os.path.join(
            config.scenes_dir,
            scene.scene_id,
            f"{scene.scene_id}.mp4",
        )
        if not os.path.exists(clip_path):
            logger.info(
                "Scene asset missing for %s — re-running module 2",
                scene.scene_id,
            )
            return False
        if scene.keyframes_dir:
            if not os.path.isdir(scene.keyframes_dir):
                logger.info(
                    "Keyframes dir missing for %s — re-running module 2",
                    scene.scene_id,
                )
                return False
            if scene.keyframe_paths:
                if not all(os.path.exists(p) for p in scene.keyframe_paths):
                    return False
            elif not (scene.first_frame_path and os.path.exists(scene.first_frame_path)):
                return False
        elif not (scene.first_frame_path and os.path.exists(scene.first_frame_path)):
            continue
    return True


def load_module2_result(config: AgentConfig) -> TimeInstructionSet:
    """Load module 2 output from workspace."""
    return TimeInstructionSet.load(config.time_instru_path)


def load_physical_shots_from_workspace(config: AgentConfig) -> List[SceneClip]:
    """Rebuild physical-shot metadata from ``shots_analysis.json`` or ``shots/`` clips."""
    from video_editing_agent.utils.ffmpeg_utils import probe_duration

    if os.path.exists(config.shots_analysis_path):
        analysis = ShotAnalysisSet.load(config.shots_analysis_path)
        return [
            SceneClip(
                scene_id=shot.shot_id,
                start_sec=shot.pyscenedetect_start_sec,
                end_sec=shot.pyscenedetect_end_sec,
                shot_id=shot.shot_id,
                shot_clip_path=shot.clip_path,
            )
            for shot in analysis.shots
        ]

    shots: List[SceneClip] = []
    if not os.path.isdir(config.shots_dir):
        return shots

    names = sorted(
        f
        for f in os.listdir(config.shots_dir)
        if f.startswith("shot_") and f.endswith(".mp4")
    )
    cursor = 0.0
    for name in names:
        shot_id = os.path.splitext(name)[0]
        shot_path = os.path.abspath(os.path.join(config.shots_dir, name))
        try:
            duration = probe_duration(shot_path)
        except Exception:
            duration = 0.0
        start_sec = cursor
        end_sec = cursor + duration
        cursor = end_sec
        shots.append(
            SceneClip(
                scene_id=shot_id,
                start_sec=start_sec,
                end_sec=end_sec,
                shot_id=shot_id,
                shot_clip_path=shot_path,
            )
        )
    return shots


def is_shot_analysis_complete(config: AgentConfig, expected_shot_count: int) -> bool:
    """Shot VLM analysis finished when ``shots_analysis.json`` matches scene count."""
    if not config.resume_from_checkpoints:
        return False
    path = config.shots_analysis_path
    if not os.path.exists(path):
        return False
    try:
        analysis = ShotAnalysisSet.load(path)
    except Exception as exc:
        logger.warning("Cannot load %s: %s", path, exc)
        return False
    if not _video_path_matches(config, analysis.source_video_path):
        logger.info("Source video changed — re-running shot VLM analysis")
        return False
    if len(analysis.shots) != expected_shot_count:
        return False
    for shot in analysis.shots:
        if not shot.clip_path or not os.path.exists(shot.clip_path):
            return False
    return True


def load_shot_analysis_result(config: AgentConfig) -> ShotAnalysisSet:
    """Load per-shot VLM analysis from workspace."""
    return ShotAnalysisSet.load(config.shots_analysis_path)


def is_entity_keyframe_grounding_complete(
    config: AgentConfig,
    expected_entity_count: int,
) -> bool:
    """Module 2 entity keyframe grounding finished when JSON matches entity count."""
    if not config.resume_from_checkpoints:
        return False
    path = config.entity_keyframe_appearances_path
    if not os.path.exists(path):
        return False
    try:
        from video_editing_agent.schemas.entity_keyframe_appearances import (
            EntityKeyframeAppearanceSet,
        )

        data = EntityKeyframeAppearanceSet.load(path)
    except Exception as exc:
        logger.warning("Cannot load %s: %s", path, exc)
        return False
    if not _video_path_matches(config, data.source_video_path):
        return False
    return len(data.entities) == expected_entity_count


def is_entity_multiview_refs_complete(
    config: AgentConfig,
    entity_instru: EntityInstructionSet,
) -> bool:
    """Entity refs done when each sighted instruction has front-view + canonical images."""
    if not config.resume_from_checkpoints:
        return False
    appearances_path = config.entity_keyframe_appearances_path
    if not os.path.exists(appearances_path):
        return False
    try:
        from video_editing_agent.schemas.entity_keyframe_appearances import (
            EntityKeyframeAppearanceSet,
        )

        appearance_set = EntityKeyframeAppearanceSet.load(appearances_path)
    except Exception as exc:
        logger.warning("Cannot load %s: %s", appearances_path, exc)
        return False
    if not _video_path_matches(config, appearance_set.source_video_path):
        return False

    ref_dir = os.path.join(config.workspace_dir, "entity_refs")
    for instr in entity_instru.instructions:
        record = appearance_set.record_for_instruction(instr.instruction_id)
        if record is None or not record.appearances:
            continue
        mv = entity_ref_multiview_path(ref_dir, instr.instruction_id)
        canonical = entity_ref_canonical_path(ref_dir, instr.instruction_id)
        if not os.path.exists(mv) or not os.path.exists(canonical):
            logger.info(
                "Front-view refs incomplete for %s — re-running entity ref builder",
                instr.instruction_id,
            )
            return False
    return True


def is_module2_5_complete(config: AgentConfig, entity_instru: EntityInstructionSet) -> bool:
    """Module 2.5 finished when canonical_edits.json marks completion."""
    if not config.resume_from_checkpoints:
        return False
    manifest_path = canonical_edits_path(config)
    if not _manifest_completed(manifest_path):
        return False
    data = _load_json(manifest_path) or {}
    canonical_paths: Dict[str, str] = data.get("canonical_paths") or {}
    ref_dir = os.path.join(config.workspace_dir, "entity_refs")
    for instr in entity_instru.instructions:
        multiview = entity_ref_multiview_path(ref_dir, instr.instruction_id)
        canonical = entity_ref_canonical_path(ref_dir, instr.instruction_id)
        if os.path.exists(multiview) and os.path.exists(canonical):
            continue
        src_path = entity_ref_src_path(ref_dir, instr.instruction_id)
        mask_path = entity_ref_mask_path(ref_dir, instr.instruction_id)
        if not os.path.exists(src_path):
            continue
        if not mask_has_content(mask_path):
            continue
        expected = canonical_paths.get(instr.instruction_id) or canonical
        if not os.path.exists(expected):
            logger.info(
                "Missing canonical ref for %s — re-running module 2.5",
                instr.instruction_id,
            )
            return False
    return True


def load_module2_5_result(config: AgentConfig) -> Dict[str, str]:
    """Load module 2.5 canonical path map from workspace."""
    data = _load_json(canonical_edits_path(config)) or {}
    paths = data.get("canonical_paths") or {}
    return {str(k): str(v) for k, v in paths.items() if v}


def save_module2_5_manifest(config: AgentConfig, canonical_paths: Dict[str, str]) -> None:
    """Persist module 2.5 completion marker."""
    path = canonical_edits_path(config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "version": "1.0",
        "completed": True,
        "canonical_paths": canonical_paths,
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def is_module3_complete(config: AgentConfig, time_instru: TimeInstructionSet) -> bool:
    """Module 3 finished when every scene has a keyframe grid or recorded skip."""
    if not config.resume_from_checkpoints:
        return False
    keyframes, skipped_scenes = load_module3_checkpoint(config)
    pending: list[str] = []
    # Check ALL scenes, not just those with requires_edit bindings, because
    # time_instru binding is no longer used to filter which scenes are processed.
    for scene in time_instru.scenes:
        scene_id = scene.scene_id
        if module3_scene_is_done(scene_id, keyframes, skipped_scenes):
            continue
        pending.append(scene_id)
    if pending:
        logger.info(
            "Module 3 incomplete — pending scenes: %s",
            ", ".join(pending),
        )
        return False
    return True


def load_module3_result(config: AgentConfig) -> Dict[str, KeyframeAnchors]:
    """Load module 3 begin keyframe map from workspace."""
    keyframes, skipped = load_module3_checkpoint(config)
    return {
        scene_id: anchors
        for scene_id, anchors in keyframes.items()
        if module3_scene_is_done(scene_id, keyframes, skipped)
    }


def save_module3_manifest(
    config: AgentConfig,
    edited_keyframes: Dict[str, KeyframeAnchors],
    skipped_scenes: Optional[Dict[str, Dict[str, object]]] = None,
    *,
    time_instru: Optional[TimeInstructionSet] = None,
) -> None:
    """Persist module 3 checkpoint (keyframes + skipped scenes)."""
    path = edited_keyframes_path(config)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    existing_keyframes, existing_skipped = load_module3_checkpoint(config)
    merged_keyframes: Dict[str, KeyframeAnchors] = dict(existing_keyframes)
    for scene_id, anchors in edited_keyframes.items():
        prev = merged_keyframes.get(scene_id, {})
        merged_keyframes[scene_id] = {**prev, **anchors}
    merged_skipped = {**existing_skipped, **(skipped_scenes or {})}

    completed = True
    if time_instru is not None:
        for scene in time_instru.scenes:
            if not module3_scene_is_done(
                scene.scene_id,
                merged_keyframes,
                merged_skipped,
            ):
                completed = False
                break

    payload = {
        "version": "1.2",
        "completed": completed,
        "keyframes": merged_keyframes,
        "skipped_scenes": merged_skipped,
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def is_module4_complete(config: AgentConfig, time_instru: TimeInstructionSet) -> bool:
    """Module 4 finished when every scene has a valid edited clip."""
    if not config.resume_from_checkpoints:
        return False
    clips = load_module4_checkpoint(config)
    pending: list[str] = []
    # Check ALL scenes, not just those with requires_edit bindings, because
    # time_instru binding is no longer used to filter which scenes are processed.
    for scene in time_instru.scenes:
        scene_id = scene.scene_id
        if module4_scene_is_done(scene_id, clips):
            continue
        pending.append(scene_id)
    if pending:
        logger.info(
            "Module 4 incomplete — pending scenes: %s",
            ", ".join(pending),
        )
        return False
    return True


def load_module4_result(config: AgentConfig) -> Dict[str, str]:
    """Load module 4 clip path map from workspace."""
    return load_module4_checkpoint(config)


def save_module4_manifest(
    config: AgentConfig,
    edited_clips: Dict[str, str],
    *,
    time_instru: Optional[TimeInstructionSet] = None,
) -> None:
    """Persist module 4 checkpoint (merge with existing clips on disk)."""
    path = edited_clips_manifest_path(config)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    merged = {**load_module4_checkpoint(config), **edited_clips}
    completed = True
    if time_instru is not None:
        for scene in time_instru.scenes:
            if not module4_scene_is_done(scene.scene_id, merged):
                completed = False
                break

    payload = {
        "version": "1.0",
        "completed": completed,
        "clips": merged,
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def is_module5_complete(config: AgentConfig) -> bool:
    """Module 5 finished when final_output.mp4 exists."""
    if not config.resume_from_checkpoints:
        return False
    path = config.final_output_path
    return os.path.exists(path) and os.path.getsize(path) > 0
