"""Load per-scene keyframe metadata written by scene re-segmentation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from video_editing_agent.schemas.scenes import SceneClip

KEYFRAMES_MANIFEST = "keyframes.json"


def load_scene_keyframe_entries(scene: SceneClip) -> List[Dict[str, Any]]:
    """Return keyframe dicts from ``keyframes.json`` or ``keyframe_paths`` fallback."""
    manifest_path = scene.keyframes_manifest_path
    if not manifest_path and scene.keyframes_dir:
        manifest_path = os.path.join(scene.keyframes_dir, KEYFRAMES_MANIFEST)

    if manifest_path and os.path.exists(manifest_path):
        try:
            data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            entries = data.get("keyframes") or []
            if entries:
                return [e for e in entries if isinstance(e, dict)]
        except (OSError, json.JSONDecodeError):
            pass

    entries: List[Dict[str, Any]] = []
    paths = scene.keyframe_paths or []
    if not paths and scene.first_frame_path:
        paths = [scene.first_frame_path]
    for idx, path in enumerate(paths, start=1):
        if not path or not os.path.exists(path):
            continue
        entries.append(
            {
                "path": path,
                "filename": os.path.basename(path),
                "timestamp_in_video_sec": float(scene.start_sec),
                "timestamp_in_scene_sec": 0.0,
                "role": "opening" if idx == 1 else "",
                "description": "",
            }
        )
    return entries
