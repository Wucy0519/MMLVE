"""Helpers for front-view entity reference generation from keyframe sightings.

Some function names still say "multiview" for compatibility with existing
pipeline call sites, but the generated entity_refs are single front-view images.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Set

from PIL import Image, ImageDraw

from video_editing_agent.schemas.entity_keyframe_appearances import KeyframeEntityAppearance
from video_editing_agent.utils.mask_utils import (
    _append_bottom_caption,
    _compose_before_after_panels,
    _label_entity_panel,
    _load_reference_caption_font,
    entity_ref_canonical_path,
    entity_ref_meta_path,
    entity_ref_multiview_edited_path,
    entity_ref_multiview_path,
    entity_ref_overlay_path,
    entity_ref_src_path,
    format_canonical_comparison_caption,
)

REFERENCE_GRID_COLS = 3
REFERENCE_VIEW_ANGLES = ("front",)


def appearances_catalog_for_selection(
    appearances: Sequence[KeyframeEntityAppearance],
) -> List[Dict[str, Any]]:
    """Build a compact JSON-serializable catalog for VLM keyframe selection."""
    catalog: List[Dict[str, Any]] = []
    for idx, app in enumerate(appearances):
        catalog.append({
            "appearance_index": idx,
            "scene_id": app.scene_id,
            "timestamp_in_video_sec": round(app.timestamp_in_video_sec, 3),
            "timestamp_in_scene_sec": round(app.timestamp_in_scene_sec, 3),
            "confidence": round(float(app.confidence), 3),
            "quality_score": round(float(app.quality_score), 1),
            "appearance_time_score": round(float(app.appearance_time_score), 1),
            "subject_features_score": round(float(app.subject_features_score), 1),
            "identification_clarity_score": round(float(app.identification_clarity_score), 1),
            "view_angle": app.view_angle or "other",
            "orientation_note": (
                "front-facing or near-front reference candidate"
                if _normalize_view(app.view_angle) in REFERENCE_VIEW_ANGLES
                else "non-front or ambiguous orientation"
            ),
            "keyframe_role": app.keyframe_role,
            "keyframe_description": app.keyframe_description,
            "visibility_state": app.visibility_state,
            "pose_and_action": app.pose_and_action,
            "location_description": app.location_description,
        })
    return catalog


def resolve_selected_appearances(
    appearances: Sequence[KeyframeEntityAppearance],
    selected_indices: Sequence[int],
) -> List[KeyframeEntityAppearance]:
    """Map VLM-selected indices back to appearance records."""
    valid = [a for a in appearances if a.keyframe_path and os.path.exists(a.keyframe_path)]
    if not valid:
        return []
    out: List[KeyframeEntityAppearance] = []
    seen: Set[int] = set()
    for raw_idx in selected_indices:
        idx = int(raw_idx)
        if idx in seen or idx < 0 or idx >= len(valid):
            continue
        seen.add(idx)
        out.append(valid[idx])
    return out


def fallback_select_reference_keyframes(
    appearances: Sequence[KeyframeEntityAppearance],
    *,
    select_count: int,
    video_duration_sec: float = 0.0,
) -> List[int]:
    """Heuristic fallback: quality_score + temporal spread + view diversity + orientation clarity."""
    valid = [
        (idx, app)
        for idx, app in enumerate(appearances)
        if app.keyframe_path and os.path.exists(app.keyframe_path)
    ]
    if not valid:
        return []
    if len(valid) <= select_count:
        return [idx for idx, _ in valid]

    duration = video_duration_sec
    if duration <= 0:
        duration = max(app.timestamp_in_video_sec for _, app in valid) or 1.0

    def time_bin(ts: float) -> int:
        ratio = max(0.0, min(0.999, ts / duration))
        return min(2, int(ratio * 3))

    selected: List[int] = []
    selected_views: Set[str] = set()

    ranked = sorted(
        valid,
        key=lambda item: (
            _orientation_select_score(item[1]),
            float(item[1].quality_score),
            float(item[1].confidence),
        ),
        reverse=True,
    )

    for bin_id in range(3):
        candidates = [
            (idx, app) for idx, app in ranked
            if time_bin(app.timestamp_in_video_sec) == bin_id and idx not in selected
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda item: (
                _view_priority(item[1].view_angle, selected_views),
                _orientation_select_score(item[1]),
                float(item[1].quality_score),
                float(item[1].confidence),
            ),
            reverse=True,
        )
        idx, app = candidates[0]
        selected.append(idx)
        selected_views.add(_normalize_view(app.view_angle))

    for idx, app in ranked:
        if len(selected) >= select_count:
            break
        if idx in selected:
            continue
        view = _normalize_view(app.view_angle)
        if view in REFERENCE_VIEW_ANGLES and view not in selected_views:
            selected.append(idx)
            selected_views.add(view)

    for idx, _app in ranked:
        if len(selected) >= select_count:
            break
        if idx not in selected:
            selected.append(idx)

    return selected[:select_count]


def _normalize_view(view_angle: str) -> str:
    key = (view_angle or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in REFERENCE_VIEW_ANGLES:
        return key
    if "three_quarter" in key or "3_4" in key:
        return "front"
    return key or "other"


def _view_priority(view_angle: str, selected_views: Set[str]) -> int:
    view = _normalize_view(view_angle)
    if view in REFERENCE_VIEW_ANGLES and view not in selected_views:
        return 10
    if view not in selected_views:
        return 5
    return 0


def _orientation_select_score(app: KeyframeEntityAppearance) -> float:
    """Boost keyframes with reliable front-facing labels for reference synthesis."""
    score = float(app.quality_score)
    view = _normalize_view(app.view_angle)
    if view in REFERENCE_VIEW_ANGLES:
        score += 8.0
    elif view == "three_quarter":
        score += 3.0
    score += float(app.identification_clarity_score) * 0.15
    return score


def format_keyframe_notes(appearances: Sequence[KeyframeEntityAppearance]) -> str:
    """Build rich text notes for image-model and VLM prompts."""
    lines: List[str] = []
    for idx, app in enumerate(appearances, start=1):
        lines.append(
            f"Keyframe {idx} (quality={app.quality_score:.0f}/100, "
            f"confidence={app.confidence:.2f}, "
            f"orientation/view_angle={app.view_angle or 'unknown'}, "
            f"scene={app.scene_id}):"
        )
        if app.keyframe_description:
            lines.append(f"  moment: {app.keyframe_description}")
        if app.scene_moment_description:
            lines.append(f"  scene: {app.scene_moment_description}")
        if app.visibility_state:
            lines.append(f"  visibility: {app.visibility_state}")
        if app.pose_and_action:
            lines.append(f"  pose/action: {app.pose_and_action}")
        if app.location_description:
            lines.append(f"  location: {app.location_description}")
        lines.append(
            f"  scores: appearance_time={app.appearance_time_score:.0f}, "
            f"subject_features={app.subject_features_score:.0f}, "
            f"clarity={app.identification_clarity_score:.0f}"
        )
        lines.append(
            f"  timestamps: video={app.timestamp_in_video_sec:.3f}s, "
            f"scene={app.timestamp_in_scene_sec:.3f}s"
        )
    return "\n".join(lines) if lines else "(no keyframe notes)"


def build_keyframe_grid_image(
    image_paths: Sequence[str],
    *,
    cols: int = REFERENCE_GRID_COLS,
    cell_size: int = 256,
) -> Image.Image:
    """Compose keyframes into a grid (default 3 columns)."""
    paths = [p for p in image_paths if p and os.path.exists(p)]
    if not paths:
        return Image.new("RGB", (cell_size * cols, cell_size * cols), (255, 255, 255))

    rows = max(1, math.ceil(len(paths) / cols))
    grid_w = cols * cell_size
    grid_h = rows * cell_size
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    for idx, path in enumerate(paths):
        row, col = divmod(idx, cols)
        img = Image.open(path).convert("RGB")
        img.thumbnail((cell_size, cell_size), Image.Resampling.LANCZOS)
        ox = col * cell_size + (cell_size - img.width) // 2
        oy = row * cell_size + (cell_size - img.height) // 2
        canvas.paste(img, (ox, oy))
    return canvas


def save_keyframe_grid(
    image_paths: Sequence[str],
    output_path: str,
    *,
    cols: int = REFERENCE_GRID_COLS,
    cell_size: int = 256,
) -> str:
    """Save a keyframe grid PNG."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    build_keyframe_grid_image(image_paths, cols=cols, cell_size=cell_size).save(output_path)
    return output_path


def build_multiview_canonical_card(
    original_path: str,
    edited_path: Optional[str],
    *,
    instruction_id: str,
    entity_id: str,
    action: str,
    subject_features: str = "",
    panel_height: int = 512,
) -> Image.Image:
    """Side-by-side card for original vs edited front-view reference images."""
    original = Image.open(original_path).convert("RGB")
    if edited_path and os.path.exists(edited_path):
        edited = Image.open(edited_path).convert("RGB")
        left = _label_entity_panel(_resize_to_height(original, panel_height), "Original (front view)")
        right = _label_entity_panel(_resize_to_height(edited, panel_height), "Edited (front view)")
        composite = _compose_before_after_panels(left, right)
    else:
        composite = _label_entity_panel(
            _resize_to_height(original, panel_height),
            "Delete target (front view)",
        )
    caption = format_canonical_comparison_caption(
        instruction_id,
        entity_id,
        action,
        subject_features,
    )
    return _append_bottom_caption(composite, caption)


def _resize_to_height(img: Image.Image, height: int) -> Image.Image:
    w, h = img.size
    if h <= 0:
        return img
    scale = height / h
    return img.resize((max(1, int(w * scale)), height), Image.Resampling.LANCZOS)


def make_square_reference_image(
    img: Image.Image,
    *,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Pad an entity reference image to a square canvas without distorting it."""
    square = max(1, img.width, img.height)
    canvas = Image.new("RGB", (square, square), background)
    canvas.paste(img.convert("RGB"), ((square - img.width) // 2, (square - img.height) // 2))
    return canvas


def ensure_square_reference_file(path: str) -> str:
    """Ensure a generated entity reference file is square on disk."""
    if not path or not os.path.exists(path):
        return path
    img = Image.open(path).convert("RGB")
    if img.width != img.height:
        make_square_reference_image(img).save(path)
    return path


def save_multiview_entity_refs(
    ref_dir: str,
    instruction_id: str,
    *,
    entity_id: str,
    action: str,
    subject_features: str,
    multiview_source_path: str,
    multiview_edited_path: Optional[str],
    top_keyframe_path: str,
    input_manifest: Dict[str, Any],
) -> Dict[str, str]:
    """Persist front-view reference bundle under ``entity_refs/``."""
    os.makedirs(ref_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    mv_out = entity_ref_multiview_path(ref_dir, instruction_id)
    make_square_reference_image(Image.open(multiview_source_path).convert("RGB")).save(mv_out)
    paths["multiview"] = mv_out
    paths["front"] = mv_out

    src_out = entity_ref_src_path(ref_dir, instruction_id)
    Image.open(top_keyframe_path).convert("RGB").save(src_out)
    paths["src"] = src_out

    overlay_out = entity_ref_overlay_path(ref_dir, instruction_id)
    overlay = Image.open(mv_out).convert("RGB")
    caption = (
        f"Front-view entity reference | instruction_id={instruction_id}"
        + (f" | entity_id={entity_id}" if entity_id else "")
        + " | reference only — not an output template"
    )
    _append_bottom_caption(overlay, caption).save(overlay_out)
    paths["overlay"] = overlay_out

    if multiview_edited_path and os.path.exists(multiview_edited_path):
        edited_out = entity_ref_multiview_edited_path(ref_dir, instruction_id)
        make_square_reference_image(Image.open(multiview_edited_path).convert("RGB")).save(edited_out)
        paths["multiview_edited"] = edited_out
        paths["front_edited"] = edited_out

    canonical_out = entity_ref_canonical_path(ref_dir, instruction_id)
    card = build_multiview_canonical_card(
        mv_out,
        paths.get("multiview_edited"),
        instruction_id=instruction_id,
        entity_id=entity_id,
        action=action,
        subject_features=subject_features,
    )
    card.save(canonical_out)
    paths["canonical"] = canonical_out

    meta_path = entity_ref_meta_path(ref_dir, instruction_id)
    payload = {
        "instruction_id": instruction_id,
        "entity_id": entity_id,
        "action": action,
        "reference_type": "front_view",
        "front_source_path": mv_out,
        "front_edited_path": paths.get("multiview_edited", ""),
        # Backward-compatible aliases for readers that still use the old keys.
        "multiview_source_path": mv_out,
        "multiview_edited_path": paths.get("multiview_edited", ""),
        "canonical_path": canonical_out,
        "input_manifest": input_manifest,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    paths["meta"] = meta_path
    return paths


def build_dev_mode_multiview_placeholder(label: str, *, size: int = 1024) -> Image.Image:
    """Simple front-view placeholder for dev_mode."""
    canvas = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = _load_reference_caption_font()
    draw.rectangle([16, 16, size - 16, size - 64], outline=(180, 180, 180), width=2)
    draw.text((32, 32), "FRONT VIEW", fill=(80, 80, 80), font=font)
    draw.text((20, size - 36), label, fill=(40, 40, 40), font=font)
    return canvas
