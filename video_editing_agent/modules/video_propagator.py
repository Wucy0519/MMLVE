"""
Module 4 — per-scene video editing guided directly by entity_refs.

For each scene:
  1. Read ``entity_keyframe_appearances.json`` to determine which edit-target
     entities appear in the scene/sub-video.
  2. For each present entity, collect its before reference
     ``entity_refs/instr_00N_ref.png``, after reference
     ``entity_refs/instr_00N_ref_front_edited.png`` (when available), the edit
     instruction, and its appearance/location notes from the grounding catalog.
  3. Build one composite entity-reference grid containing all scene edits.
  4. Feed the original scene clip + composite reference + all edit instructions
     to the video edit model in a single call.
  5. Keep the existing keyframe-grid VLM QA/retry flow for video output quality.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from video_editing_agent.clients.base import ModelApiClientBase, VideoEditRejectedError
from video_editing_agent.config import AgentConfig
from video_editing_agent.schemas.entity_keyframe_appearances import (
    EntityKeyframeAppearanceSet,
    KeyframeEntityAppearance,
)
from video_editing_agent.schemas.instructions import EntityInstructionSet
from video_editing_agent.schemas.scenes import SceneClip, TimeInstructionSet
from video_editing_agent.utils.edit_qa_utils import build_edit_retry_guidance_section
from video_editing_agent.utils.ffmpeg_utils import mux_video_with_scene_audio, probe_duration
from video_editing_agent.utils.keyframe_manifest_utils import load_scene_keyframe_entries
from video_editing_agent.utils.mask_utils import (
    entity_ref_canonical_path,
    entity_ref_multiview_edited_path,
    entity_ref_multiview_path,
    entity_ref_overlay_path,
)
from video_editing_agent.utils.scene_video_edit_utils import (
    build_keyframe_comparison_grids,
    extract_edited_keyframes_from_manifest,
    load_entity_instru_text,
    original_keyframe_paths,
)
from video_editing_agent.utils.video_chunk_utils import (
    chunk_edited_last_frame_path,
    chunk_edited_path,
    chunk_first_frame_path,
    compute_chunk_time_ranges,
    concat_edited_chunks,
    extract_last_frame,
    scene_chunks_dir,
    split_scene_video_into_chunks,
    update_chunk_manifest_entry,
)
from video_editing_agent.utils.workspace_checkpoints import (
    edited_clip_path,
    load_module4_checkpoint,
    module4_scene_is_done,
    save_module4_manifest,
)

logger = logging.getLogger(__name__)

MAX_EDIT_ATTEMPTS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Per-entity edit data collection
# ─────────────────────────────────────────────────────────────────────────────

class EntityEditSpec:
    """One scene-level entity edit assembled from grounding + entity_refs."""

    __slots__ = (
        "instruction_id",
        "entity_id",
        "edit_prompt",
        "subject_features",
        "before_ref_path",
        "after_ref_path",
        "location_description",
        "keyframe_path",
        "timestamp_in_scene_sec",
        "confidence",
        "action",
    )

    def __init__(
        self,
        instruction_id: str,
        entity_id: str,
        edit_prompt: str,
        subject_features: str,
        before_ref_path: str,
        after_ref_path: str,
        location_description: str,
        keyframe_path: str,
        timestamp_in_scene_sec: float,
        confidence: float,
        action: str,
    ) -> None:
        self.instruction_id = instruction_id
        self.entity_id = entity_id
        self.edit_prompt = edit_prompt
        self.subject_features = subject_features
        self.before_ref_path = before_ref_path
        self.after_ref_path = after_ref_path
        self.location_description = location_description
        self.keyframe_path = keyframe_path
        self.timestamp_in_scene_sec = timestamp_in_scene_sec
        self.confidence = confidence
        self.action = action

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction_id": self.instruction_id,
            "entity_id": self.entity_id,
            "edit_prompt": self.edit_prompt,
            "subject_features": self.subject_features,
            "before_ref_path": self.before_ref_path,
            "after_ref_path": self.after_ref_path,
            "location_description": self.location_description,
            "keyframe_path": self.keyframe_path,
            "timestamp_in_scene_sec": self.timestamp_in_scene_sec,
            "confidence": self.confidence,
            "action": self.action,
        }


def _best_scene_appearance(
    appearances: List[KeyframeEntityAppearance],
    scene_id: str,
    *,
    min_confidence: float,
) -> Optional[KeyframeEntityAppearance]:
    candidates = [
        a for a in appearances
        if a.scene_id == scene_id and a.present and a.confidence >= min_confidence
    ]
    if not candidates:
        # Retry with a lower threshold for non-frontal / challenging views —
        # these are valid detections that may carry lower VLM confidence due
        # to viewpoint/lighting/expression but still identify the correct entity.
        robust_min = max(0.35, min_confidence - 0.15)
        candidates = [
            a for a in appearances
            if a.scene_id == scene_id and a.present and a.confidence >= robust_min
        ]
        if not candidates:
            return None

    def _robust_sort_key(a: KeyframeEntityAppearance) -> Tuple[float, float, float]:
        # Prefer front/three-quarter views, but still accept side/back when
        # identity evidence is strong. Apply a viewpoint bonus rather than
        # penalizing non-frontal views.
        view = (a.view_angle or "").strip().lower()
        viewpoint_bonus = {
            "front": 0.0,
            "three_quarter": -2.0,
            "left": -5.0,
            "right": -5.0,
            "back": -8.0,
        }.get(view, -5.0)
        quality = float(a.quality_score or 0.0) + viewpoint_bonus
        clarity = float(a.identification_clarity_score or 0.0)
        # Don't penalize non-frontal views on confidence — the VLM prompt
        # already instructs it to keep confidence high for matching identity.
        confidence = float(a.confidence or 0.0)
        return (quality, clarity, confidence)

    return max(candidates, key=_robust_sort_key)


def _existing_first(*paths: str) -> str:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return ""


def _unique_existing_paths(paths: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        norm = os.path.normpath(os.path.abspath(path))
        if norm in seen:
            continue
        seen.add(norm)
        out.append(path)
    return out


def _collect_scene_entity_refs(
    config: AgentConfig,
    entity_instru: EntityInstructionSet,
) -> Tuple[List[str], str]:
    """Collect all entity reference images + build catalog text block.

    Returns:
        (ref_image_paths, entity_catalog_block)
    """
    ref_dir = os.path.join(config.workspace_dir, "entity_refs")
    ref_paths: List[str] = []
    catalog_lines: List[str] = []

    for idx, instr in enumerate(entity_instru.instructions, start=1):
        before = _existing_first(
            entity_ref_overlay_path(ref_dir, instr.instruction_id),
            entity_ref_multiview_path(ref_dir, instr.instruction_id),
            entity_ref_canonical_path(ref_dir, instr.instruction_id),
        )
        after = _existing_first(
            entity_ref_multiview_edited_path(ref_dir, instr.instruction_id),
            entity_ref_canonical_path(ref_dir, instr.instruction_id),
        )
        collected = _unique_existing_paths([before, after])
        ref_paths.extend(collected)

        img_note = (
            f"reference images {len(ref_paths) - len(collected) + 1}"
            if collected
            else "no reference image available"
        )
        if len(collected) > 1:
            img_note = f"reference images {len(ref_paths) - len(collected) + 1}–{len(ref_paths)}"

        catalog_lines.append(
            f"[Entity {idx}] instruction_id={instr.instruction_id}; "
            f"entity_id={instr.entity_id}; action={instr.action.value}; "
            f"subject_features: {instr.subject_features}; "
            f"edit_prompt: {instr.edit_prompt}; {img_note}"
        )

    catalog_block = "\n".join(catalog_lines) if catalog_lines else "(no entities)"
    return ref_paths, catalog_block


VOTE_RETRY_DELAYS = [2, 5]  # seconds between vote retries on empty response


async def vote_scene_should_edit(
    config: AgentConfig,
    api_client: ModelApiClientBase,
    scene: SceneClip,
    entity_instru: EntityInstructionSet,
    *,
    num_votes: int = 3,
    min_yes: int = 2,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Run VLM votes to decide whether a scene needs editing.

    Sends all scene keyframes + all entity_refs to VLM in a single call per
    vote. If >= ``min_yes`` votes say "yes, edit-target entity present",
    the scene should be edited.

    Returns:
        (should_edit, vote_results)
    """
    keyframe_paths = original_keyframe_paths(scene)
    if not keyframe_paths:
        logger.warning(
            "%s: no keyframes for existence vote — assume edit needed",
            scene.scene_id,
        )
        return True, []

    if not entity_instru.instructions:
        logger.info("%s: no edit instructions — skip", scene.scene_id)
        return False, []

    ref_paths, catalog_block = _collect_scene_entity_refs(config, entity_instru)
    if not ref_paths:
        logger.warning(
            "%s: no entity reference images found — assume edit needed",
            scene.scene_id,
        )
        return True, []

    votes: List[Dict[str, Any]] = []
    yes_count = 0

    for vote_idx in range(1, num_votes + 1):
        result = await api_client.vote_scene_entity_existence(
            keyframe_paths=keyframe_paths,
            entity_ref_image_paths=ref_paths,
            entity_catalog_block=catalog_block,
        )
        result["vote_index"] = vote_idx
        votes.append(result)

        if result.get("scene_has_edit_target"):
            yes_count += 1

        logger.info(
            "%s existence vote %d/%d: has_target=%s (yes=%d/%d so far)",
            scene.scene_id,
            vote_idx,
            num_votes,
            result.get("scene_has_edit_target"),
            yes_count,
            num_votes,
        )

        # Early exit: if enough yes votes, no need to continue.
        if yes_count >= min_yes:
            break
        # Early exit: if impossible to reach min_yes, stop.
        remaining = num_votes - vote_idx
        if yes_count + remaining < min_yes:
            break

    should_edit = yes_count >= min_yes
    logger.info(
        "%s existence vote final: should_edit=%s (yes=%d/%d, threshold=%d)",
        scene.scene_id,
        should_edit,
        yes_count,
        len(votes),
        min_yes,
    )
    return should_edit, votes


def _vote_present_instruction_ids(vote_results: List[Dict[str, Any]]) -> Dict[str, str]:
    """Extract instruction_ids that any vote marked as present.

    Returns:
        instruction_id → location_description text (from the first positive vote
        that has a location_description, falling back to identity_cues).
    """
    out: Dict[str, str] = {}
    for vote in vote_results:
        entities = vote.get("entities") or []
        if not isinstance(entities, list):
            continue
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            iid = str(ent.get("instruction_id", "") or "").strip()
            if not iid or iid in out:
                continue
            if ent.get("present"):
                loc = str(ent.get("location_description", "") or "").strip()
                if not loc:
                    loc = str(ent.get("identity_cues", "") or "").strip()
                out[iid] = loc
    return out


def collect_entity_edit_specs(
    config: AgentConfig,
    scene: SceneClip,
    entity_instru: EntityInstructionSet,
    *,
    vote_results: Optional[List[Dict[str, Any]]] = None,
) -> List[EntityEditSpec]:
    """Collect scene edits from ``entity_keyframe_appearances.json`` **and** VLM votes.

    A scene needs an edit when EITHER:
    - the grounding catalog says the entity is present in that scene, OR
    - the VLM existence vote marked the entity as present.

    This dual-source approach prevents entities missed by Module 2 grounding
    (due to dark lighting, non-frontal views, occlusion, etc.) from being
    skipped during video editing.
    """
    scene_id = scene.scene_id
    appearances_path = config.entity_keyframe_appearances_path
    instr_by_id = {i.instruction_id: i for i in entity_instru.instructions}
    ref_dir = os.path.join(config.workspace_dir, "entity_refs")
    specs: List[EntityEditSpec] = []
    covered_iids: set[str] = set()

    # ── Source 1: entity_keyframe_appearances.json ──────────────────────
    if os.path.exists(appearances_path):
        appearance_set = EntityKeyframeAppearanceSet.load(appearances_path)

        for record in appearance_set.entities:
            instr = instr_by_id.get(record.instruction_id)
            if instr is None:
                continue
            appearance = _best_scene_appearance(
                record.appearances,
                scene_id,
                min_confidence=config.entity_keyframe_min_confidence,
            )
            if appearance is None:
                continue

            before_ref = _existing_first(
                entity_ref_overlay_path(ref_dir, instr.instruction_id),
                entity_ref_multiview_path(ref_dir, instr.instruction_id),
                entity_ref_canonical_path(ref_dir, instr.instruction_id),
            )
            after_ref = _existing_first(
                entity_ref_multiview_edited_path(ref_dir, instr.instruction_id),
                entity_ref_canonical_path(ref_dir, instr.instruction_id),
            )
            if not before_ref:
                logger.warning(
                    "%s: missing before entity ref for %s — skip",
                    scene_id,
                    instr.instruction_id,
                )
                continue

            specs.append(EntityEditSpec(
                instruction_id=instr.instruction_id,
                entity_id=instr.entity_id,
                edit_prompt=instr.edit_prompt or record.edit_prompt or "",
                subject_features=instr.subject_features or record.subject_features or "",
                before_ref_path=before_ref,
                after_ref_path=after_ref,
                location_description=appearance.location_description or "",
                keyframe_path=appearance.keyframe_path or "",
                timestamp_in_scene_sec=float(appearance.timestamp_in_scene_sec or 0.0),
                confidence=float(appearance.confidence or 0.0),
                action=str(instr.action.value),
            ))
            covered_iids.add(instr.instruction_id)
    else:
        logger.warning(
            "%s: missing entity appearances catalog — %s",
            scene_id,
            appearances_path,
        )

    # ── Source 2: VLM vote results (fill in entities missed by grounding) ─
    if vote_results:
        vote_present = _vote_present_instruction_ids(vote_results)
        keyframe_entries = load_scene_keyframe_entries(scene)
        # Use the first available keyframe as fallback for vote-only entities.
        fallback_kf_path = ""
        fallback_kf_ts = 0.0
        if keyframe_entries:
            first_entry = keyframe_entries[0]
            fallback_kf_path = str(first_entry.get("path", "") or "")
            fallback_kf_ts = float(first_entry.get("timestamp_in_scene_sec", 0.0) or 0.0)

        for iid, cues in vote_present.items():
            if iid in covered_iids:
                continue
            instr = instr_by_id.get(iid)
            if instr is None:
                continue

            before_ref = _existing_first(
                entity_ref_overlay_path(ref_dir, instr.instruction_id),
                entity_ref_multiview_path(ref_dir, instr.instruction_id),
                entity_ref_canonical_path(ref_dir, instr.instruction_id),
            )
            after_ref = _existing_first(
                entity_ref_multiview_edited_path(ref_dir, instr.instruction_id),
                entity_ref_canonical_path(ref_dir, instr.instruction_id),
            )
            if not before_ref:
                logger.warning(
                    "%s: missing before entity ref for vote-identified %s — skip",
                    scene_id,
                    iid,
                )
                continue

            logger.info(
                "%s: adding vote-identified entity %s (not in appearances catalog)",
                scene_id,
                iid,
            )
            specs.append(EntityEditSpec(
                instruction_id=instr.instruction_id,
                entity_id=instr.entity_id,
                edit_prompt=instr.edit_prompt or "",
                subject_features=instr.subject_features or "",
                before_ref_path=before_ref,
                after_ref_path=after_ref,
                location_description=cues or "identified by VLM existence vote",
                keyframe_path=fallback_kf_path,
                timestamp_in_scene_sec=fallback_kf_ts,
                confidence=0.7,  # vote-confirmed, moderate confidence
                action=str(instr.action.value),
            ))
            covered_iids.add(iid)

    specs.sort(key=lambda s: (s.timestamp_in_scene_sec, s.instruction_id))
    logger.info(
        "%s: collected %d entity edit spec(s) (appearances + vote)",
        scene_id,
        len(specs),
    )
    return specs


# ─────────────────────────────────────────────────────────────────────────────
# Composite reference image builder
# ─────────────────────────────────────────────────────────────────────────────

def _paste_fit(
    canvas: Image.Image,
    path: str,
    box: Tuple[int, int, int, int],
    *,
    missing_label: str,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    if not path or not os.path.exists(path):
        draw.rectangle(box, fill=(70, 35, 35))
        draw.text((x0 + 10, y0 + 10), missing_label, fill=(255, 120, 120))
        return
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max(1, x1 - x0), max(1, y1 - y0)))
        px = x0 + (x1 - x0 - img.width) // 2
        py = y0 + (y1 - y0 - img.height) // 2
        canvas.paste(img, (px, py))
    except Exception as exc:
        logger.warning("Failed to load reference image %s: %s", path, exc)
        draw.rectangle(box, fill=(70, 35, 35))
        draw.text((x0 + 10, y0 + 10), missing_label, fill=(255, 120, 120))


def build_entity_reference_grid(
    specs: List[EntityEditSpec],
    output_path: str,
    cell_w: int = 420,
    cell_h: int = 420,
    label_h: int = 58,
) -> str:
    """Build one before/after entity_refs grid for a scene.

    Layout: one row per entity, two columns:
    LEFT = ``instr_00N_ref.png`` before/identity card;
    RIGHT = ``instr_00N_ref_front_edited.png`` after reference when available.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not specs:
        img = Image.new("RGB", (cell_w * 2, cell_h), color=(40, 40, 40))
        img.save(output_path)
        return output_path

    grid_w = cell_w * 2
    grid_h = len(specs) * (cell_h + label_h)
    grid = Image.new("RGB", (grid_w, grid_h), color=(28, 28, 28))
    draw = ImageDraw.Draw(grid)

    for row, spec in enumerate(specs):
        y = row * (cell_h + label_h)
        draw.rectangle([0, y, grid_w, y + label_h], fill=(50, 50, 50))
        label = (
            f"Row {row + 1}: {spec.instruction_id} / {spec.entity_id} | "
            f"left=BEFORE identity+description, right=AFTER edit reference"
        )
        draw.text((8, y + 8), label, fill=(255, 230, 120))
        draw.text((8, y + 30), f"edit: {spec.edit_prompt[:140]}", fill=(230, 230, 230))

        top = y + label_h
        _paste_fit(
            grid,
            spec.before_ref_path,
            (0, top, cell_w, top + cell_h),
            missing_label="BEFORE REF MISSING",
        )
        _paste_fit(
            grid,
            spec.after_ref_path,
            (cell_w, top, cell_w * 2, top + cell_h),
            missing_label=("DELETE / NO AFTER REF" if spec.action == "delete" else "AFTER REF MISSING"),
        )
        draw.text((8, top + cell_h - 22), "BEFORE: instr_00N_ref.png", fill=(180, 210, 255))
        draw.text((cell_w + 8, top + cell_h - 22), "AFTER: instr_00N_ref_front_edited.png", fill=(180, 255, 190))

    grid.save(output_path)
    return output_path


def build_per_entity_reference_cards(
    specs: List[EntityEditSpec],
    output_dir: str,
    cell_w: int = 420,
    cell_h: int = 420,
    label_h: int = 58,
) -> List[str]:
    """Build one BEFORE/AFTER reference card per entity (not a combined grid).

    Each card is a standalone image with:
    - LEFT = ``instr_00N_ref.png`` (before/identity)
    - RIGHT = ``instr_00N_ref_front_edited.png`` (after edit reference)

    Returns a list of absolute paths, one per spec (in order).
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: List[str] = []

    for idx, spec in enumerate(specs, start=1):
        card_path = os.path.join(output_dir, f"entity_ref_card_{idx:02d}_{spec.instruction_id}.png")
        grid_w = cell_w * 2
        grid_h = cell_h + label_h
        grid = Image.new("RGB", (grid_w, grid_h), color=(28, 28, 28))
        draw = ImageDraw.Draw(grid)

        # Label header
        draw.rectangle([0, 0, grid_w, label_h], fill=(50, 50, 50))
        label = (
            f"Entity {idx}: {spec.instruction_id} / {spec.entity_id} | "
            f"left=BEFORE, right=AFTER | edit: {spec.edit_prompt[:120]}"
        )
        draw.text((8, 8), label, fill=(255, 230, 120))
        draw.text((8, 30), f"action: {spec.action}", fill=(230, 230, 230))

        top = label_h
        _paste_fit(
            grid,
            spec.before_ref_path,
            (0, top, cell_w, top + cell_h),
            missing_label="BEFORE REF MISSING",
        )
        _paste_fit(
            grid,
            spec.after_ref_path,
            (cell_w, top, cell_w * 2, top + cell_h),
            missing_label=("DELETE / NO AFTER REF" if spec.action == "delete" else "AFTER REF MISSING"),
        )
        draw.text((8, top + cell_h - 22), "BEFORE", fill=(180, 210, 255))
        draw.text((cell_w + 8, top + cell_h - 22), "AFTER", fill=(180, 255, 190))

        grid.save(card_path)
        paths.append(os.path.abspath(card_path))

    return paths


def build_chunk_reference_grid(
    entity_reference_grid_path: str,
    continuity_frame_path: str,
    output_path: str,
) -> str:
    """Combine previous chunk tail frame with entity refs for a chunk call."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    entity_grid = Image.open(entity_reference_grid_path).convert("RGB")
    cont = Image.open(continuity_frame_path).convert("RGB")
    max_w = max(entity_grid.width, 840)
    cont_h = 420
    canvas = Image.new("RGB", (max_w, cont_h + entity_grid.height + 70), color=(28, 28, 28))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, max_w, 70], fill=(50, 50, 50))
    draw.text(
        (8, 10),
        "Chunk continuity + entity refs: top image = previous edited chunk tail; below = before/after entity references.",
        fill=(255, 230, 120),
    )
    cont.thumbnail((max_w, cont_h))
    canvas.paste(cont, ((max_w - cont.width) // 2, 70 + (cont_h - cont.height) // 2))
    canvas.paste(entity_grid, ((max_w - entity_grid.width) // 2, 70 + cont_h))
    canvas.save(output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Per-entity text prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_entity_edit_prompt(specs: List[EntityEditSpec]) -> str:
    """Build a single prompt containing all scene entity edits."""
    if not specs:
        return ""

    lines: List[str] = []
    lines.append("Edit the provided ORIGINAL source video clip directly.")
    lines.append(
        "The attached reference images are separate per-entity BEFORE/AFTER cards — "
        "one image per entity. Each card has LEFT=BEFORE (identity) and RIGHT=AFTER (edited appearance). "
        "Use each card to identify its corresponding target entity and the intended edited appearance."
    )
    lines.append("")
    lines.append("ENTITY EDITS TO APPLY SIMULTANEOUSLY:")

    for idx, spec in enumerate(specs, start=1):
        after = (
            f"reference card {idx} RIGHT panel"
            if spec.after_ref_path and os.path.exists(spec.after_ref_path)
            else "the textual edit instruction (delete/no after-reference image)"
        )
        lines.append("")
        lines.append(f"[Entity {idx} / Reference Card {idx}] instruction_id={spec.instruction_id}; entity_id={spec.entity_id}; action={spec.action}")
        lines.append(f"  - Target identity: reference card {idx} LEFT panel; subject_features: {spec.subject_features}")
        lines.append(f"  - Edit instruction: {spec.edit_prompt}")
        lines.append(f"  - Edited appearance guide: {after}")

    lines.append("")
    lines.append("QUALITY AND PRESERVATION REQUIREMENTS:")
    lines.append("- Apply ALL listed entity edits in one pass to the source video clip, only when the corresponding entity is visible.")
    lines.append("- Do not change any non-target entity, object, clothing, face, pose, action, gesture, or expression unless explicitly instructed.")
    lines.append("- Do not modify backgrounds, walls, floors, furniture, props, scenery, lighting, camera framing, black bars, or unedited regions.")
    lines.append("- Preserve the original motion, timing, camera movement, composition, occlusions, shadows, blur, noise, and video compression characteristics.")
    lines.append("- Blend edited attributes photorealistically into the original scene; avoid pasted stickers, over-sharp edits, identity drift, flicker, warping, duplication, or temporal inconsistency.")
    lines.append("- Never copy the reference card layout into the video; the output must remain the original video scene with only the requested entity edits.")
    lines.append("")
    lines.append("ENTITY-TO-EDIT ONE-TO-ONE MAPPING (CRITICAL — violations cause severe errors):")
    lines.append("- Each reference card (Entity 1, Entity 2, ...) corresponds to a DIFFERENT, DISTINCT individual/object in the video.")
    lines.append("- One entity in the video can ONLY be bound to ONE reference card / edit instruction. NEVER apply two different edits to the same person/object.")
    lines.append("- If two reference cards look similar, you MUST still distinguish them by the distinguishing features described in subject_features and")
    lines.append("  match each card to a SEPARATE person/object in the video. Do not merge or collapse two edit targets onto one individual.")
    lines.append("- If you cannot find a distinct second person/object for a reference card, apply only the edits you can confidently match to separate individuals,")
    lines.append("  and leave the unmatched edit for retry. Do NOT apply a second edit to an already-edited entity.")
    lines.append("- Example: If Entity 1 = 'change shirt to white' and Entity 2 = 'replace headscarf with cap', and both reference cards show men at a table,")
    lines.append("  you must find TWO DIFFERENT men — apply the shirt edit to one and the cap edit to the OTHER. Never apply both edits to the same person.")
    lines.append("")
    lines.append("ENTITY IDENTIFICATION ROBUSTNESS:")
    lines.append("- The target entity may appear from ANY camera angle: front, side (left/right profile), back, three-quarter, or overhead.")
    lines.append("- Match the entity by stable identity cues (face profile, head/hair shape, hairline, body build, body proportions, clothing, accessories)")
    lines.append("  — NOT by requiring a frontal face. Side/back/three-quarter views are valid edit targets.")
    lines.append("- Do NOT skip editing an entity because its expression, gaze, head pose, lighting, motion blur, or clothing state differs from the reference image.")
    lines.append("- Apply the edit to the correct entity throughout the clip regardless of viewpoint, expression, or lighting changes across frames.")
    lines.append("- For back/side views, adapt the edit naturally to the visible angle (e.g. hair color change on visible back-of-head hair, accessory on visible shoulder side).")
    return "\n".join(lines)


CHUNK_CONTINUITY_PROMPT = (
    "\n\nCHUNK CONTINUITY: For this later sub-clip, the top image in the attached reference grid "
    "is the previous edited chunk's last frame. Use it only as a continuity cue for edited "
    "appearance and temporal consistency at the boundary; do not copy any grid layout into the output."
)


# ─────────────────────────────────────────────────────────────────────────────
# VideoPropagator
# ─────────────────────────────────────────────────────────────────────────────

class VideoPropagator:
    """Video Propagation Agent (Module 4) — direct scene video editing."""

    def __init__(
        self,
        config: AgentConfig,
        api_client: ModelApiClientBase,
        max_concurrency: int = 2,
    ) -> None:
        self.config = config
        self.api_client = api_client
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def run(
        self,
        time_instru: TimeInstructionSet,
        entity_instru: EntityInstructionSet,
    ) -> Dict[str, str]:
        """Edit all scenes, concurrently."""
        try:
            logger.info("VideoPropagator.run — concurrency=%d", self.max_concurrency)

            if self.config.resume_from_checkpoints:
                edited_clips = load_module4_checkpoint(self.config)
            else:
                edited_clips = {}

            jobs: List[Tuple[str, asyncio.Task]] = []
            for scene in time_instru.scenes:
                scene_id = scene.scene_id
                if module4_scene_is_done(scene_id, edited_clips):
                    logger.info(
                        "Checkpoint: skip %s — edited clip already exists",
                        scene_id,
                    )
                    continue

                # ── VLM existence vote: skip scenes with no edit-target entity ──
                should_edit, vote_results = await vote_scene_should_edit(
                    self.config, self.api_client, scene, entity_instru,
                    num_votes=3, min_yes=2,
                )

                # Persist vote results for debugging/audit
                work_dir = self._scene_work_dir(scene_id)
                os.makedirs(work_dir, exist_ok=True)
                vote_sidecar = os.path.join(work_dir, "entity_existence_votes.json")
                with open(vote_sidecar, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "scene_id": scene_id,
                            "should_edit": should_edit,
                            "votes": vote_results,
                        },
                        fh, indent=2, ensure_ascii=False,
                    )

                if not should_edit:
                    logger.info(
                        "Skip %s — VLM existence vote rejected (no edit-target entity), copying original",
                        scene_id,
                    )
                    source_clip = os.path.join(
                        self.config.scenes_dir, scene_id, f"{scene_id}.mp4",
                    )
                    output_clip = edited_clip_path(self.config, scene_id)
                    if os.path.exists(source_clip):
                        import shutil
                        os.makedirs(os.path.dirname(output_clip) or ".", exist_ok=True)
                        shutil.copy2(source_clip, output_clip)
                        edited_clips[scene_id] = output_clip
                        if self.config.resume_from_checkpoints:
                            save_module4_manifest(
                                self.config,
                                {scene_id: output_clip},
                                time_instru=time_instru,
                            )
                    else:
                        logger.warning(
                            "Skip %s — source clip missing: %s",
                            scene_id,
                            source_clip,
                        )
                    continue

                task = asyncio.create_task(
                    self._edit_scene(
                        scene=scene,
                        entity_instru=entity_instru,
                        time_instru=time_instru,
                        vote_results=vote_results,
                    )
                )
                jobs.append((scene_id, task))

            if not jobs:
                logger.info(
                    "VideoPropagator.run done — %d clips (all from checkpoint or none pending)",
                    len(edited_clips),
                )
                return edited_clips

            results = await asyncio.gather(*(t for _, t in jobs), return_exceptions=True)

            errors: List[str] = []
            for (scene_id, _), result in zip(jobs, results):
                if isinstance(result, Exception):
                    errors.append(f"{scene_id}: {result}")
                else:
                    edited_clips[scene_id] = result  # type: ignore[assignment]

            if errors:
                raise RuntimeError(f"Video propagation failed: {errors[0]}")

            logger.info("VideoPropagator.run done — %d clips", len(edited_clips))
            return edited_clips

        except Exception as exc:
            logger.error("VideoPropagator.run failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Video propagation failed: {exc}") from exc

    async def _edit_scene(
        self,
        *,
        scene: SceneClip,
        entity_instru: EntityInstructionSet,
        time_instru: TimeInstructionSet,
        vote_results: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        scene_id = scene.scene_id
        async with self._semaphore:
            source_clip = os.path.join(
                self.config.scenes_dir, scene_id, f"{scene_id}.mp4",
            )
            output_clip = edited_clip_path(self.config, scene_id)
            work_dir = self._scene_work_dir(scene_id)
            os.makedirs(work_dir, exist_ok=True)

            # ── Step 1: collect per-entity edit specs ──────────────────────
            specs = collect_entity_edit_specs(
                self.config, scene, entity_instru,
                vote_results=vote_results,
            )

            # Save specs for debugging/audit
            specs_path = os.path.join(work_dir, "entity_edit_specs.json")
            with open(specs_path, "w", encoding="utf-8") as fh:
                json.dump(
                    [s.to_dict() for s in specs],
                    fh, indent=2, ensure_ascii=False,
                )

            if not specs:
                logger.info(
                    "%s: no entity edits found after spec collection — copying original video",
                    scene_id,
                )
                # No edits needed: copy source as output
                import shutil
                shutil.copy2(source_clip, output_clip)
                if self.config.resume_from_checkpoints:
                    save_module4_manifest(
                        self.config, {scene_id: output_clip}, time_instru=time_instru,
                    )
                return output_clip

            # ── Step 2: build per-entity reference cards + prompt ──────────
            # Each entity gets its own standalone BEFORE/AFTER reference card,
            # sent as a separate reference_image to the video model — preventing
            # model confusion from a single combined grid.
            ref_cards_dir = os.path.join(work_dir, "entity_ref_cards")
            per_entity_ref_paths = await asyncio.to_thread(
                build_per_entity_reference_cards, specs, ref_cards_dir,
            )
            # Also build a combined grid as fallback (for QA / sidecar).
            ref_grid_path = os.path.join(work_dir, "entity_reference_grid.png")
            await asyncio.to_thread(build_entity_reference_grid, specs, ref_grid_path)
            edit_prompt = build_entity_edit_prompt(specs)
            video_reference_image = ref_grid_path

            logger.info(
                "%s direct video edit — %d grounded entity spec(s), %d per-entity ref cards, prompt: %s",
                scene_id,
                len(specs),
                len(per_entity_ref_paths),
                edit_prompt[:200],
            )

            # ── Step 3: determine chunking ─────────────────────────────────
            scene_duration = probe_duration(source_clip)
            max_chunk = self.config.video_chunk_max_sec
            min_tail = self.config.video_chunk_min_tail_sec
            chunk_ranges = compute_chunk_time_ranges(
                scene_duration, max_chunk, min_tail,
            )

            entity_ref_paths = _unique_existing_paths(
                [ref_grid_path]
                + [s.before_ref_path for s in specs]
                + [s.after_ref_path for s in specs]
            )
            entity_instru_text = load_entity_instru_text(self.config)

            if len(chunk_ranges) <= 1:
                result = await self._edit_scene_with_qa_retry(
                    scene=scene,
                    source_clip=source_clip,
                    output_clip=output_clip,
                    work_dir=work_dir,
                    entity_ref_paths=entity_ref_paths,
                    entity_instru_text=entity_instru_text,
                    base_edit_prompt=edit_prompt,
                    reference_image_path=video_reference_image,
                    reference_image_paths=per_entity_ref_paths,
                    audio_path=scene.audio_path or "",
                    specs=specs,
                )
            else:
                logger.info(
                    "Editing %s (%.2fs) — splitting into ≤%.1fs chunks",
                    scene_id,
                    scene_duration,
                    max_chunk,
                )
                result = await self._edit_scene_chunked(
                    scene=scene,
                    source_clip=source_clip,
                    output_clip=output_clip,
                    work_dir=work_dir,
                    entity_ref_paths=entity_ref_paths,
                    entity_instru_text=entity_instru_text,
                    base_edit_prompt=edit_prompt,
                    reference_image_path=video_reference_image,
                    reference_image_paths=per_entity_ref_paths,
                    max_chunk_sec=max_chunk,
                    min_tail_sec=min_tail,
                )

            if self.config.resume_from_checkpoints:
                save_module4_manifest(
                    self.config,
                    {scene_id: result},
                    time_instru=time_instru,
                )
            return result

    def _scene_work_dir(self, scene_id: str) -> str:
        return os.path.join(self.config.scenes_dir, scene_id, "direct_video_edit")

    async def _edit_scene_with_qa_retry(
        self,
        *,
        scene: SceneClip,
        source_clip: str,
        output_clip: str,
        work_dir: str,
        entity_ref_paths: List[str],
        entity_instru_text: str,
        base_edit_prompt: str,
        reference_image_path: str,
        audio_path: str,
        specs: Optional[List[EntityEditSpec]] = None,
        reference_image_paths: Optional[List[str]] = None,
        skip_audio_mux: bool = False,
        attempt_suffix: str = "",
    ) -> str:
        """Run video edit + keyframe-grid QA; retry up to MAX_EDIT_ATTEMPTS times.

        All attempt outputs are preserved. After all attempts, VLM selects the
        best result by comparing their keyframe grids — not just the last one.
        """
        scene_id = scene.scene_id
        original_kf_paths = original_keyframe_paths(scene)
        qa_sidecar = os.path.join(work_dir, f"qa{attempt_suffix}.json")
        retry_focus = ""
        positive_focus = ""
        missing_edits_focus = ""
        qa_history: List[dict] = []
        qa_enabled = self._qa_enabled()
        max_attempts = MAX_EDIT_ATTEMPTS if qa_enabled else 1

        effective_base_prompt = base_edit_prompt

        # Track instruction_ids that have been discontinued (e.g. pasted entity detected —
        # edit too difficult, abandon rather than retry).
        discontinued_iids: set[str] = set()
        # Active specs (filtered to exclude discontinued instructions).
        active_specs: List[EntityEditSpec] = list(specs) if specs else []
        # Active per-entity reference image paths (filtered to match active_specs).
        active_ref_paths: Optional[List[str]] = reference_image_paths

        # Track all attempt outputs + their edited keyframe grids for final selection.
        attempt_outputs: List[str] = []
        attempt_grids: List[str] = []  # edited keyframe grid path per attempt

        for attempt in range(1, max_attempts + 1):
            attempt_tag = f"{attempt_suffix}.attempt{attempt}" if attempt_suffix else f".attempt{attempt}"
            attempt_out = output_clip + attempt_tag + ".mp4"

            try:
                await self.api_client.execute_direct_scene_video_edit(
                    source_clip_path=source_clip,
                    reference_image_path=reference_image_path,
                    edit_operation_prompt=effective_base_prompt,
                    output_clip_path=attempt_out,
                    audio_path=audio_path,
                    skip_audio_mux=skip_audio_mux,
                    avoid_operations=retry_focus if attempt > 1 else "",
                    positive_prompt=positive_focus if attempt > 1 else "",
                    retry_objective=missing_edits_focus if attempt > 1 else "",
                    reference_image_paths=active_ref_paths,
                    reference_image_role="entity_reference_grid",
                )
            except VideoEditRejectedError as exc:
                logger.warning(
                    "%s video edit rejected by model (attempt %d/%d): %s — "
                    "will retry",
                    scene_id,
                    attempt,
                    max_attempts,
                    exc,
                )
                if attempt >= max_attempts:
                    # All attempts exhausted — use original video.
                    logger.warning(
                        "%s — all %d attempts rejected, using original video",
                        scene_id,
                        max_attempts,
                    )
                    import shutil
                    if os.path.exists(source_clip):
                        shutil.copy2(source_clip, output_clip)
                    for path in attempt_outputs:
                        if os.path.exists(path):
                            try:
                                os.remove(path)
                            except OSError:
                                pass
                    with open(qa_sidecar, "w", encoding="utf-8") as fh:
                        json.dump(
                            {
                                "attempts": qa_history,
                                "final_passed": False,
                                "qa_enabled": qa_enabled,
                                "all_rejected": True,
                                "rejection_reason": str(exc),
                                "base_edit_operation_prompt": base_edit_prompt,
                            },
                            fh, indent=2, ensure_ascii=False,
                        )
                    return output_clip
                # Not last attempt — clean up partial output and continue to next attempt.
                if os.path.exists(attempt_out):
                    try:
                        os.remove(attempt_out)
                    except OSError:
                        pass
                continue
                return output_clip

            if not qa_enabled:
                os.replace(attempt_out, output_clip)
                return output_clip

            kf_dir = os.path.join(work_dir, f"edited_keyframes{attempt_tag}")
            edited_kf_paths = await asyncio.to_thread(
                extract_edited_keyframes_from_manifest,
                attempt_out,
                scene,
                kf_dir,
            )
            grid_dir = os.path.join(work_dir, f"grids{attempt_tag}")
            grid_a, grid_b = await asyncio.to_thread(
                build_keyframe_comparison_grids,
                original_kf_paths,
                edited_kf_paths,
                grid_dir,
            )

            attempt_outputs.append(attempt_out)
            attempt_grids.append(grid_a)

            qa = await self.api_client.validate_scene_video_edit_keyframe_grids(
                edited_keyframes_grid_path=grid_a,
                original_keyframes_grid_path=grid_b,
                entity_ref_image_paths=entity_ref_paths,
                entity_instru_json=entity_instru_text,
                edit_operation_prompt=base_edit_prompt,
            )
            qa["attempt"] = attempt
            qa_history.append(qa)
            with open(qa_sidecar, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "attempts": qa_history,
                        "final_passed": bool(qa.get("passed")),
                        "qa_enabled": qa_enabled,
                        "base_edit_operation_prompt": base_edit_prompt,
                        "effective_edit_operation_prompt": effective_base_prompt,
                        "reference_image_path": reference_image_path,
                        "reference_type": "entity_refs_before_after_grid",
                        "first_frame_constraint": False,
                        "last_positive_prompt": str(qa.get("positive_prompt", "")).strip(),
                        "last_retry_focus_prompt": str(qa.get("retry_focus_prompt", "")).strip(),
                        "last_missing_edits_prompt": str(qa.get("missing_edits_prompt", "")).strip(),
                        "discontinued_instruction_ids": sorted(discontinued_iids),
                        "active_spec_count": len(active_specs),
                    },
                    fh,
                    indent=2,
                    ensure_ascii=False,
                )

            if qa.get("passed"):
                logger.info("%s keyframe-grid QA passed (attempt %d)", scene_id, attempt)
                break

            retry_focus = str(qa.get("retry_focus_prompt", "")).strip()
            positive_focus = str(qa.get("positive_prompt", "")).strip()
            missing_edits_focus = str(qa.get("missing_edits_prompt", "")).strip()
            # Fallback: if VLM didn't produce a missing_edits_prompt, derive one
            # from failed_aspects so the retry objective is never empty.
            if not missing_edits_focus:
                failed = qa.get("failed_aspects") or []
                if isinstance(failed, list) and failed:
                    missing_edits_focus = "; ".join(
                        str(f).strip() for f in failed if str(f).strip()
                    )

            # ── Discontinue instructions with pasted entities ─────────────
            # When QA detects that an entity was "hard pasted" (a flat sticker
            # rather than a natural blend), the target is too difficult to edit
            # (e.g. too small, too distant, too occluded).  We abandon that
            # instruction on subsequent attempts instead of retrying with
            # another paste.
            if qa.get("pasted_entity_detected"):
                per_entity = qa.get("per_entity_results") or []
                if isinstance(per_entity, list):
                    for ent in per_entity:
                        if not isinstance(ent, dict):
                            continue
                        if ent.get("pasted_entity"):
                            iid = str(ent.get("instruction_id", "") or "").strip()
                            if iid and iid not in discontinued_iids:
                                discontinued_iids.add(iid)
                                logger.warning(
                                    "%s discontinuing instruction %s — pasted entity detected "
                                    "(edit too difficult, will not retry this instruction)",
                                    scene_id,
                                    iid,
                                )

            # Rebuild active specs + prompt + ref paths if any instructions
            # were discontinued.
            if discontinued_iids and active_specs:
                new_active = [s for s in active_specs if s.instruction_id not in discontinued_iids]
                if len(new_active) < len(active_specs):
                    active_specs = new_active
                    # Rebuild edit prompt from remaining specs.
                    effective_base_prompt = build_entity_edit_prompt(active_specs)
                    # Rebuild per-entity ref card paths to match active specs.
                    if reference_image_paths and active_specs:
                        ref_cards_dir = os.path.join(work_dir, "entity_ref_cards")
                        active_ref_paths = await asyncio.to_thread(
                            build_per_entity_reference_cards,
                            active_specs,
                            ref_cards_dir,
                        )
                    # Remove discontinued instruction_ids from missing_edits_focus
                    # so we don't tell the model to retry an abandoned edit.
                    if missing_edits_focus:
                        for iid in discontinued_iids:
                            # Crude but effective: drop sentences containing the iid.
                            missing_edits_focus = "; ".join(
                                part for part in missing_edits_focus.split(";")
                                if iid not in part
                            ).strip()
                    logger.info(
                        "%s rebuilt prompt for next attempt: %d active spec(s), "
                        "%d discontinued instruction(s)",
                        scene_id,
                        len(active_specs),
                        len(discontinued_iids),
                    )

            # ── LLM refinement: deduplicate & reorganize retry guidance ──
            # Send the raw QA-derived prompts to the LLM to remove redundancy,
            # identify critical problem areas, and produce a clean retry objective
            # with the avoid section emphasized at the end.
            try:
                refined = await self.api_client.refine_retry_guidance_with_llm(
                    base_edit_prompt=base_edit_prompt,
                    positive_prompt=positive_focus,
                    avoid_operations=retry_focus,
                    missing_edits_prompt=missing_edits_focus,
                    qa_feedback=str(qa.get("feedback", "") or "").strip(),
                    failed_aspects=[
                        str(f).strip()
                        for f in (qa.get("failed_aspects") or [])
                        if str(f).strip()
                    ],
                )
                positive_focus = refined.get("positive_prompt", positive_focus)
                retry_focus = refined.get("avoid_operations", retry_focus)
                missing_edits_focus = refined.get(
                    "missing_edits_prompt", missing_edits_focus
                )
                # Use the LLM-built retry_objective (includes critical problem areas)
                retry_obj = refined.get("retry_objective", "")
                if retry_obj:
                    missing_edits_focus = retry_obj
                logger.info(
                    "%s LLM retry guidance refined (attempt %d): refined=%s, critical=%s",
                    scene_id,
                    attempt,
                    refined.get("refined", False),
                    str(refined.get("critical_problem_areas", ""))[:120],
                )
            except Exception as refine_exc:
                logger.warning(
                    "%s LLM retry guidance refinement failed (attempt %d): %s — using raw guidance",
                    scene_id,
                    attempt,
                    refine_exc,
                )

            logger.warning(
                "%s keyframe-grid QA failed (attempt %d/%d): %s | avoid=%s | keep=%s",
                scene_id,
                attempt,
                max_attempts,
                qa.get("feedback", ""),
                retry_focus[:120],
                positive_focus[:120],
            )
            if attempt >= max_attempts:
                logger.warning(
                    "%s — all %d attempts exhausted, selecting best",
                    scene_id,
                    max_attempts,
                )
                break

        # ── Select the best attempt via VLM comparison ────────────────────
        best_idx = await self._select_best_attempt(
            scene_id=scene_id,
            original_kf_paths=original_kf_paths,
            attempt_grids=attempt_grids,
            entity_ref_paths=entity_ref_paths,
            entity_instru_text=entity_instru_text,
            base_edit_prompt=base_edit_prompt,
            qa_history=qa_history,
            work_dir=work_dir,
            qa_sidecar=qa_sidecar,
            attempt_suffix=attempt_suffix,
        )

        best_output = attempt_outputs[best_idx]
        logger.info("%s selected best attempt %d → %s", scene_id, best_idx + 1, best_output)

        # Move best to final output, clean up others.
        if best_output != output_clip:
            os.replace(best_output, output_clip)
        for i, path in enumerate(attempt_outputs):
            if i != best_idx and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        return output_clip

    async def _select_best_attempt(
        self,
        *,
        scene_id: str,
        original_kf_paths: List[str],
        attempt_grids: List[str],
        entity_ref_paths: List[str],
        entity_instru_text: str,
        base_edit_prompt: str,
        qa_history: List[dict],
        work_dir: str,
        qa_sidecar: str,
        attempt_suffix: str = "",
    ) -> int:
        """Use VLM to select the best attempt; fall back to QA scores on error."""
        if len(attempt_grids) <= 1:
            return 0

        # Build original grid for comparison if not already built.
        orig_grid = os.path.join(work_dir, f"original_grid{attempt_suffix}.png")
        if not os.path.exists(orig_grid) and original_kf_paths:
            from video_editing_agent.utils.multiview_ref_utils import save_keyframe_grid, REFERENCE_GRID_COLS
            await asyncio.to_thread(save_keyframe_grid, original_kf_paths, orig_grid, cols=REFERENCE_GRID_COLS)

        try:
            result = await self.api_client.select_best_video_edit_attempt(
                original_keyframes_grid_path=orig_grid,
                candidate_grid_paths=attempt_grids,
                entity_ref_image_paths=entity_ref_paths,
                entity_instru_json=entity_instru_text,
                edit_operation_prompt=base_edit_prompt,
            )
            best_idx = int(result.get("best_candidate_index", 0))
            best_idx = max(0, min(best_idx, len(attempt_grids) - 1))

            # Record selection in QA sidecar.
            try:
                with open(qa_sidecar, "r", encoding="utf-8") as fh:
                    sidecar_data = json.load(fh)
                sidecar_data["best_attempt_selection"] = result
                sidecar_data["best_attempt_index"] = best_idx
                with open(qa_sidecar, "w", encoding="utf-8") as fh:
                    json.dump(sidecar_data, fh, indent=2, ensure_ascii=False)
            except (OSError, json.JSONDecodeError):
                pass

            logger.info(
                "%s VLM best-attempt selection: index=%d, reasoning=%s",
                scene_id,
                best_idx,
                str(result.get("reasoning", ""))[:200],
            )
            return best_idx
        except Exception as exc:
            logger.warning(
                "%s VLM best-attempt selection failed: %s — falling back to QA scores",
                scene_id,
                exc,
            )

        # Fallback: pick the attempt with the highest QA score.
        best_idx = 0
        best_score = -1.0
        for i, qa in enumerate(qa_history):
            score = float(qa.get("score", 0.0) or 0.0)
            passed = bool(qa.get("passed", False))
            if passed:
                return i
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    @staticmethod
    def _qa_enabled() -> bool:
        return os.environ.get("VIDEO_EDIT_QA", "true").lower() not in (
            "0",
            "false",
            "no",
        )

    async def _edit_scene_chunked(
        self,
        *,
        scene: SceneClip,
        source_clip: str,
        output_clip: str,
        work_dir: str,
        entity_ref_paths: List[str],
        entity_instru_text: str,
        base_edit_prompt: str,
        reference_image_path: str,
        max_chunk_sec: float,
        min_tail_sec: float,
        reference_image_paths: Optional[List[str]] = None,
    ) -> str:
        """Edit long scenes by splitting into chunks; retry up to MAX_EDIT_ATTEMPTS times.

        All attempt outputs are preserved. After all attempts, VLM selects the
        best result by comparing their keyframe grids.
        """
        scene_id = scene.scene_id
        original_kf_paths = original_keyframe_paths(scene)
        qa_sidecar = os.path.join(work_dir, "qa_chunked.json")
        retry_focus = ""
        positive_focus = ""
        missing_edits_focus = ""
        qa_history: List[dict] = []
        qa_enabled = self._qa_enabled()
        max_attempts = MAX_EDIT_ATTEMPTS if qa_enabled else 1
        audio_path = scene.audio_path or ""

        attempt_staged: List[str] = []
        attempt_grids: List[str] = []

        for attempt in range(1, max_attempts + 1):
            avoid = retry_focus if attempt > 1 else ""
            positive = positive_focus if attempt > 1 else ""
            retry_obj = missing_edits_focus if attempt > 1 else ""
            try:
                staged = await self._render_chunked_edit(
                    scene=scene,
                    source_clip=source_clip,
                    work_dir=work_dir,
                    base_edit_prompt=base_edit_prompt,
                    reference_image_path=reference_image_path,
                    reference_image_paths=reference_image_paths,
                    max_chunk_sec=max_chunk_sec,
                    min_tail_sec=min_tail_sec,
                    avoid_operations=avoid,
                    positive_prompt=positive,
                    retry_objective=retry_obj,
                    attempt=attempt,
                )
            except VideoEditRejectedError as exc:
                logger.warning(
                    "%s chunked video edit rejected by model (attempt %d/%d): %s — "
                    "will retry",
                    scene_id,
                    attempt,
                    max_attempts,
                    exc,
                )
                if attempt >= max_attempts:
                    logger.warning(
                        "%s — all %d chunked attempts rejected, using original video",
                        scene_id,
                        max_attempts,
                    )
                    import shutil
                    if os.path.exists(source_clip):
                        shutil.copy2(source_clip, output_clip)
                    for path in attempt_staged:
                        if os.path.exists(path):
                            try:
                                os.remove(path)
                            except OSError:
                                pass
                    with open(qa_sidecar, "w", encoding="utf-8") as fh:
                        json.dump(
                            {
                                "attempts": qa_history,
                                "final_passed": False,
                                "qa_enabled": qa_enabled,
                                "all_rejected": True,
                                "rejection_reason": str(exc),
                                "base_edit_operation_prompt": base_edit_prompt,
                            },
                            fh, indent=2, ensure_ascii=False,
                        )
                    return output_clip
                continue

            if not qa_enabled:
                os.replace(staged, output_clip)
                return output_clip

            kf_dir = os.path.join(work_dir, f"edited_keyframes.chunked.attempt{attempt}")
            edited_kf_paths = await asyncio.to_thread(
                extract_edited_keyframes_from_manifest,
                staged,
                scene,
                kf_dir,
            )
            grid_dir = os.path.join(work_dir, f"grids.chunked.attempt{attempt}")
            grid_a, grid_b = await asyncio.to_thread(
                build_keyframe_comparison_grids,
                original_kf_paths,
                edited_kf_paths,
                grid_dir,
            )

            attempt_staged.append(staged)
            attempt_grids.append(grid_a)

            qa = await self.api_client.validate_scene_video_edit_keyframe_grids(
                edited_keyframes_grid_path=grid_a,
                original_keyframes_grid_path=grid_b,
                entity_ref_image_paths=entity_ref_paths,
                entity_instru_json=entity_instru_text,
                edit_operation_prompt=base_edit_prompt,
            )
            qa["attempt"] = attempt
            qa_history.append(qa)
            with open(qa_sidecar, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "attempts": qa_history,
                        "final_passed": bool(qa.get("passed")),
                        "qa_enabled": qa_enabled,
                        "base_edit_operation_prompt": base_edit_prompt,
                        "last_missing_edits_prompt": str(qa.get("missing_edits_prompt", "")).strip(),
                    },
                    fh,
                    indent=2,
                    ensure_ascii=False,
                )

            if qa.get("passed"):
                logger.info(
                    "%s chunked keyframe-grid QA passed (attempt %d)",
                    scene_id,
                    attempt,
                )
                break

            retry_focus = str(qa.get("retry_focus_prompt", "")).strip()
            positive_focus = str(qa.get("positive_prompt", "")).strip()
            missing_edits_focus = str(qa.get("missing_edits_prompt", "")).strip()
            if not missing_edits_focus:
                failed = qa.get("failed_aspects") or []
                if isinstance(failed, list) and failed:
                    missing_edits_focus = "; ".join(
                        str(f).strip() for f in failed if str(f).strip()
                    )

            # ── LLM refinement: deduplicate & reorganize retry guidance ──
            try:
                refined = await self.api_client.refine_retry_guidance_with_llm(
                    base_edit_prompt=base_edit_prompt,
                    positive_prompt=positive_focus,
                    avoid_operations=retry_focus,
                    missing_edits_prompt=missing_edits_focus,
                    qa_feedback=str(qa.get("feedback", "") or "").strip(),
                    failed_aspects=[
                        str(f).strip()
                        for f in (qa.get("failed_aspects") or [])
                        if str(f).strip()
                    ],
                )
                positive_focus = refined.get("positive_prompt", positive_focus)
                retry_focus = refined.get("avoid_operations", retry_focus)
                missing_edits_focus = refined.get(
                    "missing_edits_prompt", missing_edits_focus
                )
                retry_obj = refined.get("retry_objective", "")
                if retry_obj:
                    missing_edits_focus = retry_obj
                logger.info(
                    "%s chunked LLM retry guidance refined (attempt %d): refined=%s",
                    scene_id,
                    attempt,
                    refined.get("refined", False),
                )
            except Exception as refine_exc:
                logger.warning(
                    "%s chunked LLM retry guidance refinement failed (attempt %d): %s — using raw guidance",
                    scene_id,
                    attempt,
                    refine_exc,
                )

            logger.warning(
                "%s chunked keyframe-grid QA failed (attempt %d/%d): %s | avoid=%s | keep=%s | missing=%s",
                scene_id,
                attempt,
                max_attempts,
                qa.get("feedback", ""),
                retry_focus[:120],
                positive_focus[:120],
                missing_edits_focus[:120],
            )
            if attempt >= max_attempts:
                logger.warning(
                    "%s — all %d chunked attempts exhausted, selecting best",
                    scene_id,
                    max_attempts,
                )
                break

        # ── Select the best attempt via VLM comparison ────────────────────
        best_idx = await self._select_best_attempt(
            scene_id=scene_id,
            original_kf_paths=original_kf_paths,
            attempt_grids=attempt_grids,
            entity_ref_paths=entity_ref_paths,
            entity_instru_text=entity_instru_text,
            base_edit_prompt=base_edit_prompt,
            qa_history=qa_history,
            work_dir=work_dir,
            qa_sidecar=qa_sidecar,
            attempt_suffix=".chunked",
        )

        best_staged = attempt_staged[best_idx]
        logger.info("%s selected best chunked attempt %d", scene_id, best_idx + 1)

        if best_staged and os.path.exists(best_staged):
            await asyncio.to_thread(
                mux_video_with_scene_audio,
                best_staged,
                source_clip,
                output_clip,
                audio_path=audio_path or None,
            )
            # Clean up non-best staged files.
            for i, path in enumerate(attempt_staged):
                if i != best_idx and path != output_clip and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

        logger.info(
            "%s chunked direct edit done → %s",
            scene_id,
            output_clip,
        )
        return output_clip

    async def _render_chunked_edit(
        self,
        *,
        scene: SceneClip,
        source_clip: str,
        work_dir: str,
        base_edit_prompt: str,
        reference_image_path: str,
        max_chunk_sec: float,
        min_tail_sec: float,
        avoid_operations: str,
        positive_prompt: str,
        retry_objective: str = "",
        reference_image_paths: Optional[List[str]] = None,
        attempt: int,
    ) -> str:
        """Edit all chunks sequentially with entity_refs and boundary continuity.

        Each chunk still receives the entity before/after reference grid. Later chunks
        also receive a composite reference that includes the previous edited tail frame
        as a continuity cue, but it is not an output-frame template.
        """
        scene_id = scene.scene_id
        chunks_dir = scene_chunks_dir(self.config.scenes_dir, scene_id)
        chunk_attempt_dir = os.path.join(chunks_dir, f"attempt_{attempt:02d}")
        manifest_entries, _ = await asyncio.to_thread(
            split_scene_video_into_chunks,
            source_clip,
            chunk_attempt_dir,
            max_chunk_sec,
            min_tail_sec=min_tail_sec,
        )

        edited_paths: List[str] = []
        previous_edited_last_frame = ""
        effective_base = base_edit_prompt
        retry_guidance = build_edit_retry_guidance_section(
            positive_prompt=positive_prompt,
            avoid_operations=avoid_operations,
            retry_objective=retry_objective,
        )
        if retry_guidance:
            effective_base += retry_guidance

        for entry in manifest_entries:
            idx = entry.index
            chunk_src = entry.source_path
            chunk_out = chunk_edited_path(chunk_attempt_dir, idx)
            chunk_orig_first = chunk_first_frame_path(chunk_attempt_dir, idx)
            chunk_last_png = chunk_edited_last_frame_path(chunk_attempt_dir, idx)

            if idx == 0:
                ref_frame = reference_image_path
                chunk_prompt = effective_base
            else:
                if not previous_edited_last_frame or not os.path.exists(previous_edited_last_frame):
                    raise RuntimeError(
                        f"Missing edited last frame from chunk {idx - 1} for {scene_id}"
                    )
                ref_frame = os.path.join(
                    chunk_attempt_dir,
                    f"chunk_{idx:04d}_continuity_entity_reference_grid.png",
                )
                await asyncio.to_thread(
                    build_chunk_reference_grid,
                    reference_image_path,
                    previous_edited_last_frame,
                    ref_frame,
                )
                chunk_prompt = await self.api_client.derive_video_chunk_edit_operation_prompt(
                    original_chunk_first_frame_path=chunk_orig_first,
                    previous_edited_last_frame_path=previous_edited_last_frame,
                )
                chunk_prompt += "\n\n" + effective_base + CHUNK_CONTINUITY_PROMPT

            # For chunk 0, send per-entity ref cards as separate references.
            # For later chunks, only the composite continuity grid is available.
            chunk_ref_paths = reference_image_paths if idx == 0 else None

            await self.api_client.execute_direct_scene_video_edit(
                source_clip_path=chunk_src,
                reference_image_path=ref_frame,
                edit_operation_prompt=chunk_prompt,
                output_clip_path=chunk_out,
                skip_audio_mux=True,
                reference_image_paths=chunk_ref_paths,
                reference_image_role="entity_reference_grid",
            )

            # Extract last frame as a continuity cue for the next chunk.
            await asyncio.to_thread(extract_last_frame, chunk_out, chunk_last_png)
            previous_edited_last_frame = chunk_last_png
            edited_paths.append(chunk_out)

            update_chunk_manifest_entry(
                chunk_attempt_dir,
                idx,
                edited_path=chunk_out,
                edited_last_frame_path=chunk_last_png,
                edit_operation_prompt=chunk_prompt,
            )

        staged = os.path.join(work_dir, f"chunked_concat.attempt{attempt}.mp4")
        await asyncio.to_thread(
            concat_edited_chunks,
            edited_paths,
            staged,
            dedup_boundary_frames=True,
        )
        return staged
