"""Mask creation, propagation, and compositing utilities."""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from video_editing_agent.prompts.templates import KEYFRAME_PRESERVE_FRAME_STRUCTURE_CLAUSE

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

# Distinct RGB colors for multi-entity masks
ENTITY_COLORS_RGB: List[Tuple[int, int, int]] = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert ``#RRGGBB`` to RGB tuple."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def build_inpaint_edit_directives(
    edits: List[Dict[str, str]],
) -> str:
    """Build color-mask-guided edit clauses for inpainting prompt.

    Each item needs: color_name, subject_features, edit_prompt.
    """
    clauses: List[str] = []
    for item in edits:
        color_name = item.get("color_name", "red")
        subject = (item.get("subject_features") or "").strip()
        edit = (item.get("edit_prompt") or "").strip()
        clauses.append(
            f'According to the {color_name} mask indicated region in image 1, '
            f'on image 2 for "{subject}", execute: {edit}'
        )
    return "; ".join(clauses) + ("." if clauses else "")


def build_keyframe_inpaint_directives(
    edits: List[Dict[str, str]],
) -> str:
    """Build Module-3 mask-color edit clauses (legacy mask-guided inpaint)."""
    clauses: List[str] = []
    for item in edits:
        color_name = item.get("color_name", "red")
        instruction_id = (item.get("instruction_id") or "").strip()
        edit = (item.get("edit_prompt") or "").strip()
        clauses.append(
            f'According to the {color_name} mask indicated region in image 1, '
            f'on image 2, for instruction {instruction_id}, execute: {edit}. '
            f'Identify the edit target ONLY from the attached entity reference '
            f'images for {instruction_id}; do not infer the subject from text'
        )
    return "; ".join(clauses) + ("." if clauses else "")


def _clamp_one_sentence(text: str, *, max_chars: int = 140) -> str:
    """Reduce verbose location text to a single short sentence."""
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    for sep in (". ", "; ", " — ", " - ", "。 "):
        if sep in cleaned:
            cleaned = cleaned.split(sep, 1)[0].strip()
            break
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0].strip()
    if cleaned and not cleaned.endswith("."):
        cleaned += "."
    return cleaned


def compose_keyframe_edit_boundary_text(
    location_record: Dict[str, object],
    *,
    subject_features: str = "",
) -> str:
    """Build precise inclusion/exclusion boundary language for keyframe inpaint."""
    parts: List[str] = []
    includes = _as_str_list(location_record.get("edit_includes"))
    excludes = _as_str_list(location_record.get("edit_excludes"))
    boundary = str(location_record.get("boundary_notes", "") or "").strip()
    distinguishes = _as_str_list(location_record.get("distinguishes_from_other_instructions"))

    if includes:
        parts.append(f"Edit ONLY these parts of the target: {', '.join(includes)}.")
    if excludes:
        parts.append(
            "Do NOT edit, remove, erase, or alter these separate adjacent or touching "
            f"elements: {', '.join(excludes)}."
        )
    if boundary:
        parts.append(boundary)
    if distinguishes:
        parts.append(
            "This target is NOT the same as: " + "; ".join(distinguishes) + "."
        )

    subject = subject_features.strip()
    if not parts and subject:
        parts.append(
            f'Confine the edit strictly to the subject "{subject}" and its own silhouette; '
            "preserve every distinct object, prop, furniture piece, or person that is "
            "touching, overlapping, or beside the subject."
        )
    return " ".join(parts)


def compose_keyframe_edit_location_prompt(
    location_record: Dict[str, object],
    *,
    subject_features: str = "",
    color_name: str = "",
) -> str:
    """Turn structured VLM location output into one short sentence for inpaint."""
    subject = subject_features.strip()
    if not subject:
        cues = _as_str_list(location_record.get("identity_cues"))
        subject = cues[0] if cues else ""

    spatial = str(location_record.get("spatial_region", "") or "").strip()
    landmarks = _as_str_list(location_record.get("landmark_relations"))
    where = (
        spatial
        if _is_meaningful_location_text(spatial)
        else (landmarks[0] if landmarks else "")
    )

    if subject and where:
        return _clamp_one_sentence(f"{subject}, {where}")
    if subject:
        return _clamp_one_sentence(subject)

    explicit = str(location_record.get("location_edit_prompt", "") or "").strip()
    if _is_meaningful_location_text(explicit):
        return _clamp_one_sentence(explicit)

    if color_name:
        return f"The {color_name} edit target in image 1."
    return "The edit target in image 1."


def compose_keyframe_location_hint(
    location_record: Dict[str, object],
    *,
    subject_features: str = "",
) -> str:
    """Build a concise, edit-oriented location hint from structured VLM fields."""
    parts: List[str] = []

    identity: List[str] = []
    if subject_features.strip():
        identity.append(subject_features.strip())
    identity.extend(_as_str_list(location_record.get("identity_cues")))
    identity = list(dict.fromkeys(identity))
    if identity:
        parts.append("the subject with " + "; ".join(identity))

    spatial = str(location_record.get("spatial_region", "") or "").strip()
    landmarks = _as_str_list(location_record.get("landmark_relations"))
    visible = _as_str_list(location_record.get("visible_body_parts"))
    vp_det = str(location_record.get("viewpoint_in_detection", "") or "").strip()

    where_parts: List[str] = []
    if spatial and _is_meaningful_location_text(spatial):
        where_parts.append(spatial)
    where_parts.extend(landmarks[:2])
    if where_parts:
        parts.append("located at " + " — ".join(where_parts))
    if visible:
        parts.append("visible parts: " + ", ".join(visible))
    if vp_det and _is_meaningful_location_text(vp_det):
        parts.append(f"viewpoint: {vp_det}")

    if not parts:
        return ""
    return ", ".join(parts)


def build_keyframe_delete_identification_image(
    src_path: str,
    mask_path: str,
) -> Image.Image:
    """Single-panel delete-target identification crop for keyframe inpaint."""
    frame = Image.open(src_path).convert("RGB")
    mask = Image.open(mask_path).convert("RGB")
    return _label_entity_panel(
        _fit_entity_bbox_panel(frame, mask),
        "Delete target",
    )


def build_keyframe_location_edit_directives(
    edits: List[Dict[str, str]],
) -> str:
    """Build compact Module-3 edit clauses with original edit_prompt foregrounded."""
    lines: List[str] = []
    for item in edits:
        instruction_id = (item.get("instruction_id") or "").strip()
        edit = (item.get("edit_prompt") or "").strip()
        location_prompt = _clamp_one_sentence(item.get("location_prompt") or "")
        subject = (item.get("subject_features") or "").strip()
        color_name = (item.get("color_name") or "").strip()
        action = (item.get("action") or "").strip().lower()
        if not location_prompt:
            location_prompt = _clamp_one_sentence(subject) or f"instruction {instruction_id} target"
        color_tag = f" ({color_name})" if color_name else ""
        line = (
            f"[{instruction_id}]{color_tag} ORIGINAL EDIT: {edit} "
            f"| Target in image 1: {location_prompt}"
        )
        if action == "delete":
            line += (
                " | DELETE SCOPE: remove ONLY this target and naturally inpaint the "
                "revealed background from surrounding scene context — never blank, "
                "whiten, or erase the full frame"
            )
        lines.append(line)

    if not lines:
        return ""

    return (
        " ".join(lines)
        + " Execute ALL ORIGINAL EDIT lines above on image 1 (the video keyframe). "
        "Location text locates the target only; the EDIT text defines what to change. "
        "Leave all other regions unchanged. "
        + KEYFRAME_PRESERVE_FRAME_STRUCTURE_CLAUSE
    )


def load_entity_ref_overlay_guide(
    overlay_path: str,
    frame_size: Tuple[int, int],
) -> Image.Image:
    """Load overlay fusion reference, stripping optional caption band."""
    return extract_reference_frame_from_composite(
        Image.open(overlay_path).convert("RGB"),
        frame_size,
    )


def collect_instruction_entity_ref_paths(
    ref_dir: str,
    instruction_id: str,
    *,
    include_canonical: bool = True,
) -> Dict[str, str]:
    """Collect existing entity_refs assets for one instruction."""
    paths: Dict[str, str] = {}
    resolvers = [
        ("src", entity_ref_src_path),
        ("mask", entity_ref_mask_path),
        ("overlay", entity_ref_overlay_path),
    ]
    if include_canonical:
        resolvers.append(("canonical", entity_ref_canonical_path))
    for key, resolver in resolvers:
        path = resolver(ref_dir, instruction_id)
        if os.path.exists(path):
            paths[key] = path
    multiview = entity_ref_multiview_path(ref_dir, instruction_id)
    if os.path.exists(multiview):
        paths["multiview"] = multiview
        paths["front"] = multiview
    multiview_edited = entity_ref_multiview_edited_path(ref_dir, instruction_id)
    if os.path.exists(multiview_edited):
        paths["multiview_edited"] = multiview_edited
        paths["front_edited"] = multiview_edited
    return paths


def collect_keyframe_entity_ref_paths(
    ref_dir: str,
    instruction_id: str,
    *,
    action: str = "",
) -> Dict[str, str]:
    """Collect entity_refs for Module-3 keyframe inpaint.

    Delete uses src+mask identification crop only — not the before/after card whose
    empty right panel can mislead the model into whitening the whole output frame.
    """
    action_key = (action or "").strip().lower()
    if action_key == "delete":
        paths: Dict[str, str] = {}
        multiview = entity_ref_multiview_path(ref_dir, instruction_id)
        if os.path.exists(multiview):
            paths["multiview"] = multiview
            paths["front"] = multiview
        src = entity_ref_src_path(ref_dir, instruction_id)
        mask = entity_ref_mask_path(ref_dir, instruction_id)
        if os.path.exists(src):
            paths["src"] = src
        if os.path.exists(mask):
            paths["mask"] = mask
        return paths

    canonical = entity_ref_canonical_path(ref_dir, instruction_id)
    if os.path.exists(canonical):
        out: Dict[str, str] = {"canonical": canonical}
        multiview = entity_ref_multiview_path(ref_dir, instruction_id)
        if os.path.exists(multiview):
            out["multiview"] = multiview
            out["front"] = multiview
        multiview_edited = entity_ref_multiview_edited_path(ref_dir, instruction_id)
        if os.path.exists(multiview_edited):
            out["multiview_edited"] = multiview_edited
            out["front_edited"] = multiview_edited
        return out
    return collect_instruction_entity_ref_paths(
        ref_dir,
        instruction_id,
        include_canonical=True,
    )


def entity_color_name(index: int) -> str:
    """English color name for entity palette index."""
    names = ["red", "green", "blue", "yellow", "magenta", "cyan"]
    return names[index % len(names)]


def color_name_from_hex(hex_color: str) -> str:
    """Map palette hex back to English color name."""
    normalized = hex_color.upper()
    for i in range(len(ENTITY_COLORS_RGB)):
        if entity_color_hex(i).upper() == normalized:
            return entity_color_name(i)
    return "red"


ENTITY_COLOR_REGISTRY_FILENAME = "entity_color_registry.json"
DEFAULT_PALETTE_MIN_COSINE = 0.75
RELAXED_PALETTE_MIN_COSINE = 0.65
MASK_LOCATION_CONFIDENT_MIN = 0.55


class EntityColorRegistry:
    """Workspace-persistent stable ``entity_id`` → palette color mapping."""

    def __init__(self, registry_path: str) -> None:
        self.registry_path = registry_path
        self._mapping: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.registry_path):
            return
        try:
            data = json.loads(open(self.registry_path, encoding="utf-8").read())
            raw = data.get("entity_colors") or data
            if isinstance(raw, dict):
                self._mapping = {
                    str(k): str(v).upper()
                    for k, v in raw.items()
                    if k and v
                }
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to load entity color registry: %s", exc)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.registry_path) or ".", exist_ok=True)
        payload = {
            "version": "1.0",
            "entity_colors": self._mapping,
        }
        with open(self.registry_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    def color_for(self, entity_id: str) -> str:
        """Return stable palette hex for ``entity_id``, assigning on first use."""
        if not entity_id:
            return entity_color_hex(0)
        if entity_id in self._mapping:
            return self._mapping[entity_id]

        used = set(self._mapping.values())
        for i in range(len(ENTITY_COLORS_RGB)):
            hex_color = entity_color_hex(i)
            if hex_color not in used:
                self._mapping[entity_id] = hex_color
                self.save()
                return hex_color

        idx = abs(hash(entity_id)) % len(ENTITY_COLORS_RGB)
        hex_color = entity_color_hex(idx)
        self._mapping[entity_id] = hex_color
        self.save()
        return hex_color

    def build_color_maps(
        self,
        entity_ids: List[str],
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Build hex and English name maps for the given entity ids."""
        color_map = {eid: self.color_for(eid) for eid in entity_ids if eid}
        name_map = {eid: color_name_from_hex(hex_c) for eid, hex_c in color_map.items()}
        return color_map, name_map


def quantize_mask_to_palette(
    mask_img: Image.Image,
    entity_color_map: Dict[str, str],
    *,
    keep_entity_ids: Optional[List[str]] = None,
    min_magnitude: int = 12,
    min_cosine: float = DEFAULT_PALETTE_MIN_COSINE,
) -> Image.Image:
    """Assign each active pixel to exactly one palette color (winner-take-all)."""
    entity_ids = keep_entity_ids or [
        eid for eid in entity_color_map if entity_color_map.get(eid)
    ]
    palette: List[Tuple[str, Tuple[int, int, int]]] = []
    for eid in entity_ids:
        hex_color = entity_color_map.get(eid)
        if not hex_color:
            continue
        palette.append((eid, hex_to_rgb(hex_color)))

    if not palette:
        return mask_img

    arr = np.array(mask_img.convert("RGB"))
    magnitude = np.max(arr, axis=2)
    valid = magnitude >= min_magnitude
    if not np.any(valid):
        return Image.new("RGB", mask_img.size, (0, 0, 0))

    pixels = arr.astype(np.float32)
    norms = np.maximum(np.linalg.norm(pixels, axis=2), 1e-6)
    unit = pixels / norms[:, :, np.newaxis]

    best_scores = np.full(arr.shape[:2], -1.0, dtype=np.float32)
    best_rgb = np.zeros_like(arr)

    for _eid, rgb in palette:
        target = np.array(rgb, dtype=np.float32)
        target_unit = target / max(float(np.linalg.norm(target)), 1e-6)
        cosine = np.sum(unit * target_unit.reshape(1, 1, 3), axis=2)
        cosine = np.where(valid, cosine, -1.0)
        better = cosine > best_scores
        best_scores = np.where(better, cosine, best_scores)
        rgb_arr = np.array(rgb, dtype=np.uint8)
        best_rgb = np.where(better[..., np.newaxis], rgb_arr, best_rgb)

    out = np.zeros_like(arr)
    keep = best_scores >= min_cosine
    out[keep] = best_rgb[keep]
    return Image.fromarray(out, mode="RGB")


def quantize_mask_to_palette_best_effort(
    mask_img: Image.Image,
    entity_color_map: Dict[str, str],
    *,
    keep_entity_ids: Optional[List[str]] = None,
) -> Image.Image:
    """Quantize to palette; relax cosine thresholds if strict pass is empty."""
    strict = quantize_mask_to_palette(
        mask_img,
        entity_color_map,
        keep_entity_ids=keep_entity_ids,
    )
    if image_has_mask_content(strict) or not image_has_mask_content(mask_img):
        return strict

    relaxed = quantize_mask_to_palette(
        mask_img,
        entity_color_map,
        keep_entity_ids=keep_entity_ids,
        min_cosine=RELAXED_PALETTE_MIN_COSINE,
    )
    if image_has_mask_content(relaxed):
        logger.info(
            "Mask palette quantization recovered with relaxed cosine %.2f",
            RELAXED_PALETTE_MIN_COSINE,
        )
        return relaxed

    very_relaxed = quantize_mask_to_palette(
        mask_img,
        entity_color_map,
        keep_entity_ids=keep_entity_ids,
        min_cosine=0.55,
        min_magnitude=8,
    )
    if image_has_mask_content(very_relaxed):
        logger.info("Mask palette quantization recovered with cosine 0.55")
        return very_relaxed
    return strict


def save_mask_color_map_debug(
    mask_path: str,
    entity_color_map: Dict[str, str],
    entity_color_name_map: Optional[Dict[str, str]] = None,
) -> str:
    """Write per-entity color coverage stats beside a scene mask."""
    debug_path = f"{mask_path}.color_map.json"
    stats: Dict[str, object] = {"entity_colors": dict(entity_color_map)}
    if entity_color_name_map:
        stats["entity_color_names"] = dict(entity_color_name_map)

    if os.path.exists(mask_path):
        mask_img = Image.open(mask_path).convert("RGB")
        coverage: Dict[str, float] = {}
        for eid, hex_color in entity_color_map.items():
            coverage[eid] = mask_color_coverage_ratio(mask_img, hex_color)
        stats["coverage"] = coverage

    with open(debug_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    return debug_path


def entity_ref_meta_path(ref_dir: str, instruction_id: str) -> str:
    """Sidecar metadata path for an entity reference bundle."""
    return os.path.join(ref_dir, f"{instruction_id}_ref.meta.json")


def entity_ref_meta_path_from_overlay(overlay_path: str) -> str:
    """Resolve meta path adjacent to an overlay PNG."""
    if overlay_path.endswith("_ref.png"):
        return overlay_path[: -len("_ref.png")] + "_ref.meta.json"
    base, _ = os.path.splitext(overlay_path)
    return f"{base}.meta.json"


def save_entity_ref_meta(
    ref_dir: str,
    instruction_id: str,
    *,
    entity_id: str,
    color_hex: str,
    color_name: str,
) -> str:
    """Persist stable color metadata for a reference overlay."""
    path = entity_ref_meta_path(ref_dir, instruction_id)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "instruction_id": instruction_id,
        "entity_id": entity_id,
        "color_hex": color_hex.upper(),
        "color_name": color_name,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


def load_entity_ref_meta(
    ref_dir: str,
    instruction_id: str,
) -> Optional[Dict[str, str]]:
    """Load reference color metadata if present."""
    path = entity_ref_meta_path(ref_dir, instruction_id)
    if not os.path.exists(path):
        return None
    try:
        data = json.loads(open(path, encoding="utf-8").read())
        if not isinstance(data, dict):
            return None
        return {
            "entity_id": str(data.get("entity_id", "")),
            "color_hex": str(data.get("color_hex", "")).upper(),
            "color_name": str(data.get("color_name", "")),
            "instruction_id": str(data.get("instruction_id", instruction_id)),
        }
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        logger.warning("Failed to load entity ref meta %s: %s", path, exc)
        return None


def load_entity_ref_meta_from_overlay(overlay_path: str) -> Optional[Dict[str, str]]:
    """Load reference metadata from the overlay image path."""
    meta_path = entity_ref_meta_path_from_overlay(overlay_path)
    if not os.path.exists(meta_path):
        return None
    try:
        data = json.loads(open(meta_path, encoding="utf-8").read())
        if not isinstance(data, dict):
            return None
        return {
            "entity_id": str(data.get("entity_id", "")),
            "color_hex": str(data.get("color_hex", "")).upper(),
            "color_name": str(data.get("color_name", "")),
            "instruction_id": str(data.get("instruction_id", "")),
        }
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        logger.warning("Failed to load entity ref meta %s: %s", meta_path, exc)
        return None


def resolve_reference_overlay_color(
    overlay_path: str,
    *,
    fallback_hex: str = "",
) -> Tuple[str, str]:
    """Read the actual palette color used in a saved reference overlay."""
    meta = load_entity_ref_meta_from_overlay(overlay_path)
    if meta:
        hex_color = meta.get("color_hex", "").upper()
        name = meta.get("color_name", "") or color_name_from_hex(hex_color)
        if hex_color:
            return hex_color, name

    mask_path = overlay_path.replace("_ref.png", "_ref_mask.png")
    if os.path.exists(mask_path):
        mask_img = Image.open(mask_path).convert("RGB")
        best_hex = ""
        best_ratio = 0.0
        for i in range(len(ENTITY_COLORS_RGB)):
            hex_color = entity_color_hex(i)
            ratio = mask_color_coverage_ratio(mask_img, hex_color)
            if ratio > best_ratio:
                best_ratio = ratio
                best_hex = hex_color
        if best_hex and best_ratio > 0.0001:
            return best_hex, color_name_from_hex(best_hex)

    if fallback_hex:
        return fallback_hex.upper(), color_name_from_hex(fallback_hex)
    return "", ""


def _as_str_list(value: object) -> List[str]:
    """Coerce JSON list or string into a clean string list."""
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def compose_robust_location_hint(
    location_record: Dict[str, object],
    *,
    subject_features: str = "",
) -> str:
    """Build a viewpoint-robust location hint for reference-location VLM pipeline."""
    parts: List[str] = [
        "VIEWPOINT-INVARIANT MATCH: The subject may differ from the reference in pose, "
        "orientation (front/back/side), partial occlusion, blur, or scale. Match identity, not pose.",
    ]

    identity: List[str] = []
    if subject_features.strip():
        identity.append(subject_features.strip())
    identity.extend(_as_str_list(location_record.get("identity_cues")))
    identity = list(dict.fromkeys(identity))
    if identity:
        parts.append("Identity anchors: " + "; ".join(identity))

    vp_ref = str(location_record.get("viewpoint_in_reference", "") or "").strip()
    vp_det = str(location_record.get("viewpoint_in_detection", "") or "").strip()
    if vp_ref or vp_det:
        parts.append(
            f"Reference viewpoint: {vp_ref or 'unknown'}. "
            f"Detection-frame viewpoint: {vp_det or 'unknown'}."
        )

    visible = _as_str_list(location_record.get("visible_body_parts"))
    if visible:
        parts.append("Visible in image 1 (segment these parts): " + ", ".join(visible))

    spatial = str(location_record.get("spatial_region", "") or "").strip()
    landmarks = _as_str_list(location_record.get("landmark_relations"))
    where_parts: List[str] = []
    if spatial:
        where_parts.append(spatial)
    if landmarks:
        where_parts.append("; ".join(landmarks))
    if where_parts:
        parts.append("Where in image 1: " + " — ".join(where_parts))

    freeform = str(
        location_record.get("location_edit_prompt", "")
        or location_record.get("location_prompt", "")
        or "",
    ).strip()
    if freeform and freeform.lower() not in _PLACEHOLDER_LOCATION_VALUES:
        parts.append(freeform)

    parts.append(
        "Segment the full visible silhouette in image 1, including back, side, or partial views; "
        "do not require front-facing pose or identical orientation to the reference."
    )
    return " ".join(parts)


def compose_mask_detected_location_text(
    location_record: Dict[str, object],
    *,
    subject_features: str = "",
) -> str:
    """Concise location phrase merged into a located entity's segmentation line."""
    explicit = str(
        location_record.get("location_edit_prompt", "")
        or location_record.get("location_prompt", "")
        or "",
    ).strip()
    if explicit:
        return explicit

    parts: List[str] = []
    spatial = str(location_record.get("spatial_region", "") or "").strip()
    if spatial:
        parts.append(spatial)
    parts.extend(_as_str_list(location_record.get("landmark_relations"))[:2])
    vp_det = str(location_record.get("viewpoint_in_detection", "") or "").strip()
    if vp_det:
        parts.append(vp_det)
    visible = _as_str_list(location_record.get("visible_body_parts"))[:3]
    if visible:
        parts.append("visible: " + ", ".join(visible))
    if parts:
        return " — ".join(parts)
    return subject_features.strip()


def entity_location_confident_for_mask_retry(
    record: Dict[str, object],
    *,
    min_confidence: float = MASK_LOCATION_CONFIDENT_MIN,
) -> bool:
    """Return True when VLM location is strong enough to justify mask re-segmentation."""
    if not bool(record.get("present_in_frame", False)):
        return False
    if not keyframe_location_has_spatial_evidence(record):
        return False
    confidence = _location_record_confidence(record)
    if confidence is None:
        return False
    try:
        if float(confidence) < min_confidence:
            return False
    except (TypeError, ValueError):
        return False
    return bool(
        compose_robust_location_hint(record).strip()
        or compose_mask_detected_location_text(record).strip()
    )


def located_entities_missing_from_mask(
    mask_img: Image.Image,
    location_records: List[Dict[str, object]],
    color_map: Dict[str, str],
    *,
    min_confidence: float = MASK_LOCATION_CONFIDENT_MIN,
) -> List[str]:
    """Entity ids that VLM located confidently but have no palette color in the mask."""
    missing: List[str] = []
    seen: set[str] = set()
    for record in location_records:
        eid = str(record.get("entity_id", "")).strip()
        if not eid or eid in seen or eid not in color_map:
            continue
        seen.add(eid)
        if not entity_location_confident_for_mask_retry(
            record,
            min_confidence=min_confidence,
        ):
            continue
        if mask_color_coverage_ratio(mask_img, color_map[eid]) <= 0.00015:
            missing.append(eid)
    return missing


def partition_mask_location_entities(
    records: List[Dict[str, object]],
    entity_ids: List[str],
    *,
    subject_by_entity: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, str], set[str]]:
    """Split entities into located (with merged location text) vs priority-detect."""
    subject_by_entity = subject_by_entity or {}
    by_eid = {
        str(record.get("entity_id", "")).strip(): record
        for record in records
        if str(record.get("entity_id", "")).strip()
    }
    located: Dict[str, str] = {}
    focus: set[str] = set()

    for eid in entity_ids:
        if not eid:
            continue
        record = by_eid.get(eid)
        if record and bool(record.get("present_in_frame", False)):
            subject = subject_by_entity.get(eid, "").strip()
            if entity_location_confident_for_mask_retry(record):
                location_text = compose_robust_location_hint(
                    record,
                    subject_features=subject,
                ).strip()
            else:
                location_text = compose_mask_detected_location_text(
                    record,
                    subject_features=subject,
                ).strip()
            located[eid] = location_text or subject or "match the quoted description in image 1"
        else:
            focus.add(eid)

    return located, focus


def _location_record_confidence(
    location_record: Dict[str, object],
) -> Optional[float]:
    """Parse optional VLM confidence from a location record."""
    confidence = location_record.get("confidence")
    if confidence is None:
        return None
    try:
        return float(confidence)
    except (TypeError, ValueError):
        return None


def should_retry_reference_location(
    location_record: Optional[Dict[str, object]],
    *,
    min_confidence: float = 0.4,
) -> bool:
    """Return True when a batch location result should be retried per-entity."""
    if not location_record:
        return True
    if not bool(location_record.get("present_in_frame", False)):
        return True
    if not compose_robust_location_hint(location_record).strip():
        return True
    confidence = location_record.get("confidence")
    if confidence is None:
        return False
    try:
        return float(confidence) < min_confidence
    except (TypeError, ValueError):
        return False


_PLACEHOLDER_LOCATION_VALUES = frozenset({
    "",
    "none",
    "n/a",
    "na",
    "unknown",
    "nil",
    "not applicable",
    "not applicable.",
})


def _is_meaningful_location_text(value: object) -> bool:
    text = str(value or "").strip().lower()
    return bool(text) and text not in _PLACEHOLDER_LOCATION_VALUES


def keyframe_location_has_spatial_evidence(record: Dict[str, object]) -> bool:
    """Return True when the VLM gave a concrete WHERE in the target frame."""
    if _is_meaningful_location_text(record.get("location_edit_prompt")):
        return True
    if _is_meaningful_location_text(record.get("location_prompt")):
        return True
    if _is_meaningful_location_text(record.get("spatial_region")):
        return True
    if _as_str_list(record.get("landmark_relations")):
        return True
    if _as_str_list(record.get("visible_body_parts")):
        return True
    viewpoint = str(record.get("viewpoint_in_detection", "") or "").strip().lower()
    return bool(viewpoint) and viewpoint not in _PLACEHOLDER_LOCATION_VALUES


def keyframe_location_has_identity_evidence(
    record: Dict[str, object],
    *,
    subject_features: str = "",
) -> bool:
    """Return True when the record carries identity cues (reference-only is insufficient alone)."""
    return bool(_as_str_list(record.get("identity_cues")))


def normalize_keyframe_location_record(
    record: Dict[str, object],
    *,
    subject_features: str = "",
) -> Dict[str, object]:
    """Repair strict VLM outputs for side view / blur without inventing absent subjects."""
    normalized = dict(record)
    present = bool(normalized.get("present_in_frame", False))
    has_spatial = keyframe_location_has_spatial_evidence(normalized)
    confidence = _location_record_confidence(normalized)

    if present and not has_spatial:
        normalized["present_in_frame"] = False
        normalized["presence_rejected_no_spatial"] = True
        return normalized

    if present and has_spatial and (confidence is None or confidence <= 0.0):
        normalized["confidence"] = 0.6
        normalized["confidence_normalized"] = True
    elif (
        present
        and has_spatial
        and confidence is not None
        and 0.0 < confidence < 0.45
        and bool(normalized.get("viewpoint_change"))
    ):
        normalized["confidence"] = 0.5
        normalized["confidence_normalized"] = True

    return normalized


def should_retry_keyframe_location(
    location_record: Optional[Dict[str, object]],
    *,
    min_confidence: float = 0.5,
    subject_features: str = "",
) -> bool:
    """Return True when a keyframe location result should be retried per-instruction."""
    if not location_record:
        return True
    record = normalize_keyframe_location_record(
        location_record,
        subject_features=subject_features,
    )
    if not bool(record.get("present_in_frame", False)):
        return True
    if not keyframe_location_has_spatial_evidence(record):
        return True
    confidence = _location_record_confidence(record)
    if confidence is None:
        return True
    try:
        return float(confidence) < min_confidence
    except (TypeError, ValueError):
        return True


def assess_keyframe_entity_recognition(
    record: Optional[Dict[str, object]],
    *,
    min_confidence: float = 0.6,
    subject_features: str = "",
    mask_path: Optional[str] = None,
    color_hex: Optional[str] = None,
    action: str = "modify",
) -> Tuple[bool, str]:
    """Return whether keyframe entity location is confident enough to inpaint."""
    if not record:
        return False, "no location record — entity recognition failed"

    normalized = normalize_keyframe_location_record(
        record,
        subject_features=subject_features,
    )
    if not bool(normalized.get("present_in_frame", False)):
        if normalized.get("presence_rejected_no_spatial"):
            return False, "no spatial location in frame — entity not confirmed present"
        return False, "entity not present in frame — entity recognition failed"

    if not keyframe_location_has_spatial_evidence(normalized):
        return False, "no spatial location evidence — entity not confirmed in frame"

    if action.lower() != "add" and mask_path and color_hex:
        if (
            mask_has_content(mask_path)
            and not entity_mask_has_content(mask_path, color_hex)
        ):
            return (
                False,
                "no indicative mask region for entity in this scene — skip inpaint",
            )

    confidence = _location_record_confidence(normalized)
    effective_min = min_confidence
    has_viewpoint_tolerance = bool(normalized.get("viewpoint_change")) or bool(
        str(normalized.get("viewpoint_in_detection", "") or "").strip()
        and str(normalized.get("viewpoint_in_detection", "")).strip().lower()
        not in _PLACEHOLDER_LOCATION_VALUES
    )
    if has_viewpoint_tolerance and keyframe_location_has_spatial_evidence(normalized):
        effective_min = min(effective_min, 0.5)

    if confidence is None:
        return False, "missing confidence — entity recognition failed"

    try:
        score = float(confidence)
    except (TypeError, ValueError):
        return False, "invalid confidence — entity recognition failed"

    if score < effective_min:
        return (
            False,
            f"confidence {score:.0%} below {effective_min:.0%} — entity recognition failed",
        )
    return True, ""


_LOCATION_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "into", "only",
    "image", "edit", "target", "subject", "located", "visible", "parts",
    "viewpoint", "not", "must", "remain", "exactly", "frame",
})


def _location_text_tokens(text: str) -> Set[str]:
    tokens = {
        tok
        for tok in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(tok) > 2 and tok not in _LOCATION_STOPWORDS
    }
    return tokens


def keyframe_location_summary_text(
    record: Dict[str, object],
    *,
    subject_features: str = "",
) -> str:
    """Compact location text used for cross-instruction conflict checks."""
    return compose_keyframe_edit_location_prompt(
        record,
        subject_features=subject_features,
    ).strip()


def keyframe_location_pair_conflict_score(
    record_a: Dict[str, object],
    record_b: Dict[str, object],
    *,
    subject_a: str = "",
    subject_b: str = "",
    entity_id_a: str = "",
    entity_id_b: str = "",
) -> float:
    """Return 0-1 similarity; higher means more likely the same physical target."""
    if entity_id_a and entity_id_b and entity_id_a == entity_id_b:
        return 0.0

    text_a = keyframe_location_summary_text(record_a, subject_features=subject_a)
    text_b = keyframe_location_summary_text(record_b, subject_features=subject_b)
    tokens_a = _location_text_tokens(text_a)
    tokens_b = _location_text_tokens(text_b)
    overlap = 0.0
    if tokens_a and tokens_b:
        overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    region_a = str(record_a.get("spatial_region", "") or "").strip().lower()
    region_b = str(record_b.get("spatial_region", "") or "").strip().lower()
    if (
        _is_meaningful_location_text(region_a)
        and region_a == region_b
    ):
        overlap = max(overlap, 0.72)

    cues_a = set(_location_text_tokens(" ".join(_as_str_list(record_a.get("identity_cues")))))
    cues_b = set(_location_text_tokens(" ".join(_as_str_list(record_b.get("identity_cues")))))
    if cues_a and cues_b:
        cue_overlap = len(cues_a & cues_b) / len(cues_a | cues_b)
        overlap = max(overlap, cue_overlap)

    subj_a = _location_text_tokens(subject_a)
    subj_b = _location_text_tokens(subject_b)
    if subj_a and subj_b and subj_a != subj_b and overlap >= 0.45:
        overlap = max(overlap, 0.68)

    return overlap


def find_keyframe_location_conflicts(
    records_by_iid: Dict[str, Dict[str, object]],
    *,
    subject_by_instruction: Optional[Dict[str, str]] = None,
    entity_id_by_instruction: Optional[Dict[str, str]] = None,
    min_score: float = 0.62,
) -> List[Dict[str, object]]:
    """Find instruction pairs that likely map to the same physical target."""
    subject_by_instruction = subject_by_instruction or {}
    entity_id_by_instruction = entity_id_by_instruction or {}
    present_records = {
        iid: record
        for iid, record in records_by_iid.items()
        if bool(record.get("present_in_frame", False))
    }
    conflicts: List[Dict[str, object]] = []
    iids = sorted(present_records.keys())
    for idx, iid_a in enumerate(iids):
        for iid_b in iids[idx + 1:]:
            record_a = present_records[iid_a]
            record_b = present_records[iid_b]
            entity_a = str(
                record_a.get("entity_id", "")
                or entity_id_by_instruction.get(iid_a, "")
            ).strip()
            entity_b = str(
                record_b.get("entity_id", "")
                or entity_id_by_instruction.get(iid_b, "")
            ).strip()
            if entity_a and entity_b and entity_a == entity_b:
                continue
            score = keyframe_location_pair_conflict_score(
                record_a,
                record_b,
                subject_a=subject_by_instruction.get(iid_a, ""),
                subject_b=subject_by_instruction.get(iid_b, ""),
                entity_id_a=entity_a,
                entity_id_b=entity_b,
            )
            if score < min_score:
                continue
            conflicts.append({
                "instruction_id_a": iid_a,
                "instruction_id_b": iid_b,
                "entity_id_a": entity_a,
                "entity_id_b": entity_b,
                "score": round(score, 3),
                "summary_a": keyframe_location_summary_text(
                    record_a,
                    subject_features=subject_by_instruction.get(iid_a, ""),
                ),
                "summary_b": keyframe_location_summary_text(
                    record_b,
                    subject_features=subject_by_instruction.get(iid_b, ""),
                ),
            })
    return conflicts


def pick_keyframe_location_conflict_losers(
    conflicts: List[Dict[str, object]],
    records_by_iid: Dict[str, Dict[str, object]],
) -> Set[str]:
    """Choose which conflicting instructions to drop (lower confidence first)."""
    loser_scores: Dict[str, float] = {}

    def _confidence(iid: str) -> float:
        record = records_by_iid.get(iid) or {}
        value = _location_record_confidence(record)
        return float(value) if value is not None else 0.0

    for conflict in conflicts:
        iid_a = str(conflict.get("instruction_id_a", "")).strip()
        iid_b = str(conflict.get("instruction_id_b", "")).strip()
        if not iid_a or not iid_b:
            continue
        conf_a = _confidence(iid_a)
        conf_b = _confidence(iid_b)
        if conf_a < conf_b:
            loser, winner = iid_a, iid_b
            loser_conf, winner_conf = conf_a, conf_b
        elif conf_b < conf_a:
            loser, winner = iid_b, iid_a
            loser_conf, winner_conf = conf_b, conf_a
        else:
            loser, winner = sorted([iid_a, iid_b])
            loser_conf = conf_a
            winner_conf = conf_b
        loser_scores[loser] = max(loser_scores.get(loser, 0.0), float(conflict.get("score", 0.0)))
        logger.warning(
            "Keyframe location conflict: %s and %s map to the same target (score=%.2f) — "
            "prefer %s (conf=%.2f) over %s (conf=%.2f)",
            iid_a,
            iid_b,
            float(conflict.get("score", 0.0)),
            winner,
            winner_conf,
            loser,
            loser_conf,
        )
    return set(loser_scores.keys())


def format_keyframe_peer_assignment_lines(
    records_by_iid: Dict[str, Dict[str, object]],
    *,
    exclude_instruction_id: str,
    subject_by_instruction: Optional[Dict[str, str]] = None,
    entity_id_by_instruction: Optional[Dict[str, str]] = None,
) -> str:
    """Build lines describing other instructions' assigned targets for disambiguation."""
    subject_by_instruction = subject_by_instruction or {}
    entity_id_by_instruction = entity_id_by_instruction or {}
    lines: List[str] = []
    for iid, record in sorted(records_by_iid.items()):
        if iid == exclude_instruction_id:
            continue
        if not bool(record.get("present_in_frame", False)):
            continue
        subject = subject_by_instruction.get(iid, "").strip()
        entity_id = str(
            record.get("entity_id", "")
            or entity_id_by_instruction.get(iid, "")
        ).strip()
        summary = keyframe_location_summary_text(
            record,
            subject_features=subject,
        )
        lines.append(
            f'- instruction_id="{iid}"; entity_id="{entity_id}"; '
            f'subject_features="{subject}"; assigned_target="{summary}"'
        )
    return "\n".join(lines)


def keyframe_location_records_to_prompts(
    records: List[Dict[str, object]],
    *,
    subject_by_instruction: Optional[Dict[str, str]] = None,
    color_name_by_instruction: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Map keyframe location VLM records to per-instruction location edit prompts."""
    subject_by_instruction = subject_by_instruction or {}
    color_name_by_instruction = color_name_by_instruction or {}
    prompts: Dict[str, str] = {}
    for record in records:
        iid = str(record.get("instruction_id", "")).strip()
        if not iid:
            continue
        if not bool(record.get("present_in_frame", False)):
            subject = subject_by_instruction.get(iid, "").strip()
            if subject:
                prompts[iid] = _clamp_one_sentence(f"{subject} (verify in frame)")
            else:
                prompts[iid] = "Edit target (verify in frame)."
            continue
        prompts[iid] = compose_keyframe_edit_location_prompt(
            record,
            subject_features=subject_by_instruction.get(iid, ""),
            color_name=color_name_by_instruction.get(iid, ""),
        )
    return prompts


def split_entity_binary_masks(
    initial_mask_path: str,
    entity_color_map: Dict[str, str],
) -> Dict[str, np.ndarray]:
    """Split a multi-color mask into per-entity binary masks after palette quantization."""
    mask_img = Image.open(initial_mask_path).convert("RGB")
    quantized = quantize_mask_to_palette(mask_img, entity_color_map)
    arr = np.array(quantized)
    h, w = arr.shape[:2]
    result: Dict[str, np.ndarray] = {}

    for entity_id, hex_color in entity_color_map.items():
        rgb = np.array(hex_to_rgb(hex_color), dtype=np.int16)
        match = np.all(arr.astype(np.int16) == rgb.reshape(1, 1, 3), axis=2)
        binary = (match.astype(np.uint8) * 255)
        if binary.sum() == 0:
            logger.warning("Empty mask region for entity %s", entity_id)
        result[entity_id] = binary

    return result


def build_single_entity_segmentation_prompt(
    entity_id: str,
    description: str,
    color_hex: str,
    *,
    location_hint: str = "",
    instruction_id: str = "",
    anti_copy_retry: bool = False,
) -> str:
    """Build a one-entity segmentation prompt for targeted mask recovery."""
    from video_editing_agent.prompts.templates import (
        ANTI_COPY_MASK_CLAUSE,
        MASK_ANTI_STITCH_FAILURE_CLAUSE,
        MASK_SEGMENTATION_IMAGE_PROMPT,
    )

    color_name = color_name_from_hex(color_hex)
    id_suffix = f", instruction_id={instruction_id}" if instruction_id else ""
    hint_block = ""
    if location_hint.strip():
        hint_block = f"\nLocation guidance for image 1: {location_hint.strip()}"
    entity_queries = (
        f'1. Segment in image 1 ONLY: "{description.strip()}" '
        f"(entity_id={entity_id}{id_suffix}). "
        f"Search the entire frame; mark the full visible silhouette including side/back views. "
        f"Color: {color_name} ({color_hex})."
        f"{hint_block}"
    )
    anti_copy_clause = ANTI_COPY_MASK_CLAUSE if anti_copy_retry else ""
    return MASK_SEGMENTATION_IMAGE_PROMPT.format(
        entity_queries=entity_queries,
        anti_copy_clause=anti_copy_clause,
        anti_stitch_failure_clause=MASK_ANTI_STITCH_FAILURE_CLAUSE,
    )


def build_mask_segmentation_queries(
    entity_descriptions: List[Dict],
    color_map: Dict[str, str],
    *,
    located_by_entity: Optional[Dict[str, str]] = None,
    focus_entity_ids: Optional[set[str]] = None,
    instruction_labels: Optional[Dict[str, str]] = None,
) -> str:
    """Build concise per-entity segmentation lines for the image model."""
    located_by_entity = located_by_entity or {}
    instruction_labels = instruction_labels or {}
    described_ids = {
        str(desc.get("entity_id", "")).strip()
        for desc in entity_descriptions
        if str(desc.get("entity_id", "")).strip()
    }
    merged_descriptions: List[Dict] = list(entity_descriptions)
    for eid, hex_color in color_map.items():
        eid = str(eid).strip()
        if not eid or eid in described_ids:
            continue
        merged_descriptions.append({
            "entity_id": eid,
            "description": eid,
            "instruction_id": instruction_labels.get(eid, ""),
        })
        described_ids.add(eid)

    lines: List[str] = []
    for i, desc in enumerate(merged_descriptions):
        eid = desc.get("entity_id", "")
        text = (desc.get("description") or "").strip()
        hex_color = color_map.get(eid, entity_color_hex(i))
        color_name = color_name_from_hex(hex_color)
        iid = instruction_labels.get(eid, desc.get("instruction_id", "")) if instruction_labels else desc.get("instruction_id", "")
        id_suffix = f", instruction_id={iid}" if iid else ""

        if eid in located_by_entity:
            location_text = located_by_entity[eid].strip()
            line = (
                f'{i + 1}. Segment in image 1: "{text}" (entity_id={eid}{id_suffix}). '
                f"Location in image 1: {location_text}. "
                f"Color: {color_name} ({hex_color})."
            )
        elif focus_entity_ids is not None and eid in focus_entity_ids:
            line = (
                f'{i + 1}. [PRIORITY DETECT] Segment in image 1: "{text}" '
                f"(entity_id={eid}{id_suffix}). Prior reference did not confirm location — "
                f"search the entire frame and mark if present. "
                f"Color: {color_name} ({hex_color})."
            )
        else:
            line = (
                f'{i + 1}. Segment in image 1: "{text}" (entity_id={eid}{id_suffix}). '
                f"Search the entire frame; mark if present. "
                f"Color: {color_name} ({hex_color})."
            )
        lines.append(line)
    return "\n".join(lines)


def build_input_image_index_section(
    entity_descriptions: List[Dict],
    *,
    entity_references: Optional[Dict[str, str]] = None,
    reference_image_labels: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    """Build explicit image-index legend for segmentation prompts."""
    entity_references = entity_references or {}
    reference_image_labels = reference_image_labels or {}
    lines = [
        "INPUT IMAGE INDEX (CRITICAL):",
        "- Image 1 = detection frame. This is the ONLY segmentation canvas and the ONLY allowed output size.",
        "- Image 1 is the ONLY image you may paint on. Never use image 2+ as the output base.",
    ]
    for desc in entity_descriptions:
        eid = desc.get("entity_id", "")
        if eid not in entity_references:
            continue
        labels = reference_image_labels.get(eid, {})
        frame_label = labels.get("frame", "attached reference")
        lines.append(
            f"- {frame_label} = earlier reference frame with semi-transparent colored mask overlay "
            f"for {eid} (identity only; shows which entity was marked earlier; "
            f"NOT a mask template; NOT an output canvas)."
        )
    if len(lines) == 3:
        return ""
    return "\n".join(lines)


def assess_mask_candidate(
    mask_img: Image.Image,
    frame_size: Tuple[int, int],
    *,
    max_union_area: float = 0.85,
) -> List[str]:
    """Return issue codes for an indicative mask candidate (empty list = acceptable)."""
    issues: List[str] = []
    if mask_img.size != frame_size:
        issues.append("size_mismatch")
    if not image_has_mask_content(mask_img):
        issues.append("empty")
    if mask_looks_like_split_panel_artifact(mask_img, frame_size):
        issues.append("split_panel")
    if mask_union_coverage_ratio(mask_img) > max_union_area:
        issues.append("excessive_coverage")
    return issues


def build_batch_segmentation_prompt(
    entity_descriptions: List[Dict],
    color_map: Dict[str, str],
    *,
    located_by_entity: Optional[Dict[str, str]] = None,
    focus_entity_ids: Optional[set[str]] = None,
    instruction_labels: Optional[Dict[str, str]] = None,
    anti_copy_retry: bool = False,
) -> str:
    """Build full batch segmentation prompt."""
    from video_editing_agent.prompts.templates import (
        ANTI_COPY_MASK_CLAUSE,
        MASK_ANTI_STITCH_FAILURE_CLAUSE,
        MASK_SEGMENTATION_IMAGE_PROMPT,
    )

    entity_queries = build_mask_segmentation_queries(
        entity_descriptions,
        color_map,
        located_by_entity=located_by_entity,
        focus_entity_ids=focus_entity_ids,
        instruction_labels=instruction_labels,
    )
    anti_copy_clause = ANTI_COPY_MASK_CLAUSE if anti_copy_retry else ""
    return MASK_SEGMENTATION_IMAGE_PROMPT.format(
        entity_queries=entity_queries,
        anti_copy_clause=anti_copy_clause,
        anti_stitch_failure_clause=MASK_ANTI_STITCH_FAILURE_CLAUSE,
    )


def ensure_anti_copy_in_revised_prompt(revised_prompt: str) -> str:
    """Append anti-copy guidance to a model-revised segmentation prompt."""
    text = (revised_prompt or "").strip()
    if not text:
        return text
    marker = "NON-NEGOTIABLE RETRY MASK CONSTRAINTS"
    if marker in text:
        return text
    from video_editing_agent.prompts.templates import ANTI_COPY_MASK_CLAUSE

    return (
        f"{text}\n\n"
        f"{ANTI_COPY_MASK_CLAUSE}\n"
        "Apply these constraints even if they conflict with the revised prompt above. "
        "The output must stay in strict one-to-one correspondence with image 1's pixel grid. "
        "Never align the mask to any reference composite. "
        "The current frame geometry in image 1 is the only source for mask shape."
    )


def _load_reference_caption_font(size: int = 22) -> ImageFont.ImageFont:
    """Best-effort readable font for reference-composite captions."""
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arial.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_caption_lines(
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> List[str]:
    """Word-wrap caption text to fit composite width."""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    words = text.split()
    if not words:
        return [text]

    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if probe.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _format_reference_composite_caption(instruction_id: str) -> str:
    """English caption explaining overlay reference semantics."""
    return (
        f"Instruction {instruction_id} — reference frame with semi-transparent colored "
        "indicative mask overlay. For reference only to identify the entity corresponding "
        "to the mask. Do not use for mask generation."
    )


def overlay_colored_mask_on_frame(
    frame: Image.Image,
    mask: Image.Image,
    *,
    alpha: float = 0.45,
    mask_threshold: int = 12,
) -> Image.Image:
    """Blend a colored indicative mask onto a reference frame as a thin color film."""
    frame_rgb = frame.convert("RGB")
    mask_rgb = mask.convert("RGB")
    if mask_rgb.size != frame_rgb.size:
        mask_rgb = mask_rgb.resize(frame_rgb.size, Image.Resampling.NEAREST)

    frame_arr = np.array(frame_rgb, dtype=np.float32)
    mask_arr = np.array(mask_rgb, dtype=np.float32)
    active = np.max(mask_arr, axis=2) > mask_threshold
    if not np.any(active):
        return frame_rgb

    out = frame_arr.copy()
    inv = 1.0 - alpha
    for channel in range(3):
        out[..., channel] = np.where(
            active,
            frame_arr[..., channel] * inv + mask_arr[..., channel] * alpha,
            frame_arr[..., channel],
        )
    return Image.fromarray(out.clip(0, 255).astype(np.uint8), mode="RGB")


def _append_bottom_caption(image: Image.Image, caption: str, *, padding: int = 16) -> Image.Image:
    """Render caption text below an image on a white strip."""
    font = _load_reference_caption_font()
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_text_width = max(32, image.width - 2 * padding)
    lines = _wrap_caption_lines(caption, font, max_text_width)

    try:
        bbox = probe.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=6)
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        line_height = getattr(font, "size", 12) + 6
        text_height = line_height * len(lines)

    caption_height = text_height + 2 * padding
    canvas = Image.new("RGB", (image.width, image.height + caption_height), (255, 255, 255))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text(
        (padding, image.height + padding),
        "\n".join(lines),
        fill=(0, 0, 0),
        font=font,
        spacing=6,
    )
    return canvas


def entity_ref_overlay_path(ref_dir: str, instruction_id: str) -> str:
    """Path to overlay reference image with caption."""
    return os.path.join(ref_dir, f"{instruction_id}_ref.png")


def entity_ref_src_path(ref_dir: str, instruction_id: str) -> str:
    """Path to raw source frame backup for an instruction reference."""
    return os.path.join(ref_dir, f"{instruction_id}_ref_src.png")


def entity_ref_mask_path(ref_dir: str, instruction_id: str) -> str:
    """Path to single-entity indicative mask saved with the reference bundle."""
    return os.path.join(ref_dir, f"{instruction_id}_ref_mask.png")


def entity_ref_canonical_path(ref_dir: str, instruction_id: str) -> str:
    """Path to left-right canonical comparison card (original vs edited entity)."""
    return os.path.join(ref_dir, f"{instruction_id}_ref_canonical.png")


def entity_ref_multiview_path(ref_dir: str, instruction_id: str) -> str:
    """Path to synthesized front-view entity reference image.

    Kept under the historical function name so existing pipeline calls continue
    to work while entity_refs no longer produce 4-view sheets.
    """
    return os.path.join(ref_dir, f"{instruction_id}_ref_front.png")


def entity_ref_multiview_edited_path(ref_dir: str, instruction_id: str) -> str:
    """Path to edited front-view entity reference image after applying the edit instruction."""
    return os.path.join(ref_dir, f"{instruction_id}_ref_front_edited.png")


ENTITY_REF_BBOX_PADDING_RATIO = 0.25


def _fit_entity_on_white_canvas(
    crop: Image.Image,
    panel_size: int = 384,
) -> Image.Image:
    """Center-fit an image on a white square panel."""
    crop = crop.convert("RGB")
    w, h = crop.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (panel_size, panel_size), (255, 255, 255))
    scale = min(panel_size / w, panel_size / h, 1.0)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = crop.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (panel_size, panel_size), (255, 255, 255))
    ox = (panel_size - nw) // 2
    oy = (panel_size - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas


def _mask_active_bbox(
    mask_img: Image.Image,
    *,
    threshold: int = 20,
    padding_ratio: float = 0.10,
) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box of active mask pixels with proportional padding."""
    arr = np.array(mask_img.convert("RGB"), dtype=np.uint8)
    active = np.max(arr, axis=2) > threshold
    if not np.any(active):
        return None
    ys, xs = np.where(active)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    w, h = arr.shape[1], arr.shape[0]
    pad_x = max(4, int((x1 - x0) * padding_ratio))
    pad_y = max(4, int((y1 - y0) * padding_ratio))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(w, x1 + pad_x),
        min(h, y1 + pad_y),
    )


def _fit_entity_bbox_panel(
    frame_img: Image.Image,
    mask_img: Image.Image,
    *,
    panel_size: int = 384,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    padding_ratio: float = ENTITY_REF_BBOX_PADDING_RATIO,
) -> Image.Image:
    """Crop the mask-bounded entity region (rectangular box) and fit on a white panel."""
    frame = frame_img.convert("RGB")
    mask = mask_img.convert("RGB")
    if mask.size != frame.size:
        mask = mask.resize(frame.size, Image.Resampling.NEAREST)
    if bbox is None:
        bbox = _mask_active_bbox(mask, padding_ratio=padding_ratio)
    if bbox is None:
        return Image.new("RGB", (panel_size, panel_size), (255, 255, 255))
    return _fit_entity_on_white_canvas(frame.crop(bbox), panel_size)


def _label_entity_panel(panel: Image.Image, label: str) -> Image.Image:
    """Add a short label strip above an entity panel."""
    font = _load_reference_caption_font()
    strip_h = 28
    canvas = Image.new("RGB", (panel.width, panel.height + strip_h), (255, 255, 255))
    canvas.paste(panel, (0, strip_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), label, fill=(0, 0, 0), font=font)
    return canvas


def _compose_before_after_panels(
    left_panel: Image.Image,
    right_panel: Image.Image,
    *,
    gap: int = 12,
) -> Image.Image:
    """Horizontally compose two same-width entity panels with a divider."""
    if right_panel.size != left_panel.size:
        right_panel = right_panel.resize(left_panel.size, Image.Resampling.LANCZOS)
    height = max(left_panel.height, right_panel.height)
    width = left_panel.width + gap + right_panel.width
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(left_panel, (0, 0))
    canvas.paste(right_panel, (left_panel.width + gap, 0))
    divider_x = left_panel.width + gap // 2
    draw = ImageDraw.Draw(canvas)
    draw.line(
        [(divider_x, 0), (divider_x, height)],
        fill=(210, 210, 210),
        width=2,
    )
    return canvas


def format_canonical_comparison_caption(
    instruction_id: str,
    entity_id: str,
    action: str,
    subject_features: str = "",
) -> str:
    """Caption for left-right canonical comparison cards."""
    action_key = (action or "").strip().lower()
    if action_key == "delete":
        header = (
            f"DELETE TARGET | instruction_id={instruction_id} | entity_id={entity_id}"
        )
        layout = "Left: mask-bounded entity crop (original) | Right: removed (empty)."
    else:
        header = (
            f"EDIT COMPARISON | instruction_id={instruction_id} | entity_id={entity_id}"
        )
        layout = (
            "Left: mask-bounded entity crop (original) | Right: same crop after edit. "
            "Rectangular entity region only — not full scene."
        )
    subject = " ".join(str(subject_features or "").split())
    lines = [header, layout]
    if subject:
        lines.insert(1, f"Identified subject: {subject}")
    return "\n".join(lines)


def build_before_after_entity_reference_card(
    original_frame: Image.Image,
    mask_img: Image.Image,
    edited_frame: Optional[Image.Image] = None,
    *,
    instruction_id: str,
    entity_id: str,
    action: str,
    subject_features: str = "",
    panel_size: int = 384,
) -> Image.Image:
    """Left-right canonical card: mask-bounded entity crops before vs after edit."""
    mask = mask_img.convert("RGB")
    bbox = _mask_active_bbox(mask, padding_ratio=ENTITY_REF_BBOX_PADDING_RATIO)
    left = _label_entity_panel(
        _fit_entity_bbox_panel(
            original_frame,
            mask,
            panel_size=panel_size,
            bbox=bbox,
        ),
        "Original",
    )
    if edited_frame is not None:
        right = _label_entity_panel(
            _fit_entity_bbox_panel(
                edited_frame,
                mask,
                panel_size=panel_size,
                bbox=bbox,
            ),
            "Edited",
        )
    else:
        blank = Image.new("RGB", (panel_size, panel_size), (255, 255, 255))
        right = _label_entity_panel(blank, "Removed")
    composite = _compose_before_after_panels(left, right)
    caption = format_canonical_comparison_caption(
        instruction_id,
        entity_id,
        action,
        subject_features,
    )
    return _append_bottom_caption(composite, caption)


def save_before_after_entity_reference(
    original_frame_path: str,
    mask_path: str,
    output_path: str,
    *,
    edited_frame_path: Optional[str] = None,
    instruction_id: str,
    entity_id: str,
    action: str,
    subject_features: str = "",
) -> str:
    """Save a left-right canonical comparison card from mask-bounded entity crops."""
    original = Image.open(original_frame_path).convert("RGB")
    mask = Image.open(mask_path).convert("RGB")
    edited = (
        Image.open(edited_frame_path).convert("RGB")
        if edited_frame_path and os.path.exists(edited_frame_path)
        else None
    )
    card = build_before_after_entity_reference_card(
        original,
        mask,
        edited,
        instruction_id=instruction_id,
        entity_id=entity_id,
        action=action,
        subject_features=subject_features,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    card.save(output_path)
    return output_path


def format_isolated_entity_caption(
    instruction_id: str,
    entity_id: str,
    action: str,
    subject_features: str = "",
) -> str:
    """Caption for isolated entity reference cards with clear instruction/entity IDs."""
    action_key = (action or "").strip().lower()
    if action_key == "delete":
        header = (
            f"DELETE TARGET | instruction_id={instruction_id} | entity_id={entity_id}"
        )
    else:
        header = (
            f"EDIT RESULT | instruction_id={instruction_id} | entity_id={entity_id}"
        )
    subject = " ".join(str(subject_features or "").split())
    lines = [header, "Mask-bounded entity crop for identification."]
    if subject:
        lines.insert(1, f"Identified subject: {subject}")
    return "\n".join(lines)


def build_isolated_entity_reference_card(
    frame_img: Image.Image,
    mask_img: Image.Image,
    *,
    instruction_id: str,
    entity_id: str,
    action: str,
    subject_features: str = "",
    panel_size: int = 384,
) -> Image.Image:
    """Fit a mask-bounded entity crop on white background with ID caption."""
    panel = _fit_entity_bbox_panel(frame_img, mask_img, panel_size=panel_size)
    caption = format_isolated_entity_caption(
        instruction_id,
        entity_id,
        action,
        subject_features,
    )
    return _append_bottom_caption(panel, caption)


def save_isolated_entity_reference(
    frame_path: str,
    mask_path: str,
    output_path: str,
    *,
    instruction_id: str,
    entity_id: str,
    action: str,
    subject_features: str = "",
) -> str:
    """Save a mask-bounded entity crop card with entity ID caption."""
    frame = Image.open(frame_path).convert("RGB")
    mask = Image.open(mask_path).convert("RGB")
    card = build_isolated_entity_reference_card(
        frame,
        mask,
        instruction_id=instruction_id,
        entity_id=entity_id,
        action=action,
        subject_features=subject_features,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    card.save(output_path)
    return output_path


def _save_entity_ref_src_and_mask(
    ref_dir: str,
    instruction_id: str,
    first_frame_path: str,
    entity_mask: Image.Image,
) -> Tuple[str, str]:
    """Persist raw source frame and entity mask alongside overlay reference."""
    src_path = entity_ref_src_path(ref_dir, instruction_id)
    mask_path = entity_ref_mask_path(ref_dir, instruction_id)
    frame = Image.open(first_frame_path).convert("RGB")
    frame.save(src_path)
    entity_mask.save(mask_path)
    return src_path, mask_path


def build_single_entity_mask_image(
    reference_mask_path: str,
    instruction_id: str,
    reference_mask_color: str,
    frame_size: Tuple[int, int],
) -> Image.Image:
    """Extract one entity's colored mask for reference bundle storage."""
    mask = Image.new("RGB", frame_size, (0, 0, 0))
    if not reference_mask_path or not reference_mask_color:
        return mask
    if not os.path.exists(reference_mask_path):
        return mask
    try:
        mask_img = Image.open(reference_mask_path).convert("RGB")
        mask = keep_only_entity_colors(
            mask_img,
            {instruction_id: reference_mask_color},
            [instruction_id],
        )
        if mask.size != frame_size:
            mask = mask.resize(frame_size, Image.Resampling.NEAREST)
    except Exception as exc:
        logger.warning("Failed to build entity mask for %s: %s", instruction_id, exc)
    return mask


def save_entity_reference_assets(
    ref_dir: str,
    instruction_id: str,
    first_frame_path: str,
    reference_mask_path: Optional[str] = None,
    reference_mask_color: Optional[str] = None,
    *,
    entity_id: str = "",
    overlay_alpha: float = 0.45,
    output_path: Optional[str] = None,
) -> str:
    """Persist an overlay reference guide for an instruction.

    The saved image keeps the existing ``*_ref.png`` naming convention by default:
    reference frame with a semi-transparent colored indicative mask film on top,
    plus an English caption strip below explaining reference-only usage.
    """
    os.makedirs(ref_dir, exist_ok=True)
    frame_path = output_path or entity_ref_overlay_path(ref_dir, instruction_id)

    frame = Image.open(first_frame_path).convert("RGB")
    entity_mask = build_single_entity_mask_image(
        reference_mask_path or "",
        instruction_id,
        reference_mask_color or "",
        frame.size,
    )

    if output_path is None:
        _save_entity_ref_src_and_mask(ref_dir, instruction_id, first_frame_path, entity_mask)

    combined = overlay_colored_mask_on_frame(
        frame,
        entity_mask,
        alpha=overlay_alpha,
    )
    caption = _format_reference_composite_caption(instruction_id)
    combined = _append_bottom_caption(combined, caption)
    combined.save(frame_path)

    if entity_id and reference_mask_color:
        save_entity_ref_meta(
            ref_dir,
            instruction_id,
            entity_id=entity_id,
            color_hex=reference_mask_color,
            color_name=color_name_from_hex(reference_mask_color),
        )
    return frame_path


def boost_segmentation_mask_contrast(
    mask_img: Image.Image,
    *,
    min_active: int = 4,
) -> Image.Image:
    """Linear-stretch weak segmentation model outputs before palette quantization."""
    arr = np.array(mask_img.convert("RGB"))
    peak = int(np.max(arr)) if arr.size else 0
    if peak < min_active or peak >= 128:
        return mask_img
    scale = 255.0 / float(peak)
    boosted = np.clip(arr.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return Image.fromarray(boosted, mode="RGB")


def mask_has_palette_coverage(
    mask_img: Image.Image,
    color_map: Dict[str, str],
    *,
    min_ratio: float = 0.00015,
) -> bool:
    """Return True when any entity palette color covers a meaningful region."""
    if not color_map:
        return image_has_mask_content(mask_img)
    return any(
        mask_color_coverage_ratio(mask_img, hex_color) > min_ratio
        for hex_color in color_map.values()
        if hex_color
    )


def finalize_segmentation_mask(
    mask_img: Image.Image,
    color_map: Dict[str, str],
    *,
    keep_entity_ids: Optional[List[str]] = None,
) -> Image.Image:
    """Boost contrast and quantize an indicative mask to the entity palette."""
    boosted = boost_segmentation_mask_contrast(mask_img)
    return quantize_mask_to_palette_best_effort(
        boosted,
        color_map,
        keep_entity_ids=keep_entity_ids,
    )


def image_has_mask_content(mask_img: Image.Image, *, threshold: int = 20) -> bool:
    """Return True if an in-memory mask image has non-black pixels."""
    arr = np.array(mask_img.convert("RGB"))
    return bool(np.any(arr > threshold))


def mask_from_raw_segmentation_sidecar(
    raw_path: str,
    frame_size: Tuple[int, int],
) -> Optional[Image.Image]:
    """Build final mask from ``*.raw.png`` by resizing/aligning to the source frame."""
    if not raw_path or not os.path.exists(raw_path):
        return None
    aligned = align_mask_to_frame(Image.open(raw_path).convert("RGB"), frame_size)
    if not image_has_mask_content(aligned):
        return None
    return aligned


def segmentation_aligned_sidecar_path(mask_path: str) -> str:
    """Path to the aligned batch-segmentation sidecar for ``mask_0000.png``."""
    return f"{mask_path}.aligned.png"


def segmentation_retry_aligned_path(mask_path: str, entity_id: str) -> str:
    """Path to a per-entity supplement retry aligned mask sidecar."""
    return f"{mask_path}.{entity_id}.retry.aligned.png"


def restore_mask_from_segmentation_sidecars(
    mask_path: str,
    *,
    color_map: Optional[Dict[str, str]] = None,
) -> bool:
    """Rebuild ``mask_path`` from aligned / retry sidecars when the final file is empty.

    Returns True when a non-empty mask was written.
    """
    if not mask_path:
        return False

    aligned_path = segmentation_aligned_sidecar_path(mask_path)
    layers: List[Image.Image] = []
    frame_size: Optional[Tuple[int, int]] = None

    if os.path.exists(aligned_path):
        aligned = Image.open(aligned_path).convert("RGB")
        frame_size = aligned.size
        if image_has_mask_content(aligned):
            layers.append(aligned)

    if color_map:
        prefix = f"{mask_path}."
        suffix = ".retry.aligned.png"
        for retry_path in sorted(glob.glob(f"{prefix}*{suffix}")):
            if not os.path.isfile(retry_path):
                continue
            basename = os.path.basename(retry_path)
            mask_basename = os.path.basename(mask_path)
            eid = basename[len(mask_basename) + 1 : -len(suffix)]
            if eid not in color_map:
                continue
            retry = Image.open(retry_path).convert("RGB")
            if frame_size is None:
                frame_size = retry.size
            if not image_has_mask_content(retry):
                boosted = boost_segmentation_mask_contrast(retry)
                if image_has_mask_content(boosted):
                    retry = boosted
                else:
                    continue
            layer = quantize_mask_to_palette_best_effort(
                retry,
                color_map,
                keep_entity_ids=[eid],
            )
            if image_has_mask_content(layer):
                layers.append(layer)

    if not layers or frame_size is None:
        return False

    combined = (
        layers[0]
        if len(layers) == 1
        else composite_segmentation_layers(layers, frame_size)
    )
    if not image_has_mask_content(combined):
        return False

    os.makedirs(os.path.dirname(mask_path) or ".", exist_ok=True)
    combined.save(mask_path)
    return True


def ensure_segmentation_mask_output(
    mask_path: str,
    *,
    color_map: Optional[Dict[str, str]] = None,
) -> str:
    """Keep ``mask_path`` non-empty when batch / supplement sidecars still have pixels."""
    if mask_path and os.path.exists(mask_path):
        mask_img = Image.open(mask_path).convert("RGB")
        if image_has_mask_content(mask_img) or (
            color_map and mask_has_palette_coverage(mask_img, color_map)
        ):
            return mask_path

    if restore_mask_from_segmentation_sidecars(mask_path, color_map=color_map):
        logger.info("Restored empty segmentation mask from sidecars: %s", mask_path)
    return mask_path


def mask_color_coverage_ratio(
    mask_img: Image.Image,
    color_hex: str,
    *,
    min_magnitude: int = 12,
    min_cosine: float = 0.82,
) -> float:
    """Return approximate frame coverage for one palette color."""
    arr = np.array(mask_img.convert("RGB"))
    if arr.size == 0:
        return 0.0
    pixels = arr.astype(np.float32)
    magnitude = np.max(arr, axis=2)
    valid = magnitude >= min_magnitude
    if not np.any(valid):
        return 0.0

    norms = np.maximum(np.linalg.norm(pixels, axis=2), 1e-6)
    unit = pixels / norms[:, :, np.newaxis]
    target = _palette_unit_vector(color_hex)
    cosine = np.sum(unit * target.reshape(1, 1, 3), axis=2)
    matched = valid & (cosine >= min_cosine)
    return float(np.count_nonzero(matched)) / float(arr.shape[0] * arr.shape[1])


def mask_union_coverage_ratio(mask_img: Image.Image, *, threshold: int = 12) -> float:
    """Return approximate non-black mask coverage over the full frame."""
    arr = np.array(mask_img.convert("RGB"))
    if arr.size == 0:
        return 0.0
    active = np.max(arr, axis=2) >= threshold
    return float(np.count_nonzero(active)) / float(arr.shape[0] * arr.shape[1])


def keep_only_entity_colors(
    mask_img: Image.Image,
    entity_color_map: Dict[str, str],
    keep_entity_ids: List[str],
    *,
    min_magnitude: int = 12,
    min_cosine: float = DEFAULT_PALETTE_MIN_COSINE,
) -> Image.Image:
    """Keep only selected entity color regions using exclusive palette quantization."""
    if not keep_entity_ids:
        return Image.new("RGB", mask_img.size, (0, 0, 0))
    return quantize_mask_to_palette(
        mask_img,
        entity_color_map,
        keep_entity_ids=keep_entity_ids,
        min_magnitude=min_magnitude,
        min_cosine=min_cosine,
    )


def composite_segmentation_layers(
    layers: List[Image.Image],
    frame_size: Tuple[int, int],
) -> Image.Image:
    """Merge per-entity segmentation layers without blur or feathering."""
    w, h = frame_size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for layer in layers:
        arr = np.array(align_mask_to_frame(layer, frame_size).convert("RGB"))
        mag = np.max(arr, axis=2)
        active = mag > 15
        if not np.any(active):
            continue
        canvas[active] = np.maximum(canvas[active], arr[active])
    return Image.fromarray(canvas, mode="RGB")


def entity_color_hex(index: int) -> str:
    """Palette hex color for entity index."""
    r, g, b = ENTITY_COLORS_RGB[index % len(ENTITY_COLORS_RGB)]
    return f"#{r:02X}{g:02X}{b:02X}"


def render_soft_bbox_mask(
    image_size: Tuple[int, int],
    detections: List[Dict],
    entity_color_map: Dict[str, str],
    *,
    feather_radius: int = 25,
) -> Image.Image:
    """Fallback: soft elliptical indicative masks from bboxes (not hard rectangles)."""
    w, h = image_size
    hard = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(hard)

    for det in detections:
        eid = det.get("entity_id", "")
        bbox = det.get("bbox") or []
        if len(bbox) != 4:
            continue
        color = hex_to_rgb(entity_color_map.get(eid, "#FF0000"))
        x0 = int(float(bbox[0]) * w)
        y0 = int(float(bbox[1]) * h)
        x1 = int(float(bbox[2]) * w)
        y1 = int(float(bbox[3]) * h)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        rx, ry = max((x1 - x0) // 2, 1), max((y1 - y0) // 2, 1)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color)

    return to_indicative_soft_mask(hard, blur_radius=feather_radius)


def to_indicative_soft_mask(
    mask_img: Image.Image,
    *,
    blur_radius: int = 21,
    max_intensity: int = 180,
) -> Image.Image:
    """Convert any mask to a soft indicative guide (feathered, non-binary).

    Args:
        mask_img: RGB or RGBA segmentation / colored mask.
        blur_radius: Gaussian blur kernel radius for soft edges.
        max_intensity: Peak channel value (below 255 = intentionally soft).

    Returns:
        Soft RGB indicative mask.
    """
    arr = np.array(mask_img.convert("RGB")).astype(np.float32)
    if cv2 is not None and blur_radius > 0:
        k = blur_radius * 2 + 1
        arr = cv2.GaussianBlur(arr, (k, k), 0)

    # Zero out near-black background
    magnitude = np.max(arr, axis=2)
    arr[magnitude < 8] = 0

    peak = max(float(np.max(arr)), 1.0)
    arr = arr * (max_intensity / peak)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="RGB")


def extract_reference_frame_from_composite(
    ref_image: Image.Image,
    frame_size: Tuple[int, int],
) -> Image.Image:
    """Prepare a saved entity reference image for mask-segmentation APIs.

    New format: reference frame + semi-transparent mask overlay, optional caption below.
    Legacy format: side-by-side [mask | frame] composite — right panel is extracted.
    """
    fw, fh = frame_size
    w, h = ref_image.size

    # Legacy side-by-side composite.
    if w >= int(fw * 1.75):
        stitched_h = fh if h > int(fh * 1.08) else h
        stitched = ref_image.crop((0, 0, w, stitched_h))
        right = stitched.crop((fw, 0, w, stitched_h))
        if right.size != frame_size:
            right = right.resize(frame_size, Image.Resampling.LANCZOS)
        return right

    # Overlay reference: strip optional caption band, keep frame-sized content on top.
    content_h = fh
    if h > int(fh * 1.08):
        content_h = fh
    elif h < fh:
        content_h = h

    content = ref_image.crop((0, 0, w, content_h))
    if content.size != frame_size:
        content = content.resize(frame_size, Image.Resampling.LANCZOS)
    return content


def mask_looks_like_split_panel_artifact(
    mask_img: Image.Image,
    frame_size: Tuple[int, int],
    *,
    edge_fraction: float = 0.40,
) -> bool:
    """Detect masks confined to one lateral half (stitched-output artifact)."""
    mask = align_mask_to_frame(mask_img, frame_size)
    arr = np.array(mask.convert("RGB"))
    if arr.size == 0:
        return False

    w = arr.shape[1]
    active = np.max(arr, axis=2) > 12
    if not np.any(active):
        return False

    cols = np.where(np.any(active, axis=0))[0]
    x_min, x_max = int(cols[0]), int(cols[-1])
    if x_max < w * edge_fraction:
        return True
    if x_min > w * (1.0 - edge_fraction):
        return True

    span = x_max - x_min + 1
    centroid = (x_min + x_max) / 2.0
    if span < w * 0.45 and (centroid < w * 0.42 or centroid > w * 0.58):
        return True
    return False


def _mask_active_coverage(mask_img: Image.Image) -> float:
    arr = np.array(mask_img.convert("RGB"))
    if arr.size == 0:
        return 0.0
    active = np.max(arr, axis=2) > 12
    return float(np.count_nonzero(active)) / float(active.size)


def _extract_best_panel_from_stitched_mask(
    mask_img: Image.Image,
    frame_size: Tuple[int, int],
) -> Image.Image:
    """Recover a single-frame mask when the model returned a split-panel output."""
    fw, fh = frame_size
    mw, mh = mask_img.size
    content_h = min(mh, fh) if mh > int(fh * 1.08) else mh
    stitched = mask_img.crop((0, 0, mw, content_h))
    left = stitched.crop((0, 0, mw // 2, content_h)).resize(frame_size, Image.Resampling.LANCZOS)
    right = stitched.crop((mw // 2, 0, mw, content_h)).resize(frame_size, Image.Resampling.LANCZOS)

    left_cov = _mask_active_coverage(left)
    right_cov = _mask_active_coverage(right)
    if left_cov == 0.0 and right_cov == 0.0:
        return left
    if right_cov == 0.0:
        return left
    if left_cov == 0.0:
        return right
    # Prefer the panel whose content is less edge-confined.
    left_artifact = mask_looks_like_split_panel_artifact(left, frame_size)
    right_artifact = mask_looks_like_split_panel_artifact(right, frame_size)
    if left_artifact and not right_artifact:
        return right
    if right_artifact and not left_artifact:
        return left
    return left if left_cov >= right_cov else right


def align_mask_to_frame(mask_img: Image.Image, frame_size: Tuple[int, int]) -> Image.Image:
    """Resize or de-stitch indicative mask to match source frame dimensions."""
    fw, fh = frame_size
    mw, mh = mask_img.size

    if mw >= int(fw * 1.75):
        return _extract_best_panel_from_stitched_mask(mask_img, frame_size)

    if mh > int(fh * 1.25) and mw >= int(fw * 1.25):
        mask_img = mask_img.crop((0, 0, mw, fh))
        mw, mh = mask_img.size

    if mask_img.size == frame_size:
        return mask_img
    return mask_img.resize(frame_size, Image.Resampling.LANCZOS)


def render_bbox_mask(
    image_size: Tuple[int, int],
    detections: List[Dict],
    entity_color_map: Dict[str, str],
) -> Image.Image:
    """Render multi-color mask from normalized bounding boxes.

    Args:
        image_size: (width, height).
        detections: List with entity_id and bbox [x_min,y_min,x_max,y_max] 0-1.
        entity_color_map: entity_id → hex color.

    Returns:
        RGB mask PIL image.
    """
    w, h = image_size
    mask = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(mask)

    for det in detections:
        eid = det.get("entity_id", "")
        bbox = det.get("bbox") or []
        if len(bbox) != 4:
            continue
        color = hex_to_rgb(entity_color_map.get(eid, "#FF0000"))
        x0 = int(float(bbox[0]) * w)
        y0 = int(float(bbox[1]) * h)
        x1 = int(float(bbox[2]) * w)
        y1 = int(float(bbox[3]) * h)
        draw.rectangle([x0, y0, x1, y1], fill=color)

    return mask


def union_binary_mask(mask_rgb: Image.Image, entity_color_map: Dict[str, str]) -> Image.Image:
    """Build single-channel L mask (255=edit region) from multi-color mask."""
    arr = np.array(mask_rgb.convert("RGB"))
    union = np.zeros(arr.shape[:2], dtype=np.uint8)
    for hex_c in entity_color_map.values():
        rgb = np.array(hex_to_rgb(hex_c), dtype=np.uint8)
        match = np.all(np.abs(arr.astype(np.int16) - rgb.astype(np.int16)) < 30, axis=2)
        union[match] = 255
    return Image.fromarray(union, mode="L")


def mask_has_content(mask_path: str, *, threshold: int = 20) -> bool:
    """Return True if mask image has any non-empty pixels."""
    if not mask_path or not os.path.exists(mask_path):
        return False
    arr = np.array(Image.open(mask_path).convert("RGB"))
    return bool(np.any(arr > threshold))


def _palette_unit_vector(hex_color: str) -> np.ndarray:
    """Normalized RGB direction for an entity palette color."""
    rgb = np.array(hex_to_rgb(hex_color), dtype=np.float32)
    norm = float(np.linalg.norm(rgb))
    if norm < 1e-6:
        return rgb
    return rgb / norm


def extract_entity_mask(
    multi_color_mask_path: str,
    entity_color_hex: str,
    output_path: str,
    *,
    tolerance: int = 35,
    soft: bool = True,
    min_magnitude: int = 12,
    min_cosine: float = 0.82,
) -> str:
    """Extract per-entity indicative mask (soft by default).

    Uses color-direction matching so softened / blurred multi-color masks
    (peak ~180, feathered edges) still separate entities correctly.
    """
    arr = np.array(Image.open(multi_color_mask_path).convert("RGB"))
    magnitude = np.max(arr, axis=2)
    valid = magnitude >= min_magnitude

    if soft:
        pixels = arr.astype(np.float32)
        norms = np.maximum(np.linalg.norm(pixels, axis=2), 1e-6)
        unit = pixels / norms[:, :, np.newaxis]
        target = _palette_unit_vector(entity_color_hex)
        cosine = np.sum(unit * target.reshape(1, 1, 3), axis=2)
        match = valid & (cosine >= min_cosine)
    else:
        rgb = np.array(hex_to_rgb(entity_color_hex), dtype=np.int16)
        match = valid & np.all(
            np.abs(arr.astype(np.int16) - rgb.reshape(1, 1, 3)) <= tolerance,
            axis=2,
        )

    if soft:
        soft_img = to_indicative_soft_mask(
            Image.fromarray((match.astype(np.uint8) * 255), mode="L").convert("RGB"),
            blur_radius=15,
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        soft_img.save(output_path)
        return output_path

    binary = (match.astype(np.uint8) * 255)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Image.fromarray(binary, mode="L").save(output_path)
    return output_path


def entity_mask_has_content(multi_color_mask_path: str, color_hex: str) -> bool:
    """Return True if a specific entity color region exists in a multi-color mask."""
    if not multi_color_mask_path or not os.path.exists(multi_color_mask_path):
        return False
    mask_img = Image.open(multi_color_mask_path).convert("RGB")
    return mask_color_coverage_ratio(mask_img, color_hex) > 0.00015


def union_mask_from_multicolor(mask_path: str, output_path: str) -> str:
    """Build union binary mask from any non-black RGB mask."""
    arr = np.array(Image.open(mask_path).convert("RGB"))
    binary = (np.any(arr > 20, axis=2).astype(np.uint8) * 255)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Image.fromarray(binary, mode="L").save(output_path)
    return output_path


def composite_with_mask(
    original_path: str,
    edited_path: str,
    mask_path: str,
    output_path: str,
    *,
    feather: int = 3,
) -> str:
    """Paste edited region onto original using binary mask (background preserved).

    Args:
        original_path: Original frame.
        edited_path: Model-generated frame.
        mask_path: Union or binary mask (white = edit).
        output_path: Output composited image.

    Returns:
        output_path
    """
    original = Image.open(original_path).convert("RGB")
    edited = Image.open(edited_path).convert("RGB")
    if edited.size != original.size:
        edited = edited.resize(original.size, Image.LANCZOS)

    mask_img = Image.open(mask_path)
    if mask_img.mode == "RGB":
        mask_l = union_binary_mask(mask_img, {"#FFFFFF": "#FFFFFF"})
        # treat any non-black as edit if no color map
        arr = np.array(mask_img.convert("RGB"))
        gray = np.any(arr > 20, axis=2).astype(np.uint8) * 255
        mask_l = Image.fromarray(gray, mode="L")
    else:
        mask_l = mask_img.convert("L")

    if feather > 0 and cv2 is not None:
        m = np.array(mask_l)
        m = cv2.GaussianBlur(m, (feather * 2 + 1, feather * 2 + 1), 0)
        mask_l = Image.fromarray(m, mode="L")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result = Image.composite(edited, original, mask_l)
    result.save(output_path)
    return output_path


def detect_uniform_margin_bands(
    image: Image.Image,
    *,
    threshold: int = 12,
    row_coverage: float = 0.92,
    col_coverage: float = 0.92,
) -> Tuple[int, int, int, int]:
    """Detect top/bottom/left/right letterbox or pillarbox bands in pixels."""
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return (0, 0, 0, 0)

    dark = np.max(arr, axis=2) < threshold

    top = 0
    for y in range(h):
        if float(np.mean(dark[y])) >= row_coverage:
            top += 1
        else:
            break

    bottom = 0
    for y in range(h - 1, -1, -1):
        if float(np.mean(dark[y])) >= row_coverage:
            bottom += 1
        else:
            break

    left = 0
    for x in range(w):
        if float(np.mean(dark[:, x])) >= col_coverage:
            left += 1
        else:
            break

    right = 0
    for x in range(w - 1, -1, -1):
        if float(np.mean(dark[:, x])) >= col_coverage:
            right += 1
        else:
            break

    return top, bottom, left, right


def _apply_margin_bands_from_original(
    original: Image.Image,
    result: Image.Image,
    margins: Tuple[int, int, int, int],
) -> Image.Image:
    """Force letterbox/pillarbox margin pixels back to the original frame."""
    top, bottom, left, right = margins
    if top + bottom + left + right == 0:
        return result

    out = result.copy()
    o = original.convert("RGB")
    w, h = out.size
    if top > 0:
        out.paste(o.crop((0, 0, w, top)), (0, 0))
    if bottom > 0:
        out.paste(o.crop((0, h - bottom, w, h)), (0, h - bottom))
    if left > 0:
        out.paste(o.crop((0, 0, left, h)), (0, 0))
    if right > 0:
        out.paste(o.crop((w - right, 0, w, h)), (w - right, 0))
    return out


def restore_frame_margins_from_original(
    original_path: str,
    edited_path: str,
    output_path: str,
) -> str:
    """Restore letterbox/pillarbox margins and canvas size from the source frame."""
    original = Image.open(original_path).convert("RGB")
    edited = Image.open(edited_path).convert("RGB")
    if edited.size != original.size:
        edited = edited.resize(original.size, Image.Resampling.LANCZOS)
    margins = detect_uniform_margin_bands(original)
    restored = _apply_margin_bands_from_original(original, edited, margins)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    restored.save(output_path)
    return output_path


def blend_inpaint_preserve_frame_structure(
    original_path: str,
    edited_path: str,
    mask_path: str,
    output_path: str,
    *,
    feather: int = 3,
) -> str:
    """Blend inpainted edits onto the original frame without changing canvas structure.

    Keeps unmasked pixels and detected letterbox/pillarbox margins from the original
    so black bars and framing are not removed or stretched by the image model.
    """
    original = Image.open(original_path).convert("RGB")
    margins = detect_uniform_margin_bands(original)

    raw_output = output_path
    if os.path.abspath(edited_path) == os.path.abspath(output_path):
        raw_output = f"{output_path}.inpaint_raw.png"

    composite_with_mask(original_path, edited_path, mask_path, raw_output, feather=feather)

    result = Image.open(raw_output).convert("RGB")
    if result.size != original.size:
        result = result.resize(original.size, Image.Resampling.LANCZOS)
    restored = _apply_margin_bands_from_original(original, result, margins)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    restored.save(output_path)

    if raw_output != output_path and os.path.exists(raw_output):
        try:
            os.remove(raw_output)
        except OSError:
            pass
    return output_path


def crop_by_mask(
    image_path: str,
    mask_path: str,
    output_path: str,
    padding: int = 8,
) -> str:
    """Crop image to mask bounding box for reference library."""
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path)
    if mask.mode == "RGB":
        arr = np.array(mask.convert("RGB"))
        ys, xs = np.where(np.any(arr > 20, axis=2))
    else:
        arr = np.array(mask.convert("L"))
        ys, xs = np.where(arr > 20)

    if len(xs) == 0:
        img.save(output_path)
        return output_path

    x0, x1 = max(0, int(xs.min()) - padding), min(img.width, int(xs.max()) + padding)
    y0, y1 = max(0, int(ys.min()) - padding), min(img.height, int(ys.max()) + padding)
    crop = img.crop((x0, y0, x1, y1))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    crop.save(output_path)
    return output_path


def propagate_masks_optical_flow(
    frame_paths: List[str],
    initial_mask_path: str,
    output_mask_dir: str,
) -> List[str]:
    """Propagate first-frame mask through scene using OpenCV optical flow.

    Fallback VOS when the video mask API is unavailable.

    Args:
        frame_paths: Ordered scene frame paths.
        initial_mask_path: First frame multi-color or binary mask.
        output_mask_dir: Output directory for mask sequence.

    Returns:
        List of output mask paths.
    """
    if cv2 is None:
        raise RuntimeError("opencv-python is required for mask propagation")

    os.makedirs(output_mask_dir, exist_ok=True)
    if not frame_paths:
        return []

    prev_frame = cv2.imread(frame_paths[0])
    if prev_frame is None:
        raise RuntimeError(f"Cannot read frame: {frame_paths[0]}")

    prev_mask = cv2.imread(initial_mask_path)
    if prev_mask is None:
        raise RuntimeError(f"Cannot read mask: {initial_mask_path}")

    out_paths: List[str] = []
    out0 = os.path.join(output_mask_dir, "mask_0000.png")
    cv2.imwrite(out0, prev_mask)
    out_paths.append(out0)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    for i, fpath in enumerate(frame_paths[1:], start=1):
        frame = cv2.imread(fpath)
        if frame is None:
            logger.warning("Skip unreadable frame: %s", fpath)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0,
        )
        h, w = prev_mask.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[:, :, 0]).astype(np.float32)
        map_y = (grid_y + flow[:, :, 1]).astype(np.float32)
        warped = cv2.remap(prev_mask, map_x, map_y, cv2.INTER_LINEAR)

        out_p = os.path.join(output_mask_dir, f"mask_{i:04d}.png")
        cv2.imwrite(out_p, warped)
        out_paths.append(out_p)

        prev_gray = gray
        prev_mask = warped

    logger.info("Optical-flow VOS: %d masks → %s", len(out_paths), output_mask_dir)
    return out_paths


def composite_video_with_masks(
    original_clip: str,
    edited_clip: str,
    mask_dir: str,
    output_path: str,
    *,
    fps: float = 25.0,
) -> str:
    """Per-frame masked composite of edited video onto original clip.

    Preserves background from original; applies edited pixels inside mask.
    """
    if cv2 is None:
        raise RuntimeError("opencv-python is required for video compositing")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cap_orig = cv2.VideoCapture(original_clip)
    cap_edit = cv2.VideoCapture(edited_clip)
    if not cap_orig.isOpened() or not cap_edit.isOpened():
        raise RuntimeError("Cannot open video for compositing")

    w = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    mask_files = sorted(
        f for f in os.listdir(mask_dir)
        if f.startswith("mask_") and f.endswith(".png")
    )

    idx = 0
    while True:
        ret_o, frame_o = cap_orig.read()
        if not ret_o:
            break

        ret_e, frame_e = cap_edit.read()
        if not ret_e:
            cap_edit.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_e, frame_e = cap_edit.read()

        if frame_e.shape[:2] != frame_o.shape[:2]:
            frame_e = cv2.resize(frame_e, (w, h))

        if idx < len(mask_files):
            mask_bgr = cv2.imread(os.path.join(mask_dir, mask_files[idx]))
            if mask_bgr is not None:
                if mask_bgr.shape[:2] != (h, w):
                    mask_bgr = cv2.resize(mask_bgr, (w, h))
                gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
                _, bin_m = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
                bin_m = cv2.GaussianBlur(bin_m, (5, 5), 0)
                alpha = bin_m.astype(np.float32) / 255.0
                alpha = alpha[:, :, np.newaxis]
                blended = (frame_e.astype(np.float32) * alpha +
                           frame_o.astype(np.float32) * (1.0 - alpha))
                frame_o = blended.astype(np.uint8)

        writer.write(frame_o)
        idx += 1

    cap_orig.release()
    cap_edit.release()
    writer.release()

    # Re-encode to h264 for compatibility
    from video_editing_agent.utils.ffmpeg_utils import _run
    tmp = output_path + ".tmp.mp4"
    os.rename(output_path, tmp)
    _run(
        [
            "ffmpeg", "-y", "-i", tmp,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
            output_path, "-loglevel", "error",
        ],
        desc="reencode composite",
    )
    os.remove(tmp)
    logger.info("Composited video → %s", output_path)
    return output_path
