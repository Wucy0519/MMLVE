"""Build authoritative video-edit operation prompts from Module-3 planned edits."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from video_editing_agent.schemas.instructions import EntityInstructionSet

logger = logging.getLogger(__name__)

_FIRST_FRAME_CONSISTENCY = (
    "The output video's first frame must match the attached reference edited frame "
    "exactly in composition, edits, colors, and layout."
)


def _load_json(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _location_text(record: dict, prompts: Dict[str, str]) -> str:
    iid = str(record.get("instruction_id", "")).strip()
    for key in ("location_edit_prompt", "location_prompt"):
        text = str(record.get(key, "") or "").strip()
        if text:
            return text
    return str(prompts.get(iid, "") or record.get("subject_features", "") or "").strip()


def _resolve_planned_edits(
    sidecar: dict,
    entity_instru_path: str,
) -> List[dict]:
    """Return planned edit dicts with action + edit_prompt + location."""
    planned = sidecar.get("planned_edits") or []
    if planned:
        return [dict(p) for p in planned if p.get("instruction_id")]

    prompts = sidecar.get("prompts") or {}
    records = sidecar.get("records") or []
    if not records:
        return []

    instr_by_id: Dict[str, Any] = {}
    entity_data = _load_json(entity_instru_path)
    if entity_data:
        try:
            instr_set = EntityInstructionSet.load(entity_instru_path)
            instr_by_id = {i.instruction_id: i for i in instr_set.instructions}
        except (TypeError, ValueError, KeyError, OSError) as exc:
            logger.warning("Could not load entity_instru for planned edits: %s", exc)

    resolved: List[dict] = []
    for record in records:
        iid = str(record.get("instruction_id", "")).strip()
        if not iid:
            continue
        instr = instr_by_id.get(iid)
        resolved.append(
            {
                "instruction_id": iid,
                "entity_id": record.get("entity_id") or (instr.entity_id if instr else ""),
                "action": instr.action.value if instr else "modify",
                "edit_prompt": (instr.edit_prompt if instr else "").strip(),
                "subject_features": (
                    instr.subject_features if instr else record.get("subject_features", "")
                ),
                "location_prompt": _location_text(record, prompts),
            }
        )
    return [p for p in resolved if p.get("edit_prompt")]


def _format_planned_edit(item: dict) -> str:
    action = str(item.get("action", "modify") or "modify").strip().lower()
    edit = str(item.get("edit_prompt", "") or "").strip()
    location = str(
        item.get("location_edit_prompt")
        or item.get("location_prompt")
        or item.get("subject_features")
        or ""
    ).strip()

    target = f"Target: {location}. " if location else ""

    if action == "delete":
        return (
            f"DELETE — {target}{edit} "
            "The deleted target must be fully removed in every frame; "
            "do not preserve, keep, or leave this person/object visible."
        )
    if action == "add":
        return f"ADD — {target}{edit}"
    return f"MODIFY — {target}{edit}"


def build_mandatory_video_edit_operation(
    location_prompts_path: str,
    *,
    entity_instru_path: str = "",
    fallback_edit_prompt: str = "",
) -> str:
    """Compose video-edit prompt from Module-3 planned edits (authoritative over VLM diff)."""
    sidecar = _load_json(location_prompts_path)
    if not sidecar:
        return (fallback_edit_prompt or "").strip()

    planned = _resolve_planned_edits(sidecar, entity_instru_path)
    if not planned:
        return (fallback_edit_prompt or "").strip()

    clauses = [_format_planned_edit(p) for p in planned]
    edits_block = " | ".join(clauses)
    return (
        "Apply ALL of the following edits consistently across the entire video clip. "
        "These edit instructions take priority over any unchanged background or bystanders: "
        f"{edits_block} "
        f"{_FIRST_FRAME_CONSISTENCY} "
        "For regions NOT listed above only, preserve camera motion, timing, pacing, "
        "lighting, letterboxing / black bars, and natural scene continuity."
    )
