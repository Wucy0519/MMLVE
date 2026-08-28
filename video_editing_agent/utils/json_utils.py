"""JSON extraction helpers for LLM responses."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def extract_json_object(text: str) -> Dict[str, Any]:
    """Parse the first JSON object found in an LLM response string.

    Args:
        text: Raw model output possibly wrapped in markdown fences.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If no valid JSON object is found.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty LLM response")

    # Strip markdown code fences
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Find outermost { ... }
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object in response: {text[:200]}")

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : i + 1]
                return json.loads(snippet)

    raise ValueError(f"Unbalanced JSON in response: {text[:200]}")


def ensure_instruction_ids(instructions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign instruction_id / entity_id when missing."""
    out = []
    for i, item in enumerate(instructions):
        d = dict(item)
        if not d.get("instruction_id"):
            d["instruction_id"] = f"instr_{i + 1:03d}"
        if not d.get("entity_id"):
            d["entity_id"] = f"entity_{i + 1:02d}"
        d.pop("action", None)
        d.pop("needs_ref_image", None)
        d.pop("ref_subject", None)
        d.pop("ref_image_path", None)
        out.append(d)
    return out


def merge_instructions_one_per_entity(
    instructions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep one instruction row per ``entity_id``; merge duplicate entity rows."""
    if not instructions:
        return []

    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for item in instructions:
        eid = str(item.get("entity_id", "")).strip() or f"entity_{len(order) + 1:02d}"
        if eid not in merged:
            merged[eid] = dict(item)
            merged[eid]["entity_id"] = eid
            order.append(eid)
            continue

        existing = merged[eid]
        for key in ("subject_features", "edit_prompt", "success_criteria_prompt", "appearance_time_hint"):
            extra = str(item.get(key, "") or "").strip()
            if not extra:
                continue
            prev = str(existing.get(key, "") or "").strip()
            if not prev:
                existing[key] = extra
            elif extra not in prev:
                existing[key] = f"{prev}; {extra}"

        prev_scope = str(existing.get("target_instance_scope", "single") or "single").lower()
        next_scope = str(item.get("target_instance_scope", "single") or "single").lower()
        if prev_scope == "multiple" or next_scope == "multiple":
            existing["target_instance_scope"] = "multiple"
        else:
            existing["target_instance_scope"] = "single"

        existing.pop("appearance_time_sec", None)

        logger.info(
            "Merged duplicate instruction for %s into %s",
            item.get("instruction_id"),
            existing.get("instruction_id"),
        )

    return [merged[eid] for eid in order]


def normalize_instruction_instance_scope(
    instructions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Default each instruction to a single tracked instance unless explicitly plural."""
    out: List[Dict[str, Any]] = []
    for item in instructions:
        row = dict(item)
        raw = row.get("target_instance_scope", row.get("multiple_instances"))
        if isinstance(raw, bool):
            scope = "multiple" if raw else "single"
        elif isinstance(raw, (int, float)):
            scope = "multiple" if raw > 1 else "single"
        else:
            text = str(raw or "single").strip().lower()
            if text in {"multiple", "multi", "all", "many", "plural", "every"}:
                scope = "multiple"
            else:
                scope = "single"
        row["target_instance_scope"] = scope
        row.pop("multiple_instances", None)
        out.append(row)
    return out
