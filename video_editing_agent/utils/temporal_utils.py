"""Temporal conflict resolution for entity instructions."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# User mentions time only to identify WHO/WHAT, not edit window (text cues)
_IDENTIFICATION_APPEARANCE_CUE_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"视频前\s*几帧|前\s*几帧"),
        "appears in the first few frames of the video",
    ),
    (
        re.compile(r"视频\s*开头|开头\s*部分|视频\s*一开始|一开始"),
        "appears near the beginning of the video",
    ),
    (
        re.compile(r"视频\s*末尾|视频\s*结尾|最后\s*几帧|结尾\s*部分"),
        "appears near the end of the video",
    ),
    (
        re.compile(
            r"(?:出现|出现在)\s*(?:于|在)?\s*(?:视频)?\s*"
            r"(?:约|大约|around|roughly|about)?\s*第\s*(\d+(?:\.\d+)?)\s*秒"
            r"(?:\s*(?:左右|附近|前后|时))?",
            re.I,
        ),
        "appears around second {0} in the video",
    ),
    (
        re.compile(
            r"(?:视频)?\s*第\s*(\d+(?:\.\d+)?)\s*秒\s*(?:出现|左右|附近|前后)?",
            re.I,
        ),
        "appears around second {0} in the video",
    ),
    (
        re.compile(
            r"(?:appears?|appearing|shown|visible|seen)\s+(?:at|around|about)\s+"
            r"(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)?",
            re.I,
        ),
        "appears around second {0} in the video",
    ),
    (
        re.compile(
            r"(?:at|around|about)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)\s*(?:mark|point)?",
            re.I,
        ),
        "appears around second {0} in the video",
    ),
    (
        re.compile(r"(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)\s*(?:mark|point)", re.I),
        "appears around second {0} in the video",
    ),
    (
        re.compile(r"first\s+few\s+frames", re.I),
        "appears in the first few frames of the video",
    ),
    (
        re.compile(r"near\s+the\s+(?:beginning|start)\s+of\s+the\s+video", re.I),
        "appears near the beginning of the video",
    ),
    (
        re.compile(r"near\s+the\s+end\s+of\s+the\s+video", re.I),
        "appears near the end of the video",
    ),
]

_REFERENTIAL_TIME_PATTERNS = [
    re.compile(r"(?:约|大约|around|roughly|about)\s*(?:第)?\s*\d+\s*秒", re.I),
    re.compile(r"(?:第)\s*\d+\s*秒\s*(?:出现|左右|附近|前后)?", re.I),
    re.compile(
        r"(?:appears?|appearing|shown|visible|seen)\s+(?:at|around|about)\s+\d+",
        re.I,
    ),
    re.compile(r"\d+\s*(?:s|sec|seconds?)\s*(?:mark|point)", re.I),
    re.compile(r"(?:at|around|about)\s+\d+\s*(?:s|sec|seconds?)", re.I),
    re.compile(r"who\s+appears\s+(?:at|around)", re.I),
    re.compile(r"(?:出现|出现在)\s*(?:约|大约)?\s*\d+\s*秒", re.I),
    re.compile(r"前\s*几帧"),
    re.compile(r"视频\s*开头|开头\s*部分|视频\s*一开始"),
    re.compile(r"视频\s*末尾|最后\s*几帧|结尾\s*部分"),
    re.compile(r"first\s+few\s+frames", re.I),
    re.compile(r"near\s+the\s+(?:beginning|start|end)\s+of\s+the\s+video", re.I),
]

# User explicitly constrains edit to a time window
_EXPLICIT_EDIT_WINDOW_PATTERNS = [
    re.compile(r"\b(?:only|just|solely)\b", re.I),
    re.compile(r"\b(?:between|from)\s+\d+\s+(?:and|to)\s+\d+", re.I),
    re.compile(r"\b(?:仅在|只在|仅限于)\b"),
    re.compile(r"\b(?:从).{0,20}(?:到|至).{0,20}(?:秒|分钟)\b"),
    re.compile(r"\b(?:first|last)\s+\d+\s*(?:s|sec|seconds?|min|minutes?)\b", re.I),
    re.compile(r"\b(?:edit|change|modify|remove).{0,30}\b(?:between|from)\b", re.I),
]

_DEFAULT_EVENT_SCOPE = "wherever the entity appears in the video"


def _absolute_window(tc: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Return (start, end) for absolute conditions, else None."""
    if tc.get("condition_type") != "absolute":
        return None
    start = tc.get("start_sec")
    end = tc.get("end_sec")
    if start is None or end is None:
        return None
    return float(start), float(end)


def windows_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """Return True if two half-open intervals [start, end) overlap."""
    return a[0] < b[1] and b[0] < a[1]


def _combined_instruction_text(instr: Dict[str, Any], source_prompt: str = "") -> str:
    tc = instr.get("time_condition") or {}
    parts = [
        source_prompt,
        instr.get("subject_features", ""),
        instr.get("edit_prompt", ""),
        tc.get("event_description", ""),
    ]
    return " ".join(str(p) for p in parts if p)


def extract_identification_appearance_time_description(text: str) -> Optional[str]:
    """Extract referential appearance-time wording as a natural-language cue."""
    if not (text or "").strip():
        return None
    for pattern, template in _IDENTIFICATION_APPEARANCE_CUE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groups()
        if groups:
            try:
                sec = float(groups[0])
                sec_token = str(int(sec)) if sec == int(sec) else f"{sec:.1f}"
                return template.format(sec_token)
            except (TypeError, ValueError, IndexError):
                return template
        return template
    return None


def _appearance_hint_in_text(text: str, hint: str) -> bool:
    if not hint or not text:
        return False
    lowered = text.lower()
    hint_lower = hint.lower()
    if hint_lower in lowered:
        return True
    # Treat overlapping identification cues as equivalent
    extracted = extract_identification_appearance_time_description(text)
    return bool(extracted and extracted.lower() == hint_lower)


def enrich_instruction_appearance_cues(
    instr: Dict[str, Any],
    *,
    source_prompt: str = "",
) -> Dict[str, Any]:
    """Record referential appearance-time wording for entity identification."""
    out = dict(instr)
    text = _combined_instruction_text(out, source_prompt)

    hint = str(out.get("appearance_time_hint", "") or "").strip()
    if not hint:
        hint = extract_identification_appearance_time_description(text) or ""

    out.pop("appearance_time_sec", None)

    if hint:
        out["appearance_time_hint"] = hint
        subject = str(out.get("subject_features", "") or "").strip()
        if subject and not _appearance_hint_in_text(subject, hint):
            out["subject_features"] = f"{subject}; {hint}"
        elif not subject:
            out["subject_features"] = hint
    else:
        out.pop("appearance_time_hint", None)

    return out


def enrich_all_instruction_appearance_cues(
    instructions: List[Dict[str, Any]],
    *,
    source_prompt: str = "",
) -> List[Dict[str, Any]]:
    """Apply :func:`enrich_instruction_appearance_cues` to each instruction dict."""
    return [
        enrich_instruction_appearance_cues(i, source_prompt=source_prompt)
        for i in instructions
    ]


def _has_referential_timestamp(text: str) -> bool:
    return any(p.search(text) for p in _REFERENTIAL_TIME_PATTERNS)


def _has_explicit_edit_window(text: str) -> bool:
    return any(p.search(text) for p in _EXPLICIT_EDIT_WINDOW_PATTERNS)


def _is_open_ended_absolute(tc: Dict[str, Any]) -> bool:
    end = tc.get("end_sec")
    if end is None:
        return True
    try:
        return float(end) >= 9000.0
    except (TypeError, ValueError):
        return False


def normalize_referential_time_condition(
    instr: Dict[str, Any],
    *,
    source_prompt: str = "",
) -> Dict[str, Any]:
    """Coerce mis-parsed absolute timestamps used only for entity identification.

    Example: "blond male lead who appears around 30s" must not become start_sec=30.
    """
    instr = dict(instr)
    tc = dict(instr.get("time_condition") or {})
    ctype = tc.get("condition_type", "event")

    text = _combined_instruction_text(instr, source_prompt)

    if ctype == "event":
        return instr

    if ctype != "absolute":
        return instr

    # Respect explicit edit-window language from the user
    if _has_explicit_edit_window(text) and not _has_referential_timestamp(text):
        return instr

    start = tc.get("start_sec")
    open_ended = _is_open_ended_absolute(tc)
    referential = _has_referential_timestamp(text)

    should_coerce = referential or open_ended
    if not should_coerce:
        return instr

    event_desc = (tc.get("event_description") or "").strip()
    if not event_desc or referential:
        event_desc = _DEFAULT_EVENT_SCOPE

    logger.info(
        "Referential timestamp normalized for %s: absolute start=%s → event (%s)",
        instr.get("instruction_id"),
        start,
        event_desc,
    )
    instr["time_condition"] = {
        "condition_type": "event",
        "event_description": event_desc,
        "start_sec": None,
        "end_sec": None,
    }
    return instr


def normalize_all_referential_time_conditions(
    instructions: List[Dict[str, Any]],
    *,
    source_prompt: str = "",
) -> List[Dict[str, Any]]:
    """Apply :func:`normalize_referential_time_condition` to each instruction."""
    return [
        normalize_referential_time_condition(i, source_prompt=source_prompt)
        for i in instructions
    ]


def resolve_temporal_conflicts(instructions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Resolve overlapping absolute-time conflicts per entity (keep last).

    For each entity_id group:
        - Sort by list order (proxy for user intent / timeline).
        - For absolute time_condition pairs that overlap, drop earlier ones.
        - Event-based conditions are kept (scene-level dedup happens later).

    Args:
        instructions: Raw instruction dicts.

    Returns:
        Conflict-resolved list preserving original order of survivors.
    """
    if not instructions:
        return []

    by_entity: Dict[str, List[Dict[str, Any]]] = {}
    for instr in instructions:
        eid = instr.get("entity_id", "entity_unknown")
        by_entity.setdefault(eid, []).append(instr)

    survivors: List[Dict[str, Any]] = []
    dropped = 0

    for entity_id, group in by_entity.items():
        kept: List[Dict[str, Any]] = []
        for instr in group:
            tc = instr.get("time_condition") or {}
            win = _absolute_window(tc)
            if win is None:
                kept.append(instr)
                continue

            overlap_idx = None
            for j, prev in enumerate(kept):
                prev_win = _absolute_window(prev.get("time_condition") or {})
                if prev_win and windows_overlap(prev_win, win):
                    overlap_idx = j
                    break

            if overlap_idx is not None:
                logger.info(
                    "Temporal conflict on %s: dropping %s, keeping %s",
                    entity_id,
                    kept[overlap_idx].get("instruction_id"),
                    instr.get("instruction_id"),
                )
                kept[overlap_idx] = instr
                dropped += 1
            else:
                kept.append(instr)

        survivors.extend(kept)

    # Preserve original global order
    order = {instr.get("instruction_id"): i for i, instr in enumerate(instructions)}
    survivors.sort(key=lambda x: order.get(x.get("instruction_id"), 9999))

    if dropped:
        logger.info("Resolved %d temporal conflict(s)", dropped)
    return survivors
