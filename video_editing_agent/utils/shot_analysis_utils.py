"""Normalize VLM shot-analysis JSON and trim transition zones from scene ranges."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from video_editing_agent.schemas.shots import (
    ShotAnalysis,
    ShotKeyframe,
    ShotTimeRange,
    TransitionZone,
    UndetectedSubCut,
)

logger = logging.getLogger(__name__)

DEFAULT_TRANSITION_TRIM_HALF_SEC = 0.12
DEFAULT_TRANSITION_MAX_ZONE_SEC = 0.6
DEFAULT_SUB_CUT_MIN_CONFIDENCE = 0.75
DEFAULT_SUB_CUT_MIN_DURATION_SEC = 0.8
MIN_SCENE_DURATION_AFTER_TRIM_SEC = 0.12
BOUNDARY_SNAP_TOLERANCE_SEC = 0.5

# In-shot cinematography — NOT editorial sub-cuts
_FALSE_POSITIVE_TRANSITION_KEYWORDS = (
    "rack focus",
    "pull focus",
    "focus pull",
    "focus shift",
    "focus change",
    "depth of field",
    "shallow focus",
    "deep focus",
    "bokeh",
    "defocus",
    "refocus",
    "zoom",
    "push in",
    "push-in",
    "pull out",
    "pull-out",
    "dolly",
    "camera move",
    "handheld shake",
    "pan",
    "tilt",
    "lighting change",
    "exposure change",
    "subject move",
    "motion blur",
)

# True editorial blends between distinct compositions
_EDITORIAL_TRANSITION_KEYWORDS = (
    "dissolve",
    "cross-dissolve",
    "cross dissolve",
    "fade",
    "fade to black",
    "fade in",
    "fade out",
    "wipe",
    "mix",
    "blend",
    "morph",
)


@dataclass
class SubCutTrimSettings:
    """Knobs for sub-cut validation and transition trimming."""

    transition_trim_half_sec: float = DEFAULT_TRANSITION_TRIM_HALF_SEC
    transition_max_zone_sec: float = DEFAULT_TRANSITION_MAX_ZONE_SEC
    sub_cut_min_confidence: float = DEFAULT_SUB_CUT_MIN_CONFIDENCE
    sub_cut_min_duration_sec: float = DEFAULT_SUB_CUT_MIN_DURATION_SEC


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _lower_blob(*parts: str) -> str:
    return " ".join(p for p in parts if p).lower()


def _contains_keyword(text: str, keywords: Tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in keywords)


def parse_transition_boundaries(raw: Dict[str, Any]) -> List[float]:
    """Parse precise transition peak timestamps from VLM output."""
    peaks: List[float] = []
    for item in raw.get("transition_boundaries") or []:
        if not isinstance(item, dict):
            continue
        for key in ("boundary_sec_in_shot", "peak_sec_in_shot", "transition_sec_in_shot"):
            if key in item and item[key] is not None:
                try:
                    peaks.append(float(item[key]))
                except (TypeError, ValueError):
                    pass
                break
    return sorted(peaks)


def _sub_cut_boundary_points(sub_cuts: List[UndetectedSubCut]) -> List[float]:
    """Internal boundaries between consecutive sub-cuts (shot-relative)."""
    sorted_subs = sorted(sub_cuts, key=lambda s: s.start_sec_in_shot)
    points: List[float] = []
    for left, right in zip(sorted_subs, sorted_subs[1:]):
        points.append((left.end_sec_in_shot + right.start_sec_in_shot) / 2.0)
    return points


def _snap_to_subcut_boundary(
    peak_sec: float,
    sub_cuts: List[UndetectedSubCut],
    *,
    tolerance_sec: float,
) -> float:
    """Snap a transition peak to the nearest sub-cut boundary when close enough."""
    boundaries = _sub_cut_boundary_points(sub_cuts)
    if not boundaries:
        return peak_sec
    nearest = min(boundaries, key=lambda b: abs(b - peak_sec))
    if abs(nearest - peak_sec) <= tolerance_sec:
        return nearest
    return peak_sec


def validate_sub_cut_analysis(
    raw: Dict[str, Any],
    *,
    shot_duration: float,
    settings: SubCutTrimSettings,
) -> Tuple[bool, str]:
    """Return (accepted, reason). Reject focus-shift and low-confidence splits."""
    if not raw.get("has_undetected_sub_cuts"):
        return False, "not flagged"

    confidence = float(raw.get("sub_cut_detection_confidence", 0.0) or 0.0)
    if confidence < settings.sub_cut_min_confidence:
        return False, f"confidence {confidence:.2f} < {settings.sub_cut_min_confidence}"

    subs = [s for s in (raw.get("undetected_sub_cuts") or []) if isinstance(s, dict)]
    if len(subs) < 2:
        return False, "fewer than 2 sub-cuts"

    for sub in subs:
        try:
            rel_start = float(sub.get("start_sec_in_shot", 0.0) or 0.0)
            rel_end = float(sub.get("end_sec_in_shot", rel_start) or rel_start)
        except (TypeError, ValueError):
            return False, "invalid sub-cut timestamps"
        if rel_end - rel_start < settings.sub_cut_min_duration_sec:
            return False, f"sub-cut too short ({rel_end - rel_start:.2f}s)"

    blob = _lower_blob(
        str(raw.get("plot_description", "")),
        str(raw.get("sub_cut_rationale", "")),
        " ".join(
            str(s.get("sub_plot_description", "")) for s in subs if isinstance(s, dict)
        ),
    )
    for tz in raw.get("transition_zones") or []:
        if isinstance(tz, dict):
            blob += " " + str(tz.get("transition_type", ""))

    has_editorial = _contains_keyword(blob, _EDITORIAL_TRANSITION_KEYWORDS)
    has_false_positive = _contains_keyword(blob, _FALSE_POSITIVE_TRANSITION_KEYWORDS)

    if has_false_positive and not has_editorial:
        return False, "focus/camera change mistaken for editorial transition"

    for tz in raw.get("transition_zones") or []:
        if not isinstance(tz, dict):
            continue
        ttype = str(tz.get("transition_type", "") or "")
        if _contains_keyword(ttype, _FALSE_POSITIVE_TRANSITION_KEYWORDS):
            if not _contains_keyword(ttype, _EDITORIAL_TRANSITION_KEYWORDS):
                return False, f"non-editorial transition type: {ttype}"

    risks = raw.get("false_positive_risks") or []
    if isinstance(risks, list) and risks and confidence < 0.88:
        risk_text = _lower_blob(*(str(r) for r in risks))
        if _contains_keyword(risk_text, _FALSE_POSITIVE_TRANSITION_KEYWORDS):
            return False, "VLM flagged focus/camera false-positive risks"

    # Sub-cuts should partition most of the shot without huge unexplained gaps
    sorted_subs = sorted(subs, key=lambda s: float(s.get("start_sec_in_shot", 0)))
    covered = sum(
        float(s.get("end_sec_in_shot", 0)) - float(s.get("start_sec_in_shot", 0))
        for s in sorted_subs
    )
    if shot_duration > 0 and covered < shot_duration * 0.45:
        return False, "sub-cuts cover too little of the shot"

    return True, "accepted"


def parse_transition_zones(
    raw_items: List[Dict[str, Any]],
    *,
    shot_start_sec: float,
    shot_duration: float,
) -> List[TransitionZone]:
    """Parse VLM transition zones (shot-relative) into typed objects."""
    zones: List[TransitionZone] = []
    for item in raw_items or []:
        if not isinstance(item, dict):
            continue
        rel_start = float(item.get("start_sec_in_shot", 0.0) or 0.0)
        rel_end = float(item.get("end_sec_in_shot", rel_start) or rel_start)
        rel_start = _clamp(rel_start, 0.0, shot_duration if shot_duration > 0 else rel_start)
        rel_end = _clamp(rel_end, rel_start, shot_duration if shot_duration > 0 else rel_end)
        if rel_end <= rel_start:
            continue
        zones.append(
            TransitionZone(
                start_sec_in_shot=rel_start,
                end_sec_in_shot=rel_end,
                start_sec_in_video=shot_start_sec + rel_start,
                end_sec_in_video=shot_start_sec + rel_end,
                transition_type=str(item.get("transition_type", "") or "").strip(),
            )
        )
    zones.sort(key=lambda z: z.start_sec_in_shot)
    return zones


def refine_transition_zones(
    zones: List[TransitionZone],
    sub_cuts: List[UndetectedSubCut],
    boundary_peaks: List[float],
    *,
    shot_start_sec: float,
    shot_duration: float,
    settings: SubCutTrimSettings,
) -> List[TransitionZone]:
    """Narrow and re-center transition zones on sub-cut boundaries."""
    half = min(
        settings.transition_trim_half_sec,
        settings.transition_max_zone_sec / 2.0,
    )
    if half <= 0:
        return []

    peaks = boundary_peaks or []
    if not peaks and zones:
        peaks = [(z.start_sec_in_shot + z.end_sec_in_shot) / 2.0 for z in zones]
    if not peaks and len(sub_cuts) >= 2:
        peaks = _sub_cut_boundary_points(sub_cuts)

    refined: List[TransitionZone] = []
    for peak in peaks:
        peak = _clamp(peak, 0.0, shot_duration)
        peak = _snap_to_subcut_boundary(
            peak,
            sub_cuts,
            tolerance_sec=BOUNDARY_SNAP_TOLERANCE_SEC,
        )
        rel_start = max(0.0, peak - half)
        rel_end = min(shot_duration, peak + half)
        if rel_end <= rel_start:
            continue
        refined.append(
            TransitionZone(
                start_sec_in_shot=rel_start,
                end_sec_in_shot=rel_end,
                start_sec_in_video=shot_start_sec + rel_start,
                end_sec_in_video=shot_start_sec + rel_end,
                transition_type="editorial_trim",
            )
        )
    return refined


def build_trimmed_effective_ranges(
    sub_cuts: List[UndetectedSubCut],
    transition_zones: List[TransitionZone],
    *,
    shot_start_sec: float,
    shot_end_sec: float,
    plot_description: str = "",
    settings: SubCutTrimSettings | None = None,
) -> tuple[List[ShotTimeRange], List[TransitionZone]]:
    """Build scene time ranges with inter-sub-cut transition regions removed."""
    settings = settings or SubCutTrimSettings()
    shot_duration = max(0.0, shot_end_sec - shot_start_sec)
    zones = list(transition_zones)
    if not zones and len(sub_cuts) >= 2:
        zones = refine_transition_zones(
            [],
            sub_cuts,
            _sub_cut_boundary_points(sub_cuts),
            shot_start_sec=shot_start_sec,
            shot_duration=shot_duration,
            settings=settings,
        )

    if not sub_cuts:
        return (
            [
                ShotTimeRange(
                    start_sec=shot_start_sec,
                    end_sec=shot_end_sec,
                    description=plot_description,
                )
            ],
            zones,
        )

    trimmed: List[ShotTimeRange] = []
    for sub in sorted(sub_cuts, key=lambda s: s.start_sec_in_shot):
        span = _trim_sub_cut_range(
            sub.start_sec_in_shot,
            sub.end_sec_in_shot,
            zones,
        )
        if span is None:
            logger.warning(
                "Sub-cut %.2f–%.2f fully consumed by transition trim — skipped",
                sub.start_sec_in_shot,
                sub.end_sec_in_shot,
            )
            continue
        rel_start, rel_end = span
        trimmed.append(
            ShotTimeRange(
                start_sec=shot_start_sec + rel_start,
                end_sec=shot_start_sec + rel_end,
                description=sub.sub_plot_description,
            )
        )

    if not trimmed:
        whole = _trim_sub_cut_range(0.0, shot_duration, zones)
        if whole:
            rel_start, rel_end = whole
            trimmed.append(
                ShotTimeRange(
                    start_sec=shot_start_sec + rel_start,
                    end_sec=shot_start_sec + rel_end,
                    description=plot_description,
                )
            )
        else:
            trimmed.append(
                ShotTimeRange(
                    start_sec=shot_start_sec,
                    end_sec=shot_end_sec,
                    description=plot_description,
                )
            )

    return trimmed, zones


def _trim_sub_cut_range(
    rel_start: float,
    rel_end: float,
    transition_zones: List[TransitionZone],
) -> Optional[tuple[float, float]]:
    """Remove transition-zone overlap from one sub-cut span (shot-relative)."""
    start = rel_start
    end = rel_end
    for zone in transition_zones:
        tz_start = zone.start_sec_in_shot
        tz_end = zone.end_sec_in_shot
        if tz_end <= start or tz_start >= end:
            continue
        if tz_start <= start < tz_end:
            start = tz_end
        if tz_start < end <= tz_end:
            end = tz_start
        elif start < tz_start and end > tz_end:
            end = tz_start
    if end - start < MIN_SCENE_DURATION_AFTER_TRIM_SEC:
        return None
    return start, end


def keyframe_in_transition_zone(
    timestamp_in_shot_sec: float,
    transition_zones: List[TransitionZone],
) -> bool:
    """Return True when a keyframe falls inside an excluded transition region."""
    for zone in transition_zones:
        if zone.start_sec_in_shot <= timestamp_in_shot_sec < zone.end_sec_in_shot:
            return True
    return False


def ensure_opening_keyframe(
    keyframes: List[ShotKeyframe],
    *,
    shot_start_sec: float,
    plot_description: str = "",
) -> List[ShotKeyframe]:
    """Ensure the first frame (timestamp_in_shot_sec=0.0) is always a keyframe."""
    description = (plot_description or "").strip()
    if description:
        opening_desc = f"Opening frame of the shot: {description[:200]}"
    else:
        opening_desc = "Opening frame of the shot — first frame of the clip."

    for idx, kf in enumerate(keyframes):
        role = (kf.role or "").strip().lower()
        if kf.timestamp_in_shot_sec <= 0.05 or role == "opening":
            keyframes[idx] = ShotKeyframe(
                description=kf.description or opening_desc,
                timestamp_in_shot_sec=0.0,
                timestamp_in_video_sec=shot_start_sec,
                role="opening",
            )
            result = list(keyframes)
            result.sort(key=lambda k: k.timestamp_in_shot_sec)
            return result

    opening = ShotKeyframe(
        description=opening_desc,
        timestamp_in_shot_sec=0.0,
        timestamp_in_video_sec=shot_start_sec,
        role="opening",
    )
    result = [opening, *keyframes]
    result.sort(key=lambda k: k.timestamp_in_shot_sec)
    return result


_CLOSING_KEYFRAME_ROLES = frozenset({"closing", "end", "ending", "final", "outro"})


def ensure_closing_keyframe(
    keyframes: List[ShotKeyframe],
    *,
    shot_start_sec: float,
    shot_end_sec: float,
    plot_description: str = "",
) -> List[ShotKeyframe]:
    """Ensure the last frame (timestamp_in_shot_sec ≈ duration) is always a keyframe.

    If a keyframe already exists near the end of the shot, its timestamp and
    role are updated to mark it as the closing keyframe.  Otherwise a new
    closing keyframe is appended.
    """
    duration = max(0.0, shot_end_sec - shot_start_sec)
    # Don't add a closing keyframe for extremely short shots — it would
    # overlap with the opening keyframe.
    if duration < 0.1:
        return keyframes

    description = (plot_description or "").strip()
    if description:
        closing_desc = f"Closing frame of the shot: {description[:200]}"
    else:
        closing_desc = "Closing frame of the shot — last frame of the clip."

    # Use a timestamp just before the exact end to avoid ffmpeg seek-at-EOF issues
    closing_ts = duration - 0.001

    for idx, kf in enumerate(keyframes):
        role = (kf.role or "").strip().lower()
        if role in _CLOSING_KEYFRAME_ROLES or abs(kf.timestamp_in_shot_sec - duration) <= 0.15:
            keyframes[idx] = ShotKeyframe(
                description=kf.description or closing_desc,
                timestamp_in_shot_sec=closing_ts,
                timestamp_in_video_sec=shot_end_sec,
                role="closing",
            )
            result = list(keyframes)
            result.sort(key=lambda k: k.timestamp_in_shot_sec)
            return result

    closing = ShotKeyframe(
        description=closing_desc,
        timestamp_in_shot_sec=closing_ts,
        timestamp_in_video_sec=shot_end_sec,
        role="closing",
    )
    result = [*keyframes, closing]
    result.sort(key=lambda k: k.timestamp_in_shot_sec)
    return result


def normalize_shot_vlm_payload(
    raw: Dict[str, Any],
    *,
    shot_id: str,
    scene_id: str,
    clip_path: str,
    shot_start_sec: float,
    shot_end_sec: float,
    trim_settings: SubCutTrimSettings | None = None,
) -> ShotAnalysis:
    """Map VLM relative timestamps to absolute positions; trim transition gaps."""
    settings = trim_settings or SubCutTrimSettings()
    duration = max(0.0, shot_end_sec - shot_start_sec)

    keyframes: List[ShotKeyframe] = []
    for item in raw.get("keyframes") or []:
        if not isinstance(item, dict):
            continue
        desc = str(item.get("description", "") or "").strip()
        if not desc:
            continue
        rel = float(item.get("timestamp_in_shot_sec", 0.0) or 0.0)
        rel = _clamp(rel, 0.0, duration if duration > 0 else rel)
        keyframes.append(
            ShotKeyframe(
                description=desc,
                timestamp_in_shot_sec=rel,
                timestamp_in_video_sec=shot_start_sec + rel,
                role=str(item.get("role", "") or "").strip(),
            )
        )

    plot = str(raw.get("plot_description", "") or "").strip()
    accepted, reason = validate_sub_cut_analysis(
        raw,
        shot_duration=duration,
        settings=settings,
    )

    undetected: List[UndetectedSubCut] = []
    transition_zones: List[TransitionZone] = []
    has_sub_cuts = False

    if accepted:
        has_sub_cuts = True
        for item in raw.get("undetected_sub_cuts") or []:
            if not isinstance(item, dict):
                continue
            rel_start = float(item.get("start_sec_in_shot", 0.0) or 0.0)
            rel_end = float(item.get("end_sec_in_shot", rel_start) or rel_start)
            rel_start = _clamp(rel_start, 0.0, duration if duration > 0 else rel_start)
            rel_end = _clamp(rel_end, rel_start, duration if duration > 0 else rel_end)
            undetected.append(
                UndetectedSubCut(
                    start_sec_in_shot=rel_start,
                    end_sec_in_shot=rel_end,
                    start_sec_in_video=shot_start_sec + rel_start,
                    end_sec_in_video=shot_start_sec + rel_end,
                    transition_type=str(item.get("transition_type", "") or "").strip(),
                    sub_plot_description=str(
                        item.get("sub_plot_description", "") or ""
                    ).strip(),
                )
            )

        raw_zones = parse_transition_zones(
            raw.get("transition_zones") or [],
            shot_start_sec=shot_start_sec,
            shot_duration=duration,
        )
        boundary_peaks = parse_transition_boundaries(raw)
        transition_zones = refine_transition_zones(
            raw_zones,
            undetected,
            boundary_peaks,
            shot_start_sec=shot_start_sec,
            shot_duration=duration,
            settings=settings,
        )
        effective, transition_zones = build_trimmed_effective_ranges(
            undetected,
            transition_zones,
            shot_start_sec=shot_start_sec,
            shot_end_sec=shot_end_sec,
            plot_description=plot,
            settings=settings,
        )
        keyframes = [
            kf
            for kf in keyframes
            if not keyframe_in_transition_zone(kf.timestamp_in_shot_sec, transition_zones)
        ]
        logger.info(
            "Shot %s: accepted %d sub-cut(s), %d transition trim zone(s)",
            shot_id,
            len(undetected),
            len(transition_zones),
        )
    else:
        if raw.get("has_undetected_sub_cuts"):
            logger.info(
                "Shot %s: rejected sub-cut split (%s) — keeping single scene",
                shot_id,
                reason,
            )
        effective = [
            ShotTimeRange(
                start_sec=shot_start_sec,
                end_sec=shot_end_sec,
                description=plot,
            )
        ]

    keyframes = ensure_opening_keyframe(
        keyframes,
        shot_start_sec=shot_start_sec,
        plot_description=plot,
    )
    keyframes = ensure_closing_keyframe(
        keyframes,
        shot_start_sec=shot_start_sec,
        shot_end_sec=shot_end_sec,
        plot_description=plot,
    )

    return ShotAnalysis(
        shot_id=shot_id,
        scene_id=scene_id,
        clip_path=clip_path,
        pyscenedetect_start_sec=shot_start_sec,
        pyscenedetect_end_sec=shot_end_sec,
        plot_description=plot,
        keyframes=keyframes,
        has_undetected_sub_cuts=has_sub_cuts,
        undetected_sub_cuts=undetected,
        transition_zones=transition_zones,
        effective_time_ranges_in_video=effective,
    )


def resolve_shot_scene_ranges(
    shot: ShotAnalysis,
    *,
    trim_settings: SubCutTrimSettings | None = None,
) -> List[ShotTimeRange]:
    """Recompute trimmed scene ranges (e.g. when loading a saved analysis file)."""
    settings = trim_settings or SubCutTrimSettings()
    if shot.has_undetected_sub_cuts and shot.undetected_sub_cuts:
        effective, _ = build_trimmed_effective_ranges(
            shot.undetected_sub_cuts,
            shot.transition_zones,
            shot_start_sec=shot.pyscenedetect_start_sec,
            shot_end_sec=shot.pyscenedetect_end_sec,
            plot_description=shot.plot_description,
            settings=settings,
        )
        return effective
    if shot.effective_time_ranges_in_video:
        return shot.effective_time_ranges_in_video
    return [
        ShotTimeRange(
            start_sec=shot.pyscenedetect_start_sec,
            end_sec=shot.pyscenedetect_end_sec,
            description=shot.plot_description,
        )
    ]


def trim_settings_from_config(config: Any) -> SubCutTrimSettings:
    """Build trim settings from :class:`AgentConfig`."""
    return SubCutTrimSettings(
        transition_trim_half_sec=getattr(
            config, "scene_transition_trim_half_sec", DEFAULT_TRANSITION_TRIM_HALF_SEC
        ),
        transition_max_zone_sec=getattr(
            config, "scene_transition_max_zone_sec", DEFAULT_TRANSITION_MAX_ZONE_SEC
        ),
        sub_cut_min_confidence=getattr(
            config, "scene_sub_cut_min_confidence", DEFAULT_SUB_CUT_MIN_CONFIDENCE
        ),
        sub_cut_min_duration_sec=getattr(
            config, "scene_sub_cut_min_duration_sec", DEFAULT_SUB_CUT_MIN_DURATION_SEC
        ),
    )
