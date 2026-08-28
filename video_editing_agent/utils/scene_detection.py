"""PySceneDetect wrappers with multi-detector fusion to reduce missed cuts."""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

from scenedetect import (
    AdaptiveDetector,
    ContentDetector,
    HashDetector,
    HistogramDetector,
    SceneManager,
    ThresholdDetector,
    open_video,
)
from scenedetect.scene_manager import CutList

logger = logging.getLogger(__name__)

SceneSegment = Tuple[float, float]  # (start_sec, end_sec)

# Segments shorter than this are "short" and get extra scrutiny.
SHORT_SEGMENT_THRESHOLD_SEC = 1.5
# When two cuts are closer than this, they might be a missed transition
# (e.g. a quick flash or partial cut). Merge them.
MERGE_NEARBY_CUTS_SEC = 0.5
# Minimum segment duration — segments shorter than this are merged into
# their neighbors to avoid over-fragmentation of continuous video.
MIN_SEGMENT_DURATION_SEC = 1.0
# A cut is considered "very certain" (high confidence) if at least this many
# detectors agree. High-confidence cuts are kept even when they would create
# a segment shorter than MIN_SEGMENT_DURATION_SEC — i.e. we only split off a
# sub-1-second segment when we are very sure the cut is real.
HIGH_CONFIDENCE_VOTES = 3
# For short segments, re-run detection with even more aggressive thresholds
# to catch missed sub-cuts.
SHORT_SEGMENT_RECHECK_CONTENT_THRESHOLD = 15.0
SHORT_SEGMENT_RECHECK_ADAPTIVE_THRESHOLD = 1.8
SHORT_SEGMENT_RECHECK_MIN_CONTENT_VAL = 5.0
SHORT_SEGMENT_RECHECK_MIN_SCENE_LEN = 2


def _internal_cut_seconds(scene_list: CutList) -> List[float]:
    if len(scene_list) <= 1:
        return []
    return [start.get_seconds() for start, _ in scene_list[1:]]


def _run_detector(video_path: str, detector) -> CutList:
    video = open_video(video_path)
    manager = SceneManager()
    manager.add_detector(detector)
    manager.detect_scenes(video)
    return manager.get_scene_list()


def merge_cut_times(
    cut_time_lists: Sequence[Sequence[float]],
    *,
    min_gap_sec: float = 0.05,
) -> List[float]:
    """Union cut times from multiple detectors, dropping near-duplicates."""
    merged: List[float] = []
    for t in sorted({t for cuts in cut_time_lists for t in cuts}):
        if not merged or t - merged[-1] >= min_gap_sec:
            merged.append(t)
    return merged


def _vote_merge_cut_times(
    cut_time_lists: Sequence[Sequence[float]],
    *,
    min_gap_sec: float = 0.5,
    min_votes: int = 2,
) -> List[Tuple[float, int]]:
    """Merge cut times using voting consensus across detectors.

    A cut is accepted only if at least ``min_votes`` detectors found a cut
    within ``min_gap_sec`` of each other. This prevents a single detector's
    false positive from splitting a continuous shot.

    Args:
        cut_time_lists: One list of cut times per detector.
        min_gap_sec: Cuts within this window from different detectors are
            considered the same cut.
        min_votes: Minimum number of detectors that must agree on a cut.

    Returns:
        List of ``(cut_time, vote_count)`` tuples sorted by time.
    """
    if not cut_time_lists:
        return []

    # Collect all candidate cuts with their detector source index
    candidates: List[Tuple[float, int]] = []
    for det_idx, cuts in enumerate(cut_time_lists):
        for t in cuts:
            candidates.append((t, det_idx))

    candidates.sort(key=lambda x: x[0])

    # Group nearby cuts and count unique detector votes per group
    merged: List[Tuple[float, int]] = []
    i = 0
    while i < len(candidates):
        # Start a new group
        group_times: List[float] = [candidates[i][0]]
        group_detectors: set[int] = {candidates[i][1]}
        j = i + 1
        while j < len(candidates) and candidates[j][0] - group_times[-1] < min_gap_sec:
            group_times.append(candidates[j][0])
            group_detectors.add(candidates[j][1])
            j += 1

        votes = len(group_detectors)
        if votes >= min_votes:
            # Accept this cut — use the median time of the group
            group_times.sort()
            median_t = group_times[len(group_times) // 2]
            if not merged or median_t - merged[-1][0] >= min_gap_sec:
                merged.append((median_t, votes))
            logger.debug(
                "Cut at %.2fs accepted (%d detector votes)",
                median_t,
                votes,
            )
        else:
            logger.debug(
                "Cut at %.2fs rejected (only %d detector vote(s), need %d)",
                group_times[0],
                votes,
                min_votes,
            )

        i = j

    return merged


def segments_from_cuts(cut_times: Sequence[float], duration_sec: float) -> List[SceneSegment]:
    if duration_sec <= 0:
        return []
    boundaries = [0.0, *cut_times, duration_sec]
    segments: List[SceneSegment] = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end > start:
            segments.append((start, end))
    return segments or [(0.0, duration_sec)]


def _merge_nearby_cuts(
    cuts_with_votes: List[Tuple[float, int]],
    min_gap: float = MERGE_NEARBY_CUTS_SEC,
) -> List[Tuple[float, int]]:
    """Merge cuts that are very close together — likely a single transition detected as two.

    When merging two nearby cuts, the resulting vote count is the maximum of
    the two (preserving the strongest detector agreement).
    """
    if len(cuts_with_votes) <= 1:
        return cuts_with_votes
    merged: List[Tuple[float, int]] = [cuts_with_votes[0]]
    for t, votes in cuts_with_votes[1:]:
        if t - merged[-1][0] < min_gap:
            old_t, old_votes = merged[-1]
            merged[-1] = ((old_t + t) / 2.0, max(old_votes, votes))
        else:
            merged.append((t, votes))
    return merged


def _enforce_min_segment_duration(
    cuts_with_votes: List[Tuple[float, int]],
    duration_sec: float,
    min_duration: float = MIN_SEGMENT_DURATION_SEC,
    high_confidence_votes: int = HIGH_CONFIDENCE_VOTES,
) -> List[Tuple[float, int]]:
    """Remove cuts that create segments shorter than ``min_duration``.

    A cut is only removed if it is **not** high-confidence.  High-confidence
    cuts (``votes >= high_confidence_votes``) are kept even when they create
    a short segment, because we are "very certain" the cut is real.  This
    implements the rule: *unless very certain, do not split off a segment
    shorter than 1 second*.

    Iteratively removes the lowest-confidence cut that produces a short
    segment, merging that segment into its neighbor, until all remaining
    short segments are backed by high-confidence cuts (or no cuts remain).
    """
    if not cuts_with_votes or min_duration <= 0:
        return cuts_with_votes

    cuts = list(cuts_with_votes)
    changed = True
    while changed and cuts:
        changed = False
        times = [t for t, _ in cuts]
        boundaries = [0.0, *times, duration_sec]
        shortest_idx = -1
        shortest_dur = float("inf")
        shortest_votes = 0
        for i in range(len(boundaries) - 1):
            seg_dur = boundaries[i + 1] - boundaries[i]
            if seg_dur < min_duration:
                # Determine which cut bounds this segment and its votes
                if i == 0:
                    cut_idx, cut_votes = 0, cuts[0][1]
                elif i == len(boundaries) - 2:
                    cut_idx, cut_votes = len(cuts) - 1, cuts[-1][1]
                else:
                    left_dur = boundaries[i] - boundaries[i - 1]
                    right_dur = boundaries[i + 2] - boundaries[i + 1]
                    if left_dur <= right_dur:
                        cut_idx, cut_votes = i - 1, cuts[i - 1][1]
                    else:
                        cut_idx, cut_votes = i, cuts[i][1]
                # High-confidence cuts are kept — skip removal
                if cut_votes >= high_confidence_votes:
                    continue
                # Among removable cuts, pick the one producing the shortest segment
                if seg_dur < shortest_dur:
                    shortest_dur = seg_dur
                    shortest_idx = cut_idx
                    shortest_votes = cut_votes

        if shortest_idx >= 0:
            removed_t, removed_votes = cuts.pop(shortest_idx)
            logger.info(
                "Merged segment <%.1fs by removing low-confidence cut at %.2fs (%d votes)",
                shortest_dur,
                removed_t,
                removed_votes,
            )
            changed = True

    # Log any remaining short segments (backed by high-confidence cuts)
    times = [t for t, _ in cuts]
    boundaries = [0.0, *times, duration_sec]
    for i in range(len(boundaries) - 1):
        seg_dur = boundaries[i + 1] - boundaries[i]
        if seg_dur < min_duration:
            logger.info(
                "Keeping short segment %.2fs–%.2fs (%.2fs) — backed by high-confidence cut",
                boundaries[i],
                boundaries[i + 1],
                seg_dur,
            )

    return cuts


def _recheck_short_segment(
    video_path: str,
    start_sec: float,
    end_sec: float,
) -> List[float]:
    """Re-run detection on a short segment with aggressive thresholds.

    Returns any new sub-cut times found within [start_sec, end_sec].
    """
    if end_sec - start_sec >= SHORT_SEGMENT_THRESHOLD_SEC:
        return []

    logger.info(
        "Short segment %.2fs–%.2fs (%.2fs) — rechecking with aggressive thresholds",
        start_sec,
        end_sec,
        end_sec - start_sec,
    )
    detector = ContentDetector(
        threshold=SHORT_SEGMENT_RECHECK_CONTENT_THRESHOLD,
        min_scene_len=SHORT_SEGMENT_RECHECK_MIN_SCENE_LEN,
    )
    try:
        # PySceneDetect doesn't support time-range filtering directly,
        # so we run on the full video and filter results to the segment.
        scene_list = _run_detector(video_path, detector)
        all_cuts = _internal_cut_seconds(scene_list)
        sub_cuts = [c for c in all_cuts if start_sec < c < end_sec]
        if sub_cuts:
            logger.info(
                "Short segment recheck found %d additional cut(s): %s",
                len(sub_cuts),
                [f"{c:.2f}s" for c in sub_cuts],
            )
        return sub_cuts
    except Exception as exc:
        logger.warning("Short segment recheck failed: %s", exc)
        return []


def detect_scene_segments(
    video_path: str,
    *,
    mode: str = "enhanced",
    content_threshold: float = 22.0,
    adaptive_threshold: float = 2.5,
    min_content_val: float = 8.0,
    min_scene_len: int = 5,
    hash_threshold: float = 0.395,
    histogram_threshold: float = 0.05,
    duration_sec: float | None = None,
) -> List[SceneSegment]:
    """Detect physical shot boundaries as ``(start_sec, end_sec)`` pairs.

    Uses multi-detector fusion to maximize cut recall. ``enhanced`` mode runs
    4 detectors (content + adaptive + hash + histogram) and unions their results.

    Args:
        mode: ``content``, ``adaptive``, ``hybrid`` (content+adaptive),
            or ``enhanced`` (content+adaptive+hash+histogram — recommended).
        content_threshold: ContentDetector HSV delta threshold (lower = more cuts).
        adaptive_threshold: AdaptiveDetector ratio threshold (lower = more cuts).
        min_content_val: AdaptiveDetector absolute content floor (lower catches weak cuts).
        min_scene_len: Minimum frames between cuts (lower allows rapid consecutive cuts).
        hash_threshold: HashDetector threshold (lower = more cuts).
        histogram_threshold: HistogramDetector max relative difference (lower = more cuts).
    """
    from video_editing_agent.utils.ffmpeg_utils import probe_duration

    if duration_sec is None:
        duration_sec = probe_duration(video_path)

    mode = (mode or "enhanced").lower().strip()
    min_scene_len = max(0, int(min_scene_len))

    detectors: list[tuple[str, object]] = []
    if mode in ("content", "hybrid", "enhanced"):
        detectors.append(
            (
                "content",
                ContentDetector(
                    threshold=content_threshold,
                    min_scene_len=min_scene_len,
                ),
            )
        )
    if mode in ("adaptive", "hybrid", "enhanced"):
        detectors.append(
            (
                "adaptive",
                AdaptiveDetector(
                    adaptive_threshold=adaptive_threshold,
                    min_content_val=min_content_val,
                    min_scene_len=min_scene_len,
                ),
            )
        )
    if mode == "enhanced":
        detectors.append(
            (
                "hash",
                HashDetector(
                    threshold=hash_threshold,
                    min_scene_len=min_scene_len,
                ),
            )
        )
        detectors.append(
            (
                "histogram",
                HistogramDetector(
                    threshold=histogram_threshold,
                    min_scene_len=min_scene_len,
                ),
            )
        )
    if not detectors:
        raise ValueError(f"Unknown scene detect mode: {mode!r}")

    cut_lists: List[List[float]] = []
    for name, detector in detectors:
        scene_list = _run_detector(video_path, detector)
        cuts = _internal_cut_seconds(scene_list)
        cut_lists.append(cuts)
        logger.info(
            "Scene detect [%s]: %d scene(s), %d cut(s)",
            name,
            len(scene_list) or 1,
            len(cuts),
        )

    if len(cut_lists) > 1:
        # Use voting consensus: a cut is accepted only if >=2 detectors agree.
        # This prevents a single detector's false positive from splitting a
        # continuous shot into two segments.
        min_votes = min(2, len(cut_lists))
        cuts_with_votes = _vote_merge_cut_times(
            cut_lists,
            min_gap_sec=MERGE_NEARBY_CUTS_SEC,
            min_votes=min_votes,
        )
        detector_names = [name for name, _ in detectors]
        logger.info(
            "Scene detect [%s vote-merged]: %d cut(s) (need %d votes) (%s)",
            mode,
            len(cuts_with_votes),
            min_votes,
            ", ".join(f"{n}={len(c)}" for n, c in zip(detector_names, cut_lists)),
        )
    else:
        cuts_with_votes = [(t, 1) for t in cut_lists[0]] if cut_lists else []

    # ── Short segment recheck ──────────────────────────────────────────
    # Segments around 1 second are prone to missed sub-cuts. Re-run detection
    # with aggressive thresholds on these segments to catch any missed cuts.
    preliminary_cuts = [t for t, _ in cuts_with_votes]
    preliminary_segments = segments_from_cuts(preliminary_cuts, duration_sec)
    extra_cuts: List[float] = []
    for seg_start, seg_end in preliminary_segments:
        if seg_end - seg_start < SHORT_SEGMENT_THRESHOLD_SEC:
            sub_cuts = _recheck_short_segment(video_path, seg_start, seg_end)
            extra_cuts.extend(sub_cuts)

    if extra_cuts:
        # Recheck cuts are low-confidence (single detector, votes=1).
        existing_times = {t for t, _ in cuts_with_votes}
        for t in extra_cuts:
            if t not in existing_times:
                cuts_with_votes.append((t, 1))
        cuts_with_votes.sort(key=lambda x: x[0])
        cuts_with_votes = _merge_nearby_cuts(cuts_with_votes)
        logger.info(
            "Short segment recheck: added %d cut(s), total now %d",
            len(extra_cuts),
            len(cuts_with_votes),
        )

    # ── Enforce minimum segment duration ───────────────────────────────
    # Remove cuts that would create segments shorter than MIN_SEGMENT_DURATION_SEC
    # — but only if the cut is NOT high-confidence.  High-confidence cuts
    # (≥ HIGH_CONFIDENCE_VOTES detector agreement) are kept even when they
    # create a sub-1-second segment, because we are "very certain" the cut
    # is real.
    cuts_with_votes = _enforce_min_segment_duration(cuts_with_votes, duration_sec)

    cut_times = [t for t, _ in cuts_with_votes]
    segments = segments_from_cuts(cut_times, duration_sec)
    logger.info(
        "Scene detect final: mode=%s -> %d segment(s)",
        mode,
        len(segments),
    )
    return segments
