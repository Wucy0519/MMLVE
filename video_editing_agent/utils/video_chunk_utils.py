"""Split long scene clips into model-safe chunks and stitch edited results."""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

from video_editing_agent.utils.ffmpeg_utils import (
    concat_videos,
    cut_video_segment,
    extract_frame_at,
    probe_duration,
    probe_fps,
)

logger = logging.getLogger(__name__)

CHUNK_MANIFEST_FILENAME = "chunk_manifest.json"


@dataclass
class ChunkTimeRange:
    """Inclusive start, exclusive end is NOT used — end is the boundary timestamp."""

    index: int
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(self.end_sec - self.start_sec, 0.04)


@dataclass
class ChunkManifestEntry:
    index: int
    start_sec: float
    end_sec: float
    source_path: str
    edited_path: str = ""
    edited_last_frame_path: str = ""
    edit_operation_prompt: str = ""


def compute_chunk_time_ranges(
    duration_sec: float,
    max_chunk_sec: float = 10.0,
    min_tail_sec: float = 4.0,
) -> List[ChunkTimeRange]:
    """Return chunk boundaries; adjacent chunks share the boundary timestamp frame.

    Splits at *max_chunk_sec* steps. If the final segment would be shorter than
    *min_tail_sec*, it is merged into the previous segment (e.g. 24s → 10s + 14s;
    12s stays a single 12s clip).
    """
    if duration_sec <= max_chunk_sec + 1e-3:
        return [ChunkTimeRange(index=0, start_sec=0.0, end_sec=duration_sec)]

    boundaries = [0.0]
    t = max_chunk_sec
    while t < duration_sec - 1e-3:
        boundaries.append(t)
        t += max_chunk_sec
    boundaries.append(duration_sec)

    while len(boundaries) >= 3:
        last_dur = boundaries[-1] - boundaries[-2]
        if last_dur >= min_tail_sec - 1e-3:
            break
        boundaries.pop(-2)

    ranges = [
        ChunkTimeRange(index=i, start_sec=boundaries[i], end_sec=boundaries[i + 1])
        for i in range(len(boundaries) - 1)
    ]
    return ranges


def scene_chunks_dir(scenes_dir: str, scene_id: str) -> str:
    return os.path.join(scenes_dir, scene_id, "chunks")


def chunk_source_path(chunks_dir: str, index: int) -> str:
    return os.path.join(chunks_dir, f"chunk_{index:03d}_source.mp4")


def chunk_edited_path(chunks_dir: str, index: int) -> str:
    return os.path.join(chunks_dir, f"chunk_{index:03d}_edited.mp4")


def chunk_first_frame_path(chunks_dir: str, index: int) -> str:
    return os.path.join(chunks_dir, f"chunk_{index:03d}_first.png")


def chunk_edited_last_frame_path(chunks_dir: str, index: int) -> str:
    return os.path.join(chunks_dir, f"chunk_{index:03d}_edited_last.png")


def chunk_manifest_path(chunks_dir: str) -> str:
    return os.path.join(chunks_dir, CHUNK_MANIFEST_FILENAME)


def split_scene_video_into_chunks(
    source_clip_path: str,
    chunks_dir: str,
    max_chunk_sec: float = 10.0,
    *,
    min_tail_sec: float = 4.0,
    reencode: bool = True,
) -> Tuple[List[ChunkManifestEntry], bool]:
    """Split *source_clip_path* when longer than *max_chunk_sec*.

    Returns:
        (manifest_entries, was_split)
    """
    os.makedirs(chunks_dir, exist_ok=True)
    duration = probe_duration(source_clip_path)
    ranges = compute_chunk_time_ranges(duration, max_chunk_sec, min_tail_sec)
    was_split = len(ranges) > 1

    entries: List[ChunkManifestEntry] = []
    for chunk in ranges:
        src_out = chunk_source_path(chunks_dir, chunk.index)
        if not os.path.exists(src_out):
            cut_video_segment(
                source_clip_path,
                chunk.start_sec,
                chunk.end_sec,
                src_out,
                reencode=reencode,
            )
        first_png = chunk_first_frame_path(chunks_dir, chunk.index)
        if not os.path.exists(first_png):
            extract_frame_at(src_out, 0.0, first_png)
        entries.append(
            ChunkManifestEntry(
                index=chunk.index,
                start_sec=chunk.start_sec,
                end_sec=chunk.end_sec,
                source_path=src_out,
            )
        )

    save_chunk_manifest(chunks_dir, entries, was_split=was_split)
    logger.info(
        "Split %s (%.2fs) → %d chunk(s) in %s",
        os.path.basename(source_clip_path),
        duration,
        len(entries),
        chunks_dir,
    )
    return entries, was_split


def save_chunk_manifest(
    chunks_dir: str,
    entries: List[ChunkManifestEntry],
    *,
    was_split: bool,
) -> str:
    path = chunk_manifest_path(chunks_dir)
    payload = {
        "was_split": was_split,
        "chunks": [asdict(e) for e in entries],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


def load_chunk_manifest(chunks_dir: str) -> Optional[dict]:
    path = chunk_manifest_path(chunks_dir)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def update_chunk_manifest_entry(
    chunks_dir: str,
    index: int,
    *,
    edited_path: str = "",
    edited_last_frame_path: str = "",
    edit_operation_prompt: str = "",
) -> None:
    data = load_chunk_manifest(chunks_dir) or {"was_split": True, "chunks": []}
    chunks = data.get("chunks") or []
    for item in chunks:
        if int(item.get("index", -1)) == index:
            if edited_path:
                item["edited_path"] = edited_path
            if edited_last_frame_path:
                item["edited_last_frame_path"] = edited_last_frame_path
            if edit_operation_prompt:
                item["edit_operation_prompt"] = edit_operation_prompt
            break
    else:
        chunks.append(
            {
                "index": index,
                "edited_path": edited_path,
                "edited_last_frame_path": edited_last_frame_path,
                "edit_operation_prompt": edit_operation_prompt,
            }
        )
    data["chunks"] = chunks
    with open(chunk_manifest_path(chunks_dir), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def extract_last_frame(video_path: str, output_path: str) -> str:
    """Extract the last video frame."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    dur = probe_duration(video_path)
    fps = max(probe_fps(video_path), 1.0)
    ts = max(0.0, dur - 1.0 / fps)
    return extract_frame_at(video_path, ts, output_path)


def trim_skip_first_frame(input_path: str, output_path: str) -> str:
    """Drop the first frame so adjacent chunk boundaries are not duplicated."""
    from video_editing_agent.utils.ffmpeg_utils import _run

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fps = max(probe_fps(input_path), 1.0)
    frame_dur = 1.0 / fps
    _run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ss", f"{frame_dur:.6f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-an",
            output_path,
            "-loglevel", "error",
        ],
        desc="trim skip first frame",
    )
    return output_path


def concat_edited_chunks(
    edited_paths: List[str],
    output_path: str,
    *,
    dedup_boundary_frames: bool = True,
) -> str:
    """Concatenate edited chunk clips; skip duplicate boundary frames after chunk 0."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not edited_paths:
        raise ValueError("No edited chunk paths to concatenate")
    if len(edited_paths) == 1:
        if os.path.abspath(edited_paths[0]) != os.path.abspath(output_path):
            shutil.copy2(edited_paths[0], output_path)
        return output_path

    paths_for_concat: List[str] = []
    temp_trimmed: List[str] = []
    for i, path in enumerate(edited_paths):
        if i == 0 or not dedup_boundary_frames:
            paths_for_concat.append(path)
            continue
        trimmed = path + ".trim_head.mp4"
        trim_skip_first_frame(path, trimmed)
        paths_for_concat.append(trimmed)
        temp_trimmed.append(trimmed)

    list_path = output_path + ".concat_list.txt"
    with open(list_path, "w", encoding="utf-8") as fh:
        for p in paths_for_concat:
            fh.write(f"file '{os.path.abspath(p)}'\n")

    concat_videos(list_path, output_path, reencode=True)

    for p in temp_trimmed + [list_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    return output_path
