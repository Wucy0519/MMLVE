"""ffmpeg / ffprobe helpers for scene extraction and assembly."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class FfmpegError(RuntimeError):
    """ffmpeg command failed."""


def _run(cmd: List[str], *, desc: str = "ffmpeg") -> subprocess.CompletedProcess:
    """Run a subprocess and raise on failure."""
    logger.debug("Running %s: %s", desc, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "")[-800:]
        raise FfmpegError(f"{desc} failed (code={result.returncode}): {stderr}")
    return result


def probe_duration(video_path: str) -> float:
    """Return video duration in seconds."""
    result = _run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            video_path,
        ],
        desc="ffprobe duration",
    )
    try:
        return max(float(result.stdout.strip()), 0.01)
    except ValueError as exc:
        raise FfmpegError(f"Invalid duration for {video_path}") from exc


def probe_fps(video_path: str) -> float:
    """Return average frame rate."""
    result = _run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate",
            "-of", "csv=p=0",
            video_path,
        ],
        desc="ffprobe fps",
    )
    raw = result.stdout.strip()
    if "/" in raw:
        num, den = raw.split("/", 1)
        den_f = float(den) or 1.0
        return float(num) / den_f
    return float(raw or 25.0)


def probe_video_size(video_path: str) -> Tuple[int, int]:
    """Return video width and height."""
    result = _run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            video_path,
        ],
        desc="ffprobe size",
    )
    raw = result.stdout.strip()
    try:
        width_s, height_s = raw.split("x", 1)
        return int(width_s), int(height_s)
    except ValueError as exc:
        raise FfmpegError(f"Invalid video size for {video_path}: {raw!r}") from exc


def determine_resolution_mode(video_path: str) -> str:
    """Determine the processing resolution mode based on input video height.

    Returns "480p" when the video height is below 480 pixels (super-low
    resolution), otherwise "720p".  This drives model selection for both
    image editing and video editing to avoid upscaling artifacts.

    If the probe fails, falls back to "720p" (the safe default).
    """
    try:
        _, height = probe_video_size(video_path)
    except (FfmpegError, Exception) as exc:
        logger.warning(
            "Failed to probe video size for %s — defaulting to 720p: %s",
            video_path,
            exc,
        )
        return "720p"
    if height < 480:
        logger.info(
            "Input video height=%d (<480) — using 480p processing mode",
            height,
        )
        return "480p"
    logger.info(
        "Input video height=%d (>=480) — using 720p processing mode",
        height,
    )
    return "720p"


def extract_frame_at(
    video_path: str,
    timestamp_sec: float,
    output_path: str,
) -> str:
    """Extract a single frame at the given timestamp."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _run(
        [
            "ffmpeg", "-y",
            "-ss", str(max(0.0, timestamp_sec)),
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            output_path,
            "-loglevel", "error",
        ],
        desc="extract frame",
    )
    return output_path


def prune_scene_frames_keep_anchors(
    frame_dir: str,
    first_frame_path: str,
    last_frame_path: str = "",
) -> int:
    """Delete extracted scene frames except first/last anchor frames.

    Returns:
        Number of files removed.
    """
    if not frame_dir or not os.path.isdir(frame_dir):
        return 0

    keep = {
        os.path.abspath(p)
        for p in (first_frame_path, last_frame_path)
        if p
    }
    removed = 0
    for name in sorted(os.listdir(frame_dir)):
        if not (name.startswith("frame_") and name.endswith(".png")):
            continue
        path = os.path.join(frame_dir, name)
        if os.path.abspath(path) in keep:
            continue
        try:
            os.remove(path)
            removed += 1
        except OSError as exc:
            logger.warning("Failed to remove scene frame %s: %s", path, exc)
    return removed


def prune_scene_frames_keep_first(frame_dir: str, first_frame_path: str) -> int:
    """Backward-compatible wrapper — keeps first frame only."""
    return prune_scene_frames_keep_anchors(frame_dir, first_frame_path)


def extract_frames_range(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_dir: str,
    fps: float = 8.0,
) -> List[str]:
    """Extract frames for [start_sec, end_sec) at given fps."""
    os.makedirs(output_dir, exist_ok=True)
    duration = max(end_sec - start_sec, 0.04)
    pattern = os.path.join(output_dir, "frame_%04d.png")
    _run(
        [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", video_path,
            "-t", str(duration),
            "-vf", f"fps={fps}",
            pattern,
            "-loglevel", "error",
        ],
        desc="extract frames range",
    )
    frames = sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith("frame_") and f.endswith(".png")
    )
    return frames


def cut_video_segment(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    *,
    reencode: bool = False,
) -> str:
    """Cut a video segment [start_sec, end_sec)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    duration = max(end_sec - start_sec, 0.04)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
    ]
    if reencode:
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"]
    else:
        cmd += ["-c", "copy"]
    cmd += [output_path, "-loglevel", "error"]
    _run(cmd, desc="cut segment")
    return output_path


def concat_videos(
    concat_list_path: str,
    output_path: str,
    *,
    reencode: bool = True,
) -> str:
    """Concatenate videos via ffmpeg concat demuxer."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list_path,
    ]
    if reencode:
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-an"]
    else:
        cmd += ["-c", "copy"]
    cmd += [output_path, "-loglevel", "error"]
    _run(cmd, desc="concat videos")
    return output_path


def has_audio_stream(video_path: str) -> bool:
    """Return True if the video file contains an audio stream."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    return "audio" in (result.stdout or "")


def extract_audio(
    video_path: str,
    output_path: str,
) -> str:
    """Extract audio track to AAC."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn", "-acodec", "aac", "-b:a", "192k",
            output_path,
            "-loglevel", "error",
        ],
        desc="extract audio",
    )
    return output_path


def mux_video_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
    *,
    shortest: bool = True,
) -> str:
    """Mux video and audio into final output."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
    ]
    if shortest:
        cmd.append("-shortest")
    cmd += [output_path, "-loglevel", "error"]
    _run(cmd, desc="mux av")
    return output_path


def image_to_video_clip(
    image_path: str,
    output_path: str,
    duration_sec: float,
    fps: float = 25.0,
) -> str:
    """Encode a still image as a video clip (fallback)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _run(
        [
            "ffmpeg", "-y",
            "-loop", "1", "-i", image_path,
            "-t", str(duration_sec),
            "-r", str(fps),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            output_path,
            "-loglevel", "error",
        ],
        desc="image to video",
    )
    return output_path


def static_video_from_source_first_frame(
    source_clip_path: str,
    output_clip_path: str,
    *,
    frame_png_path: str = "",
) -> str:
    """Placeholder clip: freeze the source video's first frame for the clip duration."""
    png = frame_png_path or (output_clip_path + ".source_first_frame.png")
    extract_frame_at(source_clip_path, 0.0, png)
    duration = probe_duration(source_clip_path)
    image_to_video_clip(png, output_clip_path, duration)
    return output_clip_path


def static_video_from_image(
    image_path: str,
    source_clip_path: str,
    output_clip_path: str,
) -> str:
    """Placeholder clip: freeze a given image for the source clip duration."""
    duration = probe_duration(source_clip_path)
    image_to_video_clip(image_path, output_clip_path, duration)
    return output_clip_path


def resize_video_to_duration(
    input_path: str,
    output_path: str,
    target_duration: float,
    fps: float = 25.0,
) -> str:
    """Trim or loop-pad a clip to target duration."""
    src_dur = probe_duration(input_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if src_dur >= target_duration - 0.05:
        _run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-t", str(target_duration),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                output_path, "-loglevel", "error",
            ],
            desc="trim video",
        )
    else:
        loops = int(target_duration / max(src_dur, 0.1)) + 1
        _run(
            [
                "ffmpeg", "-y",
                "-stream_loop", str(loops),
                "-i", input_path,
                "-t", str(target_duration),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                output_path, "-loglevel", "error",
            ],
            desc="loop pad video",
        )
    return output_path


def extend_video_by_duplicating_frames(
    input_path: str,
    output_path: str,
    target_duration: float,
) -> str:
    """Extend a short clip to *target_duration* by duplicating middle frames.

    Stretches the PTS so frames are spread across the longer duration, but
    keeps the **original frame rate** by duplicating frames to fill gaps.
    Frames are distributed throughout the video (not just appended at the end),
    so there is no frozen last-frame section.
    """
    src_dur = probe_duration(input_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if src_dur >= target_duration - 0.02:
        if os.path.abspath(input_path) != os.path.abspath(output_path):
            shutil.copy2(input_path, output_path)
        return output_path

    src_fps = probe_fps(input_path)
    # Stretch PTS so the same frames span the target duration.
    ratio = target_duration / src_dur
    # Keep original fps — ffmpeg duplicates frames to fill the stretched timeline.
    _run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", f"setpts={ratio:.6f}*PTS",
            "-r", f"{src_fps:.4f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-an",
            output_path,
            "-loglevel", "error",
        ],
        desc="extend video by duplicating frames",
    )
    return output_path


def prepare_video_edit_source_clip(
    source_clip_path: str,
    padded_output_path: str,
    *,
    min_duration: Optional[float] = None,
    pad_duration: Optional[float] = None,
) -> Tuple[str, bool]:
    """Return the video-edit input path; extend short clips by duplicating frames.

    For clips shorter than the minimum duration, frames are duplicated throughout
    the video to extend the duration while maintaining the original frame rate.
    No last-frame cloning or looping. After editing, ``conform_video_to_source``
    trims back to the original duration.

    Returns:
        (clip_path, was_padded)
    """
    min_dur = float(
        min_duration
        if min_duration is not None
        else os.environ.get("VIDEO_EDIT_MIN_DURATION", "3")
    )
    pad_dur = float(
        pad_duration
        if pad_duration is not None
        else os.environ.get("VIDEO_EDIT_PAD_DURATION", "4")
    )
    src_dur = probe_duration(source_clip_path)
    if src_dur >= min_dur - 1e-3:
        return source_clip_path, False

    target = max(pad_dur, min_dur + 0.5)
    logger.info(
        "Video-edit input %.2fs < %.2fs — extending to %.2fs (duplicate frames, keep fps)",
        src_dur,
        min_dur,
        target,
    )
    extend_video_by_duplicating_frames(source_clip_path, padded_output_path, target)
    return padded_output_path, True


def trim_video_to_duration(
    input_path: str,
    output_path: str,
    target_duration: float,
    *,
    fps: Optional[float] = None,
) -> str:
    """Trim video to *target_duration* seconds (video-only)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    rate = fps if fps is not None else probe_fps(input_path)
    _run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-t", f"{target_duration:.4f}",
            "-r", str(rate),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-an",
            output_path,
            "-loglevel", "error",
        ],
        desc="trim video to duration",
    )
    return output_path


def conform_video_to_source(
    input_path: str,
    source_path: str,
    output_path: str,
) -> str:
    """Match source duration, resolution, and fps; output video-only MP4."""
    target_duration = probe_duration(source_path)
    width, height = probe_video_size(source_path)
    fps = probe_fps(source_path)
    src_dur = probe_duration(input_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    base = [
        "ffmpeg", "-y",
    ]
    if src_dur < target_duration - 0.05:
        loops = int(target_duration / max(src_dur, 0.1)) + 1
        base += ["-stream_loop", str(loops)]
    base += [
        "-i", input_path,
        "-t", str(target_duration),
        "-vf", (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            "setsar=1"
        ),
        "-r", str(fps),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-an",
        output_path,
        "-loglevel", "error",
    ]
    _run(base, desc="conform video")
    return output_path


def mux_video_with_scene_audio(
    video_path: str,
    source_clip_path: str,
    output_path: str,
    *,
    audio_path: Optional[str] = None,
) -> str:
    """Mux generated video with scene audio (extracted AAC preferred)."""
    if audio_path and os.path.exists(audio_path):
        return mux_video_audio(video_path, audio_path, output_path)
    return replace_video_audio_from_source(video_path, source_clip_path, output_path)


def replace_video_audio_from_source(
    video_path: str,
    source_path: str,
    output_path: str,
) -> str:
    """Copy generated video and attach source audio when present."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not has_audio_stream(source_path):
        _run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-c:v", "copy",
                "-an",
                output_path,
                "-loglevel", "error",
            ],
            desc="copy video without audio",
        )
        return output_path

    _run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", source_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path,
            "-loglevel", "error",
        ],
        desc="replace source audio",
    )
    return output_path
