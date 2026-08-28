"""
Module 5 — Video assembly via ffmpeg-python.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Dict, List, Tuple

from video_editing_agent.config import AgentConfig
from video_editing_agent.schemas.scenes import TimeInstructionSet
from video_editing_agent.utils.ffmpeg_utils import (
    concat_videos,
    extract_audio,
    has_audio_stream,
    mux_video_audio,
)

logger = logging.getLogger(__name__)


class VideoAssembler:
    """Video Assembly Agent (Module 5)."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def run(
        self,
        time_instru: TimeInstructionSet,
        edited_clips: Dict[str, str],
    ) -> str:
        """Assemble final video with original audio."""
        try:
            logger.info("VideoAssembler.run started")

            if not time_instru.scenes:
                raise RuntimeError("No scenes to assemble")

            clip_sequence = self._build_clip_sequence(time_instru, edited_clips)
            for sid, path in clip_sequence:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing clip for {sid}: {path}")

            concat_list = self._write_concat_demuxer_list(clip_sequence)
            video_only = os.path.join(self.config.workspace_dir, "concat_video_only.mp4")
            self._concatenate_video(concat_list, video_only)
            self._merge_original_audio(video_only)

            if os.path.exists(video_only):
                os.remove(video_only)

            logger.info("VideoAssembler.run done → %s", self.config.final_output_path)
            return self.config.final_output_path

        except Exception as exc:
            logger.error("VideoAssembler.run failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Video assembly failed: {exc}") from exc

    def _build_clip_sequence(
        self,
        time_instru: TimeInstructionSet,
        edited_clips: Dict[str, str],
    ) -> List[Tuple[str, str]]:
        """Ordered scene clips — edited or original."""
        scenes = sorted(time_instru.scenes, key=lambda s: s.start_sec)
        sequence: List[Tuple[str, str]] = []

        for scene in scenes:
            if scene.scene_id in edited_clips:
                sequence.append((scene.scene_id, edited_clips[scene.scene_id]))
            else:
                original = os.path.join(
                    self.config.scenes_dir,
                    scene.scene_id,
                    f"{scene.scene_id}.mp4",
                )
                sequence.append((scene.scene_id, original))

        return sequence

    def _write_concat_demuxer_list(
        self,
        clip_sequence: List[Tuple[str, str]],
    ) -> str:
        """Write ffmpeg concat demuxer list."""
        list_path = os.path.join(self.config.workspace_dir, "concat_list.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for _sid, clip_path in clip_sequence:
                escaped = os.path.abspath(clip_path).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
        logger.info("Concat list: %d clips", len(clip_sequence))
        return list_path

    def _concatenate_video(self, concat_list_path: str, output_path: str) -> None:
        """Concatenate via ffmpeg-python with subprocess fallback."""
        try:
            import ffmpeg
            (
                ffmpeg
                .input(concat_list_path, format="concat", safe=0)
                .output(output_path, vcodec="libx264", pix_fmt="yuv420p", an=None)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except Exception as exc:
            logger.warning("ffmpeg-python concat failed, using subprocess: %s", exc)
            concat_videos(concat_list_path, output_path, reencode=True)

    def _merge_original_audio(self, video_only_path: str) -> None:
        """Mux original source audio onto concatenated video."""
        if not has_audio_stream(self.config.source_video_path):
            logger.info("Source has no audio — using video-only output")
            shutil.copy2(video_only_path, self.config.final_output_path)
            return

        audio_path = os.path.join(self.config.workspace_dir, "source_audio.aac")
        try:
            extract_audio(self.config.source_video_path, audio_path)
            mux_video_audio(video_only_path, audio_path, self.config.final_output_path)
        except Exception as exc:
            logger.warning("Audio mux failed (%s) — copying video-only output", exc)
            shutil.copy2(video_only_path, self.config.final_output_path)

        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass
