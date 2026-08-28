#!/usr/bin/env python3
"""
CLI entry point for the multi-shot video editing agent.

Example::

    source env.example.sh
    python -m video_editing_agent.run_agent \\
        --video /path/to/input.mp4 \\
        --prompt "Replace the man's red shirt with a blue one" \\
        --workspace ./output/run_001

    # Offline / no API:
    python -m video_editing_agent.run_agent --video in.mp4 --prompt "..." --dev-mode
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from video_editing_agent.agent import VideoEditingAgent
from video_editing_agent.config import AgentConfig
from video_editing_agent.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-shot real-scene long video editing agent",
    )
    parser.add_argument("--video", required=True, help="Input video absolute path")
    parser.add_argument("--prompt", required=True, help="Natural-language edit prompt")
    parser.add_argument("--workspace", default="./workspace", help="Output workspace")
    parser.add_argument("--scene-threshold", type=float, default=27.0)
    parser.add_argument(
        "--scene-mode",
        choices=("content", "adaptive", "hybrid"),
        default="hybrid",
        help="Scene detector strategy (hybrid = union of content + adaptive)",
    )
    parser.add_argument(
        "--scene-min-len",
        type=int,
        default=8,
        help="Min frames between scene cuts (default 8; PySceneDetect default is 15)",
    )
    parser.add_argument("--extract-fps", type=float, default=8.0, help="Frame extraction FPS")
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--dev-mode", action="store_true", help="Skip paid API calls")
    parser.add_argument(
        "--no-mask-visibility-prefilter",
        action="store_true",
        help="Query all instructions on every scene frame (no visibility pre-check)",
    )
    parser.add_argument(
        "--no-mask-first-detection-validate",
        action="store_true",
        default=False,
        help="Skip VLM identity check when first saving an entity to entity_refs (default: validate)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run all modules even when workspace result files already exist",
    )
    parser.add_argument(
        "--no-keyframe-edit-qa",
        action="store_true",
        help="Disable VLM quality check after Module-3 keyframe inpaint (default: on)",
    )
    parser.add_argument("--log-file", default=None)
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    """Run the agent pipeline."""
    video_path = os.path.abspath(args.video)
    if not os.path.exists(video_path):
        logger.error("Video not found: %s", video_path)
        return 1

    if args.no_keyframe_edit_qa:
        keyframe_edit_qa = False
    else:
        env_qa = (
            os.environ.get("KEYFRAME_EDIT_QA")
            or os.environ.get("KEYFRAME_EDIT_QA", "true")
        ).lower()
        keyframe_edit_qa = env_qa not in ("0", "false", "no")

    config = AgentConfig(
        workspace_dir=os.path.abspath(args.workspace),
        source_video_path=video_path,
        scene_detect_threshold=args.scene_threshold,
        scene_detect_mode=args.scene_mode,
        scene_detect_min_scene_len=args.scene_min_len,
        extract_fps=args.extract_fps,
        max_propagation_concurrency=args.max_concurrency,
        dev_mode=args.dev_mode,
        mask_visibility_prefilter=not args.no_mask_visibility_prefilter,
        mask_first_detection_validate=not args.no_mask_first_detection_validate,
        keyframe_edit_qa=keyframe_edit_qa,
        resume_from_checkpoints=not args.no_resume,
    )
    agent = VideoEditingAgent(config)

    try:
        final_path = await agent.run(args.prompt)
        print(f"\n✅ Final video: {final_path}")
        return 0
    except Exception as exc:
        logger.error("Agent failed: %s", exc)
        return 1


def main(argv: list[str] | None = None) -> int:
    """Synchronous CLI entry."""
    args = parse_args(argv)
    setup_logging(log_file=args.log_file)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
