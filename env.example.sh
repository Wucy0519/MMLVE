#!/usr/bin/env bash
# =============================================================================
#  MMLVE-Agent — API configuration
#  用法 / Usage:  source env.example.sh
#
#  Fill in the two API keys below before running the agent.
#  运行前请填写下面两个 API Key（留空则无法调用真实模型）。
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 1. LLM / Vision / Image  —  OpenAI-compatible chat/completions endpoint
#    用于：指令解析、镜头理解(VLM)、关键帧编辑与质检
#    Used for: instruction parsing, shot understanding (VLM), keyframe editing & QA
# ─────────────────────────────────────────────────────────────────────────────

# API key — REQUIRED. 必填。
export OPENAI_API_KEY=""

# Endpoint. Pick the one matching your provider.
# 端点地址，根据所用服务商选择其一：
#
#   OpenAI:
#     https://api.openai.com/v1/chat/completions
#   Google Gemini (OpenAI-compatible):
#     https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
#   Any OpenAI-compatible gateway / 任意兼容网关:
#     https://<your-host>/v1/chat/completions
export OPENAI_BASE_URL="https://api.openai.com/v1/chat/completions"

# Model IDs — must be available to your API key.
# 模型名称，需确保你的 Key 有权限访问。
export LLM_TEXT_MODEL="gpt-4o-mini"        # structured JSON / reasoning
export LLM_VISION_MODEL="gpt-4o-mini"      # image & frame understanding
export LLM_IMAGE_MODEL="gpt-image-1"       # image generation / inpainting

# Gemini example / Gemini 示例:
#   export LLM_TEXT_MODEL="gemini-2.5-pro"
#   export LLM_VISION_MODEL="gemini-2.5-pro"
#   export LLM_IMAGE_MODEL="gemini-2.5-flash-image"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Video editing  —  BytePlus ModelArk (Seedance)
#    用于：将关键帧编辑结果传播到整段视频 (V2V)
#    Used for: propagating keyframe edits across the video (V2V)
#    Console: https://console.byteplus.com/ark
# ─────────────────────────────────────────────────────────────────────────────

# API key — REQUIRED for real video generation. 生成真实视频时必填。
export BYTEPLUS_API_KEY=""

export BYTEPLUS_BASE_URL="https://ark.ap-southeast.bytepluses.com/api/v3"
export VIDEO_MODEL="seedance-1-5-pro-251215"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Optional tuning / 可选调优（保持默认即可 — defaults are fine）
# ─────────────────────────────────────────────────────────────────────────────

# ── Video output ──
# Leave VIDEO_RESOLUTION unset to auto-detect from the source video.
# 不设置 VIDEO_RESOLUTION 时会根据源视频自动选择 480p / 720p。
# export VIDEO_RESOLUTION="720p"
export VIDEO_RATIO="adaptive"
export VIDEO_GENERATE_AUDIO="false"

# Async video task polling / 异步视频任务轮询
export VIDEO_POLL_SEC=10
export VIDEO_POLL_TIMEOUT=1000

# Short clips are padded before editing, then trimmed back.
# 过短片段先补长再编辑，完成后裁回原时长。
export VIDEO_EDIT_MIN_DURATION=3
export VIDEO_EDIT_PAD_DURATION=4

# Max seconds per video-edit chunk / 单次视频编辑的最大时长
export VIDEO_CHUNK_MAX_SEC=10
export VIDEO_CHUNK_MIN_TAIL_SEC=4

# ── HTTP / request limits ──
export LLM_REQUEST_TIMEOUT=300
export LLM_MAX_VISION_IMAGES=8
export LLM_HTTP_RETRIES=3

# ── Quality checks (VLM self-correction) / 质检（VLM 自我校正）──
export KEYFRAME_EDIT_QA=true    # after keyframe editing  / 关键帧编辑后
export VIDEO_EDIT_QA=true       # after video propagation / 视频传播后

# ── Scene / shot detection (PySceneDetect) ──
# Lower threshold = more cuts detected. 阈值越低，检出的镜头切换越多。
export SCENE_DETECT_MODE="enhanced"    # content | adaptive | hybrid | enhanced
export SCENE_DETECT_THRESHOLD=22
export SCENE_MIN_SCENE_LEN=5

# ── Local acceleration (Apple Silicon) / 本地加速（Apple 芯片）──
export PYTORCH_ENABLE_MPS_FALLBACK=1
