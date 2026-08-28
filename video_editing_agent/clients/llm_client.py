"""
OpenAI-compatible LLM / VLM / image-generation client.

The pipeline talks to any OpenAI-compatible ``chat/completions`` endpoint for
text reasoning, image understanding (vision), and image generation. This works
with the official OpenAI API, Google Gemini's OpenAI-compatible endpoint, or any
third-party gateway that speaks the same protocol.

Environment variables:
  OPENAI_API_KEY       API key (Bearer token). REQUIRED.
  OPENAI_BASE_URL      Chat/completions endpoint, e.g.
                       https://api.openai.com/v1/chat/completions  or
                       https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
  LLM_TEXT_MODEL       Text / JSON reasoning model (default: gpt-4o-mini)
  LLM_VISION_MODEL     Image-understanding model (default: same as text model)
  LLM_IMAGE_MODEL      Image-generation model (default: gpt-image-1)
  LLM_REQUEST_TIMEOUT  HTTP timeout in seconds (default: 300)
  LLM_MAX_VISION_IMAGES  Max frames/images per vision request (default: 8)
"""
from __future__ import annotations

import base64
import io
import os
import subprocess
import tempfile
import time
from typing import Dict, List, Optional

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import RequestException, Timeout as RequestsTimeout
from PIL import Image

DEFAULT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_TEXT_MODEL = "gpt-4o-mini"
DEFAULT_IMAGE_MODEL = "gpt-image-1"
_PLACEHOLDER_TOKENS = frozenset(
    {"", "your_token_here", "your_api_key_here", "sk-xxxx", "xxxx", "xxx"}
)


class LlmApiError(RuntimeError):
    """OpenAI-compatible API request failed."""

    def __init__(self, status: int, body: str, auth_mode: str = ""):
        self.status = status
        self.body = body
        self.auth_mode = auth_mode
        super().__init__(f"LLM API error {status}: {body[:500]}")


def _first_env(*names: str, default: str = "") -> str:
    """Return the first non-empty environment variable among ``names``."""
    for name in names:
        val = os.environ.get(name, "").strip().strip('"').strip("'")
        if val:
            return val
    return default


def _raw_token() -> str:
    return _first_env("OPENAI_API_KEY", "LLM_API_KEY")


def validate_credentials() -> None:
    """Fail fast with a helpful message if the API key looks invalid."""
    raw = _raw_token()
    if raw.lower() in _PLACEHOLDER_TOKENS:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set or still uses a placeholder.\n"
            "  1. Obtain an API key from your model provider, e.g.\n"
            "       OpenAI:  https://platform.openai.com/api-keys\n"
            "       Gemini:  https://aistudio.google.com/apikey\n"
            "  2. Fill it into env.example.sh:\n"
            "       export OPENAI_API_KEY=\"sk-...\"\n"
            "       export OPENAI_BASE_URL=\"https://api.openai.com/v1/chat/completions\"\n"
            "  3. Run: source env.example.sh"
        )


def _token() -> str:
    val = _raw_token()
    if not val:
        raise EnvironmentError(
            "API key not set. Export OPENAI_API_KEY for the chat/completions endpoint."
        )
    return val


def _api_url() -> str:
    return _first_env("OPENAI_BASE_URL", "LLM_BASE_URL", default=DEFAULT_URL)


def _timeout() -> int:
    return int(_first_env("LLM_REQUEST_TIMEOUT", default="300"))


def text_model() -> str:
    return _first_env("LLM_TEXT_MODEL", default=DEFAULT_TEXT_MODEL)


def vision_model() -> str:
    return _first_env("LLM_VISION_MODEL", default=text_model())


def image_model() -> str:
    return _first_env("LLM_IMAGE_MODEL", default=DEFAULT_IMAGE_MODEL)


def _max_vision_images() -> int:
    return int(_first_env("LLM_MAX_VISION_IMAGES", default="8"))


def _auth_headers() -> Dict[str, str]:
    """Standard OpenAI-compatible bearer auth header."""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_token()}",
    }


def _pil_to_data_url(img: Image.Image, fmt: str = "JPEG", max_side: int = 1280) -> str:
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _file_to_data_url(path: str, max_side: int = 1280) -> str:
    return _pil_to_data_url(Image.open(path), max_side=max_side)


def _build_content(prompt: str, images: Optional[List[Image.Image]] = None) -> list:
    parts: list = [{"type": "text", "text": prompt}]
    for img in (images or [])[: _max_vision_images()]:
        parts.append({
            "type": "image_url",
            "image_url": {"url": _pil_to_data_url(img)},
        })
    return parts


_RETRYABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _post_json(
    endpoint: str,
    payload: dict,
    headers: Dict[str, str],
) -> requests.Response:
    max_retries = int(_first_env("LLM_HTTP_RETRIES", default="3"))
    backoff = float(
        _first_env("LLM_HTTP_RETRY_BACKOFF", default="5")
    )

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=_timeout(),
            )
            if resp.status_code in _RETRYABLE_HTTP_STATUS and attempt < max_retries:
                wait = backoff * attempt
                print(
                    f"  ⚠️  LLM HTTP {resp.status_code} — "
                    f"retry {attempt}/{max_retries} in {wait:.0f}s"
                )
                time.sleep(wait)
                continue
            return resp
        except (RequestsConnectionError, RequestsTimeout) as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = backoff * attempt
                print(
                    f"  ⚠️  LLM connection error — "
                    f"retry {attempt}/{max_retries} in {wait:.0f}s: {exc}"
                )
                time.sleep(wait)
            else:
                raise

    if last_exc:
        raise last_exc
    raise RuntimeError("LLM request failed after retries")


def chat_completions(
    prompt: str,
    images: Optional[List[Image.Image]] = None,
    model: Optional[str] = None,
    max_tokens: int = 8192,
    url: Optional[str] = None,
) -> dict:
    payload = {
        "model": model or text_model(),
        "messages": [{"role": "user", "content": _build_content(prompt, images)}],
        "max_tokens": max_tokens,
    }
    endpoint = url or _api_url()
    headers = _auth_headers()

    resp = _post_json(endpoint, payload, headers)
    if resp.ok:
        try:
            return resp.json()
        except Exception as json_exc:
            body_preview = (resp.text or "")[:300]
            raise LlmApiError(
                resp.status_code,
                f"JSON decode failed (status {resp.status_code}): {json_exc}. "
                f"Body preview: {body_preview}",
            ) from json_exc

    if resp.status_code in (401, 403):
        _raise_auth_help(LlmApiError(resp.status_code, resp.text))
    raise LlmApiError(resp.status_code, resp.text)


def _raise_auth_help(err: LlmApiError) -> None:
    raw = _raw_token()
    masked = raw[:4] + "..." + raw[-4:] if len(raw) > 10 else "(too short)"
    raise LlmApiError(
        err.status,
        f"{err.body}\n\n"
        "API authentication failed. Check:\n"
        f"  • OPENAI_API_KEY loaded: {masked}\n"
        f"  • OPENAI_BASE_URL: {_api_url()}\n"
        "  • Confirm the key has access to the requested model\n"
        "  • For Gemini set OPENAI_BASE_URL to the OpenAI-compatible endpoint:\n"
        "    https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        err.auth_mode,
    ) from None


def _extract_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    texts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            texts.append(part.get("text", ""))
    return "".join(texts)


def _decode_data_url(url: str) -> bytes:
    if "," in url:
        url = url.split(",", 1)[1]
    return base64.b64decode(url)


def _extract_image(data: dict) -> Optional[Image.Image]:
    choices = data.get("choices") or []
    if not choices:
        return None
    msg = choices[0].get("message") or {}

    # OpenAI-style image generation may return images[] on the message.
    for images_key in ("images",):
        images = msg.get(images_key)
        if isinstance(images, list):
            for item in images:
                if isinstance(item, dict):
                    url = (
                        item.get("image_url", {}).get("url")
                        if isinstance(item.get("image_url"), dict)
                        else item.get("url")
                    )
                    b64 = item.get("b64_json")
                    if b64:
                        return Image.open(io.BytesIO(base64.b64decode(b64)))
                    if url:
                        return Image.open(io.BytesIO(_decode_data_url(url)))

    content = msg.get("content", "")
    if isinstance(content, str):
        return None

    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type", "")
        if ptype == "image_url":
            url = part.get("image_url", {}).get("url", "")
            if url:
                return Image.open(io.BytesIO(_decode_data_url(url)))
        # Gemini OpenAI-compatible responses may embed inline image data.
        if ptype in ("output_image", "image"):
            b64 = part.get("data") or part.get("b64_json")
            if b64:
                return Image.open(io.BytesIO(base64.b64decode(b64)))
    return None


def text_generate(prompt: str, model: Optional[str] = None) -> str:
    data = chat_completions(prompt, model=model or text_model())
    return _extract_text(data)


def vision_generate(
    prompt: str,
    images: List[Image.Image],
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> str:
    tokens = max_tokens if max_tokens is not None else int(
        _first_env("LLM_VISION_MAX_TOKENS", default="8192")
    )
    data = chat_completions(
        prompt, images=images, model=model or vision_model(), max_tokens=tokens,
    )
    return _extract_text(data)


def generate_image(
    prompt: str,
    reference_images: Optional[List[Image.Image]] = None,
    model: Optional[str] = None,
    save_path: Optional[str] = None,
    max_tokens: int = 4096,
) -> Optional[Image.Image]:
    try:
        data = chat_completions(
            prompt,
            images=reference_images,
            model=model or image_model(),
            max_tokens=max_tokens,
        )
    except (RequestException, LlmApiError) as exc:
        print(f"  ⚠️  Image generation request failed: {exc}")
        return None

    img = _extract_image(data)
    if img and save_path:
        img.save(save_path)
    return img


def extract_video_frames(
    video_path: str,
    max_frames: int = 12,
) -> List[Image.Image]:
    if not os.path.exists(video_path):
        return []

    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, check=True,
        )
        duration = float(result.stdout.strip())
    except Exception:
        duration = 60.0

    duration = max(duration, 0.5)
    cap = _max_vision_images()
    n = max(2, min(max_frames, cap, int(duration) + 1))
    times = [duration * i / (n - 1) for i in range(n)] if n > 1 else [0.0]

    frames: List[Image.Image] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, t in enumerate(times):
            out = os.path.join(tmp, f"frame_{i:03d}.jpg")
            cmd = [
                "ffmpeg", "-y", "-ss", str(t), "-i", video_path,
                "-frames:v", "1", "-q:v", "3", out,
                "-loglevel", "error",
            ]
            subprocess.run(cmd, capture_output=True)
            if os.path.exists(out):
                try:
                    frames.append(Image.open(out).copy())
                except Exception:
                    pass
    return frames


def analyze_video_frames(
    video_path: str,
    prompt: str,
    max_frames: int = 12,
) -> str:
    frames = extract_video_frames(video_path, max_frames=max_frames)
    if not frames:
        raise RuntimeError(f"Could not extract frames from {video_path}")

    frame_note = (
        f"\n\n[The attached images are {len(frames)} frames sampled evenly "
        f"from a video clip, in chronological order from first to last. "
        f"Analyze them as a continuous video segment.]"
    )
    analysis_tokens = int(_first_env("LLM_VIDEO_ANALYSIS_MAX_TOKENS", default="16384"))
    return vision_generate(prompt + frame_note, frames, max_tokens=analysis_tokens)


def upload_video_local(video_path: str) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    print(f"  Using local video (frame sampling): {video_path}")
    return f"local://{os.path.abspath(video_path)}"


def analyze_video_local(video_ref: str, prompt: str) -> str:
    if video_ref.startswith("local://"):
        path = video_ref[len("local://"):]
    else:
        path = video_ref
    max_frames = int(_first_env("LLM_VIDEO_MAX_FRAMES", default="12"))
    return analyze_video_frames(path, prompt, max_frames=max_frames)
