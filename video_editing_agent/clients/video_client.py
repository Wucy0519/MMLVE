"""
BytePlus ARK (ModelArk) video-edit client — Seedance video-to-video / image-to-video.

Official API used by the video editing agent for scene-level V2V edits.

Auth: BYTEPLUS_API_KEY environment variable.
Docs: https://docs.byteplus.com/en/docs/ModelArk/Video_Generation_API

The pipeline submits an editing task with the source clip as a reference video
(and, optionally, an edited first-frame image), polls until the task succeeds,
then downloads the resulting MP4.

Environment variables:
    BYTEPLUS_API_KEY          API key (REQUIRED for real video generation).
    BYTEPLUS_BASE_URL         ARK base URL (default: the ap-southeast endpoint).
    VIDEO_MODEL               Seedance model id (default: seedance-1-5-pro-251215).
    VIDEO_RESOLUTION          Output resolution, e.g. 720p / 1080p (default: 720p).
    VIDEO_RATIO               Aspect ratio hint (default: adaptive).
    VIDEO_GENERATE_AUDIO      true|false (default: false).
    VIDEO_POLL_SEC            Poll interval in seconds (default: 10).
    VIDEO_POLL_TIMEOUT        Max seconds to wait for a task (default: 1000).
"""
from __future__ import annotations

import base64
import os
import time
from typing import Any, List, Optional

import requests

DEFAULT_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"
DEFAULT_MODEL = "seedance-1-5-pro-251215"

# Content-policy / moderation markers → do not retry, fall back to original clip.
_NON_RETRYABLE_MARKERS = (
    "inappropriate content",
    "content policy",
    "content_filter",
    "safety filter",
    "moderation",
    "sensitive",
)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip().strip('"').strip("'")


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        val = _env(name)
        if val:
            return val
    return default


def _api_key() -> str:
    key = _first_env("BYTEPLUS_API_KEY", "ARK_API_KEY")
    if not key:
        raise EnvironmentError(
            "BYTEPLUS_API_KEY is not set (required for video generation).\n"
            "  Get a key from BytePlus ModelArk / ARK console and set:\n"
            "    export BYTEPLUS_API_KEY=\"...\"\n"
            "  Docs: https://docs.byteplus.com/en/docs/ModelArk/Video_Generation_API"
        )
    return key


def _base_url() -> str:
    return _first_env("BYTEPLUS_BASE_URL", "ARK_BASE_URL", default=DEFAULT_BASE_URL).rstrip("/")


def _poll_interval() -> int:
    return int(_first_env("VIDEO_POLL_SEC", default="10"))


def _timeout() -> int:
    return int(_first_env("VIDEO_POLL_TIMEOUT", default="1000"))


def _http_timeout() -> int:
    return int(_first_env("VIDEO_HTTP_TIMEOUT", default="300"))


def _resolution(resolution: Optional[str] = None) -> str:
    res = (resolution or _first_env("VIDEO_RESOLUTION", default="720p")).strip().lower()
    if res.endswith("p"):
        return res
    return res + "p"


def _ratio(ratio: Optional[str] = None) -> str:
    return (ratio or _first_env("VIDEO_RATIO", default="adaptive")).strip()


def _generate_audio_default() -> bool:
    return _first_env("VIDEO_GENERATE_AUDIO", default="false").lower() in {
        "1", "true", "yes", "on",
    }


def default_video_model() -> str:
    return _first_env("VIDEO_MODEL", default=DEFAULT_MODEL)


def _is_seedance_model(model: str) -> bool:
    """Kept for backward compatibility — all supported models use the ARK flow."""
    return True


def snap_seedance_duration(duration: float) -> int:
    """Seedance duration: 4-15 seconds."""
    seconds = int(round(duration))
    return max(4, min(seconds, 15))


def _mime_for_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
    }.get(ext, "application/octet-stream")


def _image_to_data_url(image_path: str, max_kb: int = 800) -> str:
    """Encode an image as a base64 data URL, downscaling to stay under ``max_kb``."""
    from PIL import Image
    import io

    img = Image.open(image_path).convert("RGB")
    quality = 85
    buf = io.BytesIO()
    for _ in range(5):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if buf.tell() // 1024 <= max_kb:
            break
        w, h = img.size
        img = img.resize((max(1, w // 2), max(1, h // 2)), Image.LANCZOS)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _video_to_data_url(video_path: str) -> str:
    with open(video_path, "rb") as fh:
        raw = fh.read()
    mime = _mime_for_path(video_path)
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _configured_media_url(local_path: str) -> str:
    """Optional local-path/basename → public URL mapping (VIDEO_MEDIA_URL_MAP)."""
    import json

    mapping_json = _first_env("VIDEO_MEDIA_URL_MAP")
    if not mapping_json:
        return ""
    try:
        mapping = json.loads(mapping_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid VIDEO_MEDIA_URL_MAP JSON: {exc}") from exc
    if not isinstance(mapping, dict):
        raise RuntimeError("VIDEO_MEDIA_URL_MAP must be a JSON object")
    for key in (os.path.abspath(local_path), local_path, os.path.basename(local_path)):
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _video_reference_url(video_path: str) -> str:
    return _configured_media_url(video_path) or _video_to_data_url(video_path)


def _image_reference_url(image_path: str) -> str:
    return _configured_media_url(image_path) or _image_to_data_url(image_path)


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_api_key()}",
    }


def _create_task(model: str, content: List[dict]) -> str:
    """Create an ARK video generation task; returns the task id."""
    url = f"{_base_url()}/contents/generations/tasks"
    resp = requests.post(
        url,
        headers=_headers(),
        json={"model": model, "content": content},
        timeout=_http_timeout(),
    )
    if not resp.ok:
        raise RuntimeError(f"Create video task failed (HTTP {resp.status_code}): {resp.text[:500]}")
    data = resp.json()
    task_id = data.get("id") or (data.get("data") or {}).get("id")
    if not task_id:
        raise RuntimeError(f"Create video task: no task id in response: {data}")
    return str(task_id)


def _get_task(task_id: str) -> dict:
    url = f"{_base_url()}/contents/generations/tasks/{task_id}"
    resp = requests.get(url, headers=_headers(), timeout=_http_timeout())
    if not resp.ok:
        raise RuntimeError(f"Query video task failed (HTTP {resp.status_code}): {resp.text[:500]}")
    return resp.json()


def _task_video_url(data: dict) -> str:
    content = data.get("content") or {}
    if isinstance(content, dict):
        url = content.get("video_url") or content.get("videoUrl")
        if url:
            return str(url)
    # Some gateways nest under data.
    inner = data.get("data") or {}
    if isinstance(inner, dict):
        content = inner.get("content") or {}
        if isinstance(content, dict) and content.get("video_url"):
            return str(content["video_url"])
    return ""


def _poll_task(task_id: str) -> str:
    """Poll until the task succeeds; return the resulting video URL."""
    deadline = time.time() + _timeout()
    poll = 0
    while time.time() < deadline:
        poll += 1
        if poll > 1:
            print(f"     Video polling... ({(poll - 1) * _poll_interval()}s)")
        data = _get_task(task_id)
        status = str(data.get("status") or (data.get("data") or {}).get("status") or "").lower()
        if status in ("succeeded", "success", "succeed"):
            url = _task_video_url(data)
            if url:
                return url
            raise RuntimeError("Video task succeeded but no video_url in response")
        if status in ("failed", "cancelled", "canceled"):
            err = data.get("error") or (data.get("data") or {}).get("error") or "failed"
            raise RuntimeError(f"Video task {status}: {err}")
        time.sleep(_poll_interval())
    raise TimeoutError(f"Video task {task_id} timed out after {_timeout()}s")


def _download(url: str, output_path: str) -> None:
    resp = requests.get(url, timeout=_http_timeout())
    resp.raise_for_status()
    if len(resp.content) <= 100:
        raise RuntimeError("Downloaded video empty")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(resp.content)


def _is_non_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _NON_RETRYABLE_MARKERS)


def _build_content(
    *,
    prompt: str,
    video_path: Optional[str],
    reference_image_path: Optional[str],
    reference_image_paths: Optional[List[str]],
    duration: int,
    resolution: str,
    ratio: str,
    generate_audio: bool,
) -> List[dict]:
    """Assemble the ARK ``content`` array for a video-edit / I2V task."""
    text = (
        f"{prompt} "
        f"--resolution {resolution} "
        f"--ratio {ratio} "
        f"--duration {duration} "
        f"--watermark false"
    )
    content: List[dict] = [{"type": "text", "text": text}]

    if reference_image_paths:
        for img in reference_image_paths:
            if img:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": _image_reference_url(img)},
                    "role": "reference_image",
                })
    elif reference_image_path:
        content.append({
            "type": "image_url",
            "image_url": {"url": _image_reference_url(reference_image_path)},
            "role": "first_frame",
        })

    if video_path:
        content.append({
            "type": "video_url",
            "video_url": {"url": _video_reference_url(video_path)},
            "role": "reference_video",
        })

    return content


def _run_edit(
    *,
    model: str,
    prompt: str,
    output_path: str,
    video_path: Optional[str] = None,
    reference_image_path: Optional[str] = None,
    reference_image_paths: Optional[List[str]] = None,
    duration: int = 5,
    resolution: str = "720p",
    ratio: str = "adaptive",
    generate_audio: bool = False,
) -> None:
    """Submit + poll + download a single ARK video-edit task; raises on failure."""
    content = _build_content(
        prompt=prompt,
        video_path=video_path,
        reference_image_path=reference_image_path,
        reference_image_paths=reference_image_paths,
        duration=duration,
        resolution=resolution,
        ratio=ratio,
        generate_audio=generate_audio,
    )
    task_id = _create_task(model, content)
    print(f"     Task submitted: {task_id}")
    url = _poll_task(task_id)
    _download(url, output_path)


def generate_seedance_video_edit_to_file(
    *,
    model: str,
    prompt: str,
    output_path: str,
    video_path: str,
    reference_image_path: Optional[str] = None,
    duration: Optional[float] = None,
    resolution: Optional[str] = None,
    ratio: Optional[str] = None,
    generate_audio: Optional[bool] = None,
) -> bool:
    """Submit + poll + download a Seedance video-edit task (source clip as reference).

    When ``reference_image_path`` is given it is used as the edited first frame.
    Returns True on success, False on failure.
    """
    res = _resolution(resolution)
    dur = snap_seedance_duration(duration if duration is not None else 5)
    kind = "first_frame (I2V)" if reference_image_path else "reference_video (V2V)"
    print(f"  🎬 Seedance {kind} [{model}] ({res}, {dur}s)...")
    try:
        _run_edit(
            model=model or default_video_model(),
            prompt=prompt,
            output_path=output_path,
            video_path=video_path,
            reference_image_path=reference_image_path,
            duration=dur,
            resolution=res,
            ratio=_ratio(ratio),
            generate_audio=generate_audio if generate_audio is not None else _generate_audio_default(),
        )
    except Exception as exc:
        print(f"  ❌ Seedance video-edit: {exc}")
        return False

    kb = os.path.getsize(output_path) // 1024
    print(f"  ✅ Seedance video-edit saved: {output_path} ({kb}KB)")
    return True


def generate_video_edit_to_file(
    *,
    model: str,
    prompt: str,
    output_path: str,
    video_path: str,
    reference_image_path: Optional[str] = None,
    duration_seconds: int = 5,
    aspect_ratio: str = "16:9",
    resolution: str = "720p",
) -> bool:
    """Generic single-clip video-edit entry (delegates to the Seedance flow)."""
    return generate_seedance_video_edit_to_file(
        model=model,
        prompt=prompt,
        output_path=output_path,
        video_path=video_path,
        reference_image_path=reference_image_path,
        duration=duration_seconds,
        resolution=resolution,
    )


def generate_video_edit_with_references(
    *,
    model: str,
    prompt: str,
    output_path: str,
    video_path: str,
    reference_image_path: Optional[str] = None,
    reference_image_paths: Optional[List[str]] = None,
    resolution: Optional[str] = None,
    audio_setting: Optional[str] = None,
) -> str:
    """Compatibility wrapper — routes multi-reference video edits to the ARK flow.

    Returns ``"success"`` | ``"content_rejected"`` | ``"failed"`` so existing
    call sites keep working after the switch to the official BytePlus API.
    """
    res = _resolution(resolution)
    num_refs = len(reference_image_paths) if reference_image_paths else (1 if reference_image_path else 0)
    kind = f"video-edit + {num_refs} reference_image(s)" if num_refs else "video-edit"
    effective_model = model or default_video_model()
    print(f"  🎬 Video-edit {kind} [{effective_model}] ({res})...")
    try:
        _run_edit(
            model=effective_model,
            prompt=prompt,
            output_path=output_path,
            video_path=video_path,
            reference_image_path=reference_image_path,
            reference_image_paths=reference_image_paths,
            duration=snap_seedance_duration(5),
            resolution=res,
            ratio=_ratio(None),
            generate_audio=_generate_audio_default(),
        )
    except Exception as exc:
        if _is_non_retryable(exc):
            print(f"  ⚠️ Video-edit content-policy rejection — using original video: {exc}")
            return "content_rejected"
        print(f"  ❌ Video-edit failed: {exc}")
        return "failed"

    kb = os.path.getsize(output_path) // 1024
    print(f"  ✅ Video-edit saved: {output_path} ({kb}KB)")
    return "success"
