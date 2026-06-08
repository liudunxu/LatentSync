import argparse
import json
import logging
import math
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse

import cv2
import numpy as np
import requests
import torch

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TF", "0")

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is on path so latentsync imports work
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from omegaconf import OmegaConf

RESULT_ROOT = PROJECT_DIR / "results" / "api"
INPUT_ROOT = RESULT_ROOT / "inputs"
OUTPUT_ROOT = RESULT_ROOT / "outputs"

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for directory in (INPUT_ROOT, OUTPUT_ROOT):
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "6006"))
    ffmpeg_path: str = os.getenv("FFMPEG_PATH", "./ffmpeg-4.4-amd64-static/")
    gpu_id: int = int(os.getenv("LATENTSYNC_GPU_ID", "0"))
    unet_config_path: str = os.getenv("LATENTSYNC_UNET_CONFIG", "configs/unet/stage2_512.yaml")
    inference_ckpt_path: str = os.getenv("LATENTSYNC_INFERENCE_CKPT", "checkpoints/latentsync_unet.pt")
    # Inference defaults tuned for natural-looking output (less "AI mouth"
    # crispness / over-coercion). Lower guidance + more steps + DeepCache
    # give ~2x speedup over the no-cache setting with a small quality cost
    # (slightly softer detail on edges; mouth motion stays correct). Frontend
    # can still override per request via the *_override fields.
    guidance_scale: float = float(os.getenv("LATENTSYNC_GUIDANCE_SCALE", "1.5"))
    inference_steps: int = int(os.getenv("LATENTSYNC_INFERENCE_STEPS", "40"))
    seed: int = int(os.getenv("LATENTSYNC_SEED", "1247"))
    enable_deepcache: bool = os.getenv("LATENTSYNC_ENABLE_DEEPCACHE", "1").lower() in {"1", "true", "yes"}
    max_download_bytes: int = int(os.getenv("API_MAX_DOWNLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
    download_retries: int = int(os.getenv("API_DOWNLOAD_RETRIES", "2"))
    download_retry_backoff_seconds: float = float(os.getenv("API_DOWNLOAD_RETRY_BACKOFF_SECONDS", "1.0"))
    progress_enabled: bool = os.getenv("API_PROGRESS", "1").lower() not in {"0", "false", "no", "off"}
    # CodeFormer (https://github.com/sczhou/CodeFormer) postprocess applied
    # to aligned face crops after diffusion inpainting. Disabled by
    # default -- enable per request with codeformer_enabled. Set
    # ``LATENTSYNC_CODEFORMER_PRELOAD=1`` to load the model at server
    # startup; otherwise it loads lazily on the first request that
    # asks for it.
    codeformer_checkpoint_path: str = os.getenv(
        "LATENTSYNC_CODEFORMER_CKPT", "checkpoints/codeformer/codeformer.pth"
    )
    codeformer_preload: bool = os.getenv("LATENTSYNC_CODEFORMER_PRELOAD", "0").lower() in {"1", "true", "yes"}
    codeformer_batch_size: int = int(os.getenv("LATENTSYNC_CODEFORMER_BATCH_SIZE", "8"))
    codeformer_required: bool = os.getenv("LATENTSYNC_CODEFORMER_REQUIRED", "0").lower() in {"1", "true", "yes"}


settings = Settings()
logger = logging.getLogger("latentsync.api")


# ---------------------------------------------------------------------------
# Pydantic models – kept identical to MuseTalk api.py for compatibility
# ---------------------------------------------------------------------------

class LipSyncRequest(BaseModel):
    video_url: str = Field(..., description="Source video URL")
    avatar_url: Optional[str] = Field(None, description="Reference avatar image URL")
    audio_url: str = Field(..., description="Driving audio URL")
    similarity_threshold: float = Field(0.5, ge=0.0, le=1.0)
    identity_margin: float = Field(0.05, ge=0.0, le=1.0)
    identity_cluster_threshold: float = Field(0.78, ge=0.0, le=1.0)
    default_identity_min_coverage: float = Field(0.5, ge=0.0, le=1.0)
    require_face_embedding: bool = True
    allow_crop_embedding_fallback: bool = True
    crop_embedding_min_detection_score: float = Field(0.0, ge=0.0, le=1.0)
    temporal_tracking_weight: float = Field(0.08, ge=0.0, le=0.5)
    target_fill_max_gap_seconds: float = Field(0.6, ge=0.0, le=3.0)
    target_fill_window_seconds: float = Field(2.0, ge=0.1, le=10.0)
    target_fill_min_match_ratio: float = Field(0.40, ge=0.0, le=1.0)
    target_fill_max_center_shift: float = Field(0.8, ge=0.0, le=5.0)
    target_motion_gate_enabled: bool = True
    target_motion_max_center_shift: float = Field(0.30, ge=0.0, le=5.0)
    target_motion_max_scale_change: float = Field(0.25, ge=0.0, le=2.0)
    target_fast_motion_gate_enabled: bool = True
    target_fast_motion_max_center_shift_per_frame: float = Field(0.12, ge=0.0, le=2.0)
    target_fast_motion_max_scale_change_per_frame: float = Field(0.08, ge=0.0, le=2.0)
    target_fast_motion_min_run_frames: int = Field(5, ge=1, le=120)
    lipsync_continuity_max_gap_seconds: float = Field(0.35, ge=0.0, le=2.0)
    lipsync_continuity_max_center_shift: float = Field(0.35, ge=0.0, le=5.0)
    lipsync_continuity_max_scale_change: float = Field(0.35, ge=0.0, le=2.0)
    # Mouth-region pixel diff break: complementary to the embedding
    # similarity check. When the mouth region mean abs diff between
    # consecutive aligned face crops exceeds this fraction, treat the
    # next frame as a continuity break -- catches face switches the
    # embedding check misses (similar-looking people, side faces).
    # 0 disables. Default 0.10 is above same-person expression/pose
    # diff (~0.02-0.05) and below cross-person diff (~0.10-0.30).
    lipsync_mouth_diff_break_threshold: float = Field(
        0.10, ge=0.0, le=1.0,
        description="Mouth-region mean abs diff break threshold; 0 disables.",
    )
    target_bbox_smoothing_window: int = Field(3, ge=1, le=15)
    target_bbox_smoothing_max_center_shift: float = Field(0.35, ge=0.0, le=5.0)
    identity_scan_interval: int = Field(0, ge=0, le=300, description="0 means scan about 2 frames per second")
    identity_scan_max_frames: int = Field(0, ge=0, description="0 means scan all sampled identity frames")
    identity_scan_require_landmark_match: bool = False
    min_detection_score: float = Field(0.30, ge=0.0, le=1.0)
    require_landmark_match: bool = True
    min_landmark_points: int = Field(8, ge=1, le=68)
    min_landmark_overlap: float = Field(0.08, ge=0.0, le=1.0)
    lipsync_min_segment_frames: int = Field(5, ge=1, le=300)
    lipsync_min_face_area_ratio: float = Field(0.015, ge=0.0, le=1.0)
    bbox_shift: int = 0
    extra_margin: int = Field(10, ge=0, le=100)
    parsing_mode: str = "jaw"
    blend_upper_boundary_ratio: float = Field(0.58, ge=0.0, le=1.0)
    blend_mask_blur_ratio: float = Field(0.01, ge=0.0, le=0.2)
    color_match_strength: float = Field(0.60, ge=0.0, le=1.0)
    mouth_detail_strength: float = Field(0.65, ge=0.0, le=1.0)
    # Unsharp-mask amount applied to the generated mouth region. 0 = off,
    # 1 = strong sharpen. Default 0.30 (was 0.0): the inpainter's output
    # tends to be slightly soft -- the diffusion process encourages
    # plausible-but-not-sharp, so a small amount of unsharp in the mouth
    # region recovers the high-frequency detail (teeth, lip lines, mouth
    # corners) that the generated content is missing. 0.30 is in the
    # "mild" range documented in the unsharp-mask helper; values above
    # ~0.7 start to look crunchy. Frontend can still override per request.
    mouth_sharpen_strength: float = Field(0.30, ge=0.0, le=1.0)
    # Frame-to-frame mouth stabilization: was 0.08, raised to 0.15 so the
    # EMA carryover actually damps the high-frequency jitter the diffusion
    # output tends to have. 0 disables entirely.
    mouth_temporal_stabilization_strength: float = Field(0.15, ge=0.0, le=0.6)
    mouth_temporal_stabilization_max_delta: float = Field(0.12, ge=0.0, le=2.0)
    mouth_audio_adaptive_motion_enabled: bool = True
    # Adaptive motion: 0.75 floor means even on weak/silent audio we keep
    # 75% of the generated current-frame motion (was 0.65, which over-
    # smoothed soft speech and made lips look "frozen" between syllables).
    mouth_audio_motion_min_scale: float = Field(0.75, ge=0.0, le=2.0)
    mouth_audio_motion_max_scale: float = Field(1.20, ge=0.0, le=2.0)
    # Inpaint mask override. None = use the server-side default
    # (self.config.data.mask_image_path, usually latentsync/utils/mask.png).
    # Set to "latentsync/utils/mask5.png" to use the tight mouth-only mask,
    # which keeps identity/expression intact by leaving cheeks/chin untouched.
    # Path is resolved relative to the project root at call time.
    mask_image_path: Optional[str] = Field(
        None,
        description="Override the inpaint mask path. None = use server default (mask.png).",
    )
    # Postfilter catches generated frames where the mouth ROI is clearly
    # blurry or much softer than the original mouth ROI. This is intentionally
    # conservative: difficult frames fall back to the source video instead of
    # showing smeared lips.
    quality_gate_enabled: bool = False
    quality_min_laplacian: float = Field(0.04, ge=0.0, le=2000.0)
    quality_min_sharpness_ratio: float = Field(0.05, ge=0.0, le=1.0)
    quality_ref_min_laplacian: float = Field(
        1.00,
        ge=0.0,
        le=2000.0,
        description="Only apply generated/reference sharpness-ratio fallback when the source mouth ROI is at least this sharp.",
    )
    quality_max_fallback_ratio: float = Field(
        0.80,
        ge=0.0,
        le=1.0,
        description="Disable quality fallback for this run if it would skip more than this fraction of non-prefiltered frames.",
    )
    # Side-face / fast-turn prefilters. Frames exceeding either threshold fall
    # back to the original (no inpainting), which is the right call for blurry
    # side profiles and motion-blur turns. yaw_rate is in degrees/frame, not
    # per second (28°/frame at 25fps ≈ 700°/sec).
    yaw_skip_threshold: float = Field(30.0, ge=0.0, le=90.0)
    yaw_rate_skip_threshold: float = Field(28.0, ge=0.0, le=45.0)
    # Episode-level side-face filter: when N consecutive frames exceed
    # yaw_skip_threshold, also skip `pre_pad`/`post_pad` frames of
    # transition zone around the episode (frames whose yaw is between
    # yaw_warn_threshold and yaw_skip_threshold -- the face is clearly
    # turning and the affine alignment is unreliable there). Set
    # pre_pad/post_pad to 0 to disable the padding and only do the
    # per-frame yaw skip.
    side_face_episode_pre_pad: int = Field(3, ge=0, le=30)
    side_face_episode_post_pad: int = Field(3, ge=0, le=30)
    # Warn-band ratio: yaws above `yaw_skip_threshold * ratio` but below
    # `yaw_skip_threshold` are treated as transition frames / near-profile
    # runs. Default 0.75 = warn @ 22.5° for the default 30° threshold.
    yaw_warn_threshold_ratio: float = Field(0.75, ge=0.0, le=1.0)
    side_face_warn_min_run_frames: int = Field(
        0,
        ge=0,
        le=120,
        description="Skip sustained near-profile runs above the yaw warn threshold; 0 disables.",
    )
    # Per-request inference overrides. None = use server-side setting
    # (LATENTSYNC_GUIDANCE_SCALE / LATENTSYNC_INFERENCE_STEPS / LATENTSYNC_SEED
    # env vars, or their CLI flags). Frontend (~/Downloads/dub) sends these
    # via a 质量预设 group (fast/balanced/quality) + raw fields.
    guidance_scale_override: Optional[float] = Field(
        None, ge=0.0, le=10.0, description="Classifier-free guidance scale (mouth motion strength). None = use server default."
    )
    inference_steps_override: Optional[int] = Field(
        None, ge=1, le=100, description="DDIM inference steps. None = use server default."
    )
    seed_override: Optional[int] = Field(
        None, description="RNG seed (-1 for random). None = use server default."
    )
    # DeepCache is wired at pipeline load time (see settings.enable_deepcache);
    # changing it per-request would require reloading the model. We log a
    # warning when this differs from the server setting so the caller knows
    # their hint was ignored without surprise.
    enable_deepcache_override: Optional[bool] = Field(
        None, description="Hint for DeepCache enable. None = use server default. Honored only at server startup."
    )
    # Mouth-occlusion prefilter: skip frames where the mouth is covered by
    # a hand, microphone, phone, mask, etc. Score 0..1. Default 1.0 disables
    # this heuristic because the fixed-ROI dark-pixel check is too sensitive
    # on side/profile shots and can filter most frames.
    mouth_occlusion_skip_threshold: float = Field(1.0, ge=0.0, le=1.0)
    # Motion-blur input filter: skip frames whose aligned face is too
    # smeared to inpaint cleanly. Set to 0 to disable.
    motion_blur_skip_threshold: float = Field(0.08, ge=0.0, le=10.0)
    face_jump_center_threshold: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description="Skip frames whose landmark center jumps by more than this fraction of face size; 0 disables.",
    )
    face_jump_scale_threshold: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description="Skip frames whose landmark face scale changes abruptly by more than this fraction; 0 disables.",
    )
    left_cheek_width: int = Field(75, ge=1, le=240)
    right_cheek_width: int = Field(75, ge=1, le=240)
    batch_size: int = Field(8, ge=1, le=64)
    audio_padding_length_left: int = Field(2, ge=0, le=10)
    audio_padding_length_right: int = Field(2, ge=0, le=10)
    audio_sync_offset_seconds: float = Field(0.0, ge=-0.5, le=0.5)
    audio_feature_fps: float = Field(
        0.0,
        ge=0.0,
        le=120.0,
        description="0 follows source fps, otherwise use this fps for Whisper audio features",
    )
    max_audio_feature_fps: float = Field(
        25.0,
        ge=0.0,
        le=120.0,
        description="0 disables capping; high-fps videos default to 25fps audio features",
    )
    speech_gate_enabled: bool = False
    speech_gate_relative_db: float = Field(-42.0, ge=-80.0, le=0.0)
    speech_gate_min_rms: float = Field(0.00035, ge=0.0, le=1.0)
    speech_gate_window_seconds: float = Field(0.12, ge=0.02, le=0.5)
    speech_gate_pre_roll_seconds: float = Field(0.04, ge=0.0, le=1.0)
    speech_gate_post_roll_seconds: float = Field(0.12, ge=0.0, le=1.0)
    speech_gate_fill_gap_seconds: float = Field(0.16, ge=0.0, le=1.0)
    # --- CodeFormer face-restoration postprocess (added 2026-06) ---
    # When True and the CodeFormer checkpoint is available, the aligned
    # face crops produced by LatentSync are run through CodeFormer
    # before being pasted back to the full video. This sharpens the
    # synthesized mouth and recovers identity/edge detail, but adds
    # ~1s of GPU time per ~30s of video on a modern card. Off by
    # default so the existing API behavior is unchanged unless the
    # caller opts in. Setting codeformer_enabled=True when the
    # checkpoint is missing is logged and silently skipped (no error
    # to the caller) unless ``codeformer_required`` is also set.
    codeformer_enabled: bool = Field(
        False,
        description="Run CodeFormer face restoration on the aligned 512x512 face crops before paste-back.",
    )
    codeformer_fidelity_weight: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="CodeFormer fidelity weight. 0 = sharpest (most codebook-driven, identity drift), "
                    "1 = closest to input. Default 0.7 is tuned for lip-sync output: the upstream "
                    "0.5 over-reconstructs the inpainter's output and tends to overwrite the "
                    "lipsync with a 'more typical' face. 0.85+ is safest for identity preservation.",
    )
    codeformer_adain: bool = Field(
        True,
        description="Apply CodeFormer's adaptive instance norm so the restored face's color matches the input. "
                    "Turning it off can yield a more 'CodeFormer-style' face at the cost of color drift.",
    )
    codeformer_required: bool = Field(
        settings.codeformer_required,
        description="If True and codeformer_enabled=True, fail the request when the CodeFormer checkpoint is missing.",
    )


class FaceListRequest(BaseModel):
    video_url: str = Field(..., description="Source video URL")
    similarity_threshold: float = Field(0.78, ge=0.0, le=1.0)
    frame_sample_interval: int = Field(1, ge=0, le=300, description="0 means sample about 2 frames per second; 1 scans every frame")
    max_frames: int = Field(0, ge=0, description="0 means scan all sampled frames")
    min_face_area: int = Field(100, ge=1)
    min_detection_score: float = Field(0.35, ge=0.0, le=1.0)
    require_face_embedding: bool = False
    require_landmark_match: bool = False
    min_landmark_points: int = Field(8, ge=1, le=68)
    min_landmark_overlap: float = Field(0.08, ge=0.0, le=1.0)
    crop_padding: float = Field(0.8, ge=0.0, le=1.5)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="LatentSync lip-sync API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT)), name="outputs")


@app.exception_handler(HTTPException)
async def log_http_exception(request: Request, exc: HTTPException):
    stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.error(
        "HTTP error while handling %s %s: %s\n%s",
        request.method,
        request.url,
        exc.detail,
        stack,
    )
    response = await http_exception_handler(request, exc)
    if isinstance(response, JSONResponse):
        return JSONResponse(
            status_code=exc.status_code,
            content=jsonable_encoder({"detail": exc.detail, "traceback": stack}),
            headers=exc.headers,
        )
    return response


@app.exception_handler(RequestValidationError)
async def log_validation_exception(request: Request, exc: RequestValidationError):
    stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.error(
        "Validation error while handling %s %s: %s\n%s",
        request.method,
        request.url,
        exc.errors(),
        stack,
    )
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": exc.errors(), "traceback": stack}),
    )


@app.exception_handler(Exception)
async def log_unhandled_exception(request: Request, exc: Exception):
    stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.exception("Unhandled error while handling %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content=jsonable_encoder({
            "detail": str(exc),
            "error_type": type(exc).__name__,
            "traceback": stack,
        }),
    )


# ---------------------------------------------------------------------------
# Helpers (mirroring MuseTalk api.py)
# ---------------------------------------------------------------------------

def _check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def _ensure_ffmpeg() -> None:
    if _check_ffmpeg():
        return
    path_separator = ";" if os.name == "nt" else ":"
    os.environ["PATH"] = f"{settings.ffmpeg_path}{path_separator}{os.environ.get('PATH', '')}"
    if not _check_ffmpeg():
        raise RuntimeError("ffmpeg was not found. Install ffmpeg or set FFMPEG_PATH.")


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail=f"Invalid URL: {url}")


class _RetryableDownloadError(Exception):
    pass


def _download_attempt_count() -> int:
    return max(1, settings.download_retries + 1)


def _is_retryable_download_status(status_code: int) -> bool:
    return status_code in {408, 425, 429} or 500 <= status_code < 600


def _is_retryable_download_error(exc: Exception) -> bool:
    if isinstance(exc, _RetryableDownloadError):
        return True
    if isinstance(exc, requests.HTTPError):
        response = getattr(exc, "response", None)
        status_code = response.status_code if response is not None else None
        return status_code is None or _is_retryable_download_status(status_code)
    return isinstance(exc, requests.RequestException)


def _download_retry_delay(attempt_index: int) -> float:
    return max(0.0, settings.download_retry_backoff_seconds) * (2 ** attempt_index)


def _get_download_response_once(url: str) -> requests.Response:
    response = requests.get(url, stream=True, timeout=(10, 120))
    if _is_retryable_download_status(response.status_code):
        status_code = response.status_code
        response.close()
        raise _RetryableDownloadError(f"HTTP {status_code}")
    response.raise_for_status()
    return response


def _get_download_response(url: str, label: str) -> requests.Response:
    attempts = _download_attempt_count()
    last_error = None
    attempts_made = 0
    for attempt_index in range(attempts):
        attempts_made = attempt_index + 1
        try:
            return _get_download_response_once(url)
        except _RetryableDownloadError as exc:
            last_error = exc
        except requests.RequestException as exc:
            last_error = exc

        if not _is_retryable_download_error(last_error):
            break
        if attempt_index >= attempts - 1:
            break
        logger.warning(
            "Failed to download %s on attempt %s/%s: %s; retrying",
            label,
            attempt_index + 1,
            attempts,
            last_error,
        )
        time.sleep(_download_retry_delay(attempt_index))

    raise HTTPException(
        status_code=400,
        detail=f"Failed to download {label} after {attempts_made} attempts: {last_error}",
    )


def _guess_suffix(url: str, content_type: str, allowed: set, fallback: str) -> str:
    suffix = Path(unquote(urlparse(url).path)).suffix.lower()
    if suffix in allowed:
        return suffix
    guessed = mimetypes.guess_extension((content_type or "").split(";")[0].strip())
    if guessed and guessed.lower() in allowed:
        return guessed.lower()
    return fallback


def _local_output_from_url(url: str) -> Optional[Path]:
    parsed = urlparse(url)
    path = parsed.path
    if path == "/api/download":
        query_url = parse_qs(parsed.query).get("url", [""])[0]
        if query_url:
            return _local_output_from_url(query_url)

    if not path.startswith("/outputs/"):
        return None
    relative = unquote(path[len("/outputs/"):]).lstrip("/")
    candidate = (OUTPUT_ROOT / relative).resolve()
    try:
        candidate.relative_to(OUTPUT_ROOT.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _download_to_file(url: str, dest_dir: Path, prefix: str, allowed: set, fallback: str) -> Path:
    local_path = _local_output_from_url(url)
    if local_path is not None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        suffix = local_path.suffix.lower()
        if suffix not in allowed:
            suffix = fallback
        output_path = dest_dir / f"{prefix}{suffix}"
        shutil.copyfile(local_path, output_path)
        return output_path

    _validate_url(url)
    dest_dir.mkdir(parents=True, exist_ok=True)
    attempts = _download_attempt_count()
    last_error = None
    attempts_made = 0
    for attempt_index in range(attempts):
        attempts_made = attempt_index + 1
        response = None
        temp_path = None
        try:
            response = _get_download_response_once(url)
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > settings.max_download_bytes:
                raise HTTPException(status_code=413, detail=f"{prefix} is larger than API_MAX_DOWNLOAD_BYTES")

            suffix = _guess_suffix(url, response.headers.get("content-type", ""), allowed, fallback)
            output_path = dest_dir / f"{prefix}{suffix}"
            temp_path = dest_dir / f"{prefix}{suffix}.part"
            downloaded = 0
            with temp_path.open("wb") as file_obj:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > settings.max_download_bytes:
                        raise HTTPException(status_code=413, detail=f"{prefix} is larger than API_MAX_DOWNLOAD_BYTES")
                    file_obj.write(chunk)
            temp_path.replace(output_path)
            return output_path
        except HTTPException:
            raise
        except _RetryableDownloadError as exc:
            last_error = exc
        except (requests.RequestException, OSError) as exc:
            last_error = exc
        finally:
            if response is not None:
                response.close()
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

        if not _is_retryable_download_error(last_error):
            break
        if attempt_index >= attempts - 1:
            break
        logger.warning(
            "Failed to download %s url=%s on attempt %s/%s: %s; retrying",
            prefix,
            url,
            attempt_index + 1,
            attempts,
            last_error,
        )
        time.sleep(_download_retry_delay(attempt_index))

    logger.error(
        "Giving up on %s download after %s attempts: url=%s err=%s",
        prefix,
        attempts_made,
        url,
        last_error,
    )
    raise HTTPException(
        status_code=400,
        detail=f"Failed to download {prefix} after {attempts_made} attempts: {last_error} (url={url})",
    )


def _output_url(request: Request, output_path: Path) -> str:
    relative = output_path.relative_to(OUTPUT_ROOT).as_posix()
    scheme = "https"
    host = request.headers.get("host", str(request.base_url).split("/")[2] if "://" in str(request.base_url) else f"localhost:8443")
    if ":" not in host:
        host = f"{host}:8443"
    return f"{scheme}://{host}/outputs/{relative}"


def _read_video_info(video_path: Path) -> Tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 1:
        fps = 25.0
    cap.release()
    return frame_count, fps


def _read_video_frames(video_path: Path) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 1:
        fps = 25.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frames, fps


def _clip_box(bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, ...]) -> Optional[Tuple[int, int, int, int]]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return x1, y1, x2, y2


def _box_area(bbox: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _box_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _box_iou(left: Tuple[int, int, int, int], right: Tuple[int, int, int, int]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    inter_x1 = max(lx1, rx1)
    inter_y1 = max(ly1, ry1)
    inter_x2 = min(lx2, rx2)
    inter_y2 = min(ly2, ry2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0
    union_area = _box_area(left) + _box_area(right) - inter_area
    return inter_area / union_area if union_area else 0.0


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


LMK_ADAPT_ORIGIN_ORDER = [
    1, 10, 12, 14, 16, 3, 5, 7, 0, 23, 21, 19,
    32, 30, 28, 26, 17, 43, 48, 49, 51, 50, 102, 103,
    104, 105, 101, 73, 74, 86,
]


# ---------------------------------------------------------------------------
# Runtime (loads model once and reuses across requests)
# ---------------------------------------------------------------------------

class LatentSyncApiRuntime:
    def __init__(self) -> None:
        self.loaded = False
        self.detectors_loaded = False
        self.load_lock = threading.RLock()
        self.run_lock = threading.Lock()
        self.config = None
        self.pipeline = None
        self.dtype = None
        self.face_embedder = None
        self.face_embedding_error = ""
        # CodeFormer restorer is built lazily on first use to avoid
        # ~1 GB of GPU memory when the feature is never requested.
        self.codeformer_restorer = None
        self.codeformer_load_attempted = False
        self.codeformer_load_error = ""

    def load_detectors(self) -> None:
        if self.detectors_loaded:
            return
        with self.load_lock:
            if self.detectors_loaded:
                return
            try:
                from insightface.app import FaceAnalysis

                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if torch.cuda.is_available()
                    else ["CPUExecutionProvider"]
                )
                root_path = str(PROJECT_DIR / "checkpoints" / "auxiliary")
                self.face_embedder = FaceAnalysis(
                    allowed_modules=["detection", "landmark_2d_106", "recognition"],
                    root=root_path,
                    providers=providers,
                )
                ctx_id = settings.gpu_id if torch.cuda.is_available() else -1
                self.face_embedder.prepare(ctx_id=ctx_id, det_size=(512, 512))
            except Exception as exc:
                self.face_embedding_error = str(exc)
                logger.warning("Failed to load InsightFace detector: %s", exc)
                self.face_embedder = None
            self.detectors_loaded = True

    def load(self) -> None:
        with self.load_lock:
            if self.loaded:
                return
            self.load_detectors()
            if self.loaded:
                return
            _ensure_ffmpeg()

            config_path = Path(settings.unet_config_path)
            if not config_path.is_absolute():
                config_path = PROJECT_DIR / config_path
            if not config_path.is_file():
                raise FileNotFoundError(f"UNet config not found: {config_path}")

            ckpt_path = Path(settings.inference_ckpt_path)
            if not ckpt_path.is_absolute():
                ckpt_path = PROJECT_DIR / ckpt_path

            self.config = OmegaConf.load(str(config_path))

            is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
            self.dtype = torch.float16 if is_fp16_supported else torch.float32

            from diffusers import AutoencoderKL, DDIMScheduler
            from latentsync.models.unet import UNet3DConditionModel
            from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
            from latentsync.whisper.audio2feature import Audio2Feature

            scheduler = DDIMScheduler.from_pretrained("configs")

            if self.config.model.cross_attention_dim == 768:
                whisper_model_path = "checkpoints/whisper/small.pt"
            elif self.config.model.cross_attention_dim == 384:
                whisper_model_path = "checkpoints/whisper/tiny.pt"
            else:
                raise NotImplementedError("cross_attention_dim must be 768 or 384")

            audio_encoder = Audio2Feature(
                model_path=whisper_model_path,
                device="cuda",
                num_frames=self.config.data.num_frames,
                audio_feat_length=self.config.data.audio_feat_length,
            )

            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=self.dtype)
            vae.config.scaling_factor = 0.18215
            vae.config.shift_factor = 0

            unet, _ = UNet3DConditionModel.from_pretrained(
                OmegaConf.to_container(self.config.model),
                str(ckpt_path),
                device="cpu",
            )
            unet = unet.to(dtype=self.dtype)

            self.pipeline = LipsyncPipeline(
                vae=vae,
                audio_encoder=audio_encoder,
                unet=unet,
                scheduler=scheduler,
            ).to("cuda")

            if settings.enable_deepcache:
                from DeepCache import DeepCacheSDHelper
                helper = DeepCacheSDHelper(pipe=self.pipeline)
                helper.set_params(cache_interval=3, cache_branch_id=0)
                helper.enable()

            if settings.codeformer_preload:
                restorer, err = self._get_codeformer_restorer()
                if restorer is not None and restorer.is_loaded:
                    logger.info(
                        "[LipSync] Preloaded CodeFormer restorer from %s (batch_size=%d)",
                        settings.codeformer_checkpoint_path,
                        settings.codeformer_batch_size,
                    )
                else:
                    logger.warning(
                        "[LipSync] codeformer_preload=True but model did not load: %s", err
                    )

            self.loaded = True

    def _get_codeformer_restorer(self) -> Tuple[Optional[object], str]:
        """Return the singleton :class:`CodeFormerRestorer`, building it
        on first call. Returns ``(None, reason)`` when construction
        fails so the caller can decide whether to fail the request or
        log and continue.
        """
        if self.codeformer_restorer is not None:
            return self.codeformer_restorer, ""
        if self.codeformer_load_attempted and self.codeformer_load_error:
            if (
                self.codeformer_load_error.startswith("CodeFormer checkpoint not found")
                and settings.codeformer_checkpoint_path
                and os.path.isfile(settings.codeformer_checkpoint_path)
            ):
                logger.info(
                    "[LipSync] CodeFormer checkpoint is now present; retrying load from %s",
                    settings.codeformer_checkpoint_path,
                )
                self.codeformer_load_error = ""
            else:
                return None, self.codeformer_load_error
        self.codeformer_load_attempted = True
        try:
            from latentsync.utils.codeformer_restorer import CodeFormerRestorer
        except Exception as exc:  # noqa: BLE001 -- the import is heavy
            self.codeformer_load_error = (
                f"failed to import CodeFormerRestorer: {type(exc).__name__}: {exc}"
            )
            logger.error("[LipSync] %s", self.codeformer_load_error)
            return None, self.codeformer_load_error
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.codeformer_restorer = CodeFormerRestorer(
            checkpoint_path=settings.codeformer_checkpoint_path,
            device=device,
            batch_size=settings.codeformer_batch_size,
        )
        # Eagerly probe the load so we surface "checkpoint missing" at
        # the first request rather than after the inpainter has burned
        # seconds of GPU time. The probe is cached in the restorer.
        if not self.codeformer_restorer.is_loaded:
            # _ensure_loaded is private; calling it is the only way to
            # populate _load_error before the user actually wants the
            # model. We tolerate the AttributeError on a refactor.
            try:
                self.codeformer_restorer._ensure_loaded()  # noqa: SLF001
            except Exception as exc:  # noqa: BLE001
                self.codeformer_load_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                logger.error("[LipSync] CodeFormer preload failed: %s", self.codeformer_load_error)
                self.codeformer_restorer = None
                return None, self.codeformer_load_error
            if not self.codeformer_restorer.is_loaded:
                self.codeformer_load_error = self.codeformer_restorer.load_error or "unknown"
                logger.error("[LipSync] CodeFormer preload failed: %s", self.codeformer_load_error)
                self.codeformer_restorer = None
                return None, self.codeformer_load_error
        return self.codeformer_restorer, ""

    def _detect_faces(self, frame: np.ndarray) -> List[Dict[str, object]]:
        if self.face_embedder is None:
            return []
        f_h, f_w, _ = frame.shape
        try:
            faces = self.face_embedder.get(frame)
        except Exception:
            return []
        results = []
        for face in faces:
            bbox = getattr(face, "bbox", None)
            if bbox is None:
                continue
            det_score = float(getattr(face, "det_score", 0.5))
            raw_bbox = np.array(bbox).astype(np.int_)
            w, h = raw_bbox[2] - raw_bbox[0], raw_bbox[3] - raw_bbox[1]
            if w < 16 or h < 16:
                continue
            if w / h > 4.0 or w / h < 0.1:
                continue
            if det_score < 0.25:
                continue
            lmk = getattr(face, "landmark_2d_106", None)
            if lmk is None:
                x1, y1, x2, y2 = raw_bbox.tolist()
            else:
                lmk = np.round(lmk).astype(np.int_)
                halk_face_coord = np.mean([lmk[74], lmk[73]], axis=0)
                sub_lmk = lmk[LMK_ADAPT_ORIGIN_ORDER]
                halk_face_dist = np.max(sub_lmk[:, 1]) - halk_face_coord[1]
                upper_bond = halk_face_coord[1] - halk_face_dist
                x1, y1, x2, y2 = (np.min(sub_lmk[:, 0]), int(upper_bond), np.max(sub_lmk[:, 0]), np.max(sub_lmk[:, 1]))
                if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
                    x1, y1, x2, y2 = raw_bbox.tolist()
                y2 += int((x2 - x1) * 0.1)
                x1 -= int((x2 - x1) * 0.05)
                x2 += int((x2 - x1) * 0.05)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(f_w, x2)
            y2 = min(f_h, y2)
            clipped = (x1, y1, x2, y2)
            if clipped[2] - clipped[0] < 8 or clipped[3] - clipped[1] < 8:
                continue
            embedding = getattr(face, "normed_embedding", None)
            if embedding is not None:
                embedding = _normalize_vector(np.asarray(embedding, dtype=np.float32))
            results.append({
                "bbox": clipped,
                "detection_score": det_score,
                "embedding": embedding,
            })
        return results

    @staticmethod
    def _crop_face(frame: np.ndarray, bbox: Tuple[int, int, int, int], padding: float = 0.25) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(frame.shape[1], x2 + pad_x)
        cy2 = min(frame.shape[0], y2 + pad_y)
        crop = frame[cy1:cy2, cx1:cx2]
        return crop if crop.size > 0 else None

    @staticmethod
    def _bbox_similarity(left: Tuple[int, int, int, int], right: Tuple[int, int, int, int]) -> float:
        iou = _box_iou(left, right)
        lc = _box_center(left)
        rc = _box_center(right)
        dist = math.hypot(lc[0] - rc[0], lc[1] - rc[1])
        scale = math.sqrt(max(1, _box_area(left)))
        norm_dist = max(0.0, 1.0 - dist / max(scale, 1.0))
        return 0.5 * iou + 0.5 * norm_dist

    def extract_distinct_faces(
        self,
        video_path: Path,
        output_dir: Path,
        payload: FaceListRequest,
    ) -> Dict[str, object]:
        self.load_detectors()
        with self.run_lock:
            frames, fps = _read_video_frames(video_path)
            sample_interval = payload.frame_sample_interval or max(1, int(round(fps / 2.0)))
            scanned_frames = 0
            detections = 0
            rejected_low_score = 0
            rejected_shape = 0
            rejected_landmarks = 0
            rejected_embedding = 0
            rejected_avatar_crop = 0

            face_infos: List[Dict[str, object]] = []
            frame_indices = range(0, len(frames), sample_interval)
            total_scan_frames = len(frame_indices)
            if payload.max_frames:
                total_scan_frames = min(total_scan_frames, payload.max_frames)

            for frame_index in frame_indices:
                if payload.max_frames and scanned_frames >= payload.max_frames:
                    break
                frame = frames[frame_index]
                scanned_frames += 1
                for face in self._detect_faces(frame):
                    if face["detection_score"] < payload.min_detection_score:
                        rejected_low_score += 1
                        continue
                    area = _box_area(face["bbox"])
                    if area < payload.min_face_area:
                        rejected_shape += 1
                        continue
                    aspect = (face["bbox"][2] - face["bbox"][0]) / max(1, face["bbox"][3] - face["bbox"][1])
                    if aspect < 0.3 or aspect > 2.5:
                        rejected_shape += 1
                        continue
                    if payload.require_face_embedding and face["embedding"] is None:
                        rejected_embedding += 1
                        continue
                    detections += 1
                    face["frame_index"] = frame_index
                    face_infos.append(face)

            # Cluster faces by embedding (fallback to bbox similarity)
            clusters: List[Dict[str, object]] = []
            threshold = payload.similarity_threshold
            for info in face_infos:
                embedding = info.get("embedding")
                best_cluster = None
                best_score = -1.0
                for i, cluster in enumerate(clusters):
                    cluster_embedding = cluster.get("embedding")
                    if embedding is not None and cluster_embedding is not None:
                        score = float(np.dot(embedding, cluster_embedding))
                    else:
                        score = self._bbox_similarity(info["bbox"], cluster["best_bbox"])
                    if score > best_score:
                        best_score = score
                        best_cluster = i

                if best_cluster is not None and best_score >= threshold:
                    c = clusters[best_cluster]
                    c["count"] = int(c["count"]) + 1
                    c["descriptors"].append(info)
                    area = _box_area(info["bbox"])
                    if area > int(c["max_area"]):
                        c["max_area"] = area
                        c["best_frame_index"] = info["frame_index"]
                        c["best_bbox"] = info["bbox"]
                        c["best_detection_score"] = info["detection_score"]
                else:
                    clusters.append({
                        "embedding": embedding,
                        "count": 1,
                        "max_area": _box_area(info["bbox"]),
                        "best_frame_index": info["frame_index"],
                        "best_bbox": info["bbox"],
                        "best_detection_score": info["detection_score"],
                        "descriptors": [info],
                    })

            clusters.sort(key=lambda c: (int(c["count"]), int(c["max_area"])), reverse=True)

            faces_dir = output_dir / "faces"
            faces_dir.mkdir(parents=True, exist_ok=True)
            face_paths = []
            face_items = []
            for cluster in clusters:
                frame = frames[int(cluster["best_frame_index"])]
                crop = self._crop_face(frame, cluster["best_bbox"], payload.crop_padding)
                if crop is None:
                    rejected_avatar_crop += 1
                    continue
                index = len(face_paths)
                face_path = faces_dir / f"face_{index:03d}.jpg"
                cv2.imwrite(str(face_path), crop)
                face_paths.append(face_path)
                face_items.append({
                    "path": face_path,
                    "max_area": int(cluster["max_area"]),
                    "frame_index": int(cluster["best_frame_index"]),
                    "detection_score": float(cluster["best_detection_score"]),
                    "count": int(cluster["count"]),
                })

            return {
                "face_paths": face_paths,
                "faces": face_items,
                "source_frame_count": len(frames),
                "frame_sample_interval": sample_interval,
                "scanned_frame_count": scanned_frames,
                "detected_face_count": detections,
                "rejected_low_score_count": rejected_low_score,
                "rejected_shape_count": rejected_shape,
                "rejected_landmark_count": rejected_landmarks,
                "rejected_embedding_count": rejected_embedding,
                "rejected_avatar_crop_count": rejected_avatar_crop,
                "face_identity_backend": "embedding" if payload.require_face_embedding else "visual",
            }

    @torch.no_grad()
    def synthesize(self, payload: LipSyncRequest, paths: Dict[str, Path], job_output_dir: Path, reference_embedding=None) -> Dict[str, object]:
        self.load()
        with self.run_lock:
            from accelerate.utils import set_seed

            video_path = paths["video"]
            audio_path = paths["audio"]
            output_path = job_output_dir / "result.mp4"

            # Resolve per-request overrides for inference quality. None = use
            # the server-side default loaded from env vars at startup.
            effective_guidance_scale = (
                payload.guidance_scale_override
                if payload.guidance_scale_override is not None
                else settings.guidance_scale
            )
            effective_inference_steps = (
                payload.inference_steps_override
                if payload.inference_steps_override is not None
                else settings.inference_steps
            )
            effective_seed = (
                payload.seed_override
                if payload.seed_override is not None
                else settings.seed
            )
            if payload.enable_deepcache_override is not None and payload.enable_deepcache_override != settings.enable_deepcache:
                logger.warning(
                    f"[LipSync] enable_deepcache_override={payload.enable_deepcache_override} "
                    f"differs from server default settings.enable_deepcache={settings.enable_deepcache}; "
                    "DeepCache is wired at pipeline load time -- restart the server with the new env var to apply."
                )

            # CodeFormer opt-in. The restorer is built lazily so we don't
            # burn ~1 GB of GPU on every server boot; the first request
            # that asks for it pays the load cost and subsequent calls
            # share the singleton.
            codeformer_restorer = None
            codeformer_unavailable_reason = ""
            if payload.codeformer_enabled:
                codeformer_restorer, codeformer_unavailable_reason = self._get_codeformer_restorer()
                if codeformer_restorer is None:
                    msg = (
                        "codeformer_enabled=True but the CodeFormer model is not available: "
                        f"{codeformer_unavailable_reason}"
                    )
                    if payload.codeformer_required or settings.codeformer_required:
                        raise HTTPException(status_code=503, detail=msg)
                    logger.warning(f"[LipSync] {msg}; continuing without CodeFormer postprocess")

            if effective_seed != -1:
                set_seed(effective_seed)
            else:
                torch.seed()

            logger.info(f"[LipSync] Starting pipeline: video={video_path}, audio={audio_path}, "
                         f"guidance_scale={effective_guidance_scale}, steps={effective_inference_steps}, "
                         f"seed={effective_seed}, has_reference_embedding={reference_embedding is not None}, "
                         f"codeformer_enabled={payload.codeformer_enabled}, "
                         f"codeformer_loaded={codeformer_restorer is not None and codeformer_restorer.is_loaded}, "
                         f"codeformer_fidelity_weight={payload.codeformer_fidelity_weight}, "
                         f"codeformer_adain={payload.codeformer_adain}")
            self.pipeline(
                video_path=str(video_path),
                audio_path=str(audio_path),
                video_out_path=str(output_path),
                num_frames=self.config.data.num_frames,
                num_inference_steps=effective_inference_steps,
                guidance_scale=effective_guidance_scale,
                weight_dtype=self.dtype,
                width=self.config.data.resolution,
                height=self.config.data.resolution,
                mask_image_path=payload.mask_image_path or self.config.data.mask_image_path,
                temp_dir=str(job_output_dir / "temp"),
                reference_embedding=reference_embedding,
                face_embedder=runtime.face_embedder,
                identity_similarity_threshold=payload.similarity_threshold,
                quality_gate_enabled=payload.quality_gate_enabled,
                quality_min_laplacian=payload.quality_min_laplacian,
                quality_min_sharpness_ratio=payload.quality_min_sharpness_ratio,
                quality_ref_min_laplacian=payload.quality_ref_min_laplacian,
                quality_max_fallback_ratio=payload.quality_max_fallback_ratio,
                yaw_skip_threshold=payload.yaw_skip_threshold,
                yaw_rate_skip_threshold=payload.yaw_rate_skip_threshold,
                side_face_episode_pre_pad=payload.side_face_episode_pre_pad,
                side_face_episode_post_pad=payload.side_face_episode_post_pad,
                yaw_warn_threshold_ratio=payload.yaw_warn_threshold_ratio,
                side_face_warn_min_run_frames=payload.side_face_warn_min_run_frames,
                mouth_occlusion_skip_threshold=payload.mouth_occlusion_skip_threshold,
                motion_blur_skip_threshold=getattr(payload, "motion_blur_skip_threshold", 0.08),
                face_jump_center_threshold=payload.face_jump_center_threshold,
                face_jump_scale_threshold=payload.face_jump_scale_threshold,
                lipsync_continuity_max_center_shift=payload.lipsync_continuity_max_center_shift,
                lipsync_continuity_max_scale_change=payload.lipsync_continuity_max_scale_change,
                lipsync_mouth_diff_break_threshold=payload.lipsync_mouth_diff_break_threshold,
                silent_skip_enabled=payload.speech_gate_enabled,
                silent_rms_threshold=payload.speech_gate_min_rms,
                silent_min_run_frames=max(
                    1,
                    int(round(payload.speech_gate_fill_gap_seconds * float(self.config.data.video_fps))),
                ),
                silent_pad_frames=0,
                color_match_strength=payload.color_match_strength,
                mouth_detail_strength=payload.mouth_detail_strength,
                mouth_sharpen_strength=payload.mouth_sharpen_strength,
                mouth_temporal_stabilization_strength=payload.mouth_temporal_stabilization_strength,
                mouth_temporal_stabilization_max_delta=payload.mouth_temporal_stabilization_max_delta,
                mouth_audio_adaptive_motion_enabled=payload.mouth_audio_adaptive_motion_enabled,
                mouth_audio_motion_min_scale=payload.mouth_audio_motion_min_scale,
                mouth_audio_motion_max_scale=payload.mouth_audio_motion_max_scale,
                # CodeFormer postprocess. ``codeformer_enabled`` is honoured
                # only when ``codeformer_restorer`` actually loaded; when
                # it didn't (e.g. checkpoint missing and not required) the
                # pipeline logs a warning and skips. Either way, the
                # pipeline still runs -- the failure mode is "no
                # postprocess", not "broken output".
                codeformer_enabled=payload.codeformer_enabled,
                codeformer_fidelity_weight=payload.codeformer_fidelity_weight,
                codeformer_adain=payload.codeformer_adain,
                codeformer_restorer=codeformer_restorer,
            )
            logger.info(f"[LipSync] Pipeline completed, output={output_path}")

            try:
                source_frame_count, source_fps = _read_video_info(video_path)
            except Exception:
                source_frame_count, source_fps = 0, 25.0

            # Pull real stats from the pipeline (stashed by the pipeline __call__).
            run_stats = getattr(self.pipeline, "_last_run_stats", None) or {}
            quality_fallback_frames = int(run_stats.get("quality_fallback_frames", 0))
            yaw_skip_count = int(run_stats.get("yaw_skip_count", 0))
            yaw_rate_skip_count = int(run_stats.get("yaw_rate_skip_count", 0))
            mouth_occlusion_skip_count = int(run_stats.get("mouth_occlusion_skip_count", 0))
            motion_blur_skip_count = int(run_stats.get("motion_blur_skip_count", 0))
            face_jump_skip_count = int(run_stats.get("face_jump_skip_count", 0))
            side_face_episode_extra_skip_count = int(
                run_stats.get("side_face_episode_extra_skip_count", 0)
            )
            side_face_warn_run_skip_count = int(
                run_stats.get("side_face_warn_run_skip_count", 0)
            )
            effective_generated_frames = int(
                run_stats.get("effective_generated_frames", source_frame_count)
            )
            pre_skip_frames = int(run_stats.get("pre_skip_frames", 0))
            quality_skip_frames = int(run_stats.get("quality_skip_frames", 0))
            effective_skip_frames = int(run_stats.get("effective_skip_frames", 0))
            silent_skip_frames = int(run_stats.get("silent_skip_frames", 0))
            skipped_inference_batches = int(run_stats.get("skipped_inference_batches", 0))
            skipped_inference_frames = int(run_stats.get("skipped_inference_frames", 0))
            identity_similarity_stats = run_stats.get("identity_similarity") or {}
            identity_skip_count = int(run_stats.get("identity_skip_count", 0))
            codeformer_stats = run_stats.get("codeformer") or {}
            mouth_temporal_stats = run_stats.get("mouth_temporal") or {}

            return {
                "output_path": output_path,
                "source_frame_count": source_frame_count,
                "output_frame_count": source_frame_count,
                "audio_frame_count": 0,
                "source_fps": round(float(source_fps), 6),
                "audio_feature_fps": 0.0,
                "audio_sync_offset_frames": 0,
                "audio_sync_offset_output_frames": 0,
                "audio_sync_offset_seconds": payload.audio_sync_offset_seconds,
                "effective_guidance_scale": effective_guidance_scale,
                "effective_inference_steps": effective_inference_steps,
                "effective_seed": effective_seed,
                "matched_source_frames": source_frame_count,
                "filled_source_frames": 0,
                "filtered_motion_frames": 0,
                "filtered_fast_motion_frames": 0,
                "continuity_filled_source_frames": 0,
                "filtered_small_face_frames": 0,
                "filtered_short_segment_frames": 0,
                "smoothed_source_frames": 0,
                "matched_or_filled_source_frames": source_frame_count,
                "eligible_source_frames": source_frame_count,
                "generated_output_frames": source_frame_count,
                "quality_fallback_frames": quality_fallback_frames,
                "prefilter_skip_frames": pre_skip_frames,
                "quality_skip_frames": quality_skip_frames,
                "yaw_skip_count": yaw_skip_count,
                "yaw_rate_skip_count": yaw_rate_skip_count,
                "mouth_occlusion_skip_count": mouth_occlusion_skip_count,
                "motion_blur_skip_count": motion_blur_skip_count,
                "face_jump_skip_count": face_jump_skip_count,
                "side_face_episode_extra_skip_count": side_face_episode_extra_skip_count,
                "side_face_warn_run_skip_count": side_face_warn_run_skip_count,
                "silent_skip_frames": silent_skip_frames,
                "skipped_inference_batches": skipped_inference_batches,
                "skipped_inference_frames": skipped_inference_frames,
                "effective_generated_output_frames": effective_generated_frames,
                "skipped_output_frames": effective_skip_frames,
                "best_similarity": 0.0,
                "identity_skip_count": identity_skip_count,
                "identity_similarity_min": float(identity_similarity_stats.get("min", 0.0)),
                "identity_similarity_median": float(identity_similarity_stats.get("median", 0.0)),
                "identity_similarity_max": float(identity_similarity_stats.get("max", 0.0)),
                "identity_similarity_threshold": float(run_stats.get("identity_similarity_threshold", payload.similarity_threshold)),
                "target_identity_similarity": 0.0,
                "target_identity_count": 0,
                "target_identity_coverage": 0.0,
                "target_identity_source": "none",
                "face_identity_backend": "embedding" if payload.require_face_embedding else "visual",
                "speech_gate": {
                    "enabled": payload.speech_gate_enabled,
                    "active_frames": max(0, source_frame_count - silent_skip_frames),
                    "silent_frames": silent_skip_frames,
                },
                "mouth_temporal": {
                    "stabilization_strength": float(payload.mouth_temporal_stabilization_strength),
                    "stabilization_max_delta": float(payload.mouth_temporal_stabilization_max_delta),
                    "delta_min": float(mouth_temporal_stats.get("delta_min", 0.0)),
                    "delta_median": float(mouth_temporal_stats.get("delta_median", 0.0)),
                    "delta_max": float(mouth_temporal_stats.get("delta_max", 0.0)),
                    "delta_skip_frames": int(mouth_temporal_stats.get("delta_skip_frames", 0)),
                    "stabilized_frames": int(mouth_temporal_stats.get("stabilized_frames", 0)),
                    "audio_adaptive_motion_enabled": bool(payload.mouth_audio_adaptive_motion_enabled),
                    "audio_motion_min_scale": float(mouth_temporal_stats.get("audio_motion_min_scale", 1.0)),
                    "audio_motion_median_scale": float(mouth_temporal_stats.get("audio_motion_median_scale", 1.0)),
                    "audio_motion_max_scale": float(mouth_temporal_stats.get("audio_motion_max_scale", 1.0)),
                },
                "codeformer": {
                    "requested": bool(payload.codeformer_enabled),
                    "fidelity_weight": float(payload.codeformer_fidelity_weight),
                    "required": bool(payload.codeformer_required or settings.codeformer_required),
                    # Whether the runtime actually has a loadable model.
                    "runtime_available": self.codeformer_restorer is not None
                    and self.codeformer_restorer.is_loaded,
                    "runtime_load_error": self.codeformer_load_error,
                    "checkpoint_path": settings.codeformer_checkpoint_path,
                    # Per-run stats from the pipeline layer.
                    "frames_total": int(codeformer_stats.get("frames_total", 0)),
                    "frames_enhanced": int(codeformer_stats.get("frames_enhanced", 0)),
                    "frames_fallback": int(codeformer_stats.get("frames_fallback", 0)),
                    "frames_skipped_by_pipeline": int(
                        codeformer_stats.get("frames_skipped_by_pipeline", 0)
                    ),
                    "fidelity_weight": float(codeformer_stats.get("fidelity_weight", 0.0)),
                    "adain": bool(codeformer_stats.get("adain", True)),
                    "elapsed_seconds": float(codeformer_stats.get("elapsed_seconds", 0.0)),
                    "error": codeformer_stats.get("error", ""),
                },
                "quality_ok": True,
            }


runtime = LatentSyncApiRuntime()


# ---------------------------------------------------------------------------
# Endpoints (identical paths to MuseTalk api.py)
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, object]:
    codeformer_loaded = (
        runtime.codeformer_restorer is not None
        and getattr(runtime.codeformer_restorer, "is_loaded", False)
    )
    return {
        "status": "ok",
        "detectors_loaded": runtime.detectors_loaded,
        "model_loaded": runtime.loaded,
        "face_embedding_loaded": runtime.face_embedder is not None,
        "crop_face_embedding_loaded": False,
        "face_embedding_backend": "insightface",
        "face_embedding_error": runtime.face_embedding_error,
        "port": settings.port,
        "codeformer": {
            "checkpoint_path": settings.codeformer_checkpoint_path,
            "loaded": codeformer_loaded,
            "preload_requested": settings.codeformer_preload,
            "load_error": runtime.codeformer_load_error,
        },
    }


@app.post("/api/faces")
def list_distinct_faces(payload: FaceListRequest, request: Request) -> Dict[str, object]:
    logger.info(f"[/api/faces] Request: video_url={payload.video_url}, similarity_threshold={payload.similarity_threshold}, min_face_area={payload.min_face_area}")
    job_id = uuid.uuid4().hex
    job_input_dir = INPUT_ROOT / job_id
    job_output_dir = OUTPUT_ROOT / job_id
    job_input_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)

    video_path = _download_to_file(payload.video_url, job_input_dir, "video", VIDEO_SUFFIXES, ".mp4")

    try:
        result = runtime.extract_distinct_faces(video_path, job_output_dir, payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    face_paths = result.pop("face_paths")
    face_urls = [_output_url(request, face_path) for face_path in face_paths]
    faces = []
    for face_url, item in zip(face_urls, result.pop("faces")):
        faces.append({
            "url": face_url,
            "max_area": item["max_area"],
            "frame_index": item["frame_index"],
            "detection_score": item["detection_score"],
            "count": item["count"],
        })

    response = {
        "job_id": job_id,
        "face_urls": face_urls,
        "faces": faces,
        **result,
    }
    logger.info(f"[/api/faces] Raw response: {json.dumps(response, ensure_ascii=False)[:2000]}")
    return response


@app.post("/api/lipsync")
def create_lipsync(payload: LipSyncRequest, request: Request) -> Dict[str, object]:
    if payload.parsing_mode not in {"jaw", "raw"}:
        raise HTTPException(status_code=400, detail="parsing_mode must be 'jaw' or 'raw'")

    logger.info(f"[/api/lipsync] Request: video_url={payload.video_url}, audio_url={payload.audio_url}, avatar_url={payload.avatar_url}")
    job_id = uuid.uuid4().hex
    job_input_dir = INPUT_ROOT / job_id
    job_output_dir = OUTPUT_ROOT / job_id
    job_input_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)

    video_path = _download_to_file(payload.video_url, job_input_dir, "video", VIDEO_SUFFIXES, ".mp4")
    audio_path = _download_to_file(payload.audio_url, job_input_dir, "audio", AUDIO_SUFFIXES, ".wav")
    try:
        input_paths = {"video": video_path, "audio": audio_path}
        reference_embedding = None
        if payload.avatar_url:
            avatar_downloaded = _download_to_file(payload.avatar_url, job_input_dir, "avatar", IMAGE_SUFFIXES, ".jpg")
            runtime.load_detectors()
            if runtime.face_embedder is not None:
                import cv2
                avatar_frame = cv2.imread(str(avatar_downloaded))
                if avatar_frame is not None:
                    faces = runtime.face_embedder.get(avatar_frame)
                    if faces:
                        def _avatar_face_score(face) -> float:
                            bbox = getattr(face, "bbox", None)
                            area = 0.0
                            if bbox is not None:
                                x1, y1, x2, y2 = [float(v) for v in bbox]
                                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                            return area * max(0.0, float(getattr(face, "det_score", 1.0)))

                        selected_face = max(faces, key=_avatar_face_score)
                        emb = getattr(selected_face, "normed_embedding", None)
                        if emb is not None:
                            import numpy as np
                            reference_embedding = np.asarray(emb, dtype=np.float32)
                            selected_bbox = getattr(selected_face, "bbox", None)
                            selected_score = float(getattr(selected_face, "det_score", 0.0))
                            logger.info(
                                "[LipSync] Loaded reference face embedding, faces=%d, selected_score=%.3f, selected_bbox=%s, shape=%s",
                                len(faces),
                                selected_score,
                                selected_bbox,
                                reference_embedding.shape,
                            )
                        else:
                            logger.warning(f"[LipSync] No embedding found in avatar face")
                    else:
                        logger.warning(f"[LipSync] No face detected in avatar image")
                else:
                    logger.warning(f"[LipSync] Failed to read avatar image")
            else:
                logger.warning(f"[LipSync] Face embedder not available")
        else:
            logger.info(f"[LipSync] No avatar_url provided, skipping face matching")
        result = runtime.synthesize(payload, input_paths, job_output_dir, reference_embedding=reference_embedding)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    output_path = result.pop("output_path")
    video_url = _output_url(request, output_path)
    scheme = request.url.scheme if hasattr(request, 'url') and request.url else "https"
    host = request.headers.get("host", str(request.base_url).split("/")[2] if "://" in str(request.base_url) else f"localhost:8443")
    if ":" not in host:
        host = f"{host}:8443"
    download_url = f"{scheme}://{host}/api/download?url={quote(video_url, safe='')}"
    response = {
        "job_id": job_id,
        "video_url": video_url,
        "download_url": download_url,
        **result,
    }
    logger.info(f"[LipSync] Raw response: {json.dumps(response, ensure_ascii=False)[:2000]}")
    return response


@app.get("/api/download")
def download_by_url(url: str = Query(..., description="Generated or remote video URL")):
    logger.info(f"[Download] url={url}")
    local_path = _local_output_from_url(url)
    if local_path is not None:
        file_size = local_path.stat().st_size
        logger.info(f"[Download] serving local file: {local_path}, size={file_size}")
        return FileResponse(
            str(local_path),
            filename=local_path.name,
            media_type="video/mp4",
            headers={
                "Content-Length": str(file_size),
                "Content-Disposition": f'attachment; filename="{local_path.name}"',
            },
        )

    _validate_url(url)
    try:
        response = _get_download_response(url, "URL")
    except HTTPException:
        raise

    content_length = response.headers.get("content-length")
    content_type = response.headers.get("content-type", "video/mp4")
    filename = Path(unquote(urlparse(url).path)).name or "video.mp4"
    if not filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        filename = filename + ".mp4"

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": content_type,
    }
    if content_length:
        headers["Content-Length"] = content_length

    logger.info(f"[Download] streaming from remote: filename={filename}, size={content_length or 'unknown'}")

    def iterator():
        try:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    yield chunk
        finally:
            response.close()

    return StreamingResponse(iterator(), headers=headers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LatentSync HTTP API")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--ffmpeg_path", default=settings.ffmpeg_path)
    parser.add_argument("--gpu_id", type=int, default=settings.gpu_id)
    parser.add_argument("--unet_config_path", default=settings.unet_config_path)
    parser.add_argument("--inference_ckpt_path", default=settings.inference_ckpt_path)
    parser.add_argument("--guidance_scale", type=float, default=settings.guidance_scale)
    parser.add_argument("--inference_steps", type=int, default=settings.inference_steps)
    parser.add_argument("--seed", type=int, default=settings.seed)
    parser.add_argument("--enable_deepcache", action="store_true", default=settings.enable_deepcache)
    parser.add_argument(
        "--codeformer_checkpoint_path",
        default=settings.codeformer_checkpoint_path,
        help="Path to codeformer.pth; LATENTSYNC_CODEFORMER_CKPT env var also accepted.",
    )
    parser.add_argument(
        "--codeformer_preload",
        action="store_true",
        default=settings.codeformer_preload,
        help="Load the CodeFormer model at server startup.",
    )
    parser.add_argument(
        "--codeformer_batch_size",
        type=int,
        default=settings.codeformer_batch_size,
        help="Faces per CodeFormer forward pass. 8 is a conservative default while LatentSync also occupies GPU memory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    settings.host = args.host
    settings.port = args.port
    settings.ffmpeg_path = args.ffmpeg_path
    settings.gpu_id = args.gpu_id
    settings.unet_config_path = args.unet_config_path
    settings.inference_ckpt_path = args.inference_ckpt_path
    settings.guidance_scale = args.guidance_scale
    settings.inference_steps = args.inference_steps
    settings.seed = args.seed
    settings.enable_deepcache = args.enable_deepcache
    settings.codeformer_checkpoint_path = args.codeformer_checkpoint_path
    settings.codeformer_preload = args.codeformer_preload
    settings.codeformer_batch_size = args.codeformer_batch_size

    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
