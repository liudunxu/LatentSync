import argparse
import asyncio
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
AUDIO_EMBEDS_CACHE_ROOT = RESULT_ROOT / "audio_embeds_cache"

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for directory in (INPUT_ROOT, OUTPUT_ROOT, AUDIO_EMBEDS_CACHE_ROOT):
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
    # Tier 1 -- adaptive per-frame w bucketing. ``sharp_threshold`` and
    # ``blurry_threshold`` are mouth-region Laplacian variance values
    # (typical 0.001-0.1 for 512x512 aligned faces); the per-bucket w
    # values follow. The defaults are conservative placeholders -- run
    # ``tools/calibrate_codeformer_thresholds.py`` against a real
    # short-drama sample to pick better numbers.
    codeformer_sharp_threshold: float = float(os.getenv("LATENTSYNC_CODEFORMER_SHARP_THRESHOLD", "0.05"))
    codeformer_blurry_threshold: float = float(os.getenv("LATENTSYNC_CODEFORMER_BLURRY_THRESHOLD", "0.01"))
    codeformer_w_sharp: float = float(os.getenv("LATENTSYNC_CODEFORMER_W_SHARP", "0.85"))
    codeformer_w_medium: float = float(os.getenv("LATENTSYNC_CODEFORMER_W_MEDIUM", "0.7"))
    codeformer_w_blurry: float = float(os.getenv("LATENTSYNC_CODEFORMER_W_BLURRY", "0.5"))
    # Tier 2 -- retry pass for frames that fail in the blurry bucket.
    codeformer_w_retry: float = float(os.getenv("LATENTSYNC_CODEFORMER_W_RETRY", "0.4"))
    codeformer_retry_max_frames: int = int(os.getenv("LATENTSYNC_CODEFORMER_RETRY_MAX_FRAMES", "64"))
    # Tier 3 -- mouth-ROI paste-back feather (Gaussian sigma in pixels).
    codeformer_mouth_mask_feather_sigma: float = float(
        os.getenv("LATENTSYNC_CODEFORMER_MOUTH_FEATHER_SIGMA", "5.0")
    )
    # When the mouth patch is taken from CodeFormer, its per-channel
    # mean/std is locally recomputed to match the inpainter's same-
    # region statistics, so the paste is invisible against the
    # surrounding face. 1.0 = full match (default), 0.0 = no match.
    codeformer_mouth_roi_color_match_strength: float = float(
        os.getenv("LATENTSYNC_CODEFORMER_MOUTH_ROI_COLOR_MATCH_STRENGTH", "1.0")
    )
    # Short-drama profile: when ``codeformer_short_drama_thresholds_enabled``
    # is True and the per-request ``codeformer_short_drama_profile`` is
    # also True, the three quality-check thresholds are loosened from
    # 0.20/0.15/[0.5,2.0] to the values below. Server-side defaults;
    # per-request overrides of the individual thresholds aren't
    # currently exposed (kept simple to avoid a 4-field UI).
    codeformer_short_drama_thresholds_enabled: bool = os.getenv(
        "LATENTSYNC_CODEFORMER_SHORT_DRAMA_THRESHOLDS", "1"
    ).lower() in {"1", "true", "yes"}
    codeformer_pixel_diff_short_drama: float = float(
        os.getenv("LATENTSYNC_CODEFORMER_PIXEL_DIFF_SHORT_DRAMA", "0.30")
    )
    codeformer_mouth_diff_short_drama: float = float(
        os.getenv("LATENTSYNC_CODEFORMER_MOUTH_DIFF_SHORT_DRAMA", "0.22")
    )
    codeformer_sharpness_high_short_drama: float = float(
        os.getenv("LATENTSYNC_CODEFORMER_SHARP_HIGH_SHORT_DRAMA", "2.5")
    )


settings = Settings()
logger = logging.getLogger("latentsync.api")


LANGUAGE_LIPSYNC_PRESETS = {
    "english": {
        "guidance_scale": 1.45,
        "mouth_temporal_stabilization_strength": 0.15,
        "mouth_audio_motion_min_scale": 0.75,
        "mouth_audio_motion_max_scale": 1.45,
    },
    "indonesian": {
        "guidance_scale": 1.50,
        "mouth_temporal_stabilization_strength": 0.14,
        "mouth_audio_motion_min_scale": 0.80,
        "mouth_audio_motion_max_scale": 1.60,
    },
    "filipino": {
        "guidance_scale": 1.45,
        "mouth_temporal_stabilization_strength": 0.14,
        "mouth_audio_motion_min_scale": 0.80,
        "mouth_audio_motion_max_scale": 1.55,
    },
    "malay": {
        "guidance_scale": 1.50,
        "mouth_temporal_stabilization_strength": 0.14,
        "mouth_audio_motion_min_scale": 0.80,
        "mouth_audio_motion_max_scale": 1.60,
    },
    "vietnamese": {
        "guidance_scale": 1.35,
        "mouth_temporal_stabilization_strength": 0.16,
        "mouth_audio_motion_min_scale": 0.70,
        "mouth_audio_motion_max_scale": 1.35,
    },
}

LANGUAGE_ALIASES = {
    "en": "english",
    "eng": "english",
    "english": "english",
    "id": "indonesian",
    "indonesian": "indonesian",
    "bahasa indonesia": "indonesian",
    "fil": "filipino",
    "filipino": "filipino",
    "philippines": "filipino",
    "tagalog": "filipino",
    "tl": "filipino",
    "ms": "malay",
    "malay": "malay",
    "bahasa melayu": "malay",
    "vi": "vietnamese",
    "vie": "vietnamese",
    "vietnamese": "vietnamese",
}


def _normalize_target_language(value: Optional[str]) -> str:
    if not value:
        return ""
    key = value.strip().lower().replace("_", " ").replace("-", " ")
    key = " ".join(key.split())
    return LANGUAGE_ALIASES.get(key, "")


def _payload_field_was_set(payload: BaseModel, field_name: str) -> bool:
    fields_set = getattr(payload, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(payload, "__fields_set__", set())
    return field_name in fields_set


# ---------------------------------------------------------------------------
# Pydantic models – kept identical to MuseTalk api.py for compatibility
# ---------------------------------------------------------------------------

class LipSyncRequest(BaseModel):
    video_url: str = Field(..., description="Source video URL")
    avatar_url: Optional[str] = Field(None, description="Reference avatar image URL")
    audio_url: str = Field(..., description="Driving audio URL")
    target_language: Optional[str] = Field(
        None,
        description="Optional dubbing target language preset: English, Indonesian, Filipino, Malay, Vietnamese.",
    )
    similarity_threshold: float = Field(0.5, ge=0.0, le=1.0)
    identity_margin: float = Field(0.05, ge=0.0, le=1.0)
    identity_cluster_threshold: float = Field(0.78, ge=0.0, le=1.0)
    default_identity_min_coverage: float = Field(0.5, ge=0.0, le=1.0)
    require_face_embedding: bool = True
    apply_identity_filter: bool = Field(
        False,
        description="Enable identity filtering. When True, uses avatar if provided; otherwise auto-detects the main speaker and filters to that face.",
    )
    scene_split_enabled: bool = Field(
        True,
        description="Split the video by detected scenes and run lip-sync on each scene independently, then concatenate the results.",
    )
    scene_split_threshold: float = Field(
        0.45,
        ge=0.0,
        le=1.0,
        description="Threshold for detecting scene boundaries when scene_split_enabled=True.",
    )
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
    # EMA alpha for the per-frame mouth_info (center + half-extents)
    # used to draw the dynamic inpaint mask. Higher = more weight on
    # the current frame (less lag, less mask-boundary smoothing);
    # lower = more weight on the previous frame (smoother mask,
    # but lags on fast mouth motion and can leave the inpaint region
    # offset from the real mouth on the specific frames where mouth
    # position jumps). 0.7 is the legacy default; bump toward 0.85-1.0
    # when individual frames show the inpaint region drifting off the
    # mouth onto the cheek / chin.
    aligned_mouth_ema_alpha: float = Field(0.7, ge=0.0, le=1.0)
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
    # --- HeyGen-like segment consistency (adapted from MuseTalk 4b4987a) ---
    # Refuse the time-window merge of two adjacent valid runs if a
    # hard cut is detected in the gap (a hard cut / shot boundary
    # would otherwise be bridged by a short passthrough, causing
    # the inpainter to splice across the cut). 0 disables the
    # per-frame histogram-distance check. See
    # MuseTalk docs/heygen_like_lipsync_segmentation_td.md §5.1.
    segment_consistency_hard_cut_enabled: bool = Field(
        True,
        description="When True, refuse the time-window merge if a hard cut is detected in the gap. Mirrors MuseTalk 4b4987a §5.1.",
    )
    segment_consistency_hard_cut_distance_threshold: float = Field(
        0.65,
        ge=0.0,
        le=2.0,
        description="Face-crop histogram distance above which a frame boundary is treated as a hard cut. 0.65 is conservative (obvious cuts only); 0 disables.",
    )
    # Track-aware merge. The pipeline already assigns a per-source-frame
    # ``track_id`` from ``continuity_break`` (identity / geometry /
    # mouth-region pixel diff). When two adjacent valid runs have
    # different track_ids the merge is refused even if the gap is
    # short -- a speaker switch is never bridged by a short
    # passthrough. Falls back to time-window merge when track_id is
    # missing on either side. Mirrors MuseTalk 4b4987a §5.5.
    segment_consistency_track_aware: bool = Field(
        True,
        description="When True, only merge adjacent valid runs whose track_id matches. Falls back to time-window merge when track_id is missing. Mirrors MuseTalk 4b4987a §5.5.",
    )
    # Post-merge minimum duration. After the merge, any valid run
    # whose total length is below this many seconds is forced
    # entirely to passthrough to avoid the splice artifacts that a
    # short isolated segment would produce. 0.4s keeps the guardrail
    # against isolated flicker while allowing normal short utterances /
    # short shots to still generate. 0 disables.
    # Mirrors MuseTalk 4b4987a §5.7.
    min_merged_lipsync_seconds: float = Field(
        0.4,
        ge=0.0,
        le=10.0,
        description="After segment consistency merge, valid runs shorter than this many seconds are forced to passthrough. 0 disables. Mirrors MuseTalk 4b4987a §5.7.",
    )
    # Scene-cut continuity guard. This is intentionally separate from
    # segment-consistency hard-cut detection: the latter only gates whether a
    # short passthrough gap can be merged, while this one catches adjacent
    # source-frame hard cuts and resets temporal/affine carry state at the new
    # shot. It does not skip the frame by default.
    scene_cut_break_enabled: bool = True
    scene_cut_break_threshold: float = Field(
        0.45,
        ge=0.0,
        le=1.0,
        description="Adjacent source-frame scene-cut score above this value resets temporal carry state; 0 disables.",
    )
    # Shot-level passthrough guard for short-drama production. Frame-level
    # filters can produce visible generated/source flicker inside a side-face
    # or fast-turn shot. When enabled, any shot whose prefilter skip ratio
    # crosses the threshold is kept entirely as source video.
    shot_passthrough_enabled: bool = False
    shot_passthrough_skip_ratio_threshold: float = Field(
        0.45,
        ge=0.0,
        le=1.0,
        description="When shot passthrough is enabled, force a shot to source if this fraction of its frames are already prefilter-skipped.",
    )
    shot_passthrough_min_frames: int = Field(
        8,
        ge=1,
        le=300,
        description="Minimum shot length eligible for shot-level passthrough.",
    )
    shot_passthrough_min_bad_frames: int = Field(
        3,
        ge=1,
        le=300,
        description="Minimum already-skipped frames required before forcing a whole shot to passthrough.",
    )
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
    # Frame-to-frame mouth stabilization. Keep this light: too much carryover
    # damps open-mouth frames and makes speech look under-articulated.
    mouth_temporal_stabilization_strength: float = Field(0.15, ge=0.0, le=0.6)
    mouth_temporal_stabilization_max_delta: float = Field(0.12, ge=0.0, le=2.0)
    mouth_audio_adaptive_motion_enabled: bool = True
    # Adaptive motion: preserve more current generated mouth motion,
    # especially on high-energy speech, so open-mouth frames are not pulled
    # back toward the smoothed/previous-frame mouth too aggressively.
    mouth_audio_motion_min_scale: float = Field(0.75, ge=0.0, le=2.0)
    mouth_audio_motion_max_scale: float = Field(1.60, ge=0.0, le=2.0)
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
    # Aggressive side-face passthrough. When > 0, frames with abs(yaw) in
    # the band (side_face_passthrough_yaw_threshold, yaw_skip_threshold)
    # are also marked as passthrough -- i.e. the diffusion inpainter is
    # bypassed and the original frame is kept. Useful when "side-face
    # residue ghost" artifacts dominate the output. 22.5 in effect says
    # "don't try to inpaint any non-frontal face"; the inpainter only
    # runs on faces at or below the threshold. 0 disables (legacy
    # 30°-only behavior).
    side_face_passthrough_yaw_threshold: float = Field(0.0, ge=0.0, le=90.0)
    # Episode-level side-face filter: when N consecutive frames exceed
    # yaw_skip_threshold, also skip `pre_pad`/`post_pad` frames of
    # transition zone around the episode (frames whose yaw is between
    # yaw_warn_threshold and yaw_skip_threshold -- the face is clearly
    # turning and the affine alignment is unreliable there). Set
    # pre_pad/post_pad to 0 to disable the padding and only do the
    # per-frame yaw skip.
    side_face_episode_pre_pad: int = Field(1, ge=0, le=30)
    side_face_episode_post_pad: int = Field(1, ge=0, le=30)
    # Cross-fade between inpaint and source at side-face boundaries. The
    # episode pad above creates a hard binary inpaint->source cut at the
    # warn-band edge. The blend zone softens that cut: the N inpaint
    # frames just before/after each skip block are mixed with the
    # source frame by a coefficient that ramps from 0.5 at the boundary
    # to 0 at fade_frames away. Set to 0 to disable (pure binary cut).
    # Independent of side_face_episode_pre_pad / post_pad; the blend
    # only acts on inpaint frames, never on the skip block itself, so
    # disabled pad + blend=3 still works (no boundaries to blend at).
    side_face_blend_fade_frames: int = Field(3, ge=0, le=30)
    # Warn-band ratio: yaws above `yaw_skip_threshold * ratio` but below
    # `yaw_skip_threshold` are treated as transition frames / near-profile
    # runs. Default 0.80 = warn @ 24° for the default 30° threshold, keeping
    # the side-face guardrail while avoiding over-skipping mild turns.
    yaw_warn_threshold_ratio: float = Field(0.80, ge=0.0, le=1.0)
    side_face_warn_min_run_frames: int = Field(
        0,
        ge=0,
        le=120,
        description="Skip sustained near-profile runs above the yaw warn threshold; 0 disables.",
    )
    # Time-based alternative to ``side_face_warn_min_run_frames``. The
    # warn-run skip uses ``max(min_run_frames, round(min_run_seconds *
    # fps))`` as the effective threshold. The time form is more
    # intuitive to set from a UI ("skip if the side face lasts >0.5s").
    # 0 disables (the run-skip still respects ``min_run_frames``).
    # Combined with the absolute yaw thresholds this implements the
    # "if a side face is sustained, just passthrough the whole run
    # instead of trying to inpaint" behavior.
    side_face_warn_min_run_seconds: float = Field(
        0.0,
        ge=0.0,
        le=10.0,
        description="Skip sustained near-profile runs whose duration is > this many seconds; 0 disables.",
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
    # --- CodeFormer Tier 1/2/3 (short-drama-tuned, added 2026-06) ---
    # Master switch: when True, the three toggles below are honored and
    # the looser quality-check thresholds (``codeformer_*_short_drama``
    # in ``Settings``) replace the conservative 08cb35f defaults. When
    # False, the per-tier toggles are ignored and behavior falls back
    # to the 08cb35f single-w / strict-threshold path. Default True
    # because the short-drama use case is the request shape the project
    # is tuned for; clients that want the old behavior set False.
    codeformer_short_drama_profile: bool = Field(
        True,
        description="Master switch for the short-drama-tuned CodeFormer: enables adaptive w, "
                    "looser quality-check thresholds, mouth-only paste-back, and (when "
                    "codeformer_retry_enabled is also True) the w=0.4 retry pass on the blurry "
                    "bucket. Set False to restore 08cb35f conservative behavior.",
    )
    codeformer_adaptive_w_enabled: bool = Field(
        True,
        description="Bucket frames by input mouth-region sharpness; run CodeFormer with "
                    "w_sharp=0.85, w_medium=0.7, w_blurry=0.5 per bucket. Only effective when "
                    "codeformer_short_drama_profile is also True.",
    )
    codeformer_retry_enabled: bool = Field(
        False,
        description="If a frame in the blurry bucket fails the quality check on the first pass, "
                    "re-run it with w=0.4 (more aggressive codebook). Capped at "
                    "codeformer_retry_max_frames per request. Only effective when "
                    "codeformer_short_drama_profile is also True.",
    )
    codeformer_mouth_only_paste_enabled: bool = Field(
        True,
        description="When the quality check passes, paste back only the mouth ROI from CodeFormer; "
                    "rest of the face stays as the inpainter's original output. Decouples the "
                    "aggressive-w passes from identity drift on eyes/forehead/cheeks. Only "
                    "effective when codeformer_short_drama_profile is also True.",
    )
    # Post-CodeFormer cross-frame 1-order EMA on the restored crops.
    # CodeFormer is stateless per-frame so a high-frequency flicker can
    # persist across consecutive valid frames. This EMA dampens that
    # flicker by blending each restored crop toward the previous one
    # (alpha = weight on the previous frame). 0 disables. Mirrors
    # MuseTalk commit ce7b684 (``codeformer_temporal_alpha``).
    # Track-aware mode (default True) refuses the mix across
    # speaker/identity boundaries -- a track switch with EMA on would
    # otherwise smear the old face onto the new identity for one frame
    # (a single-frame pop that downstream smoothing can amplify).
    codeformer_post_ema_alpha: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Post-CodeFormer cross-frame EMA on the restored face crops. 0 disables; 0.8 (default) "
                    "dampens per-frame CF flicker. See MuseTalk commit ce7b684.",
    )
    codeformer_post_ema_track_aware: bool = Field(
        True,
        description="When True, refuse the post-CF EMA mix across speaker/track_id boundaries. Falls "
                    "back to legacy adjacency-only rule when track_id is missing on either side.",
    )
    # --- Adaptive quality fallback (added 2026-06) ---
    # After diffusion/post-processing, evaluate a per-frame composite quality
    # score and fallback to the source frame when it is too low. Designed for
    # short-drama content where a bad generated mouth is worse than the
    # original, but isolated fallback decisions must be suppressed to avoid
    # visible flicker.
    adaptive_quality_fallback_enabled: bool = Field(
        False,
        description="Enable per-frame adaptive quality fallback. Combines mouth sharpness, "
                    "mouth-region diff, identity similarity, yaw, audio confidence and "
                    "temporal stability into a single score; frames below the adaptive "
                    "threshold are replaced with the source frame.",
    )
    adaptive_quality_fallback_threshold: float = Field(
        0.35, ge=0.0, le=1.0,
        description="Base quality threshold in [0, 1]. Lower = more tolerant of artefacts. "
                    "The actual threshold is raised automatically if the fallback ratio "
                    "would exceed adaptive_quality_fallback_max_ratio.",
    )
    adaptive_quality_fallback_max_ratio: float = Field(
        0.35, ge=0.0, le=1.0,
        description="Maximum fraction of non-prefiltered frames allowed to fallback. "
                    "If the base threshold would skip more than this, the threshold is "
                    "raised until the budget is met.",
    )
    adaptive_quality_fallback_hysteresis_frames: int = Field(
        2, ge=0, le=10,
        description="Suppress isolated single-frame fallback decisions. A run of fallback "
                    "frames shorter than or equal to this length is reverted unless it "
                    "touches the start/end of the clip.",
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
    # Filter to only return face clusters whose mouth was visibly open at
    # some sampled frame during the scan. ``mouth_motion_max_openness`` is
    # the highest dark-pixel ratio observed in the mouth region across all
    # frames that hit the cluster. A closed-mouth face (listener, profile
    # glance) typically scores < 0.08; an open-mouth face with visible
    # cavity scores 0.15-0.50. Default 0.10 drops most silent faces
    # (listener / side face) while keeping any face that visibly spoke.
    # Set to 0.0 to disable the filter and return every distinct face.
    # Mirrors MuseTalk commit 8a382f0 ("/api/faces filters silent clusters
    # by mouth openness").
    min_mouth_openness: float = Field(
        0.10,
        ge=0.0,
        le=1.0,
        description="Drop clusters whose max mouth-region dark-pixel ratio across the scan is below this. 0 disables; default 0.10 drops silent faces.",
    )


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
        # Cache avatar embeddings by URL to avoid re-extracting the same
        # reference face on repeated requests.
        self.avatar_embedding_cache: Dict[str, np.ndarray] = {}
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
                audio_embeds_cache_dir=str(AUDIO_EMBEDS_CACHE_ROOT),
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
        # The restorer is a singleton (per the lazy-load pattern); the
        # per-request ``codeformer_short_drama_profile`` flag only
        # controls the 3 Tier toggles, NOT the quality-check thresholds
        # (those are server-side via env vars). To get the original
        # 08cb35f strict thresholds, set the corresponding
        # ``LATENTSYNC_CODEFORMER_*_SHORT_DRAMA`` env vars to the old
        # values (0.20 / 0.15 / 2.0).
        self.codeformer_restorer = CodeFormerRestorer(
            checkpoint_path=settings.codeformer_checkpoint_path,
            device=device,
            batch_size=settings.codeformer_batch_size,
            fallback_pixel_diff=settings.codeformer_pixel_diff_short_drama,
            fallback_mouth_diff=settings.codeformer_mouth_diff_short_drama,
            fallback_sharpness_high=settings.codeformer_sharpness_high_short_drama,
            sharp_threshold=settings.codeformer_sharp_threshold,
            blurry_threshold=settings.codeformer_blurry_threshold,
            w_sharp=settings.codeformer_w_sharp,
            w_medium=settings.codeformer_w_medium,
            w_blurry=settings.codeformer_w_blurry,
            w_retry=settings.codeformer_w_retry,
            retry_max_frames=settings.codeformer_retry_max_frames,
            mouth_mask_feather_sigma=settings.codeformer_mouth_mask_feather_sigma,
            mouth_roi_color_match_strength=settings.codeformer_mouth_roi_color_match_strength,
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

    @staticmethod
    def _compute_mouth_openness(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Estimate how open the mouth is, in [0.0, 1.0].

        Cheap HSV dark-pixel ratio over the lower-middle third of the bbox
        (typical mouth location under a frontal face detector). A closed
        mouth (lip line only) scores 0.02-0.08; an open mouth showing
        dark cavity scores 0.15-0.50; a wide-open shout can reach 0.6+.

        Intentionally cheap (no extra model forward) so it can run on every
        accepted detection during the scan. Mirrors MuseTalk commit
        8a382f0 (``_compute_mouth_openness``); the V < 80 cutoff matches
        their tuning. Returns 0.0 on bad bbox / tiny region.
        """
        x1, y1, x2, y2 = bbox
        h, w = y2 - y1, x2 - x1
        if h <= 0 or w <= 0:
            return 0.0
        # Lower 40% of the bbox (typical mouth strip), horizontally the
        # middle 60% (avoid jaw / face edges).
        my1 = y1 + int(h * 0.60)
        my2 = y2
        mx1 = x1 + int(w * 0.20)
        mx2 = x2 - int(w * 0.20)
        if my2 - my1 < 4 or mx2 - mx1 < 4:
            return 0.0
        region = frame[my1:my2, mx1:mx2]
        if region.size == 0:
            return 0.0
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        # 80 is well below typical lip color (100-160) and well above the
        # noise floor on dark skin. Matches MuseTalk's V < 80 threshold.
        return float((v < 80).mean())

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
                    # Cheap HSV dark-pixel ratio on the lower-middle bbox --
                    # cached on the face dict so the cluster loop can take the
                    # max without recomputing. Mirrors MuseTalk 8a382f0.
                    face["mouth_openness"] = self._compute_mouth_openness(frame, face["bbox"])
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

                # Per-cluster max mouth openness -- a face that briefly
                # opened its mouth once should be kept even if 90% of its
                # sampled frames show a closed mouth. Mirrors MuseTalk
                # 8a382f0.
                info_openness = float(info.get("mouth_openness", 0.0))

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
                    if info_openness > float(c.get("mouth_motion_max_openness", 0.0)):
                        c["mouth_motion_max_openness"] = info_openness
                else:
                    clusters.append({
                        "embedding": embedding,
                        "count": 1,
                        "max_area": _box_area(info["bbox"]),
                        "best_frame_index": info["frame_index"],
                        "best_bbox": info["bbox"],
                        "best_detection_score": info["detection_score"],
                        "descriptors": [info],
                        "mouth_motion_max_openness": info_openness,
                    })

            clusters.sort(key=lambda c: (int(c["count"]), int(c["max_area"])), reverse=True)

            # Drop silent clusters: faces whose mouth never visibly opened
            # during the scan. Default threshold 0.10 keeps visible speakers
            # and drops listeners / side-glance faces. Set
            # ``min_mouth_openness=0`` on the request to disable.
            # Mirrors MuseTalk 8a382f0.
            rejected_silent_face_count = 0
            if payload.min_mouth_openness > 0.0:
                kept_clusters: List[Dict[str, object]] = []
                for c in clusters:
                    if float(c.get("mouth_motion_max_openness", 0.0)) >= payload.min_mouth_openness:
                        kept_clusters.append(c)
                    else:
                        rejected_silent_face_count += 1
                clusters = kept_clusters

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
                    "mouth_motion_max_openness": float(
                        cluster.get("mouth_motion_max_openness", 0.0)
                    ),
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
                "rejected_silent_face_count": rejected_silent_face_count,
                "min_mouth_openness_threshold": float(payload.min_mouth_openness),
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

            target_language = _normalize_target_language(payload.target_language)
            language_preset = (
                LANGUAGE_LIPSYNC_PRESETS.get(target_language, {})
                if target_language
                else {}
            )

            # Resolve per-request overrides for inference quality. None = use
            # the server-side default loaded from env vars at startup.
            effective_guidance_scale = (
                payload.guidance_scale_override
                if payload.guidance_scale_override is not None
                else language_preset.get("guidance_scale", settings.guidance_scale)
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
            effective_mouth_temporal_stabilization_strength = (
                payload.mouth_temporal_stabilization_strength
                if _payload_field_was_set(payload, "mouth_temporal_stabilization_strength")
                else language_preset.get(
                    "mouth_temporal_stabilization_strength",
                    payload.mouth_temporal_stabilization_strength,
                )
            )
            effective_mouth_audio_motion_min_scale = (
                payload.mouth_audio_motion_min_scale
                if _payload_field_was_set(payload, "mouth_audio_motion_min_scale")
                else language_preset.get(
                    "mouth_audio_motion_min_scale",
                    payload.mouth_audio_motion_min_scale,
                )
            )
            effective_mouth_audio_motion_max_scale = (
                payload.mouth_audio_motion_max_scale
                if _payload_field_was_set(payload, "mouth_audio_motion_max_scale")
                else language_preset.get(
                    "mouth_audio_motion_max_scale",
                    payload.mouth_audio_motion_max_scale,
                )
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

            effective_apply_identity_filter = bool(payload.apply_identity_filter)

            logger.info(f"[LipSync] Starting pipeline: video={video_path}, audio={audio_path}, "
                         f"guidance_scale={effective_guidance_scale}, steps={effective_inference_steps}, "
                         f"seed={effective_seed}, has_reference_embedding={reference_embedding is not None}, "
                         f"apply_identity_filter={effective_apply_identity_filter}, "
                         f"scene_split_enabled={payload.scene_split_enabled}, "
                         f"scene_split_threshold={payload.scene_split_threshold}, "
                         f"target_language={target_language or 'none'}, "
                         f"codeformer_enabled={payload.codeformer_enabled}, "
                         f"codeformer_loaded={codeformer_restorer is not None and codeformer_restorer.is_loaded}, "
                         f"codeformer_fidelity_weight={payload.codeformer_fidelity_weight}, "
                         f"codeformer_adain={payload.codeformer_adain}")
            self.pipeline(
                video_path=str(video_path),
                audio_path=str(audio_path),
                video_out_path=str(output_path),
                audio_sync_offset_seconds=payload.audio_sync_offset_seconds,
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
                apply_identity_filter=effective_apply_identity_filter,
                identity_similarity_threshold=payload.similarity_threshold,
                quality_gate_enabled=payload.quality_gate_enabled,
                quality_min_laplacian=payload.quality_min_laplacian,
                quality_min_sharpness_ratio=payload.quality_min_sharpness_ratio,
                quality_ref_min_laplacian=payload.quality_ref_min_laplacian,
                quality_max_fallback_ratio=payload.quality_max_fallback_ratio,
                yaw_skip_threshold=payload.yaw_skip_threshold,
                yaw_rate_skip_threshold=payload.yaw_rate_skip_threshold,
                side_face_passthrough_yaw_threshold=payload.side_face_passthrough_yaw_threshold,
                side_face_episode_pre_pad=payload.side_face_episode_pre_pad,
                side_face_episode_post_pad=payload.side_face_episode_post_pad,
                side_face_blend_fade_frames=payload.side_face_blend_fade_frames,
                yaw_warn_threshold_ratio=payload.yaw_warn_threshold_ratio,
                side_face_warn_min_run_frames=payload.side_face_warn_min_run_frames,
                side_face_warn_min_run_seconds=payload.side_face_warn_min_run_seconds,
                mouth_occlusion_skip_threshold=payload.mouth_occlusion_skip_threshold,
                motion_blur_skip_threshold=getattr(payload, "motion_blur_skip_threshold", 0.08),
                face_jump_center_threshold=payload.face_jump_center_threshold,
                face_jump_scale_threshold=payload.face_jump_scale_threshold,
                lipsync_continuity_max_center_shift=payload.lipsync_continuity_max_center_shift,
                lipsync_continuity_max_scale_change=payload.lipsync_continuity_max_scale_change,
                aligned_mouth_ema_alpha=payload.aligned_mouth_ema_alpha,
                lipsync_mouth_diff_break_threshold=payload.lipsync_mouth_diff_break_threshold,
                lipsync_min_segment_frames=payload.lipsync_min_segment_frames,
                segment_consistency_hard_cut_enabled=payload.segment_consistency_hard_cut_enabled,
                segment_consistency_hard_cut_distance_threshold=payload.segment_consistency_hard_cut_distance_threshold,
                segment_consistency_track_aware=payload.segment_consistency_track_aware,
                min_merged_lipsync_seconds=payload.min_merged_lipsync_seconds,
                scene_cut_break_enabled=payload.scene_cut_break_enabled,
                scene_cut_break_threshold=payload.scene_cut_break_threshold,
                lipsync_min_face_area_ratio=payload.lipsync_min_face_area_ratio,
                shot_passthrough_enabled=payload.shot_passthrough_enabled,
                shot_passthrough_skip_ratio_threshold=payload.shot_passthrough_skip_ratio_threshold,
                shot_passthrough_min_frames=payload.shot_passthrough_min_frames,
                shot_passthrough_min_bad_frames=payload.shot_passthrough_min_bad_frames,
                scene_split_enabled=payload.scene_split_enabled,
                scene_split_threshold=payload.scene_split_threshold,
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
                mouth_temporal_stabilization_strength=effective_mouth_temporal_stabilization_strength,
                mouth_temporal_stabilization_max_delta=payload.mouth_temporal_stabilization_max_delta,
                mouth_audio_adaptive_motion_enabled=payload.mouth_audio_adaptive_motion_enabled,
                mouth_audio_motion_min_scale=effective_mouth_audio_motion_min_scale,
                mouth_audio_motion_max_scale=effective_mouth_audio_motion_max_scale,
                # CodeFormer postprocess. ``codeformer_enabled`` is honoured
                # only when ``codeformer_restorer`` actually loaded; when
                # it didn't (e.g. checkpoint missing and not required) the
                # pipeline logs a warning and skips. Either way, the
                # pipeline still runs -- the failure mode is "no
                # postprocess", not "broken output". The three Tier 1/2/3
                # toggles below are gated by the per-request master
                # switch ``codeformer_short_drama_profile`` so clients
                # can opt back to the conservative 08cb35f single-w
                # behavior with a single flag flip.
                codeformer_enabled=payload.codeformer_enabled,
                codeformer_fidelity_weight=payload.codeformer_fidelity_weight,
                codeformer_adain=payload.codeformer_adain,
                codeformer_adaptive_w_enabled=(
                    payload.codeformer_adaptive_w_enabled
                    and payload.codeformer_short_drama_profile
                ),
                codeformer_retry_enabled=(
                    payload.codeformer_retry_enabled
                    and payload.codeformer_short_drama_profile
                ),
                codeformer_mouth_only_paste_enabled=(
                    payload.codeformer_mouth_only_paste_enabled
                    and payload.codeformer_short_drama_profile
                ),
                codeformer_post_ema_alpha=payload.codeformer_post_ema_alpha,
                codeformer_post_ema_track_aware=payload.codeformer_post_ema_track_aware,
                codeformer_restorer=codeformer_restorer,
                adaptive_quality_fallback_enabled=payload.adaptive_quality_fallback_enabled,
                adaptive_quality_fallback_threshold=payload.adaptive_quality_fallback_threshold,
                adaptive_quality_fallback_max_ratio=payload.adaptive_quality_fallback_max_ratio,
                adaptive_quality_fallback_hysteresis_frames=payload.adaptive_quality_fallback_hysteresis_frames,
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
            small_face_skip_count = int(run_stats.get("small_face_skip_count", 0))
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
            adaptive_quality_fallback_frames = int(
                run_stats.get("adaptive_quality_fallback_frames", 0)
            )
            adaptive_quality_fallback_enabled = bool(
                run_stats.get("adaptive_quality_fallback_enabled", False)
            )
            adaptive_quality_fallback_threshold = float(
                run_stats.get("adaptive_quality_fallback_threshold", 0.35)
            )
            adaptive_quality_fallback_max_ratio = float(
                run_stats.get("adaptive_quality_fallback_max_ratio", 0.35)
            )
            adaptive_quality_fallback_hysteresis_frames = int(
                run_stats.get("adaptive_quality_fallback_hysteresis_frames", 2)
            )
            effective_skip_frames = int(run_stats.get("effective_skip_frames", 0))
            silent_skip_frames = int(run_stats.get("silent_skip_frames", 0))
            skipped_inference_batches = int(run_stats.get("skipped_inference_batches", 0))
            skipped_inference_frames = int(run_stats.get("skipped_inference_frames", 0))
            identity_similarity_stats = run_stats.get("identity_similarity") or {}
            identity_skip_count = int(run_stats.get("identity_skip_count", 0))
            active_speaker_stats = run_stats.get("active_speaker") or {}
            codeformer_stats = run_stats.get("codeformer") or {}
            mouth_temporal_stats = run_stats.get("mouth_temporal") or {}
            segment_consistency_reasons = run_stats.get("segment_consistency") or {}
            scene_cut_break_count = int(run_stats.get("scene_cut_break_count", 0))
            shot_passthrough_shots = int(run_stats.get("shot_passthrough_shots", 0))
            shot_passthrough_frames = int(run_stats.get("shot_passthrough_frames", 0))
            generation_summary = run_stats.get("generation_summary") or {
                "total_frames": int(source_frame_count),
                "latentsync_generated_frames": int(effective_generated_frames),
                "passthrough_frames": int(effective_skip_frames),
                "passthrough_ratio": float(effective_skip_frames / max(1, source_frame_count)),
                "prefilter_passthrough_frames": int(pre_skip_frames),
                "small_face_passthrough_frames": int(small_face_skip_count),
                "shot_passthrough_frames": int(shot_passthrough_frames),
                "shot_passthrough_shots": int(shot_passthrough_shots),
                "quality_passthrough_frames": int(quality_skip_frames),
                "adaptive_quality_passthrough_frames": int(adaptive_quality_fallback_frames),
                "silent_passthrough_frames": int(silent_skip_frames),
                "skipped_inference_batches": int(skipped_inference_batches),
                "skipped_inference_frames": int(skipped_inference_frames),
                "route": "latentsync_or_passthrough",
            }

            return {
                "output_path": output_path,
                "source_frame_count": source_frame_count,
                "output_frame_count": source_frame_count,
                "audio_frame_count": 0,
                "source_fps": round(float(source_fps), 6),
                "audio_feature_fps": 0.0,
                "audio_sync_offset_frames": int(run_stats.get("audio_sync_offset_frames", 0)),
                "audio_sync_offset_output_frames": int(run_stats.get("audio_sync_offset_output_frames", 0)),
                "audio_sync_offset_seconds": float(run_stats.get("audio_sync_offset_seconds", payload.audio_sync_offset_seconds)),
                "effective_guidance_scale": effective_guidance_scale,
                "effective_inference_steps": effective_inference_steps,
                "effective_seed": effective_seed,
                "language_preset": {
                    "requested": payload.target_language or "",
                    "resolved": target_language,
                    "applied": bool(language_preset),
                    "values": dict(language_preset),
                },
                "generation_summary": dict(generation_summary),
                "retry_recommendation": dict(run_stats.get("retry_recommendation") or {}),
                "shot_summary": dict(run_stats.get("shot_summary") or {}),
                "routing_manifest": list(run_stats.get("routing_manifest") or []),
                "matched_source_frames": source_frame_count,
                "filled_source_frames": 0,
                "filtered_motion_frames": 0,
                "filtered_fast_motion_frames": 0,
                "continuity_filled_source_frames": 0,
                "filtered_small_face_frames": small_face_skip_count,
                "filtered_short_segment_frames": int(segment_consistency_reasons.get("too_short", 0)),
                "segment_consistency_reasons": dict(segment_consistency_reasons),
                "segment_consistency_hard_cut_enabled": bool(run_stats.get("segment_consistency_hard_cut_enabled", False)),
                "segment_consistency_track_aware": bool(run_stats.get("segment_consistency_track_aware", False)),
                "min_merged_lipsync_seconds": float(run_stats.get("min_merged_lipsync_seconds", 0.0)),
                "scene_cut_break_enabled": bool(run_stats.get("scene_cut_break_enabled", payload.scene_cut_break_enabled)),
                "scene_cut_break_threshold": float(run_stats.get("scene_cut_break_threshold", payload.scene_cut_break_threshold)),
                "scene_cut_break_count": scene_cut_break_count,
                "scene_split_enabled": bool(run_stats.get("scene_split_enabled", payload.scene_split_enabled)),
                "scene_split_threshold": float(run_stats.get("scene_split_threshold", payload.scene_split_threshold)),
                "scene_count": int(run_stats.get("scene_count", 1)),
                "scene_split_frames": list(run_stats.get("scene_split_frames", [])),
                "shot_passthrough_enabled": bool(run_stats.get("shot_passthrough_enabled", payload.shot_passthrough_enabled)),
                "shot_passthrough_skip_ratio_threshold": float(
                    run_stats.get("shot_passthrough_skip_ratio_threshold", payload.shot_passthrough_skip_ratio_threshold)
                ),
                "shot_passthrough_min_frames": int(run_stats.get("shot_passthrough_min_frames", payload.shot_passthrough_min_frames)),
                "shot_passthrough_min_bad_frames": int(
                    run_stats.get("shot_passthrough_min_bad_frames", payload.shot_passthrough_min_bad_frames)
                ),
                "shot_passthrough_shots": shot_passthrough_shots,
                "shot_passthrough_frames": shot_passthrough_frames,
                "smoothed_source_frames": 0,
                "matched_or_filled_source_frames": source_frame_count,
                "eligible_source_frames": source_frame_count,
                "generated_output_frames": source_frame_count,
                "quality_fallback_frames": quality_fallback_frames,
                "prefilter_skip_frames": pre_skip_frames,
                "quality_skip_frames": quality_skip_frames,
                "adaptive_quality_fallback_enabled": adaptive_quality_fallback_enabled,
                "adaptive_quality_fallback_frames": adaptive_quality_fallback_frames,
                "adaptive_quality_fallback_threshold": adaptive_quality_fallback_threshold,
                "adaptive_quality_fallback_max_ratio": adaptive_quality_fallback_max_ratio,
                "adaptive_quality_fallback_hysteresis_frames": adaptive_quality_fallback_hysteresis_frames,
                "yaw_skip_count": yaw_skip_count,
                "yaw_rate_skip_count": yaw_rate_skip_count,
                "mouth_occlusion_skip_count": mouth_occlusion_skip_count,
                "motion_blur_skip_count": motion_blur_skip_count,
                "face_jump_skip_count": face_jump_skip_count,
                "small_face_skip_count": small_face_skip_count,
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
                "active_speaker": dict(active_speaker_stats),
                "apply_identity_filter": bool(effective_apply_identity_filter),
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
                    "stabilization_strength": float(effective_mouth_temporal_stabilization_strength),
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
            "mouth_motion_max_openness": float(item.get("mouth_motion_max_openness", 0.0)),
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
async def create_lipsync(payload: LipSyncRequest, request: Request) -> Dict[str, object]:
    if payload.parsing_mode not in {"jaw", "raw"}:
        raise HTTPException(status_code=400, detail="parsing_mode must be 'jaw' or 'raw'")

    logger.info(f"[/api/lipsync] Request: video_url={payload.video_url}, audio_url={payload.audio_url}, avatar_url={payload.avatar_url}")
    job_id = uuid.uuid4().hex
    job_input_dir = INPUT_ROOT / job_id
    job_output_dir = OUTPUT_ROOT / job_id
    job_input_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)

    # Download video and audio (and avatar if provided) in parallel to reduce
    # request startup latency.
    async def _download_avatar():
        if not payload.avatar_url:
            return None
        return await asyncio.to_thread(
            _download_to_file,
            payload.avatar_url,
            job_input_dir,
            "avatar",
            IMAGE_SUFFIXES,
            ".jpg",
        )

    video_path, audio_path, avatar_downloaded = await asyncio.gather(
        asyncio.to_thread(
            _download_to_file, payload.video_url, job_input_dir, "video", VIDEO_SUFFIXES, ".mp4"
        ),
        asyncio.to_thread(
            _download_to_file, payload.audio_url, job_input_dir, "audio", AUDIO_SUFFIXES, ".wav"
        ),
        _download_avatar(),
    )
    try:
        input_paths = {"video": video_path, "audio": audio_path}
        reference_embedding = None
        if avatar_downloaded is not None:
            runtime.load_detectors()
            avatar_url = payload.avatar_url or ""
            cached = runtime.avatar_embedding_cache.get(avatar_url)
            if cached is not None:
                reference_embedding = cached
                logger.info(
                    "[LipSync] Using cached reference face embedding for avatar_url=%s, shape=%s",
                    avatar_url,
                    reference_embedding.shape,
                )
            elif runtime.face_embedder is not None:
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
                            runtime.avatar_embedding_cache[avatar_url] = reference_embedding
                            selected_bbox = getattr(selected_face, "bbox", None)
                            selected_score = float(getattr(selected_face, "det_score", 0.0))
                            logger.info(
                                "[LipSync] Loaded and cached reference face embedding, faces=%d, selected_score=%.3f, selected_bbox=%s, shape=%s",
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
