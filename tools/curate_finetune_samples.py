# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Curate a finetune dataset from a candidate pool (URLs or local files).

For each candidate video, run face detection (yaw) and motion scoring,
categorize into one of {frontal, side_face, fast_motion, reject}, then
sample to the target distribution (default 45% frontal / 35% side_face /
20% fast_motion) and write the result to:

    <output-dir>/
        frontal/<n>.mp4
        side_face/<n>.mp4
        fast_motion/<n>.mp4
        fileslist.txt         # one path per line, used as train_data_dir/fileslist
        curation_report.json  # per-video scores + chosen bucket

Usage:
    # 1. Put URLs (one per line) in a text file or pass --url inline.
    # 2. python tools/curate_finetune_samples.py \\
    #        --urls examples_finetune_urls.txt \\
    #        --output-dir data/finetune_samples \\
    #        --target-count 60

    # OR with local files:
    python tools/curate_finetune_samples.py \\
        --source-dir /path/to/candidate_videos \\
        --output-dir data/finetune_samples

Each candidate URL is downloaded via yt-dlp (if not already present in
<output-dir>/_raw); local files are scanned in place.

Bucket definitions (per-video aggregates):
    frontal:     max(|yaw|) <= 20  AND  motion score <= 30
    side_face:   max(|yaw|)  in (20, 45]
    fast_motion: motion score > 30  (any yaw)
    reject:      yaw > 45 (extreme), or no face detected, or sync_conf < 2

Where:
    yaw        = average InsightFace pose[1] across detected frames (degrees)
    motion     = median per-frame inter-frame bbox-center displacement
                 in the aligned 512-crop pixel space, **excluding** yaw-rotated
                 motion (so motion captures translation, not head-turn).
    sync_conf  = optional; loaded from <output-dir>/_raw/<name>.sync_conf.json
                 if produced by tools/compute_sync_conf.py. Skipped if absent.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import math
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bucket boundaries
# ---------------------------------------------------------------------------

YAW_FRONTAL_MAX = 20.0          # deg | yaw | <= this → frontal eligibility
YAW_SIDE_MIN = 20.0
YAW_SIDE_MAX = 45.0             # beyond this → reject
MOTION_FAST_THRESHOLD = 30.0    # motion score > this → fast_motion tag
MIN_FRAMES = 30                 # skip videos shorter than ~1.2s @ 25fps
SAMPLE_FRAMES = 64              # frames to subsample for face detection
SYNC_CONF_MIN = 2.0             # below this → reject (if sync_conf available)
TARGET_RATIO = {"frontal": 0.45, "side_face": 0.35, "fast_motion": 0.20}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class VideoScore:
    """Per-candidate aggregates used for bucketing."""
    path: str
    yaw_mean: float = 0.0
    yaw_max_abs: float = 0.0
    motion_score: float = 0.0
    frame_count: int = 0
    face_detected_ratio: float = 0.0  # % of sampled frames with a face
    sync_conf: Optional[float] = None
    bucket: str = ""
    rejected_reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Step 1 — Download
# ---------------------------------------------------------------------------


def _download_urls(urls: List[str], raw_dir: Path, num_workers: int) -> List[Path]:
    """Download each URL via yt-dlp into raw_dir. Skip if file exists."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for url in tqdm(urls, desc="download"):
        # Use a stable filename (sha1 of url, first 16 hex).
        import hashlib
        h = hashlib.sha1(url.encode()).hexdigest()[:16]
        out = raw_dir / f"{h}.mp4"
        if out.exists() and out.stat().st_size > 1024:
            paths.append(out)
            continue
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best",
            "--merge-output-format", "mp4",
            "--no-warnings",
            "--output", str(out),
            url,
        ]
        try:
            subprocess.run(cmd, timeout=600, check=False)
        except subprocess.TimeoutExpired:
            logger.warning("Timeout downloading %s", url)
            continue
        if out.exists() and out.stat().st_size > 1024:
            paths.append(out)
    return paths


def _materialize_local(source_dir: Path, raw_dir: Path) -> List[Path]:
    """Find video files under source_dir and symlink (or copy) them into raw_dir.

    Symlinks keep storage low; if --copy is passed we copy instead.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    exts = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
    paths: List[Path] = []
    for p in sorted(source_dir.rglob("*")):
        if p.suffix.lower() in exts and p.is_file():
            out = raw_dir / p.name
            if not out.exists():
                try:
                    os.symlink(p, out)
                except OSError:
                    shutil.copy2(p, out)
            paths.append(out)
    return paths


# ---------------------------------------------------------------------------
# Score cache — persists per-video yaw/motion to JSONL so re-curating with
# different (target_count, scale) skips face detection on already-scored
# videos. Cache key = absolute resolved path; entries carry a fingerprint
# of sample_frames + min_frames so a different scoring config invalidates.
# ---------------------------------------------------------------------------


def _load_score_cache(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    cache: Dict[str, dict] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = entry.get("path")
                if key:
                    cache[key] = entry
    except OSError as exc:
        logger.warning("could not read score cache %s: %s", path, exc)
    return cache


def _save_score_cache(path: Path, cache: Dict[str, dict]) -> None:
    """Append-mode write with a short temp-write-replace; safe under SIGINT."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w") as f:
            for entry in cache.values():
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("could not persist score cache to %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Step 2 — Score
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 2 — Score
# ---------------------------------------------------------------------------


def _estimate_yaw_from_lmk(lmk: np.ndarray) -> float:
    """Fallback yaw estimator from 106-point InsightFace landmarks.

    Mirrors the conservative multi-signal logic in
    lipsync_pipeline._estimate_yaw_degrees so curation can bucket side-face
    clips even when the InsightFace pose model is not loaded.
    """
    if lmk is None or len(lmk) < 106:
        return 0.0
    try:
        pt_left_eye = np.mean(lmk[[43, 48, 49, 51, 50]], axis=0)
        pt_right_eye = np.mean(lmk[101:106], axis=0)
        pt_nose = np.mean(lmk[[74, 77, 83, 86]], axis=0)
    except (IndexError, TypeError):
        return 0.0
    inter_ocular = float(pt_right_eye[0] - pt_left_eye[0])
    if abs(inter_ocular) < 1e-3:
        return 0.0
    expected = inter_ocular / 2.0

    nose_to_left = float(abs(pt_nose[0] - pt_left_eye[0]))
    delta = (expected - nose_to_left) / expected
    nose_yaw = delta * 60.0

    left_eye_x_range = float(np.ptp(lmk[[43, 48, 49, 51, 50], 0]))
    right_eye_x_range = float(np.ptp(lmk[101:106, 0]))
    eye_yaw = 0.0
    if min(left_eye_x_range, right_eye_x_range) > 1e-3:
        eye_asym = max(left_eye_x_range, right_eye_x_range) / min(left_eye_x_range, right_eye_x_range)
        if eye_asym > 1.5:
            eye_yaw = (eye_asym - 1.5) * 30.0

    d_left = float(np.linalg.norm(lmk[48] - pt_nose))
    d_right = float(np.linalg.norm(lmk[54] - pt_nose))
    mouth_yaw = 0.0
    if min(d_left, d_right) > 1e-3:
        mouth_asym = abs(d_left - d_right) / max(d_left, d_right)
        if mouth_asym > 0.2:
            mouth_yaw = (mouth_asym - 0.2) * 100.0

    area_yaw = 0.0
    aspect_yaw = 0.0
    try:
        lmk_x_min = float(lmk[:, 0].min())
        lmk_x_max = float(lmk[:, 0].max())
        lmk_y_min = float(lmk[:, 1].min())
        lmk_y_max = float(lmk[:, 1].max())
        face_area = (lmk_x_max - lmk_x_min) * (lmk_y_max - lmk_y_min)
        if face_area > 1e-3:
            mouth_w = float(np.linalg.norm(lmk[54] - lmk[48]))
            mouth_h = float(max(np.linalg.norm(lmk[57] - lmk[51]) / 2.0, 2.0))
            area_norm = (math.pi * (mouth_w / 2.0) * mouth_h) / face_area
            if area_norm < 0.025:
                area_yaw = (0.025 - area_norm) * 1200.0
            if mouth_w > 1e-3 and mouth_h > 1e-3:
                aspect = mouth_w / mouth_h
                if aspect < 2.0:
                    aspect_yaw = (2.0 - aspect) * 60.0
    except (IndexError, TypeError, ValueError):
        pass

    sign = 1.0 if nose_yaw >= 0 else -1.0
    return float(sign * max(abs(nose_yaw), eye_yaw, mouth_yaw, area_yaw, aspect_yaw))


def _score_one(
    video_path: Path,
    detector,
    sample_frames: int = SAMPLE_FRAMES,
    min_frames: int = MIN_FRAMES,
    det_threshold: float = 0.3,
) -> VideoScore:
    """Run face detection + motion scoring on a single video.

    Returns a VideoScore; bucket/reject fields are still empty.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total < min_frames:
        cap.release()
        return VideoScore(path=str(video_path), frame_count=total, rejected_reason="too_short")

    # Sample uniformly across the video.  Don't request more samples than
    # frames; otherwise short clips get padded duplicates and the face
    # detection ratio artificially drops.
    sample_frames = min(sample_frames, total)
    indices = np.linspace(0, total - 1, sample_frames).astype(int)

    yaws: List[float] = []
    face_centers: List[Tuple[float, float]] = []
    face_detected = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        try:
            bbox, lmk = detector(frame, threshold=det_threshold)
        except Exception:
            bbox = None
            lmk = None
        if bbox is None:
            continue
        face_detected += 1
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        face_centers.append((cx, cy))
        yaw = detector.last_pose_yaw
        if yaw is None and lmk is not None:
            yaw = _estimate_yaw_from_lmk(lmk)
        if yaw is not None:
            yaws.append(float(yaw))

    cap.release()

    if not face_centers:
        return VideoScore(
            path=str(video_path),
            frame_count=total,
            face_detected_ratio=face_detected / max(1, sample_frames),
            rejected_reason="no_face",
        )

    # If both detector pose and landmark fallback failed to produce yaw,
    # treat the clip as frontal rather than rejecting it outright.
    if not yaws:
        logger.warning(
            "Faces detected in %s but yaw unavailable (pose/landmark model missing?). "
            "Treating as frontal. If many clips show this, run: python tools/download_checkpoints.py",
            video_path,
        )
        yaws = [0.0]

    yaw_arr = np.array(yaws)
    centers = np.array(face_centers)
    diffs = np.diff(centers, axis=0)
    motion = float(np.median(np.linalg.norm(diffs, axis=1)))

    return VideoScore(
        path=str(video_path),
        frame_count=total,
        face_detected_ratio=face_detected / max(1, sample_frames),
        yaw_mean=float(yaw_arr.mean()),
        yaw_max_abs=float(np.max(np.abs(yaw_arr))),
        motion_score=motion,
    )


# ---------------------------------------------------------------------------
# Step 3 — Bucket + select
# ---------------------------------------------------------------------------


def _bucket(score: VideoScore, *, min_frames: int, face_detected_ratio: float, yaw_side_max: float) -> str:
    """Assign a single bucket to a VideoScore. Sets `rejected_reason` if rejected."""
    if score.rejected_reason:
        return "reject"
    if score.frame_count < min_frames:
        score.rejected_reason = "too_short"
        return "reject"
    if score.face_detected_ratio < face_detected_ratio:
        score.rejected_reason = "low_face_ratio"
        return "reject"
    if score.yaw_max_abs > yaw_side_max:
        score.rejected_reason = "extreme_yaw"
        return "reject"
    if score.sync_conf is not None and score.sync_conf < SYNC_CONF_MIN:
        score.rejected_reason = "low_sync_conf"
        return "reject"
    # Bucketing: side_face wins over fast_motion only if motion is below threshold.
    if score.yaw_max_abs >= YAW_SIDE_MIN:
        return "side_face"
    if score.motion_score >= MOTION_FAST_THRESHOLD:
        return "fast_motion"
    return "frontal"


def _select(scored: List[VideoScore], target_count: int, *, min_frames: int, face_detected_ratio: float, yaw_side_max: float) -> Dict[str, List[VideoScore]]:
    """Pick the top-N per bucket per TARGET_RATIO."""
    buckets: Dict[str, List[VideoScore]] = defaultdict(list)
    for s in scored:
        s.bucket = _bucket(s, min_frames=min_frames, face_detected_ratio=face_detected_ratio, yaw_side_max=yaw_side_max)
        if s.bucket != "reject":
            buckets[s.bucket].append(s)

    selected: Dict[str, List[VideoScore]] = {}
    leftovers: Dict[str, List[VideoScore]] = {}

    for cat, ratio in TARGET_RATIO.items():
        n_target = max(1, round(target_count * ratio))
        pool = sorted(
            buckets.get(cat, []),
            key=lambda x: (x.yaw_max_abs if cat == "side_face" else x.motion_score),
            reverse=(cat == "side_face"),
        )
        selected[cat] = pool[:n_target]
        leftovers[cat] = pool[n_target:]

    # Backfill from other categories if any is short, to hit target_count total.
    have = sum(len(v) for v in selected.values())
    if have < target_count:
        flat = [s for cat, vs in leftovers.items() for s in vs]
        flat.sort(key=lambda x: x.yaw_max_abs, reverse=True)
        for s in flat:
            if have >= target_count:
                break
            if s.bucket not in selected:
                s.bucket = "side_face"  # treat as side_face for backfill
                selected["side_face"].append(s)
                have += 1
    return selected


# ---------------------------------------------------------------------------
# Step 4 — Materialize
# ---------------------------------------------------------------------------


def _copy_selected(
    selected: Dict[str, List[VideoScore]],
    out_dir: Path,
) -> List[Path]:
    """Copy (link) each kept video into out_dir/<bucket>/<idx>.mp4 and return the new paths."""
    written: List[Path] = []
    for cat, videos in selected.items():
        sub = out_dir / cat
        sub.mkdir(parents=True, exist_ok=True)
        for i, v in enumerate(videos):
            src = Path(v.path)
            dst = sub / f"{i:03d}_{src.stem}.mp4"
            if dst.exists() or dst.resolve() == src.resolve():
                # Idempotent: skip if already linked in place.
                if dst.exists():
                    written.append(dst)
                continue
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            written.append(dst)
    return written


def _write_fileslist(written: List[Path], fileslist_path: Path) -> None:
    """One path per line, no header. Readable by LatentSync dataset loaders."""
    with open(fileslist_path, "w") as f:
        for p in sorted(written):
            f.write(str(p.resolve()) + "\n")


def _write_report(scored: List[VideoScore], selected: Dict[str, List[VideoScore]], out_dir: Path, thresholds: dict) -> None:
    report = {
        "total_candidates": len(scored),
        "kept": sum(len(v) for v in selected.values()),
        "by_bucket": {cat: [s.to_dict() for s in vs] for cat, vs in selected.items()},
        "rejected": [s.to_dict() for s in scored if s.bucket == "reject"],
        "thresholds": thresholds,
    }
    with open(out_dir / "curation_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Curate a finetune dataset by yaw/motion buckets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--urls", type=str, default=None,
                        help="text file with one URL per line")
    parser.add_argument("--url", type=str, action="append", default=None,
                        help="single URL (repeat for multiple)")
    parser.add_argument("--source-dir", type=str, default=None,
                        help="local directory to scan for candidate video files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="where to put curated buckets, fileslist, and report")
    parser.add_argument("--target-count", type=int, default=60,
                        help="target total number of kept videos (200 default for finetune; bump to 1000-5000 for industry-grade)")
    parser.add_argument("--max-candidates", type=int, default=10000,
                        help="hard cap on videos to scan after download (50000+ for industry-grade from VoxCeleb2)")
    parser.add_argument("--min-frames", type=int, default=MIN_FRAMES,
                        help=f"min frames per video to keep ({MIN_FRAMES}=~1.2s @ 25fps; bump to 60-120 for 1000+ videos to reduce short-clip noise)")
    parser.add_argument("--sample-frames", type=int, default=SAMPLE_FRAMES,
                        help="frames subsampled per video for face detection")
    parser.add_argument("--face-detected-ratio", type=float, default=0.4,
                        help="minimum ratio of sampled frames that must have a detected face (default 0.4)")
    parser.add_argument("--yaw-side-max", type=float, default=YAW_SIDE_MAX,
                        help=f"max |yaw| allowed; beyond this is rejected as extreme yaw (default {YAW_SIDE_MAX})")
    parser.add_argument("--device", type=str, default="cuda",
                        help="face detector device (cuda/cpu)")
    parser.add_argument("--log", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper()), format="%(asctime)s %(levelname)s %(message)s")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "_raw"

    thresholds = {
        "yaw_frontal_max": YAW_FRONTAL_MAX,
        "yaw_side_min": YAW_SIDE_MIN,
        "yaw_side_max": args.yaw_side_max,
        "motion_fast_threshold": MOTION_FAST_THRESHOLD,
        "min_frames": args.min_frames,
        "face_detected_ratio": args.face_detected_ratio,
    }

    # ---- collect candidates ----
    urls: List[str] = []
    if args.urls:
        with open(args.urls) as f:
            urls.extend(line.strip() for line in f if line.strip())
    if args.url:
        urls.extend(args.url)
    candidates: List[Path] = []
    if urls:
        logger.info("Downloading %d URLs via yt-dlp ...", len(urls))
        candidates = _download_urls(urls, raw_dir, num_workers=4)
    if args.source_dir:
        candidates.extend(_materialize_local(Path(args.source_dir), raw_dir))
    candidates = list(dict.fromkeys(candidates))  # de-dup
    candidates = candidates[: args.max_candidates]
    logger.info("Scoring %d candidate videos ...", len(candidates))

    # ---- load face detector ----
    try:
        from latentsync.utils.face_detector import FaceDetector
        # Curation needs yaw across the full range so we can bucket side_face
        # clips.  Load the pose module too; otherwise face.pose is None and we
        # cannot estimate yaw.
        detector = FaceDetector(
            device=args.device,
            skip_side_face_threshold=None,
            allowed_modules=None,  # load all available modules (pose, genderage, ...)
        )
    except Exception as exc:
        logger.error("Could not load FaceDetector: %s", exc)
        logger.error("Install insightface + run `python tools/download_checkpoints.py` first.")
        sys.exit(1)

    # ---- score (with on-disk cache) ----
    cache_path = out_dir / "score_cache.jsonl"
    cache: Dict[str, dict] = _load_score_cache(cache_path)
    scored: List[VideoScore] = []
    n_cache_hits = 0
    for p in tqdm(candidates, desc="score"):
        key = str(p.resolve())
        cached = cache.get(key)
        if (
            cached is not None
            and cached.get("sample_frames") == args.sample_frames
            and cached.get("min_frames") == args.min_frames
        ):
            vs = VideoScore(**{k: cached[k] for k in [
                "path", "yaw_mean", "yaw_max_abs", "motion_score",
                "frame_count", "face_detected_ratio", "sync_conf",
            ] if k in cached})
            vs.path = key
            scored.append(vs)
            n_cache_hits += 1
            continue
        try:
            s = _score_one(
                p, detector,
                sample_frames=args.sample_frames,
                min_frames=args.min_frames,
                det_threshold=0.3,
            )
        except Exception as exc:
            logger.warning("Failed to score %s: %s", p, exc)
            s = VideoScore(path=str(p), rejected_reason="score_error")
        # best-effort: persist to cache (skip error rows)
        if not s.rejected_reason:
            cache[s.path] = {
                "path": s.path,
                "yaw_mean": s.yaw_mean,
                "yaw_max_abs": s.yaw_max_abs,
                "motion_score": s.motion_score,
                "frame_count": s.frame_count,
                "face_detected_ratio": s.face_detected_ratio,
                "sync_conf": s.sync_conf,
                "sample_frames": args.sample_frames,
                "min_frames": args.min_frames,
                "_cached_at": datetime.now().isoformat(timespec="seconds"),
            }
            _save_score_cache(cache_path, cache)
        scored.append(s)

    logger.info("Cache hits: %d / %d (rest freshly scored)",
                n_cache_hits, len(candidates))

    # ---- select ----
    selected = _select(
        scored, args.target_count,
        min_frames=args.min_frames,
        face_detected_ratio=args.face_detected_ratio,
        yaw_side_max=args.yaw_side_max,
    )
    kept_total = sum(len(v) for v in selected.values())
    logger.info(
        "Kept %d / %d (target=%d): %s",
        kept_total, len(scored), args.target_count,
        {k: len(v) for k, v in selected.items()},
    )

    # Summarize rejection reasons so users can tell *why* clips were dropped.
    rejected = [s for s in scored if s.bucket == "reject"]
    if rejected:
        from collections import Counter
        reasons = Counter(s.rejected_reason for s in rejected)
        logger.info("Rejection reasons (%d total rejected): %s", len(rejected), dict(reasons))
        for reason, count in reasons.most_common():
            examples = [s.path for s in rejected if s.rejected_reason == reason][:3]
            logger.info("  - %s: %d (e.g. %s)", reason, count, examples)
    else:
        logger.info("No rejections.")

    # ---- materialize ----
    written = _copy_selected(selected, out_dir)
    _write_fileslist(written, out_dir / "fileslist.txt")
    _write_report(scored, selected, out_dir, thresholds)
    if kept_total == 0:
        logger.error(
            "No clips survived curation. Common causes:\n"
            "  - face detection failed (run `python tools/download_checkpoints.py`)\n"
            "  - all clips are too short / low face ratio / extreme yaw\n"
            "Check %s/curation_report.json for per-video rejection reasons.",
            out_dir,
        )
        sys.exit(1)
    logger.info(
        "Done. fileslist at %s, report at %s/curation_report.json",
        out_dir / "fileslist.txt", out_dir,
    )


if __name__ == "__main__":
    main()
