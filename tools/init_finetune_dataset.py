# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

"""Pre-built finetune dataset initializer.

Reads tools/prebuilt_datasets.yaml, downloads a sampled subset from
HuggingFace Hub, runs the LatentSync face detector, curates into
yaw/motion buckets, and writes a fileslist.txt ready to plug into
gradio_finetune Tab 1.

Output layout (one dataset at <output-dir>/<id>/):
    <id>/
        _raw/<hf-original-filename>.mp4       # what was downloaded
        frontal/000_<stem>.mp4                # curated, symlinked
        side_face/000_<stem>.mp4
        fast_motion/000_<stem>.mp4
        fileslist.txt                         # all curated .mp4, one per line
        curation_report.json                  # per-video scores + buckets

Usage:
    # 1. List available prebuilt recipes
    python tools/init_finetune_dataset.py --list

    # 2. Initialize one (download + curate; progress to stdout)
    python tools/init_finetune_dataset.py \\
        --dataset celebv_hq_side \\
        --output-dir data/init_finetune

    # 3. Initialize all (long; downloads GBs)
    python tools/init_finetune_dataset.py --dataset all --output-dir data/init_finetune

Notes:
    - Public HF Hub repos only; no token needed.
    - Downloads are scoped via allow_patterns and capped at n_clips so
      we don't pull the full ~300GB VoxCeleb2 just for 1k clips.
    - Re-running is idempotent: skips videos already present in _raw.
    - Curation is short — re-uses the score cache from
      curate_finetune_samples.py when present.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# HuggingFace Hub 从 0.26 开始默认走 Xet 存储后端。Xet 对匿名/部分网络
# 环境会偶发 401 (cas-server.xethub.hf.co)，且并发下载容易触发 CAS 错误。
# 强制回退到普通 HTTP/LFS 可显著提升预置数据集下载成功率。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from einops import rearrange

from latentsync.utils.affine_transform import AlignRestore
from latentsync.utils.face_detector import FaceDetector
from latentsync.utils.util import write_video_via_ffmpeg

logger = logging.getLogger(__name__)

DEFAULT_YAML = REPO_ROOT / "tools" / "prebuilt_datasets.yaml"

# Curation defaults (mirrored from curate_finetune_samples.py)
DEFAULT_TARGET_RATIO = {"frontal": 0.45, "side_face": 0.35, "fast_motion": 0.20}


def _load_recipes(yaml_path: Path) -> Dict[str, dict]:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}
    recipes = {}
    for entry in raw.get("datasets", []):
        if "id" not in entry:
            continue
        recipes[entry["id"]] = entry
    return recipes


def _list_recipes(yaml_path: Path) -> List[str]:
    recipes = _load_recipes(yaml_path)
    print(f"\nAvailable pre-built datasets ({len(recipes)}):")
    for k, v in recipes.items():
        print(f"  - {k}: {v.get('name', '(unnamed)')}")
        if v.get("description"):
            for line in v["description"].strip().split("\n"):
                print(f"      {line.strip()}")
    print(f"\nRun: python tools/init_finetune_dataset.py --dataset <id>")
    return list(recipes.keys())


def _list_repo_files_paginated(
    *,
    hf_repo: str,
    repo_type: str,
    allow_patterns: List[str],
    hf_token: Optional[str] = None,
) -> List[str]:
    """List repo files by walking each allow-pattern top-level dir via the
    tree API with cursor pagination.

    Fallback for large repos where list_repo_files(recursive=True) makes the
    gateway return 504 (e.g. hf-mirror.com on a ~23k-file repo). The tree API
    caps each page at ~1000 entries; we follow the `cursor` Link header to
    enumerate a single directory fully, one dir at a time, so each HTTP
    response stays small.

    Only the top-level directory of each allow_pattern is walked (e.g.
    "lip_sync/*.mp4" -> walk "lip_sync"). This matches how the recipes use
    allow_patterns (flat per-dir globs); deep recursive globs are not handled.
    """
    import re
    import requests

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    headers = {"authorization": f"Bearer {hf_token}"} if hf_token else {}
    repo_type_plural = "datasets" if repo_type == "dataset" else repo_type

    # Derive the set of top-level dirs to walk from allow_patterns.
    dirs: List[str] = []
    for pat in allow_patterns:
        top = pat.split("/", 1)[0]
        if top and top not in dirs:
            dirs.append(top)

    all_files: List[str] = []
    import time as _time
    for d in dirs:
        cursor: Optional[str] = None
        n_pages = 0
        while True:
            url = f"{endpoint}/api/{repo_type_plural}/{hf_repo}/tree/main/{d}"
            params: Dict[str, str] = {}
            if cursor:
                params["cursor"] = cursor
            # hf-mirror.com intermittently 504s on tree pages; retry with
            # backoff so a transient gateway timeout doesn't kill the whole walk.
            r = None
            for attempt in range(5):
                try:
                    r = requests.get(url, headers=headers, params=params, timeout=60)
                    if r.status_code != 504:
                        break
                except Exception as exc:
                    last = exc
                r = None
                backoff = 3 * (2 ** attempt)
                logger.warning(
                    "tree %s:%s page %d attempt %d/5 failed (%s); retry in %ds",
                    hf_repo, d, n_pages + 1, attempt + 1,
                    r.status_code if r is not None else "exc", backoff,
                )
                _time.sleep(backoff)
            if r is None or r.status_code != 200:
                raise RuntimeError(
                    f"tree request for {hf_repo}:{d} returned "
                    f"{r.status_code if r is not None else 'exc'} after retries"
                )
            n_pages += 1
            entries = r.json()
            for entry in entries:
                path = entry.get("path") or entry.get("rfilename")
                if path and entry.get("type") != "directory":
                    all_files.append(path)
            # Follow pagination via the Link header, if present.
            link = r.headers.get("Link", "")
            m = re.search(r"[?&]cursor=([^&;>]+)", link)
            if m:
                cursor = m.group(1)
            else:
                break
    return all_files


def _download_hf_subset(
    *,
    hf_repo: str,
    repo_type: str,
    allow_patterns: List[str],
    target_dir: Path,
    max_files: int,
    hf_token: Optional[str] = None,
) -> List[Path]:
    """Download up to max_files matching allow_patterns from an HF dataset repo.

    Idempotent: re-running skips files already present and non-empty in
    target_dir. Returns the absolute paths of the downloaded files.

    For gated datasets (VoxCeleb2 mirror, CelebV-HQ gated splits, etc.),
    pass `hf_token` (or set the HF_TOKEN / HUGGINGFACE_TOKEN env var).
    """
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is required: pip install huggingface_hub"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    api_kwargs = {"token": hf_token} if hf_token else {}
    try:
        from huggingface_hub import HfApi
        api = HfApi(**api_kwargs)
        all_files = api.list_repo_files(hf_repo, repo_type=repo_type)
    except Exception as exc:
        # Large repos (e.g. ~23k files on hf-mirror.com) often 504 on the
        # recursive tree listing. Fall back to paginated per-directory walk.
        logger.warning(
            "list_repo_files failed for %s (%s); trying paginated per-dir walk ...",
            hf_repo, type(exc).__name__,
        )
        try:
            all_files = _list_repo_files_paginated(
                hf_repo=hf_repo, repo_type=repo_type,
                allow_patterns=allow_patterns, hf_token=hf_token,
            )
        except Exception as exc2:
            raise SystemExit(
                f"❌ cannot list files for {hf_repo}: {exc}\n"
                f"   paginated fallback also failed: {exc2}\n"
                "   If this is a gated repo, pass --hf-token or set HF_TOKEN. "
                "If using a mirror, check HF_ENDPOINT / network."
            )

    # Filter to allow_patterns matches (simple glob via fnmatch).
    import fnmatch
    matched = [f for f in all_files if any(fnmatch.fnmatch(f, p) for p in allow_patterns)]

    if not matched:
        raise SystemExit(
            f"❌ no files matched allow_patterns={allow_patterns} in {hf_repo}. "
            "Check the repo id and patterns."
        )

    expected_files = matched[:max_files]
    archive_suffixes = {".tar", ".tar.gz", ".tgz", ".tbz2", ".zip"}

    def _is_archive(path: str) -> bool:
        return any(path.lower().endswith(suffix) for suffix in archive_suffixes)

    def _local_path(repo_path: str) -> Path:
        # snapshot_download / hf_hub_download preserve the repo directory
        # structure under local_dir.
        return target_dir / repo_path

    def _file_ready(path: Path) -> bool:
        return path.exists() and path.stat().st_size > 0

    # Decide what actually needs downloading.
    missing: List[str] = []
    archives_to_extract: List[Path] = []
    ready_files: List[Path] = []

    for repo_path in expected_files:
        local = _local_path(repo_path)
        if _is_archive(repo_path):
            # Already extracted mp4s present? Then skip both download & extract.
            if _file_ready(local) and any(
                p.suffix.lower() == ".mp4" and p.stat().st_size > 0
                for p in target_dir.rglob("*")
                if p != local
            ):
                logger.info("archive %s already extracted, skipping", repo_path)
                continue
            if not _file_ready(local):
                missing.append(repo_path)
            else:
                archives_to_extract.append(local)
        else:
            if _file_ready(local):
                logger.info("file already exists, skipping: %s", repo_path)
                ready_files.append(local)
            else:
                missing.append(repo_path)

    n_ready = len(ready_files) + len(archives_to_extract)
    logger.info(
        "%d/%d files already present; %d need download",
        n_ready, len(expected_files), len(missing),
    )

    if missing:
        logger.info("downloading %d files from %s ...", len(missing), hf_repo)
        # Retry snapshot_download with exponential backoff. 并发下载容易触发
        # Xet CAS 401/RuntimeError，所以 max_workers=1；若仍失败则逐文件 fallback。
        last_exc = None
        for attempt in range(3):
            try:
                snapshot_download(
                    repo_id=hf_repo,
                    repo_type=repo_type,
                    local_dir=str(target_dir),
                    allow_patterns=missing,
                    max_workers=1,
                    token=hf_token,
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                backoff = 5 * (2 ** attempt)
                logger.warning(
                    "snapshot_download attempt %d/3 failed (%s); retrying in %ds ...",
                    attempt + 1, type(exc).__name__, backoff,
                )
                import time as _time
                _time.sleep(backoff)

        # Fallback: 如果 snapshot_download 整体失败，逐文件 hf_hub_download。
        # 配合 HF_HUB_DISABLE_XET=1 后，这通常能绕过 Xet CAS 的并发/鉴权问题。
        if last_exc is not None:
            logger.warning(
                "snapshot_download failed for %s; falling back to per-file download ...",
                hf_repo,
            )
            from huggingface_hub import hf_hub_download
            failed_files: List[str] = []
            for fname in missing:
                f_last_exc = None
                for attempt in range(3):
                    try:
                        hf_hub_download(
                            repo_id=hf_repo,
                            repo_type=repo_type,
                            filename=fname,
                            local_dir=str(target_dir),
                            token=hf_token,
                        )
                        f_last_exc = None
                        break
                    except Exception as exc:
                        f_last_exc = exc
                        backoff = 5 * (2 ** attempt)
                        logger.warning(
                            "hf_hub_download %s attempt %d/3 failed (%s); retrying in %ds ...",
                            fname, attempt + 1, type(exc).__name__, backoff,
                        )
                        import time as _time
                        _time.sleep(backoff)
                if f_last_exc is not None:
                    failed_files.append(fname)
                    logger.error("hf_hub_download failed for %s: %s", fname, f_last_exc)
            if failed_files:
                raise SystemExit(
                    f"❌ download failed for {hf_repo} after per-file fallback. "
                    f"Failed files ({len(failed_files)}/{len(missing)}): {failed_files[:10]}{'...' if len(failed_files) > 10 else ''}\n"
                    f"Original snapshot_download error: {last_exc}\n"
                    "   If this repo is gated or Xet keeps failing, pass --hf-token or set HF_TOKEN."
                )

        # Re-evaluate which archives need extraction after download.
        for repo_path in expected_files:
            if not _is_archive(repo_path):
                continue
            local = _local_path(repo_path)
            if _file_ready(local) and not any(
                p.suffix.lower() == ".mp4" and p.stat().st_size > 0
                for p in target_dir.rglob("*")
                if p != local
            ):
                archives_to_extract.append(local)

    # Extract archives (idempotent: only if no extracted mp4s were found).
    if archives_to_extract:
        logger.info("extracting archives: %s", [a.name for a in archives_to_extract])
        for arc in archives_to_extract:
            _extract_archive(arc, target_dir)

    # Collect final file list.
    downloaded: List[Path] = []
    for repo_path in expected_files:
        local = _local_path(repo_path)
        if _is_archive(repo_path):
            # Return the extracted mp4s instead of the archive itself.
            for p in target_dir.rglob("*.mp4"):
                if p.stat().st_size > 0:
                    downloaded.append(p)
        else:
            if _file_ready(local):
                downloaded.append(local)

    # De-duplicate and sort for stable output.
    downloaded = sorted(set(p.resolve() for p in downloaded))
    logger.info("ready: %d files under %s", len(downloaded), target_dir)
    return downloaded


def _extract_archive(archive: Path, dest: Path) -> None:
    """Extract .tar / .tar.gz / .tgz / .tbz2 / .zip into `dest`."""
    import tarfile
    import zipfile
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    try:
        if name.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                z.extractall(dest)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as t:
                t.extractall(dest)
        else:
            logger.warning("unknown archive format, skipping: %s", archive)
    except Exception as exc:
        raise SystemExit(f"❌ failed to extract {archive}: {exc}")


def _run_curation(
    *,
    source_dir: Path,
    output_dir: Path,
    target_count: int,
    ratio: Dict[str, float],
    curate_args: Optional[Dict[str, Any]] = None,
) -> int:
    """Shell out to curate_finetune_samples.py with the right defaults.

    Returns its return code.
    """
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "curate_finetune_samples.py"),
        "--source-dir", str(source_dir),
        "--output-dir", str(output_dir),
        "--target-count", str(target_count),
    ]
    curate_args = curate_args or {}
    for key, val in curate_args.items():
        arg_name = f"--{key.replace('_', '-')}"
        cmd += [arg_name, str(val)]
    logger.info("running: %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _print_paste_able(recipe_id: str, recipe: dict, output_dir: Path, fileslist: Path) -> None:
    """Print copy-paste-able block for gradio Tab 1."""
    print()
    print("=" * 72)
    print(f"✅ Pre-built dataset ready: {recipe.get('name', recipe_id)}")
    print("=" * 72)
    print(f"   output-dir: {output_dir}")
    print(f"   fileslist:  {fileslist}")
    print()
    print("📋 Paste into gradio_finetune Tab 1:")
    print()
    print(f"   train_data_dir:   {output_dir.resolve()}")
    print(f"   train_fileslist:  {fileslist.resolve()}")
    if recipe.get("typical_use"):
        print(f"   preset (typical): {recipe['typical_use']}")
    print()
    print("   然后选 preset + 🚀 启动训练.")
    print("=" * 72)


def _read_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return fps


def _mux_audio(src_video: str, dst_video: str) -> None:
    """Copy audio stream from src_video into dst_video (overwrites dst_video)."""
    tmp_path = dst_video + ".audio_mux.tmp.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-nostdin",
        "-i", src_video,
        "-i", dst_video,
        "-c:v", "copy",
        "-c:a", "copy",
        "-map", "1:v:0",
        "-map", "0:a:0?",
        "-shortest",
        tmp_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return
    os.replace(tmp_path, dst_video)


def _align_one_video(
    src_path: Path,
    dst_path: Path,
    detector: FaceDetector,
    restorer: AlignRestore,
    resolution: int,
    det_threshold: float,
    max_fail_ratio: float,
) -> bool:
    """Align faces in one video. Returns True if output was written."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        vr = VideoReader(str(src_path), ctx=cpu(0))
        total = len(vr)
        if total < 30:
            logger.warning("skip %s: too short (%d frames)", src_path, total)
            return False
        frames = vr[:].asnumpy()
        vr.seek(0)
    except Exception as exc:
        logger.warning("failed to read %s: %s", src_path, exc)
        return False

    restorer.p_bias = None
    aligned_frames: List[np.ndarray] = []
    last_face: Optional[np.ndarray] = None
    n_fail = 0

    for frame in frames:
        try:
            bbox, lmk = detector(frame, threshold=det_threshold)
            if bbox is None or lmk is None:
                raise RuntimeError("face not detected")

            pt_left_eye = np.mean(lmk[[43, 48, 49, 51, 50]], axis=0)
            pt_right_eye = np.mean(lmk[101:106], axis=0)
            pt_nose = np.mean(lmk[[74, 77, 83, 86]], axis=0)
            landmarks3 = np.array([pt_left_eye, pt_right_eye, pt_nose])

            face, _ = restorer.align_warp_face(frame.copy(), landmarks3=landmarks3, smooth=True)
            face = cv2.resize(face, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
            last_face = face
        except Exception:
            n_fail += 1
            if last_face is None:
                # Keep looking for a usable first face rather than dropping
                # the whole clip on a single bad first frame.
                continue
            face = last_face.copy()

        aligned_frames.append(face)

    # If we never got a single usable face, the clip is unusable.
    if last_face is None:
        logger.warning("skip %s: no face detected in any frame", src_path)
        return False

    # Pad leading frames that were skipped while searching for the first face.
    n_leading_missing = total - len(aligned_frames)
    if n_leading_missing > 0:
        aligned_frames[:0] = [last_face.copy()] * n_leading_missing
        # These leading frames were already counted in n_fail above.

    # curation already gated on quality (face_detected_ratio / yaw / mouth),
    # so alignment should be a format conversion, not a second quality filter.
    # Side-face / head-turn clips legitimately fail InsightFace's frontal
    # detector on most frames; padding those misses with the last detected
    # (same-pose) frame is the normal handling. Only drop a clip that never
    # produced a single usable face. Log the fail ratio so heavy-padding
    # clips are still visible in the logs.
    fail_ratio = n_fail / total
    if fail_ratio > max_fail_ratio:
        logger.warning(
            "high fail ratio %.2f on %s (threshold %.2f); keeping — "
            "curated quality gate already passed, padding missed frames",
            fail_ratio, src_path, max_fail_ratio,
        )

    if len(aligned_frames) != total:
        logger.warning("skip %s: frame count mismatch", src_path)
        return False

    aligned_stack = np.stack(aligned_frames)
    fps = _read_video_fps(src_path)
    try:
        write_video_via_ffmpeg(str(dst_path), aligned_stack, fps=int(fps), crf=13)
        _mux_audio(str(src_path), str(dst_path))
    except Exception as exc:
        logger.warning("failed to write %s: %s", dst_path, exc)
        if os.path.exists(dst_path):
            os.unlink(dst_path)
        return False

    return True


def _preprocess_aligned(
    curated_dir: Path,
    aligned_dir: Path,
    *,
    resolution: int = 512,
    device: str = "cuda",
    det_threshold: float = 0.3,
    max_fail_ratio: float = 0.2,
    side_face_max_fail_ratio: Optional[float] = None,
) -> int:
    """Run face alignment on curated videos and write fixed-res outputs.

    `side_face_max_fail_ratio` (if set) overrides `max_fail_ratio` for clips
    in the side_face bucket. Side-face clips inherently fail InsightFace's
    frontal-biased detector on many frames even though the face is still
    present in roughly the same pose, so copying the last detected (side-face)
    frame for missed frames is geometrically defensible — unlike frontal
    clips, where a high fail ratio means the face really isn't there.
    """
    aligned_dir.mkdir(parents=True, exist_ok=True)

    # Only process the curated bucket directories (frontal / side_face /
    # fast_motion). Ignore any _raw/ or aligned/ sub-dirs that may have
    # been left behind by manual moves or older runs.
    bucket_names = ("frontal", "side_face", "fast_motion")
    src_paths = sorted(
        p
        for name in bucket_names
        for p in (curated_dir / name).rglob("*.mp4")
        if p.is_file()
    )
    if not src_paths:
        logger.warning("no curated videos found under %s", curated_dir)
        return 0

    logger.info(
        "aligning %d curated videos to %dx%d on %s "
        "(side_face max_fail_ratio=%s, others=%s) ...",
        len(src_paths), resolution, resolution, device,
        side_face_max_fail_ratio if side_face_max_fail_ratio is not None else max_fail_ratio,
        max_fail_ratio,
    )

    detector = FaceDetector(device=device, skip_side_face_threshold=None)
    restorer = AlignRestore(resolution=resolution, device=device, dtype=torch.float32)

    kept = 0
    for src in src_paths:
        rel = src.relative_to(curated_dir)
        dst = aligned_dir / rel
        bucket = rel.parts[0] if rel.parts else ""
        fail_ratio = (
            side_face_max_fail_ratio
            if bucket == "side_face" and side_face_max_fail_ratio is not None
            else max_fail_ratio
        )
        if _align_one_video(src, dst, detector, restorer, resolution, det_threshold, fail_ratio):
            kept += 1
            logger.info("aligned %s", rel)

    logger.info("alignment done: kept %d / %d", kept, len(src_paths))
    return kept


def _write_fileslist_from_dir(written: List[Path], fileslist_path: Path) -> None:
    """One path per line, no header. Readable by LatentSync dataset loaders."""
    with open(fileslist_path, "w") as f:
        for p in sorted(written):
            f.write(str(p.resolve()) + "\n")


def init_one(
    recipe_id: str,
    recipe: dict,
    *,
    output_root: Path,
    n_clips_override: Optional[int] = None,
    hf_token: Optional[str] = None,
    align: bool = True,
    align_resolution: int = 512,
    align_device: str = "cuda",
    align_det_threshold: float = 0.3,
    align_max_fail_ratio: float = 0.2,
) -> Path:
    """Initialize one prebuilt dataset. Returns the output dir."""
    output_dir = output_root / recipe_id
    raw_dir = output_dir / "_raw"
    curated_dir = output_dir / "curated"
    aligned_dir = curated_dir / "aligned"

    n_clips = n_clips_override or recipe.get("n_clips", 1000)
    ratio = recipe.get("target_ratio") or DEFAULT_TARGET_RATIO

    print(f"\n[{recipe_id}] preparing {n_clips} clips from {recipe['hf_repo']} ...")
    raw_paths = _download_hf_subset(
        hf_repo=recipe["hf_repo"],
        repo_type=recipe.get("hf_repo_type", "dataset"),
        allow_patterns=recipe.get("hf_allow", ["**/*.mp4"]),
        target_dir=raw_dir,
        max_files=n_clips,
        hf_token=hf_token,
    )
    print(f"[{recipe_id}] ready {len(raw_paths)} files under {raw_dir} (already-present files are skipped)")

    if not raw_paths:
        raise SystemExit(f"❌ [{recipe_id}] download produced no files")

    print(f"[{recipe_id}] curating with ratio={ratio}, curate_args={recipe.get('curate_args', {})} ...")
    rc = _run_curation(
        source_dir=raw_dir,
        output_dir=curated_dir,
        target_count=n_clips,
        ratio=ratio,
        curate_args=recipe.get("curate_args"),
    )
    if rc != 0:
        raise SystemExit(f"❌ [{recipe_id}] curation failed (rc={rc})")

    # Optional face-alignment preprocessing. This turns raw curated videos
    # into fixed-resolution face crops so the UNetDataset can train with
    # affine_transform=False while still seeing aligned faces.
    # Per-recipe align_args can relax thresholds for hard datasets (e.g. side-face).
    if align:
        align_args = recipe.get("align_args", {})
        _preprocess_aligned(
            curated_dir=curated_dir,
            aligned_dir=aligned_dir,
            resolution=align_resolution,
            device=align_device,
            det_threshold=align_args.get("det_threshold", align_det_threshold),
            max_fail_ratio=align_args.get("max_fail_ratio", align_max_fail_ratio),
            side_face_max_fail_ratio=align_args.get("side_face_max_fail_ratio"),
        )

    # Decide which directory supplies the training files:
    # aligned videos win when they exist; otherwise fall back to curated.
    if aligned_dir.exists() and any(aligned_dir.rglob("*.mp4")):
        fileslist_dir = aligned_dir
        fileslist_source = aligned_dir
    else:
        fileslist_dir = curated_dir
        fileslist_source = curated_dir

    # Write a top-level fileslist.txt that points at the chosen directory,
    # so the user can drop it straight into gradio Tab 1.
    curated_fileslist = curated_dir / "fileslist.txt"
    top_fileslist = output_dir / "fileslist.txt"
    target_fileslist = fileslist_dir / "fileslist.txt"
    kept = 0
    if curated_fileslist.exists():
        # If we have aligned outputs, build the fileslist from aligned dir;
        # otherwise copy the existing curated fileslist.
        if fileslist_source == aligned_dir:
            aligned_paths = sorted(p for p in aligned_dir.rglob("*.mp4") if p.is_file())
            _write_fileslist_from_dir(aligned_paths, target_fileslist)
        kept = len([line for line in target_fileslist.read_text().splitlines() if line.strip()])
        if kept == 0:
            raise SystemExit(
                f"❌ [{recipe_id}] produced an empty fileslist.\n"
                f"   Likely face detection/alignment failed.\n"
                f"   Run: python tools/download_checkpoints.py\n"
                f"   Then retry, or pass --no-align to skip alignment."
            )
        shutil.copy2(target_fileslist, top_fileslist)

    _print_paste_able(recipe_id, recipe, fileslist_dir, target_fileslist)
    return fileslist_dir


def main():
    parser = argparse.ArgumentParser(
        description="Initialize a pre-built finetune dataset from HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true",
                        help="list available prebuilt recipes and exit")
    parser.add_argument("--dataset", type=str, default=None,
                        help="dataset id from prebuilt_datasets.yaml (or 'all')")
    # Default lands under the standard finetune base dir so data stays
    # on the data disk rather than under the repo working copy.
    _DEFAULT_OUTPUT_DIR = os.environ.get(
        "LATENTSYNC_FINETUNE_DIR", "/root/autodl-tmp/latentsync_finetune"
    )
    parser.add_argument("--output-dir", type=str,
                        default=f"{_DEFAULT_OUTPUT_DIR}/init_finetune",
                        help="root directory for downloaded + curated data")
    parser.add_argument("--recipes", type=str, default=str(DEFAULT_YAML),
                        help="path to prebuilt_datasets.yaml")
    parser.add_argument("--n-clips", type=int, default=None,
                        help="override the per-dataset default clip count")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token for gated datasets. Falls back to "
                             "HF_TOKEN / HUGGINGFACE_TOKEN env var.")
    parser.add_argument("--align", action=argparse.BooleanOptionalAction, default=True,
                        help="run face alignment after curation (default: True)")
    parser.add_argument("--align-resolution", type=int, default=512,
                        help="resolution of aligned output videos (default: 512)")
    parser.add_argument("--align-device", type=str, default="cuda",
                        help="device for face alignment (default: cuda)")
    parser.add_argument("--align-det-threshold", type=float, default=0.3,
                        help="InsightFace detection threshold for alignment (default: 0.3)")
    parser.add_argument("--align-max-fail-ratio", type=float, default=0.2,
                        help="max ratio of frames allowed to fail detection (default: 0.2)")
    parser.add_argument("--log", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    recipes_path = Path(args.recipes)
    if not recipes_path.exists():
        sys.exit(f"❌ recipes yaml not found: {recipes_path}")

    recipes = _load_recipes(recipes_path)

    if args.list:
        _list_recipes(recipes_path)
        return 0

    if not args.dataset:
        sys.exit("❌ --dataset is required (or use --list to see options)")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "all":
        for recipe_id in recipes:
            try:
                init_one(recipe_id, recipes[recipe_id],
                         output_root=output_root,
                         n_clips_override=args.n_clips,
                         hf_token=hf_token,
                         align=args.align,
                         align_resolution=args.align_resolution,
                         align_device=args.align_device,
                         align_det_threshold=args.align_det_threshold,
                         align_max_fail_ratio=args.align_max_fail_ratio)
            except SystemExit as exc:
                print(f"⚠️  {exc}", file=sys.stderr)
                continue
    else:
        if args.dataset not in recipes:
            sys.exit(f"❌ unknown dataset: {args.dataset}. Run --list.")
        init_one(args.dataset, recipes[args.dataset],
                 output_root=output_root,
                 n_clips_override=args.n_clips,
                 hf_token=hf_token,
                 align=args.align,
                 align_resolution=args.align_resolution,
                 align_device=args.align_device,
                 align_det_threshold=args.align_det_threshold,
                 align_max_fail_ratio=args.align_max_fail_ratio)

    return 0


if __name__ == "__main__":
    sys.exit(main())