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
        raise SystemExit(
            f"❌ cannot list files for {hf_repo}: {exc}\n"
            "   If this is a gated repo, pass --hf-token or set HF_TOKEN."
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


def _print_paste_able(recipe_id: str, recipe: dict, output_dir: Path) -> None:
    """Print copy-paste-able block for gradio Tab 1."""
    fileslist = output_dir / "fileslist.txt"
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


def init_one(
    recipe_id: str,
    recipe: dict,
    *,
    output_root: Path,
    n_clips_override: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> Path:
    """Initialize one prebuilt dataset. Returns the output dir."""
    output_dir = output_root / recipe_id
    raw_dir = output_dir / "_raw"
    curated_dir = output_dir / "curated"

    n_clips = n_clips_override or recipe.get("n_clips", 1000)
    ratio = recipe.get("target_ratio") or DEFAULT_TARGET_RATIO

    print(f"\n[{recipe_id}] downloading {n_clips} clips from {recipe['hf_repo']} ...")
    raw_paths = _download_hf_subset(
        hf_repo=recipe["hf_repo"],
        repo_type=recipe.get("hf_repo_type", "dataset"),
        allow_patterns=recipe.get("hf_allow", ["**/*.mp4"]),
        target_dir=raw_dir,
        max_files=n_clips,
        hf_token=hf_token,
    )
    print(f"[{recipe_id}] downloaded {len(raw_paths)} files → {raw_dir}")

    if not raw_paths:
        raise SystemExit(f"❌ [{recipe_id}] download produced no files")

    # Stash the ratio so the curate call knows what bucket weights we want.
    # We pass it via env so curate_finetune_samples.py can pick it up if we
    # extend it; for now curate uses its own TARGET_RATIO constant.
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

    # Write a top-level fileslist.txt that points at the curated buckets,
    # so the user can drop it straight into gradio Tab 1.
    curated_fileslist = curated_dir / "fileslist.txt"
    top_fileslist = output_dir / "fileslist.txt"
    if curated_fileslist.exists():
        shutil.copy2(curated_fileslist, top_fileslist)

    _print_paste_able(recipe_id, recipe, curated_dir)
    return curated_dir


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
                         hf_token=hf_token)
            except SystemExit as exc:
                print(f"⚠️  {exc}", file=sys.stderr)
                continue
    else:
        if args.dataset not in recipes:
            sys.exit(f"❌ unknown dataset: {args.dataset}. Run --list.")
        init_one(args.dataset, recipes[args.dataset],
                 output_root=output_root,
                 n_clips_override=args.n_clips,
                 hf_token=hf_token)

    return 0


if __name__ == "__main__":
    sys.exit(main())