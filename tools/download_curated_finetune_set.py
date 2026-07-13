# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

"""One-shot driver: download + curate + ready-to-train dataset prep.

Wraps `tools/curate_finetune_samples.py` with finetune-specific defaults
(emphasis on side-face + fast-motion buckets) and prints a ready-to-paste
Tab 1 instruction at the end.

Pipeline:
    URLs / local files  ─►  download  ─►  face detection + motion score
                                   │
                                   ▼
                       out/{frontal,side_face,fast_motion}/
                       out/fileslist.txt
                       out/curation_report.json
                                   │
                                   ▼
                    📋 print: how to plug into gradio_finetune.py

Usage:
    # A) URLs → download → curate
    python tools/download_curated_finetune_set.py \\
        --urls tools/finetune_starter_urls.example.txt \\
        --output-dir data/finetune_samples_v1

    # B) Local pile → curate only (no download)
    python tools/download_curated_finetune_set.py \\
        --source-dir /data/my_raw_videos \\
        --output-dir data/finetune_samples_v1

After this finishes, copy-paste the printed gradio-form values into
Tab 1 (or run with --auto-launch-gradio to open the UI pre-filled).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Finetuned defaults — biased toward the typical "side-face / mouth blur"
# badcase goals. Override via CLI flags if you want a different mix.
DEFAULT_TARGET = 200
DEFAULT_RATIO = {"frontal": 0.40, "side_face": 0.40, "fast_motion": 0.20}

logger = logging.getLogger(__name__)


def _check_prereqs() -> None:
    """Verify yt-dlp is installed (only needed for download mode)."""
    if shutil.which("yt-dlp") is None:
        print(
            "⚠️  yt-dlp not found. Install with:\n"
            "    pip install -U yt-dlp\n"
            "(Required for downloading from URLs. Skipping download still "
            "works if you provide --source-dir.)"
        )


def _run_curate(
    *,
    urls: str | None,
    source_dir: str | None,
    output_dir: str,
    target_count: int,
    sample_frames: int,
    min_frames: int,
    max_candidates: int,
    device: str,
) -> int:
    """Shell out to curate_finetune_samples.py with our defaults.

    Returns its return code.
    """
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "curate_finetune_samples.py"),
        "--output-dir", output_dir,
        "--target-count", str(target_count),
        "--sample-frames", str(sample_frames),
        "--min-frames", str(min_frames),
        "--max-candidates", str(max_candidates),
        "--device", device,
    ]
    if urls:
        cmd += ["--urls", urls]
    if source_dir:
        cmd += ["--source-dir", source_dir]

    logger.info("running: %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


# Scale presets. Pick one via --scale to get a coherent default bundle;
# override individual flags if you want to fine-tune the recipe.
SCALE_PRESETS = {
    #            target,  max_candidates,  min_frames,  sample_frames
    "small":   (   200,         2000,         30,         64),
    "medium":  (  1000,        10000,         60,         64),
    "large":   (  5000,        50000,        120,        128),
}


def _print_finetune_paste_able(output_dir: Path, *, scale: str | None = None,
                              target_count: int = 0) -> None:
    """Print a copy-pasteable block the user can drop into Tab 1."""
    fileslist = output_dir / "fileslist.txt"
    report = output_dir / "curation_report.json"

    # Pick training steps scaled by data size
    if target_count >= 5000:
        preset_name = "Stage 2 (256, 推荐)"  # Full finetune on industry data
        max_steps_label = "max_train_steps=100000 (cap; lift the slider cap to 200k if 1 GPU, 50k if multi-GPU)"
        save_steps_label = "save_ckpt_steps=5000"
    elif target_count >= 1000:
        preset_name = "🎯 Badcase Fix (LoRA, 12-15GB)"
        max_steps_label = "max_train_steps=50000 (about 20h on a single GPU)"
        save_steps_label = "save_ckpt_steps=2000"
    else:
        preset_name = "🎯 Badcase Fix (LoRA, 12-15GB)"
        max_steps_label = "max_train_steps=20000 (default, ~6h on a single GPU)"
        save_steps_label = "save_ckpt_steps=1000"

    print()
    print("=" * 72)
    print(f"✅ Curated dataset ready in: {output_dir}")
    print("=" * 72)
    if report.exists():
        import json as _json
        rep = _json.loads(report.read_text())
        print(f"   scale:            {scale or '(manual)'}")
        print(f"   kept:             {rep['kept']} / {rep['total_candidates']}")
        print(f"   by bucket:")
        for cat, vs in rep["by_bucket"].items():
            print(f"     - {cat:14s}: {len(vs)}")
        if rep.get("rejected"):
            print(f"   rejected:         {len(rep['rejected'])}")
    print()
    print("📋 Paste into Tab 1 'Configure & Launch':")
    print()
    print(f"   preset_dd:        '{preset_name}'")
    print(f"   train_data_dir:   {output_dir.resolve()}")
    print(f"   train_fileslist:  {fileslist.resolve()}")
    print()
    print("   Suggested training knobs (override the preset defaults):")
    print(f"   - {max_steps_label}")
    print(f"   - {save_steps_label}")
    print("   - sync_loss_weight=0.12, num_frames=24, freeze_attn2=True")
    print()
    print("🧪 After training, evaluate:")
    print("   - Tab 3 '推理对比 base vs fine-tuned' with your best checkpoint")
    print("   - Tab 6 'Badcase 检查清单' for blurry/flicker/sync/identity numbers")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end finetune dataset prep (download + curate).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--urls", type=str, default=None,
                        help="text file with one URL per line (triggers yt-dlp download)")
    parser.add_argument("--source-dir", type=str, default=None,
                        help="local directory to scan for candidate video files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="where to put curated buckets and fileslist")
    parser.add_argument("--scale", choices=list(SCALE_PRESETS.keys()), default=None,
                        help="preset bundle: 'small' (200 LoRA quick), "
                             "'medium' (1000 finetune), 'large' (5000 industry-grade). "
                             "If set, overrides --target-count / --max-candidates / --min-frames / --sample-frames.")
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET,
                        help=f"target total kept videos (default {DEFAULT_TARGET} for finetune)")
    parser.add_argument("--max-candidates", type=int, default=10000,
                        help="hard cap on videos to scan after download")
    parser.add_argument("--min-frames", type=int, default=30,
                        help="min frames per video; bump with --scale")
    parser.add_argument("--sample-frames", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--auto-launch-gradio", action="store_true",
                        help="open gradio_finetune.py after curation")
    parser.add_argument("--log", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.urls and not args.source_dir:
        parser.error("provide --urls (text file) OR --source-dir (local pile)")

    out_dir = Path(args.output_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        resp = input(f"⚠️  {out_dir} is not empty. Overwrite? [y/N] ")
        if resp.strip().lower() != "y":
            print("Aborted.")
            return 1

    if args.urls:
        _check_prereqs()

    # Apply scale preset (overrides individual flags if set)
    if args.scale:
        target, max_cands, min_frames, sample_frames = SCALE_PRESETS[args.scale]
        logger.info("scale=%s preset: target=%d, max_candidates=%d, min_frames=%d, sample_frames=%d",
                    args.scale, target, max_cands, min_frames, sample_frames)
    else:
        target = args.target_count
        max_cands = args.max_candidates
        min_frames = args.min_frames
        sample_frames = args.sample_frames

    rc = _run_curate(
        urls=args.urls,
        source_dir=args.source_dir,
        output_dir=str(out_dir),
        target_count=target,
        sample_frames=sample_frames,
        min_frames=min_frames,
        max_candidates=max_cands,
        device=args.device,
    )
    if rc != 0:
        print(f"❌ curate_finetune_samples.py exited with rc={rc}", file=sys.stderr)
        return rc

    _print_finetune_paste_able(out_dir, scale=args.scale, target_count=target)

    if args.auto_launch_gradio:
        try:
            subprocess.Popen(
                [sys.executable, str(REPO_ROOT / "gradio_finetune.py")],
                cwd=str(REPO_ROOT),
            )
        except Exception as exc:
            logger.warning("could not auto-launch gradio: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
