"""Inference wrapper used by gradio_finetune validation/compare tabs.

Performs LoRA adapter merging and optional quality checks inside a
subprocess so the Gradio event loop is never blocked.

Example (validate):
    python -m scripts.finetune_inference validate \
        --unet_config_path configs/unet/stage2.yaml \
        --inference_ckpt_path checkpoints/latentsync_unet.pt \
        --video_path assets/demo1_video.mp4 \
        --audio_path assets/demo1_audio.wav \
        --video_out_path debug/validation_outputs/val.mp4 \
        --inference_steps 20 --guidance_scale 1.5 --seed 1247 \
        --enable_deepcache

Example (compare):
    python -m scripts.finetune_inference compare \
        --unet_config_path configs/unet/stage2.yaml \
        --base_ckpt checkpoints/latentsync_unet.pt \
        --fine_tuned_ckpt debug/unet_lora/.../checkpoints/checkpoint-5000 \
        --video_path assets/demo1_video.mp4 \
        --audio_path assets/demo1_audio.wav \
        --base_video_out debug/compare_outputs/base.mp4 \
        --ft_video_out debug/compare_outputs/ft.mp4 \
        --inference_steps 20 --guidance_scale 1.5 --seed 1247
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from latentsync.finetune import REPO_ROOT, logger
from latentsync.finetune.process import _prune_debug_files
from latentsync.finetune.validation_utils import (
    check_ckpt_compatibility,
    merge_adapter_to_temp_pt,
    prepare_temp_config,
    quick_quality_check,
    resolve_ckpt_path,
    write_result_json,
)


def _build_inference_cmd(
    unet_config_path: Path,
    ckpt_path: Path,
    video_path: str,
    audio_path: str,
    out_path: Path,
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.inference",
        "--unet_config_path",
        str(unet_config_path),
        "--inference_ckpt_path",
        str(ckpt_path),
        "--video_path",
        str(video_path),
        "--audio_path",
        str(audio_path),
        "--video_out_path",
        str(out_path),
        "--inference_steps",
        str(int(args.inference_steps)),
        "--guidance_scale",
        str(float(args.guidance_scale)),
        "--seed",
        str(int(args.seed)),
        "--temp_dir",
        str(args.temp_dir),
    ]
    if args.enable_deepcache:
        cmd.append("--enable_deepcache")
    if args.baseline_mode:
        cmd.append("--baseline_mode")
    return cmd


def _run_inference_subprocess(cmd: List[str]) -> int:
    """Run ``scripts.inference`` inheriting the wrapper's stdout/stderr.

    The Gradio side captures the wrapper's stdout into the per-run log
    file; inheriting it lands the inner inference logs in that same file
    so the UI's failure tail shows the real error (previously the inner
    logs went to a separate, UI-invisible file).
    """
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
    return proc.wait()


def _resolve_ckpt(
    ckpt_path: str,
    base_ckpt: str,
    unet_config: Path,
) -> Path:
    """Resolve checkpoint path and merge LoRA adapter on the fly if needed."""
    p = resolve_ckpt_path(ckpt_path)
    if p.is_dir() and (p / "adapter_config.json").exists():
        return merge_adapter_to_temp_pt(p, base_ckpt, unet_config)
    return p


def _run_single(
    args: argparse.Namespace,
    ckpt_path: Path,
    out_path: Path,
    unet_config_path: Path,
) -> int:
    """Run one inference and return its subprocess exit code."""
    cmd = _build_inference_cmd(
        unet_config_path,
        ckpt_path,
        args.video_path,
        args.audio_path,
        out_path,
        args,
    )
    logger.info("[finetune_inference] running: %s", " ".join(cmd))
    return _run_inference_subprocess(cmd)


def _quality_check_optional(video_path: Path, skip: bool) -> Optional[Dict[str, Any]]:
    if skip or not video_path.exists():
        return None
    return quick_quality_check(str(video_path))


def cmd_validate(args: argparse.Namespace) -> int:
    """Run single-ckpt validation."""
    start = time.time()
    unet_config = Path(args.unet_config_path)
    if not unet_config.exists():
        logger.error("config not found: %s", unet_config)
        return 1

    out_path = Path(args.video_out_path)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    # The UI pins --result_json so its poller and this process agree on
    # the filename; the timestamped name is only a fallback for manual
    # CLI runs.
    result_json = (
        Path(args.result_json) if args.result_json
        else out_dir / f"validation_{ts}.json"
    )

    _prune_debug_files(out_dir, "validation_cfg_*.yaml", keep=10)

    try:
        ckpt = _resolve_ckpt(args.inference_ckpt_path, args.base_ckpt, unet_config)
    except Exception as exc:
        logger.exception("failed to merge adapter")
        write_result_json(
            result_json,
            mode="validate",
            success=False,
            duration_sec=time.time() - start,
            error=f"LoRA merge failed: {exc}",
        )
        return 2

    tmp_cfg = prepare_temp_config(
        unet_config,
        args.resolution,
        args.inference_steps,
        args.guidance_scale,
        out_dir,
        ts,
        prefix="validation_cfg",
    )

    rc = _run_single(args, ckpt, out_path, tmp_cfg)
    if rc != 0:
        write_result_json(
            result_json,
            mode="validate",
            success=False,
            duration_sec=time.time() - start,
            error=f"inference subprocess exited with rc={rc}",
            video_out_path=out_path,
        )
        return rc

    metrics = _quality_check_optional(out_path, args.skip_quality_check)
    write_result_json(
        result_json,
        mode="validate",
        success=True,
        duration_sec=time.time() - start,
        video_out_path=out_path,
        metrics=metrics,
    )
    logger.info("[finetune_inference] validate done: %s (%.1fs)", out_path, time.time() - start)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Run base-vs-fine-tuned comparison (two sequential inferences)."""
    start = time.time()
    unet_config = Path(args.unet_config_path)
    if not unet_config.exists():
        logger.error("config not found: %s", unet_config)
        return 1

    base_video = Path(args.base_video_out)
    ft_video = Path(args.ft_video_out)
    out_dir = base_video.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    # The UI pins --result_json so its poller and this process agree on
    # the filename; the timestamped name is only a fallback for manual
    # CLI runs.
    result_json = (
        Path(args.result_json) if args.result_json
        else out_dir / f"compare_{ts}.json"
    )

    _prune_debug_files(out_dir, "compare_cfg_*.yaml", keep=10)

    tmp_cfg = prepare_temp_config(
        unet_config,
        args.resolution,
        args.inference_steps,
        args.guidance_scale,
        out_dir,
        ts,
        prefix="compare_cfg",
    )

    try:
        base_ckpt = _resolve_ckpt(args.base_ckpt, args.base_ckpt, unet_config)
        ft_ckpt = _resolve_ckpt(args.fine_tuned_ckpt, args.base_ckpt, unet_config)
    except Exception as exc:
        logger.exception("failed to merge adapter")
        write_result_json(
            result_json,
            mode="compare",
            success=False,
            duration_sec=time.time() - start,
            error=f"LoRA merge failed: {exc}",
        )
        return 2

    # Run base first, then fine-tuned.
    rc_base = _run_single(args, base_ckpt, base_video, tmp_cfg)
    if rc_base != 0:
        write_result_json(
            result_json,
            mode="compare",
            success=False,
            duration_sec=time.time() - start,
            error=f"base inference failed (rc={rc_base})",
        )
        return rc_base

    rc_ft = _run_single(args, ft_ckpt, ft_video, tmp_cfg)
    if rc_ft != 0:
        write_result_json(
            result_json,
            mode="compare",
            success=False,
            duration_sec=time.time() - start,
            error=f"fine-tuned inference failed (rc={rc_ft})",
            base_video=base_video,
        )
        return rc_ft

    base_metrics = _quality_check_optional(base_video, args.skip_quality_check)
    ft_metrics = _quality_check_optional(ft_video, args.skip_quality_check)

    write_result_json(
        result_json,
        mode="compare",
        success=True,
        duration_sec=time.time() - start,
        base_video=base_video,
        ft_video=ft_video,
        base_metrics=base_metrics,
        ft_metrics=ft_metrics,
    )
    logger.info(
        "[finetune_inference] compare done: base=%s ft=%s (%.1fs)",
        base_video, ft_video, time.time() - start,
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Finetune validation/compare inference wrapper",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Common inference args.
    def _add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--unet_config_path", type=str, required=True)
        p.add_argument("--video_path", type=str, required=True)
        p.add_argument("--audio_path", type=str, required=True)
        p.add_argument("--inference_steps", type=int, default=20)
        p.add_argument("--guidance_scale", type=float, default=1.5)
        p.add_argument("--seed", type=int, default=1247)
        p.add_argument("--resolution", type=int, default=512)
        p.add_argument("--temp_dir", type=str, default="temp")
        p.add_argument("--enable_deepcache", action="store_true")
        p.add_argument("--baseline_mode", action="store_true")
        p.add_argument("--skip_quality_check", action="store_true")
        # Pin the result JSON path so the Gradio poller and this process
        # write/read the same file (avoids a timestamp race).
        p.add_argument("--result_json", type=str, default="")

    p_val = sub.add_parser("validate")
    _add_common(p_val)
    p_val.add_argument("--inference_ckpt_path", type=str, required=True)
    p_val.add_argument("--base_ckpt", type=str, default="checkpoints/latentsync_unet.pt")
    p_val.add_argument("--video_out_path", type=str, required=True)

    p_cmp = sub.add_parser("compare")
    _add_common(p_cmp)
    p_cmp.add_argument("--base_ckpt", type=str, required=True)
    p_cmp.add_argument("--fine_tuned_ckpt", type=str, required=True)
    p_cmp.add_argument("--base_video_out", type=str, required=True)
    p_cmp.add_argument("--ft_video_out", type=str, required=True)

    args = parser.parse_args(argv)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "compare":
        return cmd_compare(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
