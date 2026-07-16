"""Shared helpers for gradio_finetune validation / compare tabs.

These helpers are used by both the Gradio UI event handlers and the
subprocess wrapper ``scripts/finetune_inference.py``. Keeping them in a
separate module avoids importing Gradio (and its heavy deps) inside the
inference subprocess.
"""
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml
from omegaconf import OmegaConf

from latentsync.finetune import FINETUNE_BASE_DIR, REPO_ROOT, logger


def _prune_merged_adapters(directory: Path, keep: int = 20) -> None:
    """Keep only the N most-recently-modified merged adapter checkpoints.

    Merged adapters are 4-5 GB each; without pruning the debug directory
    grows indefinitely. We keep the most recent ``keep`` files across all
    adapter runs.
    """
    if not directory.exists():
        return
    matches = sorted(directory.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in matches[keep:]:
        try:
            old.unlink()
            logger.info("[merged_adapters] pruned old merged ckpt: %s", old)
        except OSError:
            pass


def merge_adapter_to_temp_pt(
    adapter_dir: Path,
    base_ckpt: str,
    unet_config: Path,
) -> Path:
    """Merge a peft LoRA adapter into the base UNet and save a temporary .pt.

    Used by the validation/compare tabs so users can select a LoRA adapter
    directory directly without manually running ``scripts/merge_lora.py``
    first.

    Reuses an existing merged checkpoint when the (adapter_dir, base_ckpt)
    pair has already been processed in this session, avoiding redundant
    4-5 GB writes. Old merged checkpoints are pruned to cap disk usage.
    """
    import torch
    from peft import PeftModel

    from latentsync.models.unet import UNet3DConditionModel

    out_dir = FINETUNE_BASE_DIR / "debug" / "merged_adapters"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stable cache key for this (adapter, base) pair.
    cache_key = hashlib.md5(f"{adapter_dir.resolve()}:{base_ckpt}".encode()).hexdigest()[:12]
    cached = sorted(out_dir.glob(f"{adapter_dir.name}_{cache_key}_*.pt"))
    if cached:
        logger.info("[validation] reusing merged ckpt for %s: %s", adapter_dir, cached[-1])
        return cached[-1]

    logger.info("[validation] merging adapter %s into base %s", adapter_dir, base_ckpt)
    cfg = OmegaConf.load(unet_config)
    base, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(cfg.model),
        base_ckpt,
        device="cpu",
    )

    peft_model = PeftModel.from_pretrained(base, str(adapter_dir), device="cpu")
    merged = peft_model.merge_and_unload()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_pt = out_dir / f"{adapter_dir.name}_{cache_key}_{ts}.pt"
    torch.save({"global_step": 0, "state_dict": merged.state_dict()}, out_pt)
    logger.info("[validation] saved merged ckpt to %s", out_pt)
    _prune_merged_adapters(out_dir)
    return out_pt


def check_ckpt_compatibility(ckpt_path: Path, unet_config: Path) -> List[str]:
    """Compare a checkpoint against a UNet config and return any warnings.

    Catches the common 256/512 + mask.png/mask2.png mismatches that
    produce silent garbage rather than a clean error.
    """
    warnings: List[str] = []
    try:
        cfg = OmegaConf.load(unet_config)
        cfg_res = int(cfg.data.resolution)
        cfg_mask = str(cfg.data.mask_image_path)
    except Exception as e:
        return [f"❌ 解析 config 失败: {e}"]

    ckpt_name = ckpt_path.name.lower()
    if "512" in ckpt_name and cfg_res == 256:
        warnings.append("⚠️ ckpt 名字带 '512' 但 config resolution=256，可能不兼容")
    if "512" not in ckpt_name and cfg_res == 512 and "stage1" not in ckpt_name:
        warnings.append("⚠️ config resolution=512 但 ckpt 名字没 '512'，请确认 ckpt 真是 512 训的")
    if "mask2" in cfg_mask and "mask.png" in str(ckpt_path):
        warnings.append(f"⚠️ config 用 {cfg_mask} 但 ckpt 可能是 256 训的（用 mask.png）")
    if "mask2" not in cfg_mask and "mask2" in str(ckpt_path):
        warnings.append(f"⚠️ config 用 {cfg_mask} 但 ckpt 可能是 512 训的（用 mask2.png）")
    if "lora" in ckpt_name:
        warnings.append("⚠️ ckpt 名字含 'lora' —— 可能是 LoRA adapter，需要先 merge_lora.py")
    if "adapter" in ckpt_name:
        warnings.append("⚠️ ckpt 是 peft adapter，需要先 merge_lora.py")
    return warnings


def quick_quality_check(video_path: str) -> Dict[str, Any]:
    """Run a lightweight quality check on a single generated video.

    Computes: sharpness (Laplacian), flicker (frame diff), face
    detection rate. Skips SyncNet / HyperIQA (those are too heavy
    for a per-click UI call).
    """
    if not video_path or not Path(video_path).exists():
        return {"error": f"video not found: {video_path}"}
    try:
        from decord import VideoReader

        vr = VideoReader(video_path)
        frames = [f.asnumpy() for f in vr]
        if len(frames) < 2:
            return {"error": "video too short"}

        # sharpness per frame
        grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        laps = [cv2.Laplacian(g, cv2.CV_64F).var() for g in grays]
        sharpness_mean = float(np.mean(laps))
        blurry_count = sum(1 for v in laps if v < 50)
        blurry_ratio = blurry_count / len(frames)

        # flicker
        diffs = [
            float(np.abs(frames[i].astype(float) - frames[i - 1].astype(float)).mean())
            for i in range(1, len(frames))
        ]
        flicker = float(np.mean(diffs))

        # face detection rate (sample up to 10 frames)
        face_rate: Optional[float] = None
        try:
            from latentsync.utils.face_detector import FaceDetector

            det = FaceDetector()
            detected = 0
            sample = frames[:: max(1, len(frames) // 10)][:10]
            for f in sample:
                face, _, _ = det.detect(f)
                if face is not None:
                    detected += 1
            face_rate = detected / len(sample)
        except Exception:
            pass

        return {
            "num_frames": len(frames),
            "sharpness_mean": round(sharpness_mean, 2),
            "blurry_ratio": round(blurry_ratio, 3),
            "flicker": round(flicker, 2),
            "face_detect_rate": round(face_rate, 2) if face_rate is not None else None,
        }
    except Exception as e:
        return {"error": str(e)}


def format_validation_report(
    metrics: Dict[str, Any],
    ckpt_path: str,
    duration_sec: float,
) -> str:
    """Render a human-readable quality report."""
    if "error" in metrics:
        return f"❌ 质量检查失败: {metrics['error']}"

    lines: List[str] = []
    lines.append(f"📦 Checkpoint: {ckpt_path}")
    lines.append(f"⏱ 推理耗时: {duration_sec:.1f} 秒")
    lines.append(f"🎞 总帧数: {metrics.get('num_frames', '?')}")
    lines.append("")

    sharp = metrics.get("sharpness_mean", 0)
    blurry = metrics.get("blurry_ratio", 0)
    flicker = metrics.get("flicker", 0)
    face_rate = metrics.get("face_detect_rate")

    # sharpness
    if sharp >= 200:
        lines.append(f"✅ 清晰度 {sharp:.1f} (优秀)")
    elif sharp >= 100:
        lines.append(f"✅ 清晰度 {sharp:.1f} (良好)")
    elif sharp >= 50:
        lines.append(f"⚠️ 清晰度 {sharp:.1f} (一般，可能有点糊)")
    else:
        lines.append(f"❌ 清晰度 {sharp:.1f} (差，明显模糊)")
    lines.append(f"   嘴糊比例: {blurry*100:.1f}% (目标 < 30%)")

    # flicker
    if flicker < 4:
        lines.append(f"✅ 闪烁 {flicker:.2f} (优秀)")
    elif flicker < 8:
        lines.append(f"✅ 闪烁 {flicker:.2f} (正常)")
    else:
        lines.append(f"⚠️ 闪烁 {flicker:.2f} (偏高)")
    lines.append("")

    # face detection
    if face_rate is None:
        lines.append("ℹ️ 人脸检测不可用（缺 mediapipe / insightface）")
    elif face_rate >= 0.9:
        lines.append(f"✅ 人脸检测 {face_rate*100:.0f}% (绝大多数帧检测到)")
    elif face_rate >= 0.5:
        lines.append(f"⚠️ 人脸检测 {face_rate*100:.0f}% (部分帧未检测到)")
    else:
        lines.append(f"❌ 人脸检测 {face_rate*100:.0f}% (可能大量帧被 skip)")

    # recommendations
    lines.append("")
    lines.append("=== 建议 ===")
    if blurry > 0.3:
        lines.append("• 嘴糊比例高：考虑升 512 / 加 LPIPS / 开 CodeFormer")
    if flicker > 8:
        lines.append("• 闪烁偏高：考虑开 TREPA / Motion Module 训够 / 加时序稳定")
    if face_rate is not None and face_rate < 0.7:
        lines.append("• 人脸检测率低：检查 [FaceMatch] 日志，可能是 yaw/blur 跳太多")
    if sharp >= 100 and blurry <= 0.3 and flicker < 8:
        lines.append("✅ 整体质量良好，可以用！")
    return "\n".join(lines)


def write_result_json(
    path: Path,
    mode: str,
    success: bool,
    duration_sec: float,
    error: Optional[str] = None,
    video_out_path: Optional[Path] = None,
    metrics: Optional[Dict[str, Any]] = None,
    base_video: Optional[Path] = None,
    ft_video: Optional[Path] = None,
    base_metrics: Optional[Dict[str, Any]] = None,
    ft_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a small JSON result file next to the output video(s).

    The Gradio Timer poller reads this file to update the UI without
    blocking on heavy quality checks.
    """
    payload: Dict[str, Any] = {
        "mode": mode,
        "success": success,
        "duration_sec": round(duration_sec, 2),
        "error": error,
    }
    if video_out_path is not None:
        payload["video_out_path"] = str(video_out_path)
    if metrics is not None:
        payload["metrics"] = metrics
    if base_video is not None:
        payload["base_video"] = str(base_video)
    if ft_video is not None:
        payload["ft_video"] = str(ft_video)
    if base_metrics is not None:
        payload["base_metrics"] = base_metrics
    if ft_metrics is not None:
        payload["ft_metrics"] = ft_metrics

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_result_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read a result JSON written by ``write_result_json``."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("[validation] failed to read result json %s: %s", path, e)
        return None


def resolve_ckpt_path(ckpt_path: str) -> Path:
    """Resolve a checkpoint path that may be relative to REPO_ROOT."""
    p = Path(ckpt_path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def prepare_temp_config(
    unet_config: Path,
    resolution: int,
    inference_steps: int,
    guidance_scale: float,
    out_dir: Path,
    ts: str,
    prefix: str = "validation_cfg",
) -> Path:
    """Clone a UNet config with overridden resolution/steps/guidance."""
    base_cfg = OmegaConf.load(unet_config)
    base_cfg.data.resolution = int(resolution)
    base_cfg.run.inference_steps = int(inference_steps)
    base_cfg.run.guidance_scale = float(guidance_scale)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg = out_dir / f"{prefix}_{ts}.yaml"
    with open(tmp_cfg, "w") as f:
        yaml.dump(OmegaConf.to_container(base_cfg), f)
    return tmp_cfg
