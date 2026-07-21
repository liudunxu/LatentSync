"""Tab 3 / Tab 3.5: inference comparison and validation callbacks."""
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from latentsync.finetune import REPO_ROOT, logger
from latentsync.finetune.process import _INFERENCE, _prune_debug_files
from latentsync.finetune.utils import tail_file
from latentsync.finetune.validation_utils import (
    check_ckpt_compatibility,
    format_validation_report,
    read_result_json,
    resolve_ckpt_path,
)


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _heartbeat(prefix: str) -> str:
    state = _INFERENCE
    return f"[{_now()}] {prefix} | kind={state.kind} status={state.status} busy={state.is_busy()}"


def _build_finetune_inference_cmd(
    mode: str,
    video_path: str,
    audio_path: str,
    unet_config: Path,
    inference_steps: int,
    guidance_scale: float,
    seed: int,
    resolution: int,
    enable_deepcache: bool,
    baseline_mode: bool,
    skip_quality_check: bool,
    out_path: Path,
    ckpt_path: Optional[str] = None,
    base_ckpt: Optional[str] = None,
    fine_tuned_ckpt: Optional[str] = None,
    second_out_path: Optional[Path] = None,
    result_json: Optional[Path] = None,
) -> List[str]:
    """Build the command for ``scripts/finetune_inference.py``.

    ``result_json`` pins the JSON path the subprocess writes so the UI
    poller and the subprocess agree on the filename (both used to mint
    their own timestamps, which raced whenever startup crossed a second
    boundary and made the poller read a file that was never written).
    """
    cmd = [
        sys.executable,
        "-m",
        "scripts.finetune_inference",
        mode,
        "--unet_config_path",
        str(unet_config),
        "--video_path",
        str(video_path),
        "--audio_path",
        str(audio_path),
        "--inference_steps",
        str(int(inference_steps)),
        "--guidance_scale",
        str(float(guidance_scale)),
        "--seed",
        str(int(seed)),
        "--resolution",
        str(int(resolution)),
        "--temp_dir",
        "temp",
    ]
    if enable_deepcache:
        cmd.append("--enable_deepcache")
    if baseline_mode:
        cmd.append("--baseline_mode")
    if skip_quality_check:
        cmd.append("--skip_quality_check")
    if result_json is not None:
        cmd.extend(["--result_json", str(result_json)])

    if mode == "validate":
        assert ckpt_path is not None
        cmd.extend(["--inference_ckpt_path", str(ckpt_path)])
        cmd.extend(["--base_ckpt", str(base_ckpt) if base_ckpt else "checkpoints/latentsync_unet.pt"])
        cmd.extend(["--video_out_path", str(out_path)])
    else:
        assert base_ckpt is not None and fine_tuned_ckpt is not None and second_out_path is not None
        cmd.extend(["--base_ckpt", str(base_ckpt)])
        cmd.extend(["--fine_tuned_ckpt", str(fine_tuned_ckpt)])
        cmd.extend(["--base_video_out", str(out_path)])
        cmd.extend(["--ft_video_out", str(second_out_path)])

    return cmd


def run_compare(
    video_path: str,
    audio_path: str,
    base_ckpt: str,
    fine_tuned_ckpt: str,
    inference_steps: int,
    guidance_scale: float,
    seed: int,
    resolution: int,
    baseline_mode: bool = False,
) -> Tuple[Any, Any, Any]:
    """Kick off a background base-vs-fine-tuned comparison.

    The actual result is delivered by the Timer poller
    ``_poll_compare_state`` when the subprocess finishes.
    """
    if not video_path or not audio_path:
        raise gr.Error("请先上传视频和音频")
    if not base_ckpt or not fine_tuned_ckpt:
        raise gr.Error("请选择 base 和 fine-tuned 两个 checkpoint")
    if base_ckpt == fine_tuned_ckpt:
        raise gr.Error("两个 checkpoint 必须不同")

    if _INFERENCE.is_busy():
        return (
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=f"❌ 已有推理在运行 (kind={_INFERENCE.kind})，请先 ⏹ 取消"),
        )

    out_dir = REPO_ROOT / "debug" / "compare_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = out_dir / f"base_{ts}.mp4"
    out_ft = out_dir / f"finetuned_{ts}.mp4"
    log_path = out_dir / f"compare_{ts}.log"
    result_json = out_dir / f"compare_{ts}.json"

    _prune_debug_files(out_dir, "compare_cfg_*.yaml", keep=10)

    # 512 must run with stage2_512.yaml (mask2.png); prepare_temp_config
    # only overrides resolution, so picking stage2.yaml here would pair a
    # 512 resolution with the 256 mask and produce garbage.
    unet_config = Path(
        "configs/unet/stage2_512.yaml"
        if int(resolution) == 512
        else "configs/unet/stage2.yaml"
    )

    cmd = _build_finetune_inference_cmd(
        mode="compare",
        video_path=video_path,
        audio_path=audio_path,
        unet_config=unet_config,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        resolution=resolution,
        enable_deepcache=True,
        baseline_mode=baseline_mode,
        skip_quality_check=True,
        out_path=out_base,
        base_ckpt=base_ckpt,
        fine_tuned_ckpt=fine_tuned_ckpt,
        second_out_path=out_ft,
        result_json=result_json,
    )

    if not _INFERENCE.start(
        cmd,
        log_path,
        kind="compare",
        label=f"compare base={Path(base_ckpt).name} vs ft={Path(fine_tuned_ckpt).name}",
        result_video=out_base,
        result_json=result_json,
    ):
        return (
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value="❌ 启动失败"),
        )

    return (
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=(
            f"⏳ 对比推理启动中\n"
            f"📜 log: {log_path.relative_to(REPO_ROOT)}\n"
            f"💡 页面每秒自动刷新，完成后会在此显示两个视频"
        )),
    )


def run_validation(
    video_path: str,
    audio_path: str,
    ckpt_path: str,
    unet_config: str,
    inference_steps: int,
    guidance_scale: float,
    seed: int,
    resolution: int,
    enable_deepcache: bool,
    skip_quality_check: bool,
    baseline_mode: bool = False,
) -> Tuple[Any, Any, Any, Any, Any]:
    """Kick off Tab-3.5 single-ckpt inference in the background.

    Returns immediately with a '⏳ running' status; the actual result
    is delivered by the Timer poller ``_poll_validate_state``.
    """
    if _INFERENCE.is_busy():
        return (
            gr.update(value=None),
            gr.update(),
            gr.update(value=f"❌ 已有推理在运行 (kind={_INFERENCE.kind})，请先 ⏹ 取消"),
            gr.update(value=None),
            gr.update(interactive=True),
        )
    if not ckpt_path or not resolve_ckpt_path(ckpt_path).exists():
        return (
            gr.update(value=None),
            gr.update(),
            gr.update(value=f"❌ ckpt 不存在: {ckpt_path}"),
            gr.update(value=None),
            gr.update(interactive=True),
        )
    if not unet_config or not resolve_ckpt_path(unet_config).exists():
        return (
            gr.update(value=None),
            gr.update(),
            gr.update(value=f"❌ config 不存在: {unet_config}"),
            gr.update(value=None),
            gr.update(interactive=True),
        )

    ckpt = resolve_ckpt_path(ckpt_path)
    cfg_path = resolve_ckpt_path(unet_config)

    warnings = check_ckpt_compatibility(ckpt, cfg_path)
    merged_hint = ""
    if ckpt.is_dir() and (ckpt / "adapter_config.json").exists():
        merged_hint = "ℹ️ LoRA adapter 会在推理子进程里自动合并\n"
    warnings_text = "\n".join(warnings) if warnings else "✅ ckpt 与 config 兼容"
    if merged_hint:
        warnings_text = merged_hint + warnings_text

    _prune_debug_files(REPO_ROOT / "debug" / "validation_outputs", "validation_cfg_*.yaml", keep=10)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "debug" / "validation_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"validation_{ts}.mp4"
    log_path = out_dir / f"validation_{ts}.log"
    result_json = out_dir / f"validation_{ts}.json"

    cmd = _build_finetune_inference_cmd(
        mode="validate",
        video_path=video_path,
        audio_path=audio_path,
        unet_config=cfg_path,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        resolution=resolution,
        enable_deepcache=enable_deepcache,
        baseline_mode=baseline_mode,
        skip_quality_check=skip_quality_check,
        out_path=out_mp4,
        ckpt_path=ckpt_path,
        result_json=result_json,
    )

    if not _INFERENCE.start(
        cmd,
        log_path,
        kind="validate",
        label=f"validate ckpt={ckpt.name}",
        result_video=out_mp4,
        result_json=result_json,
    ):
        return (
            gr.update(value=None),
            gr.update(),
            gr.update(value="❌ 启动失败"),
            gr.update(value=None),
            gr.update(interactive=True),
        )

    skip_msg = " (skip quality check)" if skip_quality_check else ""
    return (
        gr.update(value=None),
        gr.update(value=warnings_text),
        gr.update(value=(
            f"⏳ validate 推理启动中{skip_msg}\n"
            f"📜 log: {log_path.relative_to(REPO_ROOT)}\n"
            f"💡 等待自动刷新，或点击 ⏹ 取消"
        )),
        gr.update(value=None),
        gr.update(interactive=False),
    )


def _consume_state():
    """Reset the one-shot inference state after a finished result."""
    _INFERENCE.status = _INFERENCE.IDLE
    _INFERENCE.result_video = None
    _INFERENCE.result_json = None


def _running_update(prefix: str) -> Tuple[Any, ...]:
    """Build the running-heartbeat update tuple for a validate poller."""
    try:
        pid = _INFERENCE.proc.pid if _INFERENCE.proc else "?"
    except Exception:
        pid = "?"
    return (
        gr.update(value=None),
        gr.update(),
        gr.update(value=_heartbeat(
            f"⏳ {_INFERENCE.label} running (pid={pid})\n"
            "💡 页面每秒自动刷新，推理完成后会在此显示视频"
        )),
        gr.update(),
        gr.update(interactive=False),
    )


def _poll_validate_state(skip_quality_check: bool) -> Tuple[Any, Any, Any, Any, Any]:
    """Timer poller for Tab 3.5 — checks _INFERENCE and delivers the result."""
    state = _INFERENCE
    logger.debug(
        "[_poll_validate_state] kind=%s status=%s busy=%s",
        state.kind, state.status, state.is_busy(),
    )

    if state.kind != "validate" or state.status not in (
        state.DONE, state.FAILED, state.CANCELLED,
    ):
        if state.is_busy():
            return _running_update("validate")
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(interactive=True),
        )

    # ---- one-shot consumption of a finished result ----
    result_video = state.result_video
    result_json = state.result_json
    exit_code = state.exit_code
    log_path = state.log_path
    label = state.label
    now = _now()

    if state.status == state.DONE:
        video_path_str = str(result_video.resolve()) if result_video else ""

        if result_video is None or not result_video.exists():
            _consume_state()
            err_text = (
                f"[{now}] ❌ 推理状态为完成，但输出视频不存在: {result_video}\n"
                f"📜 log: {log_path}\n"
                "请检查 validation 子进程日志确认视频是否写入成功。"
            )
            return (
                gr.update(value=None),
                gr.update(),
                gr.update(value=err_text),
                gr.update(value=None),
                gr.update(interactive=True),
            )

        data = read_result_json(result_json) if result_json else None
        duration = data.get("duration_sec", 0.0) if data else 0.0
        metrics = data.get("metrics") if data else None

        try:
            if skip_quality_check or metrics is None:
                report = (
                    f"[{now}] ✅ 推理完成（已跳过质量自检）\n"
                    f"📂 {video_path_str}\n"
                    f"⏱ 耗时: {duration:.1f} 秒\n"
                    f"📜 log: {log_path}"
                )
            else:
                report = format_validation_report(metrics, label, duration)
                report = f"[{now}] {report}\n📂 {video_path_str}\n📜 log: {log_path}"
        except Exception as exc:
            logger.exception("[_poll_validate_state] quality check failed")
            report = (
                f"[{now}] ✅ 推理完成，但质量自检异常: {exc}\n"
                f"📂 {video_path_str}\n"
                f"📜 log: {log_path}"
            )

        logger.info("[_poll_validate_state] validation done: %s", video_path_str)
        _consume_state()
        return (
            gr.update(value=video_path_str),
            gr.update(),
            gr.update(value=report),
            gr.update(value=video_path_str),
            gr.update(interactive=True),
        )

    if state.status == state.CANCELLED:
        _consume_state()
        return (
            gr.update(value=None),
            gr.update(),
            gr.update(value=(
                f"[{now}] ⏹ 已取消\n"
                f"📜 log: {log_path}\n"
                f"最后 30 行:\n{tail_file(log_path, 30)}"
            )),
            gr.update(value=None),
            gr.update(interactive=True),
        )

    # FAILED
    _consume_state()
    err_text = (
        f"[{now}] ❌ 推理失败 (rc={exit_code})\n"
        f"📜 log: {log_path}\n\n"
        f"最后 30 行:\n{tail_file(log_path, 30)}"
    )
    return (
        gr.update(value=None),
        gr.update(),
        gr.update(value=err_text),
        gr.update(value=None),
        gr.update(interactive=True),
    )


def _poll_compare_state() -> Tuple[Any, Any, Any]:
    """Timer poller for Tab 3 — delivers both output videos when ready."""
    state = _INFERENCE
    logger.debug(
        "[_poll_compare_state] kind=%s status=%s busy=%s",
        state.kind, state.status, state.is_busy(),
    )

    if state.kind != "compare" or state.status not in (
        state.DONE, state.FAILED, state.CANCELLED,
    ):
        if state.is_busy():
            try:
                pid = state.proc.pid if state.proc else "?"
            except Exception:
                pid = "?"
            return (
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=_heartbeat(
                    f"⏳ {state.label} running (pid={pid})\n"
                    "💡 页面每秒自动刷新，完成后会在此显示两个视频"
                )),
            )
        return (
            gr.update(),
            gr.update(),
            gr.update(interactive=True),
        )

    result_json = state.result_json
    exit_code = state.exit_code
    log_path = state.log_path
    now = _now()

    if state.status == state.DONE:
        data = read_result_json(result_json) if result_json else None
        base_video = Path(data["base_video"]) if data and data.get("base_video") else None
        ft_video = Path(data["ft_video"]) if data and data.get("ft_video") else None

        if not base_video or not ft_video or not base_video.exists() or not ft_video.exists():
            _consume_state()
            return (
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=(
                    f"[{now}] ❌ 推理状态为完成，但输出视频缺失\n"
                    f"📜 log: {log_path}\n"
                    "请检查 compare 子进程日志确认视频是否写入成功。"
                )),
            )

        duration = data.get("duration_sec", 0.0) if data else 0.0
        logger.info("[_poll_compare_state] compare done: %s %s (%.1fs)", base_video, ft_video, duration)
        _consume_state()
        return (
            gr.update(value=str(base_video.resolve())),
            gr.update(value=str(ft_video.resolve())),
            gr.update(value=(
                f"[{now}] ✅ 对比推理完成\n"
                f"⏱ 总耗时: {duration:.1f} 秒\n"
                f"📜 log: {log_path}"
            )),
        )

    if state.status == state.CANCELLED:
        _consume_state()
        return (
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=(
                f"[{now}] ⏹ 已取消\n"
                f"📜 log: {log_path}\n"
                f"最后 30 行:\n{tail_file(log_path, 30)}"
            )),
        )

    # FAILED
    _consume_state()
    err_text = (
        f"[{now}] ❌ 推理失败 (rc={exit_code})\n"
        f"📜 log: {log_path}\n\n"
        f"最后 30 行:\n{tail_file(log_path, 30)}"
    )
    return (
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=err_text),
    )


def stop_inference() -> str:
    """⏹ cancel button — sends SIGINT to the running inference subprocess group."""
    return _INFERENCE.stop()


def _cancel_validate() -> Tuple[str, Any]:
    """⏹ Tab 3.5 cancel — like stop_inference but matches the two outputs
    wired in the UI (report box + re-enable the launch button)."""
    return _INFERENCE.stop(), gr.update(interactive=True)
