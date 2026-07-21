"""Tab 1: configuration, launch, and dataset preparation callbacks."""
import os
import shlex
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, List, Tuple

import gradio as gr
import yaml
from omegaconf import OmegaConf

from latentsync.finetune import (
    REPO_ROOT,
    ASSETS_DIR,
    FINETUNE_BASE_DIR,
    PREBUILT_DATASETS_YAML,
    logger,
)
from latentsync.finetune.config import PRESETS, build_config_from_form
from latentsync.finetune.process import _TRAINER
from latentsync.finetune.utils import _resolve_output_dir, tail_file

def one_click_launch(
    preset_dd: str, train_data_dir: str, train_fileslist: str,
    val_video_path: str, val_audio_path: str, resume_ckpt: str,
    batch_size: int, num_frames: int, resolution: int, learning_rate: float,
    use_motion_module: bool, pixel_space_supervise: bool, use_syncnet: bool,
    sync_loss_weight: float, perceptual_loss_weight: float,
    recon_loss_weight: float, trepa_loss_weight: float,
    mixed_precision_training: bool, enable_gradient_checkpointing: bool,
    mask_image_path: str, save_ckpt_steps: int, max_train_steps: int,
    num_workers: int, train_output_dir: str, freeze_attn2: bool,
    val_inference_steps: int, val_guidance_scale: float, val_seed: int,
    lr_scheduler: str, lr_warmup_steps: int,
    nproc_per_node: int, master_port: int, extra_env: str,
) -> str:
    """Top-of-tab "🚀 一键启动" button.

    Auto-fills empty fields with sensible defaults and calls
    launch_training. Updates the one-click status textbox with what was
    filled in so the user knows what was inferred.
    """
    defaults_used: List[str] = []

    # Fall back to assets demo files when nothing is provided — works
    # for a smoke test and proves the pipeline end-to-end.
    if not (val_video_path or "").strip():
        val_video_path = str(ASSETS_DIR / "demo1_video.mp4")
        defaults_used.append(f"val_video_path={val_video_path}")
    if not (val_audio_path or "").strip():
        val_audio_path = str(ASSETS_DIR / "demo1_audio.wav")
        defaults_used.append(f"val_audio_path={val_audio_path}")
    if not (resume_ckpt or "").strip():
        resume_ckpt = str(REPO_ROOT / "checkpoints" / "latentsync_unet.pt")
        defaults_used.append(f"resume_ckpt={resume_ckpt}")
    if not (mask_image_path or "").strip():
        mask_image_path = "latentsync/utils/mask.png"
        defaults_used.append("mask_image_path=latentsync/utils/mask.png")

    # If still no data dir, try prebuilt dataset init for the preset's
    # typical_use. Falls back to a friendly message.
    if not (train_data_dir or "").strip() and (train_fileslist or "").strip() == "":
        hint = (
            "⚠️ train_data_dir 和 train_fileslist 都为空。\n"
            "→ 用 📚 预制数据集(同 Tab 顶部)下一份,或者自己填字段。\n"
            "   例如 Tab 1 「📚 预制数据集」选 celebv_hq_side → ⬇ → 自动填。"
        )
        return hint

    # Delegate to launch_training with the resolved args.
    result = launch_training(
        preset_dd, train_data_dir, train_fileslist,
        val_video_path, val_audio_path, resume_ckpt,
        batch_size, num_frames, resolution, learning_rate,
        use_motion_module, pixel_space_supervise, use_syncnet,
        sync_loss_weight, perceptual_loss_weight, recon_loss_weight,
        trepa_loss_weight, mixed_precision_training,
        enable_gradient_checkpointing, mask_image_path,
        save_ckpt_steps, max_train_steps, num_workers, train_output_dir,
        freeze_attn2, val_inference_steps, val_guidance_scale, val_seed,
        lr_scheduler, lr_warmup_steps,
        nproc_per_node, master_port, extra_env,
    )
    status, log_path = result
    fill_note = ""
    if defaults_used:
        fill_note = "\n🪄 自动填字段:\n  - " + "\n  - ".join(defaults_used)
    return f"{status}{fill_note}"

def launch_training(
    preset_name: str,
    train_data_dir: str,
    train_fileslist: str,
    val_video_path: str,
    val_audio_path: str,
    resume_ckpt: str,
    batch_size: int,
    num_frames: int,
    resolution: int,
    learning_rate: float,
    use_motion_module: bool,
    pixel_space_supervise: bool,
    use_syncnet: bool,
    sync_loss_weight: float,
    perceptual_loss_weight: float,
    recon_loss_weight: float,
    trepa_loss_weight: float,
    mixed_precision_training: bool,
    enable_gradient_checkpointing: bool,
    mask_image_path: str,
    save_ckpt_steps: int,
    max_train_steps: int,
    num_workers: int,
    train_output_dir: str,
    freeze_attn2: bool,
    val_inference_steps: int,
    val_guidance_scale: float,
    val_seed: int,
    lr_scheduler: str,
    lr_warmup_steps: int,
    nproc_per_node: int,
    master_port: int,
    extra_env: str,
) -> Tuple[str, str]:
    """Build a config yaml, spawn the training subprocess, return status."""
    try:
        logger.info(
            "[launch_training] called with preset=%s data_dir=%s fileslist=%s",
            preset_name, train_data_dir, train_fileslist,
        )

        if _TRAINER.is_running():
            logger.warning(
                "[launch_training] rejected: another training is already running (pid=%s)",
                _TRAINER.pid,
            )
            run_name = _TRAINER.run_dir.name if _TRAINER.run_dir else "?"
            return (
                f"❌ 训练已在运行中 (pid={_TRAINER.pid}, run={run_name})",
                "",
            )

        try:
            cfg = build_config_from_form(
                preset_name=preset_name,
                train_data_dir=train_data_dir,
                train_fileslist=train_fileslist,
                val_video_path=val_video_path,
                val_audio_path=val_audio_path,
                resume_ckpt=resume_ckpt,
                batch_size=batch_size,
                num_frames=num_frames,
                resolution=resolution,
                learning_rate=learning_rate,
                use_motion_module=use_motion_module,
                pixel_space_supervise=pixel_space_supervise,
                use_syncnet=use_syncnet,
                sync_loss_weight=sync_loss_weight,
                perceptual_loss_weight=perceptual_loss_weight,
                recon_loss_weight=recon_loss_weight,
                trepa_loss_weight=trepa_loss_weight,
                mixed_precision_training=mixed_precision_training,
                enable_gradient_checkpointing=enable_gradient_checkpointing,
                mask_image_path=mask_image_path,
                save_ckpt_steps=save_ckpt_steps,
                max_train_steps=max_train_steps,
                num_workers=num_workers,
                train_output_dir=train_output_dir,
                freeze_attn2=freeze_attn2,
                val_inference_steps=val_inference_steps,
                val_guidance_scale=val_guidance_scale,
                val_seed=val_seed,
                lr_scheduler=lr_scheduler,
                lr_warmup_steps=lr_warmup_steps,
            )
        except ValueError as exc:
            return (f"❌ 配置错误: {exc}", "")
        logger.info("[launch_training] config built, train_output_dir=%s", cfg["data"]["train_output_dir"])

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg_dir = FINETUNE_BASE_DIR / "generated_configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg_dir / f"{preset_name.split()[0].lower()}_{ts}.yaml"
        cfg["unet_config_path"] = str(cfg_path)
        with open(cfg_path, "w") as f:
            yaml.dump(OmegaConf.to_container(OmegaConf.create(cfg)), f, sort_keys=False)
        logger.info("[launch_training] config written to %s", cfg_path)

        is_syncnet = "syncnet" in PRESETS[preset_name]["config_file"]
        is_lora = bool(PRESETS[preset_name].get("lora", {}).get("enabled", False))

        if is_syncnet:
            script = "scripts.train_syncnet"
            config_arg = "--config_path"
        elif is_lora:
            script = "scripts.train_unet_lora"
            config_arg = "--unet_config_path"
        else:
            script = "scripts.train_unet"
            config_arg = "--unet_config_path"

        cmd = [
            "torchrun",
            f"--nproc_per_node={int(nproc_per_node)}",
            f"--master_port={int(master_port)}",
            "-m",
            script,
            config_arg,
            str(cfg_path),
        ]
        logger.info("[launch_training] command: %s", " ".join(shlex.quote(c) for c in cmd))

        log_dir = FINETUNE_BASE_DIR / "training_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_kind = "syncnet" if is_syncnet else ("unet_lora" if is_lora else "unet")
        log_path = log_dir / f"{log_kind}_{ts}.log"

        env = os.environ.copy()
        # Help PyTorch avoid CUDA memory fragmentation during long training runs,
        # especially at high resolution (512) where VAE decode activations are large.
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if extra_env.strip():
            for kv in extra_env.strip().splitlines():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    env[k.strip()] = v.strip()

        logger.info("[launch_training] spawning subprocess ...")
        try:
            log_f = open(log_path, "w")
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid,
            )
        except FileNotFoundError as e:
            logger.error("[launch_training] torchrun not found: %s", e)
            return (
                f"❌ 启动失败：{e}\n请确保 torchrun 在 PATH 中",
                str(log_path),
            )

        _TRAINER.proc = proc
        _TRAINER.log_path = log_path
        _TRAINER.run_dir = _resolve_output_dir(cfg["data"]["train_output_dir"])
        _TRAINER.started_at = datetime.now().isoformat(timespec="seconds")
        _TRAINER.cmd = cmd
        _TRAINER.save_state()

        status = (
            f"✅ 已启动 (pid={proc.pid})\n"
            f"📄 config: {cfg_path}\n"
            f"📜 log:    {log_path}\n"
            f"📦 产物目录: {cfg['data']['train_output_dir']}/train-<timestamp>\n"
            f"🕐 启动时间: {_TRAINER.started_at}\n\n"
            f"💡 命令行：\n  {' '.join(shlex.quote(c) for c in cmd)}"
        )
        logger.info("[launch_training] started: pid=%s log=%s", proc.pid, log_path)
        return status, str(log_path)
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("[launch_training] unhandled exception: %s", e)
        return (
            f"❌ 启动时发生异常：{e}\n\n详细堆栈：\n{tb}",
            "",
        )

def stop_training() -> str:
    if not _TRAINER.is_running():
        return "ℹ️ 没有正在运行的训练任务"
    pid = _TRAINER.pid
    _TRAINER.stop()
    return f"⏹ 已停止 (pid={pid})"
def refresh_training_log() -> Tuple[str, str]:
    """Return (latest log tail, process status) for the running/last training."""
    if not _TRAINER.log_path or not _TRAINER.log_path.exists():
        status = "ℹ️ 暂无训练日志"
        if _TRAINER.is_running():
            status = f"🟡 训练已启动但日志尚未写入 (pid={_TRAINER.pid})"
        return "(log file not found yet)", status

    log_text = tail_file(_TRAINER.log_path, n_lines=80)

    if _TRAINER.is_running():
        status = f"🟢 训练中 (pid={_TRAINER.pid}, started={_TRAINER.started_at})"
    else:
        # After the process finishes, clear persisted state.
        _TRAINER.clear_state()
        status = "ℹ️ 无正在运行的训练任务"
    return log_text, status
def ping_backend() -> str:
    """Simple connectivity check."""
    logger.info("[ping_backend] received ping")
    return f"✅ 后端连通 (pid={os.getpid()}, time={datetime.now().isoformat()})"
def debug_all_inputs(*args) -> str:
    """Accept any number of inputs and echo them back for diagnostics."""
    logger.info("[debug_all_inputs] received %d args", len(args))
    for i, arg in enumerate(args):
        logger.info("[debug_all_inputs] arg[%d] type=%s value=%r", i, type(arg).__name__, arg)
    lines = [f"收到 {len(args)} 个参数："]
    for i, arg in enumerate(args):
        preview = repr(arg)
        if len(preview) > 200:
            preview = preview[:200] + "..."
        lines.append(f"  arg[{i}] ({type(arg).__name__}): {preview}")
    return "\n".join(lines)
def _run_curate_finetune(
    urls: str,
    source_dir: str,
    output_dir: str,
    scale: str,
) -> Iterator[str]:
    """Spawn `python -m tools.download_curated_finetune_set` and stream output.

    Generator (like ``_run_init_prebuilt``): curation downloads can run for
    hours, and a blocking subprocess.run would freeze the Gradio queue for
    every other user action until it finished.
    """
    urls = (urls or "").strip()
    source_dir = (source_dir or "").strip()
    if not urls and not source_dir:
        yield "❌ 必须填 URL 列表 或 本地源目录(选一)"
        return
    if not output_dir:
        yield "❌ curated 输出目录为空"
        return

    cmd = [
        sys.executable, "-m", "tools.download_curated_finetune_set",
        "--output-dir", output_dir,
    ]
    if urls:
        cmd += ["--urls", urls]
    if source_dir:
        cmd += ["--source-dir", source_dir]
    if scale:
        cmd += ["--scale", scale]

    log_path = REPO_ROOT / "debug" / f"curate_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    yield (
        f"🚀 开始 curate\n"
        f"📜 实时日志: {log_path}\n"
        f"   可另开终端查看: tail -f {log_path}\n"
    )
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(
                cmd, cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT,
            )
            while proc.poll() is None:
                time.sleep(0.5)
                yield (
                    f"📥 curate 处理中...\n"
                    f"📜 log: {log_path}\n\n{tail_file(log_path, 40)}"
                )
    except FileNotFoundError as exc:
        yield f"❌ 启动失败: {exc}"
        return

    log_text = log_path.read_text()
    if proc.returncode != 0:
        yield (
            f"❌ curate 退出码 {proc.returncode}\n"
            f"📜 log: {log_path}\n\n最后 60 行:\n{log_text[-6000:]}"
        )
        return
    yield f"✅ 已完成\n📜 log: {log_path}\n\n{log_text[-4000:]}"
def _prebuilt_choices() -> List[str]:
    """Read tools/prebuilt_datasets.yaml and return a list of `id (name)` strings
    for the gradio Dropdown."""
    if not PREBUILT_DATASETS_YAML.exists():
        return []
    try:
        import yaml as _yaml
        raw = _yaml.safe_load(PREBUILT_DATASETS_YAML.read_text()) or {}
    except Exception as exc:
        logger.warning("could not parse %s: %s", PREBUILT_DATASETS_YAML, exc)
        return []
    out: List[str] = []
    for entry in raw.get("datasets", []):
        if "id" in entry:
            out.append(f"{entry['id']} — {entry.get('name', '')}")
    return out

def _run_init_prebuilt(
    dataset_choice: str, output_dir: str, hf_token: str,
) -> Iterator[Tuple[str, Any, Any]]:
    """Spawn `python -m tools.init_finetune_dataset` and stream output.

    Yields (log_text, train_data_dir_value, train_fileslist_value) so the
    gradio textbox updates in real time and the launch form is auto-filled
    on success.
    """
    dataset_choice = (dataset_choice or "").strip()
    output_dir = (output_dir or "").strip()
    if not dataset_choice:
        yield ("❌ 请先选一个预制数据集(下拉里选)", gr.update(), gr.update())
        return
    if not output_dir:
        yield ("❌ 输出目录为空", gr.update(), gr.update())
        return
    dataset_id = dataset_choice.split(" — ")[0].strip()

    cmd = [
        sys.executable, "-m", "tools.init_finetune_dataset",
        "--dataset", dataset_id,
        "--output-dir", output_dir,
    ]
    hf_token = (hf_token or "").strip()
    if hf_token:
        cmd += ["--hf-token", hf_token]
    log_path = REPO_ROOT / "debug" / f"init_prebuilt_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    yield (
        f"🚀 开始初始化数据集 `{dataset_id}`\n"
        f"📜 实时日志: {log_path}\n"
        f"   可另开终端查看: tail -f {log_path}\n"
        f"   (窗口下方会每 0.5s 刷新一次最新日志)\n",
        gr.update(), gr.update(),
    )

    try:
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(
                cmd, cwd=str(REPO_ROOT),
                stdout=logf, stderr=subprocess.STDOUT,
            )
            while proc.poll() is None:
                time.sleep(0.5)
                tail = tail_file(log_path, 40, missing_msg="")
                yield (
                    f"🚀 `{dataset_id}` 处理中...\n"
                    f"📜 log: {log_path}\n\n{tail}",
                    gr.update(), gr.update(),
                )
    except FileNotFoundError as exc:
        yield (f"❌ 启动失败: {exc}\n📜 log: {log_path}", gr.update(), gr.update())
        return

    log_text = log_path.read_text()
    if proc.returncode != 0:
        yield (
            f"❌ init 退出码 {proc.returncode}\n"
            f"📜 log: {log_path}\n\n最后 60 行:\n{log_text[-6000:]}",
            gr.update(), gr.update(),
        )
        return

    curated_dir = (Path(output_dir).resolve() / dataset_id / "curated")
    fileslist = curated_dir / "fileslist.txt"
    summary = (
        f"✅ 完成 — 已自动填到 launch 表单\n"
        f"   train_data_dir:  {curated_dir}\n"
        f"   train_fileslist: {fileslist}\n"
        f"📜 log: {log_path}\n\n"
        f"{log_text[-4000:]}"
    )
    yield (
        summary,
        str(curated_dir),
        str(fileslist),
    )

def _run_merge_lora(
    base_ckpt: str,
    adapter_dir: str,
    out_ckpt: str,
    push_repo: str,
) -> str:
    """Spawn `python -m scripts.merge_lora` and stream its output to a textbox."""
    base_ckpt = (base_ckpt or "").strip()
    adapter_dir = (adapter_dir or "").strip()
    out_ckpt = (out_ckpt or "").strip()
    push_repo = (push_repo or "").strip()

    if not base_ckpt or not Path(base_ckpt).exists():
        return f"❌ base UNet ckpt 不存在: {base_ckpt}"
    if not adapter_dir or not Path(adapter_dir).exists():
        return f"❌ adapter 目录不存在: {adapter_dir}"
    if not out_ckpt:
        return "❌ 合并输出路径为空"
    adapter_cfg = Path(adapter_dir) / "adapter_config.json"
    if not adapter_cfg.exists():
        return (
            f"❌ {adapter_dir} 不是 peft adapter 目录 (缺少 adapter_config.json)。"
            "请指向 train_unet_lora.py 输出的 checkpoints/<step>/ 那一层。"
        )

    cmd = [
        sys.executable, "-m", "scripts.merge_lora",
        "--base_ckpt", base_ckpt,
        "--adapter_dir", adapter_dir,
        "--out_ckpt", out_ckpt,
        "--unet_config_path", "configs/unet/stage2.yaml",
    ]
    if push_repo:
        cmd += ["--push_to_hub", push_repo]

    log_path = REPO_ROOT / "debug" / f"merge_lora_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=logf,
                stderr=subprocess.STDOUT,
                check=False,
            )
        log_text = log_path.read_text()
        if proc.returncode != 0:
            return (
                f"❌ merge_lora 退出码 {proc.returncode},完整日志:{log_path}\n\n"
                f"{log_text[-4000:]}"
            )
        summary = [
            f"✅ merged → {out_ckpt}",
        ]
        if push_repo:
            summary.append(f"☁️  pushed to https://huggingface.co/{push_repo}")
        summary.append(f"📄  full log: {log_path}")
        return "\n".join(summary) + "\n\n" + log_text[-2000:]
    except FileNotFoundError as exc:
        return f"❌ 启动失败:{exc}"
