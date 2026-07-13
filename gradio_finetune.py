"""
Gradio Fine-tuning UI for LatentSync.

A visual page for configuring, launching, and monitoring UNet / SyncNet
fine-tuning. The page has three tabs:

  Tab 1 - Configure: pick a preset (Stage1 / Stage2 / Stage2 Efficient /
           Stage2 512 / SyncNet), tweak key hyperparameters, select /
           upload the fine-tune dataset, and launch the training.
  Tab 2 - Monitor:   read the chosen run directory, render loss curves,
           the latest sync_conf curve, the most recent validation
           video, and a tail of the training log. Auto-refreshes.
  Tab 3 - Compare:   side-by-side inference of base ckpt vs the just-
           fine-tuned ckpt on the same input, for quick regression.

NOTE: actual GPU training must run on a GPU host. This UI launches the
training subprocess locally if a GPU is present, or it can be pointed at
an SSH host via the 'remote launch' option (writes a launch script).
"""

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import yaml
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "configs" / "unet"
SYNCNET_CONFIG_DIR = REPO_ROOT / "configs" / "syncnet"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
TRAIN_OUTPUT_DIR = REPO_ROOT / "debug"
ASSETS_DIR = REPO_ROOT / "assets"


# ---------------------------------------------------------------------------
# Presets - one-click sane defaults that mirror configs/unet/*.yaml
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    "Stage 1 (256, 全量训练)": {
        "config_file": "configs/unet/stage1.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 1e-5,
        "use_motion_module": False,
        "pixel_space_supervise": False,
        "use_syncnet": False,
        "sync_loss_weight": 0.05,
        "perceptual_loss_weight": 0.1,
        "recon_loss_weight": 1.0,
        "trepa_loss_weight": 10.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask.png",
        "description": "学视觉特征，不加 sync / lpips / trepa。23GB VRAM。",
    },
    "Stage 2 (256, 推荐)": {
        "config_file": "configs/unet/stage2.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 1e-5,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        "sync_loss_weight": 0.05,
        "perceptual_loss_weight": 0.1,
        "recon_loss_weight": 1.0,
        "trepa_loss_weight": 10.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask.png",
        "description": "冻结 UNet 主体，只训 motion + attn。30GB VRAM。",
    },
    "Stage 2 Efficient (256, 20GB)": {
        "config_file": "configs/unet/stage2_efficient.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 1e-5,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        "sync_loss_weight": 0.05,
        "perceptual_loss_weight": 0.1,
        "recon_loss_weight": 1.0,
        "trepa_loss_weight": 0.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask.png",
        "description": "关 TREPA，只训 motion + attn2。20GB VRAM。",
    },
    "Stage 2 512 (高分辨率)": {
        "config_file": "configs/unet/stage2_512.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        "num_frames": 16,
        "resolution": 512,
        "learning_rate": 1e-5,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        "sync_loss_weight": 0.05,
        "perceptual_loss_weight": 0.1,
        "recon_loss_weight": 1.0,
        "trepa_loss_weight": 10.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask2.png",
        "description": "512 分辨率。55GB VRAM。",
    },
    "Stage 2 LoRA (256, 12-15GB)": {
        "config_file": "configs/unet/stage2.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 5e-5,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        "sync_loss_weight": 0.05,
        "perceptual_loss_weight": 0.1,
        "recon_loss_weight": 1.0,
        "trepa_loss_weight": 10.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask.png",
        "description": "LoRA 微调：adapter 仅 ~10MB，可多任务切换。需 pip install peft。",
        "lora": {
            "enabled": True,
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["to_q", "to_k", "to_v", "to_out"],
            "qlora": False,
        },
    },
    "Stage 2 QLoRA (256, 8-10GB)": {
        "config_file": "configs/unet/stage2.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 2e-4,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        "sync_loss_weight": 0.05,
        "perceptual_loss_weight": 0.1,
        "recon_loss_weight": 1.0,
        "trepa_loss_weight": 10.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask.png",
        "description": "QLoRA：base UNet 4-bit 量化 + LoRA。需 peft + bitsandbytes。",
        "lora": {
            "enabled": True,
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["to_q", "to_k", "to_v", "to_out"],
            "qlora": True,
        },
    },
    "SyncNet 训练": {
        "config_file": "configs/syncnet/syncnet_16_pixel_attn.yaml",
        "resume_ckpt": "",
        "batch_size": 256,
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 1e-5,
        "use_motion_module": False,
        "pixel_space_supervise": False,
        "use_syncnet": False,
        "sync_loss_weight": 0.0,
        "perceptual_loss_weight": 0.0,
        "recon_loss_weight": 0.0,
        "trepa_loss_weight": 0.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": False,
        "mask_image_path": "latentsync/utils/mask.png",
        "description": "训练 StableSyncNet。batch 建议 ≥256，最好 1024。",
    },
}


# ---------------------------------------------------------------------------
# Process management for background training
# ---------------------------------------------------------------------------

class TrainingProcess:
    """Track a single background training subprocess."""

    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.log_path: Optional[Path] = None
        self.run_dir: Optional[Path] = None
        self.started_at: Optional[str] = None
        self.cmd: List[str] = []

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGINT)
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.proc = None


_TRAINER = TrainingProcess()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_run_dirs(base: Path) -> List[str]:
    """List timestamped run directories produced by train_unet.py / train_syncnet.py."""
    if not base.exists():
        return []
    runs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("train-")])
    return [str(p.relative_to(REPO_ROOT)) for p in runs]


def list_datasets() -> List[str]:
    """Candidate fine-tune datasets: any directory under preprocess/ with mp4 files."""
    candidates: List[str] = []
    for root in (REPO_ROOT, REPO_ROOT / "preprocess", REPO_ROOT / "data"):
        if not root.exists():
            continue
        for path in root.rglob("*.mp4"):
            parent = path.parent
            if parent.name in {"high_visual_quality", "segmented", "affine_transformed"}:
                rel = str(parent.relative_to(REPO_ROOT))
                if rel not in candidates:
                    candidates.append(rel)
    return sorted(candidates)


def list_checkpoints() -> List[str]:
    """Available UNet / SyncNet checkpoints under checkpoints/."""
    if not CHECKPOINT_DIR.exists():
        return []
    out: List[str] = []
    for p in sorted(CHECKPOINT_DIR.rglob("*.pt")):
        out.append(str(p.relative_to(REPO_ROOT)))
    return out


def build_config_from_form(
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
) -> Dict[str, Any]:
    """Merge user-form values with the chosen preset's defaults."""
    preset = PRESETS[preset_name]
    cfg: Dict[str, Any] = {
        "data": {
            "train_data_dir": train_data_dir or "",
            "train_fileslist": train_fileslist or "",
            "val_video_path": val_video_path or str(ASSETS_DIR / "demo1_video.mp4"),
            "val_audio_path": val_audio_path or str(ASSETS_DIR / "demo1_audio.wav"),
            "audio_embeds_cache_dir": str(REPO_ROOT / "debug" / "audio_embeds_cache"),
            "audio_mel_cache_dir": str(REPO_ROOT / "debug" / "audio_mel_cache"),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "num_frames": int(num_frames),
            "resolution": int(resolution),
            "mask_image_path": mask_image_path,
            "audio_sample_rate": 16000,
            "video_fps": 25,
            "audio_feat_length": [2, 2],
            "train_output_dir": train_output_dir or "debug/unet",
            "syncnet_config_path": preset["config_file"]
            if "syncnet" not in preset["config_file"]
            else preset["config_file"],
        },
        "ckpt": {
            "resume_ckpt_path": resume_ckpt or preset["resume_ckpt"],
            "save_ckpt_steps": int(save_ckpt_steps),
        },
        "run": {
            "pixel_space_supervise": bool(pixel_space_supervise),
            "use_syncnet": bool(use_syncnet),
            "sync_loss_weight": float(sync_loss_weight),
            "perceptual_loss_weight": float(perceptual_loss_weight),
            "recon_loss_weight": float(recon_loss_weight),
            "trepa_loss_weight": float(trepa_loss_weight),
            "guidance_scale": 1.5,
            "inference_steps": 20,
            "seed": 1247,
            "use_mixed_noise": True,
            "mixed_noise_alpha": 1,
            "mixed_precision_training": bool(mixed_precision_training),
            "enable_gradient_checkpointing": bool(enable_gradient_checkpointing),
            "max_train_steps": int(max_train_steps),
            "max_train_epochs": -1,
        },
        "optimizer": {
            "lr": float(learning_rate),
            "scale_lr": False,
            "max_grad_norm": 1.0,
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,
        },
        "model": {
            "act_fn": "silu",
            "add_audio_layer": True,
            "attention_head_dim": 8,
            "block_out_channels": [320, 640, 1280, 1280],
            "center_input_sample": False,
            "cross_attention_dim": 384,
            "down_block_types": [
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
                "DownBlock3D",
            ],
            "mid_block_type": "UNetMidBlock3DCrossAttn",
            "up_block_types": [
                "UpBlock3D",
                "CrossAttnUpBlock3D",
                "CrossAttnUpBlock3D",
                "CrossAttnUpBlock3D",
            ],
            "downsample_padding": 1,
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "in_channels": 13,
            "layers_per_block": 2,
            "mid_block_scale_factor": 1,
            "norm_eps": 1e-5,
            "norm_num_groups": 32,
            "out_channels": 4,
            "sample_size": 64,
            "resnet_time_scale_shift": "default",
            "use_motion_module": bool(use_motion_module),
            "motion_module_resolutions": [1, 2, 4, 8],
            "motion_module_mid_block": False,
            "motion_module_decoder_only": False,
            "motion_module_type": "Vanilla",
            "motion_module_kwargs": {
                "num_attention_heads": 8,
                "num_transformer_block": 1,
                "attention_block_types": ["Temporal_Self", "Temporal_Self"],
                "temporal_position_encoding": True,
                "temporal_position_encoding_max_len": 24,
                "temporal_attention_dim_div": 1,
                "zero_initialize": True,
            },
        },
    }

    # If the preset carries a LoRA block, propagate it into the generated
    # config. train_unet_lora.py will pick it up; train_unet.py will
    # simply ignore it (it has its own trainable_modules logic).
    if "lora" in preset:
        cfg["lora"] = dict(preset["lora"])
    else:
        # Default-off block so users can hand-edit the generated yaml
        cfg["lora"] = {
            "enabled": False,
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["to_q", "to_k", "to_v", "to_out"],
            "qlora": False,
            "freeze_attn2": False,
        }
    return cfg


# ---------------------------------------------------------------------------
# Tab 1: Configure & launch
# ---------------------------------------------------------------------------

def on_preset_change(preset_name: str) -> Tuple[Any, ...]:
    """When user picks a preset, fill the form fields with preset defaults."""
    preset = PRESETS[preset_name]
    return (
        preset["batch_size"],
        preset["num_frames"],
        preset["resolution"],
        preset["learning_rate"],
        preset["use_motion_module"],
        preset["pixel_space_supervise"],
        preset["use_syncnet"],
        preset["sync_loss_weight"],
        preset["perceptual_loss_weight"],
        preset["recon_loss_weight"],
        preset["trepa_loss_weight"],
        preset["mixed_precision_training"],
        preset["enable_gradient_checkpointing"],
        preset["mask_image_path"],
        preset["resume_ckpt"],
        preset["description"],
    )


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
    nproc_per_node: int,
    master_port: int,
    extra_env: str,
) -> Tuple[str, str, str]:
    """Build a config yaml, spawn the training subprocess, return status."""

    if _TRAINER.is_running():
        return (
            f"❌ 训练已在运行中 (pid={_TRAINER.proc.pid}, run={_TRAINER.run_dir.name})",
            "",
            "",
        )

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
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_dir = REPO_ROOT / "debug" / "generated_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{preset_name.split()[0].lower()}_{ts}.yaml"
    cfg["unet_config_path"] = str(cfg_path)
    with open(cfg_path, "w") as f:
        yaml.dump(OmegaConf.to_container(OmegaConf.create(cfg)), f, sort_keys=False)

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

    log_dir = REPO_ROOT / "debug" / "training_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_kind = "syncnet" if is_syncnet else ("unet_lora" if is_lora else "unet")
    log_path = log_dir / f"{log_kind}_{ts}.log"

    env = os.environ.copy()
    if extra_env.strip():
        for kv in extra_env.strip().splitlines():
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k.strip()] = v.strip()

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
        return (
            f"❌ 启动失败：{e}\n请确保 torchrun 在 PATH 中",
            str(log_path),
            "",
        )

    _TRAINER.proc = proc
    _TRAINER.log_path = log_path
    _TRAINER.run_dir = REPO_ROOT / cfg["data"]["train_output_dir"]  # will be created with timestamp
    _TRAINER.started_at = datetime.now().isoformat(timespec="seconds")
    _TRAINER.cmd = cmd

    status = (
        f"✅ 已启动 (pid={proc.pid})\n"
        f"📄 config: {cfg_path.relative_to(REPO_ROOT)}\n"
        f"📜 log:    {log_path.relative_to(REPO_ROOT)}\n"
        f"📦 产物目录: {cfg['data']['train_output_dir']}/train-<timestamp>\n"
        f"🕐 启动时间: {_TRAINER.started_at}\n\n"
        f"💡 命令行：\n  {' '.join(shlex.quote(c) for c in cmd)}"
    )
    return status, str(log_path), gr.update(choices=list_run_dirs(REPO_ROOT / cfg["data"]["train_output_dir"]))


def stop_training() -> str:
    if not _TRAINER.is_running():
        return "ℹ️ 没有正在运行的训练任务"
    pid = _TRAINER.proc.pid
    _TRAINER.stop()
    return f"⏹ 已停止 (pid={pid})"


def refresh_runs(train_output_dir: str) -> gr.update:
    base = REPO_ROOT / train_output_dir if train_output_dir else REPO_ROOT / "debug/unet"
    return gr.update(choices=list_run_dirs(base))


# ---------------------------------------------------------------------------
# Tab 2: Monitor
# ---------------------------------------------------------------------------

def tail_log(log_path: Optional[str], n_lines: int = 80) -> str:
    if not log_path or not Path(log_path).exists():
        return "(log file not found yet - training may not have started writing)"
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 50_000))
            data = f.read().decode("utf-8", errors="replace")
        return "\n".join(data.splitlines()[-int(n_lines):])
    except Exception as e:
        return f"(error reading log: {e})"


def parse_loss_chart(run_dir_path: Optional[str]) -> Optional[str]:
    """LatentSync plots loss charts via plot_loss_chart(); we surface the
    latest PNG from loss_charts/ or sync_conf_results/."""
    if not run_dir_path:
        return None
    rd = Path(run_dir_path)
    if not rd.is_absolute():
        rd = REPO_ROOT / rd
    for sub in ("loss_charts", "sync_conf_results"):
        d = rd / sub
        if d.exists():
            pngs = sorted(d.glob("*.png"), key=lambda p: p.stat().st_mtime)
            if pngs:
                return str(pngs[-1])
    return None


def list_validation_videos(run_dir_path: Optional[str]) -> List[str]:
    if not run_dir_path:
        return []
    rd = Path(run_dir_path)
    if not rd.is_absolute():
        rd = REPO_ROOT / rd
    vd = rd / "val_videos"
    if not vd.exists():
        return []
    return sorted([str(p) for p in vd.glob("*.mp4")], key=lambda p: p.stat().st_mtime, reverse=True)


def list_checkpoints_in_run(run_dir_path: Optional[str]) -> List[str]:
    if not run_dir_path:
        return []
    rd = Path(run_dir_path)
    if not rd.is_absolute():
        rd = REPO_ROOT / rd
    ck = rd / "checkpoints"
    if not ck.exists():
        return []
    return [str(p.relative_to(REPO_ROOT)) for p in sorted(ck.glob("*.pt"))]


def read_loss_from_checkpoint(ckpt_path: str) -> str:
    """Best-effort: dump global_step + a couple of scalar fields from the
    latest checkpoint so the user gets a textual progress signal without
    loading the full state."""
    if not ckpt_path or not Path(ckpt_path).exists():
        return "(checkpoint not found)"
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        info = {
            "global_step": ckpt.get("global_step", "?"),
            "state_dict_keys": len(ckpt.get("state_dict", {})),
            "train_step_list_len": len(ckpt.get("train_step_list", []))
            if "train_step_list" in ckpt else "n/a",
            "train_loss_list_len": len(ckpt.get("train_loss_list", []))
            if "train_loss_list" in ckpt else "n/a",
        }
        if "train_loss_list" in ckpt and ckpt["train_loss_list"]:
            tail = ckpt["train_loss_list"][-5:]
            info["last_5_train_losses"] = [round(float(x), 4) for x in tail]
        if "val_loss_list" in ckpt and ckpt["val_loss_list"]:
            tail = ckpt["val_loss_list"][-5:]
            info["last_5_val_losses"] = [round(float(x), 4) for x in tail]
        return json.dumps(info, indent=2)
    except Exception as e:
        return f"(could not read checkpoint: {e})"


def monitor_refresh(
    train_output_dir: str,
    selected_run: Optional[str],
    log_path: Optional[str],
) -> Tuple[str, str, Any, str, str, Any]:
    """Pull the latest snapshot. Returns: (run_dir_choices, selected_run_disp,
    loss_chart, val_video_choices, log_tail, ckpt_info)."""
    base = REPO_ROOT / train_output_dir if train_output_dir else REPO_ROOT / "debug/unet"
    run_choices = list_run_dirs(base)
    run_dir = (REPO_ROOT / selected_run) if selected_run else None
    chart = parse_loss_chart(str(run_dir) if run_dir else None)
    val_videos = list_validation_videos(str(run_dir) if run_dir else None)
    log_text = tail_log(log_path, n_lines=80)
    ckpts = list_checkpoints_in_run(str(run_dir) if run_dir else None)
    if ckpts:
        ckpt_info = read_loss_from_checkpoint(REPO_ROOT / ckpts[-1])
    else:
        ckpt_info = "(no checkpoint yet)"
    status = (
        f"📌 trainer running: {_TRAINER.is_running()} | "
        f"pid: {_TRAINER.proc.pid if _TRAINER.proc else '-'} | "
        f"started: {_TRAINER.started_at or '-'}"
    )
    return (
        gr.update(choices=run_choices),
        str(run_dir) if run_dir else "",
        chart,
        val_videos,
        log_text,
        ckpt_info,
        status,
    )


# ---------------------------------------------------------------------------
# Tab 3: Compare two checkpoints (inference)
# ---------------------------------------------------------------------------

def run_compare(
    video_path: str,
    audio_path: str,
    base_ckpt: str,
    fine_tuned_ckpt: str,
    inference_steps: int,
    guidance_scale: float,
    seed: int,
    resolution: int,
) -> Tuple[str, str]:
    """Run inference twice (base vs fine-tuned) and return the two mp4 paths."""
    if not video_path or not audio_path:
        raise gr.Error("请先上传视频和音频")
    if not base_ckpt or not fine_tuned_ckpt:
        raise gr.Error("请选择 base 和 fine-tuned 两个 checkpoint")
    if base_ckpt == fine_tuned_ckpt:
        raise gr.Error("两个 checkpoint 必须不同")

    out_dir = REPO_ROOT / "debug" / "compare_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = out_dir / f"base_{ts}.mp4"
    out_ft = out_dir / f"finetuned_{ts}.mp4"

    config_path = CONFIG_DIR / "stage2.yaml"

    def _run(ckpt_path: Path, out_path: Path) -> None:
        cmd = [
            "python",
            "-m",
            "scripts.inference",
            "--unet_config_path",
            str(config_path),
            "--inference_ckpt_path",
            str(ckpt_path),
            "--video_path",
            str(video_path),
            "--audio_path",
            str(audio_path),
            "--video_out_path",
            str(out_path),
            "--inference_steps",
            str(int(inference_steps)),
            "--guidance_scale",
            str(float(guidance_scale)),
            "--seed",
            str(int(seed)),
            "--temp_dir",
            "temp",
        ]
        # resolution override
        cfg = OmegaConf.load(config_path)
        cfg.data.resolution = int(resolution)
        cfg.run.inference_steps = int(inference_steps)
        cfg.run.guidance_scale = float(guidance_scale)
        tmp_cfg = REPO_ROOT / "debug" / f"compare_cfg_{ts}.yaml"
        tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_cfg, "w") as f:
            yaml.dump(OmegaConf.to_container(cfg), f)
        cmd[cmd.index("--unet_config_path") + 1] = str(tmp_cfg)
        log = open(out_path.with_suffix(".log"), "w")
        rc = subprocess.call(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        if rc != 0:
            raise gr.Error(f"inference failed for {ckpt_path.name}, see {out_path.with_suffix('.log')}")

    _run(REPO_ROOT / base_ckpt, out_base)
    _run(REPO_ROOT / fine_tuned_ckpt, out_ft)
    return str(out_base), str(out_ft)


# ---------------------------------------------------------------------------
# Gradio UI assembly
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="LatentSync Fine-tune Studio",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
# 🎛 LatentSync Fine-tune Studio

可视化配置 / 启动 / 监控 / 对比 UNet 与 SyncNet 的微调训练。

> ⚠️ **GPU 提示**：本页会本地拉起 `torchrun`。如果没有 CUDA，会启动失败。
> 推荐在带 GPU 的机器上启动；如果在 CPU 机器上启动，至少能在 **Tab 1** 配置并保存 yaml 供远程训练使用。
>
> 训练启动后切到 **Tab 2** 查看 loss 曲线、validation 视频、日志。
            """
        )

        # =========================================================
        # Tab 1: Configure & Launch
        # =========================================================
        with gr.Tab("1️⃣ 配置 & 启动"):
            with gr.Row():
                preset_dd = gr.Dropdown(
                    choices=list(PRESETS.keys()),
                    value="Stage 2 (256, 推荐)",
                    label="预设 (Preset)",
                    scale=2,
                )
                preset_desc = gr.Textbox(
                    label="预设说明",
                    value=PRESETS["Stage 2 (256, 推荐)"]["description"],
                    interactive=False,
                    scale=3,
                )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📂 数据集")
                    train_data_dir = gr.Textbox(
                        label="train_data_dir (目录)",
                        placeholder="data/my_high_quality_videos",
                    )
                    train_fileslist = gr.Textbox(
                        label="train_fileslist (文件列表，一行一个 mp4)",
                        placeholder="data/my_high_quality_videos/fileslist.txt",
                    )
                    val_video_path = gr.Textbox(
                        label="val_video_path",
                        value="assets/demo1_video.mp4",
                    )
                    val_audio_path = gr.Textbox(
                        label="val_audio_path",
                        value="assets/demo1_audio.wav",
                    )
                    dataset_choices = gr.Dropdown(
                        choices=list_datasets(),
                        label="或从已有数据集选 (click 后填到 train_data_dir)",
                        value=None,
                    )
                    dataset_choices.change(
                        lambda x: x,
                        inputs=dataset_choices,
                        outputs=train_data_dir,
                    )

                with gr.Column():
                    gr.Markdown("### 🏗 模型 & 训练")
                    resume_ckpt = gr.Textbox(
                        label="resume_ckpt (加载的预训练权重)",
                        value=PRESETS["Stage 2 (256, 推荐)"]["resume_ckpt"],
                    )
                    batch_size = gr.Slider(1, 64, value=1, step=1, label="batch_size")
                    num_frames = gr.Slider(8, 32, value=16, step=1, label="num_frames")
                    resolution = gr.Radio([256, 512], value=256, label="resolution")
                    learning_rate = gr.Number(value=1e-5, label="learning_rate", precision=8)

                    use_motion_module = gr.Checkbox(
                        value=True,
                        label="use_motion_module (Stage 2 必开)",
                    )
                    pixel_space_supervise = gr.Checkbox(
                        value=True,
                        label="pixel_space_supervise (Stage 2 必开)",
                    )
                    use_syncnet = gr.Checkbox(
                        value=True,
                        label="use_syncnet (Stage 2 必开)",
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⚖️ 损失权重")
                    sync_loss_weight = gr.Slider(0.0, 1.0, value=0.05, step=0.01, label="sync_loss_weight")
                    perceptual_loss_weight = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="perceptual_loss_weight (LPIPS)")
                    recon_loss_weight = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="recon_loss_weight")
                    trepa_loss_weight = gr.Slider(0.0, 50.0, value=10.0, step=1.0, label="trepa_loss_weight (0=关闭)")

                with gr.Column():
                    gr.Markdown("### ⚙️ 训练设置")
                    mixed_precision_training = gr.Checkbox(value=True, label="mixed_precision_training (fp16)")
                    enable_gradient_checkpointing = gr.Checkbox(value=True, label="enable_gradient_checkpointing")
                    mask_image_path = gr.Textbox(
                        label="mask_image_path",
                        value="latentsync/utils/mask.png",
                    )
                    save_ckpt_steps = gr.Slider(500, 50000, value=10000, step=500, label="save_ckpt_steps")
                    max_train_steps = gr.Slider(1000, 10_000_000, value=10_000_000, step=1000, label="max_train_steps")
                    num_workers = gr.Slider(0, 32, value=12, step=1, label="num_workers")
                    train_output_dir = gr.Textbox(
                        label="train_output_dir",
                        value="debug/unet",
                    )
                    nproc_per_node = gr.Slider(1, 8, value=1, step=1, label="torchrun nproc_per_node")
                    master_port = gr.Slider(20000, 30000, value=25679, step=1, label="torchrun master_port")
                    extra_env = gr.Textbox(
                        label="额外环境变量 (可选，每行 KEY=VALUE)",
                        placeholder="LATENTSYNC_GUIDANCE_SCALE=1.5\nHF_TOKEN=...",
                        lines=3,
                    )

            with gr.Row():
                launch_btn = gr.Button("🚀 启动训练", variant="primary", scale=2)
                stop_btn = gr.Button("⏹ 停止训练", variant="stop", scale=1)

            launch_status = gr.Textbox(label="启动状态", lines=10)
            log_path_state = gr.State(value="")
            launch_btn.click(
                fn=launch_training,
                inputs=[
                    preset_dd, train_data_dir, train_fileslist, val_video_path, val_audio_path,
                    resume_ckpt, batch_size, num_frames, resolution, learning_rate,
                    use_motion_module, pixel_space_supervise, use_syncnet,
                    sync_loss_weight, perceptual_loss_weight, recon_loss_weight, trepa_loss_weight,
                    mixed_precision_training, enable_gradient_checkpointing, mask_image_path,
                    save_ckpt_steps, max_train_steps, num_workers, train_output_dir,
                    nproc_per_node, master_port, extra_env,
                ],
                outputs=[launch_status, log_path_state, gr.State()],
            )
            stop_btn.click(fn=stop_training, outputs=launch_status)

            # preset → fill defaults
            preset_dd.change(
                fn=on_preset_change,
                inputs=preset_dd,
                outputs=[
                    batch_size, num_frames, resolution, learning_rate,
                    use_motion_module, pixel_space_supervise, use_syncnet,
                    sync_loss_weight, perceptual_loss_weight, recon_loss_weight, trepa_loss_weight,
                    mixed_precision_training, enable_gradient_checkpointing, mask_image_path,
                    resume_ckpt, preset_desc,
                ],
            )

        # =========================================================
        # Tab 2: Monitor
        # =========================================================
        with gr.Tab("2️⃣ 训练监控"):
            with gr.Row():
                monitor_output_dir = gr.Textbox(
                    label="train_output_dir (同 Tab1)",
                    value="debug/unet",
                )
                refresh_runs_btn = gr.Button("🔄 刷新 run 列表")

            run_dd = gr.Dropdown(label="run 目录", choices=[], value=None)
            refresh_runs_btn.click(
                fn=refresh_runs,
                inputs=monitor_output_dir,
                outputs=run_dd,
            )

            trainer_status = gr.Textbox(label="Trainer 状态", interactive=False)
            log_box = gr.Textbox(label="最新日志 (尾部 80 行)", lines=20, interactive=False)
            ckpt_info_box = gr.Textbox(label="最新 checkpoint 信息", lines=10, interactive=False)

            with gr.Row():
                with gr.Column():
                    loss_chart_img = gr.Image(label="Loss / Sync_conf 曲线", type="filepath")
                with gr.Column():
                    val_video_dd = gr.Dropdown(label="Validation 视频", choices=[])
                    val_video_player = gr.Video(label="预览", interactive=False)

            def _on_run_change(run_path):
                chart = parse_loss_chart(run_path)
                vids = list_validation_videos(run_path)
                ckpts = list_checkpoints_in_run(run_path)
                ck_info = read_loss_from_checkpoint(REPO_ROOT / ckpts[-1]) if ckpts else "(no checkpoint yet)"
                return chart, gr.update(choices=vids, value=vids[0] if vids else None), ck_info

            run_dd.change(
                fn=_on_run_change,
                inputs=run_dd,
                outputs=[loss_chart_img, val_video_dd, ckpt_info_box],
            )
            val_video_dd.change(
                fn=lambda x: x,
                inputs=val_video_dd,
                outputs=val_video_player,
            )

            monitor_btn = gr.Button("🔄 手动刷新", variant="primary")
            monitor_btn.click(
                fn=monitor_refresh,
                inputs=[monitor_output_dir, run_dd, log_path_state],
                outputs=[
                    run_dd, gr.Textbox(visible=False), loss_chart_img,
                    val_video_dd, log_box, ckpt_info_box, trainer_status,
                ],
            )

            timer = gr.Timer(value=15)
            timer.tick(
                fn=monitor_refresh,
                inputs=[monitor_output_dir, run_dd, log_path_state],
                outputs=[
                    run_dd, gr.Textbox(visible=False), loss_chart_img,
                    val_video_dd, log_box, ckpt_info_box, trainer_status,
                ],
            )

        # =========================================================
        # Tab 3: Compare
        # =========================================================
        with gr.Tab("3️⃣ 推理对比 (base vs fine-tuned)"):
            gr.Markdown(
                "上传一段视频 + 音频，分别用 base 和微调后的 checkpoint 跑推理，并排对比。"
            )
            with gr.Row():
                with gr.Column():
                    cmp_video = gr.Video(label="Input Video")
                    cmp_audio = gr.Audio(label="Input Audio", type="filepath")
                with gr.Column():
                    cmp_base = gr.Dropdown(
                        choices=list_checkpoints(),
                        label="Base checkpoint",
                    )
                    cmp_ft = gr.Dropdown(
                        choices=list_checkpoints(),
                        label="Fine-tuned checkpoint",
                    )
                    cmp_resolution = gr.Radio([256, 512], value=256, label="resolution")

            with gr.Row():
                cmp_steps = gr.Slider(10, 50, value=20, step=1, label="inference_steps")
                cmp_guidance = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="guidance_scale")
                cmp_seed = gr.Number(value=1247, label="seed", precision=0)

            cmp_btn = gr.Button("🎬 生成对比", variant="primary")

            with gr.Row():
                cmp_out_base = gr.Video(label="Base 输出")
                cmp_out_ft = gr.Video(label="Fine-tuned 输出")

            cmp_btn.click(
                fn=run_compare,
                inputs=[
                    cmp_video, cmp_audio, cmp_base, cmp_ft,
                    cmp_steps, cmp_guidance, cmp_seed, cmp_resolution,
                ],
                outputs=[cmp_out_base, cmp_out_ft],
            )

        # =========================================================
        # Tab 4: Identity Protection Strategy
        # =========================================================
        with gr.Tab("🛡️ Identity 保护策略"):
            gr.Markdown(
                """
LatentSync 用 **4 层防御** 保证只改嘴部、不改脸：

| 层 | 机制 | 在哪控制 |
|---|---|---|
| L1 | `ref_pixel_values` 提供 identity | 训练时 `UNetDataset` + 推理时实时 ref |
| L2 | UNet 训练学到"看着 ref 还原 identity" | 训练分布自动学习 |
| L3 | `paste_surrounding_pixels_back` mask 截断 | 推理时 `dynamic_region_mask` |
| L4 | `_restore_reference_detail` 高频细节贴回 | 推理时 `mouth_detail_strength` |

下面三个区块分别调 L1 / L3 / L4 的关键参数。改完点 **生成推理 yaml** 即可在 `gradio_app.py` / `api.py` 里复用。
                """
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### L1: ref 窗口策略（影响训练数据采样 + 推理 ref 选择）")
                    ref_strategy = gr.Radio(
                        choices=["random", "adjacent", "fixed_first_frame"],
                        value="random",
                        label="ref 窗口选择策略",
                        info=(
                            "random: 随机抽远端帧（论文 baseline）\n"
                            "adjacent: 抽相邻帧（identity 更稳但极端表情少）\n"
                            "fixed_first_frame: 固定用第 1 帧（一致性最强但多样性差）"
                        ),
                    )
                    ref_window_distance = gr.Slider(
                        minimum=0,
                        maximum=64,
                        value=16,
                        step=1,
                        label="ref 与 gt 的最小距离（帧）",
                        info="论文 baseline = 16 帧（约 0.64 秒）。太小容易把同一段当 ref。",
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### L3: dynamic mask 大小（控制 paste-back 范围）")
                    dynamic_mask_mode = gr.Radio(
                        choices=["conservative", "standard", "aggressive"],
                        value="standard",
                        label="dynamic mask 大小策略",
                        info=(
                            "conservative: 椭圆更小（默认 pad_width×0.8），更保守\n"
                            "standard: 论文默认（pad_width×1.5）\n"
                            "aggressive: 椭圆更大（pad_width×2.0），覆盖大笑嘴"
                        ),
                    )
                    paste_back_blur_sigma = gr.Slider(
                        minimum=0.0,
                        maximum=15.0,
                        value=7.0,
                        step=0.5,
                        label="paste back 边缘模糊 sigma (像素)",
                        info="越大 paste-back 边界越平滑，但嘴部边缘会糊",
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### L4: detail / color post-processing")
                    detail_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.65,
                        step=0.05,
                        label="mouth_detail_strength (L4 detail restore)",
                        info="越大越贴原图皮肤纹理（痣、皱纹）。>0.85 会盖掉生成的嘴型",
                    )
                    color_match_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.60,
                        step=0.05,
                        label="color_match_strength",
                        info="越大颜色越平滑（避免 mask 边界色差）。>0.9 可能过度",
                    )

            with gr.Row():
                identity_generate_btn = gr.Button("📝 生成推理 kwargs / yaml", variant="primary")
                identity_clear_btn = gr.Button("🧹 重置为默认值")

            identity_output = gr.Code(
                label="生成的推理 kwargs (Python) 和 yaml (Config)",
                language="python",
                lines=20,
            )

            identity_generate_btn.click(
                fn=generate_identity_kit,
                inputs=[
                    ref_strategy, ref_window_distance,
                    dynamic_mask_mode, paste_back_blur_sigma,
                    detail_strength, color_match_strength,
                ],
                outputs=identity_output,
            )

            identity_clear_btn.click(
                fn=reset_identity_defaults,
                outputs=[
                    ref_strategy, ref_window_distance,
                    dynamic_mask_mode, paste_back_blur_sigma,
                    detail_strength, color_match_strength,
                ],
            )

            gr.Markdown(
                """
### 使用方法

生成的 `kwargs` 可以直接传给 `LipsyncPipeline.__call__(..., **kwargs)`：

```python
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
pipe = LipsyncPipeline(...)

pipe(
    video_path=...,
    audio_path=...,
    video_out_path=...,
    inference_ckpt_path=...,
    **identity_kwargs,  # ← 把上方生成的字典展开
)
```
                """
            )

        # =========================================================
        # Tab 5: Dataset Quality Evaluation
        # =========================================================
        with gr.Tab("📊 数据集质量评估"):
            gr.Markdown(
                """
训练前先评估数据，避免训完才发现质量问题。

会跑：
1. **HyperIQA 分数**（每视频取 3 帧，看视觉质量分布）
2. **SyncNet confidence**（每视频算 AV 同步质量）
3. **文件完整性**（损坏 / 缺失 / 时长不足）
4. **统计摘要** + **潜在问题列表**
                """
            )

            with gr.Row():
                ds_dir_input = gr.Textbox(
                    label="high_visual_quality 目录",
                    placeholder="/data/voxceleb2/high_visual_quality",
                    scale=3,
                )
                ds_max_videos = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=100,
                    step=10,
                    label="最多评估视频数（采样）",
                    scale=1,
                )
                ds_eval_btn = gr.Button("🔍 开始评估", variant="primary", scale=1)

            with gr.Row():
                with gr.Column():
                    ds_stats = gr.Textbox(label="统计摘要", lines=15)
                with gr.Column():
                    ds_issues = gr.Textbox(label="潜在问题", lines=15)

            ds_chart = gr.Plot(label="HyperIQA / Sync_conf 分布")

            ds_eval_btn.click(
                fn=evaluate_dataset_quality,
                inputs=[ds_dir_input, ds_max_videos],
                outputs=[ds_stats, ds_issues, ds_chart],
            )

        # =========================================================
        # Tab 6: Badcase Checklist
        # =========================================================
        with gr.Tab("⚠️ Badcase 检查清单"):
            gr.Markdown(
                """
对生成的视频跑全套质量检查（对应 §13）：

| 检查项 | 目标 | 含义 |
|---|---|---|
| 嘴糊比例 | < 30% | Laplacian 方差低于阈值的帧占比 |
| 闪烁评分 | < 8 | 嘴部帧间平均像素差 |
| 唇音同步 | > 7 | SyncNet confidence |
| 身份保持 | > 0.8 | Face embedding 余弦相似度 |
                """
            )

            with gr.Row():
                bc_video = gr.Video(label="生成结果视频", scale=2)
                bc_reference = gr.Video(label="原始参考视频（可选，用于 identity sim）", scale=2)

            bc_check_btn = gr.Button("🔍 跑 Badcase 检测", variant="primary")

            with gr.Row():
                with gr.Column():
                    bc_blurry = gr.Number(label="嘴糊比例 (目标 < 30%)")
                    bc_flicker = gr.Number(label="闪烁评分 (目标 < 8)")
                    bc_sync = gr.Number(label="唇音同步 (目标 > 7)")
                    bc_identity = gr.Number(label="身份保持 (目标 > 0.8)")
                with gr.Column():
                    bc_report = gr.Textbox(label="诊断报告", lines=20)

            bc_check_btn.click(
                fn=run_badcase_checklist,
                inputs=[bc_video, bc_reference],
                outputs=[bc_blurry, bc_flicker, bc_sync, bc_identity, bc_report],
            )

    return demo


# ---------------------------------------------------------------------------
# Tab 4 helpers: Identity Protection Strategy
# ---------------------------------------------------------------------------

def _ref_strategy_to_dataset_kwargs(strategy: str, min_distance: int) -> Dict[str, Any]:
    """Map UI strategy choice to UNetDataset-side kwargs.

    Currently UNetDataset only supports 'random' (the baseline). For
    'adjacent' and 'fixed_first_frame' we emit a code-side patch that the
    user can drop into latentsync/data/unet_dataset.py if they want.
    """
    if strategy == "random":
        return {
            "_strategy": "random",
            "ref_min_distance": min_distance,
            "_note": "Default UNetDataset.get_frames already picks random ref "
                     "outside the gt window. No code change needed.",
        }
    if strategy == "adjacent":
        return {
            "_strategy": "adjacent",
            "ref_min_distance": min_distance,
            "_patch": (
                "# In latentsync/data/unet_dataset.py:75-80, replace:\n"
                "while True:\n"
                "    ref_start_idx = random.randint(0, total_num_frames - self.num_frames)\n"
                "    if ref_start_idx > start_idx - self.num_frames and ref_start_idx < start_idx + self.num_frames:\n"
                "        continue\n"
                "# With:\n"
                "ref_start_idx = max(0, start_idx + self.num_frames + ref_min_distance)\n"
                "if ref_start_idx + self.num_frames > total_num_frames:\n"
                "    ref_start_idx = start_idx - self.num_frames - ref_min_distance\n"
            ),
        }
    # fixed_first_frame
    return {
        "_strategy": "fixed_first_frame",
        "ref_min_distance": 0,
        "_patch": (
            "# In latentsync/data/unet_dataset.py, replace ref sampling with:\n"
            "ref_start_idx = 0  # always use first frame as ref\n"
        ),
    }


def _dynamic_mask_mode_to_params(mode: str) -> Dict[str, float]:
    """Map UI mode to lipsync_pipeline.generate_dynamic_mouth_mask kwargs."""
    presets = {
        "conservative": {"pad_width_ratio": 1.2, "pad_height_top_ratio": 1.1,
                         "pad_height_bottom_ratio": 1.8, "feather_sigma_px": 5.0},
        "standard":     {"pad_width_ratio": 1.5, "pad_height_top_ratio": 1.3,
                         "pad_height_bottom_ratio": 2.2, "feather_sigma_px": 7.0},
        "aggressive":   {"pad_width_ratio": 2.0, "pad_height_top_ratio": 1.6,
                         "pad_height_bottom_ratio": 2.6, "feather_sigma_px": 10.0},
    }
    return presets[mode]


def generate_identity_kit(
    ref_strategy: str,
    ref_window_distance: int,
    dynamic_mask_mode: str,
    paste_back_blur_sigma: float,
    detail_strength: float,
    color_match_strength: float,
) -> str:
    """Build the kwargs dict the inference pipeline should receive, plus a
    yaml-style sidecar and human-readable warnings."""
    dataset_kwargs = _ref_strategy_to_dataset_kwargs(ref_strategy, ref_window_distance)
    mask_params = _dynamic_mask_mode_to_params(dynamic_mask_mode)
    mask_params["feather_sigma_px"] = float(paste_back_blur_sigma)

    inference_kwargs = {
        # L3: dynamic mask geometry
        "dynamic_mask_pad_width_ratio": mask_params["pad_width_ratio"],
        "dynamic_mask_pad_height_top_ratio": mask_params["pad_height_top_ratio"],
        "dynamic_mask_pad_height_bottom_ratio": mask_params["pad_height_bottom_ratio"],
        # L3: paste-back blur
        "paste_back_feather_sigma_px": mask_params["feather_sigma_px"],
        # L4: detail / color post-processing
        "mouth_detail_strength": float(detail_strength),
        "color_match_strength": float(color_match_strength),
    }

    lines: List[str] = []
    lines.append("# === LatentSync Identity-Protection Kit ===")
    lines.append("# Generated by gradio_finetune.py Tab 4")
    lines.append("")
    lines.append("# ----- 1) Python kwargs to pass to LipsyncPipeline.__call__ -----")
    lines.append("inference_kwargs = {")
    for k, v in inference_kwargs.items():
        lines.append(f"    {k!r}: {v!r},")
    lines.append("}")
    lines.append("# Usage:")
    lines.append("# pipe(video_path=..., audio_path=..., video_out_path=..., **inference_kwargs)")
    lines.append("")
    lines.append("# ----- 2) Dataset-side (UNetDataset) ref strategy -----")
    lines.append("# Strategy: " + dataset_kwargs["_strategy"])
    lines.append("# ref_min_distance: " + str(dataset_kwargs["ref_min_distance"]))
    if "_patch" in dataset_kwargs:
        lines.append("# Patch (drop into latentsync/data/unet_dataset.py:75-80):")
        for pl in dataset_kwargs["_patch"].splitlines():
            lines.append("#   " + pl)
    else:
        lines.append("# " + dataset_kwargs["_note"])
    lines.append("")
    lines.append("# ----- 3) Sanity warnings -----")
    if detail_strength > 0.85:
        lines.append("# ⚠️ detail_strength > 0.85 — generated mouth shape may be washed out.")
    if color_match_strength > 0.9:
        lines.append("# ⚠️ color_match_strength > 0.9 — over-correction may erase emotion.")
    if dynamic_mask_mode == "aggressive":
        lines.append("# ⚠️ aggressive mask + ref_min_distance small — identity drift possible.")
    if ref_strategy == "fixed_first_frame":
        lines.append("# ⚠️ fixed_first_frame reduces temporal diversity; expect lower sync_conf.")
    if paste_back_blur_sigma < 3.0:
        lines.append("# ⚠️ paste_back_blur < 3 px — visible seam at mask boundary.")
    return "\n".join(lines)


def reset_identity_defaults() -> Tuple[str, int, str, float, float, float]:
    return "random", 16, "standard", 7.0, 0.65, 0.60


# ---------------------------------------------------------------------------
# Tab 5 helpers: Dataset Quality Evaluation
# ---------------------------------------------------------------------------

def _safe_hyperiqa_score(model_hyper, model_target, frames_tensor, device, transforms):
    """Run HyperIQA on 3 frames (first/middle/last) and return mean score 0-100."""
    import torchvision  # local import keeps gradio launch fast

    sampled = frames_tensor[::max(1, len(frames_tensor) // 3)][:3]
    sampled = transforms(sampled).to(device)
    paras = model_hyper(sampled)
    preds = [model_target(paras).mean().item() for _ in [0]]
    return float(preds[0]) if preds else 0.0


def evaluate_dataset_quality(
    data_dir: str, max_videos: int
) -> Tuple[str, str, Any]:
    """Walk the directory, sample up to max_videos mp4s, compute HyperIQA +
    SyncNet confidence distributions, return summary text + issues + plot."""
    import random as _random
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not data_dir or not Path(data_dir).exists():
        return (
            f"❌ 目录不存在: {data_dir}",
            "请检查路径。预期目录结构: data_dir/<speaker>/<video>.mp4 或 data_dir/*.mp4",
            None,
        )

    candidates = sorted(Path(data_dir).rglob("*.mp4"))[: int(max_videos)]
    if not candidates:
        return (f"❌ {data_dir} 下没有 .mp4 文件", "", None)

    hyperiqa_scores: List[float] = []
    sync_confs: List[float] = []
    broken: List[str] = []
    too_short: List[str] = []

    try:
        from eval.hyper_iqa import HyperNet, TargetNet
        import torchvision
        from torchvision import transforms

        device = "cuda" if torch_available() else "cpu"
        model_hyper = HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)
        ckpt = REPO_ROOT / "checkpoints/auxiliary/koniq_pretrained.pkl"
        if ckpt.exists():
            model_hyper.load_state_dict(torch_load(str(ckpt), map_location=device))
            model_hyper.eval()
            hyperiqa_available = True
        else:
            hyperiqa_available = False
        tf = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    except Exception as e:
        hyperiqa_available = False
        tf = None
        print(f"[evaluate_dataset_quality] hyperiqa unavailable: {e}")

    syncnet_available = False
    try:
        from eval.syncnet import SyncNetEval
        from eval.syncnet_detect import SyncNetDetector
        from eval.eval_sync_conf import syncnet_eval as _sync_eval

        device = "cuda" if torch_available() else "cpu"
        if (REPO_ROOT / "checkpoints/auxiliary/syncnet_v2.model").exists():
            _se = SyncNetEval(device=device)
            _se.loadParameters("checkpoints/auxiliary/syncnet_v2.model")
            _sd = SyncNetDetector(device=device, detect_results_dir="detect_results")
            syncnet_available = True
    except Exception as e:
        print(f"[evaluate_dataset_quality] syncnet unavailable: {e}")

    for p in candidates:
        try:
            from decord import VideoReader
            vr = VideoReader(str(p))
            if len(vr) < 16:
                too_short.append(str(p))
                continue
            # sample first/middle/last frame
            idxs = [0, len(vr) // 2, len(vr) - 1]
            frames = vr.get_batch(idxs).asnumpy()
            frames_t = torch_from_numpy(_rearrange(frames, "f h w c -> f c h w")).float() / 255.0

            if hyperiqa_available and tf is not None:
                paras = model_hyper(tf(frames_t.clone()).to(device))
                model_target = TargetNet(paras).to(device)
                for p_ in model_target.parameters():
                    p_.requires_grad = False
                score = model_target(paras["target_in_vec"]).mean().item()
                hyperiqa_scores.append(score)

            if syncnet_available:
                # write frames as a tiny video? Easier: only eval if video has audio
                try:
                    _, conf = _sync_eval(_se, _sd, str(p), "temp")
                    if conf is not None:
                        sync_confs.append(conf)
                except Exception:
                    pass
        except Exception as e:
            broken.append(f"{p}: {e}")

    stats_lines = [
        f"📁 目录: {data_dir}",
        f"🎬 扫描视频数: {len(candidates)}",
        f"✅ 完整视频: {len(candidates) - len(broken) - len(too_short)}",
        f"❌ 损坏: {len(broken)}",
        f"⚠️ 太短 (<16 帧): {len(too_short)}",
    ]
    if hyperiqa_scores:
        import statistics
        stats_lines += [
            "",
            "📊 HyperIQA 分数（视觉质量，0-100）:",
            f"  mean: {statistics.mean(hyperiqa_scores):.2f}",
            f"  median: {statistics.median(hyperiqa_scores):.2f}",
            f"  min: {min(hyperiqa_scores):.2f}",
            f"  max: {max(hyperiqa_scores):.2f}",
            f"  ≥40 (preprocess 阈值): {sum(1 for s in hyperiqa_scores if s >= 40)} / {len(hyperiqa_scores)}",
        ]
    else:
        stats_lines.append("\n📊 HyperIQA 不可用（检查 checkpoints/auxiliary/koniq_pretrained.pkl）")
    if sync_confs:
        import statistics
        stats_lines += [
            "",
            "🎵 SyncNet confidence（音视同步）:",
            f"  mean: {statistics.mean(sync_confs):.2f}",
            f"  median: {statistics.median(sync_confs):.2f}",
            f"  min: {min(sync_confs):.2f}",
            f"  ≥3 (preprocess 阈值): {sum(1 for s in sync_confs if s >= 3)} / {len(sync_confs)}",
        ]
    else:
        stats_lines.append("\n🎵 SyncNet 不可用（检查 checkpoints/auxiliary/syncnet_v2.model）")

    issues: List[str] = []
    if broken:
        issues.append(f"❌ {len(broken)} 个损坏视频，建议先跑 remove_broken_videos")
        for b in broken[:5]:
            issues.append(f"   - {b}")
    if too_short:
        issues.append(f"⚠️ {len(too_short)} 个视频太短（<16 帧），训练时会被跳")
    if hyperiqa_scores and statistics.mean(hyperiqa_scores) < 40:
        issues.append("❌ 平均 HyperIQA < 40：数据质量整体偏低，建议重新跑 filter_visual_quality")
    if sync_confs and statistics.mean(sync_confs) < 3:
        issues.append("❌ 平均 SyncNet conf < 3：音视不同步，建议重新跑 sync_av")
    if not issues:
        issues.append("✅ 数据集看起来健康")

    # plot
    fig = None
    if hyperiqa_scores or sync_confs:
        fig, ax1 = plt.subplots(figsize=(7, 4))
        if hyperiqa_scores:
            ax1.hist(hyperiqa_scores, bins=20, alpha=0.6, label="HyperIQA", color="blue")
            ax1.axvline(40, color="blue", linestyle="--", label="HyperIQA ≥ 40")
            ax1.set_xlabel("HyperIQA score (0-100)")
        if sync_confs:
            ax2 = ax1.twinx()
            ax2.hist(sync_confs, bins=20, alpha=0.6, label="SyncNet conf", color="green")
            ax2.axvline(3, color="green", linestyle="--", label="SyncNet ≥ 3")
            ax2.set_xlabel("SyncNet confidence")
        fig.tight_layout()

    return ("\n".join(stats_lines), "\n".join(issues), fig)


def torch_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def torch_load(path: str, map_location: str = "cpu"):
    import torch
    return torch.load(path, map_location=map_location, weights_only=True)


def torch_from_numpy(arr):
    import torch
    return torch.from_numpy(arr)


def _rearrange(tensor, pattern):
    from einops import rearrange
    return rearrange(tensor, pattern)


# ---------------------------------------------------------------------------
# Tab 6 helpers: Badcase Checklist
# ---------------------------------------------------------------------------

def run_badcase_checklist(
    video_path: str, reference_video_path: Optional[str]
) -> Tuple[float, float, float, float, str]:
    """Run all 4 badcase checks on a single generated video.

    Returns (blurry_ratio, flicker_score, sync_conf, identity_sim, report).
    """
    if not video_path:
        return 0.0, 0.0, 0.0, 0.0, "❌ 请先上传视频"

    try:
        from decord import VideoReader
        import numpy as np
        import cv2
        vr = VideoReader(video_path)
        frames = [f.asnumpy() for f in vr]
        if len(frames) < 2:
            return 0.0, 0.0, 0.0, 0.0, "❌ 视频帧数 < 2"

        # ---- 1. Blurry mouth ratio ----
        # crude: convert to grayscale, Laplacian variance per frame
        grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        laps = [cv2.Laplacian(g, cv2.CV_64F).var() for g in grays]
        blurry_count = sum(1 for v in laps if v < 50)
        blurry_ratio = blurry_count / len(laps)

        # ---- 2. Flicker score ----
        # mean abs difference between consecutive frames
        diffs = []
        for i in range(1, len(frames)):
            d = np.abs(frames[i].astype(float) - frames[i - 1].astype(float)).mean()
            diffs.append(d)
        flicker_score = float(np.mean(diffs))

        # ---- 3. SyncNet confidence ----
        sync_conf = 0.0
        try:
            from eval.syncnet import SyncNetEval
            from eval.syncnet_detect import SyncNetDetector
            from eval.eval_sync_conf import syncnet_eval as _sync_eval

            device = "cuda" if torch_available() else "cpu"
            if (REPO_ROOT / "checkpoints/auxiliary/syncnet_v2.model").exists():
                _se = SyncNetEval(device=device)
                _se.loadParameters("checkpoints/auxiliary/syncnet_v2.model")
                _sd = SyncNetDetector(device=device, detect_results_dir="detect_results")
                _, sync_conf = _sync_eval(_se, _sd, video_path, "temp")
        except Exception as e:
            sync_conf_note = f"SyncNet 评估失败: {e}"
        else:
            sync_conf_note = None

        # ---- 4. Identity similarity (only if reference provided) ----
        identity_sim = 0.0
        if reference_video_path:
            try:
                from latentsync.utils.face_detector import FaceDetector
                det = FaceDetector()
                _, _, real_emb = det.detect(VideoReader(reference_video_path)[0].asnumpy())
                _, _, gen_emb = det.detect(frames[0])
                if real_emb is not None and gen_emb is not None:
                    cos = float(np.dot(real_emb, gen_emb) / (
                        np.linalg.norm(real_emb) * np.linalg.norm(gen_emb)
                    ))
                    identity_sim = cos
            except Exception as e:
                identity_sim_note = f"identity 评估失败: {e}"
        else:
            identity_sim_note = "未提供参考视频，跳过 identity sim"

        # ---- Build report ----
        lines: List[str] = []
        lines.append(f"📊 视频: {video_path}")
        lines.append(f"📏 总帧数: {len(frames)}")
        lines.append(f"🔍 Laplacian sharpness 范围: {min(laps):.1f} ~ {max(laps):.1f}")
        lines.append("")
        if blurry_ratio < 0.30:
            lines.append(f"✅ 嘴糊比例 {blurry_ratio:.1%} < 30% (目标)")
        else:
            lines.append(f"⚠️ 嘴糊比例 {blurry_ratio:.1%} ≥ 30%")
            lines.append("   建议: 升 512 分辨率 / 加 LPIPS weight / 开 CodeFormer")
        lines.append("")
        if flicker_score < 8:
            lines.append(f"✅ 闪烁评分 {flicker_score:.2f} < 8 (目标)")
        else:
            lines.append(f"⚠️ 闪烁评分 {flicker_score:.2f} ≥ 8")
            lines.append("   建议: 恢复 TREPA=10 / Motion Module 训够 / 开时序稳定")
        lines.append("")
        if sync_conf > 7:
            lines.append(f"✅ 唇音同步 {sync_conf:.2f} > 7 (目标)")
        elif sync_conf > 4:
            lines.append(f"⚠️ 唇音同步 {sync_conf:.2f} (4-7 中等)")
            lines.append("   建议: 重训 SyncNet (batch≥1024) / 加大 sync_loss_weight")
        else:
            lines.append(f"❌ 唇音同步 {sync_conf:.2f} < 4 (差)")
            lines.append("   建议: 大概率 SyncNet 没训稳，回去查 5 因素")
        lines.append("")
        if reference_video_path:
            if identity_sim > 0.8:
                lines.append(f"✅ 身份保持 {identity_sim:.3f} > 0.8 (目标)")
            elif identity_sim > 0.6:
                lines.append(f"⚠️ 身份保持 {identity_sim:.3f} (0.6-0.8 中等)")
                lines.append("   建议: 检查 UNetDataset ref 窗口选择 / 调 identity_similarity 阈值")
            else:
                lines.append(f"❌ 身份保持 {identity_sim:.3f} < 0.6 (差)")
                lines.append("   建议: identity 严重丢失，检查 paste_surrounding_pixels_back 是否生效")
        else:
            lines.append("ℹ️ 未提供参考视频，跳过 identity sim 检查")
        if sync_conf_note:
            lines.append(f"\n⚠️ {sync_conf_note}")
        if 'identity_sim_note' in locals() and identity_sim_note:
            lines.append(f"⚠️ {identity_sim_note}")

        return blurry_ratio, flicker_score, float(sync_conf), identity_sim, "\n".join(lines)
    except Exception as e:
        return 0.0, 0.0, 0.0, 0.0, f"❌ 检测失败: {e}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()