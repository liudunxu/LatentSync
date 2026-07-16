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
import logging
import os
import sys
import shlex
import shutil
import signal
import threading
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Disable Gradio telemetry / messaging fetches so the page loads faster in
# network-restricted environments (e.g. China mainland without a proxy).
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "0")
os.environ.setdefault("GRADIO_TELEMETRY_ENABLED", "0")

import gradio as gr
import psutil
import yaml
from omegaconf import OmegaConf


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("gradio_finetune")


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "configs" / "unet"
SYNCNET_CONFIG_DIR = REPO_ROOT / "configs" / "syncnet"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

# Fine-tuning intermediates (generated configs, training logs, run outputs,
# audio embeds/mel caches) go to a separate large-disk directory by default.
# Can be overridden with LATENTSYNC_FINETUNE_DIR env var.
_FINETUNE_BASE_DIR_STR = os.environ.get("LATENTSYNC_FINETUNE_DIR", "/root/autodl-tmp/latentsync_finetune")
FINETUNE_BASE_DIR = Path(_FINETUNE_BASE_DIR_STR)
try:
    FINETUNE_BASE_DIR.mkdir(parents=True, exist_ok=True)
except OSError as e:
    logger.warning(
        "Cannot create fine-tune base dir %s (%s); falling back to %s",
        FINETUNE_BASE_DIR, e, REPO_ROOT / "debug",
    )
    FINETUNE_BASE_DIR = REPO_ROOT / "debug"
    FINETUNE_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Propagate the effective base dir to child training processes so they can
# resolve relative train_output_dir consistently.
os.environ.setdefault("LATENTSYNC_FINETUNE_DIR", str(FINETUNE_BASE_DIR))
TRAIN_OUTPUT_DIR = FINETUNE_BASE_DIR

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
        "description": (
            "⚪ Baseline LoRA。只 wrap 注意力投影(4 项),~10MB adapter。"
            "适合泛用 finetune 或快速试训。结构性脸变形不 cover。"
        ),
        "lora": {
            "enabled": True,
            "rank": 32,
            "alpha": 64,
            "dropout": 0.05,
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
            "qlora": False,
        },
        "freeze_attn2": False,
    },
    "🎯 Badcase Fix (侧脸+运动, LoRA, 12-15GB)": {
        "config_file": "configs/unet/stage2.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 5e-5,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        "sync_loss_weight": 0.02,
        "perceptual_loss_weight": 0.15,
        "recon_loss_weight": 1.0,
        "trepa_loss_weight": 10.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask.png",
        "save_ckpt_steps": 500,
        "max_train_steps": 3000,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 500,
        "description": (
            "🟢 **推荐 — 内容型 badcase**\n"
            "resolution=256 显存友好,训练快;"
            "LoRA rank=64, lr=5e-5, sync_loss=0.02, cosine+500 warmup, 每 500 步存 ckpt。\n"
            "适用:嘴型/audio 同步、嘴糊、paste-back 外溢、侧脸唇形不同步。\n"
            "注意:SyncNet 只支持 16 帧,所以不要改 num_frames。\n"
            "freeze_attn2=True 保护基础唇音同步能力。\n"
            "训练前请确认数据集已做人脸对齐(init_finetune_dataset.py --align --align-resolution 256)。"
        ),
        "lora": {
            "enabled": True,
            "rank": 64,
            "alpha": 128,
            "dropout": 0.10,
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
            "qlora": False,
        },
        "freeze_attn2": True,
    },
    "🧩 Structural Fix (LoRA + conv, 18-22GB)": {
        "config_file": "configs/unet/stage2.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 3e-5,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        "sync_loss_weight": 0.05,
        "perceptual_loss_weight": 0.20,
        "recon_loss_weight": 1.0,
        "trepa_loss_weight": 10.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask.png",
        "save_ckpt_steps": 1000,
        "max_train_steps": 30000,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 300,
        "description": (
            "🔴 **推荐 — 结构性 badcase**\n"
            "LoRA target 加 conv1/conv2/conv_shortcut/proj_in/proj_out/conv_in/conv_out\n"
            "(11 项,~25-30M params,3x capacity,够 cover 侧脸几何错位)。\n"
            "VRAM 占用 18-22GB(Lower lr 防过拟合; perceptual↑保细节)。\n"
            "内容型嘴错见 🎯 Badcase Fix; 短剧多说话人见 🎬 Short Drama; 通用 baseline 见 ⚪ Stage 2 LoRA。"
        ),
        "lora": {
            "enabled": True,
            "rank": 16,
            "alpha": 32,
            "dropout": 0.10,
            "target_modules": [
                # attention projections (latentsync/models/attention.py)
                "to_q", "to_k", "to_v", "to_out.0",
                # Resnet convs (latentsync/models/resnet.py)
                "conv1", "conv2", "conv_shortcut",
                # Attention 1×1 conv re-mappers (latentsync/models/attention.py)
                "proj_in", "proj_out",
                # UNet input/output gates (latentsync/models/unet.py)
                "conv_in", "conv_out",
            ],
            "qlora": False,
        },
        "freeze_attn2": True,
    },
    "💋 Side-Face Lip Quality (LoRA+conv, 18-22GB)": {
        "config_file": "configs/unet/stage2.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        # SyncNet 只支持 16 帧,长时序上下文需用 motion module 补偿
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 3e-5,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        # 侧脸时唇被遮 ~30-50%,必须强推 audio-driven 想象
        "sync_loss_weight": 0.12,
        # 唇部纹理在侧脸时 shading 不同,提权
        "perceptual_loss_weight": 0.25,
        "recon_loss_weight": 1.0,
        # 稍降 TREPA 让唇部允许更多形状变化(闭嘴 → 张嘴)
        "trepa_loss_weight": 8.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask.png",
        "save_ckpt_steps": 500,
        "max_train_steps": 30000,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 300,
        "description": (
            "💋 **推荐 — 侧脸唇形质量 (yaw 15-30°)**\n"
            "针对侧脸时嘴部遮挡 (~30-50%) + 唇纹理变化的双重挑战。\n"
            "LoRA target 加 conv(11 项,同 Structural),rank=48 留更多 capacity 学唇形。\n"
            "sync_loss=0.18 强推唇音同步; perceptual=0.25 锐化唇部纹理;\n"
            "num_frames=16 (SyncNet 只支持 16 帧),motion module 补偿时序连贯。\n"
            "数据:用 celebv_hq_side recipe(side_face 桶 ≥ 50%)。"
        ),
        "lora": {
            "enabled": True,
            "rank": 48,
            "alpha": 96,
            "dropout": 0.10,
            "target_modules": [
                "to_q", "to_k", "to_v", "to_out.0",
                "conv1", "conv2", "conv_shortcut",
                "proj_in", "proj_out",
                "conv_in", "conv_out",
            ],
            "qlora": False,
        },
        "freeze_attn2": True,
    },
    "🎬 Short Drama (LoRA+conv, 多说话人, 18-22GB)": {
        "config_file": "configs/unet/stage2.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        # Short drama 单段短(5-15s),长上下文稀释信号 — 回到 16
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 3e-5,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        # 短剧容错低,口型必须跟得紧
        "sync_loss_weight": 0.12,
        "perceptual_loss_weight": 0.20,
        "recon_loss_weight": 1.0,
        "trepa_loss_weight": 10.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask.png",
        # 短剧 ckpt 多 — 频繁切 → 多存点
        "save_ckpt_steps": 500,
        "max_train_steps": 25000,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 300,
        "description": (
            "🟣 **推荐 — 短剧 (多说话人/频繁切场景)**\n"
            "LoRA target 加 conv(同 Structural Fix 11 项),\n"
            "sync_loss=0.18 容错低,save_ckpt_steps=500 短段多存。\n"
            "数据准备:用 tools/preprocess_short_drama.py 把剧按场景切,\n"
            "再走 curate_finetune_samples.py 分桶。\n"
            "通用 baseline 见 ⚪ Stage 2 LoRA,单人 badcase 见 🟢 🎯 Badcase Fix。"
        ),
        "lora": {
            "enabled": True,
            "rank": 16,
            "alpha": 32,
            "dropout": 0.10,
            "target_modules": [
                # attention projections
                "to_q", "to_k", "to_v", "to_out.0",
                # Resnet convs
                "conv1", "conv2", "conv_shortcut",
                # Attention 1×1 conv re-mappers
                "proj_in", "proj_out",
                # UNet input/output gates
                "conv_in", "conv_out",
            ],
            "qlora": False,
        },
        "freeze_attn2": True,
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
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
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
# Dataset path presets - real example files so the UI can be exercised end-to-end
# ---------------------------------------------------------------------------

DATASET_PRESETS: Dict[str, Dict[str, str]] = {
    "assets 演示数据 (3 videos，可完整跑通)": {
        "train_data_dir": "assets",
        "train_fileslist": "data/demo_fileslist.txt",
        "val_video_path": "assets/demo1_video.mp4",
        "val_audio_path": "assets/demo1_audio.wav",
    },
    "assets 演示数据 (仅目录，自动生成 fileslist)": {
        "train_data_dir": "assets",
        "train_fileslist": "",
        "val_video_path": "assets/demo1_video.mp4",
        "val_audio_path": "assets/demo1_audio.wav",
    },
    "preprocess/high_visual_quality (示例路径)": {
        "train_data_dir": "preprocess/high_visual_quality",
        "train_fileslist": "preprocess/high_visual_quality/fileslist.txt",
        "val_video_path": "assets/demo1_video.mp4",
        "val_audio_path": "assets/demo1_audio.wav",
    },
    "data/train (示例路径)": {
        "train_data_dir": "data/train",
        "train_fileslist": "data/train/fileslist.txt",
        "val_video_path": "assets/demo1_video.mp4",
        "val_audio_path": "assets/demo1_audio.wav",
    },
}


# ---------------------------------------------------------------------------
# Process management for background training
# ---------------------------------------------------------------------------

def _prune_debug_files(directory: Path, pattern: str, keep: int = 10) -> None:
    """Keep only the N most-recently-modified files matching `pattern` in `directory`.

    Tab 3 / 3.5 write a fresh tmp yaml per invocation; over a long session this
    leaks disk. We cap it at `keep` files (oldest deleted first).
    """
    if not directory.exists():
        return
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in matches[keep:]:
        try:
            old.unlink()
        except OSError:
            pass


class InferenceManager:
    """Track a single background inference subprocess (used by Tab 3 / Tab 3.5).

    Tab 3 inference runs are 5-10 minutes, so blocking the Gradio event
    loop is unacceptable. We spawn via Popen in a daemon thread, return
    immediately from the click handler, and update the UI by polling
    `status` from a Timer. Only one inference at a time (mirrors
    TrainingProcess).
    """

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    CANCELLING = "cancelling"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.log_f = None
        self.log_path: Optional[Path] = None
        self.status: str = self.IDLE
        self.exit_code: Optional[int] = None
        self.result_video: Optional[Path] = None
        self.result_warning: str = ""
        self.kind: str = ""  # "compare" / "validate"
        self.label: str = ""  # human-readable description (e.g. "compare base vs ft")
        self._lock = threading.Lock()

    def is_busy(self) -> bool:
        return self.status in (self.STARTING, self.RUNNING, self.CANCELLING)

    def is_running(self) -> bool:
        return self.status == self.RUNNING and self.proc is not None and self.proc.poll() is None

    def start(
        self,
        cmd: List[str],
        log_path: Path,
        kind: str,
        label: str,
        result_video: Path,
    ) -> bool:
        """Spawn the subprocess in a daemon thread.

        Returns True on accept, False if another inference is already busy.
        """
        with self._lock:
            if self.is_busy():
                return False
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_f = open(log_path, "w")
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(REPO_ROOT),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,  # so SIGINT kills the whole process group
                )
            except FileNotFoundError:
                log_f.close()
                self.status = self.FAILED
                self.exit_code = -1
                return False

            self.proc = proc
            self.log_f = log_f
            self.log_path = log_path
            self.status = self.STARTING
            self.exit_code = None
            self.result_video = result_video
            self.result_warning = ""
            self.kind = kind
            self.label = label

        threading.Thread(
            target=self._monitor,
            args=(proc, log_f, log_path),
            daemon=True,
        ).start()
        return True

    def _monitor(self, proc: subprocess.Popen, log_f, log_path: Path) -> None:
        rc = proc.wait()
        log_f.close()
        with self._lock:
            self.exit_code = rc
            was_cancelling = self.status == self.CANCELLING
            if was_cancelling:
                self.status = self.CANCELLED
            elif rc == 0:
                self.status = self.DONE
            else:
                self.status = self.FAILED

    def stop(self) -> str:
        """Send SIGINT to the inference subprocess group. Non-blocking."""
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                return "(没有运行的推理)"
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
            except (ProcessLookupError, OSError) as exc:
                logger.warning("stop() failed: %s", exc)
            self.status = self.CANCELLING
        return "⏹ 停止信号已发出,等待子进程退出…"


_INFERENCE = InferenceManager()


class TrainingProcess:
    """Track a single background training subprocess.

    State is persisted to disk so that if the Gradio service restarts,
    we can reattach to a training subprocess that survived the restart
    (it runs in its own session via os.setsid).
    """

    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.log_path: Optional[Path] = None
        self.run_dir: Optional[Path] = None
        self.started_at: Optional[str] = None
        self.cmd: List[str] = []
        self._pid: Optional[int] = None

    @staticmethod
    def _state_path() -> Path:
        path = FINETUNE_BASE_DIR / "training_logs" / "active_trainer.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_state(self) -> None:
        """Write active trainer metadata to disk."""
        pid = self.proc.pid if self.proc else self._pid
        if pid is None:
            return
        data = {
            "pid": pid,
            "log_path": str(self.log_path) if self.log_path else None,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "started_at": self.started_at,
            "cmd": self.cmd,
        }
        try:
            self._state_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("[TrainingProcess] failed to save state: %s", e)

    def clear_state(self) -> None:
        """Remove persisted state."""
        path = self._state_path()
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                logger.warning("[TrainingProcess] failed to clear state: %s", e)

    def reattach(self) -> bool:
        """On startup, try to reattach to a training subprocess that is still alive."""
        path = self._state_path()
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("[TrainingProcess] corrupt state file %s: %s", path, e)
            self.clear_state()
            return False

        pid = data.get("pid")
        if not pid or not isinstance(pid, int):
            self.clear_state()
            return False

        if not self._pid_alive(pid):
            logger.info("[TrainingProcess] previously tracked pid=%s is no longer alive; clearing state", pid)
            self.clear_state()
            return False

        # Sanity check: the process should look like a LatentSync training job.
        try:
            cmdline = " ".join(psutil.Process(pid).cmdline() or [])
        except Exception:
            cmdline = ""
        if not any(token in cmdline for token in ("torchrun", "train_unet", "train_syncnet")):
            logger.warning(
                "[TrainingProcess] pid=%s does not look like a LatentSync training process (cmdline=%r); clearing state",
                pid, cmdline,
            )
            self.clear_state()
            return False

        self._pid = pid
        self.proc = None  # we don't have the Popen object, but we know the PID
        self.log_path = Path(data["log_path"]) if data.get("log_path") else None
        self.run_dir = Path(data["run_dir"]) if data.get("run_dir") else None
        self.started_at = data.get("started_at")
        self.cmd = data.get("cmd", [])
        logger.info(
            "[TrainingProcess] reattached to surviving training subprocess pid=%s, log=%s, run_dir=%s",
            pid, self.log_path, self.run_dir,
        )
        return True

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            proc = psutil.Process(pid)
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    def is_running(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        if self._pid is not None:
            alive = self._pid_alive(self._pid)
            if not alive:
                self._pid = None
                self.clear_state()
            return alive
        return False

    @property
    def pid(self) -> Optional[int]:
        if self.proc is not None:
            return self.proc.pid
        return self._pid

    def stop(self) -> None:
        pid = self.pid
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGINT)
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        elif pid is not None:
            # Reattached process: we only have the PID, send SIGINT to its group.
            try:
                os.killpg(os.getpgid(pid), signal.SIGINT)
                # Wait briefly for it to terminate.
                for _ in range(30):
                    if not self._pid_alive(pid):
                        break
                    time.sleep(0.5)
            except (ProcessLookupError, OSError) as e:
                logger.warning("[TrainingProcess] failed to signal pid=%s: %s", pid, e)
        self.proc = None
        self._pid = None
        self.clear_state()


_TRAINER = TrainingProcess()
# On module load, attempt to reattach to a training subprocess that survived a service restart.
_TRAINER.reattach()


def _on_page_load(train_output_dir: str = "unet"):
    """Repopulate training-status UI on page (re)load.

    The Python process keeps the trainer subprocess alive across browser
    refreshes, but the browser-side gr.State values (log_path, run_dd,
    …) reset to empty. This handler re-pulls from the in-process
    _TRAINER singleton and also refreshes the monitor tab so charts /
    logs / videos appear immediately instead of waiting for the first
    timer tick.
    """
    blank_monitor = (
        "",          # selected_run_disp
        None,        # loss_chart
        None,        # sync_conf_chart
        gr.update(), # val_video_dd
        gr.update(), # ckpt_dd
        "",          # log_box
        "",          # ckpt_info
        0.0,         # progress_pct
        "",          # progress_text
    )
    try:
        log_path = str(_TRAINER.log_path) if _TRAINER.log_path else ""
        if _TRAINER.is_running():
            launch_text = f"⏳ training running since {_TRAINER.started_at or '?'}"
            run_name = _TRAINER.run_dir.name if _TRAINER.run_dir else None
        else:
            launch_text = ""
            run_name = None

        core = _monitor_refresh_core(train_output_dir, run_name, log_path)
        run_dd_update = core[0]
        # Gradio >= 4 returns gr.update() as a plain dict.
        dd_choices = run_dd_update.get("choices") if isinstance(run_dd_update, dict) else getattr(run_dd_update, "choices", None)
        # If a training run is alive, prefer selecting it over the latest run.
        if (
            run_name
            and dd_choices
            and run_name in dd_choices
        ):
            run_dd_update = gr.update(choices=dd_choices, value=run_name)
        elif (
            run_name
            and _TRAINER.run_dir
            and dd_choices
            and str(_TRAINER.run_dir) in dd_choices
        ):
            run_dd_update = gr.update(
                choices=dd_choices, value=str(_TRAINER.run_dir)
            )

        return (
            core[8],          # trainer_status
            launch_text,
            log_path,
            run_dd_update,
            gr.update(interactive=True),
        ) + core[1:8] + core[9:]
    except Exception as exc:
        logger.exception("page-load handler failed entirely: %s", exc)
        return (
            f"⚠️ page-load handler 出错: {exc}",
            "",
            "",
            gr.update(choices=[], value=None),
            gr.update(interactive=True),
        ) + blank_monitor
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_run_dirs(base: Path) -> List[str]:
    """List timestamped run directories produced by train_unet.py / train_unet_lora.py / train_syncnet.py.

    Matches both `train-...` (full training) and `train_lora-...` (LoRA).
    Sorted by directory creation time (oldest first) so the latest run is
    last and can be auto-selected.
    """
    if not base.exists():
        return []
    runs = [
        p for p in base.iterdir()
        if p.is_dir() and (p.name.startswith("train-") or p.name.startswith("train_lora-"))
    ]
    # Sort by birth/creation time when available, falling back to ctime.
    runs.sort(key=lambda p: getattr(p.stat(), "st_birthtime", p.stat().st_ctime))
    try:
        return [str(p.relative_to(REPO_ROOT)) for p in runs]
    except ValueError:
        return [str(p) for p in runs]


def _resolve_run_dir(selected_run: Optional[str]) -> Optional[Path]:
    """Resolve a selected run path to an absolute Path."""
    if not selected_run:
        return None
    p = Path(selected_run)
    if p.is_absolute():
        return p
    # First try relative to REPO_ROOT (legacy / default small-disk layout)
    repo_candidate = REPO_ROOT / p
    if repo_candidate.exists():
        return repo_candidate
    # Then try relative to FINETUNE_BASE_DIR
    finetune_candidate = FINETUNE_BASE_DIR / p
    if finetune_candidate.exists():
        return finetune_candidate
    # Fallback: assume FINETUNE_BASE_DIR
    return finetune_candidate


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


def list_checkpoints(include_lora: bool = True) -> List[str]:
    """Available UNet / SyncNet checkpoints and LoRA adapter directories.

    Returns .pt files under checkpoints/ plus peft-format LoRA adapter
    directories found under FINETUNE_BASE_DIR/unet/*/checkpoints/ when
    ``include_lora`` is True.
    """
    out: List[str] = []
    if CHECKPOINT_DIR.exists():
        for p in sorted(CHECKPOINT_DIR.rglob("*.pt")):
            out.append(str(p.relative_to(REPO_ROOT)))

    # LoRA adapter directories produced by train_unet_lora.py
    if include_lora:
        lora_base = FINETUNE_BASE_DIR / "unet"
        if lora_base.exists():
            for ckpt_dir in sorted(lora_base.rglob("checkpoints")):
                for p in sorted(ckpt_dir.iterdir()):
                    if p.is_dir() and (p / "adapter_config.json").exists():
                        try:
                            out.append(str(p.relative_to(REPO_ROOT)))
                        except ValueError:
                            out.append(str(p))
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
    freeze_attn2: bool,
    val_inference_steps: int,
    val_guidance_scale: float,
    val_seed: int,
    lr_scheduler: str,
    lr_warmup_steps: int,
) -> Dict[str, Any]:
    """Merge user-form values with the chosen preset's defaults."""
    preset = PRESETS[preset_name]

    # Resolve train_output_dir relative to FINETUNE_BASE_DIR so the training
    # script (which runs with cwd=REPO_ROOT) puts outputs where the UI expects.
    train_output_dir = (train_output_dir or "").strip()
    if train_output_dir:
        p = Path(train_output_dir)
        if not p.is_absolute():
            train_output_dir = str(FINETUNE_BASE_DIR / p)
    else:
        train_output_dir = str(FINETUNE_BASE_DIR / "unet")

    cfg: Dict[str, Any] = {
        "data": {
            "train_data_dir": train_data_dir or "",
            "train_fileslist": train_fileslist or "",
            "val_video_path": val_video_path or str(ASSETS_DIR / "demo1_video.mp4"),
            "val_audio_path": val_audio_path or str(ASSETS_DIR / "demo1_audio.wav"),
            "audio_embeds_cache_dir": str(FINETUNE_BASE_DIR / "audio_embeds_cache"),
            "audio_mel_cache_dir": str(FINETUNE_BASE_DIR / "audio_mel_cache"),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "num_frames": int(num_frames),
            "resolution": int(resolution),
            "val_resolution": 512,
            "mask_image_path": mask_image_path,
            "audio_sample_rate": 16000,
            "video_fps": 25,
            "audio_feat_length": [2, 2],
            "train_output_dir": train_output_dir,
            # train_unet.py loads this to get the StableSyncNet checkpoint path.
            # It must point to a syncnet config, not the UNet config.
            "syncnet_config_path": str(SYNCNET_CONFIG_DIR / "syncnet_16_pixel_attn.yaml"),
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
            "guidance_scale": float(val_guidance_scale),
            "inference_steps": int(val_inference_steps),
            "seed": int(val_seed),
            "use_mixed_noise": True,
            "mixed_noise_alpha": 1,
            "mixed_precision_training": bool(mixed_precision_training),
            "enable_gradient_checkpointing": bool(enable_gradient_checkpointing),
            "max_train_steps": int(max_train_steps),
            "max_train_epochs": -1,
            "trainable_modules": ["motion_modules.", "attentions."] if use_motion_module else [],
        },
        "optimizer": {
            "lr": float(learning_rate),
            "scale_lr": False,
            "max_grad_norm": 1.0,
            "lr_scheduler": lr_scheduler,
            "lr_warmup_steps": int(lr_warmup_steps),
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
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
            "qlora": False,
            "freeze_attn2": False,
        }
    cfg["lora"]["freeze_attn2"] = bool(freeze_attn2)

    # Validate syncnet / num_frames compatibility early so the user gets a
    # clear message instead of a conv2d channel mismatch deep in training.
    if cfg["run"]["use_syncnet"] and cfg["run"]["pixel_space_supervise"]:
        import yaml as _yaml
        syncnet_cfg_path = Path(cfg["data"]["syncnet_config_path"])
        syncnet_cfg = _yaml.safe_load(syncnet_cfg_path.read_text())
        syncnet_num_frames = syncnet_cfg.get("data", {}).get("num_frames")
        if syncnet_num_frames is not None and syncnet_num_frames != cfg["data"]["num_frames"]:
            raise ValueError(
                f"SyncNet config `{syncnet_cfg_path.name}` expects num_frames={syncnet_num_frames}, "
                f"but training is configured with num_frames={cfg['data']['num_frames']}. "
                f"Please set num_frames to {syncnet_num_frames} (or disable use_syncnet)."
            )
        syncnet_ckpt = syncnet_cfg.get("ckpt", {}).get("inference_ckpt_path", "")
        if not syncnet_ckpt:
            raise ValueError(
                f"SyncNet config `{syncnet_cfg_path.name}` has no inference_ckpt_path. "
                "Please use a syncnet config that points to a valid checkpoint."
            )

    return cfg


# ---------------------------------------------------------------------------
# Tab 1: Configure & launch
# ---------------------------------------------------------------------------

def on_preset_change(preset_name: str) -> Tuple[Any, ...]:
    """When user picks a preset, fill the form fields with preset defaults."""
    preset = PRESETS[preset_name]
    # Fallback for keys older presets don't carry — current form-value semantics
    # for save_ckpt_steps / max_train_steps / lr_*. Existing Stage 1 / Stage 2
    # presets still match the form's default behavior.
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
        preset.get("save_ckpt_steps", 10000),
        preset.get("max_train_steps", 10000),
        preset.get("lr_scheduler", "constant"),
        preset.get("lr_warmup_steps", 0),
        preset["description"],
        preset.get("freeze_attn2", False),
    )


def apply_dataset_preset(preset_name: str) -> Tuple[str, str, str, str]:
    """Fill train_data_dir / train_fileslist / val paths from a dataset preset."""
    preset = DATASET_PRESETS.get(preset_name, {})
    return (
        preset.get("train_data_dir", ""),
        preset.get("train_fileslist", ""),
        preset.get("val_video_path", "assets/demo1_video.mp4"),
        preset.get("val_audio_path", "assets/demo1_audio.wav"),
    )


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
            return (
                f"❌ 训练已在运行中 (pid={_TRAINER.pid}, run={_TRAINER.run_dir.name})",
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


def _resolve_output_dir(train_output_dir: str) -> Path:
    """Resolve a possibly relative train_output_dir against FINETUNE_BASE_DIR."""
    if not train_output_dir:
        return FINETUNE_BASE_DIR / "unet"
    p = Path(train_output_dir)
    if p.is_absolute():
        return p
    return FINETUNE_BASE_DIR / p


def _list_run_dirs_for_monitor(train_output_dir: str) -> List[str]:
    """List runs from the configured output dir and, for migration convenience,
    also from the legacy REPO_ROOT-based location.
    """
    base = _resolve_output_dir(train_output_dir)
    choices = list_run_dirs(base)
    p = Path(train_output_dir or "unet")
    legacy_base = REPO_ROOT / p.name if p.is_absolute() else REPO_ROOT / p
    if legacy_base != base:
        legacy_choices = list_run_dirs(legacy_base)
        existing = set(choices)
        choices = choices + [c for c in legacy_choices if c not in existing]
    return choices


def refresh_runs(train_output_dir: str) -> gr.update:
    choices = _list_run_dirs_for_monitor(train_output_dir)
    return gr.update(choices=choices, value=choices[-1] if choices else None)


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
    """Surface the latest PNG from the run's loss_charts/ directory.

    Separate from sync_conf so the UI can show two side-by-side charts.
    Returns the path to the most-recent PNG, or None.
    """
    return _latest_png_in_subdir(run_dir_path, "loss_charts")


def parse_sync_conf_chart(run_dir_path: Optional[str]) -> Optional[str]:
    """Surface the latest PNG from the run's sync_conf_results/ directory."""
    return _latest_png_in_subdir(run_dir_path, "sync_conf_results")


def _latest_png_in_subdir(run_dir_path: Optional[str], sub: str) -> Optional[str]:
    rd = Path(run_dir_path) if run_dir_path else None
    if rd is None:
        return None
    if not rd.is_absolute():
        rd = _resolve_run_dir(run_dir_path)
        if rd is None:
            return None
    d = rd / sub
    if not d.exists():
        return None
    pngs = sorted(d.glob("*.png"), key=lambda p: p.stat().st_mtime)
    return str(pngs[-1]) if pngs else None


def list_validation_videos(run_dir_path: Optional[str]) -> List[str]:
    if not run_dir_path:
        return []
    rd = Path(run_dir_path)
    if not rd.is_absolute():
        rd = _resolve_run_dir(run_dir_path)
        if rd is None:
            return []
    vd = rd / "val_videos"
    if not vd.exists():
        return []
    return sorted([str(p) for p in vd.glob("*.mp4")], key=lambda p: Path(p).stat().st_mtime, reverse=True)


def list_checkpoints_in_run(run_dir_path: Optional[str]) -> List[str]:
    """List checkpoints in a run dir.

    Supports both full-UNet .pt checkpoints and LoRA adapter directories
    (peft format, contain adapter_config.json).
    """
    if not run_dir_path:
        return []
    rd = Path(run_dir_path)
    if not rd.is_absolute():
        rd = _resolve_run_dir(run_dir_path)
        if rd is None:
            return []
    ck = rd / "checkpoints"
    if not ck.exists():
        return []

    candidates: List[Path] = []
    # Full UNet checkpoints
    candidates.extend(ck.glob("*.pt"))
    # LoRA adapter directories
    for p in ck.iterdir():
        if p.is_dir() and (p / "adapter_config.json").exists():
            candidates.append(p)

    candidates.sort(key=lambda p: p.stat().st_mtime)
    try:
        return [str(p.relative_to(REPO_ROOT)) for p in candidates]
    except ValueError:
        return [str(p) for p in candidates]


def read_loss_from_checkpoint(ckpt_path: str) -> str:
    """Best-effort: dump global_step + a couple of scalar fields from the
    latest checkpoint so the user gets a textual progress signal without
    loading the full state.

    Supports both full-UNet .pt files and LoRA adapter directories.
    """
    if not ckpt_path:
        return "(checkpoint path empty)"
    p = Path(ckpt_path)
    if not p.exists():
        return "(checkpoint not found)"

    # LoRA adapter directory (peft format)
    if p.is_dir():
        adapter_cfg = p / "adapter_config.json"
        if adapter_cfg.exists():
            try:
                import json
                cfg = json.loads(adapter_cfg.read_text())
                # global_step is encoded in the directory name: checkpoint-XXXX
                step_str = p.name.split("-")[-1]
                try:
                    global_step = int(step_str)
                except ValueError:
                    global_step = "?"
                info = {
                    "type": "LoRA adapter",
                    "global_step": global_step,
                    "lora_rank": cfg.get("r", "?"),
                    "lora_alpha": cfg.get("lora_alpha", "?"),
                    "target_modules": cfg.get("target_modules", []),
                    "adapter_path": str(p),
                }
                return json.dumps(info, indent=2)
            except Exception as e:
                return f"(could not read adapter config: {e})"
        return "(directory is not a LoRA adapter)"

    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        info = {
            "type": "full UNet checkpoint",
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


def _run_dir_start_time(run_dir: Path) -> Optional[float]:
    """Best-effort start timestamp for a run directory.

    The directory name contains an ISO-like timestamp (e.g.
    train_lora-2026_07_14-14:12:26). Parse it when possible; otherwise fall
    back to filesystem birth/ctime.
    """
    import re
    m = re.search(r"-(\d{4}_\d{2}_\d{2}-\d{2}:\d{2}:\d{2})\Z", run_dir.name)
    if m:
        try:
            from datetime import datetime
            dt = datetime.strptime(m.group(1), "%Y_%m_%d-%H:%M:%S")
            return dt.timestamp()
        except Exception:
            pass
    stat = run_dir.stat()
    return getattr(stat, "st_birthtime", stat.st_ctime)


def _compute_progress(run_dir: Optional[Path]) -> Dict[str, Any]:
    """Pull step / max_step / elapsed / throughput / ETA for the run.

    Returns a dict; missing data fields fall back to None / 0 so the
    UI doesn't crash on freshly-started or stale runs.
    """
    if not run_dir or not Path(run_dir).exists():
        return {"step": None, "max_step": None, "elapsed_s": 0,
                "throughput": 0.0, "eta_s": None, "progress_pct": 0.0,
                "latest_loss": None}

    try:
        ckpts = list_checkpoints_in_run(str(run_dir))
        latest_ckpt = Path(ckpts[-1]) if ckpts else None
        if latest_ckpt and not latest_ckpt.is_absolute():
            latest_ckpt = REPO_ROOT / latest_ckpt
        if latest_ckpt is None:
            raise ValueError("no checkpoint")

        step = 0
        latest_loss = None
        # LoRA adapter directory: step is encoded in the directory name.
        if latest_ckpt.is_dir() and (latest_ckpt / "adapter_config.json").exists():
            step_str = latest_ckpt.name.split("-")[-1]
            step = int(step_str) if step_str.isdigit() else 0
        else:
            import torch as _torch
            ckpt = _torch.load(str(latest_ckpt), map_location="cpu", weights_only=False)
            step = int(ckpt.get("global_step", 0) or 0)
            if ckpt.get("train_loss_list"):
                latest_loss = float(ckpt["train_loss_list"][-1])
    except Exception:
        return {"step": None, "max_step": None, "elapsed_s": 0,
                "throughput": 0.0, "eta_s": None, "progress_pct": 0.0,
                "latest_loss": None}

    # max_step from the yaml config that train_unet*.py copied into the
    # run dir at startup. Skip syncnet config files which also have run.max_train_steps.
    max_step = None
    try:
        yamls = [
            y for y in Path(run_dir).glob("*.yaml")
            if not y.name.lower().startswith("syncnet")
        ]
        if yamls:
            cfg = OmegaConf.load(str(yamls[0]))
            max_step = int(cfg.run.max_train_steps)
    except Exception:
        pass

    # elapsed = wall-clock since the run started. Prefer the timestamp encoded
    # in the directory name; fall back to filesystem birth/ctime.
    elapsed_s = 0.0
    try:
        start_ts = _run_dir_start_time(Path(run_dir))
        elapsed_s = max(0.0, time.time() - start_ts)
    except Exception:
        pass

    throughput = step / elapsed_s if elapsed_s > 0 else 0.0
    eta_s = None
    progress_pct = 0.0
    if max_step and max_step > 0 and step > 0:
        progress_pct = min(100.0, 100.0 * step / max_step)
        if throughput > 0:
            remaining = max(0, max_step - step)
            eta_s = remaining / throughput
        elif elapsed_s > 0 and progress_pct > 0:
            # No throughput yet — best-effort linear extrapolation.
            eta_s = (elapsed_s / (progress_pct / 100.0)) - elapsed_s

    return {
        "step": step,
        "max_step": max_step,
        "elapsed_s": elapsed_s,
        "throughput": throughput,
        "eta_s": eta_s,
        "progress_pct": progress_pct,
        "latest_loss": latest_loss,
    }


def _format_progress_text(p: Dict[str, Any]) -> str:
    """Pretty-print progress dict for the gradio Textbox."""
    if p["step"] is None:
        return "⏸ 未启动 / 没有 checkpoint 可读"
    step = p["step"]
    max_step = p["max_step"] or "?"
    pct = p["progress_pct"]
    elapsed = p["elapsed_s"]
    throughput = p["throughput"]
    eta = p["eta_s"]
    loss = p["latest_loss"]
    loss_str = f", loss={loss:.4f}" if loss is not None else ""
    throughput_str = f"{throughput:.2f} step/s" if throughput > 0 else "—"
    if eta is None:
        eta_str = "—"
    elif eta < 60:
        eta_str = f"{eta:.0f}s"
    elif eta < 3600:
        eta_str = f"{eta/60:.1f}min"
    else:
        eta_str = f"{eta/3600:.1f}h"
    elapsed_str = (
        f"{int(elapsed//3600)}h{int((elapsed%3600)//60)}m{int(elapsed%60)}s"
        if elapsed >= 60 else f"{int(elapsed)}s"
    )
    return (
        f"📈 step {step} / {max_step}  ({pct:.1f}%){loss_str}\n"
        f"⏱ 已运行: {elapsed_str} | 速度: {throughput_str} | ETA: {eta_str}"
    )


def _run_curate_finetune(
    urls: str,
    source_dir: str,
    output_dir: str,
    scale: str,
) -> str:
    """Spawn `python -m tools.download_curated_finetune_set` and stream output."""
    urls = (urls or "").strip()
    source_dir = (source_dir or "").strip()
    if not urls and not source_dir:
        return "❌ 必须填 URL 列表 或 本地源目录(选一)"
    if not output_dir:
        return "❌ curated 输出目录为空"

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
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT)
        log_text = log_path.read_text()
        if proc.returncode != 0:
            return (
                f"❌ curate 退出码 {proc.returncode}\n"
                f"📜 log: {log_path}\n\n最后 60 行:\n{log_text[-6000:]}"
            )
        return f"✅ 已完成\n📜 log: {log_path}\n\n{log_text[-4000:]}"
    except FileNotFoundError as exc:
        return f"❌ 启动失败: {exc}"


# ---------------------------------------------------------------------------
# Pre-built dataset initializer (HF Hub auto-download + curate)
# ---------------------------------------------------------------------------

PREBUILT_DATASETS_YAML = REPO_ROOT / "tools" / "prebuilt_datasets.yaml"


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


def _tail_text(path: Path, n_lines: int = 40) -> str:
    """Return the last n lines of a text file (best-effort)."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-n_lines:])


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
                bufsize=1, universal_newlines=True,
            )
            while proc.poll() is None:
                time.sleep(0.5)
                tail = _tail_text(log_path, n_lines=40)
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


def _monitor_refresh_core(
    train_output_dir: str,
    selected_run: Optional[str],
    log_path: Optional[str],
) -> Tuple[Any, str, Any, Any, Any, str, str, str, str, float, str]:
    """Shared implementation for monitor_refresh and page-load refresh.

    Returns: (run_dir_choices, selected_run_disp, loss_chart, sync_conf_chart,
    val_video_choices, checkpoint_choices, log_tail, ckpt_info, trainer_status,
    progress_pct, progress_text).
    """
    run_choices = _list_run_dirs_for_monitor(train_output_dir)
    # Auto-select the latest run if none is selected so the page isn't blank
    # on first load. If the incoming selection is stale (e.g. browser cached a
    # path that no longer exists), reset it to avoid Gradio dropdown errors.
    if not selected_run and run_choices:
        selected_run = run_choices[-1]
    elif selected_run and selected_run not in run_choices:
        selected_run = run_choices[-1] if run_choices else None

    run_dir = _resolve_run_dir(selected_run)
    run_path = str(run_dir) if run_dir else None
    chart = parse_loss_chart(run_path)
    sync_chart = parse_sync_conf_chart(run_path)
    val_videos = list_validation_videos(run_path)
    val_video_update = gr.update(choices=val_videos, value=val_videos[0] if val_videos else None)
    ckpts = list_checkpoints_in_run(run_path)
    ckpt_update = gr.update(choices=ckpts, value=ckpts[-1] if ckpts else None)

    # Fall back to the in-process trainer log if the browser-side state is empty
    # (common right after a page refresh).
    effective_log_path = log_path
    if not effective_log_path and _TRAINER.is_running() and _TRAINER.log_path:
        effective_log_path = str(_TRAINER.log_path)
    log_text = tail_log(effective_log_path, n_lines=80)

    if ckpts:
        ckpt_path = Path(ckpts[-1])
        if not ckpt_path.is_absolute():
            ckpt_path = REPO_ROOT / ckpt_path
        ckpt_info = read_loss_from_checkpoint(str(ckpt_path))
    else:
        ckpt_info = "(no checkpoint yet)"

    # Trainer status should reflect the selected run, not just any trainer.
    if not run_dir:
        if _TRAINER.is_running():
            status = (
                f"🟢 训练进行中 (pid={_TRAINER.pid}, "
                f"started={_TRAINER.started_at or '-'}, log={_TRAINER.log_path or '-'}) | "
                f"未选择 run，请点击 '🔄 刷新 run 列表'"
            )
        else:
            status = "ℹ️ 未选择 run"
    elif _TRAINER.is_running() and _TRAINER.run_dir and run_dir.parent == _TRAINER.run_dir:
        status = (
            f"🟢 当前 run 训练中 (pid={_TRAINER.pid}, "
            f"started={_TRAINER.started_at or '-'}, log={_TRAINER.log_path or '-'})"
        )
    elif _TRAINER.is_running():
        status = (
            f"ℹ️ 有其它训练在跑 (pid={_TRAINER.pid}); "
            f"当前选中 run 未在训练"
        )
    else:
        status = "⏸ 当前选中 run 未在训练"

    progress = _compute_progress(run_dir)
    progress_pct = float(progress["progress_pct"])
    progress_text = _format_progress_text(progress)
    return (
        gr.update(choices=run_choices, value=selected_run),
        str(run_dir) if run_dir else "",
        chart,
        sync_chart,
        val_video_update,
        ckpt_update,
        log_text,
        ckpt_info,
        status,
        progress_pct,
        progress_text,
    )


def monitor_refresh(
    train_output_dir: str,
    selected_run: Optional[str],
    log_path: Optional[str],
) -> Tuple[Any, str, Any, Any, Any, str, str, str, str, float, str]:
    """Pull the latest snapshot."""
    return _monitor_refresh_core(train_output_dir, selected_run, log_path)


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
    baseline_mode: bool = False,
) -> Tuple[str, str]:
    """Run inference twice (base vs fine-tuned) and return the two mp4 paths."""
    if not video_path or not audio_path:
        raise gr.Error("请先上传视频和音频")
    if not base_ckpt or not fine_tuned_ckpt:
        raise gr.Error("请选择 base 和 fine-tuned 两个 checkpoint")
    if base_ckpt == fine_tuned_ckpt:
        raise gr.Error("两个 checkpoint 必须不同")

    _prune_debug_files(REPO_ROOT / "debug", "compare_cfg_*.yaml")
    _prune_debug_files(REPO_ROOT / "debug" / "compare_outputs", "compare_cfg_*.yaml")

    out_dir = REPO_ROOT / "debug" / "compare_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = out_dir / f"base_{ts}.mp4"
    out_ft = out_dir / f"finetuned_{ts}.mp4"

    config_path = CONFIG_DIR / "stage2.yaml"

    def _run(ckpt_path: Path, out_path: Path) -> None:
        # If a LoRA adapter directory was selected, merge it into the base UNet first.
        if ckpt_path.is_dir() and (ckpt_path / "adapter_config.json").exists():
            try:
                merged_ckpt = _merge_adapter_to_temp_pt(
                    ckpt_path,
                    base_ckpt="checkpoints/latentsync_unet.pt",
                    unet_config=config_path,
                )
                ckpt_path = merged_ckpt
            except Exception as exc:
                logger.exception("[run_compare] failed to merge adapter %s", ckpt_path)
                raise gr.Error(f"LoRA adapter 合并失败: {exc}")

        cmd = [
            sys.executable,
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
        if baseline_mode:
            cmd.append("--baseline_mode")
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
# Tab 4: Validation - run inference on a single ckpt with quality self-check
# ---------------------------------------------------------------------------

def _check_ckpt_compatibility(ckpt_path: Path, unet_config: Path) -> List[str]:
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
    if "lora" in ckpt_name.lower():
        warnings.append("⚠️ ckpt 名字含 'lora' —— 可能是 LoRA adapter，需要先 merge_lora.py")
    if "adapter" in ckpt_name.lower():
        warnings.append("⚠️ ckpt 是 peft adapter，需要先 merge_lora.py")
    return warnings


def _quick_quality_check(video_path: str) -> Dict[str, Any]:
    """Run a lightweight quality check on a single generated video.

    Computes: sharpness (Laplacian), flicker (frame diff), face
    detection rate. Skips SyncNet / HyperIQA (those are too heavy
    for a per-click UI call).
    """
    if not video_path or not Path(video_path).exists():
        return {"error": f"video not found: {video_path}"}
    try:
        from decord import VideoReader
        import numpy as np
        import cv2

        vr = VideoReader(video_path)
        frames = [f.asnumpy() for f in vr]
        if len(frames) < 2:
            return {"error": "video too short"}

        # sharpness per frame
        grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        laps = [cv2.Laplacian(g, cv2.CV_64F).var() for g in grays]
        sharpness_mean = float(np.mean(laps))
        blurry_count = sum(1 for v in laps if v < 50)
        blurry_ratio = blurry_count / len(laps)

        # flicker
        diffs = [
            float(np.abs(frames[i].astype(float) - frames[i - 1].astype(float)).mean())
            for i in range(1, len(frames))
        ]
        flicker = float(np.mean(diffs))

        # face detection rate
        try:
            from latentsync.utils.face_detector import FaceDetector
            det = FaceDetector()
            detected = 0
            for f in frames[::max(1, len(frames) // 10)][:10]:
                face, _, _ = det.detect(f)
                if face is not None:
                    detected += 1
            face_rate = detected / min(10, len(frames))
        except Exception:
            face_rate = None

        return {
            "num_frames": len(frames),
            "sharpness_mean": round(sharpness_mean, 2),
            "blurry_ratio": round(blurry_ratio, 3),
            "flicker": round(flicker, 2),
            "face_detect_rate": round(face_rate, 2) if face_rate is not None else None,
        }
    except Exception as e:
        return {"error": str(e)}


def _format_validation_report(metrics: Dict[str, Any], ckpt_path: str,
                              duration_sec: float) -> str:
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


def _merge_adapter_to_temp_pt(
    adapter_dir: Path,
    base_ckpt: str,
    unet_config: Path,
) -> Path:
    """Merge a peft LoRA adapter into the base UNet and save a temporary .pt.

    Used by Tab 3.5 so users can select a LoRA adapter directory directly
    without manually running scripts/merge_lora.py first.

    Reuses an existing merged checkpoint when the (adapter_dir, base_ckpt)
    pair has already been processed in this session, avoiding redundant
    4-5 GB writes.
    """
    from latentsync.models.unet import UNet3DConditionModel
    import hashlib
    import torch

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

    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(base, str(adapter_dir), device="cpu")
    merged = peft_model.merge_and_unload()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_pt = out_dir / f"{adapter_dir.name}_{cache_key}_{ts}.pt"
    torch.save({"global_step": 0, "state_dict": merged.state_dict()}, out_pt)
    logger.info("[validation] saved merged ckpt to %s", out_pt)
    return out_pt


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
) -> Tuple[Any, Any, Any, Any]:
    """Kick off Tab-3.5 single-ckpt inference in the background.

    Returns immediately with a '⏳ running' status; the actual result
    is delivered by the Timer poller `_poll_inference_state` which
    updates the video / report components when the subprocess finishes.
    """
    if _INFERENCE.is_busy():
        return (
            gr.update(value=None),
            gr.update(),
            gr.update(value=f"❌ 已有推理在运行 (kind={_INFERENCE.kind})，请先 ⏹ 取消"),
            gr.update(value=None),
            gr.update(interactive=True),
        )
    if not ckpt_path or not Path(ckpt_path).exists():
        return (gr.update(value=None), gr.update(), gr.update(value=f"❌ ckpt 不存在: {ckpt_path}"), gr.update(), gr.update(interactive=True))
    if not unet_config or not Path(unet_config).exists():
        return (gr.update(value=None), gr.update(), gr.update(value=f"❌ config 不存在: {unet_config}"), gr.update(), gr.update(interactive=True))

    # If the user selected a LoRA adapter directory, merge it into the base
    # UNet on-the-fly so the rest of the validation path can treat it as a
    # regular .pt checkpoint.
    ckpt = Path(ckpt_path)
    merged_from_adapter: Optional[Path] = None
    if ckpt.is_dir() and (ckpt / "adapter_config.json").exists():
        try:
            merged_from_adapter = _merge_adapter_to_temp_pt(
                ckpt,
                base_ckpt="checkpoints/latentsync_unet.pt",
                unet_config=Path(unet_config),
            )
            ckpt_path = str(merged_from_adapter)
            ckpt = merged_from_adapter
        except Exception as exc:
            logger.exception("[run_validation] failed to merge adapter %s", ckpt_path)
            return (
                gr.update(value=None),
                gr.update(),
                gr.update(value=f"❌ LoRA adapter 合并失败: {exc}"),
                gr.update(),
                gr.update(interactive=True),
            )

    warnings = _check_ckpt_compatibility(Path(ckpt_path), Path(unet_config))
    if merged_from_adapter is not None:
        warnings.insert(0, f"ℹ️ 已自动合并 LoRA adapter: {merged_from_adapter.name}")
    warnings_text = "\n".join(warnings) if warnings else "✅ ckpt 与 config 兼容"

    _prune_debug_files(REPO_ROOT / "debug" / "validation_outputs", "validation_cfg_*.yaml")

    base_cfg = OmegaConf.load(unet_config)
    base_cfg.data.resolution = int(resolution)
    base_cfg.run.inference_steps = int(inference_steps)
    base_cfg.run.guidance_scale = float(guidance_scale)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "debug" / "validation_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"validation_{ts}.mp4"
    tmp_cfg = out_dir / f"validation_cfg_{ts}.yaml"
    with open(tmp_cfg, "w") as f:
        yaml.dump(OmegaConf.to_container(base_cfg), f)

    cmd = [
        sys.executable, "-m", "scripts.inference",
        "--unet_config_path", str(tmp_cfg),
        "--inference_ckpt_path", str(ckpt_path),
        "--video_path", str(video_path),
        "--audio_path", str(audio_path),
        "--video_out_path", str(out_mp4),
        "--inference_steps", str(int(inference_steps)),
        "--guidance_scale", str(float(guidance_scale)),
        "--seed", str(int(seed)),
        "--temp_dir", "temp",
    ]
    if enable_deepcache:
        cmd.append("--enable_deepcache")
    if baseline_mode:
        cmd.append("--baseline_mode")
    log_path = out_dir / f"validation_{ts}.log"

    if not _INFERENCE.start(
        cmd, log_path,
        kind="validate",
        label=f"validate ckpt={Path(ckpt_path).name}",
        result_video=out_mp4,
    ):
        return (gr.update(value=None), gr.update(), gr.update(value="❌ 启动失败"), gr.update(), gr.update(interactive=True))

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


def _poll_inference_state(skip_quality_check: bool):
    """Timer poller — checks _INFERENCE status and delivers the result.

    Called by Tab 3.5's gr.Timer every 1s while the app is open. We always
    update the report text so the user can see the poller is alive. On a
    finished result (DONE/FAILED/CANCELLED) we push the video path + report
    into the UI and reset the one-shot state.
    """
    state = _INFERENCE
    now = datetime.now().strftime("%H:%M:%S")
    logger.debug(
        "[_poll_inference_state] kind=%s status=%s busy=%s",
        state.kind, state.status, state.is_busy(),
    )

    # Helper to build the report line that proves the poller is ticking.
    def _heartbeat(prefix: str) -> str:
        return f"[{now}] {prefix} | kind={state.kind} status={state.status} busy={state.is_busy()}"

    # Not a validation result we should consume (e.g. idle, or compare running).
    if state.kind != "validate" or state.status not in (state.DONE, state.FAILED, state.CANCELLED):
        if state.is_busy():
            try:
                pid = state.proc.pid if state.proc else "?"
            except Exception:
                pid = "?"
            return (
                gr.update(value=None),
                gr.update(),
                gr.update(value=_heartbeat(
                    f"⏳ {state.label} running (pid={pid})\n"
                    "💡 页面每秒自动刷新，推理完成后会在此显示视频"
                )),
                gr.update(),
                gr.update(interactive=False),
            )
        # Idle / no validation running: leave UI unchanged so we don't clobber
        # a previous result or spam the report when the tab is open.
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(interactive=True),
        )

    # ---- one-shot consumption of a finished result ----
    result_video = state.result_video
    exit_code = state.exit_code
    log_path = state.log_path
    label = state.label

    if state.status == state.DONE:
        # Resolve to absolute path for Gradio's Video component.
        video_path_str = str(result_video.resolve()) if result_video else ""

        if result_video is None or not result_video.exists():
            _INFERENCE.status = _INFERENCE.IDLE
            _INFERENCE.result_video = None
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

        # Build the report *before* resetting state so any exception below
        # doesn't lose the video path.
        try:
            if skip_quality_check:
                report = (
                    f"[{now}] ✅ 推理完成（已跳过质量自检）\n"
                    f"📂 {video_path_str}\n"
                    f"📜 log: {log_path}"
                )
            else:
                metrics = _quick_quality_check(video_path_str)
                report = _format_validation_report(metrics, label, duration=0.0)
                report = f"[{now}] {report}\n📂 {video_path_str}\n📜 log: {log_path}"
        except Exception as exc:
            logger.exception("[_poll_inference_state] quality check failed")
            report = (
                f"[{now}] ✅ 推理完成，但质量自检异常: {exc}\n"
                f"📂 {video_path_str}\n"
                f"📜 log: {log_path}"
            )

        logger.info("[_poll_inference_state] validation done: %s", video_path_str)
        # Consume the one-shot result.
        _INFERENCE.status = _INFERENCE.IDLE
        _INFERENCE.result_video = None
        return (
            gr.update(value=video_path_str),
            gr.update(),
            gr.update(value=report),
            gr.update(value=video_path_str),
            gr.update(interactive=True),
        )

    if state.status == state.CANCELLED:
        _INFERENCE.status = _INFERENCE.IDLE
        _INFERENCE.result_video = None
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
    _INFERENCE.status = _INFERENCE.IDLE
    _INFERENCE.result_video = None
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


def stop_inference() -> Tuple[str, Any]:
    """⏹ cancel button — sends SIGINT to the running inference subprocess group."""
    return _INFERENCE.stop(), gr.update(interactive=True)
    """Run inference with the chosen fine-tuned checkpoint, then a quick
    quality self-check. Returns (output_mp4, warnings_text, report_text,
    saved_path).
    """
    if not video_path or not audio_path:
        raise gr.Error("请先上传视频和音频")
    if not ckpt_path:
        raise gr.Error("请选择 checkpoint")
    if not unet_config:
        unet_config = str(CONFIG_DIR / "stage2.yaml")

    ckpt = Path(ckpt_path)
    cfg = Path(unet_config)
    if not ckpt.exists():
        raise gr.Error(f"checkpoint 不存在: {ckpt}")
    if not cfg.exists():
        raise gr.Error(f"unet config 不存在: {cfg}")

    # Pre-flight compatibility check
    warnings = _check_ckpt_compatibility(ckpt, cfg)
    warnings_text = "\n".join(warnings) if warnings else "✅ ckpt 与 config 兼容"

    _prune_debug_files(REPO_ROOT / "debug" / "validation_outputs", "validation_cfg_*.yaml")

    # Build a per-run config with the user-chosen resolution / steps / guidance
    base_cfg = OmegaConf.load(cfg)
    base_cfg.data.resolution = int(resolution)
    base_cfg.run.inference_steps = int(inference_steps)
    base_cfg.run.guidance_scale = float(guidance_scale)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "debug" / "validation_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"validation_{ts}.mp4"
    tmp_cfg = out_dir / f"validation_cfg_{ts}.yaml"
    with open(tmp_cfg, "w") as f:
        yaml.dump(OmegaConf.to_container(base_cfg), f)

    # Run inference
    cmd = [
        "python", "-m", "scripts.inference",
        "--unet_config_path", str(tmp_cfg),
        "--inference_ckpt_path", str(ckpt),
        "--video_path", str(video_path),
        "--audio_path", str(audio_path),
        "--video_out_path", str(out_mp4),
        "--inference_steps", str(int(inference_steps)),
        "--guidance_scale", str(float(guidance_scale)),
        "--seed", str(int(seed)),
        "--temp_dir", "temp",
    ]
    if enable_deepcache:
        cmd.append("--enable_deepcache")
    log_path = out_dir / f"validation_{ts}.log"

    t0 = time.time()
    try:
        with open(log_path, "w") as logf:
            rc = subprocess.call(cmd, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT)
    except FileNotFoundError as e:
        raise gr.Error(f"启动失败: {e}")
    duration = time.time() - t0

    if rc != 0:
        err_text = (
            f"❌ 推理失败 (rc={rc})\n"
            f"📜 完整 log: {log_path.relative_to(REPO_ROOT)}\n\n"
            f"最后 30 行:\n{tail_file(log_path, 30)}"
        )
        # Return gr.update(value=None) for the video output (Gradio >= 5
        # postprocesses None / empty list as a (path, subtitle) tuple
        # which gr.Video can't parse. gr.update keeps the previous value
        # in the UI and avoids the ValueError).
        return (
            gr.update(value=None),
            gr.update(value=warnings_text),
            gr.update(value=err_text),
            gr.update(value=str(out_mp4)),
        )

    # Quality self-check
    if skip_quality_check:
        report = (
            f"✅ 推理完成 ({duration:.1f}s)\n"
            f"⏭ 跳过质量自检（用户关闭）\n"
            f"📂 输出: {out_mp4.relative_to(REPO_ROOT)}"
        )
    else:
        metrics = _quick_quality_check(str(out_mp4))
        report = _format_validation_report(metrics, str(ckpt), duration)

    return (str(out_mp4), warnings_text, report, str(out_mp4))


def tail_file(path: Path, n_lines: int = 30) -> str:
    """Read the last N lines of a text file (best-effort)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 20_000))
            data = f.read().decode("utf-8", errors="replace")
        return "\n".join(data.splitlines()[-n_lines:])
    except Exception as e:
        return f"(log read failed: {e})"


def _resolve_training_video_path(line: str, data_dir: Optional[str]) -> Optional[Path]:
    """Best-effort resolve a path from a fileslist or scan result.

    Tries the literal path first, then relative to data_dir if provided.
    """
    candidates = [Path(line.strip())]
    if data_dir:
        candidates.append(Path(data_dir) / line.strip())
    for p in candidates:
        if p.exists():
            return p
    return None


def _list_training_videos(data_dir: str, fileslist: str) -> Tuple[List[str], str]:
    """List training videos from a fileslist or by scanning data_dir.

    Returns (video_paths, status_text).
    """
    data_dir = (data_dir or "").strip()
    fileslist = (fileslist or "").strip()

    if fileslist and Path(fileslist).exists():
        try:
            with open(fileslist, "r", encoding="utf-8") as f:
                raw_lines = [line.strip() for line in f if line.strip()]
        except Exception as exc:
            return [], f"❌ 读取 fileslist 失败: {exc}"
        resolved = []
        missing = []
        for line in raw_lines:
            p = _resolve_training_video_path(line, data_dir)
            if p is not None:
                resolved.append(str(p))
            else:
                missing.append(line)
        status = f"📁 fileslist: {len(resolved)} 个视频"
        if missing:
            status += f" ({len(missing)} 个路径未找到)"
        return resolved, status

    if data_dir and Path(data_dir).exists():
        videos = sorted([str(p) for p in Path(data_dir).rglob("*.mp4")])
        return videos, f"📁 扫描目录: {len(videos)} 个 mp4"

    return [], "⚠️ 请提供有效的 train_data_dir 或 train_fileslist"


def _analyze_training_video_yaw(video_path: str, n_frames: int = 5) -> Dict[str, Any]:
    """Sample frames from a training video and estimate face yaw.

    Returns a dict with yaw_mean, yaw_max, detect_rate, face_type and an
    optional error key. Used by the training-set preview tab to filter
    frontal / side-face samples.
    """
    try:
        from latentsync.utils.av_reader import AVReader
        from latentsync.utils.face_detector import FaceDetector
    except Exception as exc:
        return {"error": f"import failed: {exc}"}

    try:
        reader = AVReader(video_path)
        total_frames = len(reader)
        if total_frames <= 0:
            return {"error": "no frames", "yaw_mean": None, "yaw_max": None, "detect_rate": 0.0}

        if n_frames <= 1:
            indices = [0]
        else:
            indices = [int(i * (total_frames - 1) / (n_frames - 1)) for i in range(n_frames)]

        import torch

        detector = FaceDetector(
            device="cuda" if torch.cuda.is_available() else "cpu",
            allowed_modules=["detection", "landmark_2d_106", "pose"],
        )

        yaws: List[float] = []
        for idx in indices:
            _, frame = reader[idx]
            if hasattr(frame, "asnumpy"):
                frame = frame.asnumpy()
            face, _ = detector.detect(frame)
            if face is not None and detector.last_pose_yaw is not None:
                yaws.append(detector.last_pose_yaw)

        if not yaws:
            return {
                "yaw_mean": None,
                "yaw_max": None,
                "detect_rate": 0.0,
                "face_type": "unknown",
            }

        yaw_mean = sum(yaws) / len(yaws)
        yaw_max = max(yaws)
        detect_rate = len(yaws) / len(indices)
        return {
            "yaw_mean": round(yaw_mean, 1),
            "yaw_max": round(yaw_max, 1),
            "detect_rate": round(detect_rate, 2),
            "face_type": "frontal" if yaw_mean < 15.0 else "side",
        }
    except Exception as exc:
        return {"error": str(exc), "yaw_mean": None, "yaw_max": None, "detect_rate": 0.0}


def _format_preview_info(video_path: str, analysis: Dict[str, Dict[str, Any]]) -> str:
    """Render yaw/face info for the selected training video."""
    if not video_path:
        return ""
    info = analysis.get(video_path, {})
    if "error" in info:
        return f"⚠️ 分析失败: {info['error']}"
    yaw_mean = info.get("yaw_mean")
    yaw_max = info.get("yaw_max")
    detect_rate = info.get("detect_rate")
    face_type = info.get("face_type", "unknown")
    if yaw_mean is None:
        return "⚠️ 未检测到人脸"
    face_type_text = "正脸" if face_type == "frontal" else "侧脸"
    return (
        f"类型: {face_type_text}\n"
        f"平均 yaw: {yaw_mean}°\n"
        f"最大 yaw: {yaw_max}°\n"
        f"人脸检测率: {detect_rate * 100:.0f}%"
    )


def _preview_cache_path(data_dir: str, fileslist: str) -> Optional[Path]:
    """Pick a stable location for the yaw-analysis cache.

    If a fileslist is used, cache next to it; otherwise cache inside data_dir.
    Returns None if neither path is available.
    """
    fileslist = (fileslist or "").strip()
    data_dir = (data_dir or "").strip()
    if fileslist:
        return Path(fileslist).parent / "preview_analysis_cache.json"
    if data_dir:
        return Path(data_dir) / "preview_analysis_cache.json"
    return None


def _load_preview_cache(cache_path: Optional[Path], videos: List[str]) -> Optional[Dict[str, Any]]:
    """Load cached yaw analysis if the video list matches exactly."""
    if cache_path is None or not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("videos") == videos:
            logger.info("Loaded preview analysis cache from %s", cache_path)
            return data.get("analysis", {})
    except Exception as exc:
        logger.warning("Failed to read preview cache %s: %s", cache_path, exc)
    return None


def _save_preview_cache(
    cache_path: Optional[Path],
    videos: List[str],
    analysis: Dict[str, Any],
    threshold: float,
) -> None:
    """Persist yaw analysis so the next load can skip face detection."""
    if cache_path is None:
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "videos": videos,
            "analysis": analysis,
            "threshold": threshold,
            "cached_at": datetime.now().isoformat(timespec="seconds"),
        }
        cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved preview analysis cache to %s", cache_path)
    except Exception as exc:
        logger.warning("Failed to save preview cache %s: %s", cache_path, exc)


def _safe_video_update(value):
    """Coerce arbitrary value to a gr.update that's safe for gr.Video in
    Gradio 5.x.

    Newer Gradio runs gr.Video.postprocess() on every value bound to
    a Video component. If the value is None / "" / [] / a stale dict,
    the postprocess raises 'Expected lists of length 2 or tuples of
    length 2. Received: []'. This wrapper routes any unsafe value to
    gr.update(value=None) which keeps the previous video on screen.
    """
    if not value:
        return gr.update(value=None)
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return gr.update(value=None)
    return gr.update(value=value)


# ---------------------------------------------------------------------------
# Gradio UI assembly
# ---------------------------------------------------------------------------

# System font stack so the UI does not wait for Google Fonts CDN.
_SYSTEM_FONT_CSS = """
* {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial,
                 "Noto Sans", "PingFang SC", "Microsoft YaHei", sans-serif !important;
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="LatentSync Fine-tune Studio",
        theme=gr.themes.Soft(),
        css=_SYSTEM_FONT_CSS,
    ) as demo:
        gr.Markdown(
            """
# 🎛 LatentSync Fine-tune Studio

> **核心目的**：**以微调 UNet 为主**的端到端工作台。覆盖**训练 → 监控 → 验证 → 调参**全流程。
> 也支持 **SyncNet 单独训练**（Tab 1 选 SyncNet preset）。

## 推荐工作流

```
[Tab 1] 配数据集 + 选 preset (Stage 2 / LoRA / QLoRA) → 启动训练
   ↓
[Tab 2] 看 loss 曲线 / validation 视频 / 日志 / sync_conf
   ↓
[Tab 3.5] 单 ckpt 推理 + 质量自检（首次验证）
   ↓
[Tab 3]  base vs fine-tuned 并排对比
   ↓
[Tab 4 / 6] 调 Identity 保护 / 查 Badcase
```

## 6 个 Tab 一览

| Tab | 阶段 | 核心作用 |
|---|---|---|
| **1. 配置 & 启动** | 训练 | 选 preset、调超参、启动 `torchrun` |
| **2. 训练监控** | 训练 | loss 曲线、val 视频、日志、auto refresh 15s |
| **3. 推理对比** | 推理 | base vs fine-tuned 并排跑 |
| **3.5. 验证 (单 ckpt)** | 推理 | **新** 选 1 个 ckpt 跑 + 自动质量自检（嘴糊 / 闪烁 / 人脸检测）|
| **4. Identity 保护** | 推理 | 调 4 层身份保持参数，生成推理 kwargs |
| **5. 数据集质量评估** | 数据 | HyperIQA + SyncNet_conf 分布 |
| **6. Badcase 检查** | 推理 | 量化"嘴糊不糊"等指标 |

> ⚠️ **GPU 提示**：本页会本地拉起 `torchrun`。如果没有 CUDA，会启动失败。
> 推荐在带 GPU 的机器上启动；如果在 CPU 机器上启动，至少能在 **Tab 1** 配置并保存 yaml 供远程训练使用。
            """
        )

        # =========================================================
        # Tab 1: Configure & Launch
        # =========================================================
        with gr.Tab("1️⃣ 配置 & 启动"):
            # ──────────────────────────────────────────────────────────
            # 🚀 一键启动训练:顶部大按钮,自动填默认 + 启动
            # ──────────────────────────────────────────────────────────
            gr.Markdown(
                "> **懒人入口**:点一下这个按钮,自动填默认字段 + 用当前 preset 启训练。"
                "想细调的字段下面表单手动改。"
            )
            with gr.Row():
                one_click_btn = gr.Button(
                    "🚀 一键启动训练 (自动填默认 + 启 torchrun)",
                    variant="primary",
                    scale=4,
                )
            one_click_status = gr.Textbox(
                label="一键启动状态", interactive=False, lines=3,
                value="👆 点上面按钮开始。空字段会用 preset 默认值 / prebuilt 默认数据 / assets demo 视频。",
            )

            with gr.Row():
                preset_dd = gr.Dropdown(
                    choices=list(PRESETS.keys()),
                    value="Stage 2 LoRA (256, 12-15GB)",
                    label="预设 (Preset)",
                    scale=2,
                )
                preset_desc = gr.Textbox(
                    label="预设说明",
                    value=PRESETS["Stage 2 LoRA (256, 12-15GB)"]["description"],
                    interactive=False,
                    scale=3,
                )

            with gr.Row():
                freeze_attn2 = gr.Checkbox(
                    label="LoRA: 冻结 attn2 (audio cross-attn) — 防 sync 退化",
                    value=True,
                    info="仅 LoRA 生效；勾上后 attn2 的 LoRA 参数冻结,牺牲一点灵活性换取 sync_conf 稳定",
                )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📂 数据集")
                    dataset_preset_dd = gr.Dropdown(
                        choices=list(DATASET_PRESETS.keys()),
                        value="assets 演示数据 (3 videos，可完整跑通)",
                        label="数据集预设 (Dataset Preset)",
                    )
                    train_data_dir = gr.Textbox(
                        label="train_data_dir (目录)",
                        placeholder="data/my_high_quality_videos",
                        value="assets",
                    )
                    train_fileslist = gr.Textbox(
                        label="train_fileslist (文件列表，一行一个 mp4)",
                        placeholder="data/my_high_quality_videos/fileslist.txt",
                        value="data/demo_fileslist.txt",
                    )
                    val_video_path = gr.Textbox(
                        label="val_video_path",
                        value="assets/demo1_video.mp4",
                    )
                    val_audio_path = gr.Textbox(
                        label="val_audio_path",
                        value="assets/demo1_audio.wav",
                    )

                    # Quick-pick presets so the user can verify the full
                    # launch path before plugging in their own data.
                    gr.Markdown("#### 📌 常用路径示例（点击填入）")
                    gr.Examples(
                        examples=[
                            ["assets", "data/demo_fileslist.txt"],
                            ["preprocess/high_visual_quality", "preprocess/high_visual_quality/fileslist.txt"],
                            ["data/train", "data/train/fileslist.txt"],
                            ["data/my_avatar", "data/my_avatar/fileslist.txt"],
                            ["data/multilingual", "data/multilingual/fileslist.txt"],
                        ],
                        inputs=[train_data_dir, train_fileslist],
                        label=None,
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

                    dataset_preset_dd.change(
                        fn=apply_dataset_preset,
                        inputs=dataset_preset_dd,
                        outputs=[train_data_dir, train_fileslist, val_video_path, val_audio_path],
                    )

                with gr.Column():
                    gr.Markdown("### 🏗 模型 & 训练")
                    resume_ckpt = gr.Textbox(
                        label="resume_ckpt (base UNet .pt 或 LoRA adapter 目录)",
                        value=PRESETS["Stage 2 (256, 推荐)"]["resume_ckpt"],
                        info="传 .pt 表示从 base/pretrained 开始训；传 LoRA adapter 目录（含 adapter_config.json）表示从该 checkpoint 继续训练。",
                    )
                    batch_size = gr.Slider(1, 64, value=1, step=1, label="batch_size")
                    num_frames = gr.Slider(8, 32, value=16, step=1, label="num_frames")
                    resolution = gr.Radio([256, 512], value=512, label="resolution")
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
                    gr.Markdown("### 🖼 Validation 推理参数")
                    gr.Markdown(
                        "<small>每 save_ckpt_steps 步生成一段验证视频用的推理质量/速度。"
                        "自然度默认 steps=40,guidance=1.5;快速试训可降到 20/1.5。"
                        "推到 HF 后别人用同样的 ckpt + 这俩默认值就能复现。</small>"
                    )
                    val_inference_steps = gr.Slider(
                        5, 80, value=40, step=5, label="validation inference_steps (20=快, 40=自然)"
                    )
                    val_guidance_scale = gr.Slider(
                        1.0, 4.0, value=1.5, step=0.1, label="validation guidance_scale"
                    )
                    val_seed = gr.Number(
                        value=1247, label="validation seed (随机种子)", precision=0
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⚙️ 训练设置")
                    mixed_precision_training = gr.Checkbox(value=True, label="mixed_precision_training (fp16)")
                    enable_gradient_checkpointing = gr.Checkbox(value=True, label="enable_gradient_checkpointing")
                    mask_image_path = gr.Textbox(
                        label="mask_image_path",
                        value="latentsync/utils/mask.png",
                    )
                    save_ckpt_steps = gr.Slider(500, 50000, value=10000, step=500, label="save_ckpt_steps")
                    max_train_steps = gr.Slider(1000, 100000, value=10000, step=1000, label="max_train_steps (cap: 100k ≈ 1.7 days @1.5s/step)")
                    num_workers = gr.Slider(0, 32, value=12, step=1, label="num_workers")
                    lr_scheduler = gr.Dropdown(
                        choices=["constant", "cosine", "cosine_with_restarts", "linear", "polynomial"],
                        value="cosine",
                        label="lr_scheduler",
                    )
                    lr_warmup_steps = gr.Slider(
                        0, 2000, value=200, step=50, label="lr_warmup_steps (推荐 100-300 for cosine)"
                    )
                    train_output_dir = gr.Textbox(
                        label=f"train_output_dir (相对于 {FINETUNE_BASE_DIR.name})",
                        value="unet",
                    )
                    gr.Markdown(
                        f"> 📂 微调中间产物根目录：`{FINETUNE_BASE_DIR}`\n"
                        f"> 可通过环境变量 `LATENTSYNC_FINETUNE_DIR` 修改"
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
                ping_btn = gr.Button("🔍 Ping 后端", scale=1)
                debug_btn = gr.Button("🐛 Debug 输入", scale=1)

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
                    freeze_attn2,
                    val_inference_steps, val_guidance_scale, val_seed,
                    lr_scheduler, lr_warmup_steps,
                    nproc_per_node, master_port, extra_env,
                ],
                outputs=[launch_status, log_path_state],
            )
            stop_btn.click(fn=stop_training, outputs=launch_status)

            # The big top-of-tab "一键启动" button delegates to the
            # standard launch_training flow with auto-filled defaults.
            one_click_btn.click(
                fn=one_click_launch,
                inputs=[
                    preset_dd, train_data_dir, train_fileslist,
                    val_video_path, val_audio_path, resume_ckpt,
                    batch_size, num_frames, resolution, learning_rate,
                    use_motion_module, pixel_space_supervise, use_syncnet,
                    sync_loss_weight, perceptual_loss_weight, recon_loss_weight, trepa_loss_weight,
                    mixed_precision_training, enable_gradient_checkpointing, mask_image_path,
                    save_ckpt_steps, max_train_steps, num_workers, train_output_dir,
                    freeze_attn2, val_inference_steps, val_guidance_scale, val_seed,
                    lr_scheduler, lr_warmup_steps,
                    nproc_per_node, master_port, extra_env,
                ],
                outputs=one_click_status,
            )

            # ---- 预制数据集 (HF Hub 自动下载 + curate) ----
            with gr.Accordion("📚 预制数据集 (HF Hub 自动下 + curate)", open=False):
                gr.Markdown(
                    "**最省事的入口** — 从 `tools/prebuilt_datasets.yaml` 选一个,"
                    "一键下载 + face 检测 + 按 yaw/motion 分桶 + 写 fileslist.txt,"
                    "直接拿来训练。无需自己找数据源。"
                )
                with gr.Row():
                    prebuilt_dd = gr.Dropdown(
                        choices=_prebuilt_choices(),
                        label="预制数据集 (HF Hub)",
                        value=None,
                        scale=2,
                    )
                    prebuilt_target = gr.Textbox(
                        label=f"输出目录 (默认 {FINETUNE_BASE_DIR}/init_finetune)",
                        placeholder=str(FINETUNE_BASE_DIR / "init_finetune"),
                        value=str(FINETUNE_BASE_DIR / "init_finetune"),
                        scale=2,
                    )
                prebuilt_hf_token = gr.Textbox(
                    label="HF Token (可选,用于 gated 数据集,留空走 HF_TOKEN 环境变量)",
                    placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
                    type="password",
                )
                prebuilt_btn = gr.Button("⬇ 下载 + Curate", variant="primary")
                prebuilt_log = gr.Textbox(label="输出", lines=18, interactive=False)
                prebuilt_btn.click(
                    fn=_run_init_prebuilt,
                    inputs=[prebuilt_dd, prebuilt_target, prebuilt_hf_token],
                    outputs=[prebuilt_log, train_data_dir, train_fileslist],
                )

            # ---- 数据集一键准备 (download + curate) ----
            with gr.Accordion("📥 数据集一键准备 (download_curated_finetune_set)", open=False):
                gr.Markdown(
                    "端到端跑 `tools/download_curated_finetune_set.py`:给一批 URL 或本地视频,"
                    "自动按 yaw/motion 分桶,产出可直接填进上面表单的 fileslist。"
                )
                with gr.Row():
                    curate_urls = gr.Textbox(
                        label="URL 列表文件 (可空,用 --source-dir)",
                        placeholder="tools/finetune_starter_urls.example.txt",
                        scale=2,
                    )
                    curate_source_dir = gr.Textbox(
                        label="本地源目录 (可空,用 --urls)",
                        placeholder="/data/my_raw_videos",
                        scale=2,
                    )
                with gr.Row():
                    curate_output_dir = gr.Textbox(
                        label="curated 输出目录",
                        value=str(FINETUNE_BASE_DIR / "finetune_samples_v1"),
                        scale=3,
                    )
                    curate_scale = gr.Dropdown(
                        choices=["small", "medium", "large"],
                        value="small",
                        label="scale (small=200, medium=1000, large=5000)",
                        scale=2,
                    )
                with gr.Row():
                    curate_btn = gr.Button("📥 跑 download_curated_finetune_set", variant="primary")
                curate_log = gr.Textbox(label="输出", lines=18, interactive=False)
                curate_btn.click(
                    fn=_run_curate_finetune,
                    inputs=[curate_urls, curate_source_dir, curate_output_dir, curate_scale],
                    outputs=curate_log,
                )

            debug_status = gr.Textbox(label="诊断信息", lines=8, interactive=False)
            ping_btn.click(fn=ping_backend, outputs=debug_status)
            debug_btn.click(
                fn=debug_all_inputs,
                inputs=[
                    preset_dd, train_data_dir, train_fileslist, val_video_path, val_audio_path,
                    resume_ckpt, batch_size, num_frames, resolution, learning_rate,
                    use_motion_module, pixel_space_supervise, use_syncnet,
                    sync_loss_weight, perceptual_loss_weight, recon_loss_weight, trepa_loss_weight,
                    mixed_precision_training, enable_gradient_checkpointing, mask_image_path,
                    save_ckpt_steps, max_train_steps, num_workers, train_output_dir,
                    nproc_per_node, master_port, extra_env,
                ],
                outputs=debug_status,
            )

            with gr.Row():
                training_log_box = gr.Textbox(
                    label="训练日志 (实时，尾部 80 行)",
                    lines=20,
                    interactive=False,
                    value="(训练日志会在这里显示)",
                )

            refresh_log_btn = gr.Button("🔄 手动刷新训练日志", variant="secondary")
            refresh_log_btn.click(
                fn=refresh_training_log,
                outputs=[training_log_box, launch_status],
            )
            log_timer = gr.Timer(value=3)
            log_timer.tick(
                fn=refresh_training_log,
                outputs=[training_log_box, launch_status],
            )

            # preset → fill defaults
            preset_dd.change(
                fn=on_preset_change,
                inputs=preset_dd,
                outputs=[
                    batch_size, num_frames, resolution, learning_rate,
                    use_motion_module, pixel_space_supervise, use_syncnet,
                    sync_loss_weight, perceptual_loss_weight, recon_loss_weight, trepa_loss_weight,
                    mixed_precision_training, enable_gradient_checkpointing, mask_image_path,
                    resume_ckpt,
                    save_ckpt_steps, max_train_steps, lr_scheduler, lr_warmup_steps,
                    preset_desc, freeze_attn2,
                ],
            )

        # =========================================================
        # Tab 2: Monitor
        # =========================================================
        with gr.Tab("2️⃣ 训练监控"):
            with gr.Row():
                monitor_output_dir = gr.Textbox(
                    label=f"train_output_dir (相对于 {FINETUNE_BASE_DIR.name})",
                    value="unet",
                )
                refresh_runs_btn = gr.Button("🔄 刷新 run 列表")

            run_dd = gr.Dropdown(label="run 目录", choices=[], value=None, allow_custom_value=True)
            refresh_runs_btn.click(
                fn=refresh_runs,
                inputs=monitor_output_dir,
                outputs=run_dd,
            )

            trainer_status = gr.Textbox(label="Trainer 状态", interactive=False)
            log_box = gr.Textbox(label="最新日志 (尾部 80 行)", lines=20, interactive=False)
            ckpt_dd = gr.Dropdown(label="Checkpoint", choices=[], value=None)
            ckpt_info_box = gr.Textbox(label="Checkpoint 信息", lines=10, interactive=False)

            with gr.Row():
                with gr.Column():
                    loss_chart_img = gr.Image(label="Loss 曲线 (lr + total + recon + lpips + sync)", type="filepath")
                    sync_conf_img = gr.Image(label="Sync_conf 曲线 (finetune 核心信号)", type="filepath")
                with gr.Column():
                    val_video_dd = gr.Dropdown(label="Validation 视频", choices=[])
                    val_video_player = gr.Video(label="预览", interactive=False)

            def _on_run_change(run_path):
                chart = parse_loss_chart(run_path)
                sync_chart = parse_sync_conf_chart(run_path)
                vids = list_validation_videos(run_path)
                ckpts = list_checkpoints_in_run(run_path)
                ckpt_update = gr.update(choices=ckpts, value=ckpts[-1] if ckpts else None)
                if ckpts:
                    ckpt_path = Path(ckpts[-1])
                    if not ckpt_path.is_absolute():
                        ckpt_path = REPO_ROOT / ckpt_path
                    ck_info = read_loss_from_checkpoint(str(ckpt_path))
                else:
                    ck_info = "(no checkpoint yet)"
                return (
                    chart,
                    sync_chart,
                    gr.update(choices=vids, value=vids[0] if vids else None),
                    ckpt_update,
                    ck_info,
                )

            def _on_ckpt_change(run_path, ckpt_path):
                if not ckpt_path:
                    return "(no checkpoint selected)"
                p = Path(ckpt_path)
                if not p.is_absolute():
                    p = REPO_ROOT / p
                return read_loss_from_checkpoint(str(p))

            run_dd.change(
                fn=_on_run_change,
                inputs=run_dd,
                outputs=[loss_chart_img, sync_conf_img, val_video_dd, ckpt_dd, ckpt_info_box],
            )
            ckpt_dd.change(
                fn=_on_ckpt_change,
                inputs=[run_dd, ckpt_dd],
                outputs=ckpt_info_box,
            )
            val_video_dd.change(
                fn=_safe_video_update,
                inputs=val_video_dd,
                outputs=val_video_player,
            )

            monitor_btn = gr.Button("🔄 手动刷新", variant="primary")
            run_dir_hidden = gr.Textbox(visible=False)
            with gr.Row():
                progress_bar = gr.Slider(
                    0, 100, value=0, step=0.1, interactive=False,
                    label="📈 训练进度 (step / max_step, %)",
                )
                progress_text = gr.Textbox(
                    label="⏱ 耗时 / 速度 / ETA",
                    interactive=False,
                    scale=2,
                )
            monitor_btn.click(
                fn=monitor_refresh,
                inputs=[monitor_output_dir, run_dd, log_path_state],
                outputs=[
                    run_dd, run_dir_hidden, loss_chart_img, sync_conf_img,
                    val_video_dd, ckpt_dd, log_box, ckpt_info_box, trainer_status,
                    progress_bar, progress_text,
                ],
            )

            timer = gr.Timer(value=15)
            timer.tick(
                fn=monitor_refresh,
                inputs=[monitor_output_dir, run_dd, log_path_state],
                outputs=[
                    run_dd, run_dir_hidden, loss_chart_img, sync_conf_img,
                    val_video_dd, ckpt_dd, log_box, ckpt_info_box, trainer_status,
                    progress_bar, progress_text,
                ],
            )

            # ---- 合并 LoRA 子表(训练完自然接这一步) ----
            with gr.Accordion("🔀 合并 LoRA adapter (merge_lora)", open=False):
                gr.Markdown(
                    "把训练好的 LoRA adapter (~10MB) 折回 base UNet,产出可独立部署的 "
                    "`latentsync_unet.pt`。可选同时 push 到 HuggingFace Hub。"
                )
                with gr.Row():
                    merge_base_ckpt = gr.Textbox(
                        label="base UNet ckpt",
                        value="checkpoints/latentsync_unet.pt",
                        scale=2,
                    )
                    merge_adapter_dir = gr.Textbox(
                        label="adapter 目录 (训练产物, 含 adapter_config.json)",
                        placeholder="debug/unet_lora/train_lora-2025.../checkpoints/checkpoint-5000",
                        scale=3,
                    )
                with gr.Row():
                    merge_out_ckpt = gr.Textbox(
                        label="合并输出路径",
                        value=str(FINETUNE_BASE_DIR / "unet" / "merged.pt"),
                        scale=3,
                    )
                    merge_push_repo = gr.Textbox(
                        label="(可选) HF Hub repo_id,留空不 push",
                        placeholder="username/latentsync-lora-finetune-v1",
                        scale=3,
                    )
                merge_btn = gr.Button("🔀 合并 LoRA → merged.pt (可同时 push)", variant="primary")
                merge_log = gr.Textbox(label="merge 输出", lines=10, interactive=False)
                merge_btn.click(
                    fn=_run_merge_lora,
                    inputs=[merge_base_ckpt, merge_adapter_dir, merge_out_ckpt, merge_push_repo],
                    outputs=merge_log,
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
                    cmp_include_lora = gr.Checkbox(value=True, label="包含 LoRA adapter 目录")
                    cmp_base = gr.Dropdown(
                        choices=list_checkpoints(include_lora=True),
                        label="Base checkpoint",
                        allow_custom_value=True,
                    )
                    cmp_ft = gr.Dropdown(
                        choices=list_checkpoints(include_lora=True),
                        label="Fine-tuned checkpoint",
                        allow_custom_value=True,
                    )
                    cmp_resolution = gr.Radio([256, 512], value=512, label="resolution")

            with gr.Row():
                cmp_steps = gr.Slider(10, 50, value=20, step=1, label="inference_steps")
                cmp_guidance = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="guidance_scale")
                cmp_seed = gr.Number(value=1247, label="seed", precision=0)
                cmp_baseline = gr.Checkbox(value=False, label="基线模式（禁用所有质量优化）")

            with gr.Row():
                cmp_btn = gr.Button("🎬 生成对比", variant="primary", scale=3)
                cmp_cancel_btn = gr.Button("⏹ 取消当前推理", variant="stop", scale=1)
            cmp_status = gr.Textbox(label="状态", interactive=False, visible=False)

            with gr.Row():
                cmp_out_base = gr.Video(label="Base 输出")
                cmp_out_ft = gr.Video(label="Fine-tuned 输出")

            cmp_btn.click(
                fn=run_compare,
                inputs=[
                    cmp_video, cmp_audio, cmp_base, cmp_ft,
                    cmp_steps, cmp_guidance, cmp_seed, cmp_resolution,
                    cmp_baseline,
                ],
                outputs=[cmp_out_base, cmp_out_ft],
            )
            cmp_include_lora.change(
                fn=lambda incl: gr.update(choices=list_checkpoints(include_lora=incl)),
                inputs=cmp_include_lora,
                outputs=[cmp_base, cmp_ft],
            )
            cmp_cancel_btn.click(fn=stop_inference, outputs=cmp_status)

        # =========================================================
        # Tab 3.5: Validation - run inference with a single ckpt
        # =========================================================
        with gr.Tab("🧪 验证 (单 ckpt 推理)"):
            gr.Markdown(
                """
选一个 checkpoint（base / fine-tuned / LoRA-merged），上传视频和音频，
跑一次推理并自动做 **质量自检**（嘴糊比例 / 闪烁 / 人脸检测率）。

> 比 Tab 3 简单：只跑一个 ckpt，更快。  
> 跑完后输出 mp4 + 质量报告。
                """
            )
            with gr.Row():
                with gr.Column():
                    val_video = gr.Video(label="Input Video", scale=2)
                    val_audio = gr.Audio(label="Input Audio", type="filepath", scale=2)
                with gr.Column():
                    val_include_lora = gr.Checkbox(value=True, label="包含 LoRA adapter 目录")
                    val_ckpt = gr.Dropdown(
                        choices=list_checkpoints(include_lora=True),
                        label="Checkpoint（base / fine-tuned / LoRA-merged）",
                        value="checkpoints/latentsync_unet.pt" if (REPO_ROOT / "checkpoints/latentsync_unet.pt").exists() else None,
                        allow_custom_value=True,
                    )
                    val_config = gr.Dropdown(
                        choices=[
                            "configs/unet/stage2.yaml",
                            "configs/unet/stage2_512.yaml",
                            "configs/unet/stage2_efficient.yaml",
                            "configs/unet/stage2_lora.yaml",
                        ],
                        value="configs/unet/stage2.yaml",
                        label="UNet config（必须和 ckpt 匹配）",
                    )
                    val_resolution = gr.Radio([256, 512], value=512, label="resolution")

            with gr.Row():
                val_steps = gr.Slider(10, 50, value=20, step=1, label="inference_steps")
                val_guidance = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="guidance_scale")
                val_seed = gr.Number(value=1247, label="seed", precision=0)
                val_deepcache = gr.Checkbox(value=True, label="enable_deepcache (快 2x)")
                val_skip_qc = gr.Checkbox(value=True, label="跳过质量自检（更快）")
                val_baseline = gr.Checkbox(value=False, label="基线模式（禁用所有质量优化）")

            with gr.Row():
                val_btn = gr.Button("🚀 推理 + 质量自检", variant="primary", scale=3)
                val_cancel_btn = gr.Button("⏹ 取消当前推理", variant="stop", scale=1)

            val_compat = gr.Textbox(label="ckpt 兼容性检查", lines=4, interactive=False)
            val_output = gr.Video(label="生成结果", interactive=False)
            val_report = gr.Textbox(label="质量报告", lines=18, interactive=False)
            val_saved = gr.Textbox(label="保存路径", interactive=False)

            val_btn.click(
                fn=run_validation,
                inputs=[
                    val_video, val_audio, val_ckpt, val_config,
                    val_steps, val_guidance, val_seed, val_resolution,
                    val_deepcache, val_skip_qc, val_baseline,
                ],
                outputs=[val_output, val_compat, val_report, val_saved, val_btn],
            )
            val_include_lora.change(
                fn=lambda incl: gr.update(choices=list_checkpoints(include_lora=incl)),
                inputs=val_include_lora,
                outputs=val_ckpt,
            )
            val_cancel_btn.click(fn=stop_inference, outputs=[val_report, val_btn])

            val_timer = gr.Timer(value=1, active=True)
            val_timer.tick(
                fn=_poll_inference_state,
                inputs=[val_skip_qc],
                outputs=[val_output, val_compat, val_report, val_saved, val_btn],
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
                        value=0.0,
                        step=0.05,
                        label="mouth_detail_strength (L4 detail restore)",
                        info="越大越贴原图皮肤纹理（痣、皱纹）。>0.85 会盖掉生成的嘴型",
                    )
                    color_match_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
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

### Badcase → 推荐操作 速查

| 现象 | 数值信号 | 首选排查方向 | 备选:微调 |
|---|---|---|---|
| **侧脸同步弱 / 嘴唇不动** | `唇音同步 < 5` 且 yaw 大的帧 | Tab 4 把 `adaptive_quality_fallback` 打开 / 放宽 `yaw_skip_threshold` (30°→40°) | Stage 2 LoRA,feed ≥ 50 条 yaw 15-30° 样本,`freeze_attn2=True` |
| **嘴糊** (整嘴一片糊) | `嘴糊比例 > 40%` | 检查 `mask_image_path` 是否被改成 mask2/3 (baseline 是 `mask.png`) | Stage 2 LoRA rank=32 |
| **嘴唇外也糊** (paste-back 边界外溢) | 闪烁评分高 + 边缘像素突跳 | Tab 4 `dynamic_mask_mode` 切 `aggressive`,`paste_back_blur_sigma` 降到 5.0 | Stage 2 LoRA + 混合 clip / distance samples |
| **人脸快速移动时嘴糊** | 闪烁评分 > 12 + sync 抖动 | 调 `mouth_temporal_stabilization_strength`↑ 到 0.25 / `mouth_audio_motion_min_scale`↑ 到 0.9 | Stage 2 LoRA + 含 motion-blur 样本 |
| **身份漂移 (像别人了)** | `身份保持 < 0.7` | Tab 4 `ref_strategy` 改 `fixed_first_frame`,`color_match_strength`↑ 到 0.75 | LoRA 不要碰 attn1 self-attn (默认就只 wrap to_q/k/v/out.0) |
| **训练后 sync_conf 退化** | 训练前后 sync 下降 > 1 | (训练配置)勾 `freeze_attn2` 重训 | — |
| **显存 OOM** | torch OOM 日志 | 切 Stage 2 QLoRA | — |
| **生成的嘴没动但能听到声音** | `唇音同步 ≈ 1` (没生成) | 看训练日志 `[FaceMatch]` 哪个 filter 跳了 — 通常是 `yaw_skip` 或 `face_jump` | Stage 2 LoRA + 多角度训练数据 |

**数据先行原则**:finetune 只能缓解,不能根治。`Tab 5` 先跑一遍数据集质量评估,HyperIQA 分布去掉 < 40 的样本后再训,事半功倍。
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
                    bc_yaw = gr.Number(
                        label="平均 yaw (°; 0=正面, ≥15°=侧脸, ≥25°=重度侧脸)",
                        precision=1,
                    )
                with gr.Column():
                    bc_report = gr.Textbox(label="诊断报告", lines=20)
                    bc_recommendation = gr.Textbox(
                        label="🎯 finetune preset 推荐 (基于上面 5 个数字自动判定)",
                        lines=4,
                        interactive=False,
                        value="跑完上方 🔍 检测后,这里会自动出推荐 preset。",
                    )

            bc_check_btn.click(
                fn=run_badcase_checklist,
                inputs=[bc_video, bc_reference],
                outputs=[bc_blurry, bc_flicker, bc_sync, bc_identity, bc_yaw, bc_report],
            )

            # Whenever any of the 5 metric numbers change, refresh the
            # preset recommendation. Tab 6 re-run fills them all in one
            # .click event, so the user sees the recommendation update
            # immediately after the numbers settle.
            for bc_metric in (bc_blurry, bc_flicker, bc_sync, bc_identity, bc_yaw):
                bc_metric.change(
                    fn=_recommend_finetune_preset,
                    inputs=[bc_blurry, bc_flicker, bc_sync, bc_identity, bc_yaw],
                    outputs=bc_recommendation,
                )

            with gr.Accordion("🎬 短剧专项诊断 (多说话人 / 频繁切场景)", open=False):
                gr.Markdown(
                    "对短剧类输入做额外的场景/说话人数量估计。"
                    "基于 HSV histogram(场景切点)+ face bbox 聚类(说话人数),"
                    "无新依赖;若有 face_recognition,会进一步用 embedding 精确聚类。"
                )
                drama_btn = gr.Button("🎬 跑短剧场景诊断", variant="primary")
                drama_report = gr.Textbox(label="短剧诊断", lines=10, interactive=False)
                drama_btn.click(
                    fn=_diagnose_short_drama,
                    inputs=[bc_video],
                    outputs=drama_report,
                )

        # =========================================================
        # Tab 7: Training-set preview
        # =========================================================
        with gr.Tab("📁 训练集预览"):
            gr.Markdown(
                """
浏览并播放训练集中的原始视频样本。
支持从 `train_fileslist` 读取（优先），或扫描 `train_data_dir` 下的 `.mp4` 文件。
可先快速加载列表，再按需分析人脸 yaw 并筛选正脸 / 侧脸。
                """
            )
            with gr.Row():
                preview_data_dir = gr.Textbox(
                    label="train_data_dir",
                    value="",
                    scale=2,
                )
                preview_fileslist = gr.Textbox(
                    label="train_fileslist（优先使用）",
                    value="",
                    scale=2,
                )
                preview_load_btn = gr.Button("🔄 加载视频列表", variant="primary", scale=1)
                preview_analyze_btn = gr.Button("🔍 分析 yaw", variant="secondary", scale=1)

            with gr.Row():
                preview_filter = gr.Dropdown(
                    label="筛选",
                    choices=["全部", "正脸", "侧脸"],
                    value="全部",
                    scale=1,
                )
                preview_threshold = gr.Slider(
                    label="yaw 阈值 (°)，≥ 为侧脸",
                    minimum=0,
                    maximum=45,
                    value=15,
                    step=1,
                    scale=2,
                )
                preview_count = gr.Textbox(
                    label="统计",
                    value="",
                    interactive=False,
                    scale=2,
                )

            with gr.Row():
                preview_video_dd = gr.Dropdown(
                    label="选择视频",
                    choices=[],
                    value=None,
                    scale=3,
                )
                preview_yaw_info = gr.Textbox(
                    label="人脸 / yaw 信息",
                    value="",
                    lines=4,
                    interactive=False,
                    scale=1,
                )

            preview_video_player = gr.Video(label="预览", interactive=False)
            preview_videos_state = gr.State([])
            preview_analysis_state = gr.State({})

            def _load_preview_videos(data_dir: str, fileslist: str, threshold: float):
                videos, status = _list_training_videos(data_dir, fileslist)
                cache_path = _preview_cache_path(data_dir, fileslist)
                cached_analysis = _load_preview_cache(cache_path, videos)
                if cached_analysis is not None:
                    status += " | 已使用缓存"
                    return (
                        gr.update(choices=videos, value=videos[0] if videos else None),
                        status,
                        gr.update(value=None),
                        videos,
                        cached_analysis,
                        _format_preview_info(videos[0] if videos else "", cached_analysis),
                    )
                return (
                    gr.update(choices=videos, value=videos[0] if videos else None),
                    status,
                    gr.update(value=None),
                    videos,
                    {},
                    "",
                )

            def _analyze_preview_videos(
                videos: List[str],
                threshold: float,
                data_dir: str,
                fileslist: str,
            ):
                if not videos:
                    return (
                        gr.update(choices=[], value=None),
                        "⚠️ 没有视频可分析",
                        {},
                        "",
                    )
                analysis: Dict[str, Any] = {}
                frontal = 0
                side = 0
                unknown = 0
                for v in videos:
                    info = _analyze_training_video_yaw(v, n_frames=5)
                    analysis[v] = info
                    ft = info.get("face_type", "unknown")
                    if ft == "frontal":
                        frontal += 1
                    elif ft == "side":
                        side += 1
                    else:
                        unknown += 1
                status = f"已分析 {len(videos)} 个 | 正脸 {frontal} | 侧脸 {side}"
                if unknown:
                    status += f" | 未检测 {unknown}"
                _save_preview_cache(
                    _preview_cache_path(data_dir, fileslist),
                    videos,
                    analysis,
                    threshold,
                )
                return (
                    gr.update(choices=videos, value=videos[0] if videos else None),
                    status,
                    analysis,
                    _format_preview_info(videos[0] if videos else "", analysis),
                )

            def _apply_preview_filter(
                filter_type: str,
                threshold: float,
                analysis: Dict[str, Any],
                videos: List[str],
            ):
                if filter_type == "全部":
                    return (
                        gr.update(choices=videos, value=videos[0] if videos else None),
                        f"共 {len(videos)} 个视频",
                    )
                if not analysis:
                    return (
                        gr.update(choices=[], value=None),
                        "⚠️ 请先点击 🔍 分析 yaw",
                    )
                filtered: List[str] = []
                for path, info in analysis.items():
                    yaw_mean = info.get("yaw_mean")
                    is_side = yaw_mean is not None and yaw_mean >= threshold
                    if filter_type == "侧脸" and is_side:
                        filtered.append(path)
                    elif filter_type == "正脸" and not is_side and yaw_mean is not None:
                        filtered.append(path)
                return (
                    gr.update(choices=filtered, value=filtered[0] if filtered else None),
                    f"筛选后: {len(filtered)} 个视频",
                )

            def _on_preview_video_change(video_path: str, analysis: Dict[str, Any]):
                return _format_preview_info(video_path, analysis)

            preview_load_btn.click(
                fn=_load_preview_videos,
                inputs=[preview_data_dir, preview_fileslist, preview_threshold],
                outputs=[
                    preview_video_dd, preview_count, preview_video_player,
                    preview_videos_state, preview_analysis_state, preview_yaw_info,
                ],
            )
            preview_analyze_btn.click(
                fn=_analyze_preview_videos,
                inputs=[preview_videos_state, preview_threshold, preview_data_dir, preview_fileslist],
                outputs=[
                    preview_video_dd, preview_count,
                    preview_analysis_state, preview_yaw_info,
                ],
            )
            preview_filter.change(
                fn=_apply_preview_filter,
                inputs=[preview_filter, preview_threshold, preview_analysis_state, preview_videos_state],
                outputs=[preview_video_dd, preview_count],
            )
            preview_video_dd.change(
                fn=_safe_video_update,
                inputs=preview_video_dd,
                outputs=preview_video_player,
            )
            preview_video_dd.change(
                fn=_on_preview_video_change,
                inputs=[preview_video_dd, preview_analysis_state],
                outputs=preview_yaw_info,
            )

            # On page (re)load: repopulate trainer status + run dropdown
            # from the in-process _TRAINER singleton. This survives browser
            # refreshes — the Python process keeps the training subprocess
            # alive even if the user's tab disconnects, so we re-detect here.
            demo.load(
                fn=_on_page_load,
                inputs=[monitor_output_dir],
                outputs=[
                    trainer_status, launch_status, log_path_state, run_dd, monitor_btn,
                    run_dir_hidden, loss_chart_img, sync_conf_img,
                    val_video_dd, ckpt_dd, log_box, ckpt_info_box,
                    progress_bar, progress_text,
                ],
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
    return "random", 16, "standard", 7.0, 0.0, 0.0


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

# Thresholds for the 🎯 vs 🧩 recommendation. Tweak if the heuristics misfire.
_RECO_THRESHOLDS = {
    "identity_critical": 0.70,   # < 0.70 → face geometry distorted
    "identity_soft":    0.85,   # < 0.85 → suspicious, helps the multiplier
    "sync_critical":     5.0,    # < 5.0 → audio-visual sync weak
    "blurry_critical":   0.40,   # > 0.40 → 40%+ frames mouth-blurred
    "flicker_critical":  12.0,   # > 12 → temporal flicker severe
}


def _recommend_finetune_preset(
    blurry: Optional[float], flicker: Optional[float],
    sync: Optional[float], identity: Optional[float],
    avg_yaw: Optional[float] = None,
) -> str:
    """Return a multi-line recommendation string given the badcase metrics.

    Rules (in order):
      1. identity < 0.70                         → 🧩 Structural Fix
      2. content issue AND identity 0.70-0.85    → 🧩 Structural Fix (cover both)
      3. avg_yaw ≥ 18° AND content issue         → 💋 Side-Face Lip Quality
         (overrides 🎯 Content Fix for heavy side-face content bugs)
      4. content issue AND identity >= 0.85     → 🎯 Content Fix
      5. flicker > 12 alone (identity OK)       → 🎯 Content Fix
      6. identity 0.70-0.85 alone (no content)   → 🧩 Structural Fix (mild drift)
      6. all metrics OK                           → ⚪ no finetune needed
    """
    s = _RECO_THRESHOLDS
    parts = []
    if identity is not None:
        parts.append(f"id={identity:.3f}")
    if blurry is not None:
        parts.append(f"blurry={blurry*100:.0f}%")
    if flicker is not None:
        parts.append(f"flicker={flicker:.1f}")
    if sync is not None:
        parts.append(f"sync={sync:.2f}")
    metric_summary = ", ".join(parts) if parts else "(no metrics yet)"

    if identity is not None and identity < s["identity_critical"]:
        return (
            "🧩 Structural Fix (LoRA + conv, 18-22GB)\n"
            f"   {metric_summary}\n"
            f"   身份保持 = {identity:.3f} < {s['identity_critical']:.2f} — "
            "脸几何/五官位置异常,须 wrap 卷积层才能修。"
        )
    has_content_issue = (
        (sync is not None and sync < s["sync_critical"])
        or (blurry is not None and blurry > s["blurry_critical"])
        or (flicker is not None and flicker > s["flicker_critical"])
    )
    structural_suspect = (
        identity is not None
        and s["identity_critical"] <= identity < s["identity_soft"]
    )

    # Rule 2 (BEFORE structural_suspect): side-face heavy + content issue
    # → 💋. This takes priority over the 0.70-0.85 identity range because
    # for side-face videos the shape is preserved but the lips are bad;
    # identity 0.82 on a side-face clip does NOT mean structural issue.
    if (
        avg_yaw is not None and avg_yaw >= 18.0
        and identity is not None and identity >= 0.70
        and has_content_issue
    ):
        return (
            "💋 Side-Face Lip Quality (LoRA+conv, 18-22GB)\n"
            f"   {metric_summary}\n"
            f"   平均 yaw = {avg_yaw:.1f}° ≥ 18° + 内容指标越界 → "
            "侧脸唇形专项(rank=48, sync=0.18, perceptual=0.25)。"
            "\n   数据:📚 预制 celebv_hq_side (side_face 桶 ≥ 50%)。"
        )

    if has_content_issue:
        if structural_suspect:
            return (
                "🧩 Structural Fix (LoRA + conv, 18-22GB) — recommended (cover both)\n"
                f"   {metric_summary}\n"
                f"   内容型问题 + 身份保持 = {identity:.3f} 区间异常 → "
                "先攻结构,顺带 cover 内容。"
            )
        return (
            "🎯 Badcase Fix (LoRA, 12-15GB)\n"
            f"   {metric_summary}\n"
            f"   脸型 OK (id={identity:.3f}),内容指标越界 → att-only LoRA 就够。"
        )

    # has_content_issue is False from here. Now check identity drift + side-face.
    if avg_yaw is not None and avg_yaw >= 18.0:
        return (
            "⚪ 当前侧脸场景内容指标正常,不需要 finetune。"
            f"\n   {metric_summary}\n"
            f"   平均 yaw = {avg_yaw:.1f}° ≥ 18° — 侧脸场景。"
            "如需主动加强侧脸唇形质量,可走 💋 Side-Face Lip Quality。"
        )

    if identity is not None and identity < s["identity_soft"]:
        return (
            "🧩 Structural Fix (LoRA + conv, 18-22GB)\n"
            f"   {metric_summary}\n"
            f"   内容型指标正常,但身份保持 = {identity:.3f} 偏低 → 脸轮廓漂,加 conv wrap。"
        )

    return (
        "⚪ 不需要 finetune / 用 Stage 2 LoRA baseline\n"
        f"   {metric_summary}\n"
        "   五个指标都在合理范围内。生成质量可用,需要换风格/换脸再调 preset。"
    )


def _diagnose_short_drama(
    video_path: str,
) -> str:
    """Quick offline heuristic for short-drama scene/speaker counts.

    Uses only cv2 + the existing face_detector (no new deps). Two
    signals:

      1. Shot count via HSV-histogram Bhattacharyya distance between
         sampled frames (>0.4 ⇒ cut).
      2. Speaker count via clustering of detected face-bbox centroids
         across frames (KMeans K in [1, 5]). Crude but works without
         face_recognition.

    Pure-python; runs in a few seconds on CPU.
    """
    import numpy as np
    import cv2
    if not video_path:
        return "❌ 请先上传视频"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"❌ 视频无法打开: {video_path}"
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total < 30:
        cap.release()
        return f"❌ 视频太短 ({total} 帧)"

    # Sample frames evenly
    sample_n = min(64, max(20, total // 50))
    indices = np.linspace(0, total - 1, sample_n).astype(int)

    try:
        from latentsync.utils.face_detector import FaceDetector
        detector = FaceDetector(device="cpu")
    except Exception as exc:
        cap.release()
        return f"❌ face_detector 加载失败: {exc}"

    frames = []
    bboxes = []  # list of [cx, cy, w, h]
    yaws = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frames.append(frame)
        bbox, _ = detector(frame)
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            bboxes.append([cx / frame.shape[1], cy / frame.shape[0],
                           (x2 - x1) / frame.shape[1], (y2 - y1) / frame.shape[0]])
            if detector.last_pose_yaw is not None:
                yaws.append(float(detector.last_pose_yaw))
    cap.release()

    # ---- shot count ----
    shots = 1
    for i in range(1, len(frames)):
        a = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2HSV)
        b = cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)
        ha = cv2.calcHist([a], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hb = cv2.calcHist([b], [0, 1], None, [8, 8], [0, 180, 0, 256])
        cv2.normalize(ha, ha); cv2.normalize(hb, hb)
        if cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA) > 0.4:
            shots += 1

    # ---- speaker count via greedy centroid clustering ----
    speaker_count = 1
    centroids: list = []
    for bb in bboxes:
        cx, cy, w, h = bb
        placed = False
        for c in centroids:
            if (abs(cx - c[0]) < 0.18 and abs(cy - c[1]) < 0.18
                    and abs(w - c[2]) < 0.15 and abs(h - c[3]) < 0.15):
                placed = True
                break
        if not placed:
            centroids.append([cx, cy, w, h])
            speaker_count += 1
        if speaker_count >= 5:
            break

    # ---- summary text ----
    avg_yaw = float(np.mean(np.abs(yaws))) if yaws else 0.0
    duration_s = total / fps
    print_lines = [
        f"🎬 短剧场景诊断",
        f"   时长:           {duration_s:.1f}s  ({total} 帧 @ {fps:.1f}fps)",
        f"   估计场景数:     {shots}",
        f"   估计说话人数:   {speaker_count}",
        f"   平均 yaw:       {avg_yaw:.1f}°",
        f"   检出 face 帧:   {len(bboxes)} / {len(frames)}",
        "",
    ]
    if speaker_count >= 2 or shots >= 3:
        print_lines.extend([
            "📋 推荐路径:",
            "   数据:python tools/preprocess_short_drama.py --input <video> ...",
            "   训练:Tab 1 preset 选 '🎬 Short Drama (LoRA+conv, 多说话人, 18-22GB)'",
            "   数据集:先 preprocess_short_drama 切场景,再 curate_finetune_samples 分桶",
        ])
    elif speaker_count == 1 and shots <= 2:
        print_lines.append("📋 这条看起来是单场景单人,常规 🎯 Badcase Fix 就够。")
    return "\n".join(print_lines)


def run_badcase_checklist(
    video_path: str, reference_video_path: Optional[str]
) -> Tuple[float, float, float, float, float, str]:
    """Run all 5 badcase checks on a single generated video.

    Returns (blurry_ratio, flicker_score, sync_conf, identity_sim,
    avg_yaw, report). avg_yaw is the mean |yaw| in degrees across
    detected-face frames; used by _recommend_finetune_preset to spot
    side-face badcases.
    """
    if not video_path:
        return 0.0, 0.0, 0.0, 0.0, 0.0, "❌ 请先上传视频"

    try:
        from decord import VideoReader
        import numpy as np
        import cv2
        vr = VideoReader(video_path)
        frames = [f.asnumpy() for f in vr]
        if len(frames) < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0, "❌ 视频帧数 < 2"

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

        # ---- 5. Average yaw (sampled face detection) ----
        # Sample up to 16 frames evenly; run face_detector; record |yaw|.
        # Lightweight — face_detector runs in <1s for 16 frames.
        try:
            from latentsync.utils.face_detector import FaceDetector
            _fd = FaceDetector()
            sample_n = min(16, len(frames))
            sample_idx = np.linspace(0, len(frames) - 1, sample_n).astype(int)
            yaws = []
            for idx in sample_idx:
                _, _ = _fd(frames[int(idx)])
                if _fd.last_pose_yaw is not None:
                    yaws.append(abs(float(_fd.last_pose_yaw)))
            avg_yaw = float(np.mean(yaws)) if yaws else 0.0
        except Exception:
            avg_yaw = 0.0

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
        # yaw summary
        if avg_yaw > 0:
            if avg_yaw >= 25:
                lines.append("")
                lines.append(f"⚠️ 平均 yaw {avg_yaw:.1f}° ≥ 25° — 偏侧脸场景")
                lines.append("   建议: Tab 1 preset 选 💋 Side-Face Lip Quality "
                             "(sync_loss↑ perceptual↑ conv wrap)")
            elif avg_yaw >= 15:
                lines.append("")
                lines.append(f"ℹ️ 平均 yaw {avg_yaw:.1f}° (15-25° 区间) — 轻度侧脸")
                lines.append("   若同步/清晰度不佳,可考虑 💋 Side-Face Lip Quality")
            else:
                lines.append(f"\n✅ 平均 yaw {avg_yaw:.1f}° < 15° (基本正面)")

        return blurry_ratio, flicker_score, float(sync_conf), identity_sim, avg_yaw, "\n".join(lines)
    except Exception as e:
        return 0.0, 0.0, 0.0, 0.0, 0.0, f"❌ 检测失败: {e}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    allowed = [str(REPO_ROOT), str(FINETUNE_BASE_DIR), "/tmp"]
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        allowed_paths=allowed,
        show_api=False,
    )


if __name__ == "__main__":
    main()
