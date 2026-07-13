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
import shlex
import shutil
import signal
import threading
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
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
    },
    "🎯 Badcase Fix (侧脸+运动, LoRA, 12-15GB)": {
        "config_file": "configs/unet/stage2.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        "num_frames": 24,
        "resolution": 256,
        "learning_rate": 5e-5,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        "sync_loss_weight": 0.12,
        "perceptual_loss_weight": 0.15,
        "recon_loss_weight": 1.0,
        "trepa_loss_weight": 10.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask.png",
        "save_ckpt_steps": 1000,
        "max_train_steps": 20000,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 200,
        "description": (
            "🟢 **推荐 — 内容型 badcase**\n"
            "只 wrap 注意力,但 sync_loss↑到 0.12, num_frames=24,"
            "cosine+200 warmup, 每 1k 步存 ckpt。\n"
            "适用:嘴型/audio 同步、嘴糊、paste-back 外溢。\n"
            "结构性脸变形见 🧩 Structural Fix。"
        ),
        "lora": {
            "enabled": True,
            "rank": 32,
            "alpha": 64,
            "dropout": 0.05,
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
            "qlora": False,
        },
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
        "sync_loss_weight": 0.18,
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


def _on_page_load():
    """Repopulate training-status UI on page (re)load.

    The Python process keeps the trainer subprocess alive across browser
    refreshes, but the browser-side gr.State values (log_path, run_dd,
    …) reset to empty. This handler re-pulls from the in-process
    _TRAINER singleton.

    Wrapped in try/except so any failure here (slow filesystem, weird
    run_dir contents, etc.) NEVER blocks the page-load — the UI just
    falls back to idle.
    """
    try:
        if _TRAINER.is_running():
            pid = _TRAINER.proc.pid
            rc_hint = _TRAINER.proc.poll()
            rc_text = f" (rc={rc_hint})" if rc_hint is not None else ""
            trainer_text = (
                f"⏳ 训练进行中 (pid={pid}{rc_text})\n"
                f"📂 run_dir: {_TRAINER.run_dir or '(unknown)'}\n"
                "💡 点击 Tab 2 的 '🔄 刷新 run 列表' + '🔄 手动刷新' 来查看进度"
            )
            launch_text = f"⏳ training running since {_TRAINER.started_at or '?'}"
            try:
                runs = list_run_dirs(FINETUNE_BASE_DIR / "unet")
            except Exception as exc:
                logger.warning("list_run_dirs failed on page load: %s", exc)
                runs = []
            return (
                trainer_text,
                launch_text,
                gr.update(choices=runs, value=_TRAINER.run_dir.name if _TRAINER.run_dir else None),
                gr.update(interactive=True),
            )
        trainer_text = "🟢 idle — 点 '🚀 启动训练' 开始"
        launch_text = ""
        try:
            runs = list_run_dirs(FINETUNE_BASE_DIR / "unet")
        except Exception as exc:
            logger.warning("list_run_dirs failed on page load: %s", exc)
            runs = []
        return (
            trainer_text,
            launch_text,
            gr.update(choices=runs, value=None),
            gr.update(interactive=True),
        )
    except Exception as exc:
        logger.exception("page-load handler failed entirely: %s", exc)
        return (
            f"⚠️ page-load handler 出错: {exc}",
            "",
            gr.update(choices=[], value=None),
            gr.update(interactive=True),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_run_dirs(base: Path) -> List[str]:
    """List timestamped run directories produced by train_unet.py / train_syncnet.py."""
    if not base.exists():
        return []
    runs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("train-")])
    # If base is inside REPO_ROOT, keep paths relative for compact display.
    # Otherwise (e.g. /root/autodl-tmp/...), return absolute paths.
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
    freeze_attn2: bool,
    val_inference_steps: int,
    val_guidance_scale: float,
    val_seed: int,
    lr_scheduler: str,
    lr_warmup_steps: int,
) -> Dict[str, Any]:
    """Merge user-form values with the chosen preset's defaults."""
    preset = PRESETS[preset_name]
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
            "mask_image_path": mask_image_path,
            "audio_sample_rate": 16000,
            "video_fps": 25,
            "audio_feat_length": [2, 2],
            "train_output_dir": train_output_dir or str(FINETUNE_BASE_DIR / "unet"),
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
                _TRAINER.proc.pid,
            )
            return (
                f"❌ 训练已在运行中 (pid={_TRAINER.proc.pid}, run={_TRAINER.run_dir.name})",
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
            freeze_attn2=freeze_attn2,
            val_inference_steps=val_inference_steps,
            val_guidance_scale=val_guidance_scale,
            val_seed=val_seed,
            lr_scheduler=lr_scheduler,
            lr_warmup_steps=lr_warmup_steps,
        )
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
    pid = _TRAINER.proc.pid
    _TRAINER.stop()
    return f"⏹ 已停止 (pid={pid})"


def refresh_training_log() -> Tuple[str, str]:
    """Return (latest log tail, process status) for the running/last training."""
    if not _TRAINER.log_path or not _TRAINER.log_path.exists():
        status = "ℹ️ 暂无训练日志"
        if _TRAINER.is_running():
            status = f"🟡 训练已启动但日志尚未写入 (pid={_TRAINER.proc.pid})"
        return "(log file not found yet)", status

    log_text = tail_file(_TRAINER.log_path, n_lines=80)

    if _TRAINER.is_running():
        status = f"🟢 训练中 (pid={_TRAINER.proc.pid}, started={_TRAINER.started_at})"
    else:
        exit_code = _TRAINER.proc.poll() if _TRAINER.proc else "?"
        if exit_code == 0:
            status = f"✅ 训练已正常结束 (exit_code=0)"
        elif exit_code is None:
            status = "ℹ️ 无正在运行的训练任务"
        else:
            status = f"🔴 训练异常退出 (exit_code={exit_code})，请看下方日志"
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


def refresh_runs(train_output_dir: str) -> gr.update:
    base = _resolve_output_dir(train_output_dir)
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
        rd = _resolve_run_dir(run_dir_path)
        if rd is None:
            return None
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
        rd = _resolve_run_dir(run_dir_path)
        if rd is None:
            return []
    vd = rd / "val_videos"
    if not vd.exists():
        return []
    return sorted([str(p) for p in vd.glob("*.mp4")], key=lambda p: Path(p).stat().st_mtime, reverse=True)


def list_checkpoints_in_run(run_dir_path: Optional[str]) -> List[str]:
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
    try:
        return [str(p.relative_to(REPO_ROOT)) for p in sorted(ck.glob("*.pt"))]
    except ValueError:
        return [str(p) for p in sorted(ck.glob("*.pt"))]


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


def monitor_refresh(
    train_output_dir: str,
    selected_run: Optional[str],
    log_path: Optional[str],
) -> Tuple[str, str, Any, str, str, Any]:
    """Pull the latest snapshot. Returns: (run_dir_choices, selected_run_disp,
    loss_chart, val_video_choices, log_tail, ckpt_info)."""
    base = _resolve_output_dir(train_output_dir)
    run_choices = list_run_dirs(base)
    run_dir = _resolve_run_dir(selected_run)
    chart = parse_loss_chart(str(run_dir) if run_dir else None)
    val_videos = list_validation_videos(str(run_dir) if run_dir else None)
    log_text = tail_log(log_path, n_lines=80)
    ckpts = list_checkpoints_in_run(str(run_dir) if run_dir else None)
    if ckpts:
        ckpt_path = Path(ckpts[-1])
        if not ckpt_path.is_absolute():
            ckpt_path = REPO_ROOT / ckpt_path
        ckpt_info = read_loss_from_checkpoint(str(ckpt_path))
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

    _prune_debug_files(REPO_ROOT / "debug", "compare_cfg_*.yaml")
    _prune_debug_files(REPO_ROOT / "debug" / "compare_outputs", "compare_cfg_*.yaml")

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
        )
    if not ckpt_path or not Path(ckpt_path).exists():
        return (gr.update(value=None), gr.update(), gr.update(value=f"❌ ckpt 不存在: {ckpt_path}"), gr.update())
    if not unet_config or not Path(unet_config).exists():
        return (gr.update(value=None), gr.update(), gr.update(value=f"❌ config 不存在: {unet_config}"), gr.update())

    warnings = _check_ckpt_compatibility(ckpt_path, unet_config)
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
    log_path = out_dir / f"validation_{ts}.log"

    if not _INFERENCE.start(
        cmd, log_path,
        kind="validate",
        label=f"validate ckpt={Path(ckpt_path).name}",
        result_video=out_mp4,
    ):
        return (gr.update(value=None), gr.update(), gr.update(value="❌ 启动失败"), gr.update())

    skip_msg = " (skip quality check)" if skip_quality_check else ""
    return (
        gr.update(value=None),
        gr.update(value=warnings_text),
        gr.update(value=(
            f"⏳ validate 推理启动中{skip_msg}\n"
            f"📜 log: {log_path.relative_to(REPO_ROOT)}\n"
            f"💡 点击 ⏹ 取消 或 等 timer 自动刷新"
        )),
        gr.update(value=None),
    )


def _poll_inference_state(skip_quality_check: bool):
    """Timer poller — checks _INFERENCE status and delivers the result.

    Called by Tab 3.5's gr.Timer every 3s while a validation is running.
    On DONE: runs _quick_quality_check and pushes the final report +
    video path into the UI (one-shot, then resets result_video).
    """
    state = _INFERENCE

    if state.kind != "validate" or state.status not in (state.DONE, state.FAILED, state.CANCELLED):
        if state.is_busy():
            try:
                pid = state.proc.pid if state.proc else "?"
            except Exception:
                pid = "?"
            return (
                gr.update(value=None),
                gr.update(),
                gr.update(value=(
                    f"⏳ {state.label} running (pid={pid})"
                )),
                gr.update(),
            )
        return (gr.update(), gr.update(), gr.update(), gr.update())

    # ---- one-shot consumption of a finished result ----
    result_video = state.result_video
    exit_code = state.exit_code
    log_path = state.log_path
    label = state.label

    if state.status == state.DONE:
        _INFERENCE.status = _INFERENCE.IDLE
        _INFERENCE.result_video = None

        report = f"✅ 推理完成\n📂 {result_video.relative_to(REPO_ROOT)}\n📜 log: {log_path.relative_to(REPO_ROOT)}"
        if not skip_quality_check and result_video.exists():
            metrics = _quick_quality_check(str(result_video))
            report = _format_validation_report(metrics, label, duration=0.0)
        return (
            gr.update(value=str(result_video)),
            gr.update(),
            gr.update(value=report),
            gr.update(value=str(result_video)),
        )

    if state.status == state.CANCELLED:
        _INFERENCE.status = _INFERENCE.IDLE
        _INFERENCE.result_video = None
        return (
            gr.update(value=None),
            gr.update(),
            gr.update(value=(
                f"⏹ 已取消\n📜 log: {log_path.relative_to(REPO_ROOT)}\n"
                f"最后 30 行:\n{tail_file(log_path, 30)}"
            )),
            gr.update(value=None),
        )

    # FAILED
    _INFERENCE.status = _INFERENCE.IDLE
    _INFERENCE.result_video = None
    err_text = (
        f"❌ 推理失败 (rc={exit_code})\n"
        f"📜 log: {log_path.relative_to(REPO_ROOT)}\n\n"
        f"最后 30 行:\n{tail_file(log_path, 30)}"
    )
    return (
        gr.update(value=None),
        gr.update(),
        gr.update(value=err_text),
        gr.update(value=None),
    )


def stop_inference() -> str:
    """⏹ cancel button — sends SIGINT to the running inference subprocess group."""
    return _INFERENCE.stop()
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

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="LatentSync Fine-tune Studio",
        theme=gr.themes.Soft(),
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
                        value="constant",
                        label="lr_scheduler",
                    )
                    lr_warmup_steps = gr.Slider(
                        0, 2000, value=0, step=50, label="lr_warmup_steps (推荐 100-300 for cosine)"
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
                    preset_desc,
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
                if ckpts:
                    ckpt_path = Path(ckpts[-1])
                    if not ckpt_path.is_absolute():
                        ckpt_path = REPO_ROOT / ckpt_path
                    ck_info = read_loss_from_checkpoint(str(ckpt_path))
                else:
                    ck_info = "(no checkpoint yet)"
                return chart, gr.update(choices=vids, value=vids[0] if vids else None), ck_info

            run_dd.change(
                fn=_on_run_change,
                inputs=run_dd,
                outputs=[loss_chart_img, val_video_dd, ckpt_info_box],
            )
            val_video_dd.change(
                fn=_safe_video_update,
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
                    val_ckpt = gr.Dropdown(
                        choices=list_checkpoints(),
                        label="Checkpoint（base / fine-tuned）",
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
                    val_resolution = gr.Radio([256, 512], value=256, label="resolution")

            with gr.Row():
                val_steps = gr.Slider(10, 50, value=20, step=1, label="inference_steps")
                val_guidance = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="guidance_scale")
                val_seed = gr.Number(value=1247, label="seed", precision=0)
                val_deepcache = gr.Checkbox(value=True, label="enable_deepcache (快 2x)")
                val_skip_qc = gr.Checkbox(value=False, label="跳过质量自检（更快）")

            val_btn = gr.Button("🚀 推理 + 质量自检", variant="primary")

            val_compat = gr.Textbox(label="ckpt 兼容性检查", lines=4, interactive=False)
            val_output = gr.Video(label="生成结果", interactive=False)
            val_report = gr.Textbox(label="质量报告", lines=18, interactive=False)
            val_saved = gr.Textbox(label="保存路径", interactive=False)

            val_btn.click(
                fn=run_validation,
                inputs=[
                    val_video, val_audio, val_ckpt, val_config,
                    val_steps, val_guidance, val_seed, val_resolution,
                    val_deepcache, val_skip_qc,
                ],
                outputs=[val_output, val_compat, val_report, val_saved],
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
                with gr.Column():
                    bc_report = gr.Textbox(label="诊断报告", lines=20)
                    bc_recommendation = gr.Textbox(
                        label="🎯 finetune preset 推荐 (基于上面 4 个数字自动判定)",
                        lines=4,
                        interactive=False,
                        value="跑完上方 🔍 检测后,这里会自动出推荐 preset。",
                    )

            bc_check_btn.click(
                fn=run_badcase_checklist,
                inputs=[bc_video, bc_reference],
                outputs=[bc_blurry, bc_flicker, bc_sync, bc_identity, bc_report],
            )

            # Whenever any of the 4 metric numbers change, refresh the
            # preset recommendation. Tab 6 re-run fills them all in one
            # .click event, so the user sees the recommendation update
            # immediately after the numbers settle.
            for bc_metric in (bc_blurry, bc_flicker, bc_sync, bc_identity):
                bc_metric.change(
                    fn=_recommend_finetune_preset,
                    inputs=[bc_blurry, bc_flicker, bc_sync, bc_identity],
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

            # On page (re)load: repopulate trainer status + run dropdown
            # from the in-process _TRAINER singleton. This survives browser
            # refreshes — the Python process keeps the training subprocess
            # alive even if the user's tab disconnects, so we re-detect here.
            demo.load(
                fn=_on_page_load,
                outputs=[
                    trainer_status, launch_status, run_dd, monitor_btn,
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
) -> str:
    """Return a multi-line recommendation string given the 4 badcase metrics.

    Rules (in order):
      1. identity < 0.70                         → 🧩 Structural Fix
      2. content issue AND identity 0.70-0.85    → 🧩 Structural Fix (cover both)
      3. content issue AND identity >= 0.85     → 🎯 Content Fix
      4. flicker > 12 alone (identity OK)       → 🎯 Content Fix
      5. identity 0.70-0.85 alone (no content)   → 🧩 Structural Fix (mild drift)
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
    if identity is not None and identity < s["identity_soft"]:
        return (
            "🧩 Structural Fix (LoRA + conv, 18-22GB)\n"
            f"   {metric_summary}\n"
            f"   内容型指标正常,但身份保持 = {identity:.3f} 偏低 → 脸轮廓漂,加 conv wrap。"
        )
    return (
        "⚪ 不需要 finetune / 用 Stage 2 LoRA baseline\n"
        f"   {metric_summary}\n"
        "   四个指标都在合理范围内。生成质量可用,需要换风格/换脸再调 preset。"
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
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()