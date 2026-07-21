"""Shared utilities for the fine-tuning UI."""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from omegaconf import OmegaConf

from latentsync.finetune import (
    CHECKPOINT_DIR,
    FINETUNE_BASE_DIR,
    REPO_ROOT,
    logger,
)

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
    """Candidate fine-tune datasets: directories under preprocess/ or data/ with mp4 files.

    Only these two subtrees are scanned — rglob over the whole repo root
    (outputs/, debug/, checkpoints/, .git/, …) made UI startup slow on
    data-heavy machines.
    """
    candidates: List[str] = []
    for root in (REPO_ROOT / "preprocess", REPO_ROOT / "data"):
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

# Full-UNet checkpoints are multi-GB; the monitor tab refreshes every 15s
# and used to torch.load the latest checkpoint TWICE per refresh just to
# read global_step + the last few losses. Cache the extracted metadata
# keyed by (path, mtime_ns, size) so a freshly written checkpoint
# invalidates automatically. Only the latest key is kept to bound memory.
_CKPT_META_CACHE: Dict[Tuple[str, int, int], Dict[str, Any]] = {}


def _load_ckpt_meta(ckpt_path: Path) -> Dict[str, Any]:
    """Extract cheap scalar metadata from a full-UNet .pt checkpoint."""
    st = ckpt_path.stat()
    mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
    key = (str(ckpt_path), mtime_ns, st.st_size)
    cached = _CKPT_META_CACHE.get(key)
    if cached is not None:
        return cached

    import torch
    try:
        # mmap=True keeps the tensor bytes on disk; we only touch scalars.
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False, mmap=True)
    except TypeError:  # older torch without mmap support
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    meta: Dict[str, Any] = {
        "global_step": int(ckpt.get("global_step", 0) or 0),
        "state_dict_keys": len(ckpt.get("state_dict", {})),
        "train_step_list_len": len(ckpt["train_step_list"]) if "train_step_list" in ckpt else None,
        "train_loss_list_len": len(ckpt["train_loss_list"]) if "train_loss_list" in ckpt else None,
        "last_train_losses": [round(float(x), 4) for x in (ckpt.get("train_loss_list") or [])[-5:]],
        "last_val_losses": [round(float(x), 4) for x in (ckpt.get("val_loss_list") or [])[-5:]],
    }
    _CKPT_META_CACHE.clear()
    _CKPT_META_CACHE[key] = meta
    return meta


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
        meta = _load_ckpt_meta(Path(ckpt_path))
        info = {
            "type": "full UNet checkpoint",
            "global_step": meta["global_step"],
            "state_dict_keys": meta["state_dict_keys"],
            "train_step_list_len": meta["train_step_list_len"]
            if meta["train_step_list_len"] is not None else "n/a",
            "train_loss_list_len": meta["train_loss_list_len"]
            if meta["train_loss_list_len"] is not None else "n/a",
        }
        if meta["last_train_losses"]:
            info["last_5_train_losses"] = meta["last_train_losses"]
        if meta["last_val_losses"]:
            info["last_5_val_losses"] = meta["last_val_losses"]
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
            meta = _load_ckpt_meta(latest_ckpt)
            step = meta["global_step"]
            if meta["last_train_losses"]:
                latest_loss = float(meta["last_train_losses"][-1])
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
def tail_file(path, n_lines: int = 30, missing_msg: str = "(log file not found)") -> str:
    """Read the last N lines of a text file (best-effort).

    Accepts ``Path`` / ``str`` / ``None``; a missing file yields
    ``missing_msg`` instead of raising.
    """
    if not path:
        return missing_msg
    p = Path(path)
    if not p.exists():
        return missing_msg
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 50_000))
            data = f.read().decode("utf-8", errors="replace")
        return "\n".join(data.splitlines()[-int(n_lines):])
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
def _analyze_training_video_yaw(video_path: str, n_frames: int = 5, detector=None) -> Dict[str, Any]:
    """Sample frames from a training video and estimate face yaw.

    Returns a dict with yaw_mean, yaw_max, detect_rate, face_type and an
    optional error key. Used by the training-set preview tab to filter
    frontal / side-face samples.

    ``detector`` may be passed in so a batch analysis loads insightface
    once instead of once per video.
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

        if detector is None:
            import torch

            detector = FaceDetector(
                device="cuda" if torch.cuda.is_available() else "cpu",
                # Measure yaw on side faces too — the caller decides the
                # frontal/side cutoff; the default 15° skip would hide them.
                skip_side_face_threshold=None,
                allowed_modules=["detection", "landmark_2d_106"],
            )

        yaws: List[float] = []
        for idx in indices:
            _, frame = reader[idx]
            if hasattr(frame, "asnumpy"):
                frame = frame.asnumpy()
            bbox, _ = detector(frame)
            if bbox is not None and detector.last_pose_yaw is not None:
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
