"""Tab 2: training monitoring callbacks."""
from pathlib import Path
from typing import Any, Optional, Tuple

import gradio as gr

from latentsync.finetune import REPO_ROOT, logger
from latentsync.finetune.process import _TRAINER
from latentsync.finetune.utils import (
    _compute_progress,
    _format_progress_text,
    _list_run_dirs_for_monitor,
    _resolve_run_dir,
    list_checkpoints_in_run,
    list_validation_videos,
    parse_loss_chart,
    parse_sync_conf_chart,
    read_loss_from_checkpoint,
    tail_log,
)

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
