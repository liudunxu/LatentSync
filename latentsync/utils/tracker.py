"""Optional experiment-tracking hooks for training scripts.

Centralises all the third-party logger plumbing (WandB / TensorBoard /
Aim) so train_unet.py and train_unet_lora.py share the same code.

If neither wandb nor tensorboard is installed, the Tracker silently
no-ops and the training loop is unaffected. The Tracker only writes
when the active run is the main process (rank 0).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


class Tracker:
    """Unified experiment logger. Falls back to no-op if no backend is available.

    Usage:
        tracker = Tracker(project="latentsync-finetune", config=cfg, rank=0)
        for step in range(...):
            tracker.log({"train/loss": loss.item(), "train/lr": lr}, step=step)
        tracker.finish()

    Set env LATENTSYNC_TRACKER=wandb or =tensorboard to force a backend.
    Default behaviour: prefer wandb if installed, else tensorboard, else no-op.
    """

    def __init__(
        self,
        project: str = "latentsync-finetune",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        rank: int = 0,
        log_dir: Optional[str] = None,
    ) -> None:
        self.rank = rank
        self.backend = None
        if rank != 0:
            return  # only main process logs

        forced = os.environ.get("LATENTSYNC_TRACKER", "").strip().lower()
        if forced == "off":
            return

        if forced in ("", "wandb"):
            try:
                import wandb
                wandb.init(project=project, name=run_name, config=config, dir=log_dir)
                self.backend = "wandb"
                return
            except ImportError:
                if forced == "wandb":
                    print("[Tracker] wandb requested but not installed; falling back.")
        if forced in ("", "tensorboard"):
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = log_dir or f"debug/tb_logs/{run_name or 'latentsync'}"
                self.tb = SummaryWriter(tb_dir)
                self.backend = "tensorboard"
                return
            except ImportError:
                if forced == "tensorboard":
                    print("[Tracker] tensorboard requested but not available.")

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        if self.rank != 0 or self.backend is None:
            return
        if self.backend == "wandb":
            import wandb
            wandb.log(metrics, step=step)
        elif self.backend == "tensorboard":
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, step)
                elif hasattr(v, "shape"):
                    # tensors: log histogram (best-effort)
                    try:
                        self.tb.add_histogram(k, v.detach().cpu(), step)
                    except Exception:
                        pass

    def finish(self) -> None:
        if self.rank != 0 or self.backend is None:
            return
        if self.backend == "wandb":
            import wandb
            wandb.finish()
        elif self.backend == "tensorboard":
            self.tb.close()
