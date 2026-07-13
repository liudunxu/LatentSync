"""Save and restore full training state for crash recovery.

Saves a `training_state.pt` alongside the model checkpoint, containing:

  - global_step
  - optimizer.state_dict()
  - scaler.state_dict() (if mixed precision)
  - lr_scheduler.state_dict()
  - torch / cuda / numpy / random RNG state
  - best sync_conf so far (for early-stop style heuristics)

At startup, train_unet.py / train_unet_lora.py look for the most
recent training_state.pt under `output_dir` and auto-resume.

Usage:
    state = TrainingState(output_dir)
    state.save(global_step, optimizer, scaler, lr_scheduler, sync_conf)
    if state.can_resume():
        step, opt, scal, sched = state.load(device)
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch


STATE_FILENAME = "training_state.pt"


class TrainingState:
    """Crash-recoverable training state."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / STATE_FILENAME

    def can_resume(self) -> bool:
        return self.path.exists()

    def save(
        self,
        global_step: int,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler],
        lr_scheduler: Any,
        extra: Optional[dict] = None,
    ) -> None:
        """Atomically save full training state.

        The atomic rename (tmp -> real) means a crash mid-save can't
        leave a half-written file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "extra": extra or {},
        }
        tmp = self.path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, self.path)

    def load(
        self, device: torch.device
    ) -> Tuple[int, torch.optim.Optimizer, Optional[torch.amp.GradScaler], Any]:
        """Load state. Caller is responsible for re-creating optimizer /
        scheduler with the right param groups BEFORE calling .attach().

        Returns (global_step, optimizer_state_dict, scaler_state_dict,
        lr_scheduler_state_dict).
        """
        payload = torch.load(self.path, map_location=device, weights_only=False)
        # Restore RNG (so resumed training is reproducible)
        rng = payload.get("rng", {})
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].to("cpu"))
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.to("cpu") for s in rng["cuda"]])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("python") is not None:
            random.setstate(rng["python"])
        return (
            int(payload["global_step"]),
            payload.get("optimizer"),
            payload.get("scaler"),
            payload.get("lr_scheduler"),
        )

    def cleanup(self) -> None:
        if self.path.exists():
            self.path.unlink()


def find_latest_checkpoint(
    output_dir: str | Path,
    prefer_state: bool = True,
) -> Optional[Path]:
    """Look for the most recent checkpoint in `output_dir`.

    Returns the path to:
      1. training_state.pt (if prefer_state and exists), else
      2. The latest checkpoint-{step}.pt (by step number), or
      3. None if nothing found.
    """
    p = Path(output_dir)
    if not p.exists():
        return None
    state_path = p / STATE_FILENAME
    if prefer_state and state_path.exists():
        return state_path
    ckpts = sorted(
        p.glob("checkpoints/checkpoint-*.pt"),
        key=lambda q: int(q.stem.split("-")[-1]) if "-" in q.stem else 0,
        reverse=True,
    )
    return ckpts[0] if ckpts else None
