"""High-level CodeFormer wrapper used by the LipSync pipeline.

The pipeline hands us a ``(T, 3, H, W)`` tensor of aligned face crops
already in ``[-1, 1]`` and of resolution ``H == W == self.resolution``
(512 in production). Our job is to feed each crop through CodeFormer in
batches, leaving the data layout / dtype / device untouched so the
caller can paste the result back into the full-frame video with
``restore_img`` as usual.

Why a separate wrapper from ``codeformer.CodeFormer``?
  * Lazy load -- the model is ~1 GB on GPU and many requests will not
    enable it, so we don't want to allocate it at server startup.
  * Batched inference -- CodeFormer is slow per-call (~50 ms on a 512
    face on a modern GPU), so we chunk the ``T`` frames into
    ``batch_size`` slices.
  * Skip / preserve passthrough -- frames that the pipeline decided to
    skip (side profile, motion blur, occluded mouth) should *not* be
    restored, because the inpainter's output was discarded in favour
    of the source frame. Pushing them through CodeFormer would sharpen
    the un-touched source face and create a visible style mismatch.
  * Stats -- we track how many frames were actually enhanced, so the
    API can report it back to the caller.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import torch

from .codeformer import load_codeformer

logger = logging.getLogger(__name__)


@dataclass
class CodeformerStats:
    """Per-request telemetry for the API response."""

    enabled: bool = False
    loaded: bool = False
    checkpoint_path: str = ""
    fidelity_weight: float = 0.5
    batch_size: int = 8
    frames_total: int = 0
    frames_enhanced: int = 0
    frames_skipped_by_pipeline: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "loaded": self.loaded,
            "checkpoint_path": self.checkpoint_path,
            "fidelity_weight": self.fidelity_weight,
            "batch_size": self.batch_size,
            "frames_total": self.frames_total,
            "frames_enhanced": self.frames_enhanced,
            "frames_skipped_by_pipeline": self.frames_skipped_by_pipeline,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "error": self.error,
        }


class CodeFormerRestorer:
    """Lazy wrapper around the CodeFormer model.

    Lifecycle:
      * ``__init__`` only stores configuration -- no model is built.
      * The first call to :meth:`restore_faces` loads the checkpoint
        and moves the model to ``device``. Subsequent calls reuse it.
      * :meth:`restore_faces` is the only method the pipeline needs.

    Thread-safety: a single :class:`CodeFormerRestorer` instance is
    intended to be shared by the API runtime's process-locked inference
    path. We do not provide concurrent-call safety; the LatentSync API
    already serialises inference through ``runtime.run_lock``.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Union[str, torch.device] = "cuda",
        batch_size: int = 8,
        adain: bool = True,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device) if isinstance(device, str) else device
        self.batch_size = max(1, int(batch_size))
        self.adain = bool(adain)
        self._net = None  # type: Optional[torch.nn.Module]
        self._load_error: str = ""

    # -- internal --------------------------------------------------------

    def _ensure_loaded(self) -> Optional[torch.nn.Module]:
        if self._net is not None:
            return self._net
        if self._load_error:
            return None
        if not self.checkpoint_path or not os.path.isfile(self.checkpoint_path):
            self._load_error = (
                f"CodeFormer checkpoint not found at {self.checkpoint_path!r}"
            )
            logger.error(self._load_error)
            return None
        try:
            t0 = time.time()
            self._net = load_codeformer(self.checkpoint_path, device="cpu")
            self._net = self._net.to(self.device)
            logger.info(
                "[CodeFormer] Loaded weights from %s in %.2fs onto %s",
                self.checkpoint_path,
                time.time() - t0,
                self.device,
            )
        except Exception as exc:  # noqa: BLE001 -- surface to caller via stats
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("[CodeFormer] Failed to load model: %s", exc)
            self._net = None
        return self._net

    # -- public API -----------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._net is not None

    @property
    def load_error(self) -> str:
        return self._load_error

    @torch.no_grad()
    def restore_faces(
        self,
        faces: torch.Tensor,
        skip_mask: Optional[Sequence[bool]] = None,
        fidelity_weight: float = 0.5,
    ) -> tuple:
        """Restore a sequence of aligned face crops.

        Args:
            faces: ``(T, 3, H, W)`` tensor in ``[-1, 1]`` on whichever
                device the pipeline is using. ``H == W`` is required --
                CodeFormer is fully convolutional so any size is fine
                in theory, but the affine aligner produces 512x512 and
                other resolutions are almost certainly a bug.
            skip_mask: optional ``Sequence[bool]`` of length ``T`` where
                ``True`` means the caller has already replaced this
                frame with the source video. We return those entries
                untouched so the caller doesn't have to re-apply the
                ``restore_img`` skip logic.
            fidelity_weight: ``w`` parameter for the SFT blocks. Smaller
                values produce a sharper but less identity-faithful
                face; ``0.5`` is the README's balanced default.

        Returns:
            ``(restored, stats)`` -- ``restored`` has the same shape,
            dtype and device as ``faces``. ``stats`` is a
            :class:`CodeformerStats` instance summarising what
            happened. When the model fails to load, ``restored`` is
            a clone of the input and ``stats.error`` describes why.
        """
        stats = CodeformerStats(
            enabled=True,
            checkpoint_path=self.checkpoint_path,
            fidelity_weight=float(fidelity_weight),
            batch_size=self.batch_size,
        )
        if faces.dim() != 4 or faces.shape[1] != 3:
            stats.error = (
                f"Expected faces of shape (T, 3, H, W); got {tuple(faces.shape)}"
            )
            logger.error("[CodeFormer] %s", stats.error)
            return faces.clone(), stats
        if faces.shape[2] != faces.shape[3]:
            stats.error = (
                f"Expected square face crops; got H={faces.shape[2]} W={faces.shape[3]}"
            )
            logger.error("[CodeFormer] %s", stats.error)
            return faces.clone(), stats

        T = faces.shape[0]
        stats.frames_total = T
        if skip_mask is not None:
            skip_list = list(skip_mask)
            if len(skip_list) < T:
                skip_list = skip_list + [False] * (T - len(skip_list))
            elif len(skip_list) > T:
                skip_list = skip_list[:T]
        else:
            skip_list = [False] * T
        stats.frames_skipped_by_pipeline = int(sum(skip_list))

        out = faces.clone()
        eligible_indices = [i for i, skip in enumerate(skip_list) if not skip]
        if not eligible_indices:
            stats.loaded = self.is_loaded
            return out, stats

        net = self._ensure_loaded()
        if net is None:
            stats.error = self._load_error or "CodeFormer model not loaded"
            stats.loaded = False
            return out, stats
        stats.loaded = True

        # Don't move the whole (T, 3, H, W) tensor to the model device;
        # move per-batch to keep peak memory low.
        w = float(max(0.0, min(1.0, fidelity_weight)))
        device = self.device
        bs = self.batch_size
        param_dtype = next(net.parameters()).dtype
        t0 = time.time()
        enhanced = 0
        try:
            for start in range(0, len(eligible_indices), bs):
                chunk_indices = eligible_indices[start : start + bs]
                idx = torch.as_tensor(chunk_indices, device=faces.device, dtype=torch.long)
                batch = faces.index_select(0, idx)
                # Move to model device (only the slice, not the whole T).
                batch_dev = batch.to(device=device, dtype=param_dtype)
                restored, _logits, _lq = net(batch_dev, w=w, adain=self.adain)
                restored = restored.to(device=faces.device, dtype=faces.dtype)
                # Clamp because the generator occasionally steps outside
                # [-1, 1] on extreme inputs (rare but observable on
                # very out-of-distribution faces) and downstream
                # paste-back does not clamp.
                restored = restored.clamp(-1.0, 1.0)
                for k, face_index in enumerate(chunk_indices):
                    out[face_index] = restored[k]
                enhanced += len(chunk_indices)
        except Exception as exc:  # noqa: BLE001
            stats.error = f"{type(exc).__name__}: {exc}"
            logger.exception("[CodeFormer] Inference failed: %s", exc)
            return faces.clone(), stats
        stats.frames_enhanced = enhanced
        stats.elapsed_seconds = time.time() - t0
        logger.info(
            "[CodeFormer] Enhanced %d / %d faces (w=%.2f, bs=%d, %.2fs)",
            enhanced, T, w, bs, stats.elapsed_seconds,
        )
        return out, stats
