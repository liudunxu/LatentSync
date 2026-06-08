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
from typing import Dict, List, Optional, Sequence, Tuple, Union

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
    adain: bool = True
    batch_size: int = 8
    frames_total: int = 0
    frames_enhanced: int = 0
    frames_fallback: int = 0
    frames_skipped_by_pipeline: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""
    # Tier 1 (adaptive w) and Tier 2 (retry) telemetry. ``bucket_counts``
    # is keyed by bucket NAME ("sharp"/"medium"/"blurry") rather than the
    # w value, so it round-trips through JSON cleanly.
    adaptive_w_enabled: bool = False
    retry_enabled: bool = False
    mouth_only_paste_enabled: bool = False
    w_sharp: float = 0.85
    w_medium: float = 0.7
    w_blurry: float = 0.5
    w_retry: float = 0.4
    bucket_counts: Dict[str, int] = field(default_factory=dict)
    frames_retry_attempted: int = 0
    frames_retry_succeeded: int = 0

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "loaded": self.loaded,
            "checkpoint_path": self.checkpoint_path,
            "fidelity_weight": self.fidelity_weight,
            "adain": self.adain,
            "batch_size": self.batch_size,
            "frames_total": self.frames_total,
            "frames_enhanced": self.frames_enhanced,
            "frames_fallback": self.frames_fallback,
            "frames_skipped_by_pipeline": self.frames_skipped_by_pipeline,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "error": self.error,
            "adaptive_w_enabled": self.adaptive_w_enabled,
            "retry_enabled": self.retry_enabled,
            "mouth_only_paste_enabled": self.mouth_only_paste_enabled,
            "w_sharp": self.w_sharp,
            "w_medium": self.w_medium,
            "w_blurry": self.w_blurry,
            "w_retry": self.w_retry,
            "bucket_counts": dict(self.bucket_counts),
            "frames_retry_attempted": self.frames_retry_attempted,
            "frames_retry_succeeded": self.frames_retry_succeeded,
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
        # Per-frame quality-check thresholds. After the model runs we
        # compare the restored crop against the input on three cheap
        # signals (sharpness ratio, whole-face pixel diff, mouth-region
        # pixel diff) and fall back to the input on any outlier. Set
        # the thresholds to 0 to disable that particular check, or
        # `fallback_enabled=False` to disable the whole safeguard.
        fallback_enabled: bool = True,
        fallback_sharpness_low: float = 0.5,
        fallback_sharpness_high: float = 2.0,
        fallback_pixel_diff: float = 0.20,
        fallback_mouth_diff: float = 0.15,
        # Tier 1: adaptive per-frame fidelity_weight (w). When
        # ``adaptive_w_enabled`` is True, frames are bucketed by input
        # mouth-region sharpness and each bucket is run through a
        # separate batched forward with its own w. ``sharp_threshold``
        # and ``blurry_threshold`` are mouth-region Laplacian variance
        # values (typical range 0.001-0.1 for 512x512 aligned faces);
        # ``w_sharp``/``w_medium``/``w_blurry`` are the w values per
        # bucket. When ``adaptive_w_enabled`` is False, the legacy
        # single-w behavior is preserved.
        adaptive_w_enabled: bool = False,
        sharp_threshold: float = 0.05,
        blurry_threshold: float = 0.01,
        w_sharp: float = 0.85,
        w_medium: float = 0.7,
        w_blurry: float = 0.5,
        # Tier 2: retry pass for frames that fail the quality check
        # in the blurry bucket. Re-runs those frames with ``w_retry``
        # (typically more aggressive than ``w_blurry``). Capped at
        # ``retry_max_frames`` per request as a circuit-breaker.
        retry_enabled: bool = False,
        w_retry: float = 0.4,
        retry_max_frames: int = 64,
        # Tier 3: mouth-region-only paste-back. When True, only the
        # mouth ROI of the restored face is pasted back; the rest of
        # the face stays as the inpainter's original output. This
        # decouples the aggressive-w passes from identity drift on
        # the eyes/forehead/cheeks. The ROI is a fixed rectangle
        # (matches the mouth_diff check) feathered with a Gaussian of
        # ``mouth_mask_feather_sigma`` pixels to avoid a hard seam.
        mouth_only_paste_enabled: bool = False,
        mouth_mask_feather_sigma: float = 5.0,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device) if isinstance(device, str) else device
        self.batch_size = max(1, int(batch_size))
        # adain stays an instance-level default; the per-call
        # ``adain`` argument to :meth:`restore_faces` can override it
        # so a single request can opt out without rebuilding the
        # restorer.
        self.adain = bool(adain)
        self.fallback_enabled = bool(fallback_enabled)
        self.fallback_sharpness_low = float(fallback_sharpness_low)
        self.fallback_sharpness_high = float(fallback_sharpness_high)
        # Thresholds are in [0, 1] (we normalize the [-1, 1] face
        # crops to [0, 1] before comparing).
        self.fallback_pixel_diff = float(fallback_pixel_diff)
        self.fallback_mouth_diff = float(fallback_mouth_diff)
        # Tier 1
        self.adaptive_w_enabled = bool(adaptive_w_enabled)
        self.sharp_threshold = float(sharp_threshold)
        self.blurry_threshold = float(blurry_threshold)
        self.w_sharp = float(w_sharp)
        self.w_medium = float(w_medium)
        self.w_blurry = float(w_blurry)
        # Tier 2
        self.retry_enabled = bool(retry_enabled)
        self.w_retry = float(w_retry)
        self.retry_max_frames = max(0, int(retry_max_frames))
        # Tier 3
        self.mouth_only_paste_enabled = bool(mouth_only_paste_enabled)
        self.mouth_mask_feather_sigma = float(mouth_mask_feather_sigma)
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
        adain: Optional[bool] = None,
        # Per-call overrides for Tier 1/2/3. ``None`` (default) means
        # use the instance-level value set at __init__ time. Pass an
        # explicit value to flip the feature on/off for this request
        # without rebuilding the restorer.
        adaptive_w_enabled: Optional[bool] = None,
        retry_enabled: Optional[bool] = None,
        mouth_only_paste_enabled: Optional[bool] = None,
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
                face; ``0.7`` is the lip-sync-tuned default -- the
                README's ``0.5`` is balanced for real-degraded faces
                but tends to over-reconstruct the inpainter's output.
                Ignored when ``adaptive_w_enabled`` is True (in that
                case the per-bucket w from ``__init__`` is used).
            adain: per-call override for the instance's adain flag.
                ``None`` (default) uses ``self.adain``; pass an explicit
                bool to opt in/out for this request without rebuilding
                the restorer.
            adaptive_w_enabled: per-call override for the Tier 1
                feature. ``None`` uses the instance default.
            retry_enabled: per-call override for the Tier 2 feature.
                ``None`` uses the instance default.
            mouth_only_paste_enabled: per-call override for the Tier 3
                feature. ``None`` uses the instance default.

        Returns:
            ``(restored, stats)`` -- ``restored`` has the same shape,
            dtype and device as ``faces``. ``stats`` is a
            :class:`CodeformerStats` instance summarising what
            happened. When the model fails to load, ``restored`` is
            a clone of the input and ``stats.error`` describes why.
        """
        adain_enabled = self.adain if adain is None else bool(adain)
        adaptive_on = self.adaptive_w_enabled if adaptive_w_enabled is None else bool(adaptive_w_enabled)
        retry_on = self.retry_enabled if retry_enabled is None else bool(retry_enabled)
        mouth_paste_on = self.mouth_only_paste_enabled if mouth_only_paste_enabled is None else bool(mouth_only_paste_enabled)
        stats = CodeformerStats(
            enabled=True,
            checkpoint_path=self.checkpoint_path,
            fidelity_weight=float(fidelity_weight),
            adain=bool(adain_enabled),
            batch_size=self.batch_size,
            adaptive_w_enabled=adaptive_on,
            retry_enabled=retry_on,
            mouth_only_paste_enabled=mouth_paste_on,
            w_sharp=self.w_sharp,
            w_medium=self.w_medium,
            w_blurry=self.w_blurry,
            w_retry=self.w_retry,
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

        # Tier 1: if adaptive w is on, bucket eligible frames by mouth-
        # region sharpness and dispatch one forward per non-empty bucket.
        # When adaptive is off, all eligible frames go into a single
        # bucket with w=fidelity_weight (legacy single-pass behavior).
        param_dtype = next(net.parameters()).dtype
        t0 = time.time()
        enhanced = 0
        fallback = 0
        retry_attempted = 0
        retry_succeeded = 0
        bucket_counts: Dict[str, int] = {"sharp": 0, "medium": 0, "blurry": 0}
        if adaptive_on:
            sharpness = self._mouth_sharpness_batch(faces[eligible_indices])
            buckets = self._bucket_by_sharpness(
                sharpness,
                self.sharp_threshold,
                self.blurry_threshold,
                self.w_sharp,
                self.w_medium,
                self.w_blurry,
            )
            for w_val, idxs in buckets.items():
                bucket_counts[self._bucket_name_for(w_val)] = len(idxs)
            # Surface bucket counts only when adaptive is on -- the
            # non-adaptive path leaves stats.bucket_counts at the
            # dataclass default (empty dict) to avoid implying a
            # three-bucket partition that didn't happen.
            stats.bucket_counts = bucket_counts
        else:
            w = float(max(0.0, min(1.0, fidelity_weight)))
            buckets = {w: list(range(len(eligible_indices)))}

        try:
            for w_val, local_indices in buckets.items():
                if not local_indices:
                    continue
                # Map local-bucket indices back to global frame indices.
                global_indices = [eligible_indices[i] for i in local_indices]
                e, f, failed = self._run_bucket(
                    net,
                    faces,
                    out,
                    global_indices,
                    w=float(w_val),
                    adain_enabled=adain_enabled,
                    bs=self.batch_size,
                    param_dtype=param_dtype,
                    device=self.device,
                    mouth_only_paste=mouth_paste_on,
                )
                enhanced += e
                fallback += f
                # Tier 2: retry only the w_blurry bucket, only when retry
                # is enabled. Capping at retry_max_frames is a safety
                # net -- a pathological input where every frame falls
                # into the blurry bucket would otherwise spin forever.
                if retry_on and failed and abs(float(w_val) - self.w_blurry) < 1e-6:
                    cap = min(len(failed), self.retry_max_frames)
                    to_retry = failed[:cap]
                    retry_attempted += len(to_retry)
                    e2, _f2, _still_failed = self._run_bucket(
                        net,
                        faces,
                        out,
                        to_retry,
                        w=self.w_retry,
                        adain_enabled=adain_enabled,
                        bs=self.batch_size,
                        param_dtype=param_dtype,
                        device=self.device,
                        mouth_only_paste=mouth_paste_on,
                    )
                    retry_succeeded += e2
                    # Frames that fail the retry pass keep their input
                    # value (out[] was already populated with batch by
                    # the first pass's fallback branch).
        except Exception as exc:  # noqa: BLE001
            stats.error = f"{type(exc).__name__}: {exc}"
            logger.exception("[CodeFormer] Inference failed: %s", exc)
            return faces.clone(), stats
        stats.frames_enhanced = enhanced + retry_succeeded
        stats.frames_fallback = max(0, fallback - retry_succeeded)
        stats.frames_retry_attempted = retry_attempted
        stats.frames_retry_succeeded = retry_succeeded
        stats.elapsed_seconds = time.time() - t0
        # Log line lists bucket counts when adaptive w is on so the
        # operator can see the actual sharpness distribution per
        # request without grepping a separate tool.
        if adaptive_on:
            logger.info(
                "[CodeFormer] Enhanced %d / %d faces (buckets=%s, bs=%d, %.2fs, fallback=%d, "
                "retry=%d/%d)",
                enhanced, T, bucket_counts, self.batch_size,
                stats.elapsed_seconds, fallback, retry_succeeded, retry_attempted,
            )
        else:
            logger.info(
                "[CodeFormer] Enhanced %d / %d faces (w=%.2f, bs=%d, %.2fs, fallback=%d)",
                enhanced, T, float(max(0.0, min(1.0, fidelity_weight))),
                self.batch_size, stats.elapsed_seconds, fallback,
            )
        return out, stats

    # -- quality-check helpers ------------------------------------------

    @staticmethod
    def _laplacian_variance(x: torch.Tensor) -> torch.Tensor:
        """Per-sample Laplacian variance of a ``(B, 3, H, W)`` batch.

        Returns a ``(B,)`` tensor. Always runs in fp32 to avoid the
        fp16-underflow issue called out in the pipeline's
        ``_face_sharpness`` docstring.
        """
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=x.device,
        ).view(1, 1, 3, 3)
        gray = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        lap = torch.nn.functional.conv2d(gray, kernel, padding=1)
        return lap.pow(2).mean(dim=(1, 2, 3))  # (B,)

    @staticmethod
    def _quality_check_batch(
        inp: torch.Tensor,
        restored: torch.Tensor,
        sharpness_low: float,
        sharpness_high: float,
        pixel_diff: float,
        mouth_diff: float,
    ) -> torch.Tensor:
        """Per-sample OK mask. ``True`` = keep restored, ``False`` = fallback to input.

        Both inputs are ``(B, 3, H, W)`` in ``[-1, 1]``; the function
        normalises to ``[0, 1]`` for the diff thresholds. All three
        checks are vectorised over the batch; the function only does
        one implicit sync (the final bool ops). A threshold <= 0
        disables that individual check.
        """
        # Whole-face pixel diff (mean abs, normalised to [0, 1])
        if pixel_diff > 0:
            pix = (restored - inp).abs().mean(dim=(1, 2, 3)) * 0.5
            pix_ok = pix <= pixel_diff
        else:
            pix_ok = torch.ones(inp.shape[0], dtype=torch.bool, device=inp.device)

        # Mouth-region diff
        if mouth_diff > 0:
            H, W = inp.shape[-2:]
            y0, y1 = int(H * 0.55), int(H * 0.74)
            x0, x1 = int(W * 0.30), int(W * 0.70)
            mouth_inp = inp[..., y0:y1, x0:x1]
            mouth_rest = restored[..., y0:y1, x0:x1]
            mdiff = (mouth_rest - mouth_inp).abs().mean(dim=(1, 2, 3)) * 0.5
            mouth_ok = mdiff <= mouth_diff
        else:
            mouth_ok = torch.ones(inp.shape[0], dtype=torch.bool, device=inp.device)

        # Sharpness ratio. Skip the check when the input is essentially
        # flat (e.g. zero placeholders, motion-blurred) because the
        # ratio is meaningless in that regime.
        if sharpness_low > 0 or sharpness_high > 0:
            lap_inp = CodeFormerRestorer._laplacian_variance(inp)
            lap_rest = CodeFormerRestorer._laplacian_variance(restored)
            safe_lap_inp = lap_inp.clamp_min(1e-3)
            ratio = lap_rest / safe_lap_inp
            sharp_ok = torch.ones_like(ratio, dtype=torch.bool)
            if sharpness_low > 0:
                sharp_ok = sharp_ok & (ratio >= sharpness_low)
            if sharpness_high > 0:
                sharp_ok = sharp_ok & (ratio <= sharpness_high)
            # If input itself is too flat, defer (treat as OK so we
            # don't penalise the model for sharpening a flat input).
            flat_input = lap_inp < 1e-3
            sharp_ok = sharp_ok | flat_input
        else:
            sharp_ok = torch.ones(inp.shape[0], dtype=torch.bool, device=inp.device)

        return pix_ok & mouth_ok & sharp_ok

    # -- Tier 1 helpers -------------------------------------------------

    @staticmethod
    def _mouth_sharpness_batch(faces: torch.Tensor) -> torch.Tensor:
        """Per-sample mouth-region Laplacian variance, in fp32.

        Used by the adaptive-w bucketing to decide which ``w`` to apply
        to each frame. Same ROI as ``_quality_check_batch``'s mouth
        diff check (``y in [0.55, 0.74] * H, x in [0.30, 0.70] * W``).
        Returns a ``(B,)`` tensor.

        Sharpness is computed on the *input* faces, not on the
        CodeFormer output -- bucketing is about how aggressive the
        codebook should be allowed to be, which depends on the input.
        """
        H, W = faces.shape[-2:]
        y0, y1 = int(H * 0.55), int(H * 0.74)
        x0, x1 = int(W * 0.30), int(W * 0.70)
        mouth = faces[..., y0:y1, x0:x1].float()
        return CodeFormerRestorer._laplacian_variance(mouth)

    @staticmethod
    def _bucket_by_sharpness(
        sharpness,
        sharp_thr: float,
        blurry_thr: float,
        w_sharp: float,
        w_medium: float,
        w_blurry: float,
    ) -> Dict[float, List[int]]:
        """Bucket frame indices by mouth-region sharpness.

        Each input index goes into exactly one of the three buckets:

        - ``sharpness[i] >= sharp_thr`` -> ``w_sharp`` bucket
        - ``blurry_thr <= sharpness[i] < sharp_thr`` -> ``w_medium`` bucket
        - ``sharpness[i] < blurry_thr`` -> ``w_blurry`` bucket

        Returns a ``{w_value: [indices...]}`` dict with empty buckets
        omitted. The caller iterates this dict in insertion order; for
        a single-bucket input this is just one entry. Note that the
        ``w_value`` key is a float, used to dispatch a forward pass
        with that w. Stats surface the bucket counts by NAME (via
        ``_bucket_name_for``) rather than by w, so the JSON round-trip
        is clean.
        """
        buckets: Dict[float, List[int]] = {
            float(w_sharp): [],
            float(w_medium): [],
            float(w_blurry): [],
        }
        for i, s in enumerate(sharpness.tolist()):
            if s >= sharp_thr:
                key = float(w_sharp)
            elif s >= blurry_thr:
                key = float(w_medium)
            else:
                key = float(w_blurry)
            buckets[key].append(i)
        return {w: idxs for w, idxs in buckets.items() if idxs}

    def _bucket_name_for(self, w_val: float) -> str:
        """Map a bucket's w value back to its display name for stats."""
        if abs(w_val - self.w_sharp) < 1e-6:
            return "sharp"
        if abs(w_val - self.w_medium) < 1e-6:
            return "medium"
        if abs(w_val - self.w_blurry) < 1e-6:
            return "blurry"
        return f"w={w_val:.2f}"

    # -- Tier 3 helpers -------------------------------------------------

    def _mouth_mask_with_feather(
        self, H: int, W: int, device: torch.device
    ) -> torch.Tensor:
        """Fixed mouth ROI mask with optional Gaussian feather.

        Same rectangle as the ``mouth_diff`` check in
        ``_quality_check_batch``: ``y in [0.55, 0.74] * H, x in [0.30,
        0.70] * W``. With ``mouth_mask_feather_sigma <= 0`` the mask
        is a hard rectangle (useful for tests / debug); otherwise the
        boundary is feathered with a 1D Gaussian of the given sigma
        (in pixels) via two separable ``F.conv2d``s. The returned
        tensor is ``(1, 1, H, W)`` float in [0, 1] and is built on
        ``device`` so the blend below happens on the same device as
        the face tensors (which is the input device, not the model
        device, since the paste-back blend runs on the CPU path).
        """
        y0, y1 = int(H * 0.55), int(H * 0.74)
        x0, x1 = int(W * 0.30), int(W * 0.70)
        rect = torch.zeros(1, 1, H, W, device=device, dtype=torch.float32)
        rect[..., y0:y1, x0:x1] = 1.0
        sigma = self.mouth_mask_feather_sigma
        if sigma <= 0:
            return rect
        radius = max(1, int(round(3.0 * sigma)))
        k = 2 * radius + 1
        ax = torch.arange(k, dtype=torch.float32, device=device) - radius
        g1 = torch.exp(-(ax ** 2) / (2.0 * sigma * sigma))
        g1 = g1 / g1.sum()
        kx = g1.view(1, 1, 1, k)
        ky = g1.view(1, 1, k, 1)
        # reflect-pad the borders so the feather doesn't pull the mask
        # to zero at the image edge.
        tmp = torch.nn.functional.conv2d(
            torch.nn.functional.pad(rect, (radius, radius, 0, 0), mode="reflect"),
            kx,
        )
        return torch.nn.functional.conv2d(
            torch.nn.functional.pad(tmp, (0, 0, radius, radius), mode="reflect"),
            ky,
        )

    def _mouth_only_blend(
        self, batch: torch.Tensor, restored: torch.Tensor, keep: torch.Tensor
    ) -> torch.Tensor:
        """Tier 3 mouth-ROI paste-back.

        Replaces the legacy full-face ``keep * restored + (1-keep) *
        batch`` blend with a per-pixel mask: inside the mouth ROI we
        take the CodeFormer output, outside we keep the inpainter's
        output. The ``keep`` mask still gates whether the CodeFormer
        output (or the input) is used at all, regardless of region.

        Output shape: ``(B, 3, H, W)`` -- same as ``batch`` and
        ``restored``. ``keep`` is a ``(B,)`` bool tensor.
        """
        mask = self._mouth_mask_with_feather(
            batch.shape[-2], batch.shape[-1], batch.device
        )
        # mouth_paste: take restored inside the mouth, batch outside.
        mouth_paste = mask * restored + (1.0 - mask) * batch
        keep_4d = keep.float().view(-1, 1, 1, 1)
        # Final: keep the paste where keep=True, else take batch.
        return keep_4d * mouth_paste + (1.0 - keep_4d) * batch

    # -- Tier 1/2/3 bucket runner ---------------------------------------

    def _run_bucket(
        self,
        net,
        faces: torch.Tensor,
        out: torch.Tensor,
        indices: List[int],
        w: float,
        adain_enabled: bool,
        bs: int,
        param_dtype: torch.dtype,
        device: torch.device,
        mouth_only_paste: bool,
    ) -> Tuple[int, int, List[int]]:
        """Run one (w, indices) bucket of frames through CodeFormer.

        Iterates the bucket in mini-batches of size ``bs``. For each
        mini-batch:

        1. Forward through ``net`` with the bucket's ``w``.
        2. Clamp the output to ``[-1, 1]`` (rare out-of-range spikes
           on extreme inputs).
        3. Per-frame quality check; collect the indices that failed
           so the caller can re-run them as a retry pass.
        4. Blend the restored output with the input crop -- either
           the legacy full-face blend (when ``mouth_only_paste`` is
           False) or the Tier 3 mouth-ROI paste.

        Returns ``(enhanced_count, fallback_count, failed_indices)``.
        ``failed_indices`` are global frame indices, ready to feed
        into a follow-up ``_run_bucket`` call for the retry pass.
        """
        enhanced = fallback = 0
        failed: List[int] = []
        for start in range(0, len(indices), bs):
            chunk = indices[start : start + bs]
            idx_t = torch.as_tensor(chunk, device=faces.device, dtype=torch.long)
            batch = faces.index_select(0, idx_t)
            # Move to model device (only the slice, not the whole T).
            batch_dev = batch.to(device=device, dtype=param_dtype)
            restored, _logits, _lq = net(batch_dev, w=w, adain=adain_enabled)
            restored = restored.to(device=faces.device, dtype=faces.dtype)
            restored = restored.clamp(-1.0, 1.0)

            if self.fallback_enabled:
                keep = self._quality_check_batch(
                    batch.float(),
                    restored.float(),
                    sharpness_low=self.fallback_sharpness_low,
                    sharpness_high=self.fallback_sharpness_high,
                    pixel_diff=self.fallback_pixel_diff,
                    mouth_diff=self.fallback_mouth_diff,
                )
                n_fb = int((~keep).sum().item())
                fallback += n_fb
                # Collect the chunk-local positions whose keep is False;
                # the chunk is `indices[start:start+bs]`, so global
                # index = chunk[k].
                for k in range(len(chunk)):
                    if not bool(keep[k].item()):
                        failed.append(chunk[k])
            else:
                keep = torch.ones(len(chunk), dtype=torch.bool, device=faces.device)
                n_fb = 0

            if mouth_only_paste:
                out_chunk = self._mouth_only_blend(batch, restored, keep)
            else:
                mask_4d = keep.float().view(-1, 1, 1, 1)
                out_chunk = mask_4d * restored + (1.0 - mask_4d) * batch
            out_chunk = out_chunk.to(out.dtype)
            for k, fi in enumerate(chunk):
                out[fi] = out_chunk[k]
            enhanced += len(chunk) - n_fb
        return enhanced, fallback, failed
