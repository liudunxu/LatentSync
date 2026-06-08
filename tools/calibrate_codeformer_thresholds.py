"""Calibrate the CodeFormer Tier-1 sharpness bucket thresholds.

Background
----------
Tier 1 of the short-drama-tuned CodeFormer bucketed frame indices
by mouth-region Laplacian variance into "sharp" / "medium" / "blurry"
groups, each running through CodeFormer with a different fidelity
weight (``w``). The two thresholds that decide bucketing
(``LATENTSYNC_CODEFORMER_SHARP_THRESHOLD`` and
``LATENTSYNC_CODEFORMER_BLURRY_THRESHOLD``) are *estimated* defaults
in the code (0.05 and 0.01) and need to be tuned against real data.

This script scans a directory of pre-aligned 512x512 face crops
(typically produced by running the lipsync pipeline once on a
representative short-drama sample) and prints:

  * A distribution summary (min, p05, p25, p50, p75, p95, max)
  * A histogram-style bar chart
  * Suggested values for ``sharp_threshold`` (p40) and
    ``blurry_threshold`` (p10), which the operator can then write
    back to ``Settings`` via env vars.

The script only uses numpy and torch; no GPU, no model load.

Usage
-----

    # Basic: print distribution + suggested thresholds
    python -m tools.calibrate_codeformer_thresholds /path/to/face_crops

    # Custom quantiles
    python -m tools.calibrate_codeformer_thresholds /path/to/face_crops \\
        --sharp-quantile 0.5 --blurry-quantile 0.1

    # Recursively find image files
    python -m tools.calibrate_codeformer_thresholds /path/to/crops \\
        --recursive

Expected file extensions: .png, .jpg, .jpeg, .npy, .pt, .pth.
Non-image files are skipped silently.

The script is read-only -- it doesn't write anything to disk.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch


def _mouth_roi_laplacian(image: torch.Tensor) -> float:
    """Per-image Laplacian variance on the mouth ROI, matching
    ``CodeFormerRestorer._mouth_sharpness_batch``.

    ``image`` is ``(3, H, W)`` in any value range (the metric is
    range-invariant because Laplacian variance is scale-agnostic only
    up to a multiplicative constant -- here we just want the
    distribution shape, not the absolute units). Returns a Python
    float.
    """
    H, W = image.shape[-2:]
    y0, y1 = int(H * 0.55), int(H * 0.74)
    x0, x1 = int(W * 0.30), int(W * 0.70)
    mouth = image[..., y0:y1, x0:x1].float()
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    ).view(1, 1, 3, 3)
    gray = mouth.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, h, w)
    lap = torch.nn.functional.conv2d(gray, kernel, padding=1)
    return float(lap.pow(2).mean().item())


def _iter_image_files(root: Path, recursive: bool) -> Iterable[Path]:
    """Yield image files under ``root``. Filters by extension."""
    valid_ext = {".png", ".jpg", ".jpeg", ".npy", ".pt", ".pth"}
    if recursive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in valid_ext:
                yield path
    else:
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix.lower() in valid_ext:
                yield path


def _load_image(path: Path) -> Optional[torch.Tensor]:
    """Load a face crop as ``(3, H, W)`` torch tensor. Returns None on
    failure (skipped silently by the caller)."""
    try:
        suffix = path.suffix.lower()
        if suffix in {".pt", ".pth"}:
            tensor = torch.load(path, map_location="cpu", weights_only=True)
        elif suffix == ".npy":
            arr = np.load(path)
            tensor = torch.as_tensor(arr)
        else:
            # PNG/JPG via torchvision or PIL.
            try:
                from torchvision.io import read_image

                tensor = read_image(str(path))  # uint8 (3, H, W)
            except Exception:
                from PIL import Image

                img = Image.open(path).convert("RGB")
                tensor = torch.as_tensor(np.array(img)).permute(2, 0, 1)
        # Normalize to 3 channels
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        if tensor.shape[0] > 3:
            tensor = tensor[:3]
        if tensor.shape[0] < 3:
            return None
        return tensor
    except Exception as exc:  # noqa: BLE001
        print(f"  skip {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def _summary_stats(values: List[float]) -> str:
    """Format a distribution summary."""
    if not values:
        return "no samples"
    arr = np.asarray(values)
    qs = np.quantile(arr, [0.0, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 1.0])
    return (
        f"  n    = {len(values)}\n"
        f"  min   = {qs[0]:.5f}\n"
        f"  p05   = {qs[1]:.5f}\n"
        f"  p10   = {qs[2]:.5f}    <-- suggested blurry_threshold (if --blurry-quantile 0.10)\n"
        f"  p25   = {qs[3]:.5f}\n"
        f"  p40   = {qs[5]:.5f}    <-- suggested sharp_threshold (if --sharp-quantile 0.40)\n"
        f"  p50   = {qs[6]:.5f}\n"
        f"  p60   = {qs[7]:.5f}\n"
        f"  p75   = {qs[8]:.5f}\n"
        f"  p95   = {qs[9]:.5f}\n"
        f"  max   = {qs[10]:.5f}"
    )


def _ascii_histogram(values: List[float], n_buckets: int = 30, width: int = 50) -> str:
    """A tiny ASCII histogram, log-spaced if the dynamic range is wide."""
    if not values:
        return ""
    arr = np.asarray(values)
    # Use log-spaced bins if the max/min ratio is > 100; otherwise linear.
    if arr.max() > 0 and arr.max() / max(arr.min(), 1e-9) > 100:
        bins = np.logspace(np.log10(max(arr.min(), 1e-9)), np.log10(arr.max()), n_buckets + 1)
    else:
        bins = np.linspace(arr.min(), arr.max(), n_buckets + 1)
    counts, edges = np.histogram(arr, bins=bins)
    peak = int(counts.max())
    if peak == 0:
        return ""
    lines = []
    for i, c in enumerate(counts):
        bar_len = int(round((c / peak) * width))
        lines.append(
            f"  [{edges[i]:.5f}..{edges[i+1]:.5f}] "
            f"{'#' * bar_len:<{width}} {int(c)}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate CodeFormer Tier-1 mouth-region sharpness bucket thresholds."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Directory of pre-aligned 512x512 face crops (one per file).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories.",
    )
    parser.add_argument(
        "--sharp-quantile",
        type=float,
        default=0.40,
        help="Quantile at which to set the sharp threshold (default 0.40).",
    )
    parser.add_argument(
        "--blurry-quantile",
        type=float,
        default=0.10,
        help="Quantile at which to set the blurry threshold (default 0.10).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit the number of files processed (0 = no limit).",
    )
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"error: {args.root} is not a directory", file=sys.stderr)
        return 2

    print(f"Scanning {args.root} (recursive={args.recursive})...")
    files = list(_iter_image_files(args.root, args.recursive))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        print(f"error: no .png/.jpg/.npy/.pt files found under {args.root}", file=sys.stderr)
        return 2
    print(f"Found {len(files)} candidate files.")

    sharpnesses: List[float] = []
    for path in files:
        img = _load_image(path)
        if img is None:
            continue
        try:
            sharpnesses.append(_mouth_roi_laplacian(img))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if not sharpnesses:
        print("error: no usable crops found", file=sys.stderr)
        return 2

    print()
    print("Mouth-region Laplacian variance distribution:")
    print(_summary_stats(sharpnesses))
    print()
    print("Histogram:")
    print(_ascii_histogram(sharpnesses))
    print()

    arr = np.asarray(sharpnesses)
    sharp_q = float(args.sharp_quantile)
    blurry_q = float(args.blurry_quantile)
    if not 0.0 <= blurry_q < sharp_q <= 1.0:
        print(
            f"error: invalid quantiles -- need 0 <= {blurry_q} < {sharp_q} <= 1",
            file=sys.stderr,
        )
        return 2
    sharp_threshold = float(np.quantile(arr, sharp_q))
    blurry_threshold = float(np.quantile(arr, blurry_q))

    print(f"Suggested thresholds (quantiles {blurry_q:.2f} / {sharp_q:.2f}):")
    print(f"  LATENTSYNC_CODEFORMER_BLURRY_THRESHOLD = {blurry_threshold:.5f}")
    print(f"  LATENTSYNC_CODEFORMER_SHARP_THRESHOLD  = {sharp_threshold:.5f}")
    print()
    print("Write back to api.py:Settings (or set the env vars in your")
    print("server config) and restart. Re-run this script on a fresh")
    print("sample to confirm the new distribution buckets cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
