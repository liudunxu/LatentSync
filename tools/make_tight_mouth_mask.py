"""Generate a tight mouth-only mask for the lipsync inpainter.

Background story
----------------
The default `mask.png` in this repo is a "U-shaped" mask where the
inpainter is asked to regenerate the eyes, nose, cheeks, mouth and most
of the face — keeping only a chin/jaw cradle. That is way too aggressive:
the inpainter drifts identity/pose, and the temporal EMA across face
crops ends up carrying previous-frame content into the cheeks/jaw (the
"ghost" the user observed in @~/Downloads/a8.png).

This script produces a tight elliptical mask that only covers the
mouth region (lips + a small chin strip below the lower lip). All
cheeks, nose, eyes, forehead and most of the chin are preserved from
the original frame — which is what we want for identity/expression
preservation.

Geometry
--------
On the 512x512 aligned face, the canonical mouth ROI used elsewhere in
the pipeline (`_mouth_roi`, `_mouth_core_mask`) sits roughly at:

  - y: 0.55..0.74  (mouth aperture / upper lip)
  - x: 0.30..0.70  (mouth width)

We extend the mask slightly (y 0.50..0.80) to also cover the chin
swing below the lower lip, which is where the model tends to smear
when the mouth opens wide.

The output is a 256x256 PNG (matching the existing mask.png / mask2/3/4
in latentsync/utils/) with a soft Gaussian edge so that
`load_fixed_mask` resizes it to whatever resolution the pipeline asks
for, while keeping the boundary feathered.
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def make_tight_mouth_mask(
    out_path: str,
    size: int = 256,
    # Ellipse center / radius in normalized [0, 1] coordinates
    cx_norm: float = 0.50,
    cy_norm: float = 0.655,
    rx_norm: float = 0.225,
    ry_norm: float = 0.155,
    # Inner keep region (fully white) in normalized coords
    inner_cx_norm: float = 0.50,
    inner_cy_norm: float = 0.655,
    inner_rx_norm: float = 0.190,
    inner_ry_norm: float = 0.125,
    # Gaussian feather sigma in pixels of the output image
    feather_sigma_px: float = 6.0,
    # White = 1.0 (inpaint), Black = 0.0 (preserve)
) -> np.ndarray:
    """Render a soft elliptical mask covering just the mouth."""
    H = W = size
    # Outer ellipse: full white (inpaint) inside, fades to black outside
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = cx_norm * W, cy_norm * H
    rx, ry = rx_norm * W, ry_norm * H
    outer = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    # Hard step: 1.0 inside ellipse, 0.0 outside
    mask = (outer <= 1.0).astype(np.float32)
    # Feather the boundary with a Gaussian blur, then re-normalize so
    # the inner region stays at 1.0 and the outer is at 0.0.
    if feather_sigma_px > 0:
        ksize = int(2 * round(3 * feather_sigma_px) + 1)
        feathered = cv2.GaussianBlur(mask, (ksize, ksize), feather_sigma_px)
        # Re-stretch so the inner core hits 1.0 and the outer hits 0.0
        inner_cx, inner_cy = inner_cx_norm * W, inner_cy_norm * H
        inner_rx, inner_ry = inner_rx_norm * W, inner_ry_norm * H
        inner = ((xx - inner_cx) / inner_rx) ** 2 + ((yy - inner_cy) / inner_ry) ** 2
        inner_keep = (inner <= 1.0)
        # Wherever the inner ellipse is fully inside, force to 1.0
        feathered = np.where(inner_keep, 1.0, feathered)
        # The Gaussian blur can leave a faint residue outside the outer
        # ellipse; threshold it down to 0 so the cheeks stay 100% original.
        feathered = np.where(outer > 1.0, 0.0, feathered)
        mask = feathered
    # Save as 8-bit grayscale PNG (cv2.imwrite expects BGR for color,
    # but for a single-channel mask it just writes it as grayscale)
    out = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, out)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="latentsync/utils/mask5.png",
        help="output PNG path",
    )
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--feather", type=float, default=6.0)
    args = ap.parse_args()
    mask = make_tight_mouth_mask(args.out, size=args.size, feather_sigma_px=args.feather)
    # Quick stats
    inpaint_frac = float((mask > 0.5).mean())
    print(
        f"Wrote {args.out} (size={args.size}x{args.size}, "
        f"inpaint_fraction={inpaint_frac:.3f})"
    )


if __name__ == "__main__":
    main()
