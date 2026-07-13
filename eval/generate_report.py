"""Generate a self-contained HTML report summarising one (or more)
checkpoint evaluations.

Combines the per-metric numbers (§13 evaluation) with a few embedded
sample videos (real vs generated) so the result can be shared with
people who don't have the project set up.

Usage:
    python -m eval.generate_report \\
        --eval_json debug/eval_results/ckpt-5000.json \\
        --real_dir data/test/real_videos \\
        --fake_dir debug/eval_results/ckpt-5000/fake_videos \\
        --ckpt_name "fine-tuned v1" \\
        --out debug/eval_results/ckpt-5000/report.html
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List


def _file_to_data_url(p: Path) -> str:
    """Embed a small video/image as a base64 data URL so the HTML is self-contained."""
    if not p.exists() or p.stat().st_size > 50 * 1024 * 1024:
        return ""
    mime, _ = mimetypes.guess_type(str(p))
    if mime is None:
        mime = "application/octet-stream"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LatentSync Eval Report — {ckpt_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          max-width: 1000px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1 {{ border-bottom: 2px solid #444; padding-bottom: 4px; }}
  h2 {{ margin-top: 2em; color: #555; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #f5f5f5; }}
  .good {{ color: #0a7c2f; font-weight: 600; }}
  .bad  {{ color: #c0392b; font-weight: 600; }}
  .neutral {{ color: #666; }}
  .video-pair {{ display: flex; gap: 1em; align-items: flex-start; margin: 1em 0; }}
  .video-pair video {{ width: 320px; height: 180px; background: #000; }}
  .label {{ font-size: 0.9em; color: #666; margin-bottom: 4px; }}
  code {{ background: #f5f5f5; padding: 1px 4px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>LatentSync Eval Report</h1>
<p><b>Checkpoint:</b> <code>{ckpt_name}</code><br>
<b>Generated:</b> {ts}</p>

<h2>Metrics</h2>
<table>
  <tr><th>Metric</th><th>Value</th><th>Direction</th><th>Status</th></tr>
  {metric_rows}
</table>

<h2>Sample Comparisons (real vs generated)</h2>
{video_pairs_html}

</body>
</html>
"""


def _format_metric_row(name: str, value: Any, target: str, status: str) -> str:
    cls = "good" if status == "good" else ("bad" if status == "bad" else "neutral")
    return f"<tr><td>{name}</td><td>{value}</td><td>{target}</td><td class='{cls}'>{status}</td></tr>"


def _classify(name: str, value: float) -> str:
    """Heuristic thresholding for the Status column.

    These are rough targets; users should treat them as starting
    points, not absolute truths.
    """
    if value is None:
        return "n/a"
    targets = {
        "HyperIQA": ("higher", 60),
        "LPIPS":    ("lower",  0.15),
        "PSNR":     ("higher", 30),
        "SSIM":     ("higher", 0.75),
        "FVD":      ("lower",  200),
        "TREPA":    ("lower",  0.01),
        "Sync_conf":("higher", 7),
        "LMD":      ("lower",  0.4),
        "Face_sim": ("higher", 0.8),
    }
    direction, threshold = targets.get(name, ("higher", 0))
    if direction == "higher":
        return "good" if value >= threshold else "bad"
    return "good" if value <= threshold else "bad"


def _build_video_pairs(real_dir: str, fake_dir: str, max_pairs: int = 4) -> str:
    rd = Path(real_dir)
    fd = Path(fake_dir)
    if not rd.exists() or not fd.exists():
        return "<p><i>No sample videos provided.</i></p>"
    pairs: List[tuple[Path, Path]] = []
    for real in sorted(rd.glob("*.mp4")):
        # assume fake has the same stem (or stem + suffix)
        candidates = [fd / real.name, fd / (real.stem + ".mp4"), fd / real.name.replace(".mp4", ".mp4")]
        for c in candidates:
            if c.exists():
                pairs.append((real, c))
                break
        if len(pairs) >= max_pairs:
            break

    if not pairs:
        return "<p><i>No matching sample videos found.</i></p>"

    blocks: List[str] = []
    for real, fake in pairs:
        real_url = _file_to_data_url(real)
        fake_url = _file_to_data_url(fake)
        real_tag = f'<video controls src="{real_url}"></video>' if real_url else "<i>(skipped: too large)</i>"
        fake_tag = f'<video controls src="{fake_url}"></video>' if fake_url else "<i>(skipped: too large)</i>"
        blocks.append(
            f"<div class='video-pair'>"
            f"<div><div class='label'>Real: {real.name}</div>{real_tag}</div>"
            f"<div><div class='label'>Generated: {fake.name}</div>{fake_tag}</div>"
            f"</div>"
        )
    return "\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", required=True,
                        help="JSON file produced by scripts/evaluate_checkpoint.py")
    parser.add_argument("--ckpt_name", default=None, help="display name; default = filename")
    parser.add_argument("--real_dir", default=None, help="dir of real videos for sample pairs")
    parser.add_argument("--fake_dir", default=None, help="dir of generated videos")
    parser.add_argument("--max_pairs", type=int, default=4)
    parser.add_argument("--out", required=True, help="output HTML path")
    args = parser.parse_args()

    with open(args.eval_json) as f:
        metrics = json.load(f)

    ckpt_name = args.ckpt_name or Path(args.eval_json).stem
    ts = metrics.get("timestamp", "now")
    rows: List[str] = []
    for section, key, target in [
        ("single_frame", "avg_sharpness", "higher (sharper)"),
        ("single_frame", "avg_hyperiqa", "higher (better)"),
        ("single_frame", "avg_lpips",    "lower (closer to GT)"),
        ("temporal",     "fvd",          "lower (more realistic)"),
        ("temporal",     "avg_trepa",    "lower (temporally aligned)"),
        ("semantic",     "avg_sync_conf","higher (lip-sync)"),
        ("semantic",     "avg_lmd",      "lower (landmark)"),
        ("semantic",     "avg_face_sim", "higher (identity)"),
    ]:
        section_data = metrics.get(section, {})
        val = section_data.get(key)
        if val is None:
            continue
        nice_name = {
            "avg_sharpness": "Sharpness (Laplacian)",
            "avg_hyperiqa": "HyperIQA",
            "avg_lpips": "LPIPS",
            "fvd": "FVD",
            "avg_trepa": "TREPA",
            "avg_sync_conf": "Sync_conf",
            "avg_lmd": "LMD",
            "avg_face_sim": "Face_sim",
        }.get(key, key)
        rows.append(_format_metric_row(nice_name, round(val, 4), target, _classify(nice_name, val)))

    video_pairs = _build_video_pairs(args.real_dir or "", args.fake_dir or "", args.max_pairs)

    html = HTML_TEMPLATE.format(
        ckpt_name=ckpt_name,
        ts=ts,
        metric_rows="\n  ".join(rows),
        video_pairs_html=video_pairs,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
