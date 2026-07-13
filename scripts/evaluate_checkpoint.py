"""End-to-end evaluation script for a trained UNet checkpoint.

Generates inference videos for a list of test samples, then runs the
per-metric evaluators from §13 and writes a JSON + HTML report.

Usage:
    python -m scripts.evaluate_checkpoint \\
        --ckpt_path debug/unet/checkpoint-5000.pt \\
        --unet_config configs/unet/stage2.yaml \\
        --test_fileslist data/test/fileslist.txt \\
        --test_videos_dir data/test/videos \\
        --out_dir debug/eval_results/ckpt-5000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_inference(
    ckpt_path: Path,
    unet_config: Path,
    video_path: Path,
    audio_path: Path,
    out_path: Path,
    inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> bool:
    """Run inference for one (video, audio) pair via the existing script."""
    cmd = [
        "python", "-m", "scripts.inference",
        "--unet_config_path", str(unet_config),
        "--inference_ckpt_path", str(ckpt_path),
        "--video_path", str(video_path),
        "--audio_path", str(audio_path),
        "--video_out_path", str(out_path),
        "--inference_steps", str(inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--seed", str(seed),
        "--temp_dir", "temp",
    ]
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[inference] FAILED for {video_path}: {e.stderr.decode()[:200]}")
        return False


def _per_video_metrics(
    real_path: Path, gen_path: Path, do_sync: bool, do_hyperiqa: bool
) -> Dict[str, float]:
    """Compute all per-metric values for a single (real, gen) pair.

    Returns a flat dict with avg_lpips, avg_sync_conf, avg_hyperiqa etc.
    """
    out: Dict[str, float] = {}
    try:
        from decord import VideoReader
        import numpy as np
        import cv2

        vr_r = VideoReader(str(real_path))
        vr_g = VideoReader(str(gen_path))
        n = min(len(vr_r), len(vr_g), 16)
        if n < 2:
            return out
        rs = [vr_r[i].asnumpy() for i in range(n)]
        gs = [vr_g[i].asnumpy() for i in range(n)]

        # sharpness (Laplacian)
        laps = [cv2.Laplacian(cv2.cvtColor(g, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for g in gs]
        out["sharpness"] = float(np.mean(laps))

        # LPIPS (only if hyperiqa or lpips deps available)
        try:
            import lpips
            import torch
            from torchvision import transforms

            loss_fn = lpips.LPIPS(net="vgg")
            t = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
            losses = []
            for r, g in zip(rs, gs):
                h = r.shape[0] // 2
                rt = t(r)[:, h:, :].unsqueeze(0)
                gt = t(g)[:, h:, :].unsqueeze(0)
                with torch.no_grad():
                    losses.append(loss_fn(rt, gt).item())
            out["lpips"] = float(np.mean(losses))
        except Exception as e:
            print(f"[lpips] skipped: {e}")

        # SyncNet confidence
        if do_sync:
            try:
                from eval.syncnet import SyncNetEval
                from eval.syncnet_detect import SyncNetDetector
                from eval.eval_sync_conf import syncnet_eval
                import torch as _t
                device = "cuda" if _t.cuda.is_available() else "cpu"
                if (REPO_ROOT / "checkpoints/auxiliary/syncnet_v2.model").exists():
                    se = SyncNetEval(device=device)
                    se.loadParameters("checkpoints/auxiliary/syncnet_v2.model")
                    sd = SyncNetDetector(device=device, detect_results_dir="detect_results_eval")
                    _, conf = syncnet_eval(se, sd, str(gen_path), "temp")
                    out["sync_conf"] = float(conf)
            except Exception as e:
                print(f"[sync_conf] skipped: {e}")

        # HyperIQA on generated
        if do_hyperiqa:
            try:
                from eval.hyper_iqa import HyperNet, TargetNet
                import torch as _t
                import torchvision
                from torchvision import transforms
                device = "cuda" if _t.cuda.is_available() else "cpu"
                ckpt = REPO_ROOT / "checkpoints/auxiliary/koniq_pretrained.pkl"
                if ckpt.exists():
                    model_hyper = HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)
                    model_hyper.load_state_dict(_t.load(str(ckpt), map_location=device, weights_only=True))
                    model_hyper.eval()
                    tf = transforms.Compose([
                        transforms.CenterCrop(224),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ])
                    sampled = gs[::max(1, len(gs) // 3)][:3]
                    t = _t.from_numpy(__import__("numpy").stack(sampled)).permute(0, 3, 1, 2).float() / 255.0
                    with _t.no_grad():
                        paras = model_hyper(tf(t).to(device))
                    model_target = TargetNet(paras).to(device)
                    for p_ in model_target.parameters():
                        p_.requires_grad = False
                    score = model_target(paras["target_in_vec"]).mean().item()
                    out["hyperiqa"] = float(score)
            except Exception as e:
                print(f"[hyperiqa] skipped: {e}")

        return out
    except Exception as e:
        print(f"[per_video_metrics] {real_path}: {e}")
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--unet_config", default="configs/unet/stage2.yaml")
    parser.add_argument("--test_fileslist", required=True,
                        help="each line: 'video_path<TAB>audio_path' or 'video_path,audio_path'")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_inference", action="store_true",
                        help="reuse existing generated videos in out_dir/fake_videos/")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fake_dir = out_dir / "fake_videos"
    fake_dir.mkdir(parents=True, exist_ok=True)

    test_list = Path(args.test_fileslist).read_text().strip().splitlines()
    test_list = test_list[: args.num_samples]

    # ---- 1. Inference ----
    print(f"=== Step 1: Inference on {len(test_list)} samples ===")
    ckpt = Path(args.ckpt_path)
    cfg = Path(args.unet_config)
    succeeded = []
    for i, line in enumerate(test_list):
        parts = re.split(r"[\t,]", line, maxsplit=1)
        if len(parts) != 2:
            print(f"  skip bad line {i}: {line!r}")
            continue
        vp, ap = Path(parts[0].strip()), Path(parts[1].strip())
        out_mp4 = fake_dir / f"{vp.stem}.mp4"
        if args.skip_inference and out_mp4.exists():
            print(f"  [{i+1}/{len(test_list)}] reuse {out_mp4.name}")
        else:
            print(f"  [{i+1}/{len(test_list)}] {vp.name} ...")
            if _run_inference(ckpt, cfg, vp, ap, out_mp4, args.inference_steps, args.guidance_scale, args.seed):
                succeeded.append((vp, ap, out_mp4))
            else:
                continue
        if not args.skip_inference:
            succeeded.append((vp, ap, out_mp4))

    # ---- 2. Per-metric evaluation ----
    print(f"=== Step 2: Metrics on {len(succeeded)} samples ===")
    all_metrics: List[Dict[str, float]] = []
    for real_v, _, gen_v in succeeded:
        m = _per_video_metrics(real_v, gen_v, do_sync=True, do_hyperiqa=True)
        if m:
            all_metrics.append(m)
            print(f"  {real_v.name}: {m}")

    # ---- 3. Aggregate ----
    def _avg(key: str) -> Optional[float]:
        vals = [m[key] for m in all_metrics if key in m]
        return float(sum(vals) / len(vals)) if vals else None

    report = {
        "ckpt_path": str(ckpt),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": len(succeeded),
        "single_frame": {
            "avg_sharpness": _avg("sharpness"),
            "avg_lpips":     _avg("lpips"),
            "avg_hyperiqa":  _avg("hyperiqa"),
        },
        "temporal": {
            # FVD not implemented in this script (use eval/eval_fvd.py for that)
            "fvd": None,
        },
        "semantic": {
            "avg_sync_conf": _avg("sync_conf"),
        },
    }
    json_path = out_dir / "evaluation_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"=== JSON report: {json_path} ===")
    print(json.dumps(report, indent=2))

    # ---- 4. HTML report (optional) ----
    real_dir = succeeded[0][0].parent if succeeded else ""
    html_cmd = [
        "python", "-m", "eval.generate_report",
        "--eval_json", str(json_path),
        "--ckpt_name", ckpt.name,
        "--real_dir", str(real_dir),
        "--fake_dir", str(fake_dir),
        "--out", str(out_dir / "report.html"),
    ]
    try:
        subprocess.run(html_cmd, cwd=REPO_ROOT, check=False)
    except Exception as e:
        print(f"[html report] skipped: {e}")


if __name__ == "__main__":
    import re
    main()
