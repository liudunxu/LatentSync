# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Merge a LoRA adapter into the base UNet to produce a regular checkpoint.

After running scripts/train_unet_lora.py, the saved adapter is a tiny
~10MB directory (peft format). Use this script to fold it back into the
base UNet weights so the result is a drop-in replacement for the original
latentsync_unet.pt.

Usage:
    python -m scripts.merge_lora \\
        --base_ckpt checkpoints/latentsync_unet.pt \\
        --adapter_dir debug/unet_lora/train_lora-.../checkpoints/checkpoint-5000 \\
        --out_ckpt debug/unet_lora/merged.pt \\
        [--push_to_hub username/latentsync-lora-finetune-name] \\
        [--hub_token_env HF_TOKEN]

After merging, the output ckpt can be used with the standard inference
script:
    python -m scripts.inference --inference_ckpt_path debug/unet_lora/merged.pt ...

HuggingFace Hub sync (optional):
    --push_to_hub REPO_ID will upload the merge directory to REPO_ID
    on HuggingFace. The directory layout is:
        <repo>/
            latentsync_unet.pt            # merged weights (drop-in)
            adapter/                       # original LoRA adapter
            unet_config.yaml
            scheduler/                     # DDIM scheduler from configs/
            README.md                     # auto-generated model card
            .gitattributes
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


HF_DEFAULT_TOKEN_ENVS = ("HF_TOKEN", "HUGGINGFACE_TOKEN")


def _hub_token() -> str:
    """Pick up HF token from env (HF_TOKEN wins, then HUGGINGFACE_TOKEN).

    Empty string returned if not set; huggingface_hub raises a clear error
    when we try to upload, which is the right behavior.
    """
    for key in HF_DEFAULT_TOKEN_ENVS:
        val = os.environ.get(key)
        if val:
            return val
    return ""


def _write_model_card(repo_dir: Path, *, repo_id: str, base_ckpt: str,
                     adapter_dir: str, unet_config_path: str,
                     lora_rank: int | None = None) -> None:
    """Generate a HuggingFace-style model card for the merged checkpoint."""
    rank_line = f"- LoRA rank: `{lora_rank}`\n" if lora_rank else ""
    readme = repo_dir / "README.md"
    readme.write_text(f"""---
license: apache-2.0
tags:
  - lipsync
  - latent-diffusion
  - lora
  - finetuned
base_model: ByteDance/LatentSync-1.5
---

# {repo_id}

LoRA-finetuned LatentSync UNet, merged back into the base checkpoint.
Built with `scripts/merge_lora.py`.

## How to use

```python
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline

pipeline = LipsyncPipeline.from_pretrained(
    "ByteDance/LatentSync-1.5",       # base components
    unet_path="{repo_id}/latentsync_unet.pt",   # this merged file
)
```

or via the Gradio Fine-tune Studio (`gradio_finetune.py`) by selecting
this repo's `latentsync_unet.pt` as the inference checkpoint.

## Files in this repo

- `latentsync_unet.pt` — drop-in replacement for the base UNet
- `adapter/`           — original peft-format LoRA adapter (if you want to keep training)
- `unet_config.yaml`   — UNet architectural config
- `scheduler/`         — DDIM scheduler

## Provenance

{rank_line}- Base checkpoint: `{base_ckpt}`
- LoRA adapter: `{adapter_dir}`
- UNet config: `{unet_config_path}`

## Caveats

This model is a community finetune of the base LatentSync checkpoint. Use
the same audio-video preprocessing as the base model. No guarantees are
made about identity preservation, side-face robustness, or lip-sync on
out-of-distribution sources.
""")


def _gitattributes_for_lfs(repo_dir: Path) -> None:
    """Mark large weight files for LFS so they're downloadable from the Hub."""
    p = repo_dir / ".gitattributes"
    p.write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n")


def _maybe_make_lfs_pointer(repo_dir: Path) -> None:
    """If `git lfs` is installed, register the .pt file as an LFS pointer.

    Falls back silently if git-lfs is missing — HF Hub will still accept
    raw .pt files (downloadable as a single ~1.3GB blob).
    """
    try:
        subprocess.run(
            ["git", "lfs", "track", "*.pt"],
            cwd=str(repo_dir), check=True, capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass


def _upload_to_hub(repo_dir: Path, repo_id: str, token: str) -> None:
    """Upload the merge directory to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit(
            "huggingface_hub is required for --push_to_hub: pip install huggingface_hub"
        )

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(repo_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload merged LatentSync UNet (LoRA finetune)",
    )
    print(f"[merge_lora] Pushed to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", required=True, help="path to latentsync_unet.pt")
    parser.add_argument("--adapter_dir", required=True, help="LoRA adapter dir from train_unet_lora.py")
    parser.add_argument("--out_ckpt", required=True, help="output merged ckpt path")
    parser.add_argument("--unet_config_path", default="configs/unet/stage2.yaml",
                        help="UNet config (must match base ckpt)")
    parser.add_argument("--push_to_hub", default=None, metavar="REPO_ID",
                        help="if set, upload the merged ckpt + adapter + model card to this HF Hub repo (e.g. 'me/latentsync-finetune-v1')")
    parser.add_argument("--hub_token_env", default=None,
                        help="override env var name for the HF token (default: HF_TOKEN, then HUGGINGFACE_TOKEN)")
    args = parser.parse_args()

    from latentsync.models.unet import UNet3DConditionModel

    print(f"[merge_lora] Loading base UNet from {args.base_ckpt} ...")
    cfg = OmegaConf.load(args.unet_config_path)
    base = UNet3DConditionModel.from_pretrained(OmegaConf.to_container(cfg.model), args.base_ckpt, device="cpu")

    print(f"[merge_lora] Loading LoRA adapter from {args.adapter_dir} ...")
    try:
        from peft import PeftModel
    except ImportError:
        raise SystemExit("peft is required: pip install peft")

    peft_model = PeftModel.from_pretrained(base, args.adapter_dir, device="cpu")

    print(f"[merge_lora] Merging adapter into base weights ...")
    merged = peft_model.merge_and_unload()

    state = {
        "global_step": 0,
        "state_dict": merged.state_dict(),
    }
    os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)
    torch.save(state, args.out_ckpt)
    print(f"[merge_lora] Saved merged checkpoint to {args.out_ckpt}")

    # ---- Optional: HuggingFace Hub sync ----
    if args.push_to_hub:
        repo_id = args.push_to_hub
        if args.hub_token_env:
            token = os.environ.get(args.hub_token_env, "")
        else:
            token = _hub_token()

        if not token:
            sys.exit(
                "HF Hub token not set. Set HF_TOKEN (or HUGGINGFACE_TOKEN) "
                "in your env, or pass --hub_token_env MY_VAR to read a custom one."
            )

        # Build a Hub-friendly folder next to the merged .pt. Layout is:
        #   <repo>/
        #       latentsync_unet.pt
        #       adapter/                       (copy of the input adapter_dir)
        #       unet_config.yaml
        #       scheduler/                     (DDIM scheduler from configs/)
        out_path = Path(args.out_ckpt).resolve()
        repo_dir = out_path.parent / f"hub_{repo_id.replace('/', '__')}"
        repo_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, repo_dir / "latentsync_unet.pt")
        # Copy original adapter next to it so users can keep training.
        adapter_dst = repo_dir / "adapter"
        if adapter_dst.exists():
            shutil.rmtree(adapter_dst)
        shutil.copytree(args.adapter_dir, adapter_dst)
        # Copy unet config + scheduler.
        shutil.copy2(args.unet_config_path, repo_dir / "unet_config.yaml")
        scheduler_src = Path("configs/scheduler")
        if scheduler_src.exists():
            scheduler_dst = repo_dir / "scheduler"
            if scheduler_dst.exists():
                shutil.rmtree(scheduler_dst)
            shutil.copytree(scheduler_src, scheduler_dst)
        # Model card + LFS markers.
        # Pull rank if adapter_config.json is sitting beside us.
        lora_rank = None
        adapter_cfg = adapter_dst / "adapter_config.json"
        if adapter_cfg.exists():
            try:
                lora_rank = json.loads(adapter_cfg.read_text()).get("r")
            except Exception:
                lora_rank = None
        _write_model_card(
            repo_dir, repo_id=repo_id,
            base_ckpt=args.base_ckpt,
            adapter_dir=args.adapter_dir,
            unet_config_path=args.unet_config_path,
            lora_rank=lora_rank,
        )
        _gitattributes_for_lfs(repo_dir)
        _maybe_make_lfs_pointer(repo_dir)

        print(f"[merge_lora] Staging hub-ready dir at {repo_dir}")
        print(f"[merge_lora] Pushing to HF Hub repo {repo_id} ...")
        _upload_to_hub(repo_dir, repo_id=repo_id, token=token)

    print(f"[merge_lora] Done. Use it with the standard inference command.")


if __name__ == "__main__":
    main()
