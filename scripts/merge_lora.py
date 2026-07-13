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
        --out_ckpt debug/unet_lora/merged.pt

After merging, the output ckpt can be used with the standard inference
script:
    python -m scripts.inference --inference_ckpt_path debug/unet_lora/merged.pt ...
"""

import argparse
import os
import torch
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", required=True, help="path to latentsync_unet.pt")
    parser.add_argument("--adapter_dir", required=True, help="LoRA adapter dir from train_unet_lora.py")
    parser.add_argument("--out_ckpt", required=True, help="output merged ckpt path")
    parser.add_argument("--unet_config_path", default="configs/unet/stage2.yaml",
                        help="UNet config (must match base ckpt)")
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
    print(f"[merge_lora] Done. Use it with the standard inference command.")


if __name__ == "__main__":
    main()
