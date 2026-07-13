#!/bin/bash

# LoRA fine-tuning for LatentSync UNet.
# Drop-in for train_unet.sh, but uses scripts/train_unet_lora.py which
# injects peft LoRA adapters into the attention layers.

torchrun --nnodes=1 --nproc_per_node=1 --master_port=25680 -m scripts.train_unet_lora \
    --unet_config_path "configs/unet/stage2_lora.yaml"
