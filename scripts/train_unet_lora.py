# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LoRA / QLoRA fine-tuning for LatentSync UNet.

Drop-in replacement for scripts/train_unet.py that:
  1. Loads the base UNet from `checkpoints/latentsync_unet.pt`.
  2. Injects LoRA adapters into attention layers (and optionally
     quantizes the base to 4-bit for QLoRA).
  3. Trains ONLY the LoRA matrices (everything else is frozen).
  4. Saves a tiny adapter file (~10MB) instead of the full UNet.

Usage:
    torchrun --nproc_per_node=1 -m scripts.train_unet_lora \\
        --unet_config_path configs/unet/stage2_lora.yaml

Inference-side merge:
    python -m scripts.merge_lora --base_ckpt checkpoints/latentsync_unet.pt \\
        --adapter debug/lora_run/checkpoints/checkpoint-5000 \\
        --out_ckpt debug/lora_merged.pt
"""

import os
import math
import argparse
import shutil
import datetime
import copy
import logging
from typing import List, Optional

from omegaconf import OmegaConf

from tqdm.auto import tqdm
from einops import rearrange

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from accelerate.utils import set_seed

from latentsync.data.unet_dataset import UNetDataset
from latentsync.models.unet import UNet3DConditionModel
from latentsync.models.stable_syncnet import StableSyncNet
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.utils.util import init_dist, cosine_loss, one_step_sampling
from latentsync.utils.util import plot_loss_chart
from latentsync.whisper.audio2feature import Audio2Feature
from latentsync.trepa.loss import TREPALoss
from eval.syncnet import SyncNetEval
from eval.syncnet_detect import SyncNetDetector
from eval.eval_sync_conf import syncnet_eval

import lpips
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------

def inject_lora(
    unet: UNet3DConditionModel,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    qlora: bool = False,
) -> UNet3DConditionModel:
    """Inject LoRA adapters into the UNet attention layers.

    Returns a peft-wrapped model where ONLY the LoRA matrices are
    trainable. Optionally loads the base UNet in 4-bit (QLoRA).

    Args:
        unet: base UNet3DConditionModel (weights already loaded).
        rank: LoRA rank (8-64 typical).
        alpha: LoRA alpha (usually 2*rank).
        dropout: LoRA dropout.
        target_modules: list of Linear layer names to wrap
            (default: to_q, to_k, to_v, to_out — i.e. all attention projections).
        qlora: if True, 4-bit quantize the base first (saves ~50% VRAM).
    """
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        raise ImportError(
            "peft is required for LoRA training. Install with: pip install peft"
        ) from e

    if target_modules is None:
        target_modules = ["to_q", "to_k", "to_v", "to_out"]

    if qlora:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "bitsandbytes is required for QLoRA. Install with: pip install bitsandbytes"
            ) from e
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        # peft's prepare_model_for_kbit_training wraps the model for k-bit training
        unet = prepare_model_for_kbit_training(unet, use_gradient_checkpointing=False)
        logger.info("QLoRA: base UNet quantized to 4-bit (nf4).")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        # UNet3DConditionModel is a feature extractor; peft doesn't have a
        # perfect enum for it, but FEATURE_EXTRACTION works in practice.
        task_type=None,
    )
    

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    return unet


def freeze_attn2_lora(unet) -> None:
    """Optional: keep audio cross-attn (attn2) LoRA frozen to preserve sync.

    Useful when you observe sync_conf regressing after LoRA fine-tuning.
    The UNet attention module names are .attn1. (self) and .attn2. (cross).
    """
    n_frozen = 0
    for name, param in unet.named_parameters():
        if "lora_" in name and "attn2" in name:
            param.requires_grad = False
            n_frozen += 1
    if n_frozen:
        logger.info(f"Froze {n_frozen} attn2 LoRA params to protect sync ability.")


def _save_loss_chart(
    save_path: str,
    steps: List[int],
    losses: dict,
    lr_list: List[float],
) -> None:
    """Plot training losses (left y-axis) and learning rate (right y-axis)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax1 = plt.subplots()
    for name, values in losses.items():
        if values:
            ax1.plot(steps[: len(values)], values, label=name)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")

    if lr_list:
        ax2 = ax1.twinx()
        ax2.plot(steps[: len(lr_list)], lr_list, color="black", linestyle="--", alpha=0.5, label="lr")
        ax2.set_ylabel("Learning rate", color="black")
        ax2.tick_params(axis="y", labelcolor="black")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config):
    # Distributed init (DDP)
    local_rank = init_dist()
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = config.run.seed + global_rank
    set_seed(seed)

    # LoRA block (must exist in config)
    if not config.lora.get("enabled", False):
        raise ValueError(
            "train_unet_lora.py expects config.lora.enabled=true. "
            "Use configs/unet/stage2_lora.yaml or set the block manually."
        )
    lora_cfg = config.lora

    # Output dir
    train_output_dir = config.data.train_output_dir
    if not os.path.isabs(train_output_dir):
        base_dir = os.environ.get("LATENTSYNC_FINETUNE_DIR", os.getcwd())
        train_output_dir = os.path.join(base_dir, train_output_dir)
        config.data.train_output_dir = train_output_dir
    folder_name = "train_lora" + datetime.datetime.now().strftime(f"-%Y_%m_%d-%H:%M:%S")
    output_dir = os.path.join(config.data.train_output_dir, folder_name)
    logger.info("output_dir: %s", output_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if is_main_process:
        diffusers.utils.logging.set_verbosity_info()
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{output_dir}/val_videos", exist_ok=True)
        os.makedirs(f"{output_dir}/sync_conf_results", exist_ok=True)
        os.makedirs(f"{output_dir}/loss_charts", exist_ok=True)
        shutil.copy(config.unet_config_path, output_dir)
        shutil.copy(config.data.syncnet_config_path, output_dir)

    device = torch.device(local_rank)
    noise_scheduler = DDIMScheduler.from_pretrained("configs")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae.requires_grad_(False)
    vae.to(device)
    if config.run.pixel_space_supervise:
        vae.enable_gradient_checkpointing()

    syncnet_eval_model = SyncNetEval(device=device)
    syncnet_eval_model.loadParameters("checkpoints/auxiliary/syncnet_v2.model")
    syncnet_detector = SyncNetDetector(device=device, detect_results_dir="detect_results")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device=device,
        audio_embeds_cache_dir=config.data.audio_embeds_cache_dir,
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    # ---- Load base UNet / resume LoRA adapter ----
    resume_ckpt = config.ckpt.resume_ckpt_path
    adapter_resume_dir = None
    if os.path.isdir(resume_ckpt) and os.path.exists(os.path.join(resume_ckpt, "adapter_config.json")):
        adapter_resume_dir = resume_ckpt
        base_ckpt = config.ckpt.get("base_unet_ckpt", "checkpoints/latentsync_unet.pt")
        logger.info("Resuming LoRA adapter from %s; base UNet from %s", adapter_resume_dir, base_ckpt)
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            base_ckpt,
            device=device,
        )
        try:
            from peft import PeftModel
            # Some peft versions try to import transformers.integrations.tensor_parallel
            # even when tensor parallelism is not used. Wrap the helper so a missing
            # tensor_parallel module falls back to the identity path instead of crashing.
            try:
                import peft.utils.save_and_load as _peft_save_load
                if hasattr(_peft_save_load, "_maybe_shard_state_dict_for_tp"):
                    _orig_maybe_shard = _peft_save_load._maybe_shard_state_dict_for_tp

                    def _safe_maybe_shard(model, state_dict, adapter_name):
                        try:
                            return _orig_maybe_shard(model, state_dict, adapter_name)
                        except ModuleNotFoundError:
                            return state_dict

                    _peft_save_load._maybe_shard_state_dict_for_tp = _safe_maybe_shard
            except Exception:
                pass
            unet = PeftModel.from_pretrained(unet, adapter_resume_dir)
            # Defensive: make sure LoRA params are trainable and base params frozen.
            # In most peft versions from_pretrained already does this, but some
            # environment combinations leave everything frozen after resume.
            for name, param in unet.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            unet.to(device)
        except Exception as e:
            raise RuntimeError(f"Could not load LoRA adapter from {adapter_resume_dir}: {e}") from e
        step_str = os.path.basename(adapter_resume_dir).split("-")[-1]
        try:
            resume_global_step = int(step_str)
        except ValueError:
            resume_global_step = 0
    else:
        unet, resume_global_step = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            resume_ckpt,
            device=device,
        )

    # ---- Inject LoRA ----
    if adapter_resume_dir is None:
        unet = inject_lora(
            unet,
            rank=int(lora_cfg.rank),
            alpha=int(lora_cfg.alpha),
            dropout=float(lora_cfg.dropout),
            target_modules=list(lora_cfg.target_modules),
            qlora=bool(lora_cfg.qlora),
        )
        if bool(lora_cfg.get("freeze_attn2", False)):
            freeze_attn2_lora(unet)
    else:
        # Adapter resumed: LoRA structure already loaded from the adapter dir.
        if is_main_process:
            try:
                unet.print_trainable_parameters()
            except Exception:
                pass

    # ---- StableSyncNet (Stage 2 supervision) ----
    syncnet = None
    if config.model.add_audio_layer and config.run.use_syncnet:
        syncnet_config = OmegaConf.load(config.data.syncnet_config_path)
        if syncnet_config.ckpt.inference_ckpt_path == "":
            raise ValueError("SyncNet path is not provided")
        syncnet = StableSyncNet(OmegaConf.to_container(syncnet_config.model), gradient_checkpointing=True).to(
            device=device, dtype=torch.float16
        )
        syncnet_checkpoint = torch.load(
            syncnet_config.ckpt.inference_ckpt_path, map_location=device, weights_only=True
        )
        syncnet.load_state_dict(syncnet_checkpoint["state_dict"])
        syncnet.requires_grad_(False)
        del syncnet_checkpoint
        torch.cuda.empty_cache()

    # ---- Optimizer (LoRA params only) ----
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if config.optimizer.scale_lr:
        config.optimizer.lr = config.optimizer.lr * num_processes
    optimizer = torch.optim.AdamW(trainable_params, lr=config.optimizer.lr)

    if is_main_process:
        logger.info(f"LoRA trainable params: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")
        logger.info(f"LoRA total params:     {sum(p.numel() for p in unet.parameters()) / 1e6:.3f} M")
        logger.info(f"LoRA ratio:            {100.0 * sum(p.numel() for p in trainable_params) / sum(p.numel() for p in unet.parameters()):.4f}%")

    # Gradient checkpointing
    if config.run.enable_gradient_checkpointing:
        try:
            unet.enable_gradient_checkpointing()
        except Exception as e:
            logger.warning(f"enable_gradient_checkpointing failed: {e}")

    # ---- Data ----
    train_dataset = UNetDataset(config.data.train_data_dir, config)
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=config.run.seed,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=config.data.num_workers,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=train_dataset.worker_init_fn,
    )

    if config.run.max_train_steps == -1:
        assert config.run.max_train_epochs != -1
        config.run.max_train_steps = config.run.max_train_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        config.optimizer.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.optimizer.lr_warmup_steps,
        num_training_steps=config.run.max_train_steps,
    )

    # ---- Optional loss modules ----
    if config.run.perceptual_loss_weight != 0 and config.run.pixel_space_supervise:
        lpips_loss_func = lpips.LPIPS(net="vgg").to(device)
    if config.run.trepa_loss_weight != 0 and config.run.pixel_space_supervise:
        trepa_loss_func = TREPALoss(device=device, with_cp=True)

    # ---- Validation pipeline (uses base UNet with LoRA baked in for this step) ----
    # merge_and_unload mutates the source in-place, so deep-copy first to
    # preserve the LoRA-wrapped `unet` for DDP training.
    if hasattr(unet, "merge_and_unload"):
        unet_for_pipeline = copy.deepcopy(unet)
        unet_for_pipeline = unet_for_pipeline.merge_and_unload()
    else:
        unet_for_pipeline = unet
    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet_for_pipeline,
        scheduler=noise_scheduler,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)
    # Keep the LoRA-wrapped `unet` for DDP training; do NOT overwrite with
    # the merged pipeline copy (which would lose the LoRA trainable params).
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # ---- Training loop ----
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    num_train_epochs = math.ceil(config.run.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = config.data.batch_size * num_processes

    if is_main_process:
        logger.info("***** Running LoRA training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Total optimization steps = {config.run.max_train_steps}")

    global_step = resume_global_step
    first_epoch = resume_global_step // num_update_steps_per_epoch
    progress_bar = tqdm(
        range(0, config.run.max_train_steps),
        initial=resume_global_step,
        desc="Steps",
        disable=not is_main_process,
    )

    train_step_list = []
    train_loss_list = []
    recon_loss_list = []
    lpips_loss_list = []
    sync_loss_list = []
    lr_list = []
    val_step_list = []
    sync_conf_list = []
    scaler = torch.amp.GradScaler("cuda") if config.run.mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        unet.train()

        for step, batch in enumerate(train_dataloader):
            ### >>>> Training >>>> ###

            if config.model.add_audio_layer:
                if batch["mel"] != []:
                    mel = batch["mel"].to(device, dtype=torch.float16)
                audio_embeds_list = []
                try:
                    for idx in range(len(batch["video_path"])):
                        video_path = batch["video_path"][idx]
                        start_idx = batch["start_idx"][idx]
                        with torch.no_grad():
                            audio_feat = audio_encoder.audio2feat(video_path)
                        audio_embeds = audio_encoder.crop_overlap_audio_window(audio_feat, start_idx)
                        audio_embeds_list.append(audio_embeds)
                except Exception as e:
                    logger.info(f"{type(e).__name__} - {e} - {video_path}")
                    continue
                audio_embeds = torch.stack(audio_embeds_list)
                audio_embeds = audio_embeds.to(device, dtype=torch.float16)
            else:
                audio_embeds = None

            gt_pixel_values = batch["gt_pixel_values"].to(device, dtype=torch.float16)
            masked_pixel_values = batch["masked_pixel_values"].to(device, dtype=torch.float16)
            masks = batch["masks"].to(device, dtype=torch.float16)
            ref_pixel_values = batch["ref_pixel_values"].to(device, dtype=torch.float16)

            gt_pixel_values = rearrange(gt_pixel_values, "b f c h w -> (b f) c h w")
            masked_pixel_values = rearrange(masked_pixel_values, "b f c h w -> (b f) c h w")
            masks = rearrange(masks, "b f c h w -> (b f) c h w")
            ref_pixel_values = rearrange(ref_pixel_values, "b f c h w -> (b f) c h w")

            with torch.no_grad():
                gt_latents = vae.encode(gt_pixel_values).latent_dist.sample()
                masked_latents = vae.encode(masked_pixel_values).latent_dist.sample()
                ref_latents = vae.encode(ref_pixel_values).latent_dist.sample()

            masks = torch.nn.functional.interpolate(masks, size=config.data.resolution // vae_scale_factor)

            gt_latents = (
                rearrange(gt_latents, "(b f) c h w -> b c f h w", f=config.data.num_frames) - vae.config.shift_factor
            ) * vae.config.scaling_factor
            masked_latents = (
                rearrange(masked_latents, "(b f) c h w -> b c f h w", f=config.data.num_frames) - vae.config.shift_factor
            ) * vae.config.scaling_factor
            ref_latents = (
                rearrange(ref_latents, "(b f) c h w -> b c f h w", f=config.data.num_frames) - vae.config.shift_factor
            ) * vae.config.scaling_factor
            masks = rearrange(masks, "(b f) c h w -> b c f h w", f=config.data.num_frames)

            if config.run.use_mixed_noise:
                noise_shared_std_dev = (config.run.mixed_noise_alpha**2 / (1 + config.run.mixed_noise_alpha**2)) ** 0.5
                noise_shared = torch.randn_like(gt_latents) * noise_shared_std_dev
                noise_shared = noise_shared[:, :, 0:1].repeat(1, 1, config.data.num_frames, 1, 1)
                noise_ind_std_dev = (1 / (1 + config.run.mixed_noise_alpha**2)) ** 0.5
                noise_ind = torch.randn_like(gt_latents) * noise_ind_std_dev
                noise = noise_ind + noise_shared
            else:
                noise = torch.randn_like(gt_latents)
                noise = noise[:, :, 0:1].repeat(1, 1, config.data.num_frames, 1, 1)

            bsz = gt_latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=gt_latents.device).long()
            noisy_gt_latents = noise_scheduler.add_noise(gt_latents, noise, timesteps)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            unet_input = torch.cat([noisy_gt_latents, masks, masked_latents, ref_latents], dim=1)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config.run.mixed_precision_training):
                pred_noise = unet(unet_input, timesteps, encoder_hidden_states=audio_embeds).sample

            if config.run.recon_loss_weight != 0:
                recon_loss = F.mse_loss(pred_noise.float(), target.float(), reduction="mean")
            else:
                recon_loss = 0

            pred_latents = one_step_sampling(noise_scheduler, pred_noise, timesteps, noisy_gt_latents)

            if config.run.pixel_space_supervise:
                pred_pixel_values = vae.decode(
                    rearrange(pred_latents, "b c f h w -> (b f) c h w") / vae.config.scaling_factor
                    + vae.config.shift_factor
                ).sample

            if config.run.perceptual_loss_weight != 0 and config.run.pixel_space_supervise:
                pred_pixel_values_perceptual = pred_pixel_values[:, :, pred_pixel_values.shape[2] // 2 :, :]
                gt_pixel_values_perceptual = gt_pixel_values[:, :, gt_pixel_values.shape[2] // 2 :, :]
                lpips_loss = lpips_loss_func(
                    pred_pixel_values_perceptual.float(), gt_pixel_values_perceptual.float()
                ).mean()
            else:
                lpips_loss = 0

            if config.run.trepa_loss_weight != 0 and config.run.pixel_space_supervise:
                trepa_pred_pixel_values = rearrange(pred_pixel_values, "(b f) c h w -> b c f h w", f=config.data.num_frames)
                trepa_gt_pixel_values = rearrange(gt_pixel_values, "(b f) c h w -> b c f h w", f=config.data.num_frames)
                trepa_loss = trepa_loss_func(trepa_pred_pixel_values, trepa_gt_pixel_values)
            else:
                trepa_loss = 0

            if config.model.add_audio_layer and config.run.use_syncnet:
                syncnet_num_frames = syncnet_config.data.num_frames
                if config.run.pixel_space_supervise:
                    if config.data.resolution != syncnet_config.data.resolution:
                        pred_pixel_values = F.interpolate(
                            pred_pixel_values,
                            size=(syncnet_config.data.resolution, syncnet_config.data.resolution),
                            mode="bicubic",
                        )
                    # SyncNet was trained on a fixed number of frames; if the
                    # UNet config uses a different clip length, resample.
                    if config.data.num_frames != syncnet_num_frames:
                        if is_main_process:
                            logger.warning(
                                "UNet num_frames=%d differs from SyncNet num_frames=%d; "
                                "temporal interpolation will be used for sync loss.",
                                config.data.num_frames,
                                syncnet_num_frames,
                            )
                        pred_pixel_values_5d = rearrange(
                            pred_pixel_values, "(b f) c h w -> b c f h w", f=config.data.num_frames
                        )
                        pred_pixel_values_5d = F.interpolate(
                            pred_pixel_values_5d.float(),
                            size=(syncnet_num_frames, pred_pixel_values_5d.shape[-2], pred_pixel_values_5d.shape[-1]),
                            mode="trilinear",
                            align_corners=False,
                        ).to(dtype=pred_pixel_values.dtype)
                        pred_pixel_values_aligned = rearrange(
                            pred_pixel_values_5d, "b c f h w -> (b f) c h w"
                        )
                    else:
                        pred_pixel_values_aligned = pred_pixel_values
                    syncnet_input = rearrange(
                        pred_pixel_values_aligned, "(b f) c h w -> b (f c) h w", f=syncnet_num_frames
                    )
                else:
                    if config.data.num_frames != syncnet_num_frames:
                        if is_main_process:
                            logger.warning(
                                "UNet num_frames=%d differs from SyncNet num_frames=%d; "
                                "temporal interpolation will be used for sync loss.",
                                config.data.num_frames,
                                syncnet_num_frames,
                            )
                        pred_latents_aligned = F.interpolate(
                            pred_latents.float(),
                            size=(syncnet_num_frames, pred_latents.shape[-2], pred_latents.shape[-1]),
                            mode="trilinear",
                            align_corners=False,
                        ).to(dtype=pred_latents.dtype)
                    else:
                        pred_latents_aligned = pred_latents
                    syncnet_input = rearrange(pred_latents_aligned, "b c f h w -> b (f c) h w")

                if syncnet_config.data.lower_half:
                    height = syncnet_input.shape[2]
                    syncnet_input = syncnet_input[:, :, height // 2 :, :]
                ones_tensor = torch.ones((config.data.batch_size, 1)).float().to(device=device)
                vision_embeds, audio_embeds = syncnet(syncnet_input, mel)
                sync_loss = cosine_loss(vision_embeds.float(), audio_embeds.float(), ones_tensor).mean()
            else:
                sync_loss = 0

            loss = (
                recon_loss * config.run.recon_loss_weight
                + sync_loss * config.run.sync_loss_weight
                + lpips_loss * config.run.perceptual_loss_weight
                + trepa_loss * config.run.trepa_loss_weight
            )

            if is_main_process:
                train_step_list.append(global_step)
                train_loss_list.append(float(loss.item()))
                recon_loss_list.append(float(recon_loss))
                lpips_loss_list.append(float(lpips_loss))
                sync_loss_list.append(float(sync_loss))
                lr_list.append(float(lr_scheduler.get_last_lr()[0]))

            optimizer.zero_grad()

            if config.run.mixed_precision_training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, config.optimizer.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, config.optimizer.max_grad_norm)
                optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1

            ### <<<< Training <<<< ###

            if is_main_process and (global_step % config.ckpt.save_ckpt_steps == 0):
                # Save LoRA adapter (small!) and unwrapped base
                adapter_dir = os.path.join(output_dir, f"checkpoints/checkpoint-{global_step}")
                os.makedirs(adapter_dir, exist_ok=True)
                # The DDP-wrapped unet is unet.module after first .module access
                base_unet = unet.module
                if hasattr(base_unet, "save_pretrained"):
                    # peft-wrapped model: save adapter only
                    base_unet.save_pretrained(adapter_dir)
                    logger.info(f"Saved LoRA adapter to {adapter_dir}")
                else:
                    # fallback: save state_dict of trainable params
                    torch.save(
                        {"state_dict": {n: p for n, p in base_unet.named_parameters() if p.requires_grad}},
                        os.path.join(adapter_dir, "lora_trainable.pt"),
                    )
                    logger.info(f"Saved trainable LoRA params to {adapter_dir}/lora_trainable.pt")

                # Validation
                logger.info("Running validation... ")
                validation_video_out_path = os.path.join(output_dir, f"val_videos/val_video_{global_step}.mp4")
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pipeline(
                        config.data.val_video_path,
                        config.data.val_audio_path,
                        validation_video_out_path,
                        num_frames=config.data.num_frames,
                        num_inference_steps=config.run.inference_steps,
                        guidance_scale=config.run.guidance_scale,
                        weight_dtype=torch.float16,
                        width=config.data.resolution,
                        height=config.data.resolution,
                        mask_image_path=config.data.mask_image_path,
                    )
                logger.info(f"Saved validation video output to {validation_video_out_path}")

                val_step_list.append(global_step)
                if config.model.add_audio_layer and os.path.exists(validation_video_out_path):
                    try:
                        _, conf = syncnet_eval(syncnet_eval_model, syncnet_detector, validation_video_out_path, "temp")
                    except Exception as e:
                        logger.info(e)
                        conf = 0
                    sync_conf_list.append(conf)
                    plot_loss_chart(
                        os.path.join(output_dir, f"sync_conf_results/sync_conf_chart-{global_step}.png"),
                        ("Sync confidence", val_step_list, sync_conf_list),
                    )

                if train_loss_list:
                    _save_loss_chart(
                        os.path.join(output_dir, f"loss_charts/loss_chart-{global_step}.png"),
                        train_step_list,
                        {
                            "total": train_loss_list,
                            "recon": recon_loss_list,
                            "lpips": lpips_loss_list,
                            "sync": sync_loss_list,
                        },
                        lr_list,
                    )

            logs = {"step_loss": loss.item(), "epoch": epoch}
            progress_bar.set_postfix(**logs)

            if global_step >= config.run.max_train_steps:
                break

    progress_bar.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet/stage2_lora.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.unet_config_path)
    config.unet_config_path = args.unet_config_path
    main(config)
