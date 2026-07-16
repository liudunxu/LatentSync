"""Unit tests for gradio_finetune config generation."""

import pytest

from latentsync.finetune.config import (
    DEFAULT_PRESET_NAME,
    PRESETS,
    build_config_from_form,
    on_preset_change,
)


def test_default_preset_exists():
    assert DEFAULT_PRESET_NAME in PRESETS
    assert "lora" in PRESETS[DEFAULT_PRESET_NAME]
    assert PRESETS[DEFAULT_PRESET_NAME]["lora"]["enabled"] is True


def _preset_kwargs(preset_name: str):
    """Return kwargs for build_config_from_form from preset defaults."""
    (
        batch_size,
        num_frames,
        resolution,
        learning_rate,
        use_motion_module,
        pixel_space_supervise,
        use_syncnet,
        sync_loss_weight,
        perceptual_loss_weight,
        recon_loss_weight,
        trepa_loss_weight,
        mixed_precision_training,
        enable_gradient_checkpointing,
        mask_image_path,
        resume_ckpt,
        save_ckpt_steps,
        max_train_steps,
        lr_scheduler,
        lr_warmup_steps,
        _description,
        freeze_attn2,
    ) = on_preset_change(preset_name)

    return {
        "preset_name": preset_name,
        "train_data_dir": "assets",
        "train_fileslist": "data/demo_fileslist.txt",
        "val_video_path": "assets/demo1_video.mp4",
        "val_audio_path": "assets/demo1_audio.wav",
        "resume_ckpt": resume_ckpt,
        "batch_size": batch_size,
        "num_frames": num_frames,
        "resolution": resolution,
        "learning_rate": learning_rate,
        "use_motion_module": use_motion_module,
        "pixel_space_supervise": pixel_space_supervise,
        "use_syncnet": use_syncnet,
        "sync_loss_weight": sync_loss_weight,
        "perceptual_loss_weight": perceptual_loss_weight,
        "recon_loss_weight": recon_loss_weight,
        "trepa_loss_weight": trepa_loss_weight,
        "mixed_precision_training": mixed_precision_training,
        "enable_gradient_checkpointing": enable_gradient_checkpointing,
        "mask_image_path": mask_image_path,
        "save_ckpt_steps": save_ckpt_steps,
        "max_train_steps": max_train_steps,
        "num_workers": 4,
        "train_output_dir": "unet",
        "freeze_attn2": freeze_attn2,
        "val_inference_steps": 20,
        "val_guidance_scale": 1.5,
        "val_seed": 1247,
        "lr_scheduler": lr_scheduler,
        "lr_warmup_steps": lr_warmup_steps,
    }


@pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
def test_build_config_from_form_for_all_presets(preset_name):
    """Every preset must produce a valid config dict without raising."""
    kwargs = _preset_kwargs(preset_name)
    cfg = build_config_from_form(**kwargs)

    assert "data" in cfg
    assert "ckpt" in cfg
    assert "run" in cfg
    assert "optimizer" in cfg
    assert "model" in cfg
    assert "lora" in cfg

    assert cfg["data"]["num_frames"] == kwargs["num_frames"]
    assert cfg["data"]["resolution"] == kwargs["resolution"]
    assert cfg["optimizer"]["lr"] == kwargs["learning_rate"]

    # SyncNet preset has no LoRA block by default; others may.
    if preset_name == "SyncNet 训练":
        assert cfg["lora"]["enabled"] is False


def test_build_config_lora_preset_structure():
    """LoRA presets propagate rank/alpha/target_modules into cfg."""
    kwargs = _preset_kwargs("Stage 2 LoRA (256, 12-15GB)")
    cfg = build_config_from_form(**kwargs)

    assert cfg["lora"]["enabled"] is True
    assert cfg["lora"]["rank"] == 32
    assert cfg["lora"]["alpha"] == 64
    assert "to_q" in cfg["lora"]["target_modules"]


def test_build_config_syncnet_num_frames_mismatch_raises():
    """Mismatch between UNet num_frames and SyncNet config should raise."""
    kwargs = _preset_kwargs("Stage 2 (256, 全量训练)")
    kwargs["num_frames"] = 8  # SyncNet config expects 16

    with pytest.raises(ValueError) as exc_info:
        build_config_from_form(**kwargs)

    assert "num_frames" in str(exc_info.value)
