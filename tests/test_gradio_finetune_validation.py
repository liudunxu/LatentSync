"""Unit tests for gradio_finetune validation/compare helpers."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf

import gradio as gr

from latentsync.finetune import REPO_ROOT
from latentsync.finetune.process import _prune_debug_files
from latentsync.finetune.validation_utils import (
    check_ckpt_compatibility,
    format_validation_report,
    prepare_temp_config,
    quick_quality_check,
    read_result_json,
    resolve_ckpt_path,
    write_result_json,
)


def test_resolve_ckpt_path_absolute(tmp_path):
    p = tmp_path / "ckpt.pt"
    assert resolve_ckpt_path(str(p)) == p


def test_resolve_ckpt_path_relative():
    assert resolve_ckpt_path("configs/unet/stage2.yaml") == REPO_ROOT / "configs/unet/stage2.yaml"


def test_check_ckpt_compatibility_256_512_mismatch(tmp_path):
    cfg = OmegaConf.create({"data": {"resolution": 256, "mask_image_path": "latentsync/utils/mask.png"}})
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)
    warnings = check_ckpt_compatibility(Path("foo_512.pt"), cfg_path)
    assert any("512" in w for w in warnings)


def test_check_ckpt_compatibility_clean(tmp_path):
    cfg = OmegaConf.create({"data": {"resolution": 256, "mask_image_path": "latentsync/utils/mask.png"}})
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)
    warnings = check_ckpt_compatibility(Path("latentsync_unet.pt"), cfg_path)
    assert warnings == []


def test_format_validation_report_error():
    text = format_validation_report({"error": "boom"}, "ckpt.pt", 0.0)
    assert "质量检查失败" in text
    assert "boom" in text


def test_format_validation_report_good():
    metrics = {
        "num_frames": 100,
        "sharpness_mean": 150.0,
        "blurry_ratio": 0.1,
        "flicker": 5.0,
        "face_detect_rate": 0.95,
    }
    text = format_validation_report(metrics, "ckpt.pt", 12.3)
    assert "12.3" in text
    assert "✅" in text
    assert "100" in text


def test_prepare_temp_config(tmp_path):
    cfg = OmegaConf.create(
        {
            "data": {"resolution": 256, "mask_image_path": "mask.png"},
            "run": {"inference_steps": 20, "guidance_scale": 1.5},
        }
    )
    base_cfg = tmp_path / "base.yaml"
    OmegaConf.save(cfg, base_cfg)
    out = prepare_temp_config(base_cfg, 512, 40, 1.2, tmp_path, "ts")
    assert out.exists()
    loaded = OmegaConf.load(out)
    assert loaded.data.resolution == 512
    assert loaded.run.inference_steps == 40
    assert loaded.run.guidance_scale == 1.2


def test_write_and_read_result_json(tmp_path):
    path = tmp_path / "result.json"
    write_result_json(
        path,
        mode="validate",
        success=True,
        duration_sec=10.5,
        video_out_path=tmp_path / "out.mp4",
        metrics={"sharpness_mean": 100},
    )
    data = read_result_json(path)
    assert data["success"] is True
    assert data["duration_sec"] == 10.5
    assert data["metrics"]["sharpness_mean"] == 100


def test_read_result_json_missing(tmp_path):
    assert read_result_json(tmp_path / "nope.json") is None


def test_prune_debug_files_keeps_recent(tmp_path):
    for i in range(12):
        p = tmp_path / f"validation_cfg_{i:02d}.yaml"
        p.write_text("x")
    _prune_debug_files(tmp_path, "validation_cfg_*.yaml", keep=5)
    assert len(list(tmp_path.glob("validation_cfg_*.yaml"))) == 5


def test_quick_quality_check_missing():
    assert quick_quality_check("/nonexistent/path.mp4")["error"].startswith("video not found")


def test_quick_quality_check_success(tmp_path):
    class MockFrame:
        def __init__(self, arr):
            self._arr = arr

        def asnumpy(self):
            return self._arr

    frames = [MockFrame(np.full((64, 64, 3), i * 10, dtype=np.uint8)) for i in range(5)]
    mock_reader = MagicMock(return_value=frames)
    out = tmp_path / "out.mp4"
    out.touch()
    with patch.dict("sys.modules", {"decord": MagicMock(VideoReader=mock_reader)}):
        with patch("latentsync.finetune.validation_utils.Path.exists", return_value=True):
            metrics = quick_quality_check(str(out))
    assert "num_frames" in metrics
    assert metrics["num_frames"] == 5
    assert "sharpness_mean" in metrics


def test_finetune_inference_build_cmd_validate():
    from scripts.finetune_inference import _build_inference_cmd

    args = MagicMock(
        inference_steps=20,
        guidance_scale=1.5,
        seed=1247,
        temp_dir="temp",
        enable_deepcache=True,
        baseline_mode=False,
    )
    cmd = _build_inference_cmd(
        Path("cfg.yaml"),
        Path("ckpt.pt"),
        "v.mp4",
        "a.wav",
        Path("out.mp4"),
        args,
    )
    assert "scripts.inference" in cmd
    assert "--enable_deepcache" in cmd
    assert "--inference_steps" in cmd
    assert "20" in cmd


def test_finetune_inference_argparse_validate():
    from scripts.finetune_inference import main

    with patch("scripts.finetune_inference.cmd_validate") as mock_validate:
        mock_validate.return_value = 0
        rc = main(
            [
                "validate",
                "--unet_config_path",
                "configs/unet/stage2.yaml",
                "--inference_ckpt_path",
                "checkpoints/latentsync_unet.pt",
                "--video_path",
                "assets/demo1_video.mp4",
                "--audio_path",
                "assets/demo1_audio.wav",
                "--video_out_path",
                "/tmp/val.mp4",
            ]
        )
        assert rc == 0
        assert mock_validate.called


def test_finetune_inference_argparse_compare():
    from scripts.finetune_inference import main

    with patch("scripts.finetune_inference.cmd_compare") as mock_compare:
        mock_compare.return_value = 0
        rc = main(
            [
                "compare",
                "--unet_config_path",
                "configs/unet/stage2.yaml",
                "--base_ckpt",
                "checkpoints/latentsync_unet.pt",
                "--fine_tuned_ckpt",
                "debug/unet/checkpoint-1000",
                "--video_path",
                "assets/demo1_video.mp4",
                "--audio_path",
                "assets/demo1_audio.wav",
                "--base_video_out",
                "/tmp/base.mp4",
                "--ft_video_out",
                "/tmp/ft.mp4",
            ]
        )
        assert rc == 0
        assert mock_compare.called


def test_ui_inference_run_compare_rejects_same_ckpt():
    from latentsync.finetune.ui_inference import run_compare

    with pytest.raises(gr.Error):
        run_compare("v.mp4", "a.wav", "ckpt.pt", "ckpt.pt", 20, 1.5, 1247, 512)


def test_ui_inference_run_compare_rejects_missing_inputs():
    from latentsync.finetune.ui_inference import run_compare

    with pytest.raises(gr.Error):
        run_compare("", "a.wav", "a.pt", "b.pt", 20, 1.5, 1247, 512)


def test_ui_inference_run_validation_rejects_missing_ckpt():
    from latentsync.finetune.ui_inference import run_validation

    result = run_validation(
        "v.mp4", "a.wav", "/nonexistent/ckpt.pt", "configs/unet/stage2.yaml",
        20, 1.5, 1247, 512, True, True,
    )
    # Returns a 5-tuple of gr.update; third element is report text.
    assert len(result) == 5
    assert "❌ ckpt 不存在" in str(result[2])
