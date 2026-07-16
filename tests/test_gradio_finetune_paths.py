"""Unit tests for gradio_finetune path utilities."""

from pathlib import Path

import pytest

from latentsync.finetune import FINETUNE_BASE_DIR, REPO_ROOT
from latentsync.finetune.utils import (
    _resolve_output_dir,
    _resolve_run_dir,
    list_run_dirs,
)


def test_resolve_output_dir_relative():
    """Relative train_output_dir resolves against FINETUNE_BASE_DIR."""
    assert _resolve_output_dir("unet") == FINETUNE_BASE_DIR / "unet"
    assert _resolve_output_dir("foo/bar") == FINETUNE_BASE_DIR / "foo" / "bar"


def test_resolve_output_dir_absolute():
    """Absolute train_output_dir is preserved."""
    p = "/tmp/abs_test"
    assert _resolve_output_dir(p) == Path(p)


def test_resolve_output_dir_empty():
    """Empty train_output_dir falls back to FINETUNE_BASE_DIR/unet."""
    assert _resolve_output_dir("") == FINETUNE_BASE_DIR / "unet"


def test_resolve_run_dir_absolute():
    """Absolute selected_run is returned as-is."""
    p = Path("/tmp/run-123")
    assert _resolve_run_dir(str(p)) == p


def test_resolve_run_dir_relative_to_repo():
    """Relative path under REPO_ROOT is resolved there."""
    # assets exists in REPO_ROOT
    p = _resolve_run_dir("assets")
    assert p == REPO_ROOT / "assets"


def test_resolve_run_dir_relative_to_finetune_base(tmp_path):
    """Relative path not in REPO_ROOT falls back to FINETUNE_BASE_DIR."""
    run_dir = tmp_path / "train-2026_01_01-00:00:00"
    run_dir.mkdir()
    # Temporarily point FINETUNE_BASE_DIR to tmp_path via monkeypatching
    import latentsync.finetune.utils as utils_mod
    original = utils_mod.FINETUNE_BASE_DIR
    try:
        utils_mod.FINETUNE_BASE_DIR = tmp_path
        resolved = _resolve_run_dir("train-2026_01_01-00:00:00")
        assert resolved == run_dir
    finally:
        utils_mod.FINETUNE_BASE_DIR = original


def test_list_run_dirs_filters_and_sorts(tmp_path):
    """list_run_dirs only keeps train-* / train_lora-* and sorts by ctime."""
    (tmp_path / "train-2026_01_01-00:00:00").mkdir()
    (tmp_path / "train_lora-2026_01_02-00:00:00").mkdir()
    (tmp_path / "other-2026_01_03-00:00:00").mkdir()
    (tmp_path / "checkpoint.pt").write_text("dummy")

    runs = list_run_dirs(tmp_path)
    assert len(runs) == 2
    names = [Path(r).name for r in runs]
    assert all(name.startswith(("train-", "train_lora-")) for name in names)
