"""Unit tests for gradio_finetune subprocess managers."""

import json

import pytest

from latentsync.finetune.process import TrainingProcess


def test_training_process_state_roundtrip(tmp_path, monkeypatch):
    """save_state writes JSON; clear_state removes it."""
    tp = TrainingProcess()
    tp._pid = 12345
    tp.log_path = tmp_path / "train.log"
    tp.run_dir = tmp_path / "run"
    tp.started_at = "2026-01-01T00:00:00"
    tp.cmd = ["torchrun", "-m", "scripts.train_unet"]

    state_path = tmp_path / "active_trainer.json"
    # Override the staticmethod so it writes to our tmp_path.
    monkeypatch.setattr(TrainingProcess, "_state_path", staticmethod(lambda: state_path))

    tp.save_state()
    assert state_path.exists()
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["pid"] == 12345
    assert data["cmd"] == tp.cmd

    tp.clear_state()
    assert not state_path.exists()


def test_pid_alive_nonexistent():
    """_pid_alive returns False for a PID that does not exist."""
    tp = TrainingProcess()
    # PID 99999999 is extremely unlikely to exist.
    assert tp._pid_alive(99999999) is False


def test_training_process_is_running_no_proc():
    """is_running returns False when no proc or pid is set."""
    tp = TrainingProcess()
    assert tp.is_running() is False
