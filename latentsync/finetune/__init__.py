"""LatentSync fine-tuning UI package.

This package holds the Gradio Fine-tune Studio logic that was previously
monolithic in ``gradio_finetune.py``. It is split by responsibility:

- config: presets and config generation
- process: training/inference subprocess management
- utils: path resolution, chart parsing, checkpoint introspection
- ui_*: per-tab Gradio callbacks
- ui: assembles ``build_ui()``
"""

import logging
import os
from pathlib import Path

# Disable Gradio telemetry / messaging fetches so the page loads faster in
# network-restricted environments (e.g. China mainland without a proxy).
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "0")
os.environ.setdefault("GRADIO_TELEMETRY_ENABLED", "0")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("gradio_finetune")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = REPO_ROOT / "configs" / "unet"
SYNCNET_CONFIG_DIR = REPO_ROOT / "configs" / "syncnet"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

# Fine-tuning intermediates (generated configs, training logs, run outputs,
# audio embeds/mel caches) go to a separate large-disk directory by default.
# Can be overridden with LATENTSYNC_FINETUNE_DIR env var.
_FINETUNE_BASE_DIR_STR = os.environ.get(
    "LATENTSYNC_FINETUNE_DIR", "/root/autodl-tmp/latentsync_finetune"
)
FINETUNE_BASE_DIR = Path(_FINETUNE_BASE_DIR_STR)
try:
    FINETUNE_BASE_DIR.mkdir(parents=True, exist_ok=True)
except OSError as e:
    logger.warning(
        "Cannot create fine-tune base dir %s (%s); falling back to %s",
        FINETUNE_BASE_DIR,
        e,
        REPO_ROOT / "debug",
    )
    FINETUNE_BASE_DIR = REPO_ROOT / "debug"
    FINETUNE_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Propagate the effective base dir to child training processes so they can
# resolve relative train_output_dir consistently.
os.environ.setdefault("LATENTSYNC_FINETUNE_DIR", str(FINETUNE_BASE_DIR))
TRAIN_OUTPUT_DIR = FINETUNE_BASE_DIR

ASSETS_DIR = REPO_ROOT / "assets"
PREBUILT_DATASETS_YAML = REPO_ROOT / "tools" / "prebuilt_datasets.yaml"
