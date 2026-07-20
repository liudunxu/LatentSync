"""Gradio Fine-tuning UI entry point for LatentSync.

The implementation has been split into the ``latentsync.finetune`` package;
this file only sets up telemetry/logging, parses CLI args, and launches the
Gradio application built by ``latentsync.finetune.ui.build_ui``.
"""

import argparse
import os

# Some container images ship an empty/garbage OMP_NUM_THREADS, which makes
# libgomp print "Invalid value for environment variable OMP_NUM_THREADS".
# Normalize it before gradio/torch initialize OpenMP (training subprocesses
# inherit this environment, so fixing it here covers them too).
_omp = os.environ.get("OMP_NUM_THREADS")
try:
    _omp_valid = _omp is not None and all(int(p) > 0 for p in _omp.split(","))
except ValueError:
    _omp_valid = False
if not _omp_valid:
    os.environ["OMP_NUM_THREADS"] = "4"
del _omp, _omp_valid

from latentsync.finetune.ui import build_ui


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    from latentsync.finetune import FINETUNE_BASE_DIR, REPO_ROOT

    allowed = [str(REPO_ROOT), str(FINETUNE_BASE_DIR), "/tmp"]
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        allowed_paths=allowed,
        show_api=False,
    )


if __name__ == "__main__":
    main()
