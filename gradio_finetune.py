"""Gradio Fine-tuning UI entry point for LatentSync.

The implementation has been split into the ``latentsync.finetune`` package;
this file only sets up telemetry/logging, parses CLI args, and launches the
Gradio application built by ``latentsync.finetune.ui.build_ui``.
"""

import argparse

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
