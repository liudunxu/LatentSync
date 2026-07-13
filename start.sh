#!/bin/bash
# Convenience script to start the LatentSync Fine-tune Studio.
# Default port: 6006 (shared with TensorBoard if you run that too).
# Override with: ./start.sh --port 8080

set -e

# cd to the repo root regardless of where the script is invoked from
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Pick the venv python if it exists, otherwise fall back to the system one
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif [ -x "venv/bin/python" ]; then
    PY="venv/bin/python"
else
    PY="python"
fi

# Make sure gradi o + finetune deps are installed
"$PY" -c "import gradio" 2>/dev/null || {
    echo "[start.sh] gradio not installed. Installing finetune deps..."
    "$PY" -m pip install -q gradio peft bitsandbytes matplotlib scikit-image
}

echo "[start.sh] Using: $PY"
echo "[start.sh] Starting Fine-tune Studio on http://0.0.0.0:6006 ..."
echo "[start.sh] (use --port / --share / --host to override)"

exec "$PY" gradio_finetune.py "$@"
