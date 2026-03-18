#!/usr/bin/env bash
# run_probe.sh - lightweight wrapper for tasks/qwen3vl_probe.py
# Usage: ./run_probe.sh [MODEL] [IMAGE] [PROMPT] [MODE] [DTYPE] [OUTDIR]

set -euo pipefail

MODEL=${1:-Qwen/Qwen3-VL-4B-Instruct}
IMAGE=${2:-}
PROMPT=${3:-"Describe the image."}
MODE=${4:-forward}
DTYPE=${5:-bfloat16}

# default output directory under "outputs" with timestamp
OUTDIR=${6:-outputs/probe-$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUTDIR"

echo "Running probe with model=$MODEL image=$IMAGE mode=$MODE dtype=$DTYPE"
echo "Outputs will be written to $OUTDIR"

python3 tasks/qwen3vl_probe.py \
  --model-id "$MODEL" \
  --image "$IMAGE" \
  --prompt "$PROMPT" \
  --mode "$MODE" \
  --device-map auto \
  --dtype "$DTYPE" \
  --output-dir "$OUTDIR"

# smoke test
# python3 tasks/qwen3vl_probe.py \
#   --model-id Qwen/Qwen3-VL-8B-Instruct \
#   --output-dir artifacts/qwen3vl-8b-inspect
