set -euo pipefail

# default output directory under "outputs" with timestamp
OUTDIR=${6:-outputs/probe-$(date +%Y%m%d_%H%M%S)}
mkdir -p "artifacts/qwen3vl-mme"

python3 tasks/qwen3vl_probe.py \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --mme-root data/mme \
  --mme-max 10 \
  --mode forward \
  --output-dir artifacts/qwen3vl-mme
