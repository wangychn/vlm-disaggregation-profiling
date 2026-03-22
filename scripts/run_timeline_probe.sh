python3 tasks/qwen3vl_timeline_probe.py \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --mme-root data/mme \
  --mme-max 1 \
  --mode forward \
  --output-dir artifacts/qwen3vl-timeline-forward


python3 tasks/qwen3vl_timeline_probe.py \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --mme-root data/mme \
  --mme-max 1 \
  --mode generate \
  --output-dir artifacts/qwen3vl-timeline-generate

