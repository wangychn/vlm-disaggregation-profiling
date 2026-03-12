# vlm-disaggregation-profiling

Minimal Qwen3-VL probe at the moment.

It does two things:

- dumps the module tree and module names
- optionally runs one profiled `forward` or `generate` call

```
<root>/
  qwen3vl_probe.py        # main profiling CLI (now uses engine helpers)
  requirements.txt        # Python dependencies
  README.md               # this document
  scripts/                # convenience shell wrappers
    run_probe.sh          # launches qwen3vl_probe with timestamped output
  engine/                 # shared utilities for probing and datasets
    utils.py              # model tree, profiling helpers, device helpers
    profiler.py           # hook-based PyTorch profiler
    datasets.py           # loaders for COCO and other eval sets
  tasks/                  # alternate versions / experiments (probe script)
  deepstack_ref/          # reference DeepStack serving framework
  distserve_ref/          # other unrelated reference code
  data/                   # (suggested) place to keep downloaded datasets
```

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Inspect only

```bash
python3 qwen3vl_probe.py \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --output-dir artifacts/qwen3vl-inspect
```

## Inspect + profile

```bash
python3 qwen3vl_probe.py \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --image /absolute/path/to/example.png \
  --prompt "Describe the image." \
  --mode forward \
  --output-dir artifacts/qwen3vl-forward
```

## Outputs

- `module_tree.txt`
- `named_modules.txt`
- `inspection.json`
- `profile.json` if `--image` is provided
