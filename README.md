# vlm-disaggregation-profiling

Minimal Qwen3-VL probe at the moment.

It does two things:

- dumps the module tree and module names
- optionally runs one profiled `forward` or `generate` call

```
<root>/
  requirements.txt        # Python dependencies
  README.md               # this document
  scripts/                # convenience shell wrappers
    run_probe.sh          # launches the probe with timestamped output
  engine/                 # shared utilities for probing and datasets
    utils.py              # model tree, profiling helpers, device helpers
    profiler.py           # hook-based PyTorch profiler
    datasets.py           # Hugging Face MME loader and local materializer
  tasks/                  # probe entrypoints / experiments
    qwen3vl_probe.py      # main profiling CLI
  deepstack_ref/          # reference DeepStack serving framework
  distserve_ref/          # other unrelated reference code
  data/                   # (suggested) place to keep downloaded datasets
```

## Install

```bash
# Install Python dependencies
python3 -m pip install -r requirements.txt

# Run all commands from the project root directory
cd /path/to/vlm-disaggregation-profiling
```

## Inspect only

```bash
python3 tasks/qwen3vl_probe.py \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --output-dir artifacts/qwen3vl-inspect
```

## Inspect + profile

```bash
python3 tasks/qwen3vl_probe.py \
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

## Datasets

Dataset support is now centered on `lmms-lab/MME` from Hugging Face:

https://huggingface.co/datasets/lmms-lab/MME

Prepare a small local working set:

```bash
python3 -m engine.datasets \
  --root data/mme \
  --split test \
  --max-items 50
```

Or use the shell wrapper:

```bash
bash scripts/download_mme.sh data/mme test 50
```

Useful options:

- `--cache-dir /path/to/hf-cache` to control Hugging Face dataset caching
- `--max-items N` to only materialize a small subset
- `--streaming` to avoid preparing the full split up front
- `--category artwork` to keep only one MME category

Then profile from MME directly:

```bash
python3 tasks/qwen3vl_probe.py \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --mme-root data/mme \
  --mme-max 10 \
  --mode forward \
  --output-dir artifacts/qwen3vl-mme
```

## Notes

- this is intentionally Qwen3-VL-specific
- no test framework, adapter layer, or trace exporter is included
- if you want a smaller or larger model, change `--model-id`
