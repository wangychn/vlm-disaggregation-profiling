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
    datasets.py           # loaders for COCO and other eval sets
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

Helpers for common multimodal evaluation sets live in
`engine/datasets.py`.  For COCO captions you can run:

```bash
python3 -m engine.datasets --root data/coco --split val2017 --download
```

or use the Python API to download and iterate over a few examples:

```python
from pathlib import Path
from engine.datasets import download_coco, load_coco_captions

data_root = Path("data/coco")
download_coco(data_root, split="val2017")
for img_path, caption in load_coco_captions(data_root, max_items=5):
    print(img_path, caption)
```

The probe CLI supports `--coco-root` (plus `--coco-max`) so you can profile
multiple images from a dataset instead of supplying a single image and prompt.

## Notes

- this is intentionally Qwen3-VL-specific
- no test framework, adapter layer, or trace exporter is included
- if you want a smaller or larger model, change `--model-id`
