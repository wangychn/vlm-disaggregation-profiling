import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.datasets import iter_mme_examples
from engine.timeline_profiler import trace_run
from engine.utils import (
    identify_modules,
    load_qwen3vl,
    move_to_device,
    pick_device,
    prepare_inputs,
    summarize,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mme-root", required=True)
    parser.add_argument("--mme-split", default="test")
    parser.add_argument("--mme-cache-dir")
    parser.add_argument("--mme-max", type=int, default=1)
    parser.add_argument("--mme-streaming", action="store_true")
    parser.add_argument("--mme-category")
    parser.add_argument("--mode", choices=("forward", "generate"), default="forward")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--all-modules", action="store_true")
    return parser.parse_args()


def response_payload(output, processor, batch, mode):
    if mode != "generate":
        return {"output": summarize(output)}

    sequences = (output.sequences if hasattr(output, "sequences") else output).detach().cpu()
    prompt_len = batch["input_ids"].shape[-1]
    return {
        "text": processor.batch_decode(sequences[:, prompt_len:], skip_special_tokens=True)[0],
        "output": summarize(output),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_qwen3vl(
        args.model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    module_info = identify_modules(model)
    device = pick_device(model)
    targets = (
        [name for name, _ in model.named_modules() if name]
        if args.all_modules
        else (
            module_info["vision_encoder"]
            + module_info["merger"]
            + module_info["deepstack_modules"]
            + module_info["decoder_layers"]
        )
    )

    for idx, (img_path, question, meta) in enumerate(
        iter_mme_examples(
            Path(args.mme_root),
            split=args.mme_split,
            cache_dir=args.mme_cache_dir,
            max_items=args.mme_max,
            streaming=args.mme_streaming,
            category=args.mme_category,
        ),
        start=1,
    ):
        batch = prepare_inputs(
            processor,
            question=question,
            image_path=str(img_path),
            add_generation_prompt=(args.mode == "generate"),
        )
        batch = move_to_device(batch, device)

        sample_dir = output_dir / f"sample_{idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        trace_path = sample_dir / "trace.json"

        result = trace_run(
            model,
            batch,
            trace_path=str(trace_path),
            target_names=targets,
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
        )
        write_json(
            sample_dir / "trace_info.json",
            {
                "sample": {"index": idx, "image_path": str(img_path), "question": question, "mme": meta},
                "run": {
                    "mode": args.mode,
                    "response": response_payload(result["output"], processor, batch, args.mode),
                    "table": result["table"],
                },
                "trace_path": str(trace_path),
                "hooked_modules": targets,
            },
        )
        print(f"[sample {idx}] {trace_path}")


if __name__ == "__main__":
    main()
