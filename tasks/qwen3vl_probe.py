from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# helpers from engine
from engine.utils import (
    torch_dtype,
    render_tree,
    identify_modules,
    write_text,
    write_json,
    pick_device,
    move_to_device,
)


def load_model(model_id: str, dtype: str, device_map: str, trust_remote_code: bool):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    kwargs = {
        "torch_dtype": torch_dtype(dtype),
        "trust_remote_code": trust_remote_code,
    }
    if device_map != "none":
        kwargs["device_map"] = device_map

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    return model, processor

def prepare_inputs(processor, question: str, image_path: str, add_generation_prompt: bool):
    content = [
        {"type": "image", "path": image_path},
        {"type": "text", "text": question},
    ]
    messages = [{"role": "user", "content": content}]

    try:
        batch = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        batch = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
    if isinstance(batch, dict):
        batch.pop("token_type_ids", None)
    return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Qwen3-VL inspection and profiling probe.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mme-root", required=True, help="local directory used to cache selected MME examples")
    parser.add_argument("--mme-split", default="test", help="MME split to use")
    parser.add_argument("--mme-cache-dir", default=None, help="optional Hugging Face datasets cache directory")
    parser.add_argument("--mme-max", type=int, default=3, help="number of MME examples to profile")
    parser.add_argument("--mme-streaming", action="store_true", help="stream MME rows instead of preparing the full slice")
    parser.add_argument("--mme-category", default=None, help="optional single MME category filter")
    parser.add_argument("--mode", choices=("forward", "generate"), default="forward")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--all-modules", action="store_true")
    parser.add_argument("--no-sync-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(
        args.model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    # ==== Running the profiler on the loaded model ====
    module_info = identify_modules(model)
    write_text(output_dir / "module_tree.txt", render_tree(model))
    write_text(
        output_dir / "named_modules.txt",
        [f"{name or '<root>'}\t{module.__class__.__name__}" for name, module in model.named_modules()],
    )
    write_json(
        output_dir / "inspection.json",
        {
            "model_id": args.model_id,
            "model_class": model.__class__.__name__,
            "identified_modules": module_info,
            "model_forward_args": list(inspect.signature(model.forward).parameters.keys()),
        },
    )

    print(f"vision_encoder={module_info['vision_encoder']}")
    print(f"merger={module_info['merger']}")
    print(f"decoder_layers={len(module_info['decoder_layers'])}")
    print(f"deepstack_modules={module_info['deepstack_modules']}")

    from engine.datasets import iter_mme_examples
    examples = list(
        iter_mme_examples(
            Path(args.mme_root),
            split=args.mme_split,
            cache_dir=args.mme_cache_dir,
            max_items=args.mme_max,
            streaming=args.mme_streaming,
            category=args.mme_category,
        )
    )

    from engine.profiler import profile_run

    for idx, (img_path, question, _meta) in enumerate(examples, start=1):
        batch = prepare_inputs(
            processor,
            question=question,
            image_path=str(img_path),
            add_generation_prompt=(args.mode == "generate"),
        )
        batch = move_to_device(batch, pick_device(model))

        if args.all_modules:
            targets = [name for name, _ in model.named_modules() if name]
        else:
            targets = (
                module_info["vision_encoder"]
                + module_info["merger"]
                + module_info["deepstack_modules"]
                + module_info["decoder_layers"]
            )

        sample_dir = output_dir / f"sample_{idx}"
        sample_dir.mkdir(exist_ok=True)
        profile = profile_run(
            model,
            batch,
            target_names=targets,
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
            sync_each_hook=not args.no_sync_cuda,
        )
        profile["hooked_modules"] = targets
        write_json(sample_dir / "profile.json", profile)
        print(f"sample {idx} total_ms={profile['total_ms']:.3f}")
        print(f"sample {idx} peak_memory_bytes={profile['peak_memory_bytes']}")
        for item in profile["aggregated"][:10]:
            print(
                f"{item['module_name']}: calls={item['calls']} total_ms={item['total_ms']:.3f} avg_ms={item['avg_ms']:.3f}"
            )
        print("---")



if __name__ == "__main__":
    main()
