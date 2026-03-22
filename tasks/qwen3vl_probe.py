from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from engine.utils import load_qwen3vl, prepare_inputs

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


def image_size(image_path: Path) -> tuple[int | None, int | None]:
    from PIL import Image

    with Image.open(image_path) as image:
        return image.width, image.height


def short_text(value: str, limit: int = 120) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def main() -> None:
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

    for idx, (img_path, question, meta) in enumerate(examples, start=1):
        width, height = image_size(img_path)
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
        payload = {
            "sample": {
                "sample_index": idx,
                "repeat_index": 1,
                "image_path": str(img_path),
                "image_width": width,
                "image_height": height,
                "question": question,
                "question_length_chars": len(question),
                "mme": {
                    "category": meta.get("category"),
                    "question_id": meta.get("question_id"),
                    "answer": meta.get("answer"),
                },
            },
            "run": {
                "model_id": args.model_id,
                "dtype": args.dtype,
                "mode": args.mode,
                "device_map": args.device_map,
                "max_new_tokens": args.max_new_tokens,
                "sync_cuda": not args.no_sync_cuda,
                "input_token_count": profile["input_token_count"],
                "output_token_count": profile["output_token_count"],
                "generated_token_count": profile["generated_token_count"],
                "total_ms": profile["total_ms"],
                "peak_memory_bytes": profile["peak_memory_bytes"],
                "output_type": profile["output_type"],
            },
            "hooked_modules": targets,
            "top_modules": profile["aggregated"][:10],
            "module_summary": profile["aggregated"],
            "events": profile["events"],
        }
        write_json(sample_dir / "profile.json", payload)
        print(f"[sample {idx}]")
        print(f"  category={meta.get('category')} question_id={meta.get('question_id')}")
        print(f"  image={img_path.name} size={width}x{height}")
        print(f"  question={short_text(question)}")
        print(
            f"  mode={args.mode} total_ms={profile['total_ms']:.3f} peak_memory_bytes={profile['peak_memory_bytes']} "
            f"generated_tokens={profile['generated_token_count']}"
        )
        print("  top_modules:")
        for item in payload["top_modules"][:5]:
            print(
                f"    - {item['module_name']}: calls={item['calls']} total_ms={item['total_ms']:.3f} avg_ms={item['avg_ms']:.3f}"
            )
        print()



if __name__ == "__main__":
    main()
