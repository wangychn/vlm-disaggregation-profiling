from __future__ import annotations

import torch
from torch.profiler import ProfilerActivity, profile, record_function


def _register_ranges(model, target_names: list[str]):
    handles = []
    target_names = set(target_names)

    for name, module in model.named_modules():
        if name not in target_names:
            continue

        stack = []

        def pre_hook(_module, _inputs, *, name=name, stack=stack):
            ctx = record_function(name)
            ctx.__enter__()
            stack.append(ctx)

        def post_hook(_module, _inputs, _output, *, stack=stack):
            if stack:
                stack.pop().__exit__(None, None, None)

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))
    return handles


def trace_run(
    model,
    batch,
    trace_path: str,
    target_names: list[str] | None = None,
    mode: str = "forward",
    max_new_tokens: int = 16,
):
    handles = _register_ranges(model, target_names or [])
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        ) as prof, torch.no_grad(), record_function(f"model.{mode}"):
            if mode == "forward":
                output = model(**batch)
            else:
                output = model.generate(**batch, max_new_tokens=max_new_tokens)
    finally:
        for handle in handles:
            handle.remove()

    prof.export_chrome_trace(trace_path)
    return {
        "trace_path": trace_path,
        "output": output,
        "output_type": output.__class__.__name__,
        "table": prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30),
    }
