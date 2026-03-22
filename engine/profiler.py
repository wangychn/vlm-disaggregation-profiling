import time
import torch
from engine.utils import summarize


def sync_cuda(enabled: bool, value) -> None:

    if not enabled or not torch.cuda.is_available():
        return
    tensors = []

    def collect(item):
        if isinstance(item, torch.Tensor):
            tensors.append(item)
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)

    collect(value)
    for tensor in tensors:
        if tensor.device.type == "cuda":
            torch.cuda.synchronize(tensor.device)


def _token_count(value) -> int | None:
    if isinstance(value, torch.Tensor) and value.ndim >= 2:
        return int(value.shape[-1])
    if hasattr(value, "sequences") and isinstance(value.sequences, torch.Tensor) and value.sequences.ndim >= 2:
        return int(value.sequences.shape[-1])
    return None


def profile_run(model, batch, target_names: list[str], mode: str, max_new_tokens: int, sync_each_hook: bool):
    """Run a single inference pass with module-level timing and memory stats.

    This helper registers forward pre/post hooks on a subset of ``model``
    submodules identified by ``target_names``. Each hooked module records a
    timestamp before execution and again after, optionally synchronizing CUDA
    to ensure accurate GPU timings.  

    Args:
        model: A ``torch.nn.Module`` instance to execute.
        batch: Input batch dict (usually produced by a ``processor``).
        target_names: List of qualified module names to instrument.
        mode: Either ``"forward"`` for a straight call or ``"generate"`` to
            invoke ``model.generate`` with ``max_new_tokens``.
        max_new_tokens: Passed through to ``model.generate`` when applicable.
        sync_each_hook: If ``True`` invoke ``torch.cuda.synchronize`` in each
            hook to prevent asynchronous GPU timing from skewing results.

    Returns:
        A dict containing:
            * ``mode``: same as the input
            * ``total_ms``: wall-clock time for the full run
            * ``peak_memory_bytes``: CUDA peak memory if available
            * ``events``: raw per-hook timing records
            * ``aggregated``: summary statistics per module
            * ``output_type``: class name of the model output
    """

    events = []
    starts: dict[str, list[float]] = {}
    handles = []

    def pre_hook(name):
        def hook(_module, inputs):
            sync_cuda(sync_each_hook, inputs)
            starts.setdefault(name, []).append(time.perf_counter())

        return hook

    def post_hook(name):
        def hook(module, inputs, output):
            sync_cuda(sync_each_hook, output)

            start = None
            if starts.get(name):
                start = starts[name].pop()
            else:
                start = time.perf_counter()
            events.append(
                {
                    "module_name": name,
                    "module_class": module.__class__.__name__,
                    "elapsed_ms": (time.perf_counter() - start) * 1000.0,
                    "input_summary": summarize(inputs),
                    "output_summary": summarize(output),
                }
            )

        return hook

    for name, module in model.named_modules():
        if name in target_names:
            handles.append(module.register_forward_pre_hook(pre_hook(name)))
            handles.append(module.register_forward_hook(post_hook(name)))

    # reset CUDA stats
    if torch.cuda.is_available():
        for param in model.parameters():
            if param.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(param.device)
                torch.cuda.synchronize(param.device)
                break

    # execute model run
    started = time.perf_counter()
    with torch.no_grad():
        if mode == "forward":
            output = model(**batch)
        else:
            output = model.generate(**batch, max_new_tokens=max_new_tokens)

    # ensure all kernels have finished
    if torch.cuda.is_available():
        for param in model.parameters():
            if param.device.type == "cuda":
                torch.cuda.synchronize(param.device)
                break

    for handle in handles:
        handle.remove()

    aggregated = {}
    for event in events:
        item = aggregated.setdefault(
            event["module_name"],
            {
                "module_name": event["module_name"],
                "module_class": event["module_class"],
                "calls": 0,
                "total_ms": 0.0,
            },
        )
        item["calls"] += 1
        item["total_ms"] += event["elapsed_ms"]

    summary = sorted(aggregated.values(), key=lambda item: item["total_ms"], reverse=True)
    for item in summary:
        item["avg_ms"] = item["total_ms"] / item["calls"]

    peak_memory = None
    if torch.cuda.is_available():
        for param in model.parameters():
            if param.device.type == "cuda":
                peak_memory = int(torch.cuda.max_memory_allocated(param.device))
                break

    input_token_count = _token_count(batch.get("input_ids")) if isinstance(batch, dict) else None
    output_token_count = _token_count(output)
    generated_token_count = None
    if mode == "generate" and input_token_count is not None and output_token_count is not None:
        generated_token_count = max(0, output_token_count - input_token_count)

    return {
        "mode": mode,
        "total_ms": (time.perf_counter() - started) * 1000.0,
        "peak_memory_bytes": peak_memory,
        "input_token_count": input_token_count,
        "output_token_count": output_token_count,
        "generated_token_count": generated_token_count,
        "events": events,
        "aggregated": summary,
        "output_type": output.__class__.__name__,
    }
