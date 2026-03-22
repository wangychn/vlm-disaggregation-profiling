from pathlib import Path
import json

def write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")

def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

def torch_dtype(name: str):
    if name == "auto":
        return "auto"
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]

# ==== Model loading functions ====


def load_qwen3vl(model_id: str, dtype: str, device_map: str, trust_remote_code: bool):
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

    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
    )
    
    if isinstance(batch, dict):
        batch.pop("token_type_ids", None)
    return batch


# ==== Device functions ====

def pick_device(model):
    import torch
    """Choose a device where the model lives.

    Some HuggingFace models expose ``hf_device_map`` when they are loaded with
    :func:`from_pretrained` and have been dispatched across multiple devices.
    We inspect that map and return the first GPU device we see with fallbacks.
    """
    if hasattr(model, "hf_device_map"):
        for value in model.hf_device_map.values():
            if isinstance(value, int):
                return torch.device(f"cuda:{value}")
            if isinstance(value, str) and value not in {"cpu", "disk", "meta"}:
                return torch.device(value)
    return next(model.parameters()).device

def move_to_device(value, device):
    import torch
    """Recursively move tensors (and containers) onto the given device.

    The batch produced by the processor is often a nested dict/list/tuple of
    tensors. This helper walks the structure and calls ``.to(device)`` on any
    tensor it finds, while preserving the original container type. It also
    handles objects with a ``to`` method to support custom objects.

    Args:
        value: arbitrary Python object returned by the processor.
        device: a ``torch.device`` instance where tensors should be moved.

    Returns:
        A new structure mirroring ``value`` but with all tensors relocated to
        ``device``.
    """
    if hasattr(value, "to"):
        try:
            return value.to(device)
        except TypeError:
            pass
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


# ==== Model visualization functions ====


def summarize(value):
    """Produce a lightweight description of tensors and nested structures.

    Used by the profiling logic to record the shape/dtype/device of tensors
    flowing through a module, while avoiding serialization of full data.
    Walks through common container types and truncates lists or
    dicts to the first eight elements so the summary stays compact.

    Arguments:
        value: any Python object, frequently a torch.Tensor or a nested batch
            dict returned by a processor or a module.
    """
    import torch

    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, dict):
        return {str(key): summarize(item) for key, item in list(value.items())[:8]}
    if isinstance(value, (list, tuple)):
        return [summarize(item) for item in list(value)[:8]]
    return value.__class__.__name__ if hasattr(value, "__dict__") else value


def render_tree(module, name: str = "<root>", depth: int = 0, lines: list[str] | None = None) -> list[str]:
    """Recursively walk a PyTorch module hierarchy.

    Each line in the returned list contains the module name (with indentation
    proportional to its depth) and the class name of the submodule. 
    """
    if lines is None:
        lines = []
    lines.append(f"{'  ' * depth}{name}: {module.__class__.__name__}")
    for child_name, child in module.named_children():
        render_tree(child, child_name, depth + 1, lines)
    return lines


def identify_modules(model) -> dict[str, list[str]]:
    """Locate important submodules inside a vision-language model.

    The returned dictionary groups names of modules that are commonly of
    interest when profiling or inspecting Qwen3-VL-style architectures:

    - **vision_encoder**: the visual trunk, if present
    - **merger**: module responsible for combining visual and text features
    - **decoder_layers**: language model transformer layers (numbered)
    - **deepstack_modules**: any module whose name contains "deepstack"
    """
    names = {name for name, _ in model.named_modules()}
    return {
        "vision_encoder": [name for name in ("model.visual", "visual") if name in names],
        "merger": [
            name
            for name in (
                "model.visual.merger",
                "visual.merger",
                "model.visual.deepstack_merger_list",
                "visual.deepstack_merger_list",
            )
            if name in names
        ],
        "decoder_layers": sorted(
            name
            for name in names
            if ".language_model.layers." in name and name.split(".")[-1].isdigit()
        ),
        "deepstack_modules": sorted(name for name in names if "deepstack" in name.lower()),
    }



