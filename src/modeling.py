"""Model and tokenizer loading, plus the size ladder used in the paper.

The ladder is the backbone of the central claim. The hypothesis is that seed
sigma shrinks as parameter count grows, so the ladder has to span roughly an
order of magnitude in size while staying cheap enough to train ten times per
rung.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Approximate parameter counts, used for the x axis of figure 2a. Verify each
# one with `count_parameters` once the model is loaded, and report the measured
# number in the paper rather than this table.
MODEL_LADDER: List[Dict[str, Any]] = [
    {"name": "EleutherAI/pythia-160m", "params_m": 160, "family": "pythia"},
    {"name": "HuggingFaceTB/SmolLM2-135M", "params_m": 135, "family": "smollm2"},
    {"name": "EleutherAI/pythia-410m", "params_m": 410, "family": "pythia"},
    {"name": "Qwen/Qwen2.5-0.5B", "params_m": 500, "family": "qwen2.5"},
    {"name": "HuggingFaceTB/SmolLM2-360M", "params_m": 360, "family": "smollm2"},
    {"name": "EleutherAI/pythia-1.4b", "params_m": 1400, "family": "pythia"},
    {"name": "Qwen/Qwen2.5-1.5B", "params_m": 1500, "family": "qwen2.5"},
]


def resolve_dtype(dtype_name: str):
    """Map a config string to a torch dtype."""
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(
            f"unknown dtype {dtype_name!r}; choose from {sorted(mapping)}"
        )
    return mapping[dtype_name]


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(
    model_name: str,
    dtype: str = "float32",
    trust_remote_code: bool = False,
    device: str | None = None,
) -> Tuple[Any, Any]:
    """Load a causal LM and its tokenizer, ready for training or scoring."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        # Most of these small models ship without a pad token. Reusing EOS is
        # standard; the loss mask in train.py keeps it out of the objective.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=resolve_dtype(dtype),
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    return model, tokenizer


def count_parameters(model, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
