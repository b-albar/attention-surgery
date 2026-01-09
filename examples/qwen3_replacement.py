"""
Example: Replace attention or RMSNorm in a Qwen3 model.

This demonstrates the replacement workflow:
1. Load a pretrained Qwen3 model
2. Replace Attention with SigmoidAttention OR RMSNorm with DynamicTanh
"""

import torch
import copy

from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RMSNorm

from attention_surgery.utils import replace_modules
from attention_surgery.models import (
    create_attention_replacement,
    create_norm_replacement,
)
from attention_surgery.ops.dynamic_tanh import DynamicTanh
from attention_surgery.ops.sigmoid_attention import SigmoidAttention


def replace_attention(model_name: str = "Qwen/Qwen2.5-0.5B"):
    """Replace attention layers with SigmoidAttention."""
    from transformers import AutoModelForCausalLM

    print(f"Loading {model_name}...")
    original = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    modified = copy.deepcopy(original)

    replacement_fn = create_attention_replacement(SigmoidAttention, original.config)
    replaced = replace_modules(modified, Qwen2Attention, replacement_fn)
    print(f"Replaced {len(replaced)} attention modules")

    print("\nOriginal:", original.model.layers[0].self_attn)
    print("\nModified:", modified.model.layers[0].self_attn)

    return original, modified


def replace_rmsnorm(model_name: str = "Qwen/Qwen2.5-0.5B"):
    """Replace RMSNorm layers with DynamicTanh."""
    from transformers import AutoModelForCausalLM

    print(f"Loading {model_name}...")
    original = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    modified = copy.deepcopy(original)

    replacement_fn = create_norm_replacement(
        DynamicTanh,
        channels_last=True,
        alpha_init_value=0.5,
    )
    replaced = replace_modules(modified, Qwen2RMSNorm, replacement_fn)
    print(f"Replaced {len(replaced)} RMSNorm modules")

    print("\nOriginal:", original.model.layers[0].input_layernorm)
    print("\nModified:", modified.model.layers[0].input_layernorm)

    return original, modified


def main():
    print("=" * 60)
    print("Example 1: Replacing Attention Layers")
    print("=" * 60)

    try:
        replace_attention()
        print("\n✓ Attention replacement successful!")
    except Exception as e:
        print(f"\n✗ Error: {e}")

    print("\n" + "=" * 60)
    print("Example 2: Replacing RMSNorm Layers")
    print("=" * 60)

    try:
        replace_rmsnorm()
        print("\n✓ RMSNorm replacement successful!")
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    main()
