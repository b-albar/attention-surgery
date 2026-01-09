"""
Generic replacement functions for attention surgery.

This module provides utilities to create replacement functions
for attention and normalization layers.
"""

from typing import Callable


def create_attention_replacement(
    replacement_class: type,
    config,
    **kwargs,
) -> Callable:
    """
    Create a replacement function that converts attention layers.

    Args:
        replacement_class: The class to use for the replacement module.
            Must accept: feature_dim, num_heads, num_kv_heads, head_dim,
            Wq, Wk, Wv, Wo as constructor arguments.
        config: The model configuration from transformers
        **kwargs: Additional keyword arguments to pass to the replacement constructor.

    Returns:
        A function that takes an attention module and returns a replacement module
    """

    def replacement_fn(attention_module):
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = getattr(config, "head_dim", hidden_size // num_heads)

        new_attention = replacement_class(
            feature_dim=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            Wq=attention_module.q_proj,
            Wk=attention_module.k_proj,
            Wv=attention_module.v_proj,
            Wo=attention_module.o_proj,
            rotary_fn=getattr(attention_module, "rotary_fn", None),
            scale=getattr(attention_module, "scaling", None),
            **kwargs,
        )

        return new_attention

    return replacement_fn


def create_norm_replacement(
    replacement_class: type,
    **kwargs,
) -> Callable:
    """
    Create a replacement function that converts normalization layers.

    Args:
        replacement_class: The class to use for the replacement module.
            Must accept 'normalized_shape' as a constructor argument.
        **kwargs: Additional keyword arguments to pass to the replacement constructor.

    Returns:
        A function that takes a normalization module and returns a replacement module
    """

    def replacement_fn(norm_module):
        normalized_shape = norm_module.weight.shape

        new_norm = replacement_class(
            normalized_shape=normalized_shape,
            **kwargs,
        )

        return new_norm

    return replacement_fn
