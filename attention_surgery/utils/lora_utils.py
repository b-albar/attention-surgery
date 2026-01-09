"""
LoRA (Low-Rank Adaptation) utilities for attention surgery.

This module provides a simple LoRA implementation for fine-tuning
linear layers with reduced parameter count.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Set


class LoRALinear(nn.Module):
    """
    LoRA wrapper for a linear layer.

    Implements Low-Rank Adaptation: output = Wx + (BAx) * scaling
    where A and B are low-rank matrices.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            original_linear: The original linear layer to wrap
            rank: Rank of the low-rank decomposition
            alpha: Scaling factor (scaling = alpha / rank)
            dropout: Dropout probability for LoRA layers
        """
        super().__init__()

        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features
        device = original_linear.weight.device
        dtype = original_linear.weight.dtype

        # LoRA matrices: A projects down, B projects up
        # Create on the same device and dtype as original
        self.lora_A = nn.Linear(
            in_features, rank, bias=False, device=device, dtype=dtype
        )
        self.lora_B = nn.Linear(
            rank, out_features, bias=False, device=device, dtype=dtype
        )
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward pass (frozen)
        result = self.original(x)
        # Add LoRA contribution
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return result + lora_out * self.scaling

    def merge(self) -> nn.Linear:
        """
        Merge LoRA weights back into the original linear layer.

        Returns:
            A new linear layer with merged weights.
        """
        merged = nn.Linear(
            self.original.in_features,
            self.original.out_features,
            bias=self.original.bias is not None,
            device=self.original.weight.device,
            dtype=self.original.weight.dtype,
        )

        # Compute merged weight: W + BA * scaling
        with torch.no_grad():
            merged_weight = (
                self.original.weight
                + (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            )
            merged.weight.copy_(merged_weight)
            if self.original.bias is not None:
                merged.bias.copy_(self.original.bias)

        return merged


def inject_lora(
    module: nn.Module,
    target_names: Optional[Set[str]] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> List[str]:
    """
    Inject LoRA adapters into specified linear layers of a module.

    Args:
        module: The module to modify (e.g., SigmoidAttention)
        target_names: Set of attribute names to wrap with LoRA (e.g., {"Wq", "Wk", "Wv", "Wo"}).
                      If None, wraps all linear layers.
        rank: LoRA rank
        alpha: LoRA alpha (scaling = alpha / rank)
        dropout: Dropout probability

    Returns:
        List of attribute names that were wrapped with LoRA
    """
    wrapped = []

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if target_names is None or name in target_names:
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, name, lora_layer)
                wrapped.append(name)

    return wrapped


def merge_lora(module: nn.Module) -> List[str]:
    """
    Merge all LoRA layers back into regular linear layers.

    Args:
        module: The module containing LoRA layers

    Returns:
        List of attribute names that were merged
    """
    merged = []

    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            merged_layer = child.merge()
            setattr(module, name, merged_layer)
            merged.append(name)

    return merged


def inject_lora_recursive(
    model: nn.Module,
    target_module_type: type,
    target_linear_names: Optional[Set[str]] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> int:
    """
    Recursively inject LoRA into all modules of a specific type.

    Args:
        model: The root model to search
        target_module_type: Type of modules to inject LoRA into (e.g., SigmoidAttention)
        target_linear_names: Names of linear layers to wrap (e.g., {"Wq", "Wk", "Wv", "Wo"})
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: Dropout probability

    Returns:
        Total number of linear layers wrapped
    """
    total_wrapped = 0

    for module in model.modules():
        if isinstance(module, target_module_type):
            wrapped = inject_lora(
                module,
                target_names=target_linear_names,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            total_wrapped += len(wrapped)

    return total_wrapped


def merge_lora_recursive(model: nn.Module) -> int:
    """
    Recursively merge all LoRA layers in a model.

    Args:
        model: The root model to search

    Returns:
        Total number of LoRA layers merged
    """
    total_merged = 0

    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                merged_layer = child.merge()
                setattr(module, name, merged_layer)
                total_merged += 1

    return total_merged
