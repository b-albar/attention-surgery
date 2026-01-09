import torch
import torch.nn as nn
from typing import Callable, Type, Any

from .module_collector import ModuleCollector


class ScaledModule(nn.Module):
    """
    Wraps a module and applies a learnable scale to its output.

    This helps match output magnitudes between original and replacement modules
    without modifying the replacement module's implementation.
    """

    def __init__(self, module: nn.Module, init_scale: float = 1.0):
        super().__init__()
        self.module = module
        self.scale = nn.Parameter(torch.tensor(init_scale))

    def forward(self, *args, **kwargs) -> Any:
        output = self.module(*args, **kwargs)
        # Handle tuple outputs (e.g., attention returns (hidden_states, ...))
        if isinstance(output, tuple):
            return (output[0] * self.scale,) + output[1:]
        return output * self.scale


def wrap_with_scale(
    model: nn.Module,
    target_type: Type[nn.Module],
    init_scale: float = 1.0,
) -> list[ScaledModule]:
    """
    Wrap all modules of target_type with ScaledModule for learnable output scaling.

    Args:
        model: Model to modify
        target_type: Type of modules to wrap
        init_scale: Initial scale value

    Returns:
        List of ScaledModule wrappers
    """
    scaled_modules = []

    for name, module in list(model.named_modules()):
        if isinstance(module, target_type):
            scaled = ScaledModule(module, init_scale)
            scaled_modules.append(scaled)
            _set_module(model, name, scaled)

    return scaled_modules


def unwrap_scaled_modules(model: nn.Module) -> int:
    """
    Remove ScaledModule wrappers, keeping the inner module.
    Note: This discards the learned scale.

    Returns:
        Number of modules unwrapped
    """
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, ScaledModule):
            _set_module(model, name, module.module)
            count += 1
    return count


def wrap_modules(
    model: nn.Module,
    target_type: Type[nn.Module],
) -> list[ModuleCollector]:
    """Wrap all modules of target_type with ModuleCollector."""
    collectors = []

    for name, module in model.named_modules():
        if isinstance(module, target_type):
            collector = ModuleCollector(module)
            collectors.append(collector)
            # Replace in parent
            _set_module(model, name, collector)

    return collectors


def replace_modules(
    model: nn.Module,
    target_type: Type[nn.Module],
    replacement_fn: Callable[[nn.Module], nn.Module],
    layer_indices: list[int] | None = None,
) -> list[nn.Module]:
    """
    Replace modules of target_type using replacement_fn.

    Args:
        model: Model to modify
        target_type: Type of modules to replace
        replacement_fn: Function that takes original module and returns replacement
        layer_indices: If specified, only replace modules at these indices.
                      If None, replace all modules of target_type.

    Returns:
        List of new replacement modules
    """
    new_modules = []

    # Find all matching modules first
    matching_modules = []
    for name, module in list(model.named_modules()):
        if isinstance(module, target_type):
            matching_modules.append((name, module))

    # Determine which indices to replace
    if layer_indices is None:
        indices_to_replace = set(range(len(matching_modules)))
    else:
        indices_to_replace = set(layer_indices)

    # Replace only specified indices
    for idx, (name, module) in enumerate(matching_modules):
        if idx in indices_to_replace:
            new_module = replacement_fn(module)
            new_modules.append(new_module)
            _set_module(model, name, new_module)

    return new_modules


def _set_module(model: nn.Module, name: str, new_module: nn.Module):
    """Set a submodule by its dotted name path."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)
