import torch.nn as nn
from typing import Any


class ModuleCollector(nn.Module):
    """Wraps a module to collect its inputs and outputs during forward pass."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.inputs: tuple | None = None
        self.output: Any = None

    def forward(self, *args, **kwargs):
        self.inputs = (args, kwargs)
        self.output = self.module(*args, **kwargs)
        return self.output

    def clear(self):
        self.inputs = None
        self.output = None
