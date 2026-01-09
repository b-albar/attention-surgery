"""
Example: Replace LayerNorm with DynamicTanh in a simple model.

This shows the basic workflow:
1. Create/load a model
2. Replace target modules
3. Train replacement to mimic original
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy

from attention_surgery.utils import replace_modules
from attention_surgery.ops.dynamic_tanh import DynamicTanh
from attention_surgery.trainers import train_replacement


# Simple test model
class SimpleModel(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.norm(x)
        x = self.linear2(x)
        return x


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create original model
    original = SimpleModel(dim=64)

    # Create modified copy
    modified = copy.deepcopy(original)

    # Replace LayerNorm -> DynamicTanh
    def ln_to_dyt(module: nn.LayerNorm) -> DynamicTanh:
        return DynamicTanh(module.normalized_shape, channels_last=True)

    replace_modules(modified, nn.LayerNorm, ln_to_dyt)

    print("Original model:", original)
    print("\nModified model:", modified)

    # Fake training data
    data = torch.randn(100, 64)
    dataset = TensorDataset(data)

    # Wrapper to match expected format
    class DataLoaderWrapper:
        def __init__(self, dataset, batch_size=16):
            self.loader = DataLoader(dataset, batch_size=batch_size)

        def __iter__(self):
            for (batch,) in self.loader:
                yield {"x": batch}

        def __len__(self):
            return len(self.loader)

    # Adjust models for dict input
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)

    loader = DataLoaderWrapper(dataset)

    # Train
    train_replacement(
        original_model=ModelWrapper(original),
        modified_model=ModelWrapper(modified),
        target_type=DynamicTanh,
        dataloader=loader,
        epochs=10,
        lr=1e-3,
        device=device,
    )


if __name__ == "__main__":
    main()
