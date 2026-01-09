import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Type

from ..utils.module_collector import ModuleCollector
from ..utils.module_surgery import wrap_modules, _set_module

logger = logging.getLogger(__name__)


def train_with_module_mse(
    original_model: nn.Module,
    modified_model: nn.Module,
    original_target_type: Type[nn.Module],
    replacement_target_type: Type[nn.Module],
    dataloader: DataLoader,
    epochs: int = 1,
    lr: float = 1e-4,
    device: str = "cuda",
    module_loss_weight: float = 1.0,
    output_loss_weight: float = 0.0,
    block_size: int | None = None,
    epochs_per_block: int | None = None,
) -> None:
    """
    Train replacement modules using per-module MSE loss.

    This wraps both original and replacement modules with ModuleCollector
    to capture their outputs during forward pass. MSE loss is computed
    between corresponding module outputs for more direct supervision.

    Args:
        original_model: Model with original modules (frozen)
        modified_model: Model with replacement modules (to train)
        original_target_type: Type of original modules to collect outputs from
        replacement_target_type: Type of replacement modules to train
        dataloader: Training data
        epochs: Number of training epochs (used when block_size is None)
        lr: Learning rate
        device: Device to train on
        module_loss_weight: Weight for per-module MSE loss (default: 1.0)
        output_loss_weight: Weight for full model output MSE (default: 0.0)
        block_size: If set, train modules in blocks of this size sequentially
        epochs_per_block: Epochs per block (required if block_size is set)
    """
    original_model.eval()
    original_model.to(device)
    modified_model.to(device)

    # Freeze original model
    for param in original_model.parameters():
        param.requires_grad = False

    # Freeze all of modified model initially
    for param in modified_model.parameters():
        param.requires_grad = False

    # Wrap original modules with collectors
    original_collectors = wrap_modules(original_model, original_target_type)
    if not original_collectors:
        logger.warning("No original target modules found.")
        return

    # Find replacement modules and wrap with collectors
    replacement_modules = []
    replacement_collectors = []
    replacement_names = []
    for name, module in list(modified_model.named_modules()):
        if isinstance(module, replacement_target_type):
            collector = ModuleCollector(module)
            replacement_collectors.append(collector)
            replacement_modules.append(module)
            replacement_names.append(name)
            _set_module(modified_model, name, collector)

    if not replacement_collectors:
        logger.warning("No replacement target modules found.")
        return

    if len(original_collectors) != len(replacement_collectors):
        logger.warning(
            f"Number of original modules ({len(original_collectors)}) != "
            f"replacement modules ({len(replacement_collectors)})"
        )

    logger.info(
        f"Wrapped {len(original_collectors)} original and {len(replacement_collectors)} replacement modules"
    )

    # Check if any replacement modules have LoRA layers
    has_lora = False
    for module in replacement_modules:
        for child in module.modules():
            if hasattr(child, "lora_A") and hasattr(child, "lora_B"):
                has_lora = True
                break

    def enable_module_params(module: nn.Module) -> None:
        """Enable trainable params for a module (respects LoRA)."""
        if has_lora:
            for child in module.modules():
                if hasattr(child, "lora_A"):
                    for param in child.lora_A.parameters():
                        param.requires_grad = True
                if hasattr(child, "lora_B"):
                    for param in child.lora_B.parameters():
                        param.requires_grad = True
        else:
            for param in module.parameters():
                param.requires_grad = True

    def disable_module_params(module: nn.Module) -> None:
        """Disable trainable params for a module."""
        if has_lora:
            for child in module.modules():
                if hasattr(child, "lora_A"):
                    for param in child.lora_A.parameters():
                        param.requires_grad = False
                if hasattr(child, "lora_B"):
                    for param in child.lora_B.parameters():
                        param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False

    def compute_module_loss(
        active_indices: list[int], criterion: nn.Module
    ) -> torch.Tensor:
        """Compute MSE loss for active module pairs."""
        module_loss = torch.tensor(0.0, device=device)
        count = 0
        for i in active_indices:
            if i >= len(original_collectors) or i >= len(replacement_collectors):
                continue
            orig_output = original_collectors[i].output
            repl_output = replacement_collectors[i].output

            if orig_output is not None and repl_output is not None:
                # Handle tuple outputs (e.g., attention returns (hidden_states, ...))
                if isinstance(orig_output, tuple) and isinstance(repl_output, tuple):
                    module_loss = module_loss + criterion(
                        repl_output[0], orig_output[0]
                    )
                else:
                    module_loss = module_loss + criterion(repl_output, orig_output)
                count += 1

        if count > 0:
            module_loss = module_loss / count
        return module_loss

    def train_epoch(
        active_indices: list[int],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> tuple[float, float, float]:
        """Run one training epoch for active modules."""
        total_loss = 0.0
        total_module_loss = 0.0
        total_output_loss = 0.0

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Clear collectors
            for c in original_collectors:
                c.clear()
            for c in replacement_collectors:
                c.clear()

            # Forward pass through both models
            with torch.no_grad():
                original_out = original_model(**batch)

            modified_out = modified_model(**batch)

            # Compute per-module MSE loss for active modules only
            module_loss = compute_module_loss(active_indices, criterion)

            # Always compute full model output loss for monitoring
            with torch.no_grad():
                output_mse = criterion(modified_out.detach(), original_out)
            total_output_loss += output_mse.item()

            # Only include in gradient if weight > 0
            if output_loss_weight > 0:
                output_loss = criterion(modified_out, original_out)
                loss = (
                    module_loss_weight * module_loss + output_loss_weight * output_loss
                )
            else:
                loss = module_loss_weight * module_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_module_loss += module_loss.item()

        return total_loss, total_module_loss, total_output_loss

    criterion = nn.MSELoss()

    # Block-wise training
    if block_size is not None:
        if epochs_per_block is None:
            epochs_per_block = epochs  # Default to epochs if not specified

        num_modules = len(replacement_modules)
        num_blocks = (num_modules + block_size - 1) // block_size

        for block_idx in range(num_blocks):
            start_idx = block_idx * block_size
            end_idx = min(start_idx + block_size, num_modules)
            active_indices = list(range(start_idx, end_idx))
            current_block = [replacement_modules[i] for i in active_indices]

            logger.info(
                f"Training block {block_idx + 1}/{num_blocks}: modules {start_idx} to {end_idx - 1}"
            )

            # Enable params for current block only
            for module in current_block:
                enable_module_params(module)

            # Create optimizer for current block
            trainable_params = [
                p for p in modified_model.parameters() if p.requires_grad
            ]
            optimizer = torch.optim.AdamW(trainable_params, lr=lr)

            # Train current block
            for epoch in range(epochs_per_block):
                total_loss, total_module, total_output = train_epoch(
                    active_indices, optimizer, criterion
                )
                avg_loss = total_loss / len(dataloader)
                avg_module = total_module / len(dataloader)
                avg_output = total_output / len(dataloader)
                logger.info(
                    f"  Epoch {epoch + 1}/{epochs_per_block}, "
                    f"Loss: {avg_loss:.6f}, Module: {avg_module:.6f}, Logit MSE: {avg_output:.6f}"
                )

            # Freeze current block after training
            for module in current_block:
                disable_module_params(module)

    # All modules training (no blocks)
    else:
        all_indices = list(range(len(replacement_modules)))

        # Enable all replacement module params
        for module in replacement_modules:
            enable_module_params(module)

        trainable_params = [p for p in modified_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        for epoch in range(epochs):
            total_loss, total_module, total_output = train_epoch(
                all_indices, optimizer, criterion
            )
            avg_loss = total_loss / len(dataloader)
            avg_module = total_module / len(dataloader)
            avg_output = total_output / len(dataloader)
            logger.info(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Loss: {avg_loss:.6f}, Module: {avg_module:.6f}, Logit MSE: {avg_output:.6f}"
            )

    # Unwrap collectors back to original modules
    for name, module in list(original_model.named_modules()):
        if isinstance(module, ModuleCollector):
            _set_module(original_model, name, module.module)

    for name, module in list(modified_model.named_modules()):
        if isinstance(module, ModuleCollector):
            _set_module(modified_model, name, module.module)

    logger.info("Training complete. Collectors unwrapped.")
