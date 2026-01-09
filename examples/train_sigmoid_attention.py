"""
Example: Train Qwen3 Attention -> SigmoidAttention replacement with random sequences.

This demonstrates the full workflow:
1. Load a pretrained Qwen3 model
2. Replace Qwen2Attention with SigmoidAttention
3. Optionally apply LoRA to Q/K/V/O projections
4. Train the replacement to mimic the original using random token sequences
"""

import argparse
import torch
import copy
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from attention_surgery.utils import (
    replace_modules,
    inject_lora_recursive,
    merge_lora_recursive,
    wrap_with_scale,
)
from attention_surgery.ops.sigmoid_attention import SigmoidAttention
from attention_surgery.models import (
    get_qwen3_attention_class,
    create_sigmoid_attention_replacement,
)
from attention_surgery.trainers import train_with_module_mse


class TextDataset(Dataset):
    """Dataset that loads real text from HuggingFace datasets."""

    def __init__(
        self,
        tokenizer,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        split: str = "train",
        seq_length: int = 128,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            dataset_name: Name of the dataset on HuggingFace
            dataset_config: Dataset configuration
            split: Dataset split (train, validation, test)
            seq_length: Length of each sequence
            max_samples: Maximum number of samples (None for all)
        """
        from datasets import load_dataset  # type: ignore

        self.seq_length = seq_length
        self.tokenizer = tokenizer

        # Load dataset
        print(f"  Loading {dataset_name}/{dataset_config} ({split})...")
        dataset = load_dataset(dataset_name, dataset_config, split=split)

        # Tokenize all text
        print("  Tokenizing...")
        all_tokens = []
        for item in dataset:
            text = item["text"]
            if text.strip():  # Skip empty lines
                tokens = tokenizer.encode(text, add_special_tokens=False)
                all_tokens.extend(tokens)

        # Create fixed-length chunks
        self.samples = []
        for i in range(0, len(all_tokens) - seq_length, seq_length):
            chunk = all_tokens[i : i + seq_length]
            if len(chunk) == seq_length:
                self.samples.append(torch.tensor(chunk, dtype=torch.long))
                if max_samples and len(self.samples) >= max_samples:
                    break

        print(f"  Created {len(self.samples)} samples of length {seq_length}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids = self.samples[idx]
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class RandomTokenDataset(Dataset):
    """Dataset that generates random token sequences for training."""

    def __init__(
        self,
        vocab_size: int,
        seq_length: int = 128,
        num_samples: int = 1000,
        seed: int = 42,
    ):
        """
        Args:
            vocab_size: Size of the tokenizer vocabulary
            seq_length: Length of each sequence
            num_samples: Number of random sequences to generate
            seed: Random seed for reproducibility
        """
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples

        # Generate random sequences
        torch.manual_seed(seed)
        self.input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(num_samples, seq_length),
            dtype=torch.long,
        )
        # Create attention mask (all tokens are valid)
        self.attention_mask = torch.ones(num_samples, seq_length, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def collate_fn(batch):
    """Collate function for the dataloader."""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
    }


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main():
    """Run the training example."""
    parser = argparse.ArgumentParser(description="Train SigmoidAttention replacement")
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA for Q/K/V/O projections instead of full fine-tuning",
    )
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=64, help="Sequence length")
    parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of samples"
    )
    parser.add_argument(
        "--merge-lora", action="store_true", help="Merge LoRA weights after training"
    )
    parser.add_argument(
        "--layer-by-layer",
        action="store_true",
        help="Train each layer sequentially (better convergence)",
    )
    parser.add_argument(
        "--use-wikitext",
        action="store_true",
        help="Use wikitext dataset instead of random tokens",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="HuggingFace dataset name (when --use-wikitext)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration (when --use-wikitext)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="List of layer indices to replace (e.g., --layers 0 1 2). If not specified, all layers are replaced.",
    )
    parser.add_argument(
        "--use-scale",
        action="store_true",
        help="Wrap replacement modules with learnable output scaling",
    )
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Configuration
    model_name = "Qwen/Qwen2.5-0.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("Qwen3 Attention -> SigmoidAttention Training Example")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_length}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Use LoRA: {args.use_lora}")
    if args.use_lora:
        print(f"  LoRA rank: {args.lora_rank}")
        print(f"  LoRA alpha: {args.lora_alpha}")
        print(f"  Merge LoRA after training: {args.merge_lora}")
    if args.layers:
        print(f"  Replacing layers: {args.layers}")
    else:
        print("  Replacing layers: all")

    # Load tokenizer (just to get vocab size)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size
    print(f"  Vocabulary size: {vocab_size}")

    # Load original model
    print(f"\nLoading {model_name}...")
    # Use eager attention to match our SigmoidAttention implementation
    original = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,  # Use float32 for training stability
        device_map=None,  # Manual device placement
        attn_implementation="eager",  # Match our attention implementation
    )
    original = original.to(device)
    print("  Loaded successfully!")

    trainable, total = count_parameters(original)
    print(f"  Total parameters: {total:,}")

    # Create modified copy
    print("\nCreating modified model with SigmoidAttention...")
    modified = copy.deepcopy(original)

    # Get attention class and create replacement function
    Qwen2Attention = get_qwen3_attention_class()
    replacement_fn = create_sigmoid_attention_replacement(original.config)

    # Replace attention modules (only specified layers if --layers is set)
    replaced = replace_modules(
        modified, Qwen2Attention, replacement_fn, layer_indices=args.layers
    )
    print(f"  Replaced {len(replaced)} attention modules")

    # Count SigmoidAttention modules
    num_sigmoid = sum(1 for m in modified.modules() if isinstance(m, SigmoidAttention))
    print(f"  SigmoidAttention modules in modified model: {num_sigmoid}")

    # Apply learnable output scaling if requested
    if args.use_scale:
        print("\nApplying learnable output scaling...")
        scaled = wrap_with_scale(modified, SigmoidAttention, init_scale=1.0)
        print(f"  Wrapped {len(scaled)} modules with ScaledModule")

    # Apply LoRA if requested
    if args.use_lora:
        print("\nApplying LoRA to Q/K/V/O projections...")
        # Target the Wq, Wk, Wv, Wo linear layers in SigmoidAttention
        num_lora = inject_lora_recursive(
            modified,
            target_module_type=SigmoidAttention,
            target_linear_names={"Wq", "Wk", "Wv", "Wo"},
            rank=args.lora_rank,
            alpha=args.lora_alpha,
        )
        print(f"  Applied LoRA to {num_lora} linear layers")

        # Show parameter reduction
        trainable_before = sum(
            p.numel() for p in modified.parameters() if p.requires_grad
        )
        # After LoRA, only LoRA params should be trainable
        for param in modified.parameters():
            param.requires_grad = False
        # Enable LoRA params
        for module in modified.modules():
            if hasattr(module, "lora_A"):
                for param in module.lora_A.parameters():
                    param.requires_grad = True
            if hasattr(module, "lora_B"):
                for param in module.lora_B.parameters():
                    param.requires_grad = True

        trainable_after, _ = count_parameters(modified)
        print(f"  Trainable parameters with LoRA: {trainable_after:,}")
        print(
            f"  Parameter reduction: {(1 - trainable_after / trainable_before) * 100:.1f}%"
        )

    # Create dataset
    if args.use_wikitext:
        print("\nLoading text dataset from HuggingFace...")
        dataset = TextDataset(
            tokenizer=tokenizer,
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            split="train",
            seq_length=args.seq_length,
            max_samples=args.num_samples,
        )
    else:
        print("\nCreating random token dataset...")
        dataset = RandomTokenDataset(
            vocab_size=vocab_size,
            seq_length=args.seq_length,
            num_samples=args.num_samples,
            seed=42,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Number of batches: {len(dataloader)}")

    # Verify outputs before training
    print("\nVerifying model outputs before training...")
    sample_batch = next(iter(dataloader))
    sample_batch = {k: v.to(device) for k, v in sample_batch.items()}

    with torch.no_grad():
        original_out = original(**sample_batch)
        modified_out = modified(**sample_batch)

        # Compare logits
        if hasattr(original_out, "logits"):
            mse_before = torch.nn.functional.mse_loss(
                modified_out.logits, original_out.logits
            ).item()
            print(f"  MSE loss before training: {mse_before:.6f}")

    # Train the replacement
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    # Wrap models for training (the trainer expects dict input/output)
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask=None, **kwargs):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits  # Return just the logits for MSE loss

    wrapped_original = ModelWrapper(original)
    wrapped_modified = ModelWrapper(modified)

    # Get the original attention class for module collection
    Qwen2Attention = get_qwen3_attention_class()

    if args.layer_by_layer:
        print("\nUsing layer-by-layer training...")
        train_with_module_mse(
            original_model=wrapped_original,
            modified_model=wrapped_modified,
            original_target_type=Qwen2Attention,
            replacement_target_type=SigmoidAttention,
            dataloader=dataloader,
            criterion=torch.nn.MSELoss(),
            block_size=1,  # Train one layer at a time
            epochs_per_block=args.epochs,
            lr=args.lr,
            device=device,
            module_loss_weight=1.0,
            output_loss_weight=0.01,
        )
    else:
        print("\nUsing per-module training (all layers)...")
        train_with_module_mse(
            original_model=wrapped_original,
            modified_model=wrapped_modified,
            original_target_type=Qwen2Attention,
            replacement_target_type=SigmoidAttention,
            dataloader=dataloader,
            criterion=torch.nn.MSELoss(),
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            module_loss_weight=1.0,
            output_loss_weight=0.01,
        )

    # Merge LoRA if requested
    if args.use_lora and args.merge_lora:
        print("\nMerging LoRA weights...")
        num_merged = merge_lora_recursive(modified)
        print(f"  Merged {num_merged} LoRA layers")

    # Verify outputs after training
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)

    print("\nVerifying model outputs after training...")
    modified.eval()
    with torch.no_grad():
        modified_out = modified(**sample_batch)

        if hasattr(original_out, "logits"):
            mse_after = torch.nn.functional.mse_loss(
                modified_out.logits, original_out.logits
            ).item()
            print(f"  MSE loss after training: {mse_after:.6f}")

    # Text generation comparison
    print("\n" + "=" * 70)
    print("Text Generation Comparison")
    print("=" * 70)

    test_prompts = [
        "The capital of France is",
        "In the year 2050, artificial intelligence will",
        "To make a good pizza, you need",
    ]

    original.eval()
    modified.eval()

    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 50)

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate with original model
        with torch.no_grad():
            # NOTE: use_cache=False because our SigmoidAttention doesn't implement KV cache
            original_output = original.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=False,
            )
        original_text = tokenizer.decode(original_output[0], skip_special_tokens=True)

        # Generate with modified model
        with torch.no_grad():
            modified_output = modified.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=False,
            )
        modified_text = tokenizer.decode(modified_output[0], skip_special_tokens=True)

        print(f"Original: {original_text}")
        print(f"Modified: {modified_text}")

        # Check if they match
        if original_text == modified_text:
            print("✓ Outputs match!")
        else:
            print("✗ Outputs differ")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
