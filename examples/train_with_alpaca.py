"""
Example: Train attention replacement using the Alpaca dataset.

Usage:
    python examples/train_with_alpaca.py
    python examples/train_with_alpaca.py --epochs 5 --max-samples 2000
"""

import argparse
import torch
import copy
from torch.utils.data import DataLoader

from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

from attention_surgery.utils import replace_modules
from attention_surgery.ops.sigmoid_attention import SigmoidAttention
from attention_surgery.models import create_attention_replacement
from attention_surgery.trainers import (
    train_with_module_mse,
    HuggingFaceTextDataset,
    create_instruct_format_fn,
    collate_fn,
)


def main():
    parser = argparse.ArgumentParser(description="Train with Alpaca dataset")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    original = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)

    # Create modified model with SigmoidAttention
    print("Creating modified model...")
    modified = copy.deepcopy(original)
    replacement_fn = create_attention_replacement(SigmoidAttention, original.config)
    replaced = replace_modules(modified, Qwen2Attention, replacement_fn)
    print(f"  Replaced {len(replaced)} attention modules")

    # Create format function for Alpaca dataset
    format_fn = create_instruct_format_fn(
        instruction_col="instruction",
        input_col="input",
        output_col="output",
    )

    # Load Alpaca dataset
    print("\nLoading Alpaca dataset...")
    dataset = HuggingFaceTextDataset(
        tokenizer=tokenizer,
        dataset_name="tatsu-lab/alpaca",
        seq_length=args.seq_length,
        max_samples=args.max_samples,
        format_fn=format_fn,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Measure MSE before training
    sample_batch = {k: v.to(device) for k, v in next(iter(dataloader)).items()}
    with torch.no_grad():
        orig_out = original(**sample_batch).logits
        mod_out = modified(**sample_batch).logits
        mse_before = torch.nn.functional.mse_loss(mod_out, orig_out).item()
    print(f"\nMSE before training: {mse_before:.6f}")

    # Train
    print("\nTraining...")

    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask=None, **kwargs):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    train_with_module_mse(
        original_model=ModelWrapper(original),
        modified_model=ModelWrapper(modified),
        original_target_type=Qwen2Attention,
        replacement_target_type=SigmoidAttention,
        dataloader=dataloader,
        criterion=torch.nn.MSELoss(),
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    # Measure MSE after training
    modified.eval()
    with torch.no_grad():
        mod_out = modified(**sample_batch).logits
        mse_after = torch.nn.functional.mse_loss(mod_out, orig_out).item()
    print(f"\nMSE after training: {mse_after:.6f}")
    print(f"Improvement: {(1 - mse_after / mse_before) * 100:.1f}%")

    # Quick generation test
    print("\n" + "=" * 50)
    print("Generation Test")
    print("=" * 50)

    prompt = "### Instruction:\nWhat is the capital of France?\n\n### Response:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        orig_text = tokenizer.decode(
            original.generate(**inputs, max_new_tokens=30, use_cache=False)[0],
            skip_special_tokens=True,
        )
        mod_text = tokenizer.decode(
            modified.generate(**inputs, max_new_tokens=30, use_cache=False)[0],
            skip_special_tokens=True,
        )

    print(f"Original: {orig_text}")
    print(f"Modified: {mod_text}")
    print("\nDone!")


if __name__ == "__main__":
    main()
