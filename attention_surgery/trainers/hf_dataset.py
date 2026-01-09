"""
HuggingFace dataset utilities for attention surgery training.

This module provides flexible dataset classes for loading and preprocessing
datasets from HuggingFace for use in module replacement training.
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, Callable


class HuggingFaceTextDataset(Dataset):
    """
    Dataset that loads text data from HuggingFace datasets.

    Supports various text formats:
    - Simple text datasets (wikitext, c4, etc.) via text_column
    - Instruction datasets (alpaca, dolly, etc.) via format_fn
    """

    def __init__(
        self,
        tokenizer,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        split: str = "train",
        seq_length: int = 512,
        max_samples: Optional[int] = None,
        text_column: Optional[str] = None,
        format_fn: Optional[Callable] = None,
        add_eos: bool = True,
        streaming: bool = False,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            dataset_name: Name of the dataset on HuggingFace (e.g., "tatsu-lab/alpaca")
            dataset_config: Dataset configuration/subset (optional)
            split: Dataset split (train, validation, test)
            seq_length: Maximum sequence length (longer sequences are truncated)
            max_samples: Maximum number of samples (None for all)
            text_column: Column name containing text (required if format_fn not provided)
            format_fn: Function to format each example into text.
                       Signature: format_fn(example: dict) -> str
                       Use create_instruct_format_fn() to create one.
            add_eos: Whether to add EOS token at the end of sequences
            streaming: Whether to use streaming mode for large datasets
        """
        from datasets import load_dataset

        if text_column is None and format_fn is None:
            raise ValueError("Either text_column or format_fn must be provided")

        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.add_eos = add_eos

        # Load dataset
        print(f"  Loading dataset: {dataset_name}")
        if dataset_config:
            print(f"  Config: {dataset_config}")

        load_kwargs = {"split": split}
        if streaming:
            load_kwargs["streaming"] = True

        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, **load_kwargs)
        else:
            dataset = load_dataset(dataset_name, **load_kwargs)

        # Process samples
        print("  Processing samples...")
        self.samples = []

        if streaming:
            iterator = iter(dataset)
        else:
            iterator = dataset

        for i, item in enumerate(iterator):
            if max_samples and i >= max_samples:
                break

            # Get text from the example
            if format_fn:
                text = format_fn(item)
            else:
                text = item[text_column]

            if not text or not text.strip():
                continue

            # Tokenize
            tokens = tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=seq_length,
            )

            # Add EOS if requested and not already present
            if add_eos and tokenizer.eos_token_id is not None:
                if len(tokens) == 0 or tokens[-1] != tokenizer.eos_token_id:
                    tokens = tokens[: seq_length - 1] + [tokenizer.eos_token_id]

            # Pad to seq_length
            attention_mask = [1] * len(tokens)
            if len(tokens) < seq_length:
                padding_length = seq_length - len(tokens)
                tokens = tokens + [tokenizer.pad_token_id or 0] * padding_length
                attention_mask = attention_mask + [0] * padding_length

            self.samples.append(
                {
                    "input_ids": torch.tensor(tokens, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                }
            )

        print(f"  Created {len(self.samples)} samples of max length {seq_length}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def create_instruct_format_fn(
    instruction_col: str = "instruction",
    input_col: Optional[str] = "input",
    output_col: Optional[str] = "output",
    instruction_prefix: str = "### Instruction:\n",
    input_prefix: str = "\n\n### Input:\n",
    output_prefix: str = "\n\n### Response:\n",
    include_output: bool = True,
) -> Callable[[dict], str]:
    """
    Create a format function for instruction-style datasets.

    Args:
        instruction_col: Column name for instruction text
        input_col: Column name for optional input/context (None to skip)
        output_col: Column name for response/output (None to skip)
        instruction_prefix: Prefix before instruction text
        input_prefix: Prefix before input text (only used if input exists)
        output_prefix: Prefix before output text
        include_output: Whether to include output in formatted text

    Returns:
        A format function: (example: dict) -> str

    Example:
        # For Alpaca-style datasets
        format_fn = create_instruct_format_fn(
            instruction_col="instruction",
            input_col="input",
            output_col="output",
        )

        # For datasets with different column names
        format_fn = create_instruct_format_fn(
            instruction_col="question",
            input_col="context",
            output_col="answer",
        )

        # For prompt-only (no response)
        format_fn = create_instruct_format_fn(
            instruction_col="instruction",
            include_output=False,
        )
    """

    def format_fn(example: dict) -> str:
        parts = []

        # Instruction (required)
        instruction = example.get(instruction_col, "")
        parts.append(f"{instruction_prefix}{instruction}")

        # Input (optional)
        if input_col:
            input_text = example.get(input_col, "")
            if input_text and input_text.strip():
                parts.append(f"{input_prefix}{input_text}")

        # Output
        if include_output and output_col:
            output = example.get(output_col, "")
            parts.append(f"{output_prefix}{output}")
        else:
            parts.append(output_prefix.rstrip())

        return "".join(parts)

    return format_fn


def collate_fn(batch):
    """Standard collate function for HuggingFace datasets."""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
    }
