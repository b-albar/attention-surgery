
<div align="center">

# Attention Surgery - WIP

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
</div>

This project aims to explore alternative attention mechanisms to the standard softmax-based attention used in LLMs. Attention surgery refers to the process of replacing only the attention layers in a pre-trained LLM and re-training the attention layers to mimic the original model.

The goal is to obtain more efficient models while reaching close to the same performance as the original model.

## Supported Models

- **Qwen3** (via Qwen2 architecture from transformers)

## Replacement Modules

- **SigmoidAttention**: Replace softmax-based attention with sigmoid activation
- **DynamicTanh**: Replace normalization layers with learnable tanh activation

## Quick Start

```python
from transformers import AutoModelForCausalLM
from src.utils import replace_modules
from src.models import get_qwen3_attention_class, create_sigmoid_attention_replacement
import copy

# Load and clone model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
modified = copy.deepcopy(model)

# Replace attention modules
Qwen2Attention = get_qwen3_attention_class()
replacement_fn = create_sigmoid_attention_replacement(model.config)
replace_modules(modified, Qwen2Attention, replacement_fn)

# Train to mimic original (see docs for full example)
```

## Examples

```bash
# Simple module replacement demo
PYTHONPATH=. python examples/simple_replacement.py

# Qwen3 model replacement
PYTHONPATH=. python examples/qwen3_replacement.py
```

See [docs/qwen3.md](docs/qwen3.md) for detailed Qwen3 usage.
