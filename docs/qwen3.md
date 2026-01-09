# Attention Surgery

Replace attention or normalization layers in transformer models.

## Quick Start

### 1. Replace Attention Layers

```python
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
import copy

from attention_surgery.utils import replace_modules
from attention_surgery.models import create_attention_replacement
from attention_surgery.ops.sigmoid_attention import SigmoidAttention

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
modified = copy.deepcopy(model)

# Create replacement function and replace
replacement_fn = create_attention_replacement(SigmoidAttention, model.config)
replace_modules(modified, Qwen2Attention, replacement_fn)
```

### 2. Replace Normalization Layers

```python
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from attention_surgery.models import create_norm_replacement
from attention_surgery.ops.dynamic_tanh import DynamicTanh

# Create replacement function with parameters
replacement_fn = create_norm_replacement(
    DynamicTanh,
    channels_last=True,
    alpha_init_value=0.5,
)
replace_modules(modified, Qwen2RMSNorm, replacement_fn)
```

### 3. Train Replacement

```python
from torch.utils.data import DataLoader
from attention_surgery.trainers import (
    train_with_module_mse,
    HuggingFaceTextDataset,
    create_instruct_format_fn,
    collate_fn,
)

# Create format function for instruction datasets
format_fn = create_instruct_format_fn(
    instruction_col="instruction",
    input_col="input",
    output_col="output",
)

# Load dataset
dataset = HuggingFaceTextDataset(
    tokenizer=tokenizer,
    dataset_name="tatsu-lab/alpaca",
    seq_length=256,
    max_samples=1000,
    format_fn=format_fn,
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Train
train_with_module_mse(
    original_model=original,
    modified_model=modified,
    original_target_type=Qwen2Attention,
    replacement_target_type=SigmoidAttention,
    dataloader=dataloader,
    criterion=torch.nn.MSELoss(),
    epochs=3,
    lr=1e-3,
    device="cuda",
)
```

### Format Function Options

```python
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

# Or use text_column for simple text datasets
dataset = HuggingFaceTextDataset(
    tokenizer=tokenizer,
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    text_column="text",
)
```

## Examples

See the `examples/` directory:

- `train_with_alpaca.py` - Train with Alpaca dataset
- `train_sigmoid_attention.py` - Train with wikitext or random tokens
- `qwen3_replacement.py` - Basic replacement examples
