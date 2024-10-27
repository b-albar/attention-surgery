from transformers import AutoModelForCausalLM

from modules.sigmoid_attention import SigmoidAttention

def load_llama(path, method="sigmoid_attention"):
    model = AutoModelForCausalLM.from_pretrained(path)

    for layer in model.model.layers:
        if method == "sigmoid_attention":
            layer.self_attn = SigmoidAttention(model.config.hidden_size, model.config.num_attention_heads, \
                Wq=layer.self_attn.q_proj.weight, \
                Wk=layer.self_attn.k_proj.weight, \
                Wv=layer.self_attn.v_proj.weight)
        else:
            raise ValueError(f"Unknown method: {method}")

    return model
