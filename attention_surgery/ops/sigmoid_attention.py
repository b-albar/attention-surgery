import torch
import torch.nn as nn
import numpy as np


class SigmoidAttention(nn.Module):
    """
    Sigmoid-based attention mechanism with support for Grouped Query Attention (GQA).

    In GQA, K and V projections have fewer heads than Q, which are repeated to match Q.
    This is more memory efficient than standard multi-head attention.
    """

    def __init__(
        self,
        feature_dim,
        num_heads,
        num_kv_heads=None,
        head_dim=None,
        Wq=None,
        Wk=None,
        Wv=None,
        Wo=None,
        rotary_fn=None,
        scale=None,
    ):
        """
        Args:
            feature_dim: Hidden dimension of the model
            num_heads: Number of query attention heads
            num_kv_heads: Number of key/value attention heads (for GQA). Defaults to num_heads.
            head_dim: Dimension per head. If None, inferred from feature_dim // num_heads.
            Wq, Wk, Wv, Wo: Optional pre-existing projection layers
            rotary_fn: Optional rotary position encoding function
            scale: Optional scaling factor for attention scores
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Calculate head_dim - either provided or inferred
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            self.head_dim = feature_dim // num_heads

        # Q projects to num_heads * head_dim
        q_dim = self.num_heads * self.head_dim
        # K, V project to num_kv_heads * head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.Wq = Wq if Wq is not None else nn.Linear(feature_dim, q_dim)
        self.Wk = Wk if Wk is not None else nn.Linear(feature_dim, kv_dim)
        self.Wv = Wv if Wv is not None else nn.Linear(feature_dim, kv_dim)
        self.Wo = Wo if Wo is not None else nn.Linear(feature_dim, feature_dim)

        # Optional rotary position encoding function
        # Should be a callable: (q, k, **kwargs) -> (q_rotated, k_rotated)
        self.rotary_fn = rotary_fn

        # Scaling factor for attention scores (default: 1/sqrt(head_dim))
        self.scale = scale if scale is not None else (1.0 / np.sqrt(self.head_dim))

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeat KV heads to match the number of query heads (for GQA).

        Args:
            x: Tensor of shape (batch, num_kv_heads, seq_len, head_dim)

        Returns:
            Tensor of shape (batch, num_heads, seq_len, head_dim)
        """
        if self.num_kv_groups == 1:
            return x
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x = x.unsqueeze(2)  # (batch, num_kv_heads, 1, seq_len, head_dim)
        x = x.expand(batch, num_kv_heads, self.num_kv_groups, seq_len, head_dim)
        x = x.reshape(batch, self.num_heads, seq_len, head_dim)
        return x

    def forward(
        self,
        hidden_states=None,
        attention_mask=None,
        position_embeddings=None,
        **kwargs,
    ):
        """
        Forward pass

        Args:
            hidden_states: Input tensor (batch, seq_len, feature_dim)
            attention_mask: Optional causal attention mask
            position_embeddings: Tuple of (cos, sin) for rotary embeddings
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Tuple of (output, None) to match transformers attention interface
        """
        if hidden_states is None:
            raise ValueError("'hidden_states' must be provided")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Compute sigmoid bias: -log(seq_len) to normalize sigmoid attention
        # This helps sigmoid attention sum to approximately 1 like softmax
        seq_len = input_shape[-1]  # Get sequence length
        sigmoid_bias = torch.tensor(
            -float(torch.log(torch.tensor(float(seq_len)))),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # Project and reshape
        query_states = self.Wq(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.Wk(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.Wv(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply rotary position encoding
        if self.rotary_fn is not None and position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = self.rotary_fn(
                query_states, key_states, cos, sin
            )

        # Repeat KV for GQA
        key_states = self._repeat_kv(key_states)
        value_states = self._repeat_kv(value_states)

        # Compute attention weights
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        )

        # Apply causal mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Use SIGMOID with trainable bias instead of softmax
        attn_weights = torch.sigmoid(attn_weights + sigmoid_bias)

        attn_output = torch.matmul(attn_weights, value_states)

        # Transpose and reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.Wo(attn_output)

        return attn_output, None
