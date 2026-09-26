import torch
import torch.nn as nn
import math

from .rope import RoPE

class Attention(nn.Module):
    """
    Multi-head self-attention.

    Projects input into Q, K, and V,
    applies RoPE to Q and K,
    then performs causal self-attention.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias= False)
        self.k_proj = nn.Linear(d_model, d_model, bias= False)
        self.v_proj = nn.Linear(d_model, d_model, bias= False)
        self.o_proj = nn.Linear(d_model, d_model, bias= False)

        self.rope = RoPE(self.head_dim)

    def forward(self, x: torch.Tensor, past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None, use_cache: bool = False):
        """
        Input
        [B, T, D]
            ↓
        Q / K / V projections
        [B, T, D]
            ↓
        Split into attention heads
        [B, T, H, head_dim]
            ↓
        Transpose
        [B, H, T, head_dim]
            ↓
        Apply RoPE to Q and K
        [B, H, T, head_dim]
            ↓
        Q @ Kᵀ / √head_dim
        [B, H, T, T]
            ↓
        Causal mask + softmax
        [B, H, T, T]
            ↓
        Attention x V
        [B, H, T, head_dim]
            ↓
        Merge attention heads
        [B, T, D]
            ↓
        Output projection
        [B, T, D]
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)

        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)

        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.transpose(1, 2)

        past_len = 0

        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.size(2) # past_k = [batch, num_heads, past_len or sequence, head_dim]

        q = self.rope(q, position_offset= past_len)
        k = self.rope(k, position_offset= past_len)

        if past_key_value is not None:
            k = torch.cat([past_k, k], dim= 2)
            v = torch.cat([past_v, v], dim= 2)

        scores = (q @ k.transpose(-2, -1) / math.sqrt(self.head_dim))

        total_len = k.size(2)

        query_positions = torch.arange(past_len, past_len + seq_len, device= x.device)
        key_positions = torch.arange(total_len, device= x.device)

        causal_mask = (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1))
        masked = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = masked.softmax(dim=-1)
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        
        output = self.o_proj(attn_output)

        if use_cache:
            return output, (k, v)
        
        return output