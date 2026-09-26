import torch
import torch.nn as nn

from .rmsnorm import RMSNorm
from .attention import Attention
from .swiglu import SwiGLU

class TransformerBlock(nn.Module):
    """
    Transformer decoder block.

    Applies:
    RMSNorm → Attention → Residual
    RMSNorm → SwiGLU → Residual
    """

    def __init__(self, d_model: int, num_heads: int, intermediate_size: int):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size

        self.input_rmsnorm = RMSNorm(d_model)
        self.attention = Attention(d_model, num_heads)

        self.output_rmsnorm = RMSNorm(d_model)
        self.swiglu = SwiGLU(d_model, intermediate_size)

    def forward(self, x: torch.Tensor, past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None, use_cache: bool = False):
        """
        Input [B,T,D]
         ↓
        RMSNorm → [B,T,D]
         ↓
        Multi-Head Self-Attention + RoPE
         ↓
        Attention output [B,T,D]
         ↓
        Residual connection → [B,T,D]
         ↓
        RMSNorm → [B,T,D]
         ↓
        SwiGLU → [B,T,D]
         ↓
        Residual connection → [B,T,D]
         ↓
        Output [B,T,D]
        """

        residual = x
        x = self.input_rmsnorm(x)

        if use_cache:
            x, present_key_value = self.attention(x, past_key_value= past_key_value, use_cache=True)
        else:
            x = self.attention(x)
        
        x = residual + x

        residual = x
        x = self.output_rmsnorm(x)
        x = self.swiglu(x)
        x = residual + x

        if use_cache:
            return x, present_key_value

        return x
