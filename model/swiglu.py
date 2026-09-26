import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network.

    Projects the input through up and gate paths,
    applies SiLU to the gate, then projects back.
    """

    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()

        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        
        up = self.up_proj(x)
        gate = F.silu(self.gate_proj(x))
        hidden = up * gate
        output = self.down_proj(hidden)

        return output