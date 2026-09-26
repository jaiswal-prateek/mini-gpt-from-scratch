# Traditional positional encoding adds a position vector to the representation,
# whereas RoPE rotates pairs of dimensions in Q and K according to position.

import torch
import torch.nn as nn

class RoPE(nn.Module):
    """
    Rotary Positional Embedding.

    Applies position-dependent rotations to pairs
    of dimensions in Q and K vectors.
    """

    def __init__(self, head_dim: int, base: float= 10000.0):
        super().__init__()

        self.head_dim = head_dim
        self.base = base
        x = torch.arange(0, head_dim, 2, dtype=torch.float32)
        self.register_buffer("inv_freq", 1.0 / (base ** (x / head_dim)))

    def forward(self, x: torch.Tensor, position_offset: int = 0):
        """
        x shape:
        [batch_size, num_heads, sequence_length, head_dim]
        """

        batch_size, num_heads, seq_len, head_dim = x.shape
        positions = torch.arange(position_offset, position_offset + seq_len, device= x.device)

        angles = positions.unsqueeze(1) * self.inv_freq.unsqueeze(0)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        x_pairs = x.reshape(batch_size, num_heads, seq_len, head_dim // 2, 2)
        x1 = x_pairs[..., 0]
        x2 = x_pairs[..., 1]

        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x2 * cos + x1 * sin

        rotated_pairs = torch.stack([x1_rotated, x2_rotated], dim= -1)
        output = rotated_pairs.reshape(batch_size, num_heads, seq_len, head_dim)

        return output