import torch
import torch.nn as nn

from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock

class GPT(nn.Module):

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, intermediate_size: int, num_blocks: int):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.num_blocks = num_blocks

        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, intermediate_size) for _ in range(num_blocks)])

        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, past_key_values: tuple[tuple[torch.Tensor, torch.Tensor] | None, ...] | None = None, use_cache: bool = False):
        """
        Token IDs
        [B,T]
         ↓
        Token Embedding
        [B,T,D]
         ↓
        Transformer Blocks × N
        [B,T,D]
         ↓
        Final RMSNorm
        [B,T,D]
         ↓
        LM Head
        [B,T,V]
         ↓
        Logits
        """

        x = self.embed_tokens(input_ids)

        present_key_values = []

        for block_idx, block in enumerate(self.blocks):
            past_key_value = None

            if past_key_values is not None:
                past_key_value = past_key_values[block_idx]

            if use_cache:
                x, present_key_value = block(x, past_key_value=past_key_value, use_cache=True)
                present_key_values.append(present_key_value)
            else:
                x = block(x)

        x = self.norm(x)
        x = self.lm_head(x)

        if use_cache:
            return x, tuple(present_key_values)

        return x