import torch

from config import (
    TOKENIZER_PATH,
    VOCAB_SIZE,
    D_MODEL,
    NUM_HEADS,
    INTERMEDIATE_SIZE,
    NUM_BLOCKS,
    MAX_SEQ_LEN,
)
from model.gpt import GPT
from tokenizer.bpe import ByteLevelBPETokenizer


CHECKPOINT_PATH = "checkpoints/best_model.pt"

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.9


def get_device():
    """Select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_tokenizer():
    """Load the trained tokenizer."""
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.load(TOKENIZER_PATH)
    return tokenizer


def load_model(device):
    """Load the trained Mini GPT checkpoint."""
    model = GPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        num_blocks=NUM_BLOCKS,
    ).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model


def apply_top_k(logits: torch.Tensor, top_k: int | None):
    """Keep only the top-k highest-scoring tokens."""
    if top_k is None:
        return logits

    top_k = min(top_k, logits.size(-1))

    top_k_values, _ = torch.topk(logits, top_k, dim=-1)

    kth_value = top_k_values[:, -1].unsqueeze(-1)

    return torch.where(
        logits < kth_value,
        torch.full_like(logits, float("-inf")),
        logits,
    )


def apply_top_p(logits: torch.Tensor, top_p: float | None):
    """Keep the smallest set of tokens whose probability mass reaches top-p."""
    if top_p is None:
        return logits

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
        dim=-1,
    )

    sorted_probs = torch.softmax(sorted_logits, dim=-1)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_remove = cumulative_probs > top_p

    sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
    sorted_remove[:, 0] = False

    remove_mask = torch.zeros_like(sorted_remove, dtype=torch.bool)

    remove_mask.scatter_(
        1,
        sorted_indices,
        sorted_remove,
    )

    return logits.masked_fill(
        remove_mask,
        float("-inf"),
    )


@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_new_tokens, temperature, top_k, top_p):
    """
    Generate text using autoregressive decoding with KV caching.

    The prompt is processed once. Each subsequent token uses
    the cached keys and values from previous tokens.
    """
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be between 0 and 1")

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    token_ids = tokenizer.encode(prompt)

    if len(token_ids) >= MAX_SEQ_LEN:
        raise ValueError(
            f"Prompt is too long: {len(token_ids)} tokens. "
            f"Maximum context length is {MAX_SEQ_LEN}."
        )

    generated_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    # Prefill: process the complete prompt once and build the KV cache.
    logits, past_key_values = model(
        generated_ids,
        use_cache=True,
    )

    eos_id = tokenizer.special_tokens["<|endoftext|>"]

    max_tokens = min(
        max_new_tokens,
        MAX_SEQ_LEN - generated_ids.size(1),
    )

    for _ in range(max_tokens):
        next_token_logits = logits[:, -1, :]

        # Apply temperature.
        next_token_logits = next_token_logits / temperature

        # Apply top-k filtering.
        next_token_logits = apply_top_k(
            next_token_logits,
            top_k,
        )

        # Apply top-p filtering.
        next_token_logits = apply_top_p(
            next_token_logits,
            top_p,
        )

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1,
        )

        # Sample the next token from the filtered distribution.
        next_token_id = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        # Stop when the model generates EOS.
        if next_token_id.item() == eos_id:
            break

        generated_ids = torch.cat(
            [generated_ids, next_token_id],
            dim=1,
        )

        # Decode only the new token using the existing KV cache.
        logits, past_key_values = model(
            next_token_id,
            past_key_values=past_key_values,
            use_cache=True,
        )

    return tokenizer.decode(generated_ids[0].tolist())


def main():
    device = get_device()

    print(f"Device: {device}")

    tokenizer = load_tokenizer()
    model = load_model(device)

    prompt = "who wanted to play"

    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
    )

    print()
    print("Prompt:")
    print(prompt)

    print()
    print("Temperature:", TEMPERATURE)
    print("Top K:", TOP_K)
    print("Top P:", TOP_P)

    print()
    print("Generated text:")
    print(generated_text)


if __name__ == "__main__":
    main()