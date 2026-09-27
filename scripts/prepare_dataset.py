from pathlib import Path
import json
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TRAIN_PROCESSED_PATH,
    VAL_PROCESSED_PATH,
    TOKENIZER_PATH,
    MAX_SEQ_LEN,
)
from tokenizer.bpe import ByteLevelBPETokenizer


def prepare_dataset(data_path: str, output_path: str, tokenizer: ByteLevelBPETokenizer, max_seq_len: int):
    """
    Tokenize, pack, and save a dataset for causal language-model training.

    Each document is terminated with an EOS token, all tokens are
    concatenated into one continuous stream, and the stream is split
    into fixed-length input and target sequences.
    """
    all_token_ids = []

    with open(data_path, "r", encoding="utf-8") as file:
        for line in file:
            story = json.loads(line)["text"]
            token_ids = tokenizer.encode(story)

            # Mark the end of each story.
            token_ids.append(tokenizer.special_tokens["<|endoftext|>"])

            all_token_ids.extend(token_ids)

    tokens = torch.tensor(all_token_ids, dtype=torch.long)

    # Create continuous next token input and target streams
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    num_sequences = len(input_tokens) // max_seq_len

    input_tokens = input_tokens[:num_sequences * max_seq_len]
    target_tokens = target_tokens[:num_sequences * max_seq_len]

    input_ids = input_tokens.reshape(num_sequences, max_seq_len)
    target_ids = target_tokens.reshape(num_sequences, max_seq_len)

    data = {
        "input_ids": input_ids,
        "target_ids": target_ids,
    }

    output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(data, output_path)

    print(f"Saved: {output_path}")
    print(f"Input shape:  {input_ids.shape}")
    print(f"Target shape: {target_ids.shape}")
    print(f"Total tokens: {len(tokens)}")


def main():
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.load(TOKENIZER_PATH)

    prepare_dataset(TRAIN_DATA_PATH, TRAIN_PROCESSED_PATH, tokenizer, MAX_SEQ_LEN)
    prepare_dataset(VAL_DATA_PATH, VAL_PROCESSED_PATH, tokenizer, MAX_SEQ_LEN)


if __name__ == "__main__":
    main()