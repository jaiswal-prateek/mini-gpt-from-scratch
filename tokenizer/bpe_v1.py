import json

class ByteLevelBPETokenizer:
    """
    Educational byte-level BPE tokenizer.

    Implements the core BPE training and tokenization process from scratch,
    including vocabulary building, merge rules, merge ranks, special tokens,
    encoding, decoding, and tokenizer persistence.

    This implementation prioritizes clarity and learning over performance.
    """

    def __init__(self):
        self.vocab = {}
        self.merges = {}
        self.merge_ranks = {}
        self.special_tokens = {}
        self.next_token_id = 0

    def initialize_vocab(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.next_token_id = 256

    def add_special_tokens(self, special_tokens):
        for token in special_tokens:
            self.special_tokens[token] = self.next_token_id
            self.next_token_id += 1

    def count_pairs(self, corpus):
        pair_counts = {}

        for tokens in corpus:
            for pair in zip(tokens, tokens[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        return pair_counts

    def merge_pair(self, tokens, pair, new_token_id):
        merged = []
        i = 0

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and (tokens[i], tokens[i + 1]) == pair
            ):
                merged.append(new_token_id)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1

        return merged

    def train(self, texts, vocab_size):
        if not self.vocab:
            self.initialize_vocab()

        max_merges = vocab_size - 256 - len(self.special_tokens)

        corpus = [list(text.encode("utf-8")) for text in texts]

        for rank in range(max_merges):
            pair_counts = self.count_pairs(corpus)

            if not pair_counts:
                break

            pair = max(pair_counts, key=pair_counts.get)

            new_token_id = self.next_token_id
            self.next_token_id += 1

            self.merges[pair] = new_token_id
            self.merge_ranks[pair] = rank

            self.vocab[new_token_id] = (self.vocab[pair[0]] + self.vocab[pair[1]])

            corpus = [self.merge_pair(tokens, pair, new_token_id) for tokens in corpus]

            # print(
            #     f"Merge {rank + 1}/{max_merges}: "
            #     f"{pair} -> {new_token_id}"
            # )

    def _encode_text(self, text):
        tokens = list(text.encode("utf-8"))

        while True:
            pair_candidates = []

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])

                if pair in self.merge_ranks:
                    pair_candidates.append(
                        (self.merge_ranks[pair], pair)
                    )

            if not pair_candidates:
                break

            _, best_pair = min(pair_candidates)

            tokens = self.merge_pair(
                tokens,
                best_pair,
                self.merges[best_pair]
            )

        return tokens

    def encode(self, text):
        tokens = []
        remaining = text

        for special_token, token_id in self.special_tokens.items():
            parts = remaining.split(special_token)

            for i, part in enumerate(parts):
                tokens.extend(self._encode_text(part))

                if i < len(parts) - 1:
                    tokens.append(token_id)

            remaining = ""

        if remaining:
            tokens.extend(self._encode_text(remaining))

        return tokens

    def decode(self, tokens):
        reverse_special_tokens = {
            token_id: token
            for token, token_id in self.special_tokens.items()
        }

        text = ""
        byte_data = b""

        for token in tokens:
            if token in reverse_special_tokens:
                text += byte_data.decode("utf-8")
                byte_data = b""
                text += reverse_special_tokens[token]
            else:
                byte_data += self.vocab[token]

        text += byte_data.decode("utf-8")

        return text

    def save(self, path):
        data = {
            "vocab": {
                str(token_id): list(token_bytes)
                for token_id, token_bytes in self.vocab.items()
            },
            "merges": [
                {
                    "pair": list(pair),
                    "token_id": token_id
                }
                for pair, token_id in self.merges.items()
            ],
            "merge_ranks": [
                {
                    "pair": list(pair),
                    "rank": rank
                }
                for pair, rank in self.merge_ranks.items()
            ],
            "special_tokens": self.special_tokens,
            "next_token_id": self.next_token_id
        }

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.vocab = {
            int(token_id): bytes(token_bytes)
            for token_id, token_bytes in data["vocab"].items()
        }

        self.merges = {
            tuple(item["pair"]): item["token_id"]
            for item in data["merges"]
        }

        self.merge_ranks = {
            tuple(item["pair"]): item["rank"]
            for item in data["merge_ranks"]
        }

        self.special_tokens = data["special_tokens"]
        self.next_token_id = data["next_token_id"]