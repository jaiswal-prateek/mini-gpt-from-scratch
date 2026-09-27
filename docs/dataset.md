# Dataset Pipeline

The training data flows through a separate preprocessing stage before it reaches the model.

```text
data/raw/train.jsonl
        ↓
scripts/prepare_dataset.py
        │
        ├── tokenize
        ├── add EOS
        ├── concatenate
        ├── shift input/target
        └── pack into [N, MAX_SEQ_LEN]
        ↓
data/processed/train.pt
        ↓
data/dataset.py
        │
        └── load + expose samples
        ↓
DataLoader
        ↓
train.py
        ↓
Mini GPT
```

## Responsibilities

| Component | Responsibility |
|---|---|
| `data/raw/` | Original TinyStories data |
| `prepare_dataset.py` | Tokenization, EOS insertion, shifting, and sequence packing |
| `data/processed/` | Training-ready tensors |
| `dataset.py` | Load processed tensors and expose individual samples |
| `DataLoader` | Batch and shuffle samples |
| `train.py` | Forward pass, loss, backpropagation, optimization, and validation |

## Training Sample

After preprocessing, each sample contains:

```text
input_ids  → [T]
target_ids → [T]
```

For our current configuration:

```text
T = 512
```

The target sequence is shifted by one token:

```text
Input:   A B C D E
Target:  B C D E F
```

The model therefore learns to predict the next token at every position.

## Current Dataset

Training:

```text
Sequences: 9,634
Sequence length: 512
Shape: [9634, 512]
```

Validation:

```text
Sequences: 778
Sequence length: 512
Shape: [778, 512]
```

The final incomplete sequence from each continuous token stream is discarded because it does not contain a complete `MAX_SEQ_LEN` sequence.

## Design Principle

Preprocessing and training are deliberately separated.

```text
Raw data
   ↓
Preprocessing
   ↓
Training-ready data
   ↓
Dataset
   ↓
DataLoader
   ↓
Training
```

This means `train.py` does not need to tokenize or repack the raw dataset every time training starts.