# [WORK IN PROGRESS]

# Mini GPT From Scratch

A hands-on project to understand how a GPT-style language model works by building the major components from scratch in PyTorch.

The project starts with raw text and progressively builds:

```text
Raw Data
   ↓
Data Exploration
   ↓
Byte-Level BPE Tokenizer
   ↓
Token IDs
   ↓
Embeddings
   ↓
Transformer Architecture
   ↓
Attention + RoPE
   ↓
Feed-Forward Network
   ↓
Mini GPT
   ↓
Training
   ↓
Text Generation
```

The goal is not to build the largest or most capable model.

The goal is to understand what is happening inside a language model by implementing and experimenting with the individual components ourselves.

---

## What This Project Is

This is a learning-focused implementation of a small GPT-style decoder-only Transformer.

The project covers the path from:

```text
text
→ bytes
→ tokens
→ embeddings
→ attention
→ transformer blocks
→ logits
→ next-token prediction
→ trained language model
```

Most components are implemented directly in Python/PyTorch rather than hidden behind high-level libraries.

The implementation is intentionally small enough to inspect, modify, and experiment with.

---

# Project Roadmap

The project is being developed in stages.

```text
Phase 1
Data
  ↓
Explore and understand the training corpus

Phase 2
Tokenizer
  ↓
Build Byte-Level BPE from scratch

Phase 3
Transformer
  ↓
Build GPT architecture from scratch

Phase 4
Training
  ↓
Train Mini GPT on TinyStories

Phase 5
Inference
  ↓
Generate text from the trained model

Phase 6
Experiments
  ↓
Understand what changes model behavior and performance
```

The README will evolve with the project, so the sections below follow the same order.

---

# 1. Data Exploration

Before building a model, we first inspect the data.

The initial experiments use the TinyStories dataset.

The first notebook explores:

- number of stories
- text length
- character distribution
- common words/patterns
- training and validation data
- basic corpus statistics

The purpose of this stage is to understand what the model will actually be trained on rather than treating the dataset as a black box.

Notebook:

```text
notebooks/01_raw_data_exploration.ipynb
```

---

# 2. Building a Byte-Level BPE Tokenizer

A language model does not directly consume text.

The text must first be converted into token IDs.

The tokenizer learning path started from a simple toy BPE implementation and progressively became more realistic.

```text
Text
 ↓
UTF-8 bytes
 ↓
Initial byte vocabulary
 ↓
BPE merges
 ↓
Subword tokens
 ↓
Token IDs
```

The final tokenizer is implemented from scratch in:

```text
tokenizer/bpe.py
```

The tokenizer supports:

- 256 initial byte tokens
- BPE vocabulary expansion
- merge rules
- merge ranks
- pre-tokenization
- frequency-weighted BPE training
- special tokens
- encoding
- decoding
- save/load

---

## Tokenizer Evolution

Three implementations were explored while learning.

```text
BPE v1
   ↓
Naive full-corpus implementation

BPE v2
   ↓
Incremental pair-statistics experiment

Final bpe.py
   ↓
Pre-tokenized + frequency-weighted BPE
```

### v1 — Naive BPE

The first implementation repeatedly scanned the full corpus:

```text
Full corpus
   ↓
Count all pairs
   ↓
Select most frequent pair
   ↓
Merge across corpus
   ↓
Repeat
```

This was simple and useful for understanding the algorithm, but became expensive as the number of merges increased.

---

### v2 — Incremental Pair Statistics

The second implementation attempted to avoid repeatedly scanning everything by maintaining:

```text
pair_counts
pair_to_sequences
```

Only sequences affected by the selected pair were updated.

However, the additional Python bookkeeping introduced enough overhead that it was actually slower in our benchmark.

```text
~1M characters
512 vocabulary

v1 → 20.58 sec
v2 → 35.37 sec
```

This was kept as an optimization experiment because understanding why an optimization does not help is also useful.

---

### Final `bpe.py`

The final implementation changes the representation of the training corpus.

Instead of repeatedly processing every occurrence:

```text
Raw text
   ↓
Pre-tokenization
   ↓
Unique pieces + frequency
   ↓
Frequency-weighted pair counts
   ↓
BPE merges
```

For approximately 1M characters:

```text
Total pre-tokenized pieces: 242,691
Unique pieces:               4,876
```

Repeated pieces can therefore be represented once while retaining their corpus frequency.

This significantly reduced training time.

```text
~1M characters
512 vocabulary

Final bpe.py → 0.80 sec
```

With the full 8,192-token target on the same development sample:

```text
Characters:     ~1.0M
Vocabulary:     8,192
BPE merges:     7,935
Training time:  12.24 sec
```

The full tokenizer was subsequently trained on the complete TinyStories training corpus.

---

# 3. Final TinyStories Tokenizer

The final tokenizer was trained only on the training corpus.

```text
Training stories:     25,000
Training characters:  20,087,753
Vocabulary size:       8,192
BPE merges:             7,935
```

The trained tokenizer is saved as:

```text
artifacts/tinystories_bpe_8192.json
```

The tokenizer was then used to encode both the training and validation datasets.

### Training token statistics

```text
Characters:       20,087,753
Tokens:            4,907,617
Characters/token:      ~4.09
Tokens/character:      ~0.244
```

### Validation token statistics

```text
Stories:           2,000
Characters:        1,617,811
Tokens:              396,525
Characters/token:      ~4.08
Tokens/character:      ~0.245
```

So the Mini GPT training corpus contains approximately:

```text
4.91M tokens
```

This gives us the actual token budget for the model rather than estimating it from character counts.

---

# 4. Tokenizer Validation

The tokenizer supports the complete:

```text
encode → decode
```

round trip.

Special tokens are handled separately from normal BPE merges.

For example:

```text
"play<|endoftext|>played"
```

is represented as:

```text
BPE tokens
+
<|endoftext|>
+
BPE tokens
```

The tokenizer also supports persistence:

```text
Train
 ↓
save()
 ↓
tokenizer JSON
 ↓
load()
 ↓
new tokenizer instance
 ↓
encode()
 ↓
decode()
```

Save/load round-trip tests pass successfully.

---

# 5. Transformer Architecture

With the tokenizer complete, the next stage is to build the language model itself.

The model will follow the decoder-only Transformer architecture used by GPT-style models.

High-level flow:

```text
Token IDs
   ↓
Token Embeddings
   ↓
Transformer Block
   ↓
Transformer Block
   ↓
...
   ↓
Final Normalization
   ↓
Language Model Head
   ↓
Logits
```

Each Transformer block contains:

```text
Input
 ↓
RMSNorm
 ↓
Multi-Head Self-Attention
 ↓
Residual Connection
 ↓
RMSNorm
 ↓
Feed-Forward Network
 ↓
Residual Connection
```

The project will implement and study:

- token embeddings
- positional information
- Q/K/V projections
- multi-head attention
- causal masking
- RoPE
- RMSNorm
- SwiGLU
- residual connections
- language-model head

---

# 6. Attention

The attention mechanism will be implemented directly rather than using a high-level Transformer implementation.

The core operation is:

```text
Q = XWq
K = XWk
V = XWv

Attention(Q,K,V)
=
softmax(QKᵀ / √d)
V
```

The project will explore:

```text
single-head attention
        ↓
multi-head attention
        ↓
causal attention
        ↓
RoPE
        ↓
KV cache
```

The objective is to understand what each matrix and tensor represents rather than treating attention as one opaque operation.

---

# 7. Transformer Building Blocks

The model will be assembled from small independent components.

Expected structure:

```text
mini_gpt/
├── attention.py
├── embeddings.py
├── model.py
├── rmsnorm.py
├── rope.py
├── swiglu.py
└── transformer_block.py
```

Each component will first be tested independently before being combined into the full model.

---

# 8. Training Mini GPT

Once the architecture is complete, the tokenizer will convert TinyStories into token IDs.

The training pipeline will become:

```text
TinyStories
   ↓
Final BPE tokenizer
   ↓
~4.91M training tokens
   ↓
Input sequences
   ↓
Mini GPT
   ↓
Logits
   ↓
Next-token loss
   ↓
Backpropagation
   ↓
Optimizer
   ↓
Updated weights
```

The model will learn using next-token prediction.

For example:

```text
Input:

The little dog

Target:

little dog ran
```

The model learns to predict the next token at every position.

---

# 9. Training Objective

The primary training objective will be causal language modeling.

For a sequence:

```text
t1 t2 t3 t4
```

the model learns:

```text
t1 → t2
t1 t2 → t3
t1 t2 t3 → t4
```

The loss will be cross-entropy between:

```text
predicted next-token distribution
```

and:

```text
actual next token
```

---

# 10. Validation

A separate validation corpus will be kept outside tokenizer training and model training.

The validation pipeline will be:

```text
Validation text
   ↓
Already-trained tokenizer
   ↓
Validation token IDs
   ↓
Mini GPT
   ↓
Validation loss
```

This allows us to compare:

```text
Training loss
vs
Validation loss
```

and understand overfitting and generalization.

---

# 11. Text Generation

After training, the model will generate text autoregressively.

```text
Prompt
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Mini GPT
 ↓
Next-token probabilities
 ↓
Sampling
 ↓
New token
 ↓
Append token
 ↓
Repeat
```

Generation experiments will include:

- greedy decoding
- temperature
- top-k sampling
- top-p sampling
- EOS stopping
- KV cache

---

# 12. KV Cache

During generation, the model repeatedly processes the growing sequence.

Without a KV cache:

```text
token 1
token 1 + token 2
token 1 + token 2 + token 3
...
```

The previous attention keys and values are repeatedly recomputed.

With a KV cache:

```text
Prompt
 ↓
Compute K/V once
 ↓
Cache K/V
 ↓
Generate next token
 ↓
Compute only new Q/K/V
 ↓
Append new K/V to cache
 ↓
Repeat
```

The project will implement and verify the cache by comparing cached and non-cached outputs.

---

# 13. Experiments

The project is intentionally experiment-driven.

Examples include:

```text
Tokenizer:
- vocabulary size
- pre-tokenization
- BPE merge count
- token compression
- training performance

Model:
- hidden size
- number of layers
- attention heads
- context length
- feed-forward size

Training:
- learning rate
- batch size
- sequence length
- optimizer
- number of training steps

Generation:
- temperature
- top-k
- top-p
- greedy decoding
- KV cache
```

The objective is to measure what changes rather than relying only on intuition.

---

# Repository Structure

```text
mini-gpt-from-scratch/
│
├── artifacts/
│   ├── tinystories_bpe_8192_dev.json
│   └── tinystories_bpe_8192.json
│
├── data/
│   ├── raw/
│   │   ├── train.jsonl
│   │   └── val.jsonl
│   │
│   └── processed/
│
├── docs/
│   ├── tokenizer.md
│   ├── RoPE.md
│   └── transformer_block.md
│
├── notebooks/
│   ├── 01_raw_data_exploration.ipynb
│   ├── 02_bpe_tokenizer.ipynb
│   ├── 03_byte_level_bpe.ipynb
│   └── 04_tokenizer_on_tinystories.ipynb
│
├── scripts/
│   └── download_data.py
│
├── tokenizer/
│   ├── bpe.py
│   ├── bpe_v1.py
│   └── bpe_v2.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

As the model implementation is added, the repository will expand with the model, training, and inference components.

---

# Learning Order

The recommended way to follow this repository is:

```text
01. Explore the data
        ↓
02. Understand tokenization
        ↓
03. Build toy BPE
        ↓
04. Build Byte-Level BPE
        ↓
05. Optimize tokenizer representation
        ↓
06. Train final tokenizer
        ↓
07. Understand embeddings
        ↓
08. Build attention
        ↓
09. Add RoPE
        ↓
10. Build Transformer block
        ↓
11. Build Mini GPT
        ↓
12. Train on TinyStories
        ↓
13. Validate
        ↓
14. Generate text
        ↓
15. Add KV cache
        ↓
16. Run experiments
```

The notebooks are intended to explain and experiment with concepts.

The Python modules contain the reusable implementations.

The `docs/` directory captures deeper technical notes and implementation decisions.

---

# What This Project Is Not

This is not intended to be:

- a production LLM
- a reproduction of a large commercial model
- a replacement for optimized tokenizer libraries
- a benchmark-focused implementation

The focus is understanding.

The implementations are deliberately small enough to read from beginning to end.

---

# Current Status

### Completed

```text
Data exploration                         ✅
Toy BPE                                  ✅
Byte-level BPE                           ✅
Merge ranks                              ✅
Special-token handling                   ✅
Encode / decode                          ✅
Save / load                              ✅
Pre-tokenization                         ✅
Frequency-weighted BPE                   ✅
Tokenizer optimization experiments      ✅
Final 8,192-token tokenizer              ✅
Full TinyStories tokenization            ✅
Train / validation token statistics     ✅
```

### Next

```text
Transformer architecture
      ↓
Attention
      ↓
RoPE
      ↓
RMSNorm
      ↓
SwiGLU
      ↓
Transformer block
      ↓
Mini GPT
      ↓
Training
      ↓
Generation
```

---

# Key Learning Principle

The project follows one rule throughout:

> Do not just use the component. Build a small version of it, inspect it, test it, and understand why it works.

For each major component, the progression is:

```text
Concept
 ↓
Small implementation
 ↓
Test
 ↓
Experiment
 ↓
Measure
 ↓
Improve
 ↓
Document
```

The goal is to finish the project with a working Mini GPT, but more importantly, to understand the path from raw text to trained language-model weights.