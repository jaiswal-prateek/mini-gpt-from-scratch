# Mini GPT From Scratch

A hands-on, from-scratch implementation of a small GPT style language model in PyTorch for learning how LLMs work - from byte-level BPE tokenization and embeddings to Transformer attention, RoPE, training, KV caching, and text generation.

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
Transformer Components
   ↓
Mini GPT
   ↓
Training
   ↓
KV Cache
   ↓
Text Generation
   ↓
Sampling Experiments
```

The goal is not to build the largest or most capable model.

The goal is to understand what is happening inside a language model by implementing, testing, training, and experimenting with the individual components ourselves.

---

## What This Project Is

This is a learning-focused implementation of a small GPT-style decoder-only Transformer.

The project covers the path from:

```text
text
 ↓
bytes
 ↓
tokens
 ↓
embeddings
 ↓
attention
 ↓
transformer blocks
 ↓
logits
 ↓
next-token prediction
 ↓
trained language model
 ↓
text generation
```

Most core components are implemented directly in Python/PyTorch rather than hidden behind high-level Transformer libraries.

The implementation is intentionally small enough to inspect, modify, and experiment with.

---

# Project Roadmap

The project is being developed in stages:

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

The README follows the same order as the learning process.

---

# 1. Data Exploration

Before building a model, we first inspect the data.

The project uses the TinyStories dataset.

The initial exploration covers:

- Number of stories
- Text length
- Character distribution
- Common patterns
- Training and validation data
- Basic corpus statistics

The purpose is to understand what the model will actually be trained on rather than treating the dataset as a black box.

Notebook:

```text
notebooks/01_raw_data_exploration.ipynb
```

---

# 2. Building a Byte-Level BPE Tokenizer

A language model does not directly consume text.

The text must first be converted into token IDs.

The tokenizer learning path started with a simple toy BPE implementation and progressively became a more practical byte-level implementation.

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

The final tokenizer is implemented in:

```text
tokenizer/bpe.py
```

It supports:

- 256 initial byte tokens
- BPE vocabulary expansion
- Merge rules
- Merge ranks
- Pre-tokenization
- Frequency-weighted BPE training
- Special tokens
- Encoding
- Decoding
- Save/load

---

## Tokenizer Evolution

Three implementations were explored while learning:

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

The first implementation repeatedly scanned the full corpus.

The second explored incremental pair statistics to reduce repeated work, but the additional Python bookkeeping made it slower in the benchmark used during development.

The final implementation changed the representation of the training corpus by using pre-tokenized pieces and their frequencies, significantly reducing tokenizer training time.

These experiments are retained because understanding why an optimization does or does not help is part of the learning process.

---

# 3. Final TinyStories Tokenizer

The final tokenizer was trained only on the TinyStories training corpus.

```text
Training stories:       25,000
Training characters:    20,087,753
Vocabulary size:        8,192
BPE merges:             7,935
```

The trained tokenizer is saved as:

```text
artifacts/tinystories_bpe_8192.json
```

### Training token statistics

```text
Characters:             20,087,753
Tokens:                  4,907,617
Characters/token:           ~4.09
Tokens/character:            ~0.244
```

### Validation token statistics

```text
Stories:                  2,000
Characters:               1,617,811
Tokens:                     396,525
Characters/token:             ~4.08
Tokens/character:              ~0.245
```

The final training pipeline contains approximately 4.93M tokens after adding EOS tokens and constructing the continuous training stream.

---

# 4. Tokenizer Validation

The tokenizer supports the complete:

```text
encode → decode
```

round trip.

Special tokens are handled separately from normal BPE merges.

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

With the tokenizer complete, the next stage was building the language model itself.

The model follows a decoder-only Transformer architecture.

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
Final RMSNorm
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
Multi-Head Self-Attention + RoPE
 ↓
Residual Connection
 ↓
RMSNorm
 ↓
SwiGLU
 ↓
Residual Connection
 ↓
Output
```

The project implements:

- Token embeddings
- Q/K/V projections
- Multi-head attention
- Causal masking
- RoPE
- RMSNorm
- SwiGLU
- Residual connections
- Language-model head

The model components are located under:

```text
model/
├── attention.py
├── gpt.py
├── rmsnorm.py
├── rope.py
├── swiglu.py
└── transformer_block.py
```

---

# 6. Mini GPT Configuration

The current baseline model uses:

```text
Vocabulary size:       8,192
d_model:               512
Transformer blocks:    4
Attention heads:       8
Head dimension:        64
SwiGLU size:           2,048
Context length:        512
Parameters:             ~25.2M
```

The model is intentionally small enough to train locally while still containing the major components found in modern decoder-only language models.

---

# 7. Attention

Attention is implemented directly rather than using a high-level Transformer implementation.

The core operation is:

```text
Q = XWq
K = XWk
V = XWv

Attention(Q, K, V)
=
softmax(QKᵀ / √d) V
```

The implementation includes:

```text
Q/K/V projections
      ↓
Multi-head attention
      ↓
Causal masking
      ↓
RoPE
      ↓
KV cache
```

The goal is to understand what each matrix and tensor represents rather than treating attention as one opaque operation.

---

# 8. Transformer Building Blocks

The model is assembled from small independent components.

The main structure is:

```text
RMSNorm
   ↓
Attention + RoPE
   ↓
Residual
   ↓
RMSNorm
   ↓
SwiGLU
   ↓
Residual
```

Each major component was tested independently before being combined into the full model.

The corresponding learning notebooks are under:

```text
notebooks/
```

and deeper implementation notes are captured under:

```text
docs/
```

---

# 9. Dataset Preparation

Once the tokenizer and model were complete, TinyStories was converted into training sequences.

The pipeline is:

```text
Raw stories
   ↓
Tokenizer
   ↓
Add EOS token
   ↓
Concatenate into continuous token stream
   ↓
Shift input / target
   ↓
Pack into 512-token sequences
   ↓
DataLoader
   ↓
Mini GPT
```

The important detail is that the input/target shift happens before splitting the continuous stream into fixed-length sequences.

This allows the model to learn across sequence boundaries in the packed token stream.

The reusable dataset implementation is:

```text
data/dataset.py
```

Dataset preparation is handled by:

```text
scripts/prepare_dataset.py
```

---

# 10. Training Mini GPT

The model is trained using causal language modeling.

For a sequence:

```text
t1 t2 t3 t4
```

the model learns:

```text
t1       → t2
t1 t2    → t3
t1 t2 t3 → t4
```

The training pipeline is:

```text
Token IDs
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

The current training configuration is:

```text
Epochs:             5
Batch size:         32
Learning rate:      3e-4
Optimizer:          AdamW
Weight decay:       0.1
Warmup ratio:       5%
Gradient clipping:  1.0
Scheduler:          Warmup + cosine decay
```

Training automatically uses:

```text
CUDA → MPS → CPU
```

depending on hardware availability.

---

# 11. Training Results

The first complete baseline training run used the full prepared TinyStories training dataset.

```text
Model parameters:        25,170,432
Training sequences:           9,634
Validation sequences:           778
Sequence length:                512
Optimizer steps:              ~1,505
Training tokens:         ~4.93M
```

### Loss progression

| Epoch | Train Loss | Validation Loss |
|------:|-----------:|----------------:|
| 1 | 4.1669 | 2.9104 |
| 2 | 2.6256 | 2.4617 |
| 3 | 2.2914 | 2.2613 |
| 4 | 2.1016 | 2.1685 |
| 5 | 1.9976 | **2.1337** |

The validation loss improved throughout the five-epoch baseline run.

Final validation perplexity:

```text
Perplexity = exp(2.1337)

≈ 8.45
```

This baseline provides a reference point for future model and training experiments.

---

# 12. Validation

A separate validation corpus is kept outside model training.

The validation pipeline is:

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
   ↓
Perplexity
```

The current baseline achieved:

```text
Final train loss:       1.9976
Best validation loss:   2.1337
Validation perplexity:  ~8.45
```

Future changes to the model or training configuration can be compared against this baseline.

---

# 13. Checkpoints

Training saves two checkpoints:

```text
checkpoints/
├── best_model.pt
└── latest_model.pt
```

The checkpoints contain the model and training state needed to resume or run inference.

They are intentionally excluded from Git because each checkpoint is a large binary file.

The repository therefore contains the implementation, configuration, notebooks, and documentation rather than storing the trained weights directly in Git.

---

# 14. KV Cache

During autoregressive generation, the model repeatedly processes a growing sequence.

Without a KV cache, previously computed attention keys and values are repeatedly recomputed.

With a KV cache:

```text
Prompt
   ↓
Compute K/V
   ↓
Cache K/V
   ↓
Generate next token
   ↓
Compute new Q/K/V
   ↓
Append new K/V
   ↓
Repeat
```

The KV cache is implemented in the attention, Transformer block, and GPT modules.

The cached generation path was tested against full-sequence generation and produces matching outputs for the corresponding positions.

---

# 15. Text Generation

After training, the model can generate text autoregressively.

The generation process is:

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

The generation implementation supports:

- Temperature
- Top-K sampling
- Top-P sampling
- EOS stopping
- Maximum new tokens
- KV cache

Implementation:

```text
generate.py
```

The current model generates TinyStories-style continuations.

It is not an instruction-tuned chatbot, so prompts such as questions do not necessarily produce direct answers. The model was trained primarily on next-token prediction.

---

# 16. Generation Playground

A lightweight local frontend was built to experiment with the trained model interactively.

The playground exposes:

```text
Temperature
Top-K
Top-P
Max Tokens
```

It also displays next-token probabilities for generated tokens.

This makes it possible to observe how sampling parameters affect generation.

For example:

```text
Lower temperature
   ↓
Sharper probability distribution
   ↓
More predictable generation
```

while:

```text
Higher temperature
   ↓
Flatter probability distribution
   ↓
More varied generation
```

The playground is primarily a learning and experimentation tool.

### Running the Playground

```bash
python -m uvicorn frontend.app.tokenizer.api:app --host 127.0.0.1 --port 8070
```

Open `http://127.0.0.1:8070` in a browser.

To stop the server:

```bash
pkill -f "uvicorn frontend"
```

---

# 17. Experiments

The project is intentionally experiment-driven.

Current and planned experiments include:

### Tokenizer

- Vocabulary size
- BPE merge count
- Pre-tokenization
- Token compression
- Training performance

### Model

- Number of Transformer blocks
- d_model
- Attention heads
- Context length
- Feed-forward size

### Training

- Learning rate
- Batch size
- Sequence length
- Optimizer
- Number of training steps
- Training duration

### Generation

- Temperature
- Top-K
- Top-P
- Greedy decoding
- KV cache

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
│   ├── processed/
│   │   ├── train.pt
│   │   └── val.pt
│   └── dataset.py
│
├── docs/
│   ├── dataset.md
│   ├── tokenizer.md
│   ├── RoPE.md
│   └── transformer_block.md
│
├── model/
│   ├── attention.py
│   ├── gpt.py
│   ├── rmsnorm.py
│   ├── rope.py
│   ├── swiglu.py
│   └── transformer_block.py
│
├── notebooks/
│   ├── 01_raw_data_exploration.ipynb
│   ├── 02_bpe_tokenizer.ipynb
│   ├── 03_byte_level_bpe.ipynb
│   ├── 04_tokenizer_on_tinystories.ipynb
│   ├── 05_embeddings.ipynb
│   ├── 06_rmsnorm.ipynb
│   ├── 07_rope.ipynb
│   ├── 08_attention.ipynb
│   ├── 09_transformer_block.ipynb
│   ├── 10_mini_gpt.ipynb
│   ├── 11_kv_cache.ipynb
│   ├── 12_dataset.ipynb
│   ├── 13_token_stream.ipynb
│   └── 14_dataloader.ipynb
│
├── scripts/
│   ├── download_data.py
│   └── prepare_dataset.py
│
├── tokenizer/
│   ├── bpe.py
│   ├── bpe_v1.py
│   └── bpe_v2.py
│
├── frontend/
│
├── config.py
├── train.py
├── generate.py
├── README.md
├── requirements.txt
└── .gitignore
```

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
08. Build RMSNorm
        ↓
09. Build RoPE
        ↓
10. Build attention
        ↓
11. Build Transformer block
        ↓
12. Build Mini GPT
        ↓
13. Prepare training data
        ↓
14. Train Mini GPT
        ↓
15. Add KV cache
        ↓
16. Generate text
        ↓
17. Experiment with sampling
        ↓
18. Evaluate and improve the baseline
```

The notebooks are intended to explain and experiment with concepts.

The Python modules contain the reusable implementations.

The `docs/` directory captures deeper technical notes and implementation decisions.

---

# What This Project Is Not

This is not intended to be:

- A production LLM
- A reproduction of a large commercial model
- A replacement for optimized tokenizer libraries
- A benchmark-focused implementation

The focus is understanding.

The implementations are deliberately small enough to read from beginning to end.

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

The goal is to finish the project with a working Mini GPT, but more importantly, to understand the path from raw text to trained language-model weights and generated text.

---

# Baseline Snapshot

The current baseline:

```text
Model:                  Mini GPT
Parameters:             25.17M
Vocabulary size:        8,192
d_model:                512
Transformer blocks:     4
Attention heads:        8
Head dimension:         64
SwiGLU size:            2,048
Context length:         512

Training stories:       25,000
Training tokens:        ~4.93M
Validation stories:     2,000
Validation tokens:      ~0.40M

Training epochs:        5
Final train loss:       1.9976
Best validation loss:   2.1337
Validation perplexity:  ~8.45

KV cache:               Enabled
Temperature sampling:   Enabled
Top-K sampling:         Enabled
Top-P sampling:         Enabled
```

This baseline will serve as the reference point for future experiments.