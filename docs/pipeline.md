# End-to-End Pipeline

This document traces every computation from raw text to generated output.

Each step includes the tensor shape and the file responsible.

---

# Overview

```text
Raw Stories
   ↓
Tokenizer Training
   ↓
Dataset Preparation
   ↓
Training
   ↓
Text Generation
```

---

# 1. Raw Data

The project starts with TinyStories stored as JSONL.

```text
data/raw/train.jsonl     25,000 stories    ~20M characters
data/raw/val.jsonl        2,000 stories    ~1.6M characters
```

Each line is a JSON object:

```json
{"text": "Once upon a time, there was a little girl named Lily..."}
```

File: `scripts/download_data.py`

---

# 2. Tokenizer Training

Before anything else, a tokenizer must be trained on the raw text.

```text
Raw stories
   ↓
Pre-tokenize into pieces
   ↓
Count unique pieces + frequency
   ↓
UTF-8 encode each unique piece
   ↓
Frequency-weighted pair counting
   ↓
Select most frequent pair
   ↓
Create new token
   ↓
Store merge rule + rank
   ↓
Merge pair in all unique pieces
   ↓
Repeat 7,935 times
   ↓
Save vocabulary + merges + special tokens
```

Result:

```text
256 byte tokens + 1 special token + 7,935 merged tokens = 8,192 total
```

Saved as: `artifacts/tinystories_bpe_8192.json`

File: `tokenizer/bpe.py` → `ByteLevelBPETokenizer.train()`

---

# 3. Encoding Text into Token IDs

When text needs to become numbers:

```text
"Once upon a time"
   ↓
Split on special tokens
   ↓
Pre-tokenize each part
   ["Once", " upon", " a", " time"]
   ↓
For each piece:
   ↓
   UTF-8 bytes
   [79, 110, 99, 101]
   ↓
   Find all applicable learned merges
   ↓
   Apply lowest-rank merge first
   ↓
   Repeat until no merge applies
   ↓
   Token IDs for this piece
   ↓
Concatenate all piece IDs
   ↓
[430, 437, 259, 398]
```

File: `tokenizer/bpe.py` → `ByteLevelBPETokenizer.encode()`

---

# 4. Dataset Preparation

The raw stories are converted into training-ready tensors.

```text
For each story in train.jsonl:
   ↓
   Encode with tokenizer
   ↓
   Append EOS token (ID 256)
   ↓
Concatenate ALL story tokens into one stream
   ↓
[t₀, t₁, t₂, t₃, ..., t₄,₉₃₃,₂₅₀]

   ↓
Shift to create input and target:

   input_tokens  = stream[:-1]     all tokens except the last
   target_tokens = stream[1:]      all tokens except the first

   ↓
Pack into fixed-length sequences:

   input_ids  = input_tokens.reshape(N, 512)
   target_ids = target_tokens.reshape(N, 512)

   ↓
Save as .pt file
```

Result:

```text
Train:  input_ids [9634, 512]    target_ids [9634, 512]
Val:    input_ids [778, 512]     target_ids [778, 512]
```

Saved as: `data/processed/train.pt` and `data/processed/val.pt`

File: `scripts/prepare_dataset.py` → `prepare_dataset()`

---

# 5. DataLoader

The saved tensors are loaded and served in batches.

```text
data/processed/train.pt
   ↓
TinyStoriesDataset
   loads input_ids [9634, 512]
   loads target_ids [9634, 512]
   ↓
DataLoader
   batch_size = 32
   shuffle = True
   ↓
Each batch:
   input_ids  [32, 512]     (integers, token IDs)
   target_ids [32, 512]     (integers, shifted by 1)
```

File: `data/dataset.py` → `TinyStoriesDataset`

---

# 6. Token Embedding

The first operation inside the model.

Token IDs are looked up in a learned embedding table.

```text
input_ids [B, T]
   ↓
nn.Embedding(8192, 512)
   ↓
x [B, T, 512]
```

Each of the 8,192 possible tokens has a learned 512-dimensional vector.

The embedding table is a weight matrix:

```text
[8192, 512]
```

Looking up token ID 430 returns row 430 of this matrix.

File: `model/gpt.py` → `GPT.embed_tokens`

---

# 7. Transformer Block (× 4)

The embedded sequence passes through 4 identical blocks.

Each block:

```text
x [B, T, 512]
   ↓
┌─────────────────────────────────────┐
│                                     │
│   save as residual                  │
│   ↓                                 │
│   RMSNorm              [B, T, 512]  │
│   ↓                                 │
│   Multi-Head Attention  [B, T, 512]  │
│   ↓                                 │
│   + residual            [B, T, 512]  │
│   ↓                                 │
│   save as residual                  │
│   ↓                                 │
│   RMSNorm              [B, T, 512]  │
│   ↓                                 │
│   SwiGLU               [B, T, 512]  │
│   ↓                                 │
│   + residual            [B, T, 512]  │
│                                     │
└─────────────────────────────────────┘
   ↓
x [B, T, 512]
```

File: `model/transformer_block.py` → `TransformerBlock.forward()`

The following sections break down each component inside the block.

---

# 8. RMSNorm

Normalizes the hidden state before attention and before the feed-forward network.

```text
x [B, T, 512]
   ↓
Compute RMS across last dimension:

   rms = sqrt( mean(x², dim=-1) + eps )
         [B, T, 1]
   ↓
Normalize:

   x_norm = x / rms
            [B, T, 512]
   ↓
Scale by learned parameter:

   output = x_norm * gamma
            [B, T, 512]
```

Learnable parameters:

```text
gamma [512]     initialized to ones
```

File: `model/rmsnorm.py` → `RMSNorm.forward()`

---

# 9. Multi-Head Attention

This is the core mechanism that lets each token attend to previous tokens.

### 9a. Q / K / V Projections

Three separate linear projections, no bias.

```text
x [B, T, 512]
   ↓
Q = x @ Wq     [B, T, 512]
K = x @ Wk     [B, T, 512]
V = x @ Wv     [B, T, 512]
```

Learnable parameters:

```text
Wq [512, 512]
Wk [512, 512]
Wv [512, 512]
```

### 9b. Split into Heads

The 512-dimensional vectors are reshaped into 8 heads of 64 dimensions each.

```text
Q [B, T, 512]
   ↓
reshape to [B, T, 8, 64]
   ↓
transpose to [B, 8, T, 64]
```

Same for K and V.

After this step:

```text
Q [B, 8, T, 64]
K [B, 8, T, 64]
V [B, 8, T, 64]
```

### 9c. RoPE (Rotary Positional Embedding)

Applied to Q and K only. Not applied to V.

RoPE encodes position by rotating pairs of dimensions.

```text
Q [B, 8, T, 64]
   ↓
Compute position indices:
   positions = [0, 1, 2, ..., T-1]
   ↓
Compute rotation angles:
   inv_freq = 1 / (10000 ^ (2i / 64))     for i = 0..31
              [32]
   ↓
   angles = positions × inv_freq
            [T, 32]
   ↓
   cos = cos(angles)    [T, 32]
   sin = sin(angles)    [T, 32]
   ↓
Split Q into pairs:
   Q reshaped to [B, 8, T, 32, 2]
   q1 = Q[..., 0]    [B, 8, T, 32]
   q2 = Q[..., 1]    [B, 8, T, 32]
   ↓
Rotate:
   q1_rot = q1 * cos - q2 * sin
   q2_rot = q2 * cos + q1 * sin
   ↓
Reassemble:
   Q_rotated [B, 8, T, 64]
```

Same rotation applied to K.

File: `model/rope.py` → `RoPE.forward()`

### 9d. Attention Scores

```text
Q [B, 8, T, 64]
K [B, 8, T, 64]
   ↓
scores = Q @ Kᵀ / sqrt(64)
         [B, 8, T, T]
```

Each entry `scores[b, h, i, j]` measures how much token `i` should attend to token `j` in head `h`.

### 9e. Causal Mask

The model must not look at future tokens.

```text
scores [B, 8, T, T]
   ↓
Build mask:
   position j is visible to position i only if j <= i

   ┌                         ┐
   │  ok   -inf  -inf  -inf  │   token 0 sees only token 0
   │  ok    ok   -inf  -inf  │   token 1 sees tokens 0-1
   │  ok    ok    ok   -inf  │   token 2 sees tokens 0-2
   │  ok    ok    ok    ok   │   token 3 sees tokens 0-3
   └                         ┘
   ↓
masked_scores [B, 8, T, T]
```

### 9f. Softmax

```text
masked_scores [B, 8, T, T]
   ↓
softmax along last dimension (key dimension)
   ↓
attn_weights [B, 8, T, T]
```

Each row sums to 1. The `-inf` entries become 0 after softmax.

### 9g. Weighted Sum of Values

```text
attn_weights [B, 8, T, T]
V            [B, 8, T, 64]
   ↓
attn_output = attn_weights @ V
              [B, 8, T, 64]
```

Each token's output is a weighted combination of all visible tokens' V vectors.

### 9h. Merge Heads

```text
attn_output [B, 8, T, 64]
   ↓
transpose to [B, T, 8, 64]
   ↓
reshape to [B, T, 512]
```

### 9i. Output Projection

```text
[B, T, 512]
   ↓
@ Wo
   ↓
[B, T, 512]
```

Learnable parameters:

```text
Wo [512, 512]
```

File: `model/attention.py` → `Attention.forward()`

### 9j. Residual Connection

```text
attention_output [B, T, 512]
   ↓
+ original input (saved before RMSNorm)
   ↓
x [B, T, 512]
```

---

# 10. SwiGLU Feed-Forward Network

After attention, each token is processed independently through a gated feed-forward network.

```text
x [B, T, 512]
   ↓
Two parallel projections:

   up   = x @ W_up       [B, T, 2048]
   gate = x @ W_gate     [B, T, 2048]
   ↓
Apply SiLU activation to gate only:

   gate = SiLU(gate)     [B, T, 2048]

   SiLU(x) = x * sigmoid(x)
   ↓
Element-wise multiply:

   hidden = up * gate    [B, T, 2048]
   ↓
Project back down:

   output = hidden @ W_down    [B, T, 512]
```

Learnable parameters:

```text
W_up   [512, 2048]
W_gate [512, 2048]
W_down [2048, 512]
```

File: `model/swiglu.py` → `SwiGLU.forward()`

### Residual Connection

```text
swiglu_output [B, T, 512]
   ↓
+ input (saved before second RMSNorm)
   ↓
x [B, T, 512]
```

This completes one Transformer block. The output feeds into the next block.

---

# 11. After All Blocks

After passing through all 4 Transformer blocks:

```text
x [B, T, 512]
   ↓
Final RMSNorm
   [B, T, 512]
   ↓
Language Model Head (linear projection, no bias):
   x @ W_lm_head
   ↓
logits [B, T, 8192]
```

Learnable parameters:

```text
W_lm_head [512, 8192]
```

`logits[b, t, v]` is the raw score for token `v` being the next token after position `t`.

File: `model/gpt.py` → `GPT.forward()`

---

# 12. Training: Loss Computation

The logits are compared against the actual next tokens.

```text
logits     [B, T, 8192]
target_ids [B, T]
   ↓
Reshape:
   logits     → [B*T, 8192]
   target_ids → [B*T]
   ↓
Cross-entropy loss:

   For each position:
      softmax(logits) → probability distribution over 8,192 tokens
      loss = -log(probability of the correct target token)
   ↓
   Average across all positions
   ↓
scalar loss
```

File: `train.py` → `loss_fn()`

---

# 13. Training: Backpropagation and Optimization

```text
loss (scalar)
   ↓
loss.backward()
   computes gradients for every learnable parameter
   ↓
Gradient clipping:
   clip_grad_norm_(parameters, max_norm=1.0)
   prevents exploding gradients
   ↓
optimizer.step()
   AdamW updates all parameters
   weight_decay = 0.1
   learning_rate = 3e-4 (with warmup + cosine decay)
   ↓
scheduler.step()
   adjusts learning rate

   First 5% of steps:  linear warmup from 0 → 3e-4
   Remaining steps:    cosine decay from 3e-4 → ~3e-5
```

File: `train.py` → `main()`

---

# 14. Training: One Full Step Summary

```text
DataLoader
   ↓
input_ids [32, 512]     target_ids [32, 512]
   ↓
Embedding                [32, 512, 512]
   ↓
Transformer Block 0      [32, 512, 512]
   RMSNorm → Attention + RoPE → Residual → RMSNorm → SwiGLU → Residual
   ↓
Transformer Block 1      [32, 512, 512]
   ↓
Transformer Block 2      [32, 512, 512]
   ↓
Transformer Block 3      [32, 512, 512]
   ↓
Final RMSNorm            [32, 512, 512]
   ↓
LM Head                  [32, 512, 8192]
   ↓
Cross-Entropy Loss       scalar
   ↓
Backward                 gradients for all 25.17M parameters
   ↓
Clip Gradients
   ↓
AdamW Step
   ↓
Scheduler Step
```

---

# 15. Validation

Same forward pass, no gradient computation.

```text
val_loader
   ↓
input_ids [32, 512]     target_ids [32, 512]
   ↓
model.eval()
   ↓
with torch.no_grad():
   ↓
   Forward pass → logits [32, 512, 8192]
   ↓
   Cross-entropy loss → scalar
   ↓
Average loss across all validation batches
   ↓
Validation loss
```

File: `train.py` → `evaluate()`

---

# 16. Checkpoints

After each epoch:

```text
Save latest_model.pt     always

If val_loss < best_val_loss:
   Save best_model.pt
```

Each checkpoint contains:

```text
model_state_dict
optimizer_state_dict
scheduler_state_dict
epoch
step
train_loss
val_loss
```

File: `train.py` → `save_checkpoint()`

---

# 17. Generation: Loading the Model

```text
config.py
   ↓
   VOCAB_SIZE = 8192
   D_MODEL = 512
   NUM_HEADS = 8
   NUM_BLOCKS = 4
   INTERMEDIATE_SIZE = 2048
   ↓
Create empty GPT model
   ↓
Load checkpoints/best_model.pt
   ↓
Load model_state_dict into model
   ↓
model.eval()
```

File: `generate.py` → `load_model()`

---

# 18. Generation: Prefill (Processing the Prompt)

```text
"Once upon a time"
   ↓
tokenizer.encode()
   ↓
[430, 437, 259, 398]
   ↓
Tensor [1, 4]
   ↓
Full forward pass with use_cache=True:

   Embedding           [1, 4, 512]
   ↓
   Block 0             [1, 4, 512]    → cache K₀ [1, 8, 4, 64]  V₀ [1, 8, 4, 64]
   Block 1             [1, 4, 512]    → cache K₁ [1, 8, 4, 64]  V₁ [1, 8, 4, 64]
   Block 2             [1, 4, 512]    → cache K₂ [1, 8, 4, 64]  V₂ [1, 8, 4, 64]
   Block 3             [1, 4, 512]    → cache K₃ [1, 8, 4, 64]  V₃ [1, 8, 4, 64]
   ↓
   Final RMSNorm       [1, 4, 512]
   ↓
   LM Head             [1, 4, 8192]
   ↓
logits [1, 4, 8192]
past_key_values = ((K₀,V₀), (K₁,V₁), (K₂,V₂), (K₃,V₃))
```

We only need `logits[:, -1, :]` — the prediction after the last prompt token.

File: `generate.py` → `generate()`

---

# 19. Generation: Sampling One Token

```text
logits[:, -1, :] → next_token_logits [1, 8192]
   ↓
Temperature scaling:
   next_token_logits = next_token_logits / 0.8
   ↓
Top-K filtering (K=50):
   Keep only the 50 highest-scoring tokens
   Set the rest to -inf
   ↓
Top-P filtering (P=0.9):
   Sort tokens by probability (descending)
   Compute cumulative probability
   Remove tokens after cumulative > 0.9
   Set removed to -inf
   ↓
Softmax:
   probabilities [1, 8192]
   (most entries are 0 due to filtering)
   ↓
Sample:
   next_token_id = multinomial(probabilities, 1)
   ↓
   e.g. token ID 44 (",")
```

File: `generate.py` → `apply_top_k()`, `apply_top_p()`

---

# 20. Generation: Autoregressive Decoding with KV Cache

After sampling one token, only that single token is processed.

```text
Step 1 (after prefill):

   new_token [1, 1]     (just the sampled token)
   ↓
   Embedding            [1, 1, 512]
   ↓
   Block 0:
      Q from new token only           [1, 8, 1, 64]
      K from new token only           [1, 8, 1, 64]
      V from new token only           [1, 8, 1, 64]
      ↓
      RoPE with position_offset = 4   (prompt was 4 tokens)
      ↓
      Append new K to cached K₀:      [1, 8, 5, 64]
      Append new V to cached V₀:      [1, 8, 5, 64]
      ↓
      Q [1, 8, 1, 64] @ K [1, 8, 5, 64]ᵀ → scores [1, 8, 1, 5]
      ↓
      Causal mask (token 4 can see tokens 0-4)
      ↓
      softmax → attn_weights [1, 8, 1, 5]
      ↓
      attn_weights @ V → [1, 8, 1, 64]
      ↓
      Merge heads → [1, 1, 512]
      ↓
      Output projection → [1, 1, 512]
      ↓
      + residual → RMSNorm → SwiGLU → + residual
   ↓
   Block 1, 2, 3: same pattern, each appending to its own cache
   ↓
   Final RMSNorm        [1, 1, 512]
   ↓
   LM Head              [1, 1, 8192]
   ↓
   Sample next token
   ↓
   Repeat

Step 2:
   position_offset = 5
   Cache grows to [1, 8, 6, 64] per block
   ...

Until EOS token or max_new_tokens reached.
```

File: `model/attention.py` → KV cache logic in `Attention.forward()`

---

# 21. Generation: Decoding Back to Text

```text
All generated token IDs:
   [430, 437, 259, 398, 44, 400, 283, 259, 389, 473, ...]
   ↓
tokenizer.decode()
   ↓
For each token ID:
   Look up bytes in vocabulary
   ↓
Concatenate all bytes
   ↓
UTF-8 decode
   ↓
"Once upon a time, there was a little girl..."
```

File: `tokenizer/bpe.py` → `ByteLevelBPETokenizer.decode()`

---

# 22. Complete Generation Flow

```text
"Once upon a time"
   ↓
tokenizer.encode()                          tokenizer/bpe.py
   [430, 437, 259, 398]
   ↓
Prefill forward pass                        model/gpt.py
   Embedding → 4 Blocks → RMSNorm → LM Head
   ↓
logits [1, 4, 8192] + KV cache
   ↓
Take logits[:, -1, :]                       generate.py
   ↓
Temperature → Top-K → Top-P → Sample
   ↓
token ID 44 (",")
   ↓
Feed [1, 1] into model with cache           model/gpt.py
   ↓
logits [1, 1, 8192] + updated cache
   ↓
Temperature → Top-K → Top-P → Sample
   ↓
token ID 400 (" there")
   ↓
... repeat ...
   ↓
EOS or max tokens
   ↓
tokenizer.decode()                          tokenizer/bpe.py
   ↓
"Once upon a time, there was a little girl named Lily."
```

---

# 23. Parameter Count

Where the 25.17M parameters live:

```text
Embedding:
   embed_tokens          8192 × 512         =  4,194,304

Per Transformer Block (× 4):
   input_rmsnorm.gamma   512                 =        512
   q_proj                512 × 512           =    262,144
   k_proj                512 × 512           =    262,144
   v_proj                512 × 512           =    262,144
   o_proj                512 × 512           =    262,144
   rope.inv_freq         32                  =         32  (buffer, not learned)
   output_rmsnorm.gamma  512                 =        512
   up_proj               512 × 2048          =  1,048,576
   gate_proj             512 × 2048          =  1,048,576
   down_proj             2048 × 512          =  1,048,576

   Block total                               =  4,195,328

4 blocks                                     = 16,781,312

Final:
   norm.gamma            512                 =        512
   lm_head               512 × 8192          =  4,194,304

Total                                        = 25,170,432
```

---

# 24. File Map

Which file does what in the pipeline:

```text
config.py                    All hyperparameters and paths
tokenizer/bpe.py             Tokenizer: train, encode, decode, save, load
scripts/prepare_dataset.py   Raw stories → packed token tensors
data/dataset.py              Load tensors, expose samples
model/rmsnorm.py             RMSNorm normalization
model/rope.py                Rotary positional embedding
model/attention.py           Multi-head attention + causal mask + KV cache
model/swiglu.py              SwiGLU feed-forward network
model/transformer_block.py   One block: RMSNorm → Attention → Residual → RMSNorm → SwiGLU → Residual
model/gpt.py                 Full model: Embedding → N Blocks → RMSNorm → LM Head
train.py                     Training loop, optimizer, scheduler, checkpoints
generate.py                  Load model, temperature, top-k, top-p, KV cache generation
```
