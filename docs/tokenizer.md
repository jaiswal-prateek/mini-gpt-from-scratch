# Tokenizer Evolution — BPE v1 → v2 → Final bpe.py

This document captures how the Byte-Level BPE tokenizer evolved during the Mini GPT project.

The goal was not only to build a tokenizer, but to understand:

- how BPE learns a vocabulary
- how encoding uses learned merge priorities
- why a naive implementation becomes slow
- what we tried to optimize
- why v2 did not help
- why the final implementation is much faster
- how the final tokenizer differs from the initial implementation

---

# 1. High-Level Evolution

```text
                         BYTE-LEVEL BPE

          ┌───────────────────┼────────────────────┐
          │                   │                    │
          ▼                   ▼                    ▼

       BPE v1              BPE v2             Final bpe.py
       Naive               Incremental        Frequency-based
       baseline            update attempt     pre-tokenized

          │                   │                    │
          ▼                   ▼                    ▼

   Full corpus          Full corpus          Pre-tokenize text
   every merge          every merge          ↓
          │             + pair index         Unique pieces
          │             + affected sets      + frequencies
          ▼                   │                    │
  Recount ALL pairs           ▼                    ▼
  every iteration       Update affected      Weighted pair counts
          │             sequences only             │
          ▼                   │                    ▼
  Merge ALL sequences        │              Merge unique pieces
  every iteration            │                    │
          │                   │                    ▼
          ▼                   ▼                  Faster
       Simple              Slower              final version
       baseline
```

---

# 2. What All Versions Have in Common

All three versions implement the same fundamental BPE idea.

```text
Text
 ↓
UTF-8 bytes
 ↓
Initial byte vocabulary
 ↓
Count adjacent token pairs
 ↓
Select highest-frequency pair
 ↓
Create a new token
 ↓
Store merge rule
 ↓
Merge the pair
 ↓
Repeat
```

The main difference between the versions is **how much text we process for every merge**.

---

# 3. BPE v1 — Naive Baseline

File:

```text
tokenizer/bpe_v1.py
```

## Main idea

The first implementation operates directly on the full text sequences.

For every BPE merge:

```text
Full corpus
   ↓
Count every adjacent pair
   ↓
Find the most frequent pair
   ↓
Merge that pair across the entire corpus
   ↓
Repeat
```

## Training flow

```text
Raw text
   │
   ▼
UTF-8 bytes
   │
   ▼
Corpus of byte sequences
   │
   ▼
Count pairs across ALL sequences
   │
   ▼
Most frequent pair
   │
   ▼
Create new token
   │
   ├── add to vocabulary
   ├── store merge rule
   └── store merge rank
   │
   ▼
Merge pair across ALL sequences
   │
   ▼
Repeat until target vocabulary size
```

### What happens inside one merge

Suppose:

```text
(p, l)
```

is the most frequent pair.

The tokenizer:

```text
1. finds how often (p,l) occurs

2. creates a new token ID

3. stores:
      (p,l) → token ID

4. stores:
      (p,l) → merge rank

5. scans the entire corpus

6. replaces every matching (p,l)
```

Then the next iteration starts from scratch.

## Main inefficiency

For every merge:

```text
Count all pairs again
+
Scan all sequences again
```

So approximately:

```text
Full corpus
×
Number of merges
```

With an 8,192-token vocabulary:

```text
8192 - 256 byte tokens - special tokens
≈ 7,935 merges
```

That means thousands of full-corpus passes.

## Benchmark

On approximately 1M characters:

```text
Vocabulary = 512
Merges     = 255

Training time ≈ 20.58 sec
```

The implementation is easy to understand, which makes it a good baseline.

---

# 4. BPE v2 — Incremental Pair Statistics

File:

```text
tokenizer/bpe_v2.py
```

## Why v2?

The obvious problem with v1 was:

```text
recount the entire corpus
```

after every merge.

So v2 attempted to maintain information about:

```text
pair counts

+

which sequences contain each pair
```

## Main flow

```text
Raw text
   │
   ▼
UTF-8 bytes
   │
   ▼
Build initial pair statistics
   │
   ├── pair_counts
   └── pair_to_sequences
   │
   ▼
Select most frequent pair
   │
   ▼
Find sequences affected by that pair
   │
   ▼
Remove old pair statistics
   │
   ▼
Merge only affected sequences
   │
   ▼
Add updated pair statistics
   │
   ▼
Repeat
```

Instead of doing:

```text
ALL sequences
```

after every merge, v2 tried to do:

```text
ONLY affected sequences
```

## What changed internally?

v2 introduced bookkeeping structures such as:

```text
pair_counts
pair_to_sequences
```

Conceptually:

```text
Pair:

(p,l)

pair_counts:

(p,l) → 8

pair_to_sequences:

(p,l) → {sequence 0, sequence 1, ...}
```

After merging `(p,l)`, only those sequences need to be updated.

## Why didn't it work?

The optimization added substantial Python overhead:

```text
Counter creation
set management
pair → sequence tracking
removing statistics
adding statistics
```

For our corpus size, that bookkeeping cost more than the scanning it was trying to avoid.

## Benchmark

Same benchmark:

```text
~1M characters
Vocabulary = 512
Merges     = 255
```

Result:

```text
v1 → 20.58 sec
v2 → 35.37 sec
```

So v2 was actually:

```text
SLOWER
```

than v1.

## Takeaway

The idea of incremental updates is valid, but this particular Python implementation did not produce a performance improvement.

This experiment was useful because it showed that:

> Avoiding full scans is not automatically faster if the bookkeeping required to do so is more expensive.

---

# 5. Final bpe.py — Pre-tokenized + Frequency-Based BPE

File:

```text
tokenizer/bpe.py
```

This is the final implementation we use for the project.

Instead of giving BPE every occurrence of every word independently, we first pre-tokenize the text and collapse repeated pieces into:

```text
unique piece
+
frequency
```

## Main idea

```text
Raw text
   ↓
Pre-tokenization
   ↓
Unique pieces + frequency
   ↓
UTF-8 bytes
   ↓
Frequency-weighted pair counting
   ↓
BPE merges
```

---

# 6. Why Pre-tokenization Helps

Our original representation could contain many repeated occurrences:

```text
"the"
"the"
"the"
"the"
...
```

Instead of processing every occurrence separately, we can represent:

```text
"the" → frequency = 9485
```

For our ~1M-character sample:

```text
Total pre-tokenized pieces = 242,691

Unique pieces              = 4,876
```

So we reduced:

```text
242,691 occurrences
```

to:

```text
4,876 unique pieces
+
their frequencies
```

This is the major source of the final implementation's speedup.

---

# 7. Pre-tokenization

Our current educational pre-tokenizer uses:

```text
words
numbers
punctuation
whitespace
```

and attaches a leading space to words/numbers where possible.

For example:

```text
"the cat is here"
```

becomes conceptually:

```text
["the", " cat", " is", " here"]
```

rather than:

```text
["the", " ", "cat", " ", "is", " ", "here"]
```

This is useful because recurring patterns such as:

```text
" the"
" and"
" to"
" was"
```

can become reusable BPE pieces.

---

# 8. Final Training Flow

```text
Raw TinyStories
      │
      ▼
Pre-tokenization
      │
      ▼
Unique piece → frequency
      │
      ▼
UTF-8 encode each unique piece
      │
      ▼
Frequency-weighted pair counts
      │
      ▼
Select globally most frequent pair
      │
      ▼
Create new token
      │
      ├── add to vocabulary
      ├── store merge rule
      └── store merge rank
      │
      ▼
Merge selected pair
inside unique pieces
      │
      ▼
Repeat
```

---

# 9. Frequency-Weighted Pair Counting

This is the important change in the final implementation.

Suppose:

```text
" the" → frequency 9,484
```

and the byte sequence contains:

```text
(t, h)
```

Instead of:

```text
(t,h) += 1
```

we do:

```text
(t,h) += 9,484
```

So pair counts represent the total frequency of the pair across the original corpus.

Conceptually:

```text
Unique piece: " the"

Frequency:    9484

     │
     ▼

byte sequence
     │
     ▼
adjacent pairs
     │
     ▼
each pair contributes 9484
```

This allows BPE to operate on the smaller set of unique pieces while preserving the frequency information of the original corpus.

---

# 10. Encoding Flow

Training and encoding must use the same pre-tokenization boundaries.

```text
Raw text
   │
   ▼
Handle special tokens
   │
   ▼
Pre-tokenize normal text
   │
   ▼
Encode each piece separately
   │
   ▼
UTF-8 bytes
   │
   ▼
Find applicable learned merges
   │
   ▼
Choose lowest merge rank
   │
   ▼
Merge
   │
   ▼
Repeat until no merge applies
   │
   ▼
Token IDs
```

For example:

```text
"played"

   ↓

pre-tokenization

   ↓

"played"

   ↓

UTF-8 bytes

   ↓

BPE merges

   ↓

[learned token IDs]
```

The tokenizer does not explicitly solve:

```text
"minimum possible number of tokens"
```

Instead it follows the learned merge ranking.

---

# 11. Merge Ranks

Every learned merge is assigned a rank.

For example:

```text
(p,l)       → rank 0
(256,a)     → rank 1
(257,y)     → rank 2
```

Lower rank means:

```text
learned earlier
=
higher priority
```

During encoding:

```text
current tokens
      ↓
find all applicable learned pairs
      ↓
select pair with lowest rank
      ↓
merge it
      ↓
repeat
```

This is why intermediate tokens can exist in the vocabulary while a larger token is used during encoding.

For example:

```text
pl
pla
play
playe
played
```

can all exist in the vocabulary.

---

# 12. Vocabulary Structure

The tokenizer maintains:

```text
vocab
merges
merge_ranks
special_tokens
```

## Vocabulary

Answers:

> What does each token ID represent?

Example:

```text
256 → b'pl'
257 → b'pla'
258 → b'play'
259 → b'playe'
261 → b'played'
```

## Merge rules

Answers:

> Which pair created this token?

Example:

```text
(112,108) → 256
(256,97)  → 257
(257,121) → 258
```

## Merge ranks

Answers:

> In what priority order should merges be applied?

Example:

```text
(112,108) → rank 0
(256,97)  → rank 1
(257,121) → rank 2
```

## Special tokens

Answers:

> Which tokens are handled outside normal BPE?

Example:

```text
<|endoftext|> → 256
```

---

# 13. Decode Flow

Decoding is simpler than encoding.

```text
Token IDs
   │
   ▼
Vocabulary lookup
   │
   ▼
Byte sequences
   │
   ▼
Concatenate bytes
   │
   ▼
UTF-8 decode
   │
   ▼
Original text
```

Special tokens are handled separately:

```text
EOS token ID
   ↓
<|endoftext|>
```

Decoding does not need merge ranks because the vocabulary already tells us what bytes each token represents.

---

# 14. Save / Load

A trained tokenizer needs to persist its learned state.

```text
Training
   ↓
vocab
merges
merge_ranks
special_tokens
   ↓
save()
   ↓
tinystories_bpe_8192.json
```

Later:

```text
tinystories_bpe_8192.json
   ↓
load()
   ↓
new tokenizer instance
   ↓
encode()
decode()
```

We verified:

```text
train
→ save
→ load
→ encode
→ decode
```

and the round trip succeeded.

---

# 15. Final Tokenizer

The final tokenizer was trained on:

```text
Training stories:    25,000
Training characters: 20,087,753
```

Target vocabulary:

```text
8,192 tokens
```

The vocabulary consists of:

```text
256 base byte tokens
+
7,935 learned BPE tokens
+
1 special token
=
8,192 total token IDs
```

The final tokenizer was saved as:

```text
artifacts/tinystories_bpe_8192.json
```

---

# 16. Performance Comparison

All benchmarks were performed on approximately 1M characters using the same basic setup.

```text
Target vocabulary = 512
BPE merges         = 255
```

| Version | Main Approach | Time |
|---|---|---:|
| v1 | Full corpus rescanned every merge | 20.58 sec |
| v2 | Incremental pair/sequence bookkeeping | 35.37 sec |
| Final bpe.py | Pre-tokenization + unique pieces + frequencies | 0.80 sec |

For the final implementation with the actual target:

```text
Characters        ≈ 1,000,670
Vocabulary size   = 8,192
BPE merges        = 7,935
Training time     = 12.24 sec
```

The final tokenizer was subsequently trained on the full TinyStories training corpus:

```text
Training stories:    25,000
Training characters: 20,087,753
Vocabulary size:      8,192
BPE merges:            7,935
Training time:        ~50.60 sec
```

---

# 17. Tokenization Statistics

The final tokenizer produced the following statistics on the full training corpus:

```text
Training characters: 20,087,753
Training tokens:      4,907,617

Characters/token:     ~4.09
Tokens/character:     ~0.244
```

On the validation corpus:

```text
Validation stories:   2,000
Validation characters: 1,617,811
Validation tokens:       396,525

Characters/token:     ~4.08
Tokens/character:     ~0.245
```

The tokenizer was trained only on the training corpus and then used to encode validation data.

---

# 18. Why the Final Implementation Is Faster

The biggest difference is the representation of the training data.

### v1

```text
~1M characters
      ↓
many individual token occurrences
      ↓
process repeatedly
```

### Final bpe.py

```text
~1M characters
      ↓
pre-tokenization
      ↓
4,876 unique pieces
+
frequency
      ↓
BPE
```

Instead of processing every occurrence independently, repeated pieces are represented once and weighted by their frequency.

---

# 19. Why the Final Implementation Is Still Educational, Not Production Grade

The final implementation is much better than v1 for our project, but it is still not a production tokenizer implementation.

The main reason is that every merge still performs Python-level work over the unique pieces.

Production implementations use:

```text
optimized data structures
efficient pair management
optimized memory representation
compiled implementations
parallelism where appropriate
```

Libraries such as Hugging Face Tokenizers and OpenAI's tokenizer implementations are heavily optimized for this reason.

Our goal here is different:

> Understand the algorithm first, understand the performance bottleneck second, and only then use a production tokenizer implementation when needed.

---

# 20. Final Evolution

The tokenizer evolved like this:

```text
BPE v1
──────────────

Full corpus
     ↓
Recount everything
     ↓
Merge everything
     ↓
Repeat

Simple
Easy to understand
Slow at scale

        │
        ▼

BPE v2
──────────────

Full corpus
     ↓
Build pair index
     ↓
Find affected sequences
     ↓
Update statistics
     ↓
Repeat

More bookkeeping
Actually slower in our benchmark
Useful optimization experiment

        │
        ▼

Final bpe.py
──────────────

Raw text
     ↓
Pre-tokenization
     ↓
Unique pieces + frequency
     ↓
Frequency-weighted pairs
     ↓
Merge unique pieces
     ↓
Repeat

Much faster
Closer to practical BPE training structure
Still intentionally educational
```

---

# 21. Final Learning

The main lesson from the three implementations is:

```text
The BPE algorithm itself is simple.

The difficult part is making BPE efficient at scale.
```

The large performance improvement did not come from changing the fundamental idea of BPE.

It came from changing:

```text
what data we process
+
how often we process it
+
how we represent repeated patterns
```

The evolution was therefore:

```text
Algorithm
   ↓
Naive implementation
   ↓
Performance bottleneck
   ↓
Optimization attempt
   ↓
Benchmark
   ↓
Better data representation
   ↓
Much faster implementation
```

This progression is useful beyond tokenizers and is a recurring pattern in ML systems engineering.