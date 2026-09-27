# Mini GPT Baseline Evaluation

This document captures the behavior of the first complete Mini GPT baseline after training.

The purpose is to record what worked, what limitations were observed, and what should be investigated in the next optimization phase.

---

## 1. Baseline

The model was trained with:

```text
Parameters:             25.17M
Vocabulary size:        8,192
d_model:                512
Transformer blocks:     4
Attention heads:        8
Context length:         512

Training tokens:        ~4.93M
Training epochs:        5
Batch size:             32
Learning rate:          3e-4
Optimizer:              AdamW

Best validation loss:   2.1337
Validation perplexity:  ~8.45
```

The baseline successfully completed the full pipeline:

```text
Raw text
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Mini GPT
   ↓
Training
   ↓
Validation
   ↓
KV-cache generation
   ↓
Sampling
   ↓
Generated text
```

---

## 2. Training Findings

The model learned effectively during the baseline run.

| Epoch | Train Loss | Validation Loss |
|------:|-----------:|----------------:|
| 1 | 4.1669 | 2.9104 |
| 2 | 2.6256 | 2.4617 |
| 3 | 2.2914 | 2.2613 |
| 4 | 2.1016 | 2.1685 |
| 5 | 1.9976 | 2.1337 |

### Observations

- Training loss decreased substantially.
- Validation loss also improved at every epoch.
- There was no obvious overfitting during the five-epoch baseline run.
- The final validation perplexity was approximately `8.45`.
- The model clearly learned statistical patterns and the style of the TinyStories corpus.
- The model is capable of producing meaningful TinyStories-style continuations.

---

## 3. Generation Behavior

The model was tested with different prompts and sampling configurations.

The generation pipeline uses:

```text
Temperature
Top-K
Top-P
KV Cache
Maximum new tokens
EOS stopping
```

The examples below are representative observations from the baseline model.

---

## Example 1 — Normal Temperature

### Prompt

```text
There was a girl named
```

### Example response

```text
There was a girl named Lily. Lily loved to play with her toys and eat yummy food. One day, she found a big, red ball in her yard. She was very happy and wanted to play with it.
```

### Observation

At approximately `temperature = 1.0`, the model generally produces:

- Fluent text
- Familiar TinyStories patterns
- Reasonable local coherence
- Different story directions across different samples

The model has clearly learned the style and vocabulary distribution of the training corpus.

However, coherence can deteriorate as the generation becomes longer.

---

## 4. Temperature Experiment

The same or similar prompts were tested with different temperatures.

### Temperature ≈ 0.1–0.2

#### Prompt

```text
There was a girl named
```

#### Example responses

```text
There was a girl named Lily. Lily loved to play with her toy car. Lily loved to play with her toys and she was very happy.
```

Another run produced:

```text
There was a girl named Lily. Lily was a very curious girl. She loved to play with her ball and go to the park.
```

### Observation

Low temperature makes the probability distribution very sharp.

The model therefore tends to select high-probability continuations repeatedly.

This produces:

- More predictable generations
- More repetition
- Stronger preference for common training patterns

Importantly, low temperature does **not** improve the model's underlying knowledge.

If the model has learned an incorrect or awkward association, low temperature can make that behavior more consistent.

---

### Temperature = 1.0

At `temperature = 1.0`, the model samples from its original probability distribution.

Example behavior:

```text
There was a girl named Sue who found a magic car...
```

Another run may produce:

```text
There was a girl named Lily who was curious and had a dog named Max...
```

### Observation

The output varies between runs while generally maintaining the TinyStories style.

This provides a useful balance between predictability and variation for this small model.

---

### Temperature = 2.0

#### Prompt

```text
There was a girl named
```

#### Example responses

```text
There was a girl named Lily and she wanted an emergent at the park. She picked him one piece of a cup and tasted cool down like a fish.
```

Another example:

```text
There was a girl named Sue who did help from Tom's hand. The post just needed a battery...
```

### Observation

At high temperature, the probability distribution becomes flatter.

Lower-probability tokens therefore have a much greater chance of being sampled.

The model starts producing:

- Unusual word combinations
- Grammatical errors
- Semantic inconsistencies
- Less coherent stories

This is particularly noticeable with a relatively small 25M parameter model.

---

## 5. Question-Like Prompts

The model was also tested with prompts that resemble questions.

### Prompt

```text
Who was Tim
```

### Example response

```text
Who was Tim? Tim was very happy to play with his friends. He had a big toy car and wanted to go outside...
```

Other runs produced unrelated TinyStories-style continuations.

### Observation

The model does not reliably answer questions.

This is expected because the model was trained using causal next-token prediction rather than instruction tuning or question-answer training.

A question-like prompt is therefore treated primarily as another sequence that needs continuation.

```text
previous tokens
      ↓
predict next token
```

---

## 6. Sampling Findings

Temperature, Top-K, and Top-P affect generation differently.

### Temperature

Controls how sharply the probability distribution is concentrated.

```text
Lower temperature
    ↓
Sharper distribution
    ↓
More predictable output
```

```text
Higher temperature
    ↓
Flatter distribution
    ↓
More varied output
```

### Top-K

Restricts sampling to the K highest-probability tokens.

```text
All vocabulary
     ↓
Select K highest-probability tokens
     ↓
Sample from those tokens
```

### Top-P

Selects the smallest set of tokens whose cumulative probability reaches the chosen probability threshold.

```text
Sort tokens by probability
        ↓
Accumulate probabilities
        ↓
Keep tokens until cumulative probability ≥ P
        ↓
Sample
```

The generation playground was used to observe these effects interactively.

Future experiments should isolate each parameter rather than changing several parameters simultaneously.

---

## 7. KV Cache Validation

The KV-cache implementation was compared with normal full-sequence generation.

The corresponding outputs matched, validating the cache implementation for generation.

Conceptually:

```text
Full-sequence generation
          ≈
KV-cache generation
```

The KV cache therefore changes the computation strategy rather than changing the model's predicted output.

---

## 8. What Is Working Well

The baseline successfully demonstrates the complete language-model pipeline.

### Architecture

- Token embeddings work.
- RMSNorm works.
- RoPE works.
- Multi-head causal attention works.
- SwiGLU works.
- Transformer blocks compose correctly.
- Mini GPT produces correctly shaped logits.

### Training

- The model trains successfully on the full dataset.
- Training loss decreases substantially.
- Validation loss also decreases.
- The baseline reaches approximately `8.45` validation perplexity.

### Generation

- The trained model generates coherent TinyStories-style text.
- Sampling parameters visibly affect generation behavior.
- EOS stopping works.
- KV-cache generation works.

### Learning value

The project now provides a complete path from:

```text
Raw text
   ↓
Tokenizer
   ↓
Training data
   ↓
Transformer
   ↓
Training
   ↓
Inference
   ↓
Sampling
```

---

## 9. Current Limitations

The model is intentionally small and trained on a relatively small corpus.

Observed limitations include:

- Weak semantic consistency over longer generations
- Repetition in some generations
- Grammatical and semantic errors
- Question-like prompts do not reliably produce direct answers
- High temperature quickly reduces coherence
- Low temperature can reinforce repetitive or incorrect patterns
- The model does not behave like an instruction-tuned assistant

These limitations are useful because they provide concrete targets for future experiments.

---

## 10. Next Optimization Phase

The baseline should now be treated as the reference model.

Future experiments should change one major variable at a time and compare against the baseline.

```text
Baseline
   ↓
Change one variable
   ↓
Train
   ↓
Evaluate
   ↓
Compare
   ↓
Keep / reject
   ↓
Document
```

### Training

Potential experiments:

- More training epochs
- Learning-rate changes
- Batch-size changes
- Scheduler changes
- Longer training runs

### Data

Potential experiments:

- More training data
- Data quality
- Sequence packing strategy
- Dataset composition

### Model

Potential experiments:

- Larger `d_model`
- More Transformer blocks
- Different attention configuration
- Larger or smaller feed-forward dimension
- Longer context length

### Generation

Potential experiments:

- Systematic Top-K experiments
- Systematic Top-P experiments
- Temperature comparison
- Greedy vs sampling
- Repetition analysis

---

## 11. Baseline Reference

The current baseline should remain unchanged while optimization experiments are performed.

```text
Model parameters:        25.17M
Vocabulary size:         8,192
d_model:                 512
Transformer blocks:      4
Attention heads:         8
Context length:          512

Training tokens:         ~4.93M
Training epochs:         5

Final train loss:        1.9976
Best validation loss:    2.1337
Validation perplexity:   ~8.45

KV cache:                Enabled
Temperature:             Enabled
Top-K:                   Enabled
Top-P:                   Enabled
```

This document should evolve as experiments are run and new findings are established.