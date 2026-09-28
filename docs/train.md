# Training — How Mini GPT Learns

Training is the process that turns the Mini GPT architecture from a randomly initialized model into a model that can predict the next token.

At a high level, the training loop repeatedly performs:

```text
Input tokens
    ↓
Model forward pass
    ↓
Logits
    ↓
Cross-entropy loss
    ↓
Backpropagation
    ↓
Gradients
    ↓
Gradient clipping
    ↓
AdamW parameter update
    ↓
Learning-rate scheduler update
    ↓
Next batch
```

After all batches in an epoch, the model is evaluated on the validation dataset and checkpoints are saved.

The important part is understanding that **three different states are involved during training**:

```text
Model parameters
    ↓
are updated by the optimizer

Gradients
    ↓
are calculated by backpropagation
and cleared before the next update

Learning rate
    ↓
is controlled by the scheduler
and determines the scale of future updates
```

These are related, but they are not the same thing.

---

## 1. The Complete Training Flow

The training process in this project can be understood as one continuous pipeline:

```text
                         TRAINING
                            │
                            ▼
                    Get one batch
                            │
                            ▼
                  Move tensors to device
                            │
                            ▼
                    Forward pass
                            │
                            ▼
                   Model produces logits
                            │
                            ▼
                  Calculate cross-entropy
                            │
                            ▼
                         Loss
                            │
                            ▼
                  Clear old gradients
                            │
                            ▼
                    Backpropagation
                            │
                            ▼
                       Gradients
                            │
                            ▼
                  Gradient clipping
                            │
                            ▼
                    AdamW optimizer
                            │
                            ▼
                  Update model weights
                            │
                            ▼
                  Update learning rate
                            │
                            ▼
                      Next batch
```

At the end of an epoch:

```text
                    EPOCH COMPLETE
                           │
                           ▼
                    Switch to eval mode
                           │
                           ▼
                    Run validation
                           │
                           ▼
                     Validation loss
                           │
                 ┌─────────┴─────────┐
                 │                   │
                 ▼                   ▼
          Save latest          Check if best
          checkpoint                │
                                    ▼
                            Save best checkpoint
```

This is the core training engine implemented in `train.py`.

---

## 2. Batch → Forward Pass → Loss

The dataset provides:

```text
input_ids  → tokens the model sees
target_ids → correct next tokens
```

For this project, a batch has:

```text
input_ids:
[batch_size, sequence_length]

[32, 512]
```

The model produces:

```text
logits:
[batch_size, sequence_length, vocab_size]

[32, 512, 8192]
```

This means that for every position in every sequence, the model produces 8192 scores — one for every possible token in the vocabulary.

The training objective is next-token prediction:

```text
Input:

The dog chased the

Target:

dog chased the cat
```

The model therefore needs to predict the target token at every position.

The loss is calculated with:

```python
loss = loss_fn(
    logits.reshape(-1, VOCAB_SIZE),
    target_ids.reshape(-1),
)
```

The reshape converts:

```text
logits
[32, 512, 8192]
```

into:

```text
[16384, 8192]
```

because:

```text
32 × 512 = 16,384
```

So CrossEntropyLoss receives:

```text
16,384 predictions
×
8,192 possible classes
```

while the targets become:

```text
[16,384]
```

Each element in the target identifies the correct next token for one prediction.

The result is a single scalar:

```text
loss = ...
```

The objective of training is to reduce this value.

---

## 3. Forward Pass Does Not Change the Model

This distinction is important.

When we execute:

```python
logits = model(input_ids)
```

the model performs a forward pass:

```text
input
  ↓
embedding
  ↓
Transformer blocks
  ↓
final normalization
  ↓
LM head
  ↓
logits
```

At this point, the model parameters have **not been updated**.

The forward pass only calculates predictions.

Then:

```python
loss = loss_fn(...)
```

measures how far those predictions are from the correct targets.

Only after backpropagation and the optimizer step do the parameters change.

---

## 4. Gradients: How Does the Model Know What to Change?

Once the loss has been calculated:

```python
loss.backward()
```

performs backpropagation.

Conceptually, for every trainable parameter, PyTorch calculates:

```text
∂Loss / ∂Parameter
```

This gradient tells us how the loss would change if that parameter changed slightly.

A simplified view is:

```text
Loss
 │
 ├── gradient for parameter 1
 ├── gradient for parameter 2
 ├── gradient for parameter 3
 ├── gradient for parameter 4
 └── ...
```

After:

```python
loss.backward()
```

the gradients are stored on the model parameters.

For example:

```python
parameter.grad
```

contains the gradient associated with that parameter.

This is why backpropagation and optimization should be viewed as two separate operations:

```text
backward()
    ↓
calculate gradients

optimizer.step()
    ↓
use gradients to update parameters
```

Backpropagation determines **how the parameters should change**.

The optimizer actually **changes them**.

---

## 5. Why Gradients Must Be Cleared

Before calculating gradients for a new batch, the code runs:

```python
optimizer.zero_grad()
```

PyTorch accumulates gradients by default.

For example, imagine:

```text
Batch 1 gradient = 0.3
Batch 2 gradient = 0.2
```

If the old gradient is not cleared, the next gradient can effectively accumulate:

```text
0.3 + 0.2 = 0.5
```

That is not the behavior wanted for ordinary batch-by-batch training.

Therefore each training iteration starts by clearing the gradients:

```text
old gradients
    ↓
zero_grad()
    ↓
fresh gradients for current batch
```

The core sequence is therefore:

```python
optimizer.zero_grad()

logits = model(input_ids)

loss = loss_fn(...)

loss.backward()
```

---

## 6. Gradient Clipping

After backpropagation, the model has gradients.

The code then applies:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    GRAD_CLIP,
)
```

with:

```python
GRAD_CLIP = 1.0
```

The purpose is to protect the optimizer from unusually large gradient updates.

Without clipping, an unusually large gradient can produce a very large parameter update and destabilize training.

Gradient clipping limits the overall gradient norm.

Importantly, this does **not** mean:

```text
every gradient is individually restricted to [-1, 1]
```

Instead, the collection of gradients is treated through its overall norm and scaled when necessary so that the norm does not exceed the configured maximum.

The conceptual flow is:

```text
Backpropagation
      ↓
Gradients
      ↓
Are gradients unusually large?
      ↓
Clip if necessary
      ↓
Optimizer
```

For this project:

```text
Maximum gradient norm = 1.0
```

Gradient clipping is therefore a **stability mechanism**. It does not teach the model anything by itself; it prevents unstable updates from disrupting the learning process.

---

## 7. AdamW: Updating the Model

After gradients have been calculated and clipped:

```python
optimizer.step()
```

updates the model parameters.

The optimizer used in this project is:

```python
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
```

with:

```text
Learning rate = 3e-4
Weight decay  = 0.1
```

A simple gradient-descent update can be thought of as:

```text
new_parameter
=
old_parameter
-
learning_rate × gradient
```

AdamW is more sophisticated than basic gradient descent.

It keeps running statistics of gradients and uses those statistics to adapt parameter updates.

Conceptually:

```text
Current gradient
       +
gradient history
       ↓
     AdamW
       ↓
parameter update
```

AdamW also uses decoupled weight decay.

The two concepts serve different purposes:

```text
Learning rate
    → controls the scale of parameter updates

Weight decay
    → regularizes parameter values
```

So `weight_decay=0.1` should not be interpreted as another form of learning rate.

---

## 8. Learning Rate Is Not Constant

The optimizer is initialized with:

```python
lr=LEARNING_RATE
```

where:

```python
LEARNING_RATE = 3e-4
```

However, the actual learning rate changes during training because this project uses a scheduler.

The training therefore does not simply use:

```text
3e-4
3e-4
3e-4
3e-4
...
```

Instead, it follows:

```text
Warmup
   ↓
Increase learning rate
   ↓
Reach base learning rate
   ↓
Cosine decay
   ↓
Finish at a smaller learning rate
```

This is controlled by:

```python
scheduler = create_scheduler(
    optimizer,
    total_steps,
)
```

---

## 9. Warmup

The project uses:

```python
WARMUP_RATIO = 0.05
```

The total number of training steps is:

```python
total_steps = len(train_loader) * NUM_EPOCHS
```

For the current dataset:

```text
~301 batches per epoch
×
5 epochs
≈
1,505 total optimizer steps
```

The warmup period is therefore approximately:

```text
1,505 × 0.05
≈
75 steps
```

During these initial steps, the learning rate increases gradually.

The scheduler calculates:

```python
float(step + 1) / warmup_steps
```

This produces a multiplier that gradually moves toward:

```text
1.0
```

Because `LambdaLR` works with a multiplier, the actual learning rate is conceptually:

```text
actual learning rate
=
base learning rate
×
scheduler multiplier
```

So the training begins with a much smaller learning rate and gradually reaches the base learning rate.

The purpose of warmup is to make the initial optimization phase less aggressive.

---

## 10. Cosine Learning-Rate Decay

After warmup, the scheduler switches to cosine decay.

The code calculates:

```python
progress = (
    step - warmup_steps
) / max(
    1,
    total_steps - warmup_steps,
)
```

This converts the remaining training period into a normalized value:

```text
progress = 0 → beginning of decay
progress = 1 → end of training
```

The schedule then uses:

```python
0.1 + 0.9 * 0.5 * (
    1.0 + math.cos(math.pi * progress)
)
```

The important behavior is:

```text
Beginning of cosine phase
    multiplier ≈ 1.0

End of cosine phase
    multiplier ≈ 0.1
```

Therefore, with a base learning rate of:

```text
3e-4
```

the schedule approximately becomes:

```text
             Learning Rate

3e-4  ────────╮
               ╲
                ╲
                 ╲
                  ╲
                   ╲
                    ╲
3e-5  ───────────────╰────────

       warmup        cosine decay
```

The schedule does not decay all the way to zero. It finishes at approximately 10% of the base learning rate.

So the overall learning-rate strategy is:

```text
Small LR
   ↓
Warmup
   ↓
3e-4
   ↓
Cosine decay
   ↓
~3e-5
```

---

## 11. Why the Scheduler Is Separate From AdamW

AdamW and the scheduler have different responsibilities.

```text
AdamW
    ↓
decides how gradients are converted into parameter updates

Scheduler
    ↓
decides how large the learning-rate scale should be over time
```

The relationship is:

```text
                     Learning-rate scheduler
                              │
                              ▼
                         Current LR
                              │
                              ▼
Gradients ────────────────→ AdamW
                              │
                              ▼
                       Updated parameters
```

The scheduler does not update model weights directly.

`optimizer.step()` changes the weights.

`scheduler.step()` changes the learning rate used by future optimizer updates.

---

## 12. Ordering of the Training Operations

The order in `train.py` is:

```python
optimizer.zero_grad()

logits = model(input_ids)

loss = loss_fn(...)

loss.backward()

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    GRAD_CLIP,
)

optimizer.step()

scheduler.step()
```

This ordering represents the complete optimization cycle.

```text
1. Clear previous gradients
          ↓
2. Forward pass
          ↓
3. Calculate loss
          ↓
4. Backpropagate
          ↓
5. Obtain gradients
          ↓
6. Clip gradients if necessary
          ↓
7. AdamW updates model parameters
          ↓
8. Scheduler updates learning rate
          ↓
9. Process next batch
```

A useful way to remember the distinction is:

```text
backward()
    = "How should each parameter change?"

optimizer.step()
    = "Actually change the parameters."

scheduler.step()
    = "What learning rate should the next update use?"
```

---

## 13. Step vs Epoch

A **training step** corresponds to one batch being processed.

With:

```text
batch size = 32
```

one step processes approximately:

```text
32 sequences
```

The sequence is:

```text
one batch
    ↓
forward
    ↓
loss
    ↓
backward
    ↓
optimizer update
```

An **epoch** is one complete pass through the training dataset.

For this project:

```text
~9,634 training sequences
÷
32 sequences per batch
≈
301 steps per epoch
```

With:

```text
5 epochs
```

the training performs approximately:

```text
301 × 5
≈
1,505 optimizer steps
```

Therefore:

```text
step ≠ epoch
```

A step is one parameter-update iteration.

An epoch is one pass through the dataset.

This distinction is particularly important because the learning-rate scheduler operates in terms of training steps.

---

## 14. Training Mode vs Evaluation Mode

During training, the code uses:

```python
model.train()
```

This places the model into training mode.

At validation time:

```python
model.eval()
```

switches it to evaluation mode.

This matters for layers whose behavior differs between training and evaluation, such as Dropout or BatchNorm.

The current Mini GPT does not rely heavily on those behaviors, but using the correct mode is still important and makes the training code robust as the architecture evolves.

The validation code therefore starts with:

```python
model.eval()
```

and uses:

```python
with torch.no_grad():
```

---

## 15. `torch.no_grad()` During Validation

Validation does not update the model.

We only want to answer:

> How well does the current model perform on data it is not training on?

There is no need to calculate gradients.

Therefore:

```python
with torch.no_grad():
```

disables gradient tracking for the validation forward passes.

This saves memory and computation.

It is important to distinguish:

```text
model.eval()
    → changes model behavior to evaluation mode

torch.no_grad()
    → disables gradient tracking
```

They solve different problems and are commonly used together during validation.

The validation flow is therefore:

```text
Validation batch
      ↓
model.eval()
      ↓
Forward pass
      ↓
Loss
      ↓
No backward pass
      ↓
Next validation batch
```

---

## 16. Training Loss vs Validation Loss

During training, the model sees:

```text
training data
```

and its parameters are updated after every batch.

At the end of the epoch, the model is evaluated on:

```text
validation data
```

without updating the weights.

The two metrics answer different questions:

```text
Training loss
    → How well is the model fitting the training data?

Validation loss
    → How well does the current model perform on unseen validation data?
```

This is why validation loss is useful for deciding which checkpoint is the best one.

The training code calculates:

```python
train_loss = total_train_loss / len(train_loader)
```

and:

```python
val_loss = evaluate(
    model,
    val_loader,
    loss_fn,
    device,
)
```

The resulting values are reported after every epoch.

---

## 17. Checkpointing the Complete Training State

The project saves more than model weights.

A checkpoint contains:

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "epoch": epoch,
    "step": step,
    "train_loss": train_loss,
    "val_loss": val_loss,
}
```

The three most important states are:

```text
Model state
    → learned parameters

Optimizer state
    → optimizer history and state required for continuation

Scheduler state
    → current position in the learning-rate schedule
```

The epoch, step, and losses provide training metadata.

This makes the checkpoint much more useful than saving only:

```python
model.state_dict()
```

If training needs to be resumed, the optimizer and scheduler states are important because training should continue from the same optimization state rather than starting those components from scratch.

---

## 18. Latest vs Best Checkpoint

The project maintains two checkpoints.

### Latest checkpoint

```text
checkpoints/latest_model.pt
```

This represents the most recent training state.

After every epoch it is overwritten with the latest state.

### Best checkpoint

```text
checkpoints/best_model.pt
```

This represents the model with the lowest validation loss observed so far.

The logic is:

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    save_checkpoint(...)
```

Initially:

```python
best_val_loss = float("inf")
```

so the first validation result automatically becomes the current best.

As training continues, only an improvement in validation loss replaces the best checkpoint.

This creates a useful separation:

```text
latest
    → Where training currently ended

best
    → Best validation performance observed during training
```

For generation and evaluation, the best validation checkpoint is generally the more meaningful baseline.

---

## 19. One Complete Training Step

Putting everything together, one batch of the current Mini GPT training looks like this:

```text
                  INPUT BATCH
                [32, 512] tokens
                       │
                       ▼
                Move to device
                       │
                       ▼
                  Mini GPT
                       │
                       ▼
              LOGITS [32,512,8192]
                       │
                       ▼
               CrossEntropyLoss
                       │
                       ▼
                      LOSS
                       │
                       ▼
             loss.backward()
                       │
                       ▼
                  GRADIENTS
                       │
                       ▼
             Gradient clipping
                       │
                       ▼
                   AdamW
                       │
                       ▼
              UPDATED WEIGHTS
                       │
                       ▼
                scheduler.step()
                       │
                       ▼
               UPDATED LR
                       │
                       ▼
                 NEXT BATCH
```

There are therefore two different things being updated after each batch:

```text
Model parameters
    → updated by AdamW

Learning rate
    → updated by the scheduler
```

The gradients are temporary information calculated for the current optimization step.

```text
Gradients
    → calculated by backward()
    → used by optimizer.step()
    → cleared before the next batch
```

---

## 20. Complete Epoch Flow

One complete epoch looks like:

```text
                  TRAINING DATASET
                         │
                         ▼
                    DataLoader
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
           Batch 1               Batch 2
              │                     │
              ▼                     ▼
        Forward + Loss        Forward + Loss
              │                     │
              ▼                     ▼
          Backward               Backward
              │                     │
              ▼                     ▼
           Clip                  Clip
              │                     │
              ▼                     ▼
          AdamW                  AdamW
              │                     │
              ▼                     ▼
        Scheduler              Scheduler
              │                     │
              └──────────┬──────────┘
                         │
                       ...
                         │
                         ▼
                    Last batch
                         │
                         ▼
                  Epoch complete
                         │
                         ▼
                  Validation
                         │
                         ▼
                   Val loss
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       Save latest             Better than best?
                                     │
                                     ▼
                               Save best
```

This is the full lifecycle of one epoch in the current implementation.

---

## 21. Training State at Any Point

At a particular training step, it is useful to think of the system as having three separate moving pieces:

```text
┌─────────────────────────────────────────┐
│             MODEL PARAMETERS            │
│                                         │
│ Embeddings                              │
│ Attention weights                       │
│ SwiGLU weights                          │
│ Normalization parameters                │
│ LM head                                 │
└─────────────────────────────────────────┘
                    ▲
                    │
              optimizer.step()
                    │
                    │
┌───────────────────┴─────────────────────┐
│                GRADIENTS                │
│                                         │
│ Calculated from current loss            │
│ Used for the current parameter update   │
│ Cleared before the next batch           │
└───────────────────▲─────────────────────┘
                    │
               loss.backward()
                    │
                    │
┌───────────────────┴─────────────────────┐
│                   LOSS                  │
│                                         │
│ Measures next-token prediction error    │
└───────────────────▲─────────────────────┘
                    │
               model(input)
                    │
                    │
┌───────────────────┴─────────────────────┐
│                  INPUT                  │
│                                         │
│ Token IDs                               │
└─────────────────────────────────────────┘


Learning-rate scheduler
        │
        ▼
Controls the learning-rate scale used
by future optimizer updates.
```

This mental model is useful because it prevents several concepts from being conflated.

---

## 22. The Core Training Mental Model

The entire process can be reduced to:

```text
1. Give the model tokens.

2. The model predicts the next token at every position.

3. Compare those predictions with the correct next tokens.

4. Produce one loss value.

5. Backpropagation calculates how every parameter contributed
   to that loss.

6. Gradient clipping prevents unusually large updates.

7. AdamW uses the gradients to update the parameters.

8. The learning-rate scheduler controls how aggressively
   future updates are made.

9. Repeat for every batch.

10. Evaluate on validation data after each epoch.

11. Save the latest state and preserve the best validation state.
```

The most important relationships are:

```text
                 Forward pass
                      │
                      ▼
                   Loss
                      │
                      ▼
                Backpropagation
                      │
                      ▼
                 Gradients
                      │
                      ▼
             Gradient clipping
                      │
                      ▼
                   AdamW
                      │
                      ▼
              Model parameters
                      ▲
                      │
             Learning rate
                      ▲
                      │
              LambdaLR scheduler
```

And the most important distinction to retain is:

```text
Backward pass
    = calculates gradients

Optimizer
    = updates model parameters

Scheduler
    = changes the learning rate used by the optimizer

Validation
    = measures the current model without updating it

Checkpoint
    = stores the state needed to preserve or resume training
```

This is the optimization loop underneath the Mini GPT. Once this flow is clear, concepts such as different optimizers, learning-rate schedules, gradient accumulation, mixed precision, distributed training, and scaling experiments can be understood as variations or extensions of this same core loop.