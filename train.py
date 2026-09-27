import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from config import (
    TRAIN_PROCESSED_PATH,
    VAL_PROCESSED_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    VOCAB_SIZE,
    D_MODEL,
    NUM_HEADS,
    INTERMEDIATE_SIZE,
    NUM_BLOCKS,
)
from data.dataset import TinyStoriesDataset
from model.gpt import GPT


CHECKPOINT_DIR = Path("checkpoints")
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pt"
LATEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "latest_model.pt"

WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.05
GRAD_CLIP = 1.0


def get_device():
    """Select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def create_scheduler(optimizer, total_steps: int):
    """Create a warmup + cosine learning-rate schedule."""
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / warmup_steps

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)

        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def evaluate(model, data_loader, loss_fn, device):
    """Evaluate the model without calculating gradients."""
    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            logits = model(input_ids)

            loss = loss_fn(
                logits.reshape(-1, VOCAB_SIZE),
                target_ids.reshape(-1),
            )

            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches


def save_checkpoint(path, model, optimizer, scheduler, epoch, step, train_loss, val_loss):
    """Save model and training state."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    torch.save(checkpoint, path)


def main():
    device = get_device()
    print(f"Device: {device}")

    # Load preprocessed datasets.
    train_dataset = TinyStoriesDataset(TRAIN_PROCESSED_PATH)
    val_dataset = TinyStoriesDataset(VAL_PROCESSED_PATH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Steps per epoch: {len(train_loader)}")

    # Create the Mini GPT model.
    model = GPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        num_blocks=NUM_BLOCKS,
    ).to(device)

    num_parameters = sum(p.numel() for p in model.parameters())

    print(f"Model parameters: {num_parameters:,}")

    # Create loss function and optimizer.
    loss_fn = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = create_scheduler(optimizer, total_steps)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_loss = float("inf")

    print(f"Total training steps: {total_steps}")
    print(f"Initial learning rate: {LEARNING_RATE}")
    print()

    # Train for multiple epochs.
    for epoch in range(NUM_EPOCHS):
        model.train()

        total_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            # Forward pass.
            logits = model(input_ids)

            # Calculate next-token prediction loss.
            loss = loss_fn(
                logits.reshape(-1, VOCAB_SIZE),
                target_ids.reshape(-1),
            )

            # Backward pass.
            optimizer.zero_grad()
            loss.backward()

            # Prevent unusually large gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            # Update model parameters.
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            global_step += 1

            if (batch_idx + 1) % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]

                print(
                    f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                    f"Step {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {current_lr:.6f}"
                )

        train_loss = total_train_loss / len(train_loader)

        # Evaluate after each epoch.
        val_loss = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
        )

        current_lr = scheduler.get_last_lr()[0]

        print()
        print(f"Epoch {epoch + 1} complete")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss:   {val_loss:.4f}")
        print(f"LR:         {current_lr:.6f}")

        # Save latest checkpoint.
        save_checkpoint(
            LATEST_CHECKPOINT_PATH,
            model,
            optimizer,
            scheduler,
            epoch + 1,
            global_step,
            train_loss,
            val_loss,
        )

        # Save best validation checkpoint.
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                BEST_CHECKPOINT_PATH,
                model,
                optimizer,
                scheduler,
                epoch + 1,
                global_step,
                train_loss,
                val_loss,
            )

            print("Saved new best checkpoint.")

        print()

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Latest checkpoint: {LATEST_CHECKPOINT_PATH}")
    print(f"Best checkpoint:    {BEST_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()