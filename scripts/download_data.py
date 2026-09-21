from pathlib import Path
import json

from datasets import load_dataset

DATASET_NAME = "karpathy/tinystories-gpt4-clean"

TRAIN_START = 20_000
TRAIN_COUNT = 25_000

VAL_START = 10_000
VAL_COUNT = 2_000

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

dataset = load_dataset(DATASET_NAME, split="train")

train_data = dataset.select(
    range(TRAIN_START, TRAIN_START + TRAIN_COUNT)
)

val_data = dataset.select(
    range(VAL_START, VAL_START + VAL_COUNT)
)

with open(RAW_DIR / "train.jsonl", "w", encoding="utf-8") as file:
    for row in train_data:
        file.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")

with open(RAW_DIR / "val.jsonl", "w", encoding="utf-8") as file:
    for row in val_data:
        file.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")

print(f"Saved training stories: {len(train_data)}")
print(f"Saved validation stories: {len(val_data)}")