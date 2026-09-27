import torch
from torch.utils.data import Dataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class TinyStoriesDataset(Dataset):
    """
    Dataset for preprocessed TinyStories sequences.

    Each sample contains:
    input_ids  → [T]
    target_ids → [T]
    """

    def __init__(self, data_path: str):
        data_path = PROJECT_ROOT / data_path
        data = torch.load(data_path, map_location="cpu")

        self.input_ids = data["input_ids"]
        self.target_ids = data["target_ids"]

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.input_ids[idx],
            "target_ids": self.target_ids[idx],
        }