# Model Configuration

VOCAB_SIZE = 8192
D_MODEL = 512
NUM_HEADS = 8
NUM_BLOCKS = 4
INTERMEDIATE_SIZE = 2048
MAX_SEQ_LEN = 512


# Training Configuration

BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 5


# Tokenizer Configuration

TOKENIZER_PATH = "artifacts/tinystories_bpe_8192.json"

# Data Configuration

TRAIN_DATA_PATH = "data/raw/train.jsonl"
VAL_DATA_PATH = "data/raw/val.jsonl"

TRAIN_PROCESSED_PATH = "data/processed/train.pt"
VAL_PROCESSED_PATH = "data/processed/val.pt"