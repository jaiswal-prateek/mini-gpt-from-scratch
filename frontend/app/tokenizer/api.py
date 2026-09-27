"""
Unified FastAPI backend for Mini GPT — Tokenizer + LLM Playground.

Serves:
  /api/encode, /api/decode, /api/vocab, /api/merges, /api/info   (tokenizer)
  /api/generate                                                    (LLM)
  /                                                                (frontend)
"""

import sys
import json
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from tokenizer.bpe import ByteLevelBPETokenizer
from model.gpt import GPT
from config import (
    VOCAB_SIZE, D_MODEL, NUM_HEADS, INTERMEDIATE_SIZE,
    NUM_BLOCKS, MAX_SEQ_LEN, TOKENIZER_PATH,
)

app = FastAPI(title="Mini GPT Playground")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Paths ────────────────────────────────────────────────
ARTIFACTS = PROJECT_ROOT / "artifacts"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
STATIC_DIR = Path(__file__).parent.parent / "static"

TOKENIZER_FILES = {
    "tinystories_8192": ARTIFACTS / "tinystories_bpe_8192.json",
    "tinystories_8192_dev": ARTIFACTS / "tinystories_bpe_8192_dev.json",
}

# ── State ────────────────────────────────────────────────
tokenizers: dict[str, ByteLevelBPETokenizer] = {}
model = None
device = None


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_tokenizer(name: str) -> ByteLevelBPETokenizer:
    if name not in tokenizers:
        tok = ByteLevelBPETokenizer()
        tok.load(str(TOKENIZER_FILES[name]))
        tokenizers[name] = tok
    return tokenizers[name]


@app.on_event("startup")
def startup():
    global model, device
    for name in TOKENIZER_FILES:
        get_tokenizer(name)

    device = get_device()
    model = GPT(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE, num_blocks=NUM_BLOCKS,
    ).to(device)
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded on {device}  |  params: {sum(p.numel() for p in model.parameters()):,}")


# ══════════════════════════════════════════════════════════
#  TOKENIZER ENDPOINTS
# ══════════════════════════════════════════════════════════

class EncodeRequest(BaseModel):
    text: str
    tokenizer: str = "tinystories_8192"

class DecodeRequest(BaseModel):
    token_ids: list[int]
    tokenizer: str = "tinystories_8192"


@app.post("/api/encode")
def encode(req: EncodeRequest):
    tok = get_tokenizer(req.tokenizer)
    token_ids = tok.encode(req.text)
    tokens = []
    for tid in token_ids:
        if tid in tok.special_tokens.values():
            text = next(k for k, v in tok.special_tokens.items() if v == tid)
        else:
            text = tok.vocab[tid].decode("utf-8", errors="replace")
        tokens.append({"id": tid, "text": text})
    return {"tokens": tokens, "token_ids": token_ids, "num_tokens": len(token_ids), "num_chars": len(req.text)}


@app.post("/api/decode")
def decode(req: DecodeRequest):
    tok = get_tokenizer(req.tokenizer)
    return {"text": tok.decode(req.token_ids)}


@app.get("/api/vocab")
def vocab(tokenizer: str = "tinystories_8192", search: str = "", offset: int = 0, limit: int = 100):
    tok = get_tokenizer(tokenizer)
    rev = {v: k for k, v in tok.special_tokens.items()}
    items = []
    for tid in sorted(tok.vocab.keys()):
        raw = tok.vocab[tid]
        text = rev.get(tid, raw.decode("utf-8", errors="replace"))
        items.append({
            "id": tid, "text": text,
            "bytes": " ".join(f"{b:02x}" for b in raw),
            "length": len(raw),
            "is_byte_token": tid < 256,
            "is_special": tid in rev,
        })
    for st, sid in tok.special_tokens.items():
        if sid not in tok.vocab:
            items.append({"id": sid, "text": st, "bytes": "", "length": 0, "is_byte_token": False, "is_special": True})
    items.sort(key=lambda x: x["id"])
    if search:
        q = search.lower()
        items = [i for i in items if q in i["text"].lower() or q in str(i["id"])]
    total = len(items)
    return {"items": items[offset:offset+limit], "total": total, "offset": offset, "limit": limit}


@app.get("/api/merges")
def merges(tokenizer: str = "tinystories_8192", offset: int = 0, limit: int = 100):
    tok = get_tokenizer(tokenizer)
    rev = {v: k for k, v in tok.special_tokens.items()}
    def tt(tid):
        if tid in rev: return rev[tid]
        if tid in tok.vocab: return tok.vocab[tid].decode("utf-8", errors="replace")
        return f"[{tid}]"
    items = []
    for pair, rank in sorted(tok.merge_ranks.items(), key=lambda x: x[1]):
        nid = tok.merges[pair]
        items.append({"rank": rank, "pair": list(pair), "pair_text": [tt(pair[0]), tt(pair[1])], "new_id": nid, "new_text": tt(nid)})
    total = len(items)
    return {"items": items[offset:offset+limit], "total": total, "offset": offset, "limit": limit}


@app.get("/api/info")
def info(tokenizer: str = "tinystories_8192"):
    tok = get_tokenizer(tokenizer)
    extra = len([s for s in tok.special_tokens if tok.special_tokens[s] not in tok.vocab])
    return {
        "name": tokenizer,
        "vocab_size": len(tok.vocab) + extra,
        "num_merges": len(tok.merges),
        "num_special_tokens": len(tok.special_tokens),
        "special_tokens": tok.special_tokens,
        "available_tokenizers": list(TOKENIZER_FILES.keys()),
    }


# ══════════════════════════════════════════════════════════
#  LLM GENERATION ENDPOINT
# ══════════════════════════════════════════════════════════

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9


def apply_top_k(logits, top_k):
    if top_k is None or top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    vals, _ = torch.topk(logits, top_k, dim=-1)
    return torch.where(logits < vals[:, -1:], torch.full_like(logits, float("-inf")), logits)


def apply_top_p(logits, top_p):
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(sorted_probs, dim=-1)
    remove = cum > top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    mask = torch.zeros_like(remove, dtype=torch.bool)
    mask.scatter_(1, sorted_idx, remove)
    return logits.masked_fill(mask, float("-inf"))


@torch.no_grad()
def generate_step_by_step(prompt_text, max_new_tokens, temperature, top_k, top_p):
    """
    Yield one dict per generated token with probability info.
    Each dict: {token_text, token_id, candidates: [{text, id, prob, selected}...]}
    """
    tok = get_tokenizer("tinystories_8192")
    token_ids = tok.encode(prompt_text)

    if len(token_ids) >= MAX_SEQ_LEN:
        yield {"error": f"Prompt too long: {len(token_ids)} tokens (max {MAX_SEQ_LEN})"}
        return

    ids_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    logits, past_kv = model(ids_tensor, use_cache=True)
    eos_id = tok.special_tokens.get("<|endoftext|>")
    max_tokens = min(max_new_tokens, MAX_SEQ_LEN - ids_tensor.size(1))

    rev_special = {v: k for k, v in tok.special_tokens.items()}

    def token_text(tid):
        if tid in rev_special:
            return rev_special[tid]
        if tid in tok.vocab:
            return tok.vocab[tid].decode("utf-8", errors="replace")
        return f"[{tid}]"

    for _ in range(max_tokens):
        next_logits = logits[:, -1, :]

        # Raw probabilities before filtering (for display)
        raw_probs = torch.softmax(next_logits / max(temperature, 1e-8), dim=-1).squeeze(0)

        # Apply sampling filters
        filtered = next_logits / max(temperature, 1e-8)
        filtered = apply_top_k(filtered, top_k)
        filtered = apply_top_p(filtered, top_p)
        probs = torch.softmax(filtered, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1)
        selected_id = next_id.item()

        if selected_id == eos_id:
            yield {"done": True, "reason": "eos"}
            return

        # Build candidate list: tokens that survived top-p filtering
        # Use raw_probs for display, but mark which survived filtering
        sorted_raw, sorted_idx = torch.sort(raw_probs, descending=True)
        cum_probs = torch.cumsum(sorted_raw, dim=-1)

        candidates = []
        for i in range(len(sorted_raw)):
            tid = sorted_idx[i].item()
            p = sorted_raw[i].item()
            if p < 0.001 and len(candidates) >= 5:
                break
            # Stop after cumulative > top_p + some margin, but show at least top candidates
            if cum_probs[i].item() > (top_p + 0.05) and len(candidates) >= 3:
                break
            if len(candidates) >= 20:
                break
            survived = probs[0, tid].item() > 0
            candidates.append({
                "id": tid,
                "text": token_text(tid),
                "prob": round(p, 5),
                "cum_prob": round(cum_probs[i].item(), 5),
                "selected": tid == selected_id,
                "in_sample_pool": survived,
            })

        yield {
            "token": token_text(selected_id),
            "token_id": selected_id,
            "prob": round(raw_probs[selected_id].item(), 5),
            "candidates": candidates,
        }

        ids_tensor = next_id
        logits, past_kv = model(next_id, past_key_values=past_kv, use_cache=True)

    yield {"done": True, "reason": "max_tokens"}


@app.post("/api/generate")
async def generate_endpoint(req: GenerateRequest):
    """Stream token-by-token generation as server-sent events."""
    async def event_stream():
        for step in generate_step_by_step(
            req.prompt, req.max_new_tokens, req.temperature, req.top_k, req.top_p
        ):
            yield f"data: {json.dumps(step)}\n\n"
            await asyncio.sleep(0)  # yield control so stream flushes
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/model_info")
def model_info():
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu")
    return {
        "vocab_size": VOCAB_SIZE,
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "num_blocks": NUM_BLOCKS,
        "intermediate_size": INTERMEDIATE_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_loss": round(ckpt.get("train_loss", 0), 4),
        "val_loss": round(ckpt.get("val_loss", 0), 4),
        "epoch": ckpt.get("epoch"),
        "device": str(device),
    }


# ══════════════════════════════════════════════════════════
#  SERVE FRONTEND
# ══════════════════════════════════════════════════════════

@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
