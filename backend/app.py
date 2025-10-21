from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import Dict, List, Optional
import os
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
import tiktoken
# import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")
VOCAB = enc.n_vocab


def encode(text: str) -> List[int]:
    return enc.encode(text)


def decode(ids: List[int]) -> str:
    return enc.decode(ids)


def _corpus_path() -> Path:
    override = os.environ.get("BIGRAM_CORPUS_PATH")
    return Path(override) if override else Path(__file__).resolve().parent.parent / "words.txt"


def _load_corpus_bigram_counts() -> Dict[int, Dict[int, int]]:
    with open('words.txt', 'r') as f:
        content = f.read()
        counts = defaultdict(lambda: defaultdict(int))
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            ids = encode(line)
            for a, b in zip(ids[:-1], ids[1:]):
                counts[a][b] += 1
    return {prev: dict(next_counts) for prev, next_counts in counts.items()}


CORPUS_BIGRAM_COUNTS: Dict[int, Dict[int, int]] = _load_corpus_bigram_counts()


MODELS: Dict[str, HookedTransformer] = {
    "t1": HookedTransformer.from_pretrained("attn-only-1l", device=DEVICE).to(DEVICE).eval(),
    "t2": HookedTransformer.from_pretrained("attn-only-2l", device=DEVICE).to(DEVICE).eval(),
}


def make_attn_only(
    vocab_size: int,
    n_layers: int,
    d_model: int = 256,
    n_heads: int = 4,
    n_ctx: int = 256,
) -> HookedTransformer:
    from transformer_lens import HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_head=d_model // n_heads,
        n_ctx=n_ctx,
        d_vocab=vocab_size,
        attn_only=True,
        use_attn_result=True,
        device=DEVICE,
    )
    return HookedTransformer(cfg).to(DEVICE)


@torch.no_grad()
def logits_zero_attn_path(model: HookedTransformer, tokens: torch.Tensor) -> torch.Tensor:
    E = model.W_E
    P = model.W_pos
    positions = torch.arange(tokens.size(1), device=tokens.device)
    x = E[tokens] + P[positions]
    x_last = x[:, -1, :]
    x_last = model.ln_final(x_last)
    logits = x_last @ model.W_U
    return logits


def _resolve_top_k(top_k: int, vocab_size: int) -> int:
    if top_k <= 0 or top_k > vocab_size:
        return vocab_size
    return top_k


def _build_bigram_logit_map(ids: List[int], vocab: int, laplace: float = 1.0) -> Dict[int, Optional[torch.Tensor]]:
    logit_map: Dict[int, Optional[torch.Tensor]] = {}
    if CORPUS_BIGRAM_COUNTS:
        unique_prev = {int(pid) for pid in ids[:-1]}
        print('unique_prev', unique_prev)
        for prev_id in unique_prev:
            next_counts = CORPUS_BIGRAM_COUNTS.get(prev_id)
            if not next_counts:
                logit_map[prev_id] = None  # Explicitly mark as missing
                continue
            vec = torch.full((vocab,), laplace, dtype=torch.float32, device=DEVICE)
            for next_id, count in next_counts.items():
                vec[next_id] += float(count)
            probs = vec / vec.sum()
            logit_map[prev_id] = torch.log(probs)
        # Ensure all unique_prev tokens are in the map (either with logits or None)
        for prev_id in unique_prev:
            if prev_id not in logit_map:
                logit_map[prev_id] = None
    return logit_map


@torch.no_grad()
def _run_with_cache(model: HookedTransformer, toks: torch.Tensor):
    # Capture both attention patterns and value vectors
    pattern_names = [f"blocks.{layer}.attn.hook_pattern" for layer in range(model.cfg.n_layers)]
    value_names = [f"blocks.{layer}.attn.hook_v" for layer in range(model.cfg.n_layers)]
    names = pattern_names + value_names
    
    logits, cache = model.run_with_cache(
        toks,
        names_filter=lambda n: n in names,
        return_type="logits",
    )
    return logits.squeeze(0), cache


@torch.no_grad()
def _calculate_value_weighted_attention(cache: Dict[str, torch.Tensor], n_layers: int, t: int) -> List[List[List[float]]]:
    """
    Calculate value-weighted attention patterns.
    For each layer and head, compute attention weights scaled by the norm of value vectors.
    """
    value_weighted_attn = []
    
    for layer in range(n_layers):
        layer_patterns = []
        
        # Get attention patterns for this layer
        pattern_key = f"blocks.{layer}.attn.hook_pattern"
        value_key = f"blocks.{layer}.attn.hook_v"
        
        if pattern_key not in cache or value_key not in cache:
            continue
            
        attn_pattern = cache[pattern_key][0, :, t-1, :t]  # [n_heads, t]
        value_vectors = cache[value_key][0, :, :t, :]     # [n_heads, t, d_head]
        
        n_heads = attn_pattern.size(0)
        
        for head in range(n_heads):
            # Get attention weights for this head
            attn_weights = attn_pattern[head]  # [t]
            
            # Get value vectors for this head
            head_values = value_vectors[head]  # [t, d_head]
            
            # Calculate norms of value vectors
            value_norms = torch.norm(head_values, dim=-1)  # [t]
            
            # Calculate value-weighted attention
            value_weighted = attn_weights * value_norms  # [t]
            
            layer_patterns.append(value_weighted.detach().cpu().tolist())
        
        value_weighted_attn.append(layer_patterns)
    
    return value_weighted_attn


def _skip_trigram_positions(ids: List[int]) -> List[int]:
    positions: List[int] = []
    for t in range(2, len(ids)):
        a, b = ids[t - 2], ids[t - 1]
        for idx in range(0, t - 1):
            if ids[idx] == a and idx + 1 < t - 1 and ids[idx + 1] == b:
                positions.append(t - 1)
                break
    return sorted(set(positions))


def _topk_pack(vec: torch.Tensor, topk: int) -> List[Dict[str, float]]:
    probs = F.softmax(vec, dim=-1)
    k = min(topk, vec.size(-1))
    tv, ti = torch.topk(vec, k)
    return [
        {
            "token": decode([int(i)]),
            "id": int(i),
            "logit": float(val),
            "prob": float(probs[int(i)]),
        }
        for val, i in zip(tv.tolist(), ti.tolist())
    ]


def analyze_text(text: str, *, top_k: int = 10, models: Optional[Dict[str, HookedTransformer]] = None):
    if models is None:
        models = MODELS
    # load text from hp.txt
    # with open('hp.txt', 'r') as f:
    #     text = f.read()
    ids = encode(text)
    if len(ids) < 2:
        raise ValueError("Need at least two tokens")
    toks = torch.tensor(ids, device=DEVICE, dtype=torch.long).unsqueeze(0)
    T = toks.size(1)

    bigram_logit_map = _build_bigram_logit_map(ids, VOCAB)

    logits1_full, cache1 = _run_with_cache(models["t1"], toks)
    logits2_full, cache2 = _run_with_cache(models["t2"], toks)

    skip_positions = set(_skip_trigram_positions(ids))

    tokens_info = [{"id": int(i), "text": decode([int(i)])} for i in ids]
    positions = []

    for t in range(1, T):
        prev_id = ids[t - 1]
        next_id = ids[t]

        bigram_logits = bigram_logit_map.get(prev_id)
        bigram_available = bigram_logits is not None
        l1 = logits1_full[t - 1]
        l2 = logits2_full[t - 1]

        attn1 = [[[] for _ in range(models["t1"].cfg.n_heads)] for _ in range(models["t1"].cfg.n_layers)]
        attn2 = [[[] for _ in range(models["t2"].cfg.n_heads)] for _ in range(models["t2"].cfg.n_layers)]

        for layer in range(models["t1"].cfg.n_layers):
            for head in range(models["t1"].cfg.n_heads):
                attn1[layer][head] = cache1[f"blocks.{layer}.attn.hook_pattern"][0, head, t, :].detach().cpu().tolist()
        for layer in range(models["t2"].cfg.n_layers):
            for head in range(models["t2"].cfg.n_heads):
                attn2[layer][head] = cache2[f"blocks.{layer}.attn.hook_pattern"][0, head, t, :].detach().cpu().tolist()


        # Calculate value-weighted attention patterns
        # value_weighted_attn1 = _calculate_value_weighted_attention(cache1, models["t1"].cfg.n_layers, t)
        # value_weighted_attn2 = _calculate_value_weighted_attention(cache2, models["t2"].cfg.n_layers, t)

        match_index = None
        for idx in range(t - 1, -1, -1):
            if ids[idx] == next_id:
                match_index = idx
                break

        attn_match = None
        if match_index is not None:
            col = match_index
            sum_t1 = float(sum(layer[:, col].sum() for layer in [
                torch.tensor(layer_heads) for layer_heads in attn1
            ]))
            sum_t2 = float(sum(layer[:, col].sum() for layer in [
                torch.tensor(layer_heads) for layer_heads in attn2
            ]))
            attn_match = {"t1": sum_t1, "t2": sum_t2}

        # Calculate losses - handle missing bigram data
        loss_bigram = float(-bigram_logits[next_id].item()) if bigram_available else None
        loss_t1 = float(-F.log_softmax(l1, dim=-1)[next_id].item())
        loss_t2 = float(-F.log_softmax(l2, dim=-1)[next_id].item())

        # Prepare topk data - handle missing bigram data
        topk_data = {
            "t1": _topk_pack(l1, top_k),
            "t2": _topk_pack(l2, top_k),
        }
        if bigram_available:
            topk_data["bigram"] = _topk_pack(bigram_logits, top_k)
        else:
            topk_data["bigram"] = None

        positions.append(
            {
                "t": t,
                "context_token": tokens_info[t - 1],
                "next_token": tokens_info[t],
                "topk": topk_data,
                "attn": {
                    "t1": attn1,
                    "t2": attn2,
                },
                "value_weighted_attn": {
                    # "t1": value_weighted_attn1,
                    # "t2": value_weighted_attn2,
                },
                "losses": {
                    "bigram": loss_bigram,
                    "t1": loss_t1,
                    "t2": loss_t2,
                },
                "bigram_available": bigram_available,
                "match_index": match_index,
                "match_attention": attn_match,
                "skip_trigram": t - 1 in skip_positions,
            }
        )

    print('config', models["t1"].cfg);

    return {
        "tokens": tokens_info,
        "positions": positions,
        "device": DEVICE,
        "t1_layers": models["t1"].cfg.n_layers,
        "t1_heads": models["t1"].cfg.n_heads,
        "t2_layers": models["t2"].cfg.n_layers,
        "t2_heads": models["t2"].cfg.n_heads,
    }


class AnalyzeReq(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    top_k: int = 10


class AnalyzeResp(BaseModel):
    tokens: List[Dict[str, object]]
    positions: List[Dict[str, object]]
    device: str
    t1_layers: int
    t1_heads: int
    t2_layers: int
    t2_heads: int


class AblateHeadReq(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    position: int
    model_name: str  # "t1" or "t2"
    layer: int
    head: int
    top_k: int = 10


class AblateHeadResp(BaseModel):
    with_head: List[Dict[str, object]]
    without_head: List[Dict[str, object]]
    delta_positive: List[Dict[str, object]]
    delta_negative: List[Dict[str, object]]


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@torch.no_grad()
def ablate_head_analysis(
    text: str,
    position: int,
    model_name: str,
    layer: int,
    head: int,
    top_k: int = 10,
    models: Optional[Dict[str, HookedTransformer]] = None,
):
    if models is None:
        models = MODELS

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    model = models[model_name]
    ids = encode(text)

    if position < 0 or position >= len(ids):
        raise ValueError(f"Position {position} out of range for text with {len(ids)} tokens")

    if layer < 0 or layer >= model.cfg.n_layers:
        raise ValueError(f"Layer {layer} out of range for model with {model.cfg.n_layers} layers")

    if head < 0 or head >= model.cfg.n_heads:
        raise ValueError(f"Head {head} out of range for model with {model.cfg.n_heads} heads")

    toks = torch.tensor(ids, device=DEVICE, dtype=torch.long).unsqueeze(0)

    # 1) Baseline logits at the position of interest
    logits_base = model(toks)[0, position]  # [vocab]

    # 2) Re-run with the chosen head zeroed
    def zero_one_head_hook(v, hook):
        # v: [batch, seq, n_heads, d_head] *before* W_O ("result" in TLens)
        v[:, :, head, :] = 0.0
        return v

    logits_no_head = model.run_with_hooks(
        toks,
        fwd_hooks=[(f"blocks.{layer}.attn.hook_result", zero_one_head_hook)]
    )[0, position]  # [vocab]

    delta_logits = logits_base - logits_no_head  # the head's exact contribution to each vocab logit

    # Get top-k results
    topk_with = _topk_pack(logits_base, top_k)
    topk_without = _topk_pack(logits_no_head, top_k)
    topk_delta = _topk_pack(delta_logits, top_k)      # tokens most *helped* by the head
    topk_delta_neg = _topk_pack(-delta_logits, top_k) # tokens most *hurt* by the head

    return {
        "with_head": topk_with,
        "without_head": topk_without,
        "delta_positive": topk_delta,
        "delta_negative": topk_delta_neg,
    }


@app.post("/api/analyze", response_model=AnalyzeResp)
def analyze(req: AnalyzeReq):
    try:
        return analyze_text(req.text, top_k=req.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/ablate-head", response_model=AblateHeadResp)
def ablate_head(req: AblateHeadReq):
    try:
        return ablate_head_analysis(
            text=req.text,
            position=req.position,
            model_name=req.model_name,
            layer=req.layer,
            head=req.head,
            top_k=req.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
