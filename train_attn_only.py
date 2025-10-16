# train_attn_only.py
import math, os, json, argparse, random, torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformer_lens import HookedTransformer, HookedTransformerConfig
import tiktoken
from contextlib import nullcontext
from time import time

# ---------- args ----------
p = argparse.ArgumentParser()
p.add_argument("--corpus_path", type=str, required=True, help="Text file (UTF-8). Use a few hundred MB for best results.")
p.add_argument("--out_dir", type=str, default="checkpoints")
p.add_argument("--ctx", type=int, default=256)
p.add_argument("--d_model", type=int, default=256)
p.add_argument("--n_heads", type=int, default=4)
p.add_argument("--steps_1layer", type=int, default=20000)
p.add_argument("--steps_2layer", type=int, default=40000)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--grad_accum", type=int, default=2)
p.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
p.add_argument("--seed", type=int, default=1337)
p.add_argument("--eval_every", type=int, default=1000)
p.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints and train models from scratch.")
p.add_argument("--bigram_dim", type=int, default=128)
p.add_argument("--bigram_steps", type=int, default=2000)
p.add_argument("--bigram_batch", type=int, default=4096)
p.add_argument("--bigram_lr", type=float, default=1e-2)
p.add_argument("--bigram_eval_batches", type=int, default=200)
p.add_argument("--bigram_eps", type=float, default=1e-3)
args = p.parse_args()

random.seed(args.seed); torch.manual_seed(args.seed)

enc = tiktoken.get_encoding("gpt2")
VOCAB = enc.n_vocab

# ---------- data ----------
print("loading corpus…")
text = Path(args.corpus_path).read_text(encoding="utf-8")
ids = enc.encode(text)  # GPT-2 BPE ids
tokens = torch.tensor(ids, dtype=torch.long, device=args.device)

# make a small held-out slice for eval
split = int(tokens.numel() * 0.98)
train_tokens = tokens[:split]
val_tokens   = tokens[split:]

def get_batch_from(arr, T=args.ctx, B=args.batch_size):
    ix = torch.randint(0, arr.numel() - T - 1, (B,), device=args.device)
    x = torch.stack([arr[i:i+T] for i in ix], dim=0)
    y = torch.stack([arr[i+1:i+T+1] for i in ix], dim=0)
    return x, y

def get_bigram_batch(arr, B=None):
    B = B or args.bigram_batch
    ix = torch.randint(0, arr.numel() - 1, (B,), device=args.device)
    x = arr[ix]
    y = arr[ix + 1]
    return x, y

@torch.no_grad()
def estimate_loss(model, iters=200):
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x,y = get_batch_from(val_tokens)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        tot += loss.item()
    model.train()
    return tot / iters

@torch.no_grad()
def estimate_bigram_loss(model, arr, batches):
    was_training = model.training
    model.eval()
    tot = 0.0
    for _ in range(batches):
        x, y = get_bigram_batch(arr)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        tot += loss.item()
    if was_training:
        model.train()
    return tot / max(batches, 1)


def make_bigram_model(dim: int):
    model = nn.Sequential(
        nn.Embedding(VOCAB, dim),
        nn.Linear(dim, VOCAB, bias=False),
    )
    return model.to(args.device)

def make_attn_only(n_layers: int):
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_head=args.d_model // args.n_heads,
        n_ctx=args.ctx,
        d_vocab=VOCAB,
        attn_only=True,
        use_attn_result=True,
        device=args.device,
    )
    return HookedTransformer(cfg).to(args.device)

# ---------- train loop ----------
def train_model(n_layers, steps, save_name, ckpt_every=2000):
    model = make_attn_only(n_layers)
    ckpt_path = Path(args.out_dir) / f"{save_name}.pt"
    start_step = 0

    if not args.fresh and ckpt_path.exists():
        data = torch.load(ckpt_path, map_location=args.device)
        state_dict = data.get("state_dict", data)
        try:
            model.load_state_dict(state_dict)
            start_step = int(data.get("trained_steps", 0))
            print(
                f"[{save_name}] loaded checkpoint from {ckpt_path} (trained_steps {start_step})",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[{save_name}] warning: failed to load checkpoint {ckpt_path}: {exc}. Training from scratch.",
                flush=True,
            )
            start_step = 0

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda") if args.device.startswith("cuda") else None
    autocast_ctx = torch.amp.autocast("cuda") if args.device.startswith("cuda") else nullcontext

    model.train()
    ema = None
    ema_beta = 0.98
    t0 = time()

    # optional initial validation snapshot for early overfit detection
    if args.eval_every:
        init_loss = estimate_loss(model, iters=100)
        init_ppl = math.exp(min(init_loss, 20))
        print(
            f"[{save_name}] step {start_step:>6} | eval_loss {init_loss:.3f} (ppl {init_ppl:.1f})",
            flush=True,
        )

    if start_step >= steps:
        print(
            f"[{save_name}] checkpoint already reached target steps ({start_step} >= {steps}); skipping training.",
            flush=True,
        )
        return ckpt_path

    step = start_step

    try:
        for step in range(start_step + 1, steps + 1):
            opt.zero_grad(set_to_none=True)
            with autocast_ctx():
                loss_accum = 0.0
                for _ in range(args.grad_accum):
                    x, y = get_batch_from(train_tokens)
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1)
                    ) / args.grad_accum
                    loss_accum += loss.item()
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

            # update train EMA and emit logging
            ema = loss_accum if ema is None else (ema_beta * ema + (1 - ema_beta) * loss_accum)
            elapsed = time() - t0
            progress = step - start_step
            it_per_s = progress / max(elapsed, 1e-6)
            eta_s = (steps - step) / max(it_per_s, 1e-6)
            ppl = math.exp(min(ema, 20))  # guard overflow

            msg = (
                f"[{save_name}] step {step:>6} "
                f"train_loss {loss_accum:.3f} ema {ema:.3f} (ppl {ppl:.1f}) "
                f"ETA {eta_s/60:.1f}m"
            )

            do_eval = args.eval_every and (step % args.eval_every == 0 or step == steps)
            if do_eval:
                eval_loss = estimate_loss(model, iters=100)
                eval_ppl = math.exp(min(eval_loss, 20))
                msg += f" | eval_loss {eval_loss:.3f} (ppl {eval_ppl:.1f})"

            print(msg, flush=True)

            # periodic checkpoint (optional)
            if ckpt_every and step % ckpt_every == 0:
                out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
                ckpt = out_dir / f"{save_name}_step{step}.pt"
                torch.save(
                    {
                        "config": model.cfg.to_dict(),
                        "state_dict": model.state_dict(),
                        "trained_steps": int(step),
                    },
                    ckpt,
                )
    except KeyboardInterrupt:
        eval_loss = estimate_loss(model, iters=100)
        eval_ppl = math.exp(min(eval_loss, 20))
        print(
            f"\n[{save_name}] interrupted at step {step:>6} | eval_loss {eval_loss:.3f} (ppl {eval_ppl:.1f})",
            flush=True,
        )
        print("[interrupt] saving partial checkpoint…")
    finally:
        out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = out_dir / f"{save_name}.pt"
        torch.save(
            {
                "config": model.cfg.to_dict(),
                "state_dict": model.state_dict(),
                "trained_steps": int(step),
            },
            ckpt,
        )
        print(f"saved {ckpt}")
    return ckpt

# ---------- bigram baseline ----------
def train_bigram(path: Path):
    start_step = 0
    dim = args.bigram_dim
    model = make_bigram_model(dim)

    if not args.fresh and path.exists():
        data = torch.load(path, map_location=args.device)
        saved_cfg = data.get("config", {})
        saved_dim = int(saved_cfg.get("embedding_dim", dim))
        if saved_dim != dim:
            print(
                f"[bigram] overriding requested dim {dim} with saved dim {saved_dim} to resume.",
                flush=True,
            )
        dim = saved_dim
        model = make_bigram_model(dim)
        try:
            model.load_state_dict(data["state_dict"])
            start_step = int(data.get("metrics", {}).get("trained_steps", 0))
            print(
                f"[bigram] loaded checkpoint from {path} (trained_steps {start_step})",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[bigram] warning: failed to load {path}: {exc}. Retraining.", flush=True)
            model = make_bigram_model(args.bigram_dim)
            start_step = 0

    opt = torch.optim.AdamW(model.parameters(), lr=args.bigram_lr, betas=(0.9, 0.99), weight_decay=0.0)
    ema = None
    ema_beta = 0.98
    final_step = start_step
    total_steps = start_step + max(args.bigram_steps, 0)

    if args.bigram_steps > 0:
        for step in range(start_step + 1, total_steps + 1):
            opt.zero_grad(set_to_none=True)
            x, y = get_bigram_batch(train_tokens)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=args.bigram_eps)
            loss.backward()
            opt.step()

            ema = loss.item() if ema is None else ema_beta * ema + (1 - ema_beta) * loss.item()

            if (
                step == start_step + 1
                or step % max(args.eval_every, 1) == 0
                or step == total_steps
            ):
                train_loss = estimate_bigram_loss(model, train_tokens, batches=args.bigram_eval_batches)
                val_loss = estimate_bigram_loss(model, val_tokens, batches=args.bigram_eval_batches)
                print(
                    f"[bigram] step {step:>6} train_loss {train_loss:.3f} "
                    f"val_loss {val_loss:.3f} ema {ema:.3f}",
                    flush=True,
                )
            final_step = step

    # always report final metrics
    train_loss = estimate_bigram_loss(model, train_tokens, batches=args.bigram_eval_batches)
    val_loss = estimate_bigram_loss(model, val_tokens, batches=args.bigram_eval_batches)

    torch.save(
        {
            "config": {
                "vocab": VOCAB,
                "embedding_dim": dim,
                "device": args.device,
                "label_smoothing": args.bigram_eps,
            },
            "state_dict": model.state_dict(),
            "metrics": {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "trained_steps": int(final_step),
            },
        },
        path,
    )
    print(
        f"saved bigram to {path} (train_loss {train_loss:.3f}, val_loss {val_loss:.3f})",
        flush=True,
    )

out = Path(args.out_dir); out.mkdir(exist_ok=True, parents=True)
train_bigram(out / "bigram.pt")
ck1 = train_model(1, args.steps_1layer, "attn_only_L1")
ck2 = train_model(2, args.steps_2layer, "attn_only_L2")

# save a small manifest
(Path(args.out_dir) / "manifest.json").write_text(json.dumps({
    "tokenizer": "gpt2-bpe",
    "bigram": "bigram.pt",
    "one_layer": Path(ck1).name,
    "two_layer": Path(ck2).name,
}, indent=2))
