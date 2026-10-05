"""Training loop for the inversion model (stage 1) and the corrector (stage 2).

Usage:
  python -m v2t.train --stage inversion --data data/corpus --out ckpt/inverter --epochs 1
  python -m v2t.train --stage corrector --init ckpt/inverter --data data/corpus --out ckpt/corrector
"""
import argparse, math, os, time, json
import numpy as np
import torch
import torch._dynamo as torch_dynamo
from .model import Vec2TextQwen, get_tokenizer
from .data import Corpus


def length_bucketed_batches(lengths, bs, rng, mega=64):
    """Shuffle, then sort inside mega-chunks so each batch has similar lengths (less padding)."""
    idx = rng.permutation(len(lengths))
    out = []
    for start in range(0, len(idx), bs * mega):
        chunk = idx[start:start + bs * mega]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        bats = [chunk[i:i + bs] for i in range(0, len(chunk), bs)]
        rng.shuffle(bats)
        out += bats
    return out


@torch.no_grad()
def evaluate(model, corpus, bs=128, max_batches=20):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, min(len(corpus), bs * max_batches), bs):
        b = corpus.batch(range(i, min(i + bs, len(corpus))))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, nt = model(b["e_target"], b["target_ids"], b.get("e_hyp"), b.get("hyp_ids"))
        tot += loss.item() * nt
        n += nt
    model.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["inversion", "corrector"], required=True)
    ap.add_argument("--data", default="data/corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None, help="inverter checkpoint to init the corrector from")
    ap.add_argument("--n_prefix", type=int, default=8)
    ap.add_argument("--base", default=None, help="base LM for a new inverter (default: Qwen/Qwen3-0.6B-Base)")
    ap.add_argument("--freeze_embed", action="store_true", help="freeze the (tied) token embedding matrix")
    ap.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw", help="adamw8bit = bitsandbytes 8-bit states for the LM params")
    ap.add_argument("--max_tokens", type=int, default=32, help="max text length in tokens (stored in the checkpoint)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=256, help="samples per optimizer step")
    ap.add_argument("--micro_bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr_proj", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N training samples")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_bs", type=int, default=64, help="validation batch size (logits are fp32: keep small for big models/long texts)")
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from --out (train_state.json, optim.pt) if present")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--pad_to", type=int, default=8, help="with --compile: pad sequence lengths to a multiple of this")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tok = get_tokenizer()
    corr = args.stage == "corrector"
    train = Corpus(args.data, "train", tok, with_hyp=corr, limit=args.limit, max_hyp=args.max_tokens + 32)
    val = Corpus(args.data, "val", tok, with_hyp=corr, max_hyp=args.max_tokens + 32)
    print(f"train={len(train)} val={len(val)}", flush=True)

    state_path = os.path.join(args.out, "train_state.json")
    start_step = 0
    if args.resume and os.path.exists(state_path):
        start_step = json.load(open(state_path))["step"]
        print(f"resuming from {args.out} at step {start_step}", flush=True)
        model = Vec2TextQwen.load(args.out, dtype=torch.float32, device="cpu")
    elif args.init:
        init_cfg = json.load(open(os.path.join(args.init, "v2t_config.json")))
        if corr and not init_cfg["corrector"]:
            model = Vec2TextQwen.from_inverter(args.init)        # new corrector from an inverter
        else:
            assert init_cfg["corrector"] == corr, "checkpoint type does not match --stage"
            model = Vec2TextQwen.load(args.init, dtype=torch.float32, device="cpu")  # warm start, same type
    else:
        assert not corr, "--init required for the corrector"
        model = Vec2TextQwen(n_prefix=args.n_prefix, corrector=False, **({"base": args.base} if args.base else {}))
    model.max_tokens = model.cfg["max_tokens"] = args.max_tokens
    if args.freeze_embed:
        model.lm.get_input_embeddings().weight.requires_grad_(False)
    model.cuda().train()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"base={model.cfg['base']} trainable params={n_train/1e6:.0f}M", flush=True)
    pad_to = 1
    if args.compile:
        torch_dynamo.config.cache_size_limit = 64
        model.lm.model = torch.compile(model.lm.model, dynamic=False)
        pad_to = args.pad_to

    proj_params = [p for n, p in model.named_parameters() if n.startswith("proj_")]
    lm_params = [p for n, p in model.named_parameters() if not n.startswith("proj_") and p.requires_grad]
    groups = [{"params": lm_params, "lr": args.lr}, {"params": proj_params, "lr": args.lr_proj}]
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, betas=(0.9, 0.98), weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), weight_decay=0.01, fused=True)
    optim_path = os.path.join(args.out, "optim.pt")
    if start_step and os.path.exists(optim_path):
        opt.load_state_dict(torch.load(optim_path, map_location="cuda"))
        print("optimizer state restored", flush=True)
    accum = args.bs // args.micro_bs
    lengths = np.array([len(x) + (len(h) if corr else 0) for x, h in zip(train.ids, train.hyp_ids or train.ids)])
    steps_per_epoch = len(train) // args.bs
    total_steps = args.max_steps or int(steps_per_epoch * args.epochs)
    print(f"steps/epoch={steps_per_epoch} total_steps={total_steps} accum={accum}", flush=True)

    def lr_scale(step):
        if step < args.warmup:
            return step / args.warmup
        p = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train_log.jsonl"), "a")
    step, t0, tok_count, run_loss = 0, time.time(), 0, 0.0
    t_win = t0
    done = False
    while not done:
        batches = length_bucketed_batches(lengths, args.bs, rng)
        for bidx in batches:
            if step < start_step:  # replay the same batch order and skip what was already trained
                step += 1
                continue
            s = lr_scale(step)
            opt.param_groups[0]["lr"] = args.lr * s
            opt.param_groups[1]["lr"] = args.lr_proj * s
            loss_acc = 0.0
            for m in range(accum):
                sub = bidx[m * args.micro_bs:(m + 1) * args.micro_bs]
                if len(sub) == 0 or (args.compile and len(sub) < args.micro_bs):
                    continue
                b = train.batch(sub)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, nt = model(b["e_target"], b["target_ids"], b.get("e_hyp"), b.get("hyp_ids"), pad_to=pad_to)
                (loss / accum).backward()
                loss_acc += loss.item() / accum
                tok_count += nt
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run_loss = 0.98 * run_loss + 0.02 * loss_acc if step > 1 else loss_acc
            if step % 20 == 0:
                now = time.time()
                rate = 20 / (now - t_win)
                t_win = now
                print(f"step {step}/{total_steps} loss {loss_acc:.3f} (ema {run_loss:.3f}) lr {args.lr*s:.2e} "
                      f"{rate:.2f} it/s {tok_count/(now-t0):.0f} tgt-tok/s mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB "
                      f"eta {(total_steps-step)/rate/60:.0f} min", flush=True)
            if step % args.eval_every == 0 or step == total_steps:
                vl = evaluate(model, val, bs=args.eval_bs, max_batches=2560 // args.eval_bs)
                print(f"  [eval] step {step} val_loss {vl:.4f}", flush=True)
                log.write(json.dumps({"step": step, "train_ema": run_loss, "val": vl, "time": time.time() - t0}) + "\n")
                log.flush()
            if step % args.save_every == 0 or step == total_steps:
                model.save(args.out)
                torch.save(opt.state_dict(), optim_path)
                json.dump({"step": step}, open(state_path, "w"))
                print(f"  saved -> {args.out}", flush=True)
            if step >= total_steps:
                done = True
                break
    print("done", flush=True)


if __name__ == "__main__":
    main()
