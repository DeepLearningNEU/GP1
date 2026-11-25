"""Train the CLIP -> GPT-2 captioner (prefix or cross-attention injection) on pre-computed CLIP features.

Example:
  python scripts/train_caption.py --data ./data --injection prefix --epochs 5
Outputs in results/m2_caption/<injection>/:
  train_log.csv          step-level training loss / learning rate / time
  epoch_log.csv          per-epoch train loss, val loss, val perplexity
  samples_by_stage.jsonl greedy captions for fixed val images before training, early in training, after each epoch
  config.json
Checkpoint: checkpoints/m2_<injection>.pt
"""
import argparse, csv, json, math, random, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast, get_linear_schedule_with_warmup

from caption_model import ClipCaptioner, GPT2_NAME
from decoding import greedy

ROOT = Path(__file__).resolve().parents[1]


def device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class CaptionPairs(Dataset):
    """One item per (image, caption) pair; every COCO caption of every image is used."""

    def __init__(self, split_dir, injection, max_caps=None):
        self.rows = [json.loads(l) for l in open(split_dir / "captions.jsonl")]
        self.cls = torch.nn.functional.normalize(torch.load(split_dir / "clip_cls.pt")["emb"], dim=-1)
        self.patches = np.load(split_dir / "clip_patches.npy", mmap_mode="r") if injection == "cross" else None
        self.items = [(i, c) for i, r in enumerate(self.rows) for c in r["captions"][:max_caps]]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, cap = self.items[k]
        out = {"img": i, "cls": self.cls[i], "caption": cap}
        if self.patches is not None:
            out["patches"] = torch.from_numpy(np.array(self.patches[i], dtype=np.float32))
        return out


def make_collate(tok, max_tokens=40):
    """Pads every batch to the same length (max_tokens + BOS + EOS): fixed shapes keep Apple-MPS
    from compiling a new GPU graph for every batch length (CUDA does not care)."""
    eos = tok.eos_token_id

    def collate(batch):
        seqs = [[eos] + tok(b["caption"])["input_ids"][:max_tokens] + [eos] for b in batch]
        L = max_tokens + 2
        ids = torch.full((len(seqs), L), eos)
        mask = torch.zeros(len(seqs), L, dtype=torch.long)
        labels = torch.full((len(seqs), L), -100)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s)
            mask[j, :len(s)] = 1
            labels[j, 1:len(s)] = torch.tensor(s[1:])     # caption tokens + EOS; BOS and padding ignored
        out = {"input_ids": ids, "attention_mask": mask, "labels": labels,
               "cls": torch.stack([b["cls"] for b in batch]), "img": [b["img"] for b in batch]}
        if "patches" in batch[0]:
            out["patches"] = torch.stack([b["patches"] for b in batch])
        return out
    return collate


def get_cond(model, batch, dev):
    if model.injection == "prefix":
        return model.condition(clip_emb=batch["cls"].to(dev))
    return model.condition(patches=batch["patches"].to(dev))


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    tot, n = 0.0, 0
    for b in loader:
        out = model(get_cond(model, b, dev), b["input_ids"].to(dev), b["attention_mask"].to(dev), b["labels"].to(dev))
        k = (b["labels"][:, 1:] != -100).sum().item()
        tot, n = tot + out.loss.item() * k, n + k
    model.train()
    return tot / n


@torch.no_grad()
def sample_captions(model, ds, idx, tok, dev, stage, path):
    model.eval()
    b = make_collate(tok)([{"img": i, "cls": ds.cls[i], "caption": "",
                            **({"patches": torch.from_numpy(np.array(ds.patches[i], dtype=np.float32))}
                               if ds.patches is not None else {})} for i in idx])
    caps = [tok.decode(c).strip() for c in greedy(model, get_cond(model, b, dev))]
    with open(path, "a") as f:
        for i, c in zip(idx, caps):
            f.write(json.dumps({"stage": stage, "file_name": ds.rows[i]["file_name"], "generated": c,
                                "reference": ds.rows[i]["captions"][0]}) + "\n")
    model.train()
    return caps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--injection", choices=["prefix", "cross"], default="prefix")
    p.add_argument("--prefix_len", type=int, default=10)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr_new", type=float, default=2e-4, help="mapping network / projection / cross-attention")
    p.add_argument("--lr_gpt", type=float, default=5e-5, help="pre-trained GPT-2 weights")
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--early_sample_steps", type=int, nargs="*", default=[100, 300])
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=0, help="stop early (debug)")
    p.add_argument("--workers", type=int, default=4 if torch.cuda.is_available() else 0)
    p.add_argument("--finetune", choices=["full", "lora", "frozen"], default="full", help="how the GPT-2 weights are trained")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lr_lora", type=float, default=2e-4)
    p.add_argument("--results", default="m2_caption", help="results/<results>/<tag>/")
    p.add_argument("--tag", default=None, help="output sub-folder / checkpoint name (default: injection)")
    p.add_argument("--ckpt_dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--test_images", type=int, default=0,
                   help="hold out the first N val images as a test set: excluded from val loss / checkpoint selection")
    p.add_argument("--save_best", action="store_true", help="keep the checkpoint with the lowest val loss")
    a = p.parse_args()

    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = device()
    tag = a.tag or a.injection
    out = ROOT / "results" / a.results / tag
    out.mkdir(parents=True, exist_ok=True)
    a.ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = a.ckpt_dir / (f"m2_{a.injection}.pt" if a.results == "m2_caption" and tag == a.injection else f"{a.results}_{tag}.pt")
    for f in ["train_log.csv", "epoch_log.csv", "samples_by_stage.jsonl"]:
        (out / f).unlink(missing_ok=True)

    tok = GPT2TokenizerFast.from_pretrained(GPT2_NAME)
    tr, va = CaptionPairs(a.data / "train", a.injection), CaptionPairs(a.data / "val", a.injection)
    collate = make_collate(tok)
    tl = DataLoader(tr, batch_size=a.bs, shuffle=True, collate_fn=collate, num_workers=a.workers)
    vl = DataLoader(va, batch_size=64, shuffle=False, collate_fn=collate, num_workers=a.workers)
    if a.test_images:   # test images are never used for val loss / model selection
        va.items = [(i, c) for i, c in va.items if i >= a.test_images]
        vl = DataLoader(va, batch_size=64, shuffle=False, collate_fn=collate, num_workers=a.workers)
    pool = range(a.test_images) if a.test_images else range(len(va.rows))
    sample_idx = random.Random(0).sample(pool, a.n_samples)

    model = ClipCaptioner(a.injection, prefix_len=a.prefix_len, finetune=a.finetune, lora_r=a.lora_r).to(dev)
    named = [(n, p_) for n, p_ in model.named_parameters() if p_.requires_grad]
    new = [p_ for n, p_ in named if n.startswith("mapper") or "crossattention" in n or "ln_cross_attn" in n]
    lora = [p_ for n, p_ in named if "lora_" in n]
    old = [p_ for n, p_ in named if not (n.startswith("mapper") or "crossattention" in n or "ln_cross_attn" in n or "lora_" in n)]
    groups = [{"params": new, "lr": a.lr_new}, {"params": old, "lr": a.lr_gpt}, {"params": lora, "lr": a.lr_lora}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.01)
    total = a.epochs * len(tl) if not a.max_steps else a.max_steps
    sched = get_linear_schedule_with_warmup(opt, a.warmup, total)

    n_new, n_all = sum(x.numel() for x in new), sum(x.numel() for x in model.parameters())
    n_train = sum(p_.numel() for _, p_ in named)
    cfg = vars(a) | {"data": str(a.data), "ckpt_dir": str(a.ckpt_dir), "tag": tag, "checkpoint": str(ckpt), "device": dev,
                     "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev, "train_pairs": len(tr), "train_images": len(tr.rows),
                     "val_pairs": len(va), "val_images": len(va.rows), "steps": total,
                     "params_total_M": round(n_all / 1e6, 1), "params_new_M": round(n_new / 1e6, 1), "params_trainable_M": round(n_train / 1e6, 2),
                     "clip": "openai/clip-vit-large-patch14 (frozen)", "decoder": GPT2_NAME}
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    print(json.dumps(cfg, indent=1))

    samples = out / "samples_by_stage.jsonl"
    v0 = evaluate(model, vl, dev)
    print(f"[epoch 0] untrained val loss {v0:.4f} ppl {math.exp(v0):.1f}")
    for c in sample_captions(model, va, sample_idx, tok, dev, "untrained", samples)[:3]:
        print("   ", c)
    with open(out / "epoch_log.csv", "w", newline="") as f:
        csv.writer(f).writerows([["epoch", "step", "train_loss", "val_loss", "val_ppl", "minutes"], [0, 0, "", round(v0, 4), round(math.exp(v0), 2), 0]])
    tw = csv.writer(open(out / "train_log.csv", "w", newline=""))
    tw.writerow(["step", "epoch", "train_loss", "lr_new", "lr_other", "seconds"])   # lr_other: GPT-2 (full) or LoRA

    t0, step, run_loss, run_n = time.time(), 0, 0.0, 0
    best, best_ep = float("inf"), 0
    model.train()
    for ep in range(1, a.epochs + 1):
        ep_loss, ep_n = 0.0, 0
        for b in tl:
            loss = model(get_cond(model, b, dev), b["input_ids"].to(dev), b["attention_mask"].to(dev), b["labels"].to(dev)).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            run_loss += loss.item(); run_n += 1; ep_loss += loss.item(); ep_n += 1
            if step % a.log_every == 0:
                lrs = sched.get_last_lr()
                tw.writerow([step, ep, round(run_loss / run_n, 4), f"{lrs[0]:.2e}", f"{lrs[-1]:.2e}", round(time.time() - t0, 1)])
                print(f"step {step:5d}/{total} | epoch {ep} | loss {run_loss / run_n:.4f} | {time.time() - t0:6.0f}s", flush=True)
                run_loss, run_n = 0.0, 0
            if step in a.early_sample_steps:
                caps = sample_captions(model, va, sample_idx, tok, dev, f"step {step}", samples)
                print(f"   [samples @ step {step}]", caps[0], "|", caps[1])
            if a.max_steps and step >= a.max_steps:
                break
        vloss = evaluate(model, vl, dev)
        mins = (time.time() - t0) / 60
        with open(out / "epoch_log.csv", "a", newline="") as f:
            csv.writer(f).writerow([ep, step, round(ep_loss / ep_n, 4), round(vloss, 4), round(math.exp(vloss), 2), round(mins, 1)])
        caps = sample_captions(model, va, sample_idx, tok, dev, f"epoch {ep}", samples)
        print(f"[epoch {ep}] train {ep_loss / ep_n:.4f} | val {vloss:.4f} (ppl {math.exp(vloss):.1f}) | {mins:.1f} min")
        for c in caps[:3]:
            print("   ", c)
        if a.save_best and vloss < best:
            best, best_ep = vloss, ep
            torch.save({"state_dict": model.state_dict(), "config": cfg | {"epoch": ep, "val_loss": vloss}}, ckpt)
            print(f"   new best val loss -> saved {ckpt}")
        if a.max_steps and step >= a.max_steps:
            break

    if not a.save_best:
        best_ep = ep
        torch.save({"state_dict": model.state_dict(), "config": cfg | {"epoch": ep}}, ckpt)
    json.dump(cfg | {"best_epoch": best_ep, "best_val_loss": round(best, 4) if a.save_best else None,
                     "train_minutes": round((time.time() - t0) / 60, 1)}, open(out / "config.json", "w"), indent=2)
    print(f"checkpoint {ckpt} (epoch {best_ep})")


if __name__ == "__main__":
    main()
