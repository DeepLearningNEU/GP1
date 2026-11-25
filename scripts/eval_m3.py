"""Milestone 3 evaluation: caption metrics for trained captioners under different decoding settings.

Decoding specs (--strategies):
  greedy            argmax decoding
  beam:K            beam search with K beams (length-normalized)
  sample:T          pure sampling from softmax(logits / T)            (top_p = 1)
  nucleus:P:T       nucleus (top-p) sampling with temperature T

Metrics on the held-out test images (first --n_images val2017 images, never used for checkpoint selection):
  BLEU-1..4, METEOR, ROUGE-L, CIDEr (pycocoevalcap; captions lower-cased, punctuation removed)
  CLIPScore = 2.5 * max(cos(CLIP image, CLIP caption), 0)  -- reference-free image-caption alignment
  avg words, distinct-1/2, unique-caption share, repeated-bigram share, seconds per image

Example:
  python scripts/eval_m3.py --data $GP1_DATA --tags prefix_full --strategies greedy beam:3 sample:0.7
Writes results/m3/<tag>/metrics.csv (one row per strategy) and results/m3/<tag>/captions.jsonl
(every caption with its per-image CIDEr and CLIPScore).
"""
import argparse, csv, json, re, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from transformers import CLIPModel, CLIPProcessor, GPT2TokenizerFast

from caption_model import ClipCaptioner, GPT2_NAME
from decoding import decode
from train_caption import CaptionPairs, device

ROOT = Path(__file__).resolve().parents[1]


def parse(spec):
    """'beam:5' -> ('beam', {'beam': 5}, 'beam width', 5) etc."""
    kind, *v = spec.split(":")
    if kind == "greedy":
        return "greedy", {}, "greedy", None
    if kind == "beam":
        return "beam", {"beam": int(v[0])}, "beam width", int(v[0])
    if kind == "sample":
        return "nucleus", {"top_p": 1.0, "temperature": float(v[0])}, "temperature", float(v[0])
    if kind == "nucleus":
        return "nucleus", {"top_p": float(v[0]), "temperature": float(v[1])}, "top-p", float(v[0])
    raise ValueError(spec)


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def distinct(caps, n):
    grams = [tuple(w[i:i + n]) for w in (c.split() for c in caps) for i in range(len(w) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


@torch.no_grad()
def clip_cos(clip, proc, img_emb, caps, dev):
    t = []
    for i in range(0, len(caps), 64):
        x = proc(text=caps[i:i + 64], return_tensors="pt", padding=True, truncation=True).to(dev)
        t.append(clip.get_text_features(**x).float().cpu())
    t = torch.nn.functional.normalize(torch.cat(t), dim=-1)
    return (torch.nn.functional.normalize(img_emb, dim=-1) * t).sum(-1)


def load_model(path, dev):
    ck = torch.load(path, map_location="cpu")
    c = ck["config"]
    m = ClipCaptioner(c["injection"], prefix_len=c["prefix_len"], finetune=c.get("finetune", "full"), lora_r=c.get("lora_r", 8))
    m.load_state_dict(ck["state_dict"])
    return m.to(dev).eval(), c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--results", default="m3")
    p.add_argument("--ckpt_dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--strategies", nargs="+", default=["greedy", "beam:3"])
    p.add_argument("--n_images", type=int, default=1000)
    p.add_argument("--bs", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    dev = device()
    tok = GPT2TokenizerFast.from_pretrained(GPT2_NAME)
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    scorers = {"bleu": Bleu(4), "meteor": Meteor(), "rouge": Rouge(), "cider": Cider()}
    va = {}

    for tag in a.tags:
        model, cfg = load_model(a.ckpt_dir / f"{a.results}_{tag}.pt", dev)
        inj = cfg["injection"]
        if inj not in va:
            va[inj] = CaptionPairs(a.data / "val", inj)
        ds = va[inj]
        idx = list(range(min(a.n_images, len(ds.rows))))
        refs = {i: [norm(c) for c in ds.rows[i]["captions"]] for i in idx}
        raw_cls = torch.load(a.data / "val" / "clip_cls.pt")["emb"][idx]
        out = ROOT / "results" / a.results / tag
        out.mkdir(parents=True, exist_ok=True)
        rows, cap_rows = [], []
        for spec in a.strategies:
            strategy, kw, family, value = parse(spec)
            g = torch.Generator().manual_seed(a.seed)
            caps, t0 = [], time.time()
            for s in range(0, len(idx), a.bs):
                chunk = idx[s:s + a.bs]
                if inj == "prefix":
                    cond = model.condition(clip_emb=ds.cls[chunk].to(dev))
                else:
                    cond = model.condition(patches=torch.from_numpy(np.array(ds.patches[chunk], dtype=np.float32)).to(dev))
                extra = {"generator": g} if strategy == "nucleus" else {}
                caps += [tok.decode(c).strip() for c in decode(model, cond, strategy, **kw, **extra)]
            sec = (time.time() - t0) / len(idx)
            nc = [norm(c) or "empty" for c in caps]
            hyp = {i: [c] for i, c in zip(idx, nc)}
            bleu, _ = scorers["bleu"].compute_score(refs, hyp, verbose=0)
            meteor, _ = scorers["meteor"].compute_score(refs, hyp)
            rouge, _ = scorers["rouge"].compute_score(refs, hyp)
            cider, cider_each = scorers["cider"].compute_score(refs, hyp)
            cos = clip_cos(clip, proc, raw_cls, caps, dev)
            row = {"tag": tag, "injection": inj, "finetune": cfg.get("finetune", "full"), "strategy": spec,
                   "family": family, "value": value, "BLEU-1": bleu[0], "BLEU-2": bleu[1], "BLEU-3": bleu[2],
                   "BLEU-4": bleu[3], "METEOR": meteor, "ROUGE-L": rouge, "CIDEr": cider,
                   "CLIPScore": (2.5 * cos.clamp(min=0)).mean().item(),
                   "avg_words": float(np.mean([len(c.split()) for c in nc])),
                   "distinct-1": distinct(nc, 1), "distinct-2": distinct(nc, 2),
                   "unique_captions": len(set(nc)) / len(nc),
                   "repeated_bigram": float(np.mean([max(Counter(zip(w, w[1:])).values(), default=0) > 1
                                                     for w in (c.split() for c in nc)])),
                   "sec_per_image": sec, "n_images": len(idx), "epoch": cfg.get("epoch")}
            rows.append(row)
            cap_rows += [{"strategy": spec, "file_name": ds.rows[i]["file_name"], "caption": c,
                          "cider": round(float(ce), 4), "clipscore": round(2.5 * max(float(co), 0), 4)}
                         for i, c, ce, co in zip(idx, caps, cider_each, cos.tolist())]
            print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                        if k in ("strategy", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "CLIPScore", "distinct-2", "sec_per_image")}, flush=True)
        with open(out / "metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
        with open(out / "captions.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in cap_rows)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
