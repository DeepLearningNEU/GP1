"""Compare decoding strategies (greedy / beam search / nucleus sampling) for a trained captioner.

Example:
  python scripts/eval_decoding.py --data ./data --injection prefix
Writes to results/m2_caption/<injection>/:
  decoding_comparison.csv   BLEU-1/4, CIDEr, CLIPScore, length, diversity, speed per strategy
  decoding_captions.jsonl   every generated caption
"""
import argparse, csv, json, re, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from transformers import CLIPModel, CLIPProcessor, GPT2TokenizerFast

from caption_model import ClipCaptioner, GPT2_NAME
from decoding import decode
from train_caption import CaptionPairs, device

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = {
    "greedy": ("greedy", {}),
    "beam-3": ("beam", {"beam": 3}),
    "beam-5": ("beam", {"beam": 5}),
    "nucleus p=0.5": ("nucleus", {"top_p": 0.5}),
    "nucleus p=0.9": ("nucleus", {"top_p": 0.9}),
}


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def distinct(caps, n):
    grams = [tuple(w[i:i + n]) for w in (c.split() for c in caps) for i in range(len(w) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


@torch.no_grad()
def clip_score(clip, proc, img_emb, caps, dev):
    """CLIPScore (Hessel et al. 2021): 2.5 * max(cos(image, text), 0), reference-free."""
    t = []
    for i in range(0, len(caps), 64):
        x = proc(text=caps[i:i + 64], return_tensors="pt", padding=True, truncation=True).to(dev)
        t.append(clip.get_text_features(**x).float().cpu())
    t = torch.nn.functional.normalize(torch.cat(t), dim=-1)
    cos = (torch.nn.functional.normalize(img_emb, dim=-1) * t).sum(-1)
    return (2.5 * cos.clamp(min=0)).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--injection", choices=["prefix", "cross"], default="prefix")
    p.add_argument("--n_images", type=int, default=500)
    p.add_argument("--bs", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    dev = device()
    out = ROOT / "results" / "m2_caption" / a.injection
    ck = torch.load(ROOT / "checkpoints" / f"m2_{a.injection}.pt", map_location="cpu")
    model = ClipCaptioner(a.injection, prefix_len=ck["config"]["prefix_len"])
    model.load_state_dict(ck["state_dict"]); model.to(dev).eval()
    tok = GPT2TokenizerFast.from_pretrained(GPT2_NAME)
    va = CaptionPairs(a.data / "val", a.injection)
    idx = list(range(min(a.n_images, len(va.rows))))           # metrics are computed on these images
    # also caption the fixed sample images logged during training (for the galleries), metrics exclude them
    pos = {r["file_name"]: i for i, r in enumerate(va.rows)}
    sp = out / "samples_by_stage.jsonl"
    extra_idx = [] if not sp.exists() else [i for i in dict.fromkeys(pos[json.loads(l)["file_name"]] for l in open(sp)) if i not in set(idx)]
    gen_idx = idx + extra_idx
    refs = {i: [norm(c) for c in va.rows[i]["captions"]] for i in idx}
    raw_cls = torch.load(a.data / "val" / "clip_cls.pt")["emb"]

    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    rows, all_caps = [], []
    for name, (strategy, kw) in STRATEGIES.items():
        g = torch.Generator().manual_seed(a.seed)
        caps, t0 = [], time.time()
        for s in range(0, len(gen_idx), a.bs):
            chunk = gen_idx[s:s + a.bs]
            if a.injection == "prefix":
                cond = model.condition(clip_emb=va.cls[chunk].to(dev))
            else:
                cond = model.condition(patches=torch.from_numpy(np.array(va.patches[chunk], dtype=np.float32)).to(dev))
            extra = {"generator": g} if strategy == "nucleus" else {}
            caps += [tok.decode(c).strip() for c in decode(model, cond, strategy, **kw, **extra)]
        sec = (time.time() - t0) / len(gen_idx)
        all_gen, caps = caps, caps[:len(idx)]
        hyp = {i: [norm(c) or "empty"] for i, c in zip(idx, caps)}
        bleu, _ = Bleu(4).compute_score(refs, hyp, verbose=0)
        cider, _ = Cider().compute_score(refs, hyp)
        nc = [norm(c) for c in caps]
        row = {"strategy": name, "BLEU-1": bleu[0], "BLEU-4": bleu[3], "CIDEr": cider,
               "CLIPScore": clip_score(clip, proc, raw_cls[idx], caps, dev),
               "avg_words": np.mean([len(c.split()) for c in nc]),
               "distinct-1": distinct(nc, 1), "distinct-2": distinct(nc, 2),
               "unique_captions": len(set(nc)) / len(nc),
               "repeated_bigram_rate": np.mean([max(Counter(zip(w, w[1:])).values(), default=0) > 1 for w in (c.split() for c in nc)]),
               "sec_per_image": sec}
        rows.append(row)
        all_caps += [{"strategy": name, "file_name": va.rows[i]["file_name"], "caption": c, "in_metrics": i in set(idx)}
                     for i, c in zip(gen_idx, all_gen)]
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    with open(out / "decoding_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
    with open(out / "decoding_captions.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in all_caps)
    print("wrote", out / "decoding_comparison.csv")


if __name__ == "__main__":
    main()
