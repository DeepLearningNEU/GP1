"""Milestone 2 figures: training curves and caption galleries.

Writes to results/m2_caption/:
  loss_curves.png             train loss (per 50 steps) + val loss (per epoch) for both injection methods
  gallery_<injection>.png     sample val images with captions from greedy / beam-5 / nucleus p=0.9
  progress_<injection>.png    the same images captioned before training, early in training and after training
"""
import argparse, csv, json, textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "m2_caption"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#ecebe7", "#ffffff"
COLORS = {"prefix": "#2a78d6", "cross": "#eb6834"}
LABEL = {"prefix": "Prefix conditioning", "cross": "Cross-attention"}
plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 10, "axes.edgecolor": "#d6d5d0",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def read_csv(p):
    return list(csv.DictReader(open(p))) if p.exists() else []


def loss_curves():
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=180)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    for inj in ["prefix", "cross"]:
        tr, ep = read_csv(OUT / inj / "train_log.csv"), read_csv(OUT / inj / "epoch_log.csv")
        if not tr:
            continue
        c = COLORS[inj]
        ax.plot([int(r["step"]) for r in tr], [float(r["train_loss"]) for r in tr], color=c, lw=1.6, alpha=0.9,
                label=f"{LABEL[inj]} — train")
        pts = [(int(r["step"]), float(r["val_loss"])) for r in ep if int(r["epoch"]) > 0]
        ax.plot(*zip(*pts), color=c, lw=0, marker="o", ms=7, mec=SURF, mew=1.5, label=f"{LABEL[inj]} — val (per epoch)")
    ax.set(xlabel="Training step (batch 32)", ylabel="Cross-entropy loss (per token)", ylim=(1.0, 4.6))
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "loss_curves.png", facecolor=SURF); plt.close(fig)


def gallery(data, inj, rows_by_name, title_lines, path, n=8):
    names = list(rows_by_name)[:n]
    cols = 4
    fig, axes = plt.subplots(2, cols, figsize=(15, 11.5), dpi=130)
    fig.patch.set_facecolor(SURF)
    for ax, name in zip(axes.flat, names):
        ax.imshow(Image.open(data / "val" / name).resize((320, 320)))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        txt = "\n".join(textwrap.fill(f"{lab}: {rows_by_name[name].get(lab, '')}", 46) for lab in title_lines)
        ax.set_xlabel(txt, fontsize=8.6, color=INK, loc="left", labelpad=6)
    fig.tight_layout(); fig.savefig(path, facecolor=SURF); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    a = p.parse_args()
    loss_curves()
    for inj in ["prefix", "cross"]:
        sp = OUT / inj / "samples_by_stage.jsonl"
        if not sp.exists():
            continue
        samples = [json.loads(l) for l in open(sp)]
        stages = list(dict.fromkeys(s["stage"] for s in samples))
        keep = ["step 100", "step 300", stages[-1]]   # untrained output is unrelated news text; described in the report
        by = {}
        for s in samples:
            by.setdefault(s["file_name"], {"reference": s["reference"]})[s["stage"]] = s["generated"]
        gallery(a.data, inj, by, keep + ["reference"], OUT / f"progress_{inj}.png")
        dc = OUT / inj / "decoding_captions.jsonl"
        if dc.exists():
            dec = {}
            for r in map(json.loads, open(dc)):
                if r["file_name"] in by:
                    dec.setdefault(r["file_name"], {"reference": by[r["file_name"]]["reference"]})[r["strategy"]] = r["caption"]
            dec = {k: dec[k] for k in by if k in dec}
            gallery(a.data, inj, dec, ["greedy", "beam-5", "nucleus p=0.9", "reference"], OUT / f"gallery_{inj}.png")
        # small copies of the sample images so the report can be built without the dataset
        (OUT / "sample_images").mkdir(exist_ok=True)
        for name in by:
            Image.open(a.data / "val" / name).convert("RGB").resize((300, 300)).save(OUT / "sample_images" / Path(name).name, quality=90)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
