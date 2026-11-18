"""Pre-compute frozen CLIP ViT-L/14 image features for captioning, plus all reference captions.

For each split in <data>/<split>/ (built by prepare_coco.py) writes:
  clip_cls.pt       {"file_name": [...], "emb": (N, 768) float32}  projected [CLS] embedding  -> prefix injection
  clip_patches.npy  (N, 257, 1024) float16 memmap, last hidden states (CLS + 16x16 patches) -> cross-attention
  captions.jsonl    {"file_name", "captions": [all COCO captions]}  (5 per image; used for training + references)
"""
import argparse, json, zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

CLIP_NAME = "openai/clip-vit-large-patch14"


def device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def all_captions(raw_dir, split):
    with zipfile.ZipFile(raw_dir / "annotations_trainval2017.zip") as z:
        ann = json.load(z.open(f"annotations/captions_{split}2017.json"))
    names = {im["id"]: im["file_name"] for im in ann["images"]}
    caps = {}
    for a in ann["annotations"]:
        caps.setdefault(names[a["image_id"]], []).append(" ".join(a["caption"].strip().split()))
    return caps


@torch.no_grad()
def run(split_dir, raw_dir, split, clip, proc, dev, bs):
    rows = [json.loads(l) for l in open(split_dir / "metadata.jsonl")]
    caps = all_captions(raw_dir, split)
    with open(split_dir / "captions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"file_name": r["file_name"], "captions": caps[Path(r["file_name"]).name]}) + "\n")

    n = len(rows)
    patches = np.lib.format.open_memmap(split_dir / "clip_patches.npy", mode="w+", dtype=np.float16, shape=(n, 257, 1024))
    cls = torch.empty(n, 768)
    for i in tqdm(range(0, n, bs), desc=f"CLIP {split}"):
        ims = [Image.open(split_dir / r["file_name"]).convert("RGB") for r in rows[i:i + bs]]
        px = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
        out = clip.vision_model(pixel_values=px)
        cls[i:i + len(ims)] = clip.visual_projection(out.pooler_output).float().cpu()
        patches[i:i + len(ims)] = out.last_hidden_state.half().cpu().numpy()
    patches.flush()
    torch.save({"file_name": [r["file_name"] for r in rows], "emb": cls}, split_dir / "clip_cls.pt")
    print(f"{split}: {n} images | cls {tuple(cls.shape)} | patches {patches.shape} | "
          f"{sum(len(caps[Path(r['file_name']).name]) for r in rows)} captions")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--bs", type=int, default=32)
    a = p.parse_args()
    dev = device()
    clip = CLIPModel.from_pretrained(CLIP_NAME).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_NAME)
    for s in a.splits:
        run(a.data / s, a.data / "raw", s, clip, proc, dev, a.bs)
