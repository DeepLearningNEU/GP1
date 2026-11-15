"""Build the image-caption sets from COCO 2017.

train: N random train2017 images (downloaded one by one) -> fine-tuning set
val:   all 5000 val2017 images -> FID/IS reference set + prompts for evaluation
Output per split: <out>/<split>/images/*.jpg (512x512) + metadata.jsonl {"file_name", "text"}
(metadata.jsonl is the Hugging Face "imagefolder" format used by diffusers training scripts.)
"""
import argparse, json, random, zipfile, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image
from tqdm import tqdm

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/{split}2017/{name}"


def load_captions(raw_dir, split):
    """Return {image_file_name: [captions]} for a split, downloading annotations once."""
    ann_zip = raw_dir / "annotations_trainval2017.zip"
    if not ann_zip.exists():
        ann_zip.write_bytes(requests.get(ANN_URL, timeout=600).content)
    with zipfile.ZipFile(ann_zip) as z:
        ann = json.load(z.open(f"annotations/captions_{split}2017.json"))
    names = {im["id"]: im["file_name"] for im in ann["images"]}
    caps = {}
    for a in ann["annotations"]:
        caps.setdefault(names[a["image_id"]], []).append(" ".join(a["caption"].split()))
    return caps


def pick_caption(captions, rng, lo=5, hi=30):
    """Clean step: keep captions with lo..hi words (fits CLIP's 77 tokens), pick one at random."""
    ok = [c for c in captions if lo <= len(c.split()) <= hi]
    return rng.choice(ok) if ok else None


def resize_crop(img, size):
    """Resize short side to `size`, then center crop to size x size."""
    img = img.convert("RGB")
    s = size / min(img.size)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
    l, t = (img.width - size) // 2, (img.height - size) // 2
    return img.crop((l, t, l + size, t + size))


def process(split, name, out_img, size):
    dst = out_img / name
    if dst.exists():
        return True
    try:
        r = requests.get(IMG_URL.format(split=split, name=name), timeout=60)
        resize_crop(Image.open(io.BytesIO(r.content)), size).save(dst, quality=95)
        return True
    except Exception as e:
        print("skip", name, e)
        return False


def build(split, n, raw_dir, out_dir, size, seed):
    rng = random.Random(seed)
    caps = load_captions(raw_dir, split)
    names = sorted(caps)
    if n < len(names):
        names = rng.sample(names, n)
    rows = [(nm, c) for nm in names if (c := pick_caption(caps[nm], rng))]

    out_img = out_dir / split / "images"
    out_img.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as ex:
        ok = list(tqdm(ex.map(lambda r: process(split, r[0], out_img, size), rows), total=len(rows), desc=split))

    with open(out_dir / split / "metadata.jsonl", "w") as f:
        for (nm, c), good in zip(rows, ok):
            if good:
                f.write(json.dumps({"file_name": f"images/{nm}", "text": c}) + "\n")
    print(f"{split}: {sum(ok)} pairs written to {out_dir / split}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_train", type=int, default=5000)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    raw = a.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    build("train", a.n_train, raw, a.out, a.size, a.seed)
    build("val", 5000, raw, a.out, a.size, a.seed)
