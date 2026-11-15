"""Text -> image with Stable Diffusion; saves each image and a captioned grid.

Example:
  python scripts/generate.py --prompts prompts/m1_test.txt --out results/m1_test
"""
import argparse, json, time
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

p = argparse.ArgumentParser()
p.add_argument("--prompts", type=Path, required=True, help="text file, one prompt per line")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--model", default=MODEL)
p.add_argument("--steps", type=int, default=50)
p.add_argument("--guidance", type=float, default=7.5)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

prompts = [l.strip() for l in open(a.prompts) if l.strip()]
a.out.mkdir(parents=True, exist_ok=True)
pipe = StableDiffusionPipeline.from_pretrained(a.model, torch_dtype=torch.float16).to("cuda")
pipe.set_progress_bar_config(disable=True)

images, log = [], []
for i, prompt in enumerate(prompts):
    g = torch.Generator("cuda").manual_seed(a.seed + i)
    t0 = time.time()
    img = pipe(prompt, num_inference_steps=a.steps, guidance_scale=a.guidance, generator=g).images[0]
    img.save(a.out / f"{i:02d}.png")
    images.append(img)
    log.append({"id": i, "prompt": prompt, "seconds": round(time.time() - t0, 2)})
    print(log[-1])

# captioned grid
fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4.8))
for ax, img, prompt in zip(axes, images, prompts):
    ax.imshow(img); ax.axis("off"); ax.set_title(prompt, fontsize=9, wrap=True)
plt.tight_layout()
plt.savefig(a.out / "grid.png", dpi=120)

cfg = {k: str(v) for k, v in vars(a).items()} | {"scheduler": type(pipe.scheduler).__name__}
json.dump({"config": cfg, "runs": log}, open(a.out / "log.json", "w"), indent=2)
