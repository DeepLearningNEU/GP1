# GP1 — From Words to Worlds: Text-to-Image with Transformers + Diffusion

IE 7615 Deep Learning for AI — Generative Project.
Text-to-image pipeline: **CLIP ViT-L/14 text encoder (Transformer)** → **Stable Diffusion v1.5 (latent diffusion UNet)**, fine-tuned with LoRA on COCO Captions and evaluated with FID / IS.

## Repository layout

| Path | Purpose |
|---|---|
| `scripts/prepare_coco.py` | Builds the datasets: 5000 COCO train2017 pairs (fine-tuning) + 5000 val2017 images (FID/IS reference) |
| `scripts/dataset.py` | PyTorch `Dataset`: 512×512 image normalized to [-1, 1] + CLIP token ids |
| `scripts/generate.py` | Prompt → image with Stable Diffusion; saves images, a grid, and a JSON log |
| `notebooks/01_image_embedding.ipynb` | Milestone 1 baseline: image → CLIP embedding, GPT-2 caption tokenization, 5 image:caption test runs |
| `prompts/` | Prompt lists |
| `slurm/` | Job scripts for the NEU Explorer cluster |
| `results/m1_image_caption/` | The 5 image:caption test runs (figure, similarity matrix, JSON) |
| `results/m1_test/` | 5 prompt → image samples from Stable Diffusion v1.5 |
| `docs/proposal.pdf` | Project proposal |
| `scripts/extract_clip_features.py` | M2: frozen CLIP ViT-L/14 features ([CLS] embedding + 257 patch tokens) and all COCO captions |
| `scripts/caption_model.py` | M2: CLIP → GPT-2 captioner with prefix conditioning or cross-attention injection |
| `scripts/train_caption.py` | M2: training; writes step/epoch logs and captions of 8 fixed val images during training |
| `scripts/decoding.py`, `scripts/eval_decoding.py` | M2: greedy / beam search / nucleus sampling, scored with BLEU, CIDEr, CLIPScore, diversity |
| `scripts/m2_figures.py`, `scripts/m2_report.py` | M2: figures and the report PDF |
| `results/m2_caption/` | M2 training logs, caption samples, decoding comparison, figures |
| `docs/milestone2_report.pdf` | Milestone 2 report (front page, training log, early caption samples, observations) |
| `scripts/eval_m3.py` | M3: BLEU-1..4, METEOR, ROUGE-L, CIDEr, CLIPScore and diversity for any decoding setting (greedy / beam:K / sample:T / nucleus:P:T) |
| `notebooks/03_evaluation.ipynb` | M3 evaluation notebook: all tables and plots (fine-tuning strategies, beam width, temperature, injection method, decoding, qualitative) |
| `slurm/m3_gpu.sbatch`, `slurm/m3_notebook.sbatch` | M3 jobs: train full / LoRA / frozen GPT-2 for one injection method and evaluate; then run the notebook |
| `results/m3/` | M3 metrics per model (`<injection>_<finetune>/metrics.csv`, `captions.jsonl`), `tables/`, `figures/` |
| `docs/milestone3_report.pdf` | Milestone 3 report (methods and results) |

## Data

COCO 2017 Captions (downloaded by `prepare_coco.py`, not stored in git):

- **train**: 5000 random train2017 images (seed 42). Each keeps one caption chosen at random among those with 5–30 words, so every caption fits CLIP's 77-token limit.
- **val**: all 5000 val2017 images, used as the FID/IS reference set; their captions are the evaluation prompts.
- Images are resized on the short side and center-cropped to 512×512. `dataset.py` normalizes them to [-1, 1].
- Output format: `<split>/images/*.jpg` plus `<split>/metadata.jsonl` with `{"file_name", "text"}` (Hugging Face imagefolder format).

## Setup and running (NEU Explorer HPC)

```bash
git clone git@github.com:DeepLearningNEU/GP1.git && cd GP1
sbatch slurm/setup_env.sbatch      # conda env at ~/gp1_env (Python 3.11, CUDA 12.1 PyTorch)
sbatch slurm/prepare_data.sbatch   # COCO -> /scratch/$USER/gp1/data, caches SD v1.5 weights
sbatch slurm/m1_gpu.sbatch         # runs the M1 notebook + 5 prompt->image samples on a GPU
sbatch slurm/m2_gpu.sbatch         # M2: CLIP features, train both captioners, decoding study, figures (~1.5 h on a P100)
sbatch slurm/m3_gpu.sbatch prefix  # M3: full / LoRA / frozen GPT-2 + evaluation (also run with: cross)
sbatch slurm/m3_notebook.sbatch    # M3: evaluation notebook -> results/m3/tables, results/m3/figures
```

Paths are set in `slurm/env.sh` (`GP1_DATA`, `HF_HOME`). On another machine, install `requirements.txt`, set the same variables, and run the scripts directly:

```bash
python scripts/prepare_coco.py --out ./data
python scripts/generate.py --prompts prompts/m1_test.txt --out results/m1_test --steps 50 --guidance 7.5
```

## Milestones

- [x] M1 — Setup and dataset preparation: environment, 5000 COCO pairs, image → embedding notebook with GPT-2 tokenization, 5 image:caption test runs, proposal
- [x] M2 — CLIP image encoder + GPT-2 decoder (prefix conditioning and cross-attention), baseline captions, decoding strategies
- [x] M3 — Evaluation: BLEU / METEOR / ROUGE-L / CIDEr / CLIPScore on 1,000 held-out images; beam width, temperature, injection method; full vs. LoRA vs. prefix tuning
- [ ] M4 — Final report, gallery, presentation
