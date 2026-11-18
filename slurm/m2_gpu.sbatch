#!/bin/bash
#SBATCH --job-name=gp1_m2
#SBATCH --partition=courses-gpu
#SBATCH --account=ie7615.202710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.log
# Milestone 2: CLIP ViT-L/14 encoder + GPT-2 decoder captioning.
#   1. frozen CLIP features (CLS embedding + patch tokens) and all 5 COCO captions per image
#   2. train with prefix conditioning and with cross-attention injection
#   3. compare decoding strategies (greedy / beam / nucleus) on 500 val images
#   4. figures for the report
set -e
source slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
[ -f "$GP1_DATA/val/clip_patches.npy" ] || python scripts/extract_clip_features.py --data "$GP1_DATA"
for INJ in prefix cross; do
  python scripts/train_caption.py --data "$GP1_DATA" --injection $INJ --epochs 5
  python scripts/eval_decoding.py --data "$GP1_DATA" --injection $INJ --n_images 500
done
python scripts/m2_figures.py --data "$GP1_DATA"
cp "logs/gp1_m2-${SLURM_JOB_ID}.log" results/m2_caption/train_console.log
