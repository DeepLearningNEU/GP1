#!/bin/bash
#SBATCH --job-name=gp1_m1
#SBATCH --partition=courses-gpu
#SBATCH --account=ie7615.202710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.log
# Milestone 1 GPU work: image->embedding notebook (incl. 5 image:caption test runs) + 5 prompt->image samples.
source slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
jupyter nbconvert --to notebook --execute --inplace notebooks/01_image_embedding.ipynb
python scripts/generate.py --prompts prompts/m1_test.txt --out results/m1_test
