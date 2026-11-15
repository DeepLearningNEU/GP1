#!/bin/bash
#SBATCH --job-name=gp1_data
#SBATCH --partition=courses
#SBATCH --account=ie7615.202710
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.log
# Download + preprocess COCO (5000 train pairs, 5000 val images) and cache SD v1.5 weights.
source slurm/env.sh
python scripts/prepare_coco.py --out "$GP1_DATA" --n_train 5000
python -c "from diffusers import StableDiffusionPipeline as P; import torch; P.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5', torch_dtype=torch.float16)"
