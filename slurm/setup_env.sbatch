#!/bin/bash
#SBATCH --job-name=gp1_setup
#SBATCH --partition=courses
#SBATCH --account=ie7615.202710
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.log
# Create the conda env at ~/gp1_env and install pinned dependencies.
set -e
module load miniconda3/25.9.1
# conda-forge only: the default channels require an interactive ToS acceptance
conda create -y -p "$HOME/gp1_env" -c conda-forge --override-channels python=3.11
source activate "$HOME/gp1_env"
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
pip install -r requirements.txt
python -c "import torch, diffusers, transformers; print(torch.__version__, diffusers.__version__, transformers.__version__)"
