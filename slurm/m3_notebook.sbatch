#!/bin/bash
#SBATCH --job-name=gp1_m3nb
#SBATCH --partition=courses
#SBATCH --account=ie7615.202710
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.log
# Milestone 3: run the evaluation notebook (tables + plots) after both m3_gpu.sbatch jobs finished.
set -e
export GP1_DATA=/scratch/$USER/gp1/data
cd notebooks && ~/gp1_env/bin/jupyter nbconvert --to notebook --execute --inplace 03_evaluation.ipynb
echo NOTEBOOK_OK
