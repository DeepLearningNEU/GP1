#!/bin/bash
#SBATCH --job-name=gp1_m3
#SBATCH --partition=courses-gpu
#SBATCH --account=ie7615.202710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.log
# Milestone 3 for one injection method (pass prefix or cross):  sbatch slurm/m3_gpu.sbatch prefix
#   1. train GPT-2 with three fine-tuning strategies: full / LoRA / frozen (prefix tuning)
#      first 1000 val images are a held-out test set; the checkpoint with the lowest loss on the other 4000 is kept
#   2. evaluate on the 1000 test images: BLEU, METEOR, ROUGE-L, CIDEr, CLIPScore
#      full model: beam width and temperature sweeps; LoRA / frozen: greedy, beam-3, nucleus
set -e
INJ=${1:?usage: sbatch slurm/m3_gpu.sbatch prefix|cross}
source slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
CK=$SCRATCH_DIR/checkpoints
COMMON="--data $GP1_DATA --injection $INJ --results m3 --ckpt_dir $CK --test_images 1000 --save_best --epochs 5"
for FT in full lora frozen; do
  python scripts/train_caption.py $COMMON --finetune $FT --tag ${INJ}_${FT}
done
SWEEP="greedy beam:2 beam:3 beam:5 beam:8 beam:10 sample:0.25 sample:0.5 sample:0.75 sample:1.0 sample:1.25 sample:1.5 nucleus:0.5:1.0 nucleus:0.9:1.0"
python scripts/eval_m3.py --data $GP1_DATA --ckpt_dir $CK --tags ${INJ}_full --strategies $SWEEP
python scripts/eval_m3.py --data $GP1_DATA --ckpt_dir $CK --tags ${INJ}_lora ${INJ}_frozen --strategies greedy beam:3 nucleus:0.9:1.0
cp "logs/gp1_m3-${SLURM_JOB_ID}.log" "results/m3/train_console_${INJ}.log"
