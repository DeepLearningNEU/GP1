# Shared settings for all jobs (source this). Edit paths for your own account.
module load miniconda3/25.9.1
source activate "$HOME/gp1_env"
export SCRATCH_DIR=/scratch/$USER/gp1
export GP1_DATA=$SCRATCH_DIR/data
export HF_HOME=$SCRATCH_DIR/hf_cache
