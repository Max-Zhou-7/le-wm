#!/bin/bash
#SBATCH -J gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x_%j.log

# Run one command on a GPU node (lewm env). HF cache on scratch.
# Usage: sbatch -J <name> [--gres=gpu:a100:1 --time=..] run_gpu.sbatch python ...
source ~/.bashrc
conda activate ${ENV:-lewm}
set -e
cd "$SLURM_SUBMIT_DIR"
export STABLEWM_HOME=/scratch/$USER/stable-wm
export HF_HOME=/scratch/$USER/hf_cache
mkdir -p $HF_HOME
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  cmd: $*"
"$@"
echo "RUN OK"
