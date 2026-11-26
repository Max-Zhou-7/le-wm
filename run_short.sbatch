#!/bin/bash
#SBATCH -J short
#SBATCH -p short
#SBATCH --constraint=ib
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x_%j.log

# Run one command on a CPU node (ib constraint: Lance crashes on old c3xxx CPUs).
# Usage: [ENV=lewm] sbatch -J <name> run_short.sbatch python hot3d_tools/actions.py add $STABLEWM_HOME/datasets
source ~/.bashrc
conda activate ${ENV:-lewm}
set -e
cd "$SLURM_SUBMIT_DIR"
export STABLEWM_HOME=/scratch/$USER/stable-wm
echo "host $(hostname)  cmd: $*"
"$@"
echo "RUN OK"
