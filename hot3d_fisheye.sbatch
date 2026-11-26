#!/bin/bash
#SBATCH -J hot3d-fish
#SBATCH -p short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x_%A_%a.log

# Save downsampled raw fisheye frames + online calibration (for TimeWarp), one array task
# per line of a list file. Output: /scratch/$USER/hot3d/fisheye512/<seq>.npz (copy to home after).
# Usage: sbatch --array=1-33 hot3d_fisheye.sbatch hot3d_tools/valtest_sequences.txt
LIST=${1:?sequence list file}
SIZE=${SIZE:-512}
SEQ=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST" | tr -d '[:space:]')
[ -n "$SEQ" ] || { echo "no sequence on line $SLURM_ARRAY_TASK_ID"; exit 1; }
source ~/.bashrc
conda activate ~/envs/hot3d
set -e
cd "$SLURM_SUBMIT_DIR"
ROOT=/scratch/$USER/hot3d
OUT=$ROOT/fisheye$SIZE/$SEQ.npz
mkdir -p $ROOT/fisheye$SIZE
if [ -f "$OUT" ]; then echo "already done: $OUT"; exit 0; fi
PYTHONPATH=$HOME/ondemand/dev/LeWM_forecasting/hot3d/hot3d \
  python hot3d_tools/fisheye.py save $ROOT/aria/$SEQ $ROOT/assets/assets "$OUT" --size $SIZE \
  --check $ROOT/proc/fov110/$SEQ.npz
echo "FISHEYE OK $SEQ"
