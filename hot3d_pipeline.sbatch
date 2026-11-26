#!/bin/bash
#SBATCH -J hot3d-pipe
#SBATCH -p short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x_%A_%a.log

# Per sequence: download -> preprocess to npz -> (optionally) delete recording.vrs.
# One array task per line of a sequence list file.
# Usage:
#   sbatch --array=1-5 hot3d_pipeline.sbatch hot3d_tools/pilot_sequences.txt        # keep VRS
#   DELETE_VRS=1 sbatch --array=1-198%20 hot3d_pipeline.sbatch <list>             # delete VRS after
# Small non-VRS files (GT csv/jsonl, masks, metadata, mps) are always kept.
LIST=${1:?sequence list file}
FOV=${FOV:-110}
DELETE_VRS=${DELETE_VRS:-0}
SEQ=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST" | tr -d '[:space:]')
[ -n "$SEQ" ] || { echo "no sequence on line $SLURM_ARRAY_TASK_ID"; exit 1; }
source ~/.bashrc
conda activate ~/envs/hot3d
set -e
ROOT=/scratch/$USER/hot3d
DIR=$ROOT/aria/$SEQ
OUT=$ROOT/proc/fov${FOV}/$SEQ.npz
mkdir -p $ROOT/proc/fov${FOV}
cd "$SLURM_SUBMIT_DIR"
echo "== $SEQ  fov=$FOV delete_vrs=$DELETE_VRS"

if [ -f "$OUT" ]; then echo "already done: $OUT"; exit 0; fi
if [ ! -f "$DIR/recording.vrs" ]; then
  bash hot3d_download.sbatch aria "$SEQ" "main_vrs ground_truth hand_data mps_slam_trajectories mps_slam_calibration"
fi

PYTHONPATH=$HOME/ondemand/dev/LeWM_forecasting/hot3d/hot3d \
  python hot3d_tools/preprocess_sequence.py "$DIR" $ROOT/assets/assets "$OUT.tmp.npz" --fov $FOV
mv "$OUT.tmp.npz" "$OUT"

if [ "$DELETE_VRS" = 1 ]; then
  rm -f "$DIR/recording.vrs"
  echo "deleted $DIR/recording.vrs"
fi
echo "PIPELINE OK $SEQ"
