#!/bin/bash
#SBATCH -J eval2
#SBATCH -p short
#SBATCH --constraint=ib
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x_%j.log

# Val evaluation v2 (gaps 1..32 = 100-3200 ms, floor, Kalman, optional wrist noise) for one model.
# Usage: sbatch -J e2-<NAME> hot3d_eval_v2.sbatch <NAME> <POLICY> <VARIANT> <EMB_DIR> [noise sigmas...]
#   LeWM:   NAME=lewm_head  POLICY=hot3d_head/weights_epoch_40.pt  EMB_DIR=/scratch/.../emb/hot3d_head_ep40
#   DINOv2: NAME=dino_best_head POLICY=feat:dino_best_head  EMB_DIR=/scratch/.../emb/dino_best_head (made here)
# Outputs: <EMB_DIR>/gap_eval_val_k32.json, rq2_val_k32.json; figs/*_k32_val.png; copies in ~/hot3d_results/v2/<NAME>/
NAME=${1:?}; P=${2:?}; V=${3:?}; O=${4:?}; shift 4
source ~/.bashrc
conda activate lewm
set -e
cd "$SLURM_SUBMIT_DIR"
export STABLEWM_HOME=/scratch/$USER/stable-wm
export HF_HOME=/scratch/$USER/hf_cache
DS=$STABLEWM_HOME/datasets
if [[ $P == feat:* ]]; then        # DINOv2 model: link the shared features + probes
    ALL=/scratch/$USER/hot3d/emb/dinov2s_all
    mkdir -p $O
    for f in emb_val.npy rows_val.npy probes.npz probe_report.json; do ln -sf $ALL/$f $O/$f; done
    python hot3d_tools/lewm_forecast.py example $P $V $O $DS figs/pred_example_${NAME}_val.png
fi
NOISE=""; [ $# -gt 0 ] && NOISE="--wrist-noise $*"
python hot3d_tools/gap_eval.py $O $DS /scratch/$USER/hot3d/fisheye512 --split val --policy $P --lewm $P --variant $V \
    --out $O/gap_eval_val_k32.json $NOISE
python hot3d_tools/fig_gap.py $O/gap_eval_val_k32.json figs/gap_eval_${NAME}_k32_val.png "$NAME ($V), val, gaps 100-3200 ms"
python hot3d_tools/rq2.py $P $V $O $DS figs/rq2_${NAME}_k32 --json-out $O/rq2_val_k32.json
R=~/hot3d_results/v2/$NAME
mkdir -p $R
cp $O/gap_eval_val_k32.json $O/rq2_val_k32.json $R/
cp figs/*${NAME}*k32* $R/ 2>/dev/null || true
cp figs/pred_example_${NAME}_val.* $R/ 2>/dev/null || true
echo "EVAL2 OK $NAME"
