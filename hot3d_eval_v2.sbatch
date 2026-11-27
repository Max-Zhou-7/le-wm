#!/bin/bash
#SBATCH -J eval2
#SBATCH -p short
#SBATCH --constraint=ib
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x_%j.log

# Evaluation v2 (gaps 1..32 = 100-3200 ms, oracle floor, Kalman, optional wrist noise, RQ2) for one model.
# Usage: [SPLIT=val|test] sbatch -J e2-<NAME> hot3d_eval_v2.sbatch <NAME> <POLICY> <VARIANT> <EMB_DIR> [noise sigmas...]
#   LeWM:   NAME=lewm_head  POLICY=hot3d_head/weights_epoch_40.pt  EMB_DIR=/scratch/.../emb/hot3d_head_ep40
#   DINOv2: NAME=dino_best_head POLICY=feat:dino_best_head  EMB_DIR=/scratch/.../emb/dino_best_head (made here)
# SPLIT=test (only in the approved test run) reuses the val choices: Kalman (q, r) per gap from
# ~/hot3d_results/v2/<NAME>/gap_eval_val_k32.json and RQ2 thresholds from rq2_val_k32.json.
# Outputs: <EMB_DIR>/gap_eval_<S>_k32.json, rq2_<S>_k32.json; figs/*_k32_<S>.png; copies in ~/hot3d_results/v2/<NAME>/
NAME=${1:?}; P=${2:?}; V=${3:?}; O=${4:?}; shift 4
S=${SPLIT:-val}
source ~/.bashrc
conda activate lewm
set -e
cd "$SLURM_SUBMIT_DIR"
export STABLEWM_HOME=/scratch/$USER/stable-wm
export HF_HOME=/scratch/$USER/hf_cache
DS=$STABLEWM_HOME/datasets
R=~/hot3d_results/v2/$NAME
mkdir -p $R
if [[ $P == feat:* ]]; then        # DINOv2 model: link the shared features + probes
    ALL=/scratch/$USER/hot3d/emb/dinov2s_all
    mkdir -p $O
    for f in emb_$S.npy rows_$S.npy probes.npz probe_report.json; do ln -sf $ALL/$f $O/$f; done
    if [ "$S" = val ]; then
        python hot3d_tools/lewm_forecast.py example $P $V $O $DS figs/pred_example_${NAME}_val.png
        mkdir -p ~/lewm_checkpoints/$NAME
        cp $STABLEWM_HOME/checkpoints/${P#feat:}/* ~/lewm_checkpoints/$NAME/
    fi
fi
EXTRA=""; [ $# -gt 0 ] && EXTRA="--wrist-noise $*"
RQ=""
if [ "$S" = test ]; then
    EXTRA="$EXTRA --kalman-params $R/gap_eval_val_k32.json"
    RQ="--split test --params $R/rq2_val_k32.json"
fi
python hot3d_tools/gap_eval.py $O $DS /scratch/$USER/hot3d/fisheye512 --split $S --policy $P --lewm $P --variant $V \
    --out $O/gap_eval_${S}_k32.json $EXTRA
python hot3d_tools/fig_gap.py $O/gap_eval_${S}_k32.json figs/gap_eval_${NAME}_k32_$S.png "$NAME ($V), $S, gaps 100-3200 ms"
python hot3d_tools/rq2.py $P $V $O $DS figs/rq2_${NAME}_k32 --json-out $O/rq2_${S}_k32.json $RQ
cp $O/gap_eval_${S}_k32.json $O/rq2_${S}_k32.json $R/
cp figs/*${NAME}_k32* $R/ 2>/dev/null || true
cp figs/pred_example_${NAME}_val.* $R/ 2>/dev/null || true
echo "EVAL2 OK $NAME $S"
