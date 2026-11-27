#!/bin/bash
# THE single test run (submit only after the user's OK). Prep (GPU, restores + encodes test), then one
# CPU eval job per model with SPLIT=test: Kalman (q, r) and RQ2 thresholds frozen on val.
# Usage (login node is fine, it only submits): bash hot3d_test_submit.sh
set -e
cd ~/ondemand/dev/LeWM_forecasting/le-wm
E=/scratch/$USER/hot3d/emb
PJ=$(sbatch --parsable hot3d_test_prep.sbatch)
echo "prep $PJ"
for V in none head head_hands; do
    N=""; [ $V = head_hands ] && N="0.5 1 2"
    SPLIT=test sbatch --dependency=afterok:$PJ -J t2-lewm_$V hot3d_eval_v2.sbatch lewm_$V hot3d_$V/weights_epoch_40.pt $V $E/hot3d_${V}_ep40
    SPLIT=test sbatch --dependency=afterok:$PJ -J t2-dino_best_$V hot3d_eval_v2.sbatch dino_best_$V feat:dino_best_$V $V $E/dino_best_$V $N
done
squeue -u $USER
