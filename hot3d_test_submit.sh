#!/bin/bash
# THE single test run (submit only after the user's OK). Prep (GPU, restores + encodes test), then one
# CPU eval job per model with SPLIT=test: Kalman (q, r, phi) and RQ2 thresholds frozen on val
# (~/hot3d_results/v3/<NAME>/*_val_k32.json), then the recording-level bootstrap + CI figure on test.
# Usage (login node is fine, it only submits): bash hot3d_test_submit.sh
set -e
cd ~/ondemand/dev/LeWM_forecasting/le-wm
E=/scratch/$USER/hot3d/emb
PJ=$(sbatch --parsable hot3d_test_prep.sbatch)
echo "prep $PJ"
J=""
for V in none head head_hands; do
    N=""; [ $V = head_hands ] && N="0.5 1 2"
    J="$J:$(SPLIT=test sbatch --parsable --dependency=afterok:$PJ --time=04:00:00 --cpus-per-task=8 --mem=48G -J t3-lewm_$V \
        hot3d_eval_v2.sbatch lewm_$V hot3d_$V/weights_epoch_40.pt $V $E/hot3d_${V}_ep40)"
    J="$J:$(SPLIT=test sbatch --parsable --dependency=afterok:$PJ --time=04:00:00 --cpus-per-task=8 --mem=48G -J t3-dino_best_$V \
        hot3d_eval_v2.sbatch dino_best_$V feat:dino_best_$V $V $E/dino_best_$V $N)"
done
cat > logs/test_boot.sh <<'EOS'
set -e
python hot3d_tools/boot.py ~/hot3d_results/v3 ~/hot3d_results/v3/boot_test.md --split test
python hot3d_tools/fig_boot.py ~/hot3d_results/v3 figs/boot_ci_test.png --split test
cp figs/boot_ci_test.png ~/hot3d_results/v3/
EOS
sbatch --dependency=afterok$J -J t3-boot run_short.sbatch bash logs/test_boot.sh
squeue -u $USER
