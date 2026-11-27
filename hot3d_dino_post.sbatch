#!/bin/bash
#SBATCH -J dino-post
#SBATCH -p short
#SBATCH --constraint=ib
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -o logs/%x_%j.log

# After train_feat.py for one variant: val-only gap eval (incl. TimeWarp encoded with DINOv2), prediction
# example, RQ2 simulation, figures; copy results + weights to home. Never touches test.
# Usage: sbatch --dependency=afterok:<trainjob>:<probejob> -J dpost-<V> hot3d_dino_post.sbatch <V>
V=${1:?variant none|head|head_hands}
source ~/.bashrc
conda activate lewm
set -e
cd "$SLURM_SUBMIT_DIR"
export STABLEWM_HOME=/scratch/$USER/stable-wm
export HF_HOME=/scratch/$USER/hf_cache
DS=$STABLEWM_HOME/datasets
NAME=dino_cls_grid_$V
P=$STABLEWM_HOME/checkpoints/$NAME
ALL=/scratch/$USER/hot3d/emb/dinov2s_all
O=/scratch/$USER/hot3d/emb/$NAME            # per-variant dir: links to the shared features + probes
mkdir -p $O
for f in emb_val.npy rows_val.npy probes.npz probe_report.json; do ln -sf $ALL/$f $O/$f; done
python hot3d_tools/lewm_forecast.py example feat:$P $V $O $DS figs/pred_example_${NAME}_val.png
python hot3d_tools/gap_eval.py $O $DS /scratch/$USER/hot3d/fisheye512 --split val --policy feat:$P --lewm feat:$P --variant $V
python hot3d_tools/fig_gap.py $O/gap_eval_val.json figs/gap_eval_${NAME}_val.png "LeWM predictor on frozen DINOv2-S (CLS+2x2 grid), $V, val"
python hot3d_tools/rq2.py feat:$P $V $O $DS figs/rq2_$NAME
R=~/hot3d_results/$NAME
mkdir -p $R ~/lewm_checkpoints/$NAME
cp $O/*.json $R/ ; cp figs/*${NAME}* $R/ 2>/dev/null || true
cp $P/featwm.pt $P/config.json $P/train_hist.json ~/lewm_checkpoints/$NAME/
cmp -s $P/featwm.pt ~/lewm_checkpoints/$NAME/featwm.pt && echo "backed up weights"
echo "DINO POST OK $V"
