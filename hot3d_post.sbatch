#!/bin/bash
#SBATCH -J hot3d-post
#SBATCH -p short
#SBATCH --constraint=ib
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH -o logs/%x_%j.log

# After a HOT3D training run: back up weights, encode, probes, prediction example, val gap eval, figures.
# Usage: sbatch --dependency=afterany:<trainjob> -J post-<V> hot3d_post.sbatch <V> <trainjob>
V=${1:?variant none|head|head_hands}
TJ=${2:?training job id}
source ~/.bashrc
conda activate lewm
set -e
cd "$SLURM_SUBMIT_DIR"
export STABLEWM_HOME=/scratch/$USER/stable-wm
DS=$STABLEWM_HOME/datasets
CK=$STABLEWM_HOME/checkpoints/hot3d_$V
N=$(ls $CK/weights_epoch_*.pt | sed 's/.*weights_epoch_//; s/\.pt//' | sort -n | tail -1)
P=hot3d_$V/weights_epoch_$N.pt
echo "variant $V, train job $TJ, last epoch $N"

# 1. weights to home (every 5th epoch + the last), checked by size
H=~/lewm_checkpoints/hot3d_$V
mkdir -p $H
cp $CK/config.json $H/
for f in $CK/weights_epoch_*.pt; do
    e=$(basename $f .pt | sed 's/weights_epoch_//')
    if [ $((e % 5)) -eq 0 ] || [ "$e" = "$N" ]; then cp $f $H/; cmp -s $f $H/$(basename $f) && echo "backed up $(basename $f)"; fi
done
grep -a "created run_dir" logs/hot3d-${V}_${TJ}.log | head -1 | grep -o "/home[^ ]*" > $H/run_dir.txt || true
cp logs/hot3d-${V}_${TJ}.log $H/ || true

# 2-4. embeddings, probes, figures, gap eval
O=/scratch/$USER/hot3d/emb/hot3d_${V}_ep$N
python hot3d_tools/encode.py $P $DS $O --splits train --stride 3 --workers 12
python hot3d_tools/encode.py $P $DS $O --splits val,test --stride 1 --workers 12
python hot3d_tools/probes.py $O $DS --up z
python hot3d_tools/lewm_forecast.py example $P $V $O $DS figs/pred_example_${V}_ep${N}_val.png
python hot3d_tools/gap_eval.py $O $DS /scratch/$USER/hot3d/fisheye512 --split val --policy $P --lewm $P --variant $V
python hot3d_tools/fig_gap.py $O/gap_eval_val.json figs/gap_eval_${V}_ep${N}_val.png "HOT3D LeWM $V epoch $N, val"
RD=$(cat $H/run_dir.txt 2>/dev/null || true)
[ -n "$RD" ] && python hot3d_tools/fig_train.py figs/train_hot3d_${V}_ep${N}.png "$V=$RD/metrics.csv"

# 5. results to home (small: probes, reports, embeddings ~100 MB)
R=~/hot3d_results/hot3d_${V}_ep$N
mkdir -p $R
cp $O/*.json $O/probes.npz $R/
cp figs/*${V}_ep${N}* $R/ 2>/dev/null || true
echo "POST OK $V epoch $N"
