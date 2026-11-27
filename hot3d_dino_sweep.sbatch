#!/bin/bash
#SBATCH -J dsweep
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:55:00
#SBATCH -o logs/%x_%j.log

# Val sweep of the predictor on frozen DINOv2 features (cls_grid), one motion variant per job.
# Each config: train_feat.py, early stopping on the val-table loss (--epochs 12 --patience 3).
# Picks the config with the lowest val-table loss -> $SW/<V>/best.txt; copies it to checkpoints/dino_best_<V>.
# Usage: sbatch -J dsweep-<V> hot3d_dino_sweep.sbatch <V>
V=${1:?variant none|head|head_hands}
source ~/.bashrc
conda activate lewm
set -e
cd "$SLURM_SUBMIT_DIR"
export STABLEWM_HOME=/scratch/$USER/stable-wm
DS=$STABLEWM_HOME/datasets
F=/scratch/$USER/hot3d/emb/dinov2s_all
SW=$STABLEWM_HOME/checkpoints/dino_sweep/$V
mkdir -p $SW
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
# name hidden depth mlp_dim dropout wd
CFGS="h384_d6_do0.1_wd1e-3:384:6:2048:0.1:1e-3
h384_d6_do0.3_wd1e-3:384:6:2048:0.3:1e-3
h384_d6_do0.1_wd5e-2:384:6:2048:0.1:5e-2
h256_d4_do0.1_wd1e-3:256:4:1024:0.1:1e-3
h256_d4_do0.2_wd1e-3:256:4:1024:0.2:1e-3
h256_d4_do0.3_wd1e-3:256:4:1024:0.3:1e-3
h256_d4_do0.2_wd5e-2:256:4:1024:0.2:5e-2
h192_d2_do0.1_wd1e-3:192:2:768:0.1:1e-3
h192_d2_do0.2_wd1e-3:192:2:768:0.2:1e-3
h192_d2_do0.3_wd1e-3:192:2:768:0.3:1e-3
h192_d2_do0.2_wd5e-2:192:2:768:0.2:5e-2
h192_d4_do0.2_wd1e-2:192:4:768:0.2:1e-2"
for c in $CFGS; do
    IFS=: read -r name h d m do wd <<< "$c"
    [ -f $SW/$name/train_hist.json ] && { echo "skip $name (done)"; continue; }
    python hot3d_tools/train_feat.py $F $DS $SW/$name --version cls_grid --variant $V \
        --epochs 12 --patience 3 --hidden $h --depth $d --mlp-dim $m --dropout $do --wd $wd
done
python - $SW <<'PY'
import json, sys, glob, os
sw = sys.argv[1]
res = []
for f in sorted(glob.glob(f"{sw}/*/train_hist.json")):
    h = json.load(open(f)); res.append((h["best_val_table"], h["best_epoch"], os.path.basename(os.path.dirname(f))))
res.sort()
for v, e, n in res: print(f"{n:24s} val-table {v:.4f} (epoch {e})")
json.dump([dict(name=n, val_table=v, best_epoch=e) for v, e, n in res], open(f"{sw}/sweep.json", "w"), indent=1)
open(f"{sw}/best.txt", "w").write(res[0][2] + "\n")
print("BEST", res[0][2])
PY
B=$(cat $SW/best.txt)
mkdir -p $STABLEWM_HOME/checkpoints/dino_best_$V
cp $SW/$B/featwm.pt $SW/$B/config.json $SW/$B/train_hist.json $STABLEWM_HOME/checkpoints/dino_best_$V/
echo $B > $STABLEWM_HOME/checkpoints/dino_best_$V/sweep_name.txt
echo "SWEEP OK $V best $B"
