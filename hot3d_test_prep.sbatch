#!/bin/bash
#SBATCH -J test-prep
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH -o logs/%x_%j.log

# Prepare the single approved TEST run. Restores from home whatever scratch lost (wipe: first Tuesday
# of the month), then encodes the test split if its features are missing:
#   Lance tables      ~/hot3d_lance/fov110/            -> $STABLEWM_HOME/datasets/
#   fisheye 512       ~/hot3d_fisheye/fisheye512/      -> /scratch/$USER/hot3d/fisheye512/
#   LeWM weights      ~/lewm_checkpoints/hot3d_<V>/weights_epoch_40.pt + config.json
#   DINOv2 predictors ~/lewm_checkpoints/dino_best_<V>/
#   probes            ~/hot3d_results/hot3d_<V>_ep40/probes.npz, ~/hot3d_results/dinov2s_all/probes.npz
#   test features     LeWM: encode.py (stride 1); DINOv2-S cls_grid: dino_encode.py --allow-test (stride 1)
source ~/.bashrc
conda activate lewm
set -e
cd "$SLURM_SUBMIT_DIR"
export STABLEWM_HOME=/scratch/$USER/stable-wm
export HF_HOME=/scratch/$USER/hf_cache
DS=$STABLEWM_HOME/datasets
E=/scratch/$USER/hot3d/emb
mkdir -p $DS $E $HF_HOME /scratch/$USER/hot3d
for t in train val test; do
    [ -d $DS/hot3d_${t}_fov110.lance ] || { echo "restore $t lance"; cp -r ~/hot3d_lance/fov110/hot3d_${t}_fov110.lance $DS/; }
done
for f in hot3d_fov110_meta.json hot3d_splits.json; do [ -f $DS/$f ] || cp ~/hot3d_lance/fov110/$f $DS/ 2>/dev/null || true; done
[ -d /scratch/$USER/hot3d/fisheye512 ] || { echo "restore fisheye"; cp -r ~/hot3d_fisheye/fisheye512 /scratch/$USER/hot3d/; }
for V in none head head_hands; do
    C=$STABLEWM_HOME/checkpoints/hot3d_$V
    mkdir -p $C
    [ -f $C/weights_epoch_40.pt ] || cp ~/lewm_checkpoints/hot3d_$V/weights_epoch_40.pt ~/lewm_checkpoints/hot3d_$V/config.json $C/
    C=$STABLEWM_HOME/checkpoints/dino_best_$V
    [ -f $C/featwm.pt ] || { mkdir -p $C; cp ~/lewm_checkpoints/dino_best_$V/* $C/; }
    O=$E/hot3d_${V}_ep40
    mkdir -p $O
    [ -f $O/probes.npz ] || cp ~/hot3d_results/hot3d_${V}_ep40/probes.npz $O/
    [ -f $O/emb_test.npy ] || python hot3d_tools/encode.py hot3d_$V/weights_epoch_40.pt $DS $O --splits test --stride 1 --workers 12
done
A=$E/dinov2s_all
mkdir -p $A
[ -f $A/probes.npz ] || cp ~/hot3d_results/dinov2s_all/probes.npz ~/hot3d_results/dinov2s_all/probe_report.json $A/
[ -f $A/emb_test.npy ] || python hot3d_tools/dino_encode.py $DS $A --splits test --stride 1 --allow-test
ls -la $A $E/hot3d_head_ep40
echo "TEST PREP OK"
