#!/bin/bash
#SBATCH -J hot3d-lance
#SBATCH -p short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --constraint=ib
# (ib = cascadelake/skylake nodes; lance crashes with "Illegal instruction" on old c3xxx CPUs)
#SBATCH --time=06:00:00
#SBATCH -o logs/%x_%j.log

# Build HOT3D Lance tables from the npz files, verify them, and copy them to home.
# Usage: sbatch hot3d_lance.sbatch [build|verify|copy|all]   (default all)
# Lance goes to $STABLEWM_HOME/datasets (scratch); the copy goes to ~/hot3d_lance/fov$FOV.
# The npz files are NOT copied to home.
STEP=${1:-all}
FOV=${FOV:-110}
source ~/.bashrc
conda activate lewm
set -e
cd "$SLURM_SUBMIT_DIR"
NPZ=/scratch/$USER/hot3d/proc/fov$FOV
OUT=$STABLEWM_HOME/datasets
HOME_DST=$HOME/hot3d_lance/fov$FOV

if [ "$STEP" = build ] || [ "$STEP" = all ]; then
  python hot3d_tools/build_lance.py $NPZ hot3d_tools/splits.json $OUT --fov $FOV
  cp hot3d_tools/splits.json $OUT/hot3d_splits.json
  du -sh $OUT/hot3d_*_fov$FOV.lance
fi
if [ "$STEP" = verify ] || [ "$STEP" = all ]; then
  python hot3d_tools/verify_lance.py $NPZ $OUT --fov $FOV | tee /tmp/verify_$SLURM_JOB_ID.txt
  grep -q "VERIFY OK" /tmp/verify_$SLURM_JOB_ID.txt
fi
if [ "$STEP" = copy ] || [ "$STEP" = all ]; then
  need=$(du -sk $OUT/hot3d_*_fov$FOV.lance | awk '{s+=$1} END {print s}')
  used=$(du -sk $HOME | awk '{print $1}')
  limit=$((75 * 1024 * 1024))
  echo "need ${need} KB, home used ${used} KB, limit ${limit} KB"
  if [ $((used + need + 3 * 1024 * 1024)) -gt $limit ]; then
    echo "NOT ENOUGH HOME SPACE (keeping 3 GB margin); not copying"; exit 1
  fi
  mkdir -p $HOME_DST
  rsync -a $OUT/hot3d_*_fov$FOV.lance $OUT/hot3d_fov${FOV}_meta.json $OUT/hot3d_splits.json $HOME_DST/
  # check the copy: same file list and sizes
  (cd $OUT && find hot3d_*_fov$FOV.lance -type f -printf "%p %s\n" | sort) > /tmp/a_$SLURM_JOB_ID
  (cd $HOME_DST && find hot3d_*_fov$FOV.lance -type f -printf "%p %s\n" | sort) > /tmp/b_$SLURM_JOB_ID
  cmp /tmp/a_$SLURM_JOB_ID /tmp/b_$SLURM_JOB_ID && echo "COPY OK -> $HOME_DST" && du -sh $HOME_DST
fi
echo "LANCE DONE ($STEP)"
