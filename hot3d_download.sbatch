#!/bin/bash
#SBATCH -J hot3d-dl
#SBATCH -p short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x_%j.log

# Download HOT3D data with the official downloader, into scratch (re-creatable).
# Usage:
#   sbatch hot3d_download.sbatch assets
#   sbatch hot3d_download.sbatch aria "<seq1> <seq2> ..." "<data types>"
#   e.g. sbatch hot3d_download.sbatch aria P0003_c701bd11 "main_vrs ground_truth hand_data"
# Link JSONs: ~/ondemand/dev/LeWM_forecasting/hot3d_links/ (expire 14 days after issue).
WHAT=${1:?assets or aria}
SEQS=${2:-}
TYPES=${3:-all}
source ~/.bashrc
conda activate ~/envs/hot3d
set -e
LINKS=$HOME/ondemand/dev/LeWM_forecasting/hot3d_links
OUT=/scratch/$USER/hot3d
mkdir -p "$OUT"
cd $HOME/ondemand/dev/LeWM_forecasting/hot3d/hot3d/data_downloader

if [ "$WHAT" = assets ]; then
  yes y | python dataset_downloader_base_main.py -c $LINKS/Hot3DAssets_download_urls.json \
    -o "$OUT/assets" -l assets -d assets
else
  [ -n "$SEQS" ] || { echo "give sequence names"; exit 1; }
  # the downloader takes "all" or numeric indices, so map names -> indices
  if [ "$TYPES" != all ]; then
    TYPES=$(python -c "
import sys
from utils import load_data_groups_from_cdn
g = [k for k in load_data_groups_from_cdn(sys.argv[1]) if k != 'mps_slam_summary']
print(' '.join(str(g.index(t)) for t in sys.argv[2:]))
" $LINKS/Hot3DAria_download_urls.json $TYPES)
  fi
  yes y | python dataset_downloader_base_main.py -c $LINKS/Hot3DAria_download_urls.json \
    -o "$OUT/aria" -l $SEQS -d $TYPES
fi
echo "DOWNLOAD DONE"
du -sh "$OUT"/*
