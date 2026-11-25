#!/bin/bash
#SBATCH -J hot3d-setup
#SBATCH -p short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x_%j.log

# Installs the HOT3D toolkit into its OWN conda env (never the lewm env).
# Env: ~/envs/hot3d (home, survives scratch wipes). Caches go to scratch.
# Repo: ~/ondemand/dev/LeWM_forecasting/hot3d (cloned if missing).
# Torch is the CPU build: we only use it to read data (saves ~2 GB of home quota).
source ~/.bashrc
set -e
ENV=$HOME/envs/hot3d
REPO=$HOME/ondemand/dev/LeWM_forecasting/hot3d
export CONDA_PKGS_DIRS=/scratch/$USER/conda_pkgs
export PIP_CACHE_DIR=/scratch/$USER/pip_cache

[ -d "$REPO" ] || git clone https://github.com/facebookresearch/hot3d.git "$REPO"
[ -d "$ENV" ] || conda create -y -p "$ENV" python=3.10 pip
conda activate "$ENV"
case "$(which python)" in "$ENV"/*) ;; *) echo "wrong python: $(which python)"; exit 1;; esac

# versions from the HOT3D README (conda section)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install projectaria_tools==1.5.1 requests rerun-sdk==0.16.0 vrs matplotlib
# optional hand deps (MANO hands need the MANO model files, licensed separately)
# chumpy's setup.py imports pip, so it must build without isolation
python -m pip install 'git+https://github.com/vchoutas/smplx.git'
python -m pip install --no-build-isolation 'git+https://github.com/mattloper/chumpy'
python -m pip check || true

cd "$REPO/hot3d"
python -c "
import projectaria_tools, torch, vrs
from dataset_api import Hot3dDataProvider
print('torch', torch.__version__)
print('HOT3D SETUP OK')
"
du -sh "$ENV"
