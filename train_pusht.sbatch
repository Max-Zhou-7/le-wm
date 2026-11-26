#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.log

MODE=${1:-full}
source ~/.bashrc
conda activate lewm
export PYTHONUNBUFFERED=1
export STABLEWM_HOME=/scratch/zhou.zilu/stable-wm
cd "$SLURM_SUBMIT_DIR"

echo "Mode: $MODE  Node: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'; print('torch', torch.__version__, 'GPU OK:', torch.cuda.get_device_name(0))" || exit 1

# LOCAL_COPY=<table.lance>: copy that dataset to node-local /tmp first (avoids re-reading it from
# scratch over the network when page cache is short; two HOT3D runs on one node fell to <1 it/s).
if [ -n "$LOCAL_COPY" ]; then
    LD=/tmp/lewm_$SLURM_JOB_ID
    mkdir -p $LD/datasets
    trap "rm -rf $LD" EXIT
    t0=$(date +%s)
    cp -r "$STABLEWM_HOME/datasets/$LOCAL_COPY" $LD/datasets/
    echo "local copy of $LOCAL_COPY: $(du -sh $LD/datasets | cut -f1) in $(( $(date +%s) - t0 )) s"
    export LOCAL_DATASET_DIR=$LD
fi

CMD="python train.py data=${DATA:-pusht} wandb.enabled=false ${*:2}"
if [ "$MODE" = "smoke" ]; then
    timeout ${SMOKE_MIN:-15}m $CMD; rc=$?
    [ $rc -eq 124 ] && { echo "SMOKE OK"; exit 0; }
    exit $rc
else
    $CMD
fi
