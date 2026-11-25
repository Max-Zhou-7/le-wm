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

CMD="python train.py data=pusht wandb.enabled=false ${*:2}"
if [ "$MODE" = "smoke" ]; then
    timeout 15m $CMD; rc=$?
    [ $rc -eq 124 ] && { echo "SMOKE OK"; exit 0; }
    exit $rc
else
    $CMD
fi
