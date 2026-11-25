#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x_%j.log

# Usage: sbatch eval_pusht.sbatch <policy> [extra overrides]
# policy is relative to $STABLEWM_HOME/checkpoints/: a folder with one .pt + config.json,
# or a .pt file next to a config.json. Examples: pusht_released, lewm/weights_epoch_6.pt
POLICY=${1:?give a policy, e.g. pusht_released}
source ~/.bashrc
conda activate lewm
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
set -e

# Released checkpoint: download and convert once, if missing
if [ "$POLICY" = "pusht_released" ] && [ ! -f "$STABLEWM_HOME/checkpoints/pusht_released/weights.pt" ]; then
  [ -f "$STABLEWM_HOME/hf_pusht/weights.pt" ] || hf download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht
  python convert_hf_pusht.py
  ls -lh $STABLEWM_HOME/checkpoints/pusht_released/
fi

python eval.py --config-name=pusht.yaml policy=$POLICY world.max_episode_steps=50 ${*:2}
