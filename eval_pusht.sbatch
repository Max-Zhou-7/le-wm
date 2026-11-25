#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x_%j.log

# Usage: sbatch eval_pusht.sbatch <policy path relative to $STABLEWM_HOME> [extra overrides]
POLICY=${1:?give a policy, e.g. pusht/lewm}
source ~/.bashrc
conda activate lewm
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
set -e

# Released checkpoint: download and convert once, if missing
if [ "$POLICY" = "pusht/lewm" ] && [ ! -f "$STABLEWM_HOME/pusht/lewm_object.ckpt" ]; then
  hf download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht
  python convert_hf_pusht.py
  ls -lh $STABLEWM_HOME/pusht/
fi

python eval.py --config-name=pusht.yaml policy=$POLICY world.max_episode_steps=50 ${*:2}
