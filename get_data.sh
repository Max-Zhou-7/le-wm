#!/bin/bash
: "${STABLEWM_HOME:?STABLEWM_HOME is not set}"
set -e
D=$STABLEWM_HOME/datasets
mkdir -p $D && cd $D

[ -f pusht_expert_train.h5 ] || {
  hf download quentinll/lewm-pusht --repo-type dataset --include "*.zst" --local-dir $D
  zstd -d pusht_expert_train.h5.zst
}

swm convert pusht_expert_train pusht_expert_train.lance --source-format hdf5 -f lance
echo "--- converted output:"; find $D -maxdepth 1 -name "*.lance"

python - << 'PY'
import glob, os, stable_worldmodel as swm
path = glob.glob(os.environ["STABLEWM_HOME"] + "/datasets/*.lance")[0]
ds = swm.data.load_dataset(path, num_steps=4)
print("LANCE OK:", path, "| length:", len(ds))
PY

rm -f pusht_expert_train.h5 pusht_expert_train.h5.zst
