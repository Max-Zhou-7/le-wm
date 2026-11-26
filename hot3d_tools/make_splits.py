"""Make train/val/test splits for HOT3D Aria recordings.

test  = all recordings of 2 GT participants (picked with a fixed seed so the
        test set has 15-25 recordings)
val   = random recordings from the other 7 GT participants (--n-val)
train = the remaining GT recordings
nogt  = recordings of the HOT3D test participants (no hand/object GT); kept
        apart, usable only for pose-free (pixel-only) training

Usage: python hot3d_tools/make_splits.py <Hot3DAria_download_urls.json> <out.json> [--seed 0]
"""
import argparse
import json
import random

NO_GT = {"P0004", "P0005", "P0006", "P0008", "P0016", "P0020"}  # HOT3D test participants

ap = argparse.ArgumentParser()
ap.add_argument("links")
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-val", type=int, default=12)
a = ap.parse_args()

names = sorted(json.load(open(a.links))["sequences"])
by_p = {}
for n in names:
    by_p.setdefault(n.split("_")[0], []).append(n)
gt_parts = sorted(p for p in by_p if p not in NO_GT)

rng = random.Random(a.seed)
while True:
    test_p = sorted(rng.sample(gt_parts, 2))
    if 15 <= sum(len(by_p[p]) for p in test_p) <= 25:
        break
test = [n for p in test_p for n in by_p[p]]
rest = [n for p in gt_parts if p not in test_p for n in by_p[p]]
val = sorted(rng.sample(rest, a.n_val))
train = sorted(set(rest) - set(val))
nogt = [n for p in sorted(NO_GT & set(by_p)) for n in by_p[p]]

out = dict(seed=a.seed, test_participants=test_p, train=train, val=val, test=test, nogt=nogt)
json.dump(out, open(a.out, "w"), indent=1)
print("GT participants:", {p: len(by_p[p]) for p in gt_parts})
print("test participants:", test_p)
for k in ["train", "val", "test", "nogt"]:
    print(f"{k}: {len(out[k])} recordings, participants {sorted({n.split('_')[0] for n in out[k]})}")
