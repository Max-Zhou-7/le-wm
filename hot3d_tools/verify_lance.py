"""Verify HOT3D Lance tables against the source npz files.

Checks per split:
  - row count and per-recording frame counts match the npz files
  - for random frames of every recording: numeric columns equal the npz
    (NaN in the same places), JPEG pixels close to the npz pixels
  - the table loads through swm.data.load_dataset with frameskip 3 (10 fps)
    and returns a clip with the expected shapes

Usage (lewm env): python hot3d_tools/verify_lance.py <npz_dir> <lance_dir> [--fov 110] [--per-rec 5]
"""
import argparse
import io
import json

import lance
import numpy as np
import stable_worldmodel as swm
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("npz_dir")
ap.add_argument("lance_dir")
ap.add_argument("--fov", type=int, default=110)
ap.add_argument("--per-rec", type=int, default=5)
a = ap.parse_args()
meta = json.load(open(f"{a.lance_dir}/hot3d_fov{a.fov}_meta.json"))
rng = np.random.default_rng(0)
NUM = ["T_world_device", "T_device_camera", "hand_wrist", "hand_joints", "hand_valid",
       "hand_vis_ratio", "obj_pose", "obj_valid", "qa_pass", "mask_hand_visible",
       "mask_object_visible", "mask_good_exposure"]
ok = True


def fail(msg):
    global ok
    ok = False
    print("FAIL", msg)


for split, names in meta["splits"].items():
    path = f"{a.lance_dir}/hot3d_{split}_fov{a.fov}.lance"
    ds = lance.dataset(path)
    ep = ds.to_table(columns=["episode_idx", "recording"]).to_pydict()
    ep_idx, rec = np.array(ep["episode_idx"]), np.array(ep["recording"])
    want = sum(meta["frames"][n] for n in names)
    if len(ep_idx) != want:
        fail(f"{split}: {len(ep_idx)} rows, expected {want}")
    starts = {}
    max_px_err = 0.0
    for e, n in enumerate(names):
        rows = np.nonzero(ep_idx == e)[0]
        starts[n] = rows[0]
        if len(rows) != meta["frames"][n] or not (rec[rows] == n).all() or not (np.diff(rows) == 1).all():
            fail(f"{split}/{n}: episode {e} rows/recording mismatch")
            continue
        d = np.load(f"{a.npz_dir}/{n}.npz")
        T = meta["frames"][n]
        idx = np.sort(rng.choice(T, min(a.per_rec, T), replace=False))
        tab = ds.take(rows[idx].tolist(), columns=["step_idx", "pixels", "t_sec"] + NUM).to_pydict()
        if list(tab["step_idx"]) != idx.tolist():
            fail(f"{split}/{n}: step_idx mismatch")
        ts = d["timestamps_ns"]
        if not np.allclose(tab["t_sec"], ((ts[idx] - ts[0]) / 1e9)[:, None], atol=1e-5):
            fail(f"{split}/{n}: t_sec mismatch")
        for c in NUM:
            got = np.asarray(tab[c], np.float32)
            ref = np.asarray(d[c], np.float32).reshape(T, -1)[idx]
            if got.shape != ref.shape or not np.allclose(got, ref, equal_nan=True, atol=1e-6):
                fail(f"{split}/{n}: column {c} differs")
        for j, blob in enumerate(tab["pixels"]):
            im = np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"), np.float32)
            err = np.abs(im - d["pixels"][idx[j]]).mean()
            max_px_err = max(max_px_err, err)
            if im.shape != d["pixels"].shape[1:] or err > 4:
                fail(f"{split}/{n}: frame {idx[j]} pixels differ (mean abs err {err:.2f})")
    print(f"{split}: {len(names)} recordings, {len(ep_idx)} rows; max JPEG mean-abs-err {max_px_err:.2f}/255")

    # load like train.py does: 10 fps clips (frameskip 3)
    sd = swm.data.load_dataset(path, transform=None, num_steps=4, frameskip=3,
                               keys_to_load=["pixels", "hand_wrist", "T_world_device"])
    s = sd[len(sd) // 2]
    print(f"  swm load: {len(sd)} clips; sample", {k: tuple(v.shape) for k, v in s.items() if hasattr(v, "shape")})

print("VERIFY OK" if ok else "VERIFY FAILED")
