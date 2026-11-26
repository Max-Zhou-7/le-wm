"""Build Lance tables (one per split) from preprocessed HOT3D npz files.

Uses stable_worldmodel's LanceWriter (same format as Push-T): one episode per
recording, all 30 fps frames, `pixels` as JPEG (quality 95), other columns as
float32 lists. Booleans are stored as 0/1.
LanceDB rejects NaN in float-list columns, so missing poses are stored as 0
and flagged by head_valid / hand_valid / hand_joints_valid / obj_valid;
hand_vis_ratio is -1 where HOT3D has no 2D box row.
Training at 10 fps = frameskip 3 at load time.

Columns:
  pixels                      224x224 RGB (pinhole, not rotated)
  t_sec                       seconds since the first frame of the recording
  T_world_device [7], T_device_camera [7]      [tx,ty,tz,qw,qx,qy,qz]
  head_valid [1]
  hand_wrist [14] (left 7, right 7), hand_joints [2*J], hand_valid [2], hand_joints_valid [2], hand_vis_ratio [2]
  obj_pose [O*7], obj_valid [O]                (O and object order: see the sidecar json)
  qa_pass, mask_hand_visible, mask_object_visible, mask_good_exposure
  recording, participant (strings)
A sidecar <out_dir>/hot3d_fov<F>_meta.json stores K, fov, obj_uids, splits, and
per-recording frame counts.

Usage (lewm env):
  python hot3d_tools/build_lance.py <npz_dir> <splits.json> <out_dir> [--fov 110]
"""
import argparse
import json
import os
import time

import numpy as np
from stable_worldmodel.data.formats.lance import LanceWriter

ap = argparse.ArgumentParser()
ap.add_argument("npz_dir")
ap.add_argument("splits")
ap.add_argument("out_dir")
ap.add_argument("--fov", type=int, default=110)
ap.add_argument("--splits-to-build", default="train,val,test,nogt")
a = ap.parse_args()
splits = json.load(open(a.splits))
meta = dict(fov_deg=a.fov, splits={}, frames={}, missing=[])


def episode(path):
    d = np.load(path)
    T = len(d["timestamps_ns"])
    ts = d["timestamps_ns"]
    f = lambda x, fill=0.0: np.nan_to_num(np.asarray(x, np.float32).reshape(T, -1), nan=fill)
    isval = lambda x: ~np.isnan(np.asarray(x, np.float32)).any(-1)
    ep = {
        "pixels": d["pixels"],
        "t_sec": f((ts - ts[0]) / 1e9),
        "T_world_device": f(d["T_world_device"]),
        "T_device_camera": f(d["T_device_camera"]),
        "head_valid": f(isval(d["T_world_device"]) & isval(d["T_device_camera"])),
        "hand_wrist": f(d["hand_wrist"]),
        "hand_joints": f(d["hand_joints"]),
        "hand_valid": f(isval(d["hand_wrist"])),
        "hand_joints_valid": f(isval(d["hand_joints"])),
        "hand_vis_ratio": f(d["hand_vis_ratio"], fill=-1.0),
        "obj_pose": f(d["obj_pose"]),
        "obj_valid": f(isval(d["obj_pose"])),
        "qa_pass": f(d["qa_pass"]),
        "mask_hand_visible": f(d["mask_hand_visible"]),
        "mask_object_visible": f(d["mask_object_visible"]),
        "mask_good_exposure": f(d["mask_good_exposure"]),
        "recording": np.array([str(d["recording"])] * T),
        "participant": np.array([str(d["participant"])] * T),
    }
    if "K" not in meta:
        meta["K"] = d["K"].tolist()
        meta["obj_uids"] = d["obj_uids"].tolist()
        meta["hand_joints_dim"] = int(d["hand_joints"].shape[-1])
    else:
        assert np.allclose(meta["K"], d["K"]) and meta["obj_uids"] == d["obj_uids"].tolist()
    # LanceWriter expects a list of per-frame values for each column
    ep = {k: list(v) for k, v in ep.items()}
    return ep, T


for split in a.splits_to_build.split(","):
    out = f"{a.out_dir}/hot3d_{split}_fov{a.fov}.lance"
    names = [n for n in splits[split] if os.path.exists(f"{a.npz_dir}/{n}.npz")]
    meta["missing"] += [n for n in splits[split] if n not in names]
    meta["splits"][split] = names
    t0 = time.time()

    def gen():
        for i, n in enumerate(names):
            ep, T = episode(f"{a.npz_dir}/{n}.npz")
            meta["frames"][n] = T
            print(f"[{split}] {i + 1}/{len(names)} {n} T={T} {time.time() - t0:.0f}s", flush=True)
            yield ep

    with LanceWriter(out, mode="overwrite") as w:
        w.write_episodes(gen())
    print(f"WROTE {out}: {len(names)} recordings, {sum(meta['frames'][n] for n in names)} frames", flush=True)

json.dump(meta, open(f"{a.out_dir}/hot3d_fov{a.fov}_meta.json", "w"), indent=1)
print("missing npz:", meta["missing"])
