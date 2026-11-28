"""Segment label per gap-eval window and gap, from GT poses only (no model):
  python hot3d_tools/segments.py <lance_dir> <split> <out.npz> [--fov 110]

Over the gap interval (raw rows r, r+3, ..., r+3g; 10 fps), compared with row r:
  1 interaction : some object (valid at r and at the step) moves >= OBJ_M in the world frame
  2 head-only   : not interaction, every valid wrist stays within WRIST_M (world), and the head turns >= HEAD_DEG
                  or moves >= HEAD_M
  0 other       : everything else (still scenes; hands moving without moving an object)
Thresholds frozen on val (2026-11-28). Anchors = gap_eval.anchors (same windows, same order; checked in boot.py).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap_eval import FS, GAPS, anchors, load_split  # noqa: E402

OBJ_M, WRIST_M, HEAD_DEG, HEAD_M = 0.02, 0.05, 5.0, 0.05
NAMES = {0: "other", 1: "interaction", 2: "head_only"}


def labels(D, R, g):
    steps = R[:, None] + FS * np.arange(1, g + 1)[None]          # [n,g]
    r0 = R[:, None]
    ov = D["ov"][r0] & D["ov"][steps]                             # [n,g,O]
    dob = np.linalg.norm(D["obj_w"][steps] - D["obj_w"][r0], axis=-1)
    obj = np.where(ov, dob, 0).max(axis=(1, 2))
    hv = D["hav"][r0] & D["hav"][steps]                           # [n,g,2]
    dw = np.linalg.norm(D["wrist_w"][steps] - D["wrist_w"][r0], axis=-1)
    wr = np.where(hv, dw, 0).max(axis=(1, 2))
    Rr = D["Rwc"][R]                                               # [n,3,3]
    Rs = D["Rwc"][steps]                                           # [n,g,3,3]
    tr = np.einsum("nij,ngij->ng", Rr, Rs)
    ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).max(1)
    tm = np.linalg.norm(D["twc"][steps] - D["twc"][r0], axis=-1).max(1)
    lab = np.zeros(len(R), np.int8)
    inter = obj >= OBJ_M
    lab[inter] = 1
    lab[~inter & (wr < WRIST_M) & ((ang >= HEAD_DEG) | (tm >= HEAD_M))] = 2
    return lab, dict(obj=obj, wrist=wr, head_deg=ang, head_m=tm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lance_dir"); ap.add_argument("split"); ap.add_argument("out")
    ap.add_argument("--fov", type=int, default=110); ap.add_argument("--every", type=int, default=15)
    a = ap.parse_args()
    D = load_split(a.lance_dir, a.split, a.fov)
    R = anchors(D, a.every)
    out = dict(rec=np.asarray(D["rec"][R]).astype(str), R=R)
    print(f"{a.split}: {len(R)} windows. Shares other / interaction / head-only:")
    for g in GAPS:
        lab, q = labels(D, R, g)
        out[f"lab_g{g}"] = lab
        for k, v in q.items():
            out[f"{k}_g{g}"] = v.astype(np.float32)
        sh = [np.mean(lab == i) for i in (0, 1, 2)]
        print(f"  {100 * g:5d} ms  " + "  ".join(f"{NAMES[i]} {s:.3f}" for i, s in zip((0, 1, 2), sh)), flush=True)
    np.savez_compressed(a.out, **out)
    print("SAVED", a.out)


if __name__ == "__main__":
    main()
