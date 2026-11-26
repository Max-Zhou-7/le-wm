"""HOT3D action column: head motion + each wrist's movement per 30 fps step.

action[t] describes the motion from raw frame t to t+1 (same convention as Push-T:
the action taken after observation t). With frameskip 3, swm stacks rows s, s+1, s+2,
i.e. the motion from frame s to s+3.

All motion is expressed in the CAMERA frame at time t (camera = RGB 214-1, not rotated,
x right / y down / z forward as in the stored pinhole images):
  action[0:3]   head (camera) translation, cm
  action[3:6]   head (camera) rotation, rotation vector, degrees
  action[6:9]   left wrist displacement (world motion, rotated into camera(t) axes), cm
  action[9:12]  right wrist displacement, cm
  action[12:15] valid flags: head, left, right (1 = the numbers above are real)
action_valid [3] = action[12:15] (separate column for loss masks / normalizers).

A part is valid only if its pose is valid at t AND t+1, the head is valid at t,
and t+1 - t <= 50 ms. Invalid parts are 0 with flag 0. The last row of a recording
has no t+1, so it is all 0. RULE: code must use the flags; a 0 is not "no motion".

Usage (lewm env, on a short node with --constraint=ib):
  python hot3d_tools/actions.py add <lance_dir> [--fov 110] [--splits train,val,test,nogt]
  python hot3d_tools/actions.py stats <lance_dir>
"""
import argparse

import numpy as np

MAX_DT_S = 0.050
ACTION_DIM = 15
PARTS = ["head", "left", "right"]


def quat_to_mat(q):
    """q [...,4] as w,x,y,z -> [...,3,3]."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-12)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def mat_to_rotvec(R):
    """[...,3,3] -> rotation vector [...,3] (radians). Accurate except very near 180 deg."""
    c = ((np.trace(R, axis1=-2, axis2=-1) - 1) / 2).clip(-1, 1)
    ang = np.arccos(c)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(ang)
    small = s < 1e-6
    k = np.where(small, 0.5, ang / (2 * np.where(small, 1, s)))  # ang/(2 sin ang) -> 1/2 as ang -> 0
    return v * k[..., None]


def pose7_to_Rt(p):
    """[...,7] (tx,ty,tz,qw,qx,qy,qz) -> R [...,3,3], t [...,3]."""
    return quat_to_mat(p[..., 3:7]), p[..., 0:3]


def compute_actions(t_sec, T_world_device, T_device_camera, head_valid, hand_wrist, hand_valid,
                    max_dt=MAX_DT_S):
    """One recording. Shapes: t_sec [T], T_* [T,7], head_valid [T], hand_wrist [T,14] (L,R),
    hand_valid [T,2]. Returns action [T,15] float32."""
    t = np.asarray(t_sec, np.float64).reshape(-1)
    T = len(t)
    out = np.zeros((T, ACTION_DIM), np.float64)
    if T < 2:
        return out.astype(np.float32)
    Rwd, twd = pose7_to_Rt(np.asarray(T_world_device, np.float64))
    Rdc, tdc = pose7_to_Rt(np.asarray(T_device_camera, np.float64))
    Rwc = Rwd @ Rdc                                   # camera -> world
    twc = np.einsum("tij,tj->ti", Rwd, tdc) + twd
    hv = np.asarray(head_valid).reshape(-1) > 0.5
    hw = np.asarray(hand_wrist, np.float64).reshape(T, 2, 7)[..., :3]
    hav = np.asarray(hand_valid).reshape(T, 2) > 0.5
    step_ok = np.diff(t) <= max_dt                    # [T-1]

    a, b = slice(0, T - 1), slice(1, T)
    RcT = np.swapaxes(Rwc[a], -1, -2)                 # world -> camera(t)
    # relative camera motion: T_c(t)_c(t+1) = T_wc(t)^-1 T_wc(t+1)
    R_rel = RcT @ Rwc[b]
    t_rel = np.einsum("tij,tj->ti", RcT, twc[b] - twc[a])
    head_ok = hv[a] & hv[b] & step_ok
    out[a, 0:3] = t_rel * 100.0
    out[a, 3:6] = np.degrees(mat_to_rotvec(R_rel))
    out[a, 12] = head_ok
    for k in range(2):
        ok = hav[a, k] & hav[b, k] & hv[a] & step_ok   # hv[a]: need camera(t) axes
        d = np.einsum("tij,tj->ti", RcT, hw[b, k] - hw[a, k]) * 100.0
        out[a, 6 + 3 * k: 9 + 3 * k] = d
        out[a, 12 + 1 + k] = ok
    # zero every invalid part (never leave numbers computed from 0-filled poses)
    out[a, 0:6] *= head_ok[:, None]
    out[a, 6:9] *= out[a, 13:14]
    out[a, 9:12] *= out[a, 14:15]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


def _episodes(tab):
    ep = np.asarray(tab["episode_idx"]).reshape(-1)
    cuts = np.flatnonzero(np.diff(ep)) + 1
    return np.split(np.arange(len(ep)), cuts)


COLS = ["episode_idx", "step_idx", "t_sec", "T_world_device", "T_device_camera", "head_valid",
        "hand_wrist", "hand_valid"]


def table_actions(ds):
    tab = ds.to_table(columns=COLS)
    order = np.lexsort((np.asarray(tab["step_idx"]).reshape(-1), np.asarray(tab["episode_idx"]).reshape(-1)))
    assert (order == np.arange(len(order))).all(), "rows are not in (episode, step) order"
    cols = {c: np.asarray(tab[c].to_pylist(), np.float64) for c in COLS[2:]}
    act = np.zeros((len(order), ACTION_DIM), np.float32)
    for idx in _episodes({"episode_idx": np.asarray(tab["episode_idx"].to_pylist())}):
        act[idx] = compute_actions(*(cols[c][idx] for c in COLS[2:]))
    return act


def add(a):
    import lance
    import pyarrow as pa
    for split in a.splits.split(","):
        path = f"{a.lance_dir}/hot3d_{split}_fov{a.fov}.lance"
        ds = lance.dataset(path)
        if "action" in ds.schema.names:
            print(f"{split}: already has action, skip", flush=True)
            continue
        act = table_actions(ds)
        flat = pa.array(act.reshape(-1), pa.float32())
        flat_v = pa.array(act[:, 12:15].reshape(-1), pa.float32())
        new = pa.table({"action": pa.FixedSizeListArray.from_arrays(flat, ACTION_DIM),
                        "action_valid": pa.FixedSizeListArray.from_arrays(flat_v, 3)})
        ds.add_columns(pa.RecordBatchReader.from_batches(new.schema, new.to_batches(max_chunksize=65536)))
        ds = lance.dataset(path)
        chk = np.asarray(ds.to_table(columns=["action"])["action"].to_pylist(), np.float32)
        assert np.array_equal(chk, act), "action column read back differs"
        v = act[:, 12:15].mean(0)
        print(f"{split}: added action {act.shape}, valid head/left/right "
              f"{v[0]:.3f}/{v[1]:.3f}/{v[2]:.3f}", flush=True)


def stats(a):
    import lance
    for split in a.splits.split(","):
        ds = lance.dataset(f"{a.lance_dir}/hot3d_{split}_fov{a.fov}.lance")
        tab = ds.to_table(columns=["action", "T_world_device", "T_device_camera", "head_valid"])
        act = np.asarray(tab["action"].to_pylist(), np.float32)
        print(f"== {split}: {len(act)} rows")
        # which world axis is up? camera y points down in the image, so its mean world direction ~ down
        hv = np.asarray(tab["head_valid"].to_pylist()).reshape(-1) > 0.5
        Rwd, twd = pose7_to_Rt(np.asarray(tab["T_world_device"].to_pylist(), np.float64)[hv])
        Rdc, _ = pose7_to_Rt(np.asarray(tab["T_device_camera"].to_pylist(), np.float64)[hv])
        Rwc = Rwd @ Rdc
        print(f"  head pos mean {twd.mean(0).round(2)} std {twd.std(0).round(2)} | camera y-axis "
              f"(image down) mean world dir {Rwc[:, :, 1].mean(0).round(2)}, z-axis (forward) "
              f"{Rwc[:, :, 2].mean(0).round(2)}")
        names = ["head_tx", "head_ty", "head_tz", "head_rx", "head_ry", "head_rz",
                 "L_x", "L_y", "L_z", "R_x", "R_y", "R_z"]
        for j, n in enumerate(names):
            m = act[:, 12 + (0 if j < 6 else 1 if j < 9 else 2)] > 0.5
            x = act[m, j]
            print(f"  {n:8s} valid {m.mean():.3f}  mean {x.mean():+.3f}  std {x.std():.3f}  "
                  f"p99|x| {np.percentile(np.abs(x), 99):.3f}  max|x| {np.abs(x).max():.2f}")


def _selftest():
    rng = np.random.default_rng(0)
    T = 50
    t = np.arange(T) / 30.0
    t[30:] += 0.1                                     # one 133 ms gap between rows 29 and 30
    # camera moving at 30 cm/s along its own z, rotating 30 deg/s about its own y; device == camera
    ang = np.radians(30.0) * t
    q = np.stack([np.cos(ang / 2), 0 * ang, np.sin(ang / 2), 0 * ang], -1)
    R = quat_to_mat(q)
    pos = np.cumsum(np.concatenate([[[0, 0, 0]], np.einsum("tij,j->ti", R[:-1], [0, 0, 1]) * 0.3 * np.diff(t)[:, None]]), 0)
    Twd = np.concatenate([pos, q], -1)
    Tdc = np.tile([0, 0, 0, 1, 0, 0, 0], (T, 1)).astype(float)
    hw = np.zeros((T, 14)); hw[:, 3] = hw[:, 10] = 1
    hw[:, 0:3] = pos + [0.1, 0, 0.4]                  # left wrist moves with the head (world)
    hw[:, 7:10] = [0.2, 0.0, 0.5]                     # right wrist fixed in the world
    hv = np.ones(T); hav = np.ones((T, 2)); hav[10, 1] = 0
    A = compute_actions(t, Twd, Tdc, hv, hw, hav)
    dt = 1 / 30
    assert np.allclose(A[0, 0:3], [0, 0, 30 * dt], atol=1e-6), A[0, 0:3]
    assert np.allclose(A[0, 3:6], [0, 30 * dt, 0], atol=1e-6), A[0, 3:6]
    assert np.allclose(A[0, 9:12], 0) and A[0, 14] == 1
    assert A[29, 12:15].sum() == 0 and A[29, :12].sum() == 0, "gap step must be invalid"
    assert A[9, 14] == 0 and A[10, 14] == 0 and A[11, 14] == 1, "hand invalid at 10 -> steps 9,10 invalid"
    assert A[-1].sum() == 0
    # left wrist world displacement rotated into camera(t): |d| = 1 cm per frame
    assert np.allclose(np.linalg.norm(A[5, 6:9]), 30 * dt, atol=1e-6)
    print("SELFTEST OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["add", "stats", "selftest"])
    ap.add_argument("lance_dir", nargs="?")
    ap.add_argument("--fov", type=int, default=110)
    ap.add_argument("--splits", default="train,val,test,nogt")
    a = ap.parse_args()
    {"add": add, "stats": stats, "selftest": lambda a: _selftest()}[a.cmd](a)
