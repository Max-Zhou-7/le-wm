"""TimeWarp baseline (B2): rotation-only reprojection from the ORIGINAL fisheye frame at time t
into the 110 deg pinhole view of time t+g, using the known head rotation (H = K R K^-1 in the
pinhole case; here the source is the fisheye, so we unproject pinhole rays, rotate, and project
with the fisheye model). Head translation, hand and object motion are not compensated.

Fisheye624 projection is reimplemented in numpy (vectorized); `selftest` checks it against
projectaria_tools CameraCalibration.project (hot3d env).

  python hot3d_tools/timewarp.py selftest <fisheye.npz>                (hot3d env, PYTHONPATH=hot3d/hot3d)
  python hot3d_tools/timewarp.py identity <fisheye.npz> <lance_dir> <split> <recording>   (lewm env)
      warp with R = I and compare to the stored pinhole frames (should match ~1-2.5/255)
"""
import argparse
import math
import sys

import numpy as np
from PIL import Image


def fisheye624_project(P, params):
    """P [...,3] camera-frame points, params [15] = f, cx, cy, k0..k5, p0, p1, s0..s3
    (FisheyeRadTanThinPrism, single focal). Returns uv [...,2] (pixels, full res), ok [...]."""
    f, cx, cy = params[0], params[1], params[2]
    k = params[3:9]
    p0, p1 = params[9], params[10]
    s = params[11:15]
    z = P[..., 2]
    ok = z > 1e-9
    zs = np.where(ok, z, 1.0)
    a, b = P[..., 0] / zs, P[..., 1] / zs
    r = np.sqrt(a * a + b * b)
    th = np.arctan(r)
    th2 = th * th
    poly = np.ones_like(th)
    thp = np.ones_like(th)
    for ki in k:
        thp = thp * th2
        poly = poly + ki * thp
    thd = th * poly
    scale = np.where(r > 1e-12, thd / np.where(r > 1e-12, r, 1.0), 1.0)
    x, y = a * scale, b * scale
    rr = x * x + y * y
    # tangential
    t = 2 * (x * p0 + y * p1)
    xd = x * (t + 1) + rr * p0
    yd = y * (t + 1) + rr * p1
    # thin prism
    xd = xd + s[0] * rr + s[1] * rr * rr
    yd = yd + s[2] * rr + s[3] * rr * rr
    return np.stack([f * xd + cx, f * yd + cy], -1), ok


def pinhole_rays(fov, render):
    """Unit-z rays [render,render,3] of the linear camera used by preprocess_sequence.py
    (principal point (render-1)/2: checked by `identity`)."""
    f = (render / 2) / math.tan(math.radians(fov / 2))
    c = (render - 1) / 2
    u, v = np.meshgrid(np.arange(render), np.arange(render))
    return np.stack([(u - c) / f, (v - c) / f, np.ones_like(u, dtype=np.float64)], -1)


def bilinear(img, uv):
    """img [H,W,3] uint8, uv [...,2] pixel coords (pixel centers at integers). Out-of-image -> 0."""
    H, W = img.shape[:2]
    x, y = uv[..., 0], uv[..., 1]
    inside = (x >= -0.5) & (x <= W - 0.5) & (y >= -0.5) & (y <= H - 0.5)
    x = np.clip(x, 0, W - 1); y = np.clip(y, 0, H - 1)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1)
    wx, wy = (x - x0)[..., None], (y - y0)[..., None]
    im = img.astype(np.float32)
    out = (im[y0, x0] * (1 - wx) * (1 - wy) + im[y0, x1] * wx * (1 - wy)
           + im[y1, x0] * (1 - wx) * wy + im[y1, x1] * wx * wy)
    return out * inside[..., None]


class Warper:
    def __init__(self, fov=110, out_size=224, render=448):
        self.out_size, self.render = out_size, render
        self.rays = pinhole_rays(fov, render)

    def __call__(self, fish_img, params_full, full_size, R_src_dst=None, valid_radius=None, max_angle=None):
        """fish_img: fisheye image at time t (size x size, downsampled from full_size).
        R_src_dst: rotation taking directions in the target camera (t+g) into the source
        camera (t), i.e. R_c(t)<-c(t+g) = R_wc(t)^T R_wc(t+g). None = identity."""
        prm = np.asarray(params_full, np.float64)
        d = self.rays if R_src_dst is None else self.rays @ np.asarray(R_src_dst).T
        uv, ok = fisheye624_project(d, prm)
        # projectaria blacks out rays more than max_solid_angle (1 rad = 57.3 deg here) from the lens
        # axis, and outside the valid radius (1415 px, no effect): the stored frames' black corners
        # (~12%). Same here: the target view keeps its fixed corners (identity rays), and source
        # rays outside the limit are black too.
        def inside(q, uvq):
            m = q[..., 2] / np.linalg.norm(q, axis=-1) >= math.cos(max_angle) if max_angle else np.ones(q.shape[:-1], bool)
            if valid_radius is not None and np.isfinite(valid_radius):
                m &= ((uvq[..., 0] - prm[1]) ** 2 + (uvq[..., 1] - prm[2]) ** 2) <= valid_radius ** 2
            return m
        ok = ok & inside(d, uv)
        if R_src_dst is not None:
            ok &= inside(self.rays, fisheye624_project(self.rays, prm)[0])
        s = fish_img.shape[0] / full_size
        uv = (uv + 0.5) * s - 0.5                  # same as CameraCalibration.rescale (half-pixel rule)
        out = bilinear(fish_img, uv) * ok[..., None]
        out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        if self.render != self.out_size:
            out = np.asarray(Image.fromarray(out).resize((self.out_size, self.out_size), Image.BOX))
        return out


def selftest(path):
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from fisheye import FisheyeRecording
    from projectaria_tools.core.calibration import CameraCalibration, CameraModelType
    from projectaria_tools.core.sophus import SE3
    f = FisheyeRecording(path)
    d = f.d
    full = int(d["full_size"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for i in [0, len(f) // 2, len(f) - 1]:
        prm = d["proj_params"][i].astype(np.float64)
        T = np.eye(4); T[:3] = d["T_device_camera"][i]
        c = CameraCalibration(str(d["label"]), getattr(CameraModelType, str(d["model_type"])), prm,
                              SE3.from_matrix(T), full, full, float(d["valid_radius"][i]),
                              float(d["max_solid_angle"][i]), str(d["serial"]))
        # random directions within ~70 deg of the optical axis
        ang = np.radians(rng.uniform(0, 70, 2000)); az = rng.uniform(0, 2 * np.pi, 2000)
        P = np.stack([np.sin(ang) * np.cos(az), np.sin(ang) * np.sin(az), np.cos(ang)], -1) * rng.uniform(0.2, 3, (2000, 1))
        mine, _ = fisheye624_project(P, prm)
        ref = np.array([c.project_no_checks(p) for p in P])
        err = np.abs(mine - ref).max()
        worst = max(worst, err)
        print(f"frame {i}: model {d['model_type']} n_params {len(prm)} max |uv - projectaria| = {err:.2e} px")
    assert worst < 1e-3, worst
    print("SELFTEST OK")


def identity(a):
    import lance
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from fisheye import FisheyeRecording
    f = FisheyeRecording(a.fisheye)
    ds = lance.dataset(f"{a.lance_dir}/hot3d_{a.split}_fov110.lance")
    tab = ds.to_table(columns=["recording"])
    recs = np.asarray(tab["recording"].to_pylist())
    rows = np.flatnonzero(recs == a.recording)
    assert len(rows) == len(f), (len(rows), len(f))
    W = Warper()
    from torchvision.io import decode_jpeg
    import torch
    errs = []
    for i in np.linspace(0, len(f) - 1, 5).astype(int):
        blob = ds.take([int(rows[i])], columns=["pixels"])["pixels"].to_pylist()[0]
        ref = decode_jpeg(torch.frombuffer(bytearray(blob), dtype=torch.uint8)).permute(1, 2, 0).numpy()
        out = W(f.image(i), f.d["proj_params"][i], int(f.d["full_size"]), valid_radius=f.d["valid_radius"][i],
                max_angle=float(f.d["max_solid_angle"][i]))
        errs.append(np.abs(out.astype(np.float32) - ref).mean())
    print(f"identity warp vs stored Lance frames: mean abs err {np.round(errs, 2).tolist()} /255")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "identity"])
    ap.add_argument("fisheye")
    ap.add_argument("lance_dir", nargs="?")
    ap.add_argument("split", nargs="?")
    ap.add_argument("recording", nargs="?")
    a = ap.parse_args()
    selftest(a.fisheye) if a.cmd == "selftest" else identity(a)
