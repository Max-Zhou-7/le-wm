"""LeWM forecasts on HOT3D (method M) + the prediction-example figure.

Rollout (same as jepa.JEPA.rollout, history 3 at 10 fps):
  embeddings z at rows r-6, r-3, r (frozen encoder+projector = the stride-1 emb_<split>.npy of THIS
  checkpoint) and actions a_j for 10 fps steps j = -2 .. K-1, where a_j = raw action rows
  r+3j, r+3j+1, r+3j+2 (45 numbers = motion from frame r+3j to r+3j+3). The predictor maps
  (z_j, a_j) for the last 3 steps to z_{j+1}; forecasts are fed back autoregressively.
Actions during the gap come from the GT poses (head: known from SLAM/IMU; hands: ORACLE hand
motion, which a real system would not have during a dropped-frame gap).
Actions are normalized exactly as in training (utils.Hot3dActionTransform, stats from the
valid rows of the train table) and masked by the variant (none / head / head_hands).

  python hot3d_tools/lewm_forecast.py example <policy> <variant> <emb_dir> <lance_dir> <out.png>
      [--recording P...] [--anchor-row N]
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from gap_eval import FS, Probes, anchors, load_split, to_cam  # noqa: E402

PART_NAMES = ["head t (cm)", "head rot (deg)", "L wrist (cm)", "R wrist (cm)", "flags h/L/R"]


def action_norm(lance_dir, fov=110):
    """Train-table stats, same as utils.get_hot3d_action_transform."""
    import lance
    act = np.asarray(lance.dataset(f"{lance_dir}/hot3d_train_fov{fov}.lance").to_table(columns=["action"])
                     ["action"].to_pylist(), np.float32)
    mean, std = np.zeros(12, np.float32), np.ones(12, np.float32)
    for part, (lo, hi) in enumerate([(0, 6), (6, 9), (9, 12)]):
        v = act[act[:, 12 + part] > 0.5, lo:hi]
        mean[lo:hi], std[lo:hi] = v.mean(0), np.maximum(v.std(0, ddof=1), 1e-6)
    return torch.from_numpy(mean), torch.from_numpy(std)


def raw_actions(D, R, K):
    """[n, 3+K-1? no: steps j=-2..K-1 -> 2+K steps, 45] raw (unnormalized) actions."""
    steps = np.arange(-2, K)
    rows = R[:, None, None] + FS * steps[None, :, None] + np.arange(FS)[None, None, :]   # [n,S,3]
    return D["action"][rows].reshape(len(R), len(steps), FS * 15)


class Forecaster:
    def __init__(self, policy, variant, lance_dir, dev=None):
        from encode import load_model
        from utils import Hot3dActionTransform
        self.dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(policy).to(self.dev).eval()
        mean, std = action_norm(lance_dir)
        self.tf = Hot3dActionTransform(mean, std, variant)

    def norm(self, a):
        n, S, A = a.shape
        return self.tf(torch.from_numpy(np.asarray(a, np.float32)).reshape(n * S, A)).reshape(n, S, A)

    @torch.no_grad()
    def rollout(self, z_hist, acts, K, batch=512):
        """z_hist [n,3,D], acts raw [n,2+K,45] -> forecasts [n,K,D] (k = 1..K)."""
        out = []
        for b in range(0, len(z_hist), batch):
            z = torch.from_numpy(np.asarray(z_hist[b:b + batch], np.float32)).to(self.dev)
            a = self.norm(acts[b:b + batch]).to(self.dev)
            ae = self.model.action_encoder(a)                     # [B, 2+K, A]
            preds = []
            for k in range(K):
                p = self.model.predict(z[:, -3:], ae[:, k:k + 3])[:, -1:]   # uses a_{k-2..k}
                z = torch.cat([z, p], 1)
                preds.append(p)
            out.append(torch.cat(preds, 1).float().cpu().numpy())
        return np.concatenate(out)


def fmt_action(a45):
    lines = []
    for sub in range(3):
        a = a45[sub * 15:(sub + 1) * 15]
        f = a[12:15]
        part = lambda lo, ok: ("[" + " ".join(f"{x:+5.2f}" for x in a[lo:lo + 3]) + "]") if ok else "[   invalid   ]"
        lines.append(f"  +{sub}/30s  head t {part(0, f[0])} rot {part(3, f[0])}  L {part(6, f[1])}  "
                     f"R {part(9, f[2])}  flags {int(f[0])}{int(f[1])}{int(f[2])}")
    return lines


def example(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figs import frames, show
    K, KS = 8, [1, 4, 8]
    D = load_split(a.lance_dir, "val", 110, extra=["action"])
    Z = np.load(f"{a.emb_dir}/emb_val.npy").astype(np.float32)
    assert len(Z) == D["N"], "need stride-1 val embeddings of this checkpoint"
    probes = Probes(f"{a.emb_dir}/probes.npz")
    R = anchors(D, 15)
    if a.anchor_row >= 0:
        r = a.anchor_row
    else:
        # a window with both hands valid at every target and clear hand motion
        cand = [x for x in R if (not a.recording or D["rec"][x] == a.recording)
                and all(D["hav"][x + FS * k].all() for k in KS)]
        mot = [np.linalg.norm(D["wrist_w"][x + FS * 8] - D["wrist_w"][x], axis=-1).sum() for x in cand]
        r = cand[int(np.argsort(mot)[len(mot) * 3 // 4])]   # 75th percentile of hand motion
    F = Forecaster(a.policy, a.variant, a.lance_dir)
    acts = raw_actions(D, np.array([r]), K)                  # [1, 2+K, 45]
    zf = F.rollout(Z[[r - 2 * FS, r - FS, r]][None], acts, K)[0]   # [K, D]
    zrep = Z[r]
    # nearest real val frame to each forecast (L2 in embedding space)
    Zt = torch.from_numpy(Z)

    def nearest(z):
        d = torch.cdist(torch.from_numpy(z[None]), Zt)[0]
        i = int(torch.argmin(d))
        return i, float(d[i])

    rec, t0 = D["rec"][r], D["t"][r]
    txt = [f"val {rec}, anchor row {r} (t = {t0:.2f} s). Variant {a.variant}. Policy {a.policy}.",
           "Actions fed to LeWM (raw units; each 10 fps step = 3 x 30 fps sub-steps; camera frame of the",
           "sub-step's start; x right / y down / z forward of the UNROTATED image). Steps -2..0 = history.",
           "Hand motion during the gap is GT (oracle)."]
    for j in range(acts.shape[1]):
        txt.append(f"step {j - 2:+d} (frames {r + FS * (j - 2)} -> {r + FS * (j - 1)}):")
        txt += fmt_action(acts[0, j])
    txt.append("")
    txt.append("Wrist positions in camera(target) frame, cm  [x, y, z]   (error = |pred - true|)")
    gt_all, rows_img = {}, {}
    fig = plt.figure(figsize=(15, 19))
    gs = fig.add_gridspec(5, 5, height_ratios=[1, 1, 1, 0.9, 1.6])
    imgs = frames(D, [r] + [r + FS * k for k in KS])
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(show(imgs[0])); ax.set_title(f"last real frame t\n(row {r})", fontsize=8); ax.axis("off")
    for c, k in enumerate(KS):
        tg = r + FS * k
        gt = to_cam(D, np.array([tg]), D["wrist_w"][[tg]])[0]
        ok = D["hav"][tg]
        lines = []
        for name, z in [("LeWM", zf[k - 1]), ("repeat-last", zrep), ("probe on real z_t+k", Z[tg])]:
            pw = probes.wrist(z[None])[0]
            e = np.linalg.norm(pw - gt, axis=-1)
            lines.append(f"  {name:20s} L {np.round(pw[0], 1)} ({e[0]:.1f})  R {np.round(pw[1], 1)} ({e[1]:.1f})")
        txt.append(f"k={k} ({100 * k} ms)  TRUE  L {np.round(gt[0], 1) if ok[0] else 'invalid'}  "
                   f"R {np.round(gt[1], 1) if ok[1] else 'invalid'}")
        txt += lines
        # images: real t+k, nearest to LeWM forecast, nearest to repeat-last
        nl, dl = nearest(zf[k - 1])
        nimg = frames(D, [nl])
        drep = np.linalg.norm(zrep - Z[tg])
        dtrue = np.linalg.norm(zf[k - 1] - Z[tg])
        lab = lambda i: (f"same rec, {D['t'][i] - D['t'][tg]:+.2f} s from t+{k}" if D["rec"][i] == rec
                         else f"other rec {D['rec'][i]}")
        for row, (im, title) in enumerate([
                (imgs[1 + c], f"real t+{k} ({100 * k} ms)"),
                (nimg[0], f"nearest val frame to LeWM forecast\n{lab(nl)}\nd={dl:.2f} (d to true z: {dtrue:.2f})"),
                (imgs[0], f"repeat-last = frame t\n(d to true z_t+{k}: {drep:.2f})")]):
            ax = fig.add_subplot(gs[row, 1 + c])
            ax.imshow(show(im)); ax.set_title(title, fontsize=7); ax.axis("off")
    axt = fig.add_subplot(gs[1:3, 0]); axt.axis("off")
    axt.text(0, 1, "columns: k = 1, 4, 8\nrow 1: real frame\nrow 2: nearest val frame\n  to LeWM forecast\n"
             "row 3: repeat-last (frame t;\n  its nearest frame is itself)\n\nFrames rotated 90 deg cw\nfor display.",
             fontsize=8, va="top", family="monospace")
    axw = fig.add_subplot(gs[3, 1:]); axw.axis("off")
    wl = txt[txt.index("Wrist positions in camera(target) frame, cm  [x, y, z]   (error = |pred - true|)"):]
    axw.text(0, 1, "\n".join(wl), fontsize=7, va="top", family="monospace")
    axa = fig.add_subplot(gs[4, :]); axa.axis("off")
    al = txt[:txt.index("Wrist positions in camera(target) frame, cm  [x, y, z]   (error = |pred - true|)")]
    axa.text(0, 1, "\n".join(al), fontsize=6.3, va="top", family="monospace")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    with open(a.out.rsplit(".", 1)[0] + ".txt", "w") as f:
        f.write("\n".join(txt) + "\n")
    print("\n".join(txt))
    print("SAVED", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["example"])
    ap.add_argument("policy")
    ap.add_argument("variant")
    ap.add_argument("emb_dir")
    ap.add_argument("lance_dir")
    ap.add_argument("out")
    ap.add_argument("--recording", default="")
    ap.add_argument("--anchor-row", type=int, default=-1)
    example(ap.parse_args())
