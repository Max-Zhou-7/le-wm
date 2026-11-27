"""RQ2 (val only): surprise vs frames skipped, and surprise-based vs fixed-rate capture at equal budgets.

Model: a LeWM checkpoint + its stride-1 val embeddings (head-only variant by default). 10 fps timeline
(every 3rd raw row). Surprise of frame t = ||z_hat_t - z_t|| / zscale, z_hat_t = LeWM forecast from the
3 previous timeline entries and the GT head actions (zscale = mean ||z - mean z|| over val, the same
normalization as gap_eval's emb_err).

Part 1 (anchors every 15 rows, as gap_eval): open-loop rollout k = 1..8 steps (100-800 ms skipped);
surprise vs k, plus repeat-last distance ||z_t - z_t+k||.

Part 2 (streaming): each recording is cut into segments at any raw time step > 50 ms. The first 3
frames of a segment are always kept. At every later step t the device holds buf[t-3..t-1] (real z
for kept frames, its own forecasts for skipped ones), forecasts z_hat_t, then decides whether to keep
frame t:
  fixed N      keep every N-th frame
  surprise     keep if surprise(t) > tau   (needs z_t: "encode every frame, store/transmit few")
  head-motion  keep if the head rotation accumulated since the last kept frame > theta deg (from the
               head action; needs no image)
Kept: buf[t] = z_t; skipped: buf[t] = z_hat_t. tau / theta are set so the kept fraction matches
each fixed-rate budget (grid pass, then interpolation, then a final pass).
Scores over all steps t >= 3 (kept frames count as 0 error):
  feature: ||buf_t - z_t|| / zscale (LeWM forecast fills skipped frames); also with repeat-last-kept z
  pixels:  MAE (/255) between the real frame t and the last kept frame (repeat-last-kept rendering:
           there is no decoder for the LeWM forecast). Whole 224 image incl. the fixed black corners.
NOTE: feature-space distance is LeWM's own training objective, so it favours LeWM-based fills.

  python hot3d_tools/rq2.py <policy> <variant> <emb_dir> <lance_dir> <out_prefix>
      out: <out_prefix>_surprise_vs_skip_val.png, _capture_val.png, _timeline_val.png, <emb_dir>/rq2_val.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from gap_eval import FS, anchors, load_split  # noqa: E402
from lewm_forecast import Forecaster, raw_actions  # noqa: E402
from windows import valid_starts  # noqa: E402

FIXED_N = [2, 3, 4, 6, 8]


def segments(D):
    """List of arrays of raw rows on the 10 fps timeline; consecutive entries are 3 rows apart and every
    window [r_j, r_j + 3] has no raw step > 50 ms (so action j = rows r_j..r_j+2 is clean)."""
    segs = []
    for e in np.unique(D["ep"]):
        idx = np.flatnonzero(D["ep"] == e)
        ok = valid_starts(D["t"][idx], FS + 1)            # window rows s..s+3
        cur = [idx[0]]
        for s in range(0, len(idx) - FS, FS):
            if ok[s]:
                cur.append(idx[s + FS])
            else:
                if len(cur) >= 4:
                    segs.append(np.array(cur))
                cur = [idx[s + FS]]
        if len(cur) >= 4:
            segs.append(np.array(cur))
    return segs


class Stream:
    def __init__(self, F, D, Z, segs, zscale):
        self.F, self.zscale = F, zscale
        S, J = len(segs), max(len(s) for s in segs)
        self.S, self.J = S, J
        self.L = np.array([len(s) for s in segs])
        self.rows = np.zeros((S, J), int)
        Dm = Z.shape[1]
        self.Zs = torch.zeros(S, J, Dm)
        A = F.model.action_encoder.embed[-1].out_features
        self.AE = torch.zeros(S, J, A)
        self.rot = np.zeros((S, J))                       # head rotation (deg) of step j -> j+1
        for i, sg in enumerate(segs):
            self.rows[i, :len(sg)] = sg
            self.Zs[i, :len(sg)] = torch.from_numpy(Z[sg])
            acts = D["action"][sg[:, None] + np.arange(FS)[None]]           # [n,3,15]
            hv = acts[..., 12] > 0.5
            self.rot[i, :len(sg)] = (np.linalg.norm(acts[..., 3:6], axis=-1) * hv).sum(1)
            a45 = acts.reshape(len(sg), 1, FS * 15)
            with torch.no_grad():
                self.AE[i, :len(sg)] = F.model.action_encoder(F.norm(a45).to(F.dev))[:, 0].cpu()

    @torch.no_grad()
    def run(self, configs, batch=4096):
        """configs: list of (kind, param). Returns dict of per-config arrays [C,S,J]."""
        C, S, J = len(configs), self.S, self.J
        dev = self.F.dev
        Zs, AE = self.Zs.to(dev), self.AE.to(dev)
        buf = Zs[:, :3].unsqueeze(0).repeat(C, 1, 1, 1)                     # [C,S,3,D] rolling
        keep = np.zeros((C, S, J), bool); keep[:, :, :3] = True
        lastk = np.zeros((C, S, J), int); lastk[:, :, :3] = np.arange(3)
        surpr = np.full((C, S, J), np.nan)
        ferr = np.zeros((C, S, J))
        rerr = np.zeros((C, S, J))                                          # repeat-last-kept z error
        acc = np.zeros((C, S))
        kind = np.array([k for k, _ in configs]); par = np.array([p for _, p in configs], float)
        for t in range(3, J):
            alive = t < self.L                                              # [S]
            if not alive.any():
                break
            si = np.flatnonzero(alive)
            x = buf[:, si].reshape(-1, 3, buf.shape[-1])
            c = AE[si, t - 3:t].unsqueeze(0).expand(C, -1, -1, -1).reshape(-1, 3, AE.shape[-1])
            pred = torch.cat([self.F.model.predict(x[b:b + batch], c[b:b + batch])[:, -1]
                              for b in range(0, len(x), batch)]).reshape(C, len(si), -1)
            zt = Zs[si, t].unsqueeze(0)
            s = (torch.linalg.norm(pred - zt, dim=-1) / self.zscale).cpu().numpy()     # [C,n]
            acc[:, si] += self.rot[si, t - 1][None]
            k = np.zeros((C, len(si)), bool)
            m = kind == "fixed"; k[m] = np.broadcast_to(((t % par[m].astype(int)) == 0)[:, None], (m.sum(), len(si)))
            m = kind == "surprise"; k[m] = s[m] > par[m][:, None]
            m = kind == "head"; k[m] = acc[m][:, si] > par[m][:, None]
            new = torch.where(torch.from_numpy(k).to(dev)[..., None], zt.expand_as(pred), pred)
            buf[:, si] = torch.cat([buf[:, si, 1:], new[:, :, None]], 2)
            keep[:, si, t] = k
            lastk[:, si, t] = np.where(k, t, lastk[:, si, t - 1])
            surpr[:, si, t] = s
            ferr[:, si, t] = np.where(k, 0, s)
            lk = lastk[:, si, t]
            zl = Zs[torch.from_numpy(np.broadcast_to(si, lk.shape).copy()).long(), torch.from_numpy(lk).long()]
            rerr[:, si, t] = (torch.linalg.norm(zl - zt, dim=-1) / self.zscale).cpu().numpy()
            acc[:, si] = np.where(k, 0, acc[:, si])
        valid = np.arange(J)[None] < self.L[:, None]
        valid[:, :3] = False
        return dict(keep=keep, lastk=lastk, surpr=surpr, ferr=ferr, rerr=rerr, valid=valid)


def frac(R):
    v = R["valid"]
    return np.array([R["keep"][c][v].mean() for c in range(len(R["keep"]))])


def pix_cache(D, rows_needed):
    import lance  # noqa: F401
    from figs import frames
    rows_needed = np.unique(rows_needed)
    imgs = {}
    for b in range(0, len(rows_needed), 512):
        rr = rows_needed[b:b + 512]
        for r, im in zip(rr, frames(D, rr)):
            imgs[int(r)] = im
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("variant")
    ap.add_argument("emb_dir")
    ap.add_argument("lance_dir")
    ap.add_argument("out_prefix")
    ap.add_argument("--grid", type=int, default=40)
    a = ap.parse_args()
    t0 = time.time()
    D = load_split(a.lance_dir, "val", 110, extra=["action"])
    Z = np.load(f"{a.emb_dir}/emb_val.npy").astype(np.float32)
    assert len(Z) == D["N"]
    zscale = float(np.linalg.norm(Z - Z.mean(0), axis=1).mean())
    F = Forecaster(a.policy, a.variant, a.lance_dir)
    rep = dict(policy=a.policy, variant=a.variant, split="val", zscale=zscale,
               note="feature-space distance is LeWM's own training objective, so it favours LeWM-based fills")

    # Part 1
    K = 8
    R = anchors(D, 15)
    zh = np.stack([Z[R - 2 * FS], Z[R - FS], Z[R]], 1)
    zf = F.rollout(zh, raw_actions(D, R, K), K)                             # [n,K,D]
    true = np.stack([Z[R + FS * k] for k in range(1, K + 1)], 1)
    sur = np.linalg.norm(zf - true, axis=-1) / zscale
    rep_ = np.linalg.norm(Z[R][:, None] - true, axis=-1) / zscale
    from actions import mat_to_rotvec
    turn = np.array([[np.degrees(np.linalg.norm(mat_to_rotvec(D["Rwc"][r].T @ D["Rwc"][r + FS * k]))) for k in range(1, K + 1)]
                     for r in R])
    rep["part1"] = dict(n=int(len(R)), k=list(range(1, K + 1)),
                        surprise_mean=sur.mean(0).tolist(), surprise_median=np.median(sur, 0).tolist(),
                        surprise_p25=np.percentile(sur, 25, 0).tolist(), surprise_p75=np.percentile(sur, 75, 0).tolist(),
                        repeat_mean=rep_.mean(0).tolist(),
                        corr_surprise_turn=[float(np.corrcoef(sur[:, k], turn[:, k])[0, 1]) for k in range(K)])
    print("part1 surprise mean by k:", np.round(sur.mean(0), 3), "repeat:", np.round(rep_.mean(0), 3),
          f"({time.time() - t0:.0f}s)", flush=True)

    # Part 2
    segs = segments(D)
    St = Stream(F, D, Z, segs, zscale)
    print(f"part2: {len(segs)} segments, {int(sum(St.L))} timeline frames", flush=True)
    fixed = [("fixed", n) for n in FIXED_N]
    Rf = St.run(fixed)
    ffrac = frac(Rf)
    s_all = Rf["surpr"][0][Rf["valid"]]
    taus = np.quantile(s_all, np.linspace(0.02, 0.98, a.grid))
    thetas = np.quantile(St.rot[St.rot > 0], np.linspace(0.02, 0.98, a.grid)) * np.linspace(0.5, 12, a.grid)
    grid = [("surprise", float(x)) for x in taus] + [("head", float(x)) for x in thetas]
    Rg = St.run(grid)
    gfrac = frac(Rg)
    print(f"grid pass done ({time.time() - t0:.0f}s): surprise frac {gfrac[:a.grid].min():.3f}-{gfrac[:a.grid].max():.3f}, "
          f"head frac {gfrac[a.grid:].min():.3f}-{gfrac[a.grid:].max():.3f}", flush=True)
    final = list(fixed)
    for kind, lo in [("surprise", 0), ("head", a.grid)]:
        p = np.array([g[1] for g in grid[lo:lo + a.grid]]); fr = gfrac[lo:lo + a.grid]
        o = np.argsort(fr)
        for f in ffrac:
            final.append((kind, float(np.interp(f, fr[o], p[o]))))
    Rr = St.run(final)
    fr = frac(Rr)
    # pixels: real t vs last kept
    v = Rr["valid"]
    need = set()
    for c in range(len(final)):
        need.update(St.rows[v].tolist())
        need.update(St.rows[np.nonzero(v)[0], Rr["lastk"][c][v]].tolist())
    imgs = pix_cache(D, np.array(sorted(need)))
    print(f"decoded {len(imgs)} frames ({time.time() - t0:.0f}s)", flush=True)
    cache = {}

    def mae(r1, r2):
        if r1 == r2:
            return 0.0
        key = (r1, r2)
        if key not in cache:
            cache[key] = float(np.abs(imgs[r1].astype(np.int16) - imgs[r2].astype(np.int16)).mean()) / 255
        return cache[key]
    res = []
    si, ti = np.nonzero(v)
    for c, (kind, par) in enumerate(final):
        lk = Rr["lastk"][c][si, ti]
        pe = np.array([mae(int(St.rows[s, t]), int(St.rows[s, l])) for s, t, l in zip(si, ti, lk)])
        skipped = ~Rr["keep"][c][v]
        res.append(dict(kind=kind, param=par, kept_frac=float(fr[c]), budget=f"1/{FIXED_N[c % len(FIXED_N)]}",
                        feat_err_lewm=float(Rr["ferr"][c][v].mean()), feat_err_repeat=float(Rr["rerr"][c][v].mean()),
                        feat_err_lewm_skipped=float(Rr["ferr"][c][v][skipped].mean()) if skipped.any() else 0.0,
                        pix_mae_repeat=float(pe.mean()), pix_mae_repeat_skipped=float(pe[skipped].mean()) if skipped.any() else 0.0))
        r = res[-1]
        print(f"{kind:9s} {r['budget']:4s} param {par:8.3f} kept {r['kept_frac']:.3f} | feature err LeWM fill {r['feat_err_lewm']:.4f} "
              f"repeat fill {r['feat_err_repeat']:.4f} | pixel MAE x255 {255 * r['pix_mae_repeat']:.2f}", flush=True)
    rep["part2"] = dict(n_segments=len(segs), n_frames=int(v.sum()), results=res)
    # surprise vs frames since last kept (fixed N = 8 run)
    c8 = FIXED_N.index(8)
    gap_since = (ti - Rr["lastk"][c8][si, ti - 1])
    s8 = Rr["surpr"][c8][si, ti]
    rep["part2"]["surprise_vs_since_kept_N8"] = {int(g): float(s8[gap_since == g].mean()) for g in np.unique(gap_since)}
    json.dump(rep, open(f"{a.emb_dir}/rq2_val.json", "w"), indent=1)

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = np.arange(1, K + 1) * 100
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    ax[0].plot(ks, sur.mean(0), "o-", label="LeWM forecast (mean)")
    ax[0].fill_between(ks, np.percentile(sur, 25, 0), np.percentile(sur, 75, 0), alpha=0.25, label="LeWM 25-75%")
    ax[0].plot(ks, rep_.mean(0), "s--", color="gray", label="repeat last frame (mean)")
    ax[0].set_xlabel("frames skipped (ms)"); ax[0].set_ylabel("surprise ||z_hat - z|| / zscale")
    ax[0].set_title(f"Surprise vs gap, {len(R)} val windows", fontsize=9); ax[0].legend(fontsize=7)
    kk = 3
    ax[1].scatter(turn[:, kk], sur[:, kk], s=3, alpha=0.3)
    ax[1].set_xlabel(f"head turn during the gap (deg), k = {kk + 1} (400 ms)"); ax[1].set_ylabel("surprise")
    ax[1].set_title(f"Surprise vs head turn at 400 ms (r = {rep['part1']['corr_surprise_turn'][kk]:.2f})", fontsize=9)
    g = sorted(rep["part2"]["surprise_vs_since_kept_N8"])
    ax[2].plot(g, [rep["part2"]["surprise_vs_since_kept_N8"][x] for x in g], "o-")
    ax[2].set_xlabel("frames since the last kept frame (streaming, keep 1 in 8)"); ax[2].set_ylabel("surprise")
    ax[2].set_title("Streaming: surprise grows while frames are skipped", fontsize=9)
    fig.suptitle(f"RQ2 surprise, LeWM {a.variant} ({a.policy}), val. Feature distance = LeWM's own objective (favours LeWM).", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{a.out_prefix}_surprise_vs_skip_val.png", dpi=110)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    cols = {"fixed": "tab:gray", "surprise": "tab:red", "head": "tab:blue"}
    names = {"fixed": "fixed rate", "surprise": "surprise trigger (needs z_t)", "head": "head-motion trigger"}
    for kind in ["fixed", "surprise", "head"]:
        rr = sorted([r for r in res if r["kind"] == kind], key=lambda r: r["kept_frac"])
        x = [r["kept_frac"] for r in rr]
        ax[0].plot(x, [r["feat_err_lewm"] for r in rr], "o-", color=cols[kind], label=names[kind])
        ax[1].plot(x, [r["feat_err_repeat"] for r in rr], "o-", color=cols[kind], label=names[kind])
        ax[2].plot(x, [255 * r["pix_mae_repeat"] for r in rr], "o-", color=cols[kind], label=names[kind])
    for j, (t_, yl) in enumerate([("feature error, LeWM forecast fills skipped frames", "||z_fill - z|| / zscale"),
                                  ("feature error, last kept frame fills", "||z_fill - z|| / zscale"),
                                  ("pixel error, last kept frame fills", "MAE (/255)")]):
        ax[j].set_xlabel("fraction of frames kept (budget)"); ax[j].set_ylabel(yl); ax[j].set_title(t_, fontsize=9)
        ax[j].legend(fontsize=7); ax[j].set_xscale("log")
    fig.suptitle(f"RQ2 capture at equal budgets, val ({rep['part2']['n_frames']} frames at 10 fps). "
                 "Feature error favours LeWM (its own objective); pixel error is neutral.", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{a.out_prefix}_capture_val.png", dpi=110)

    # timeline example: longest segment, budget 1/4
    s0 = int(np.argmax(St.L)); n4 = FIXED_N.index(4)
    cf = n4; cs = len(FIXED_N) + n4
    L0 = min(St.L[s0], 300)
    fig, ax = plt.subplots(2, 1, figsize=(15, 5), sharex=True)
    tt = np.arange(3, L0) / 10
    for j, (c, nm) in enumerate([(cf, "fixed 1 in 4"), (cs, f"surprise trigger (kept {fr[cs]:.2f})")]):
        ax[j].plot(tt, Rr["surpr"][c][s0, 3:L0], lw=0.8, label="surprise of the frame (vs forecast)")
        ax[j].plot(tt, Rr["ferr"][c][s0, 3:L0], lw=0.8, label="error after the decision")
        kk_ = np.flatnonzero(Rr["keep"][c][s0, 3:L0])
        ax[j].plot(tt[kk_], np.zeros(len(kk_)), "|", color="k", ms=10, label="kept frame")
        ax[j].set_ylabel("/ zscale"); ax[j].set_title(nm, fontsize=9); ax[j].legend(fontsize=7, loc="upper right")
    ax[1].set_xlabel(f"time (s), val {D['rec'][St.rows[s0, 0]]}")
    fig.tight_layout(); fig.savefig(f"{a.out_prefix}_timeline_val.png", dpi=110)
    print("SAVED figures", a.out_prefix, f"({time.time() - t0:.0f}s)")
    print("RQ2 OK")


if __name__ == "__main__":
    main()
