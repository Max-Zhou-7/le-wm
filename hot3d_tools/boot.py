"""Recording-level bootstrap 95% CIs for the gap eval and RQ2 (per-sample files from gap_eval.py / rq2.py).

Resampling unit = recording (windows / frames of one recording are not independent). B = 1000 resamples,
seed 0, the same resample weights for every number of one split, so paired differences are paired.
Every mean = mean over the pooled samples of the resampled recordings (sample-weighted, as the point value).
Paired differences: per sample (same window and hand / frame), method - baseline, where both are finite.
  gap eval: baseline = repeat last frame (B0_repeat; A_repeat for the all-gap-frames section)
  RQ2 capture: baseline = fixed rate at the same budget;  RQ2 surprise: LeWM surprise - repeat-last distance

  python hot3d_tools/boot.py <results_v2_dir> <out.md> [--split val] [--B 1000]
writes <results_v2_dir>/<NAME>/boot_<split>.json and the markdown tables.
"""
import argparse
import json
import os

import numpy as np

G = [1, 2, 4, 8, 16, 32]
MODELS = ["lewm_none", "lewm_head", "lewm_head_hands", "dino_best_none", "dino_best_head", "dino_best_head_hands"]


class Boot:
    def __init__(self, recs_all, B=1000, seed=0):
        self.names = sorted(set(recs_all))
        nr = len(self.names)
        self.W = np.random.default_rng(seed).multinomial(nr, np.full(nr, 1.0 / nr), size=B).astype(np.float64)

    def rid(self, rec):
        m = {n: i for i, n in enumerate(self.names)}
        return np.array([m[r] for r in rec])

    def stat(self, x, rid):
        """x [n] or [n,k] per window/frame; rid [n]. -> (mean, lo, hi, n)."""
        x = np.asarray(x, np.float64)
        r = np.broadcast_to(rid.reshape((-1,) + (1,) * (x.ndim - 1)), x.shape).reshape(-1)
        x = x.reshape(-1)
        m = np.isfinite(x)
        if not m.any():
            return None
        nr = len(self.names)
        S = np.bincount(r[m], x[m], minlength=nr); C = np.bincount(r[m], minlength=nr).astype(float)
        with np.errstate(all="ignore"):           # spurious matmul warnings on some BLAS builds
            WS, WC = self.W @ S, self.W @ C
        bs = WS[WC > 0] / WC[WC > 0]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return dict(mean=float(S.sum() / C.sum()), lo=float(lo), hi=float(hi), n=int(m.sum()))

    def diff(self, x, y, rid):
        x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
        return self.stat(np.where(np.isfinite(x) & np.isfinite(y), x - y, np.nan), rid)


def f(s, d=1, sign=False):
    if s is None:
        return "–"
    fmt = f"{{:{'+' if sign else ''}.{d}f}}"
    return f"{fmt.format(s['mean'])} [{fmt.format(s['lo'])}, {fmt.format(s['hi'])}]"


def gap_tables(name, S, bt, L, out):
    rid = bt.rid(S["rec"])
    keys = [k for k in S.files if "|" in k]
    get = lambda sec, m, g, met: S[f"{sec}|{m}|g{g}|{met}"] if f"{sec}|{m}|g{g}|{met}" in S.files else None
    meths = sorted({k.split("|")[1] for k in keys if k.startswith("methods|")})
    lm = [m for m in meths if m.startswith("M_lewm") and "noise" not in m]
    order = (["oracle", "B0_repeat", "B2_timewarp_fill", "B2_timewarp", "B1_linear", "hold_gt", "kalman_gt", "kalman_damped_gt",
              "kalman_probe", "kalman_damped_probe"] + lm + sorted(m for m in meths if "noise" in m))
    order = [m for m in order if m in meths]
    res = {}
    for sec, base in [("methods", "B0_repeat"), ("gapall", "A_repeat")]:
        ms = order if sec == "methods" else [m for m in ["A_oracle", "A_repeat", "A_lewm", "I_nearest", "I_linear", "I_lewm_endfix"]
                                             if any(k.startswith(f"gapall|{m}|") for k in keys)]
        for m in ms:
            for g in G:
                for met in ["wrist", "wrist_above_floor", "up", "emb", "pix"]:
                    x = get(sec, m, g, met)
                    if x is None:
                        continue
                    st = bt.stat(x, rid)
                    b = get(sec, base, g, met)
                    dd = bt.diff(x, b, rid) if (b is not None and m != base and met != "wrist_above_floor") else None
                    res[f"{sec}|{m}|g{g}|{met}"] = dict(stat=st, diff_vs_repeat=dd)
    json.dump(res, open(out, "w"), indent=1)
    R = lambda sec, m, g, met, which="stat": (res.get(f"{sec}|{m}|g{g}|{met}") or {}).get(which)
    hdr = "| method | " + " | ".join(f"{100 * g} ms" for g in G) + " |\n|---|" + "---|" * len(G)
    L.append(f"\n## {name}\n")
    for met, ttl, d in [("wrist", "Wrist error (cm), last frame of the gap: mean [95% CI]", 1),
                        ("wrist_above_floor", "Wrist error above the oracle floor (cm)", 1),
                        ("up", "Head tilt error (deg)", 1),
                        ("emb", "Feature error (relative; the predictor's own objective, favours LeWM)", 2)]:
        L.append(f"\n**{ttl}**\n\n" + hdr)
        for m in order:
            if R("methods", m, 1, met) is None or (met == "wrist_above_floor" and m == "oracle"):
                continue
            L.append(f"| {m} | " + " | ".join(f(R("methods", m, g, met), d) for g in G) + " |")
        if met == "wrist_above_floor":
            continue
        L.append(f"\n*Paired difference vs repeat-last ({met}; negative = better than repeat-last)*\n\n" + hdr)
        for m in order:
            if m == "B0_repeat" or R("methods", m, 1, met, "diff_vs_repeat") is None:
                continue
            L.append(f"| {m} | " + " | ".join(f(R("methods", m, g, met, "diff_vs_repeat"), d, True) for g in G) + " |")
    pm = [m for m in ["B0_repeat", "B2_timewarp_fill", "B2_timewarp"] if R("methods", m, 1, "pix")]
    if pm:
        L.append("\n**Pixel MAE (/255; every 20th window), and paired difference vs repeat-last**\n\n" + hdr)
        for m in pm:
            L.append(f"| {m} | " + " | ".join(f({k: v * 255 if k != 'n' else v for k, v in R('methods', m, g, 'pix').items()}, 1)
                                              for g in G) + " |")
        for m in pm[1:]:
            L.append(f"| {m} − repeat | " + " | ".join(
                f({k: v * 255 if k != 'n' else v for k, v in R('methods', m, g, 'pix', 'diff_vs_repeat').items()}, 1, True) for g in G) + " |")
    am = [m for m in ["A_oracle", "A_repeat", "A_lewm", "I_nearest", "I_linear", "I_lewm_endfix"] if R("gapall", m, 1, "wrist")]
    if am:
        for met, ttl, d in [("wrist", "Interpolation ablation: wrist error (cm) averaged over ALL dropped frames of the gap "
                                      "(next frame at (g+1) x 100 ms)", 1),
                            ("emb", "Interpolation ablation: feature error, all dropped frames (favours LeWM)", 2)]:
            L.append(f"\n**{ttl}**\n\n" + hdr)
            for m in am:
                L.append(f"| {m} | " + " | ".join(f(R("gapall", m, g, met), d) for g in G) + " |")
            L.append("\n*Paired difference vs A_repeat*\n\n" + hdr)
            for m in am:
                if m != "A_repeat":
                    L.append(f"| {m} | " + " | ".join(f(R("gapall", m, g, met, "diff_vs_repeat"), d, True) for g in G) + " |")
    return res


def rq2_tables(name, S, bt, L, out):
    res = {}
    r1 = bt.rid(S["p1_rec"])
    for i, g in enumerate(G):
        res[f"p1|g{g}|surprise"] = dict(stat=bt.stat(S["p1_surprise"][:, g - 1], r1),
                                        diff_vs_repeat=bt.diff(S["p1_surprise"][:, g - 1], S["p1_repeat"][:, g - 1], r1))
        res[f"p1|g{g}|repeat"] = dict(stat=bt.stat(S["p1_repeat"][:, g - 1], r1))
    r2 = bt.rid(S["p2_rec"])
    for k in S.files:
        if not k.startswith("p2|"):
            continue
        _, kind, bud, met = k.split("|")
        base = f"p2|fixed|{bud}|{met}"
        res[k] = dict(stat=bt.stat(S[k], r2), diff_vs_fixed=bt.diff(S[k], S[base], r2) if kind != "fixed" else None)
    json.dump(res, open(out, "w"), indent=1)
    L.append(f"\n### RQ2, {name}\n")
    L.append("Surprise vs gap (LeWM forecast vs real frame, / zscale; LeWM's own feature space), and the paired difference "
             "vs the repeat-last distance\n")
    L.append("| | " + " | ".join(f"{100 * g} ms" for g in G) + " |\n|---|" + "---|" * len(G))
    L.append("| surprise | " + " | ".join(f(res[f"p1|g{g}|surprise"]["stat"], 3) for g in G) + " |")
    L.append("| repeat-last distance | " + " | ".join(f(res[f"p1|g{g}|repeat"]["stat"], 3) for g in G) + " |")
    L.append("| surprise − repeat | " + " | ".join(f(res[f"p1|g{g}|surprise"]["diff_vs_repeat"], 3, True) for g in G) + " |")
    nm = {"fixed": "fixed rate", "surprise": "surprise ORACLE", "lastsurp": "surprise realistic", "head": "head motion"}
    L.append("\nCapture at equal budgets: value [95% CI]; Δ = paired difference vs fixed rate [95% CI] (negative = better)\n")
    L.append("| budget | policy | kept | feat err, forecast fills | Δ | pixel MAE /255, last kept fills | Δ |\n|---|---|---|---|---|---|---|")
    for bud in ["1/2", "1/4", "1/8"]:
        for kd in ["fixed", "surprise", "lastsurp", "head"]:
            k = f"p2|{kd}|{bud}"
            if f"{k}|feat_lewm" not in res:
                continue
            sc = lambda s: None if s is None else {a: (v * 255 if a != "n" else v) for a, v in s.items()}
            L.append(f"| {bud} | {nm[kd]} | {res[k + '|kept']['stat']['mean']:.3f} | {f(res[k + '|feat_lewm']['stat'], 3)} | "
                     f"{f(res[k + '|feat_lewm']['diff_vs_fixed'], 3, True)} | {f(sc(res[k + '|pix']['stat']), 1)} | "
                     f"{f(sc(res[k + '|pix']['diff_vs_fixed']), 1, True)} |")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("md")
    ap.add_argument("--split", default="val"); ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    L = [f"# Recording-level bootstrap, {a.split} ({a.B} resamples, 95% percentile CIs)\n"]
    first = True
    for m in MODELS:
        p = f"{a.root}/{m}/gap_eval_{a.split}_k32_samples.npz"
        if not os.path.exists(p):
            continue
        S = np.load(p)
        p2 = f"{a.root}/{m}/rq2_{a.split}_k32_samples.npz"
        S2 = np.load(p2) if os.path.exists(p2) else None
        bt = Boot(list(S["rec"]) + (list(S2["p1_rec"]) + list(S2["p2_rec"]) if S2 is not None else []), a.B)
        if first:
            L.append(f"{len(bt.names)} recordings, {len(S['rec'])} windows. Wide CIs come from few recordings.\n")
            first = False
        gap_tables(m, S, bt, L, f"{a.root}/{m}/boot_{a.split}.json")
        if S2 is not None:
            rq2_tables(m, S2, bt, L, f"{a.root}/{m}/boot_rq2_{a.split}.json")
        print("done", m, flush=True)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("SAVED", a.md)


if __name__ == "__main__":
    main()
