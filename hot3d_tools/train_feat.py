"""Train the LeWM predictor on FROZEN DINOv2 features (featwm.FeatWM). No encoder training, no SIGReg
(SIGReg regularizes the encoder, which is frozen here).

Samples = LeWM clips: 4 frames at frameskip 3 (rows s, s+3, s+6, s+9) whose raw rows [s, s+12) pass the
50 ms rule; actions rows s..s+11 -> (4, 45), normalized and masked exactly as in train.py
(utils.Hot3dActionTransform, valid-row stats of the train table, variant none / head / head_hands).
Loss = MSE(pred(z_0..2, a_0..2), z_1..3) in standardized feature space (num_preds = 1, as LeWM).
Hold-out: random 10% of clips (seed 3072, like train.py), plus the val TABLE loss each epoch.
All features are held on the GPU (1.4 GB for cls_grid fp16).

  python hot3d_tools/train_feat.py <feat_dir> <lance_dir> <out_dir> --version cls_grid --variant head
      [--epochs 40 --batch 256 --lr 3e-4]
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
from featwm import FeatWM, VERSIONS  # noqa: E402
from lewm_forecast import action_norm  # noqa: E402
from windows import valid_starts  # noqa: E402

FS, T = 3, 4


def table(lance_dir, split):
    import lance
    tab = lance.dataset(f"{lance_dir}/hot3d_{split}_fov110.lance").to_table(columns=["episode_idx", "t_sec", "action"])
    ep = np.asarray(tab["episode_idx"].to_pylist()).reshape(-1)
    t = np.asarray(tab["t_sec"].to_pylist(), np.float64).reshape(-1)
    act = np.asarray(tab["action"].to_pylist(), np.float32)
    starts = []
    for e in np.unique(ep):
        idx = np.flatnonzero(ep == e)
        ok = valid_starts(t[idx], T * FS)
        starts.append(idx[:len(ok)][ok])
    return np.concatenate(starts), act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feat_dir")
    ap.add_argument("lance_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--version", default="cls_grid", choices=list(VERSIONS))
    ap.add_argument("--variant", default="head")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=3072)
    a = ap.parse_args()
    from utils import Hot3dActionTransform
    dev = "cuda"
    torch.manual_seed(a.seed)
    Dm = VERSIONS[a.version]
    data = {}
    for split in ["train", "val"]:
        rows = np.load(f"{a.feat_dir}/rows_{split}.npy")
        F = np.load(f"{a.feat_dir}/emb_{split}.npy", mmap_mode="r")[:, :Dm]
        assert (rows == np.arange(len(rows))).all(), "need stride-1 features"
        starts, act = table(a.lance_dir, split)
        data[split] = dict(F=torch.from_numpy(np.ascontiguousarray(F)).to(dev), starts=starts, act_raw=act)
        print(f"{split}: {len(rows)} frames, {len(starts)} clips", flush=True)
    Ftr = data["train"]["F"].float()
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    del Ftr
    am, asd = action_norm(a.lance_dir)
    tf = Hot3dActionTransform(am, asd, a.variant)
    for d in data.values():       # the transform is per raw row (15 numbers): normalize every row once
        d["act"] = tf(torch.from_numpy(d.pop("act_raw"))).to(dev)                   # [N,15]
    model = FeatWM(version=a.version).to(dev)
    model.mu.copy_(mu); model.sd.copy_(sd)
    st = data["train"]["starts"]
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(st))
    nho = len(st) // 10
    tr_idx, ho_idx = st[perm[nho:]], st[perm[:nho]]

    def batch(split, s):
        d = data[split]
        s = torch.from_numpy(s).to(dev)
        fr = s[:, None] + FS * torch.arange(T, device=dev)[None]                    # [B,4]
        z = (d["F"][fr].float() - mu) / sd
        ar = s[:, None] + torch.arange(T * FS, device=dev)[None]                     # [B,12]
        acts = d["act"][ar].reshape(len(s), T, FS * 15)
        return z, acts

    def loss_of(z, acts):
        ae = model.action_encoder(acts)
        pred = model.predict_std(z[:, :3], ae[:, :3])
        return (pred - z[:, 1:]).pow(2).mean()

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_ep = len(tr_idx) // a.batch
    total = steps_ep * a.epochs
    warm = min(2000, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(i, total) / total)))

    @torch.no_grad()
    def evaluate(split, idx, n=20000):
        model.eval()
        idx = idx[np.random.default_rng(0).permutation(len(idx))[:n]]
        ls = []
        for b in range(0, len(idx), 1024):
            z, acts = batch(split, idx[b:b + 1024])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ls.append(loss_of(z, acts).float().item() * len(z))
        model.train()
        return sum(ls) / len(idx)

    hist, t0 = [], time.time()
    print(f"version {a.version} variant {a.variant}: {len(tr_idx)} train clips, {steps_ep} steps/epoch, {a.epochs} epochs", flush=True)
    for ep in range(a.epochs):
        p = np.random.default_rng(a.seed + ep).permutation(len(tr_idx))
        run = 0.0
        for i in range(steps_ep):
            z, acts = batch("train", tr_idx[p[i * a.batch:(i + 1) * a.batch]])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_of(z, acts)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item()
        h = dict(epoch=ep + 1, train=run / steps_ep, holdout=evaluate("train", ho_idx),
                 val_table=evaluate("val", data["val"]["starts"]), sec=time.time() - t0)
        hist.append(h)
        print(f"epoch {ep + 1}/{a.epochs} train {h['train']:.4f} holdout {h['holdout']:.4f} val-table {h['val_table']:.4f} "
              f"({h['sec']:.0f}s)", flush=True)
        if (ep + 1) % 10 == 0 or ep + 1 == a.epochs:
            model.save(a.out_dir)
    json.dump(dict(args=vars(a), hist=hist), open(f"{a.out_dir}/train_hist.json", "w"), indent=1)
    print("TRAIN FEAT OK", a.out_dir)


if __name__ == "__main__":
    main()
