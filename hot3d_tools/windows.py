"""Time-gap rule for HOT3D clips and eval windows.

Rule (user, 2026-11-26): skip any training clip or eval gap window whose raw
frames contain a time step > 50 ms (computed from `t_sec`). Some recordings
have RGB drops of up to 200 ms.

A window = raw rows [start, start + span) of one recording. For swm clips,
span = num_steps * frameskip (same as swm's Dataset.clip_indices).

Use in training:
    ds = swm.data.load_dataset(name, num_steps=4, frameskip=3, ...)
    keep = swm_valid_indices(ds, t_sec_of_table)        # t_sec column, all rows
    ds = torch.utils.data.Subset(ds, keep)

Report (lewm env):
    python hot3d_tools/windows.py <lance_dir> [--fov 110] [--spans 12,30,60,90]
"""
import argparse

import numpy as np

MAX_DT_S = 0.050


def valid_starts(t_sec, span, max_dt=MAX_DT_S):
    """For one recording: bool array, True where window [s, s+span) has no step > max_dt."""
    t = np.asarray(t_sec, np.float64).reshape(-1)
    n = len(t) - span + 1
    if n <= 0:
        return np.zeros(0, bool)
    bad = np.concatenate([[0], np.cumsum(np.diff(t) > max_dt)])  # bad[i] = # bad steps in rows 0..i
    # steps inside the window are (s -> s+1) ... (s+span-2 -> s+span-1)
    return (bad[span - 1: span - 1 + n] - bad[:n]) == 0


def swm_valid_indices(ds, t_sec_all, max_dt=MAX_DT_S):
    """Indices into an swm dataset (its clip_indices) that pass the rule.
    t_sec_all: the table's t_sec column for all rows, in table order."""
    t = np.asarray(t_sec_all, np.float64).reshape(-1)
    ok = {}
    for ep, (off, L) in enumerate(zip(ds.offsets, ds.lengths)):
        ok[ep] = valid_starts(t[off: off + L], ds.span, max_dt)
    return [i for i, (ep, s) in enumerate(ds.clip_indices) if ok[ep][s]]


if __name__ == "__main__":
    import lance

    ap = argparse.ArgumentParser()
    ap.add_argument("lance_dir")
    ap.add_argument("--fov", type=int, default=110)
    ap.add_argument("--spans", default="12,30,60,90")
    a = ap.parse_args()
    spans = [int(x) for x in a.spans.split(",")]
    for split in ["train", "val", "test", "nogt"]:
        tab = lance.dataset(f"{a.lance_dir}/hot3d_{split}_fov{a.fov}.lance").to_table(
            columns=["episode_idx", "t_sec"]).to_pydict()
        ep = np.asarray(tab["episode_idx"])
        t = np.asarray(tab["t_sec"], np.float64).reshape(-1)
        cuts = np.flatnonzero(np.diff(ep)) + 1
        eps = np.split(t, cuts)
        n_gap_recs = sum((np.diff(e) > MAX_DT_S).any() for e in eps)
        line = [f"{split}: {len(eps)} recs ({n_gap_recs} with a step > 50 ms)"]
        for span in spans:
            v = [valid_starts(e, span) for e in eps]
            tot = sum(len(x) for x in v)
            rm = tot - sum(int(x.sum()) for x in v)
            line.append(f"span {span}: removes {rm}/{tot} ({100 * rm / max(tot, 1):.2f}%)")
        print(" | ".join(line))
