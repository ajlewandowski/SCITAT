"""
Step 4 - Score every subset against the lidar
=============================================
For each subset: load its pair DSMs onto the lidar grid, fuse them (per-cell median,
the usual multi-date satellite fusion), and compare with the lidar on the scoring
mask (valid lidar, away from the tile edge, water excluded).

Measured quantities, defined the same way as SCITAT's predictedErrorM:
  completeness  share of mask cells reconstructed with |error| < --blunder (default 5 m)
  sigma_z       RMS error of those cells (m) - height precision of reconstructed points
  err_rms       sqrt(C sigma_z^2 + (1 - C) miss^2), miss = --miss-penalty (default 10 m,
                must match the model's errMissM)
  also: raw RMSE, NMAD, median bias, best horizontal shift (cells), cells in mask

Alignment: the DFC2019 Track 3 RPCs were already adjusted to the lidar, so by default
only a vertical median bias is removed (--no-bias keeps it in; the model's systematic
floor errSysM then absorbs it). --shift-search N also searches whole-cell horizontal
shifts of up to N cells.

Failed pairs: by default a subset that uses a pair with no dsm.tif is skipped. With
--failed-as-empty, a pair that ran and failed (it has a log.txt) counts as reconstructing
nothing, so its subsets are scored with that pair contributing no points - a failure is
then part of the measured result. --exclude-images drops every subset that uses the
listed images (e.g. cloudy or snowy ones found with checkFailedPairs.py).

Output: <work>/measured.csv, and <work>/fused/<subset_id>.tif if --save-fused.

Run
  python scoreReconstructions.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dfc2019Common import LidarTile, toGrid, saveGrid, WORK_DIR


def bestShift(dz, n):
    """Whole-cell shift (dr, dc) within +/-n minimising the median |error|."""
    if n <= 0:
        return 0, 0
    best, arg = np.inf, (0, 0)
    for dr in range(-n, n + 1):
        for dc in range(-n, n + 1):
            v = np.roll(np.roll(dz, dr, 0), dc, 1)
            m = np.nanmedian(np.abs(v - np.nanmedian(v)))
            if m < best:
                best, arg = m, (dr, dc)
    return arg


def score(fused, lidar, mask, blunder, miss, removeBias=True, shift=0):
    dz0 = fused - lidar.z
    dr, dc = bestShift(np.where(mask, dz0, np.nan), shift) if shift else (0, 0)
    if dr or dc:
        fused = np.roll(np.roll(fused, dr, 0), dc, 1)
    dz = (fused - lidar.z)[mask]
    fin = np.isfinite(dz)
    bias = float(np.nanmedian(dz)) if fin.any() else 0.0
    e = dz - bias if removeBias else dz
    good = fin & (np.abs(e) < blunder)
    C = float(good.mean())
    sig = float(np.sqrt(np.mean(e[good] ** 2))) if good.any() else np.nan
    err = float(np.sqrt(C * (sig if np.isfinite(sig) else 0) ** 2 + (1 - C) * miss ** 2))
    nmad = float(1.4826 * np.nanmedian(np.abs(e[fin] - np.nanmedian(e[fin])))) if fin.any() else np.nan
    return dict(completeness=C, sigma_z_m=sig, err_rms_m=err, raw_rmse_m=float(np.sqrt(np.nanmean(dz ** 2))) if fin.any() else np.nan,
                nmad_m=nmad, bias_m=bias, valid_share=float(fin.mean()), shift_rows=dr, shift_cols=dc,
                n_mask=int(mask.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", type=Path, default=WORK_DIR, help=f"working folder (default {WORK_DIR})")
    p.add_argument("--blunder", type=float, default=5.0, help="errors beyond this (m) count as misses")
    p.add_argument("--miss-penalty", type=float, default=10.0, help="= the model's errMissM")
    p.add_argument("--margin", type=int, default=20, help="edge cells left out of scoring")
    p.add_argument("--exclude-classes", default="9", help="DFC classes left out (9 water; add 5 for trees)")
    p.add_argument("--no-bias", action="store_true", help="keep the median vertical bias in the error")
    p.add_argument("--shift-search", type=int, default=0, help="search horizontal shifts up to N cells")
    p.add_argument("--fusion", default="median", choices=["median", "mean"])
    p.add_argument("--save-fused", action="store_true")
    p.add_argument("--failed-as-empty", action="store_true",
                   help="a pair that ran and failed counts as reconstructing nothing")
    p.add_argument("--exclude-images", default="",
                   help="comma-separated image ids; subsets using any of them are left out")
    args = p.parse_args()

    img = pd.read_csv(args.work / "images.csv")
    subsets = pd.read_csv(args.work / "subsets.csv")
    info = img.groupby("tile").first()
    excl = tuple(int(c) for c in args.exclude_classes.split(",") if c.strip())
    rows, missingPairs, failedEmpty = [], set(), set()
    drop = {i.strip() for i in args.exclude_images.split(",") if i.strip()}
    if drop:
        n0 = len(subsets)
        subsets = subsets[[not (set(s.split(";")) & drop) for s in subsets.image_ids]]
        print(f"--exclude-images: {n0 - len(subsets)} subsets using {len(drop)} images left out")
    for tile, sub in subsets.groupby("tile"):
        cls = info.loc[tile, "cls"] if isinstance(info.loc[tile, "cls"], str) and info.loc[tile, "cls"] else None
        L = LidarTile(info.loc[tile, "lidar"], cls)
        mask = L.evalMask(args.margin, excl)
        cache = {}

        def pairDSM(a, b):
            key = f"{a}__{b}" if (args.work / "pairs" / f"{a}__{b}").exists() else f"{b}__{a}"
            if key not in cache:
                f = args.work / "pairs" / key / "dsm.tif"
                if f.exists():
                    cache[key] = toGrid(f, L)
                elif args.failed_as_empty and (args.work / "pairs" / key / "log.txt").exists():
                    cache[key] = np.full((L.nrows, L.ncols), np.nan, np.float32)
                    failedEmpty.add(key)
                else:
                    cache[key] = None
                    missingPairs.add(key)
            return cache[key]

        nDone = 0
        for r in sub.itertuples():
            ids = r.image_ids.split(";")
            stack = [pairDSM(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]
            stack = [s for s in stack if s is not None]
            npairs = len(ids) * (len(ids) - 1) // 2
            if len(stack) < npairs:
                continue                       # a pair failed or has not run: skip the subset
            st = np.stack(stack)
            with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore", RuntimeWarning)
                fused = np.nanmedian(st, 0) if args.fusion == "median" else np.nanmean(st, 0)
            s = score(fused, L, mask, args.blunder, args.miss_penalty, not args.no_bias, args.shift_search)
            rows.append(dict(subset_id=r.subset_id, tile=tile, kind=r.kind, n=r.n, image_ids=r.image_ids, **s))
            if args.save_fused:
                (args.work / "fused").mkdir(exist_ok=True)
                saveGrid(fused, L, args.work / "fused" / f"{r.subset_id}.tif")
            nDone += 1
        print(f"{tile}: scored {nDone} of {len(sub)} subsets")
    df = pd.DataFrame(rows)
    df.to_csv(args.work / "measured.csv", index=False)
    print(f"\n{len(df)} subsets scored -> {args.work / 'measured.csv'}")
    if failedEmpty:
        print(f"  {len(failedEmpty)} failed pairs counted as reconstructing nothing")
    if missingPairs:
        print(f"  {len(missingPairs)} pair DSMs missing (failed or not run); subsets using them were skipped")
    if len(df):
        print(df.groupby("n")[["completeness", "sigma_z_m", "err_rms_m", "bias_m"]].median().round(3).to_string())
        if (df.shift_rows.abs() + df.shift_cols.abs()).max() > 0:
            print("  horizontal shifts found:", df[["shift_rows", "shift_cols"]].value_counts().head().to_dict())


if __name__ == "__main__":
    main()
