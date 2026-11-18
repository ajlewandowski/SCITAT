"""
Step 5 - Fit the predicted-error model to the measured errors
=============================================================
Fits the parameters of imageGeometry.predictedErrorM so the predicted height
precision and completeness of every subset match what was measured against the
lidar (step 4). Each tile's own lidar DSM is registered as the occlusion scene, so
the buildings are real and only the matching / systematic parameters are fitted.

Fitted by default
  errSigma0Pix        matching precision (px) of a small-convergence pair
  errConvGrowth       how fast precision worsens with convergence
  errScaleExp         how fast it worsens with GSD mismatch
  errMatch50Deg       convergence where matching succeeds half the time
  errMatchWidthDeg    width of that fall-off
  errSysM             systematic floor (m)
  errDateMatchTauDays, errDateSigmaGrowth   date-gap term (only if every image has a date)
Held fixed
  errFusionEff = 1    it scales every set's variance the same way as errSigma0Pix,
                      so the two cannot be told apart; errSigma0Pix absorbs it
  errMissM            must equal scoreReconstructions --miss-penalty (default 10 m)

Fit: least squares on  log(predicted sigma_z / measured sigma_z)  and
(predicted - measured completeness) / 0.05, weighted by --w-completeness.

Checks
  before / after: rank agreement (Spearman) and median predicted/measured, for E_rms,
  sigma_z and completeness; with --cv, leave-one-tile-out (refit without the tile,
  then predict it) - the honest test of whether the calibration transfers.

Outputs (in --out, default <work>/calibration)
  calibration.json                 parameters (load with --calibration in the tests, or
                                   imageGeometry.loadErrorCalibration)
  predictions.csv                  measured and predicted values per subset
  1_predicted_vs_measured.png      default vs calibrated, sigma_z / completeness / E_rms
  2_residuals.png                  residuals vs convergence, number of views, date gap
  3_leave_tile_out.png             (with --cv)

Run
  python calibrateErrorModel.py --synthetic            # self-test: recover known parameters
  python calibrateErrorModel.py --cv
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

from dfc2019Common import (WORK_DIR, LidarTile, makeGeometry, predict, designSubsets, scitatAngles, imageGeometry,
                           WV3_ALT_KM)

# name: (low, high, log-scale)
SPACE = {"errSigma0Pix": (0.05, 3.0, True), "errConvGrowth": (0.0, 6.0, False),
         "errScaleExp": (0.0, 3.0, False), "errMatch50Deg": (8.0, 70.0, False),
         "errMatchWidthDeg": (0.5, 15.0, True), "errSysM": (0.0, 3.0, False),
         "errDateMatchTauDays": (5.0, 3000.0, True), "errDateSigmaGrowth": (0.0, 6.0, False)}
DATE_PARAMS = ("errDateMatchTauDays", "errDateSigmaGrowth")


# =====================================================================================
# Data
# =====================================================================================
def registerTiles(img, tiles, cellM, nPoints, margin, exclude, rng):
    """Register each tile's lidar as a scene for the model; scoring mask, block-averaged."""
    for tile in tiles:
        r = img[img.tile == tile].iloc[0]
        cls = r.cls if isinstance(r.cls, str) and r.cls else None
        L = LidarTile(r.lidar, cls)
        dsm, f = L.sceneForModel(cellM)
        m = L.evalMask(margin, exclude)[:dsm.shape[0] * f, :dsm.shape[1] * f]
        m = m.reshape(dsm.shape[0], f, dsm.shape[1], f).mean(axis=(1, 3)) > 0.5
        registerScene(f"dfc_{tile}", dsm, L.res * f, m, nPoints, rng)


def registerScene(name, dsm, cell, mask, nPoints, rng):
    """registerScene with a random subsample of the mask (speed: the fit evaluates every
    subset hundreds of times)."""
    mask = mask.copy()
    idx = np.flatnonzero(mask)
    if nPoints and len(idx) > nPoints:
        keep = np.zeros(mask.size, bool)
        keep[rng.choice(idx, nPoints, replace=False)] = True
        mask = keep.reshape(mask.shape)
    imageGeometry.registerScene(name, dsm, cellM=cell, margin=0, mask=mask)


def buildSets(img, meas):
    """One record per measured subset: scene, angles, dates, measured values."""
    by = img.set_index("image_id")
    sets = []
    for r in meas.itertuples():
        ids = r.image_ids.split(";")
        rows = by.loc[ids]
        dates = rows["date"].to_numpy() if "date" in rows and rows["date"].notna().all() else None
        sets.append(dict(subset_id=r.subset_id, tile=r.tile, kind=r.kind, n=len(ids), scene=f"dfc_{r.tile}",
                         angles=rows[["along_deg", "cross_deg"]].to_numpy(float),
                         dates=None if dates is None else np.array(dates, dtype="datetime64[D]"),
                         C=r.completeness, S=r.sigma_z_m, E=r.err_rms_m))
    return sets


def pairStats(geo, s):
    """Mean convergence and mean date gap of a set (for residual plots)."""
    u = geo.lookVectors(s["angles"], WV3_ALT_KM)
    c = np.degrees(np.arccos(np.clip(u @ u.T, -1, 1)))[np.triu_indices(len(u), 1)]
    g = np.nan
    if s["dates"] is not None:
        g = float(np.mean(geo.dateGaps(s["dates"])[np.triu_indices(len(u), 1)]))
    return float(np.mean(c)), g


# =====================================================================================
# Model wrapper
# =====================================================================================
class Fitter:
    def __init__(self, sets, names, missM, useDates, wC):
        self.sets, self.names, self.wC = sets, list(names), wC
        self.geo = makeGeometry("WorldView3")
        self.geo.errFusionEff = 1.0
        self.geo.errMissM = missM
        self.useDates = useDates
        if not useDates:
            self.geo.errDateMatchTauDays, self.geo.errDateSigmaGrowth = None, 0.0

    def toX(self, vals):
        return np.array([np.log(vals[n]) if SPACE[n][2] else vals[n] for n in self.names], float)

    def fromX(self, x):
        return {n: float(np.exp(v) if SPACE[n][2] else v) for n, v in zip(self.names, x)}

    def bounds(self):
        lo = [np.log(SPACE[n][0]) if SPACE[n][2] else SPACE[n][0] for n in self.names]
        hi = [np.log(SPACE[n][1]) if SPACE[n][2] else SPACE[n][1] for n in self.names]
        return np.array(lo), np.array(hi)

    def setParams(self, vals):
        for k, v in vals.items():
            setattr(self.geo, k, v)

    def predictAll(self, sets=None):
        out = []
        for s in (sets or self.sets):
            r = predict(self.geo, s["scene"], s["angles"], s["dates"] if self.useDates else None)
            out.append((r["completeness"], r["sigmaZM"], r["errRmsM"]))
        return np.array(out, float)

    def residuals(self, x, sets=None):
        self.setParams(self.fromX(x))
        sets = sets or self.sets
        P = self.predictAll(sets)
        Cm = np.array([s["C"] for s in sets]); Sm = np.array([s["S"] for s in sets])
        okS = (Cm > 0.1) & np.isfinite(Sm) & (Sm > 0) & np.isfinite(P[:, 1]) & (P[:, 1] > 0)
        rS = np.where(okS, np.log(np.where(okS, P[:, 1], 1) / np.where(okS, Sm, 1)), 0.0)
        rC = self.wC * (P[:, 0] - Cm) / 0.05
        return np.concatenate([rS, rC])

    def fit(self, start, sets=None, verbose=1):
        x0 = np.clip(self.toX(start), *self.bounds())
        lo, hi = self.bounds()
        res = least_squares(self.residuals, x0, bounds=(lo, hi), args=(sets,), x_scale="jac",
                            diff_step=1e-3, max_nfev=60 * (len(x0) + 1), verbose=verbose)
        vals = self.fromX(res.x)
        self.setParams(vals)
        return vals, res


def summary(meas, P):
    """Rank agreement and median ratio for C, sigma_z, E."""
    out = {}
    for j, (k, lab) in enumerate((("C", "completeness"), ("S", "sigma_z"), ("E", "err_rms"))):
        m = np.array([s[k] for s in meas], float)
        p = P[:, j]
        ok = np.isfinite(m) & np.isfinite(p) & (m > 0) & (p > 0)
        rho = spearmanr(m[ok], p[ok])[0] if ok.sum() > 3 else np.nan
        ratio = float(np.median(p[ok] / m[ok])) if ok.any() else np.nan
        logrms = float(np.sqrt(np.mean(np.log(p[ok] / m[ok]) ** 2))) if ok.any() else np.nan
        rmsd = float(np.sqrt(np.mean((p[ok] - m[ok]) ** 2))) if ok.any() else np.nan
        out[lab] = dict(spearman=float(rho), median_ratio=ratio, rms_log_ratio=logrms, rms_diff=rmsd)
    return out


def printSummary(title, s):
    print(f"  {title}")
    for k, v in s.items():
        tail = (f"RMS difference {v['rms_diff']:.3f}" if k == "completeness"
                else f"typical factor off ×{np.exp(v['rms_log_ratio']):.2f}")
        print(f"    {k:<13s} Spearman ρ {v['spearman']:+.3f}   median predicted/measured {v['median_ratio']:.2f}   {tail}")


# =====================================================================================
# Synthetic self-test
# =====================================================================================
TRUTH = dict(errSigma0Pix=0.45, errConvGrowth=1.5, errScaleExp=1.0, errMatch50Deg=33.0, errMatchWidthDeg=4.0,
             errSysM=0.5, errDateMatchTauDays=90.0, errDateSigmaGrowth=1.0)


def syntheticData(work, rng, nTiles=6, nImages=24, noise=0.08):
    """Fake DFC-like data: city DSMs, WorldView-3 views, dates over two years, and
    'measured' errors from the model with TRUTH parameters plus noise."""
    work.mkdir(parents=True, exist_ok=True)
    imgRows, measRows = [], []
    geo = makeGeometry("WorldView3")
    geo.errFusionEff, geo.errMissM = 1.0, 10.0
    for k, v in TRUTH.items():
        setattr(geo, k, v)
    for t in range(nTiles):
        tile = f"SYN_{t:03d}"
        size, n = 200, int(rng.integers(20, 70))
        dsm = np.zeros((size, size))
        for _ in range(n):
            w, l = rng.integers(8, 30, size=2)
            r, c = rng.integers(0, size - w), rng.integers(0, size - l)
            dsm[r:r + w, c:c + l] = np.maximum(dsm[r:r + w, c:c + l], rng.uniform(4, 10 + 4 * t))
        m = np.zeros_like(dsm, bool); m[40:-40, 40:-40] = True
        registerScene(f"dfc_{tile}", dsm, 1.0, m, 4000, rng)
        inc = rng.uniform(3, 40, nImages); az = rng.uniform(0, 360, nImages)
        dates = np.datetime64("2014-06-01") + rng.integers(0, 730, nImages).astype("timedelta64[D]")
        along, cross = scitatAngles(az, inc)
        ti = pd.DataFrame(dict(tile=tile, image_id=[f"{tile}_{i:03d}" for i in range(nImages)],
                               sat_az_deg=az, incidence_deg=inc, along_deg=along, cross_deg=cross,
                               date=dates.astype(str), doy=[int(str(d)[5:7]) * 30 for d in dates]))
        imgRows.append(ti)
        _, subs = designSubsets(ti, poolSize=9, nRandom=3, nSpread=2, nNarrow=2, sizes=(3, 4, 6, 8),
                                nTriples=6, rng=rng)
        by = ti.set_index("image_id")
        for kind, s in subs:
            rows = by.loc[s]
            r = predict(geo, f"dfc_{tile}", rows[["along_deg", "cross_deg"]].to_numpy(),
                        np.array(rows.date.to_numpy(), dtype="datetime64[D]"))
            C = float(np.clip(r["completeness"] + rng.normal(0, 0.02), 0, 1))
            S = float(r["sigmaZM"] * np.exp(rng.normal(0, noise))) if np.isfinite(r["sigmaZM"]) else np.nan
            E = float(np.sqrt(C * (S if np.isfinite(S) else 0) ** 2 + (1 - C) * 100))
            measRows.append(dict(subset_id=f"{tile}_s{len(measRows):05d}", tile=tile, kind=kind, n=len(s),
                                 image_ids=";".join(s), completeness=C, sigma_z_m=S, err_rms_m=E))
    img = pd.concat(imgRows, ignore_index=True)
    meas = pd.DataFrame(measRows)
    img.to_csv(work / "images.csv", index=False); meas.to_csv(work / "measured.csv", index=False)
    return img, meas


# =====================================================================================
# Figures
# =====================================================================================
def figures(out, sets, P0, P1, geo, cvP=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = sorted({s["kind"] for s in sets})
    cols = dict(zip(kinds, ["#2a6fdb", "#e0663d", "#3aa76d", "#9b59b6", "#c9a227", "#555555"]))
    meas = np.array([[s["C"], s["S"], s["E"]] for s in sets], float)
    labels = ("completeness", "height precision σz (m)", "E_rms (m)")
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), layout="constrained")
    for row, (P, name) in enumerate(((P0, "default parameters"), (P1, "calibrated"))):
        for j, ax in enumerate(axes[row]):
            x, y = meas[:, j], P[:, j]
            ok = np.isfinite(x) & np.isfinite(y) & ((x > 0) & (y > 0) if j else True)
            for k in kinds:
                kk = ok & np.array([s["kind"] == k for s in sets])
                ax.scatter(x[kk], y[kk], s=14, color=cols[k], alpha=0.75, label=k, edgecolor="none")
            if ok.any():
                lo, hi = min(x[ok].min(), y[ok].min()), max(x[ok].max(), y[ok].max())
                ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            if j:
                ax.set_xscale("log"); ax.set_yscale("log")
            rho = spearmanr(x[ok], y[ok])[0] if ok.sum() > 3 else np.nan
            ax.set_title(f"{name}: ρ = {rho:+.2f}", fontsize=10)
            ax.set_xlabel(f"measured {labels[j]}"); ax.set_ylabel(f"predicted {labels[j]}")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Predicted vs measured, every subset", fontsize=13)
    fig.savefig(out / "1_predicted_vs_measured.png", dpi=130); plt.close(fig)

    conv, gap = np.array([pairStats(geo, s) for s in sets]).T
    n = np.array([s["n"] for s in sets])
    okS = np.isfinite(meas[:, 1]) & np.isfinite(P1[:, 1]) & (meas[:, 1] > 0)
    rS = np.full(len(sets), np.nan); rS[okS] = np.log(P1[okS, 1] / meas[okS, 1])
    rC = P1[:, 0] - meas[:, 0]
    xs = [(conv, "mean pair convergence (°)"), (n + np.random.default_rng(0).uniform(-0.2, 0.2, len(n)), "views in the set"),
          (gap, "mean pair date gap (days)")]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout="constrained")
    for j, (x, lab) in enumerate(xs):
        for i, (r, rl) in enumerate(((rS, "log(predicted / measured σz)"), (rC, "predicted − measured completeness"))):
            ax = axes[i, j]
            ax.scatter(x, r, s=10, alpha=0.6, color="#2a6fdb", edgecolor="none")
            ax.axhline(0, color="k", lw=1)
            ax.set_xlabel(lab); ax.set_ylabel(rl); ax.grid(alpha=0.3)
    fig.suptitle("Calibrated-model residuals: a trend means the model's form is missing something", fontsize=12)
    fig.savefig(out / "2_residuals.png", dpi=130); plt.close(fig)

    if cvP is not None:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
        for j, ax in enumerate(axes):
            x, y = meas[:, j], cvP[:, j]
            ok = np.isfinite(x) & np.isfinite(y) & ((x > 0) & (y > 0) if j else True)
            tiles = sorted({s["tile"] for s in sets})
            for t in tiles:
                kk = ok & np.array([s["tile"] == t for s in sets])
                ax.scatter(x[kk], y[kk], s=12, alpha=0.75, label=t)
            if ok.any():
                lo, hi = min(x[ok].min(), y[ok].min()), max(x[ok].max(), y[ok].max())
                ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            if j:
                ax.set_xscale("log"); ax.set_yscale("log")
            rho = spearmanr(x[ok], y[ok])[0] if ok.sum() > 3 else np.nan
            ax.set_title(f"held-out tiles: ρ = {rho:+.2f}", fontsize=10)
            ax.set_xlabel(f"measured {labels[j]}"); ax.set_ylabel(f"predicted {labels[j]}"); ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7, ncol=2)
        fig.suptitle("Leave-one-tile-out: each tile predicted by a model fitted without it", fontsize=12)
        fig.savefig(out / "3_leave_tile_out.png", dpi=130); plt.close(fig)


# =====================================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", type=Path, default=None, help=f"working folder (default {WORK_DIR})")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fit", default=None, help="comma-separated parameters to fit (default: see above)")
    p.add_argument("--no-dates", action="store_true", help="leave out the date-gap term")
    p.add_argument("--miss-penalty", type=float, default=10.0, help="= scoreReconstructions --miss-penalty")
    p.add_argument("--w-completeness", type=float, default=1.0)
    p.add_argument("--cell", type=float, default=1.0, help="model scene cell size (m)")
    p.add_argument("--points", type=int, default=6000, help="scene points per tile used by the model")
    p.add_argument("--margin", type=int, default=20, help="= scoreReconstructions --margin")
    p.add_argument("--exclude-classes", default="9", help="= scoreReconstructions --exclude-classes")
    p.add_argument("--min-completeness", type=float, default=0.0, help="drop subsets measured below this")
    p.add_argument("--cv", action="store_true", help="leave-one-tile-out check (one refit per tile)")
    p.add_argument("--pipeline", default="s2p", help="name of the MVS pipeline, stored in the JSON")
    p.add_argument("--synthetic", action="store_true", help="self-test on fake data with known parameters")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if not args.synthetic:
        args.work = args.work or WORK_DIR
    if args.synthetic:
        args.work = args.work or Path(__file__).resolve().parent / "synthetic"
        print("Synthetic self-test: true parameters", TRUTH)
        img, meas = syntheticData(args.work, rng)
    else:
        img = pd.read_csv(args.work / "images.csv")
        meas = pd.read_csv(args.work / "measured.csv")
        meas = meas[meas.completeness >= args.min_completeness]
        excl = tuple(int(c) for c in args.exclude_classes.split(",") if c.strip())
        registerTiles(img, sorted(meas.tile.unique()), args.cell, args.points, args.margin * 1, excl, rng)
    out = args.out or args.work / "calibration"
    out.mkdir(parents=True, exist_ok=True)

    sets = buildSets(img, meas)
    useDates = (not args.no_dates) and all(s["dates"] is not None for s in sets)
    names = [n.strip() for n in args.fit.split(",")] if args.fit else \
        [n for n in SPACE if useDates or n not in DATE_PARAMS]
    print(f"{len(sets)} subsets on {meas.tile.nunique()} tiles; fitting {', '.join(names)}"
          + ("" if useDates else "  (no date term: some images have no date)"))

    F = Fitter(sets, names, args.miss_penalty, useDates, args.w_completeness)
    default = dict(errSigma0Pix=0.3, errConvGrowth=0.5, errScaleExp=1.0, errMatch50Deg=45.0, errMatchWidthDeg=3.0,
                   errSysM=0.5 * float(F.geo.sens[3]), errDateMatchTauDays=180.0, errDateSigmaGrowth=0.5)
    # "default" predictions = the model as shipped (no date term, fusion 0.7)
    g0 = makeGeometry("WorldView3"); g0.errMissM = args.miss_penalty
    P0 = np.array([[r["completeness"], r["sigmaZM"], r["errRmsM"]] for r in
                   (predict(g0, s["scene"], s["angles"]) for s in sets)])
    s0 = summary(sets, P0)

    t = time.time()
    vals, res = F.fit({k: default[k] for k in names}, verbose=0)
    P1 = F.predictAll()
    s1 = summary(sets, P1)
    print(f"\nFit finished in {time.time() - t:.0f} s ({res.nfev} model evaluations, cost {res.cost:.2f})")
    print("\nParameters" + ("  (fitted | true)" if args.synthetic else ""))
    for k in names:
        tv = f" | {TRUTH[k]:.3g}" if args.synthetic and k in TRUTH else ""
        print(f"  {k:<22s} {vals[k]:.4g}{tv}")
    print()
    printSummary("default model:", s0)
    printSummary("calibrated:", s1)

    cvP, scv = None, None
    if args.cv:
        tiles = sorted({s["tile"] for s in sets})
        cvP = np.full_like(P1, np.nan)
        for tile in tiles:
            train = [s for s in sets if s["tile"] != tile]
            test = [i for i, s in enumerate(sets) if s["tile"] == tile]
            G = Fitter(train, names, args.miss_penalty, useDates, args.w_completeness)
            G.fit(vals, verbose=0)
            cvP[test] = G.predictAll([sets[i] for i in test])
            print(f"  held out {tile}: done")
        scv = summary(sets, cvP)
        printSummary("leave-one-tile-out (held-out predictions):", scv)

    rows = []
    for s, a, b in zip(sets, P0, P1):
        rows.append(dict(subset_id=s["subset_id"], tile=s["tile"], kind=s["kind"], n=s["n"],
                         meas_completeness=s["C"], meas_sigma_m=s["S"], meas_err_m=s["E"],
                         default_completeness=a[0], default_sigma_m=a[1], default_err_m=a[2],
                         cal_completeness=b[0], cal_sigma_m=b[1], cal_err_m=b[2]))
    pr = pd.DataFrame(rows)
    if cvP is not None:
        pr["cv_completeness"], pr["cv_sigma_m"], pr["cv_err_m"] = cvP.T
    pr.to_csv(out / "predictions.csv", index=False)

    params = {k: getattr(F.geo, k) for k in imageGeometry.ERR_PARAMS}
    params["errFusionEff"] = 1.0
    if not useDates:
        params["errDateMatchTauDays"], params["errDateSigmaGrowth"] = None, 0.0
    doc = dict(params=params, fitted=names, sensor="WorldView3", pipeline=args.pipeline,
               data="synthetic self-test" if args.synthetic else "DFC2019 Track 3",
               n_subsets=len(sets), tiles=sorted(meas.tile.unique().tolist()),
               scoring=dict(miss_penalty_m=args.miss_penalty, margin_cells=args.margin,
                            exclude_classes=args.exclude_classes),
               fit_stats=dict(default=s0, calibrated=s1, leave_tile_out=scv))
    (out / "calibration.json").write_text(json.dumps(doc, indent=2, default=float))
    figures(out, sets, P0, P1, F.geo, cvP)
    print(f"\nWrote {out / 'calibration.json'}, predictions.csv and figures to {out}")


if __name__ == "__main__":
    main()
