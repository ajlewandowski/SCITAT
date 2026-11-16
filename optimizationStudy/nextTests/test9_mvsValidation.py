"""
Test 9 - Does the cost predict real reconstruction error?
=========================================================
Question: a low cost is only useful if it means a low 3-D error. This test checks
the cost against an independent error model, and prepares the sets you would
reconstruct with a real MVS pipeline (S2P, Ames Stereo Pipeline, ...).

Part A - Monte Carlo triangulation simulator (runs here, minutes)
  A synthetic city: flat ground with random rectangular buildings (a DSM) on a
  grid whose cell size is the sensor's nadir GSD. For every image set:
    visibility   each ground/roof point is seen by a view only if the ray toward
                 that satellite clears every building (ray marching on the DSM)
    matching     a point is reconstructable from a pair of views only if both see
                 it AND their convergence angle is within --conv-band (default
                 5-45 deg, the range reported as matchable for satellite MVS)
    noise        each view measures with error sigmaPix x GSD / cos(off-nadir)
                 perpendicular to its ray
    triangulate  weighted least-squares intersection of the usable rays
  Outputs per set: height RMSE on reconstructed points, completeness (fraction of
  points reconstructed) and an "error-or-miss" RMSE where a missed point counts
  as --miss-penalty metres. None of this uses the cost function, so agreement is
  real evidence and disagreement shows what the cost is missing.
  NOT modeled: image texture, radiometry, shadows, seasonal/temporal change,
  orbit-model (RPC) errors. Those need real data - that is Part B.

Part B - Sets for a real MVS run
  Picks --n-export sets that span the cost range and writes their pointing angles
  (along/cross and off-nadir/azimuth) to mvs_validation_sets.csv. Reconstruct
  those, measure RMSE against lidar / a reference DSM, then save a CSV with
  columns  set_id, rmse_m, completeness  and run again with
  --mvs-results that.csv  to get the cost-vs-real-error plot.
  (Public multi-view satellite data with lidar truth, such as the IARPA
  multi-view stereo benchmark over Buenos Aires, works for this - choose the
  images whose angles are closest to each exported set.)

Outputs (in --out, default ./results/test9_mvsValidation)
  simulated_errors.csv             every set: cost, terms, simulated metrics
  mvs_validation_sets.csv          Part B sets to reconstruct
  1_scene_and_error_maps.png       the synthetic scene, a visibility mask, error maps
  2_cost_vs_simulated_error.png    cost vs RMSE / completeness / error-or-miss
  3_term_correlations.png          which cost term best predicts the simulated error
  4_cost_bins.png                  simulated error by cost quintile
  5_real_mvs.png                   (only with --mvs-results)

Run
  python test9_mvsValidation.py --quick          # ~1 min
  python test9_mvsValidation.py                  # ~5 min
  python test9_mvsValidation.py --mvs-results my_s2p_results.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import spearmanr

from studyCommon import (CostEvaluator, COST_TERMS, TERM_LABELS, TERM_COLORS, SERIES, SEQ, INK2,
                         angleAxes, anglesToPolar, lookVectors, pairGeometry, polarToAngles,
                         save, style, commonArgs, Timer, randomSet, incidenceAngle,
                         makeEvaluator, resolveArgs)


# =====================================================================================
# Synthetic scene and simulator
# =====================================================================================
def makeCity(rng, n=120, cell=2.0, nBuildings=60, hRange=(5, 40), wRange=(3, 12)):
    """DSM (n x n, metres) with flat ground and random rectangular buildings."""
    dsm = np.zeros((n, n))
    for _ in range(nBuildings):
        w, l = rng.integers(*wRange, size=2)
        r, c = rng.integers(0, n - w), rng.integers(0, n - l)
        dsm[r:r + w, c:c + l] = np.maximum(dsm[r:r + w, c:c + l], rng.uniform(*hRange))
    return dsm


class Simulator:
    """Visibility + matching + noisy least-squares triangulation on a DSM."""

    def __init__(self, dsm, cell, gsd0, sigmaPix, convBand, missPenalty, rng, margin=10, altKm=None):
        """altKm: satellite altitude for spherical-Earth geometry (None = flat Earth)."""
        self.dsm, self.cell, self.gsd0, self.altKm = dsm, cell, gsd0, altKm
        self.sigmaPix, self.convBand, self.miss = sigmaPix, convBand, missPenalty
        self.rng = rng
        n = dsm.shape[0]
        # Evaluate points away from the edges so rays have scene to cross
        rr, cc = np.meshgrid(np.arange(margin, n - margin), np.arange(margin, n - margin), indexing="ij")
        self.r, self.c = rr.ravel(), cc.ravel()
        self.z = dsm[self.r, self.c]
        self._vis = {}

    def visible(self, along, cross):
        """Boolean visibility of every evaluation point from one view (cached)."""
        key = (round(along, 3), round(cross, 3))
        if key in self._vis:
            return self._vis[key]
        off, az = anglesToPolar([[along, cross]])
        off = np.radians(incidenceAngle(off[0], self.altKm))   # angle at the ground
        if off < np.radians(0.25):
            v = np.ones(len(self.z), bool)
        else:
            # Ground direction from the point toward the satellite = opposite of pointing azimuth.
            # Grid axes: rows = along-track, cols = cross-track.
            dr, dc = -np.cos(np.radians(az[0])), -np.sin(np.radians(az[0]))
            step = 0.5                                        # cells per ray step
            rise = step * self.cell / np.tan(off)             # metres gained per step
            K = int(np.ceil((self.dsm.max() - self.dsm.min()) / rise)) + 1
            v = np.ones(len(self.z), bool)
            n = self.dsm.shape[0]
            for k in range(1, K + 1):
                rr = np.rint(self.r + k * step * dr).astype(int)
                cc = np.rint(self.c + k * step * dc).astype(int)
                inside = (rr >= 0) & (rr < n) & (cc >= 0) & (cc < n)
                h = np.zeros(len(self.z))
                h[inside] = self.dsm[rr[inside], cc[inside]]
                v &= ~(inside & (h > self.z + k * rise + 1e-6))
        self._vis[key] = v
        return v

    def slantRangeOverH(self, offDeg):
        """Slant range / altitude for off-nadir angle(s): 1/cos on a flat Earth."""
        th = np.radians(np.asarray(offDeg, float))
        if self.altKm is None:
            return 1 / np.cos(th)
        R, h = 6378137.0, self.altKm * 1e3
        lam = np.radians(incidenceAngle(np.degrees(th), self.altKm)) - th     # central angle
        return np.sqrt(R ** 2 + (R + h) ** 2 - 2 * R * (R + h) * np.cos(lam)) / h

    def run(self, angles, reps=3):
        """Returns dict(rmse, completeness, le90, missRmse, dzMap) for one image set."""
        angles = np.asarray(angles, float)
        V = len(angles)
        u = lookVectors(angles, self.altKm)                   # satellite -> ground, z down
        off, _ = anglesToPolar(angles)
        # measurement error perpendicular to the ray = sigmaPix x IFOV x slant range
        sig = self.sigmaPix * self.gsd0 * self.slantRangeOverH(off)
        vis = np.column_stack([self.visible(a, c) for a, c in angles])          # (M, V)
        conv, _, _ = pairGeometry(angles, self.altKm)
        i, j = np.triu_indices(V, 1)
        okPair = (conv >= self.convBand[0]) & (conv <= self.convBand[1])
        used = np.zeros_like(vis)
        for a, b in zip(i[okPair], j[okPair]):
            both = vis[:, a] & vis[:, b]
            used[:, a] |= both
            used[:, b] |= both
        rec = used.sum(1) >= 2
        P = np.eye(3)[None] - u[:, :, None] * u[:, None, :]                       # (V, 3, 3)
        W = used[rec] / sig ** 2                                                    # (m, V)
        A = np.einsum("mv,vij->mij", W, P)
        dz = np.zeros((reps, rec.sum()))
        for rp in range(reps):
            e = self.rng.normal(size=(rec.sum(), V, 3)) * sig[None, :, None]
            b = np.einsum("mv,vij,mvj->mi", W, P, e)
            dz[rp] = np.linalg.solve(A, b[..., None])[..., 0][:, 2]
        rmsePt = np.sqrt(np.mean(dz ** 2, 0))                                     # per point over reps
        comp = rec.mean()
        rmse = np.sqrt(np.mean(rmsePt ** 2)) if rec.any() else np.nan
        missR = np.sqrt((np.sum(rmsePt ** 2) + (~rec).sum() * self.miss ** 2) / len(rec))
        dzMap = np.full(self.dsm.shape, np.nan)            # nan = outside the evaluated area
        dzMap[self.r, self.c] = -1.0                        # -1 = evaluated but not reconstructed
        dzMap[self.r[rec], self.c[rec]] = rmsePt
        return dict(rmse=rmse, completeness=comp, missRmse=missR,
                    le90=np.percentile(np.abs(dz), 90) if rec.any() else np.nan, dzMap=dzMap)


def bootRho(x, y, rng, n=500):
    """Spearman rho with a 95% bootstrap interval."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    r = spearmanr(x, y).statistic
    bs = []
    for _ in range(n):
        k = rng.integers(0, len(x), len(x))
        if np.ptp(x[k]) > 0 and np.ptp(y[k]) > 0:
            bs.append(spearmanr(x[k], y[k]).statistic)
    return r, np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)


# =====================================================================================
def main():
    p = commonArgs(__doc__.split("\n")[1], "test9_mvsValidation")
    p.add_argument("--n-random", type=int, default=150, help="random sets (3..15 images)")
    p.add_argument("--grid", type=int, default=120, help="scene size in cells")
    p.add_argument("--conv-band", default="5,45", help="matchable convergence range (deg)")
    p.add_argument("--miss-penalty", type=float, default=10.0, help="error charged for a missed point (m)")
    p.add_argument("--reps", type=int, default=3, help="Monte Carlo noise draws per set")
    p.add_argument("--n-export", type=int, default=12, help="sets to export for a real MVS run")
    p.add_argument("--mvs-results", default=None, help="CSV with set_id, rmse_m[, completeness]")
    args = resolveArgs(p.parse_args())
    if args.quick:
        args.n_random, args.grid, args.reps = 30, 80, 2
    style()
    out = args.out
    rng = np.random.default_rng(args.seed)
    ev = makeEvaluator(args)
    gsd0 = ev.sens[3]
    convBand = tuple(float(v) for v in args.conv_band.split(","))

    dsm = makeCity(np.random.default_rng([args.seed, 1]), n=args.grid, cell=gsd0)
    sim = Simulator(dsm, gsd0, gsd0, ev.sigmaPix, convBand, args.miss_penalty,
                    np.random.default_rng([args.seed, 2]),
                    altKm=None if args.flat_earth else ev.alt)

    # ------------------------------------------------------------------ the set library
    sets = []
    for k in (2, 3, 5, 8):
        for t in np.arange(5, args.max_off + 0.1, 5):
            sets.append(("ring", polarToAngles(np.full(k, t), 360.0 * np.arange(k) / k)))
    for ta in np.arange(10, args.max_off + 0.1, 10):
        sets.append(("single pass", np.column_stack([[-ta, 0, ta], [0, 0, 0]]).astype(float)))
    for _ in range(args.n_random):
        sets.append(("random", randomSet(rng, int(rng.integers(3, 16)), args.max_off)))

    rows, maps = [], []
    with Timer(f"Simulating {len(sets)} sets on a {args.grid}×{args.grid} scene "
               f"(cell = nadir GSD = {gsd0:.2f} m)"):
        for sid, (kind, s) in enumerate(sets):
            r = ev.evaluate(s)
            if not r["ok"]:
                continue
            m = sim.run(s, args.reps)
            maps.append(m["dzMap"])
            rows.append({"set_id": sid, "kind": kind, "nImages": len(s), **{k: r[k] for k in COST_TERMS},
                         "cost": r["cost"], "sim_rmse_m": m["rmse"], "sim_le90_m": m["le90"],
                         "sim_completeness": m["completeness"], "sim_missRmse_m": m["missRmse"],
                         "angles": ";".join(f"{a:.2f},{c:.2f}" for a, c in s)})
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "simulated_errors.csv", index=False)

    # ------------------------------------------------------------------ figure 1: scene
    iBest, iWorst = df.cost.idxmin(), df.cost.idxmax()
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.4), layout="constrained")
    ext = (0, args.grid * gsd0, 0, args.grid * gsd0)
    im = axes[0].imshow(dsm, cmap=SEQ, extent=ext, origin="lower")
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="height (m)")
    axes[0].set_title("Synthetic scene (DSM)")
    vis = np.full(dsm.shape, np.nan)
    vis[sim.r, sim.c] = sim.visible(25.0, 0.0)
    axes[1].imshow(vis, cmap=ListedColormap(["#3d3d3a", "#cde2fb"]), extent=ext, origin="lower")
    axes[1].set_title("Visible from a 25° fore view (dark = hidden)")
    both = np.concatenate([maps[iBest].ravel(), maps[iWorst].ravel()])
    vmax = np.nanpercentile(both[both >= 0], 98)
    for ax, i, lab in ((axes[2], iBest, "lowest-cost set"), (axes[3], iWorst, "highest-cost set")):
        m = np.ma.masked_less(np.nan_to_num(maps[i], nan=-2.0), 0)     # hide misses and border
        im = ax.imshow(m, cmap=SEQ, vmin=0, vmax=vmax, extent=ext, origin="lower")
        missed = np.where(maps[i] == -1.0, 1.0, np.nan)
        ax.imshow(missed, cmap=ListedColormap(["#e34948"]), extent=ext, origin="lower")
        r = df.loc[i]
        ax.set_title(f"{lab}: cost {r.cost:.3f}\nRMSE {r.sim_rmse_m:.2f} m, "
                     f"completeness {r.sim_completeness:.0%} (red = not reconstructed)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, label="height error RMS (m)")
    for ax in axes:
        ax.set_xlabel("m"); ax.set_ylabel("m"); ax.grid(False)
    fig.suptitle("Test 9 – Independent error simulator: occlusion + matching + noisy triangulation",
                 fontsize=12)
    save(fig, out, "1_scene_and_error_maps")

    # ------------------------------------------------------------------ figure 2: cost vs metrics
    brng = np.random.default_rng(0)
    metrics = [("sim_rmse_m", "height RMSE on reconstructed points (m)"),
               ("sim_completeness", "completeness (fraction reconstructed)"),
               ("sim_missRmse_m", f"error-or-miss RMSE (m, miss = {args.miss_penalty:g} m)")]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), layout="constrained")
    kinds = ["random", "ring", "single pass"]
    for ax, (mcol, mlab) in zip(axes, metrics):
        for col, kd in zip(SERIES, kinds):
            d = df[df.kind == kd]
            ax.scatter(d.cost, d[mcol], s=22, color=col, edgecolor="k", lw=0.3, alpha=0.85, label=kd)
        r, ci = bootRho(df.cost.to_numpy(), df[mcol].to_numpy(), brng)
        ax.set_xlabel("cost"); ax.set_ylabel(mlab)
        ax.set_title(f"Spearman ρ = {r:+.2f}  (95% CI {ci[0]:+.2f} to {ci[1]:+.2f})", fontsize=10)
    axes[0].legend(fontsize=8)
    fig.suptitle("Test 9 – Does a lower cost mean a lower simulated 3-D error? "
                 "(want ρ > 0 for errors, ρ < 0 for completeness)", fontsize=12)
    save(fig, out, "2_cost_vs_simulated_error")

    # ------------------------------------------------------------------ figure 3: per-term rho
    target = df.sim_missRmse_m.to_numpy()
    names = list(COST_TERMS) + ["cost"]
    res = []
    for k in names:
        x = df[k].to_numpy()
        res.append(bootRho(x, target, brng) if np.ptp(x) > 1e-9 else (np.nan, (np.nan, np.nan)))
    fig, ax = plt.subplots(figsize=(10, 5.2), layout="constrained")
    for i, (k, (r, ci)) in enumerate(zip(names, res)):
        col = TERM_COLORS.get(k, INK2)
        if np.isnan(r):
            ax.annotate("constant", (i, 0), ha="center", va="bottom", fontsize=8, color=INK2)
            continue
        ax.bar(i, r, color=col, width=0.6)
        ax.errorbar(i, r, yerr=[[r - ci[0]], [ci[1] - r]], color="k", capsize=4, lw=1)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(names)), [TERM_LABELS.get(k, "Total cost") for k in names], rotation=15)
    ax.set_ylabel("Spearman ρ with error-or-miss RMSE")
    ax.set_title("Test 9 – Which cost term tracks the simulated error? (bars = 95% bootstrap CI)")
    save(fig, out, "3_term_correlations")

    # ------------------------------------------------------------------ figure 4: bins
    df["costBin"] = pd.qcut(df.cost, 5, labels=[f"Q{i}" for i in range(1, 6)])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for ax, (mcol, mlab) in zip(axes, (metrics[2], metrics[1])):
        groups = [df[df.costBin == b][mcol].to_numpy() for b in df.costBin.cat.categories]
        bp = ax.boxplot(groups, widths=0.55, showfliers=False, patch_artist=True, medianprops=dict(color="k"))
        for b in bp["boxes"]:
            b.set_facecolor(SERIES[0]); b.set_alpha(0.35)
        ax.set_xticks(range(1, 6), ["Q1\n(lowest cost)", "Q2", "Q3", "Q4", "Q5\n(highest cost)"])
        ax.set_ylabel(mlab)
    fig.suptitle("Test 9 – Simulated error by cost quintile", fontsize=12)
    save(fig, out, "4_cost_bins")

    # ------------------------------------------------------------------ Part B: export
    q = np.linspace(0, 1, args.n_export)
    srt = df.sort_values("cost").reset_index(drop=True)
    pick = srt.iloc[np.unique((q * (len(srt) - 1)).round().astype(int))]
    exp = []
    for _, r in pick.iterrows():
        A = np.array([[float(v) for v in s.split(",")] for s in r.angles.split(";")])
        off, az = anglesToPolar(A)
        for k, ((a, c), o, z) in enumerate(zip(A, off, az)):
            exp.append({"set_id": r.set_id, "cost": r.cost, "image": k + 1, "along_deg": a, "cross_deg": c,
                        "offnadir_deg": o, "pointing_azimuth_deg": z,
                        "sim_missRmse_m": r.sim_missRmse_m})
    pd.DataFrame(exp).to_csv(out / "mvs_validation_sets.csv", index=False)
    print(f"\nExported {len(pick)} sets spanning cost {pick.cost.min():.3f}–{pick.cost.max():.3f} "
          f"to mvs_validation_sets.csv")

    if args.mvs_results:
        real = pd.read_csv(args.mvs_results).merge(df[["set_id", "cost", "sim_missRmse_m"]], on="set_id")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
        for ax, x, lab in ((axes[0], "cost", "cost"), (axes[1], "sim_missRmse_m", "simulated error-or-miss RMSE (m)")):
            ax.scatter(real[x], real.rmse_m, s=60, color=SERIES[0], edgecolor="k", lw=0.5)
            for _, r in real.iterrows():
                ax.annotate(str(r.set_id), (r[x], r.rmse_m), xytext=(4, 3), textcoords="offset points", fontsize=7)
            rr = spearmanr(real[x], real.rmse_m).statistic
            ax.set_xlabel(lab); ax.set_ylabel("real MVS RMSE (m)"); ax.set_title(f"Spearman ρ = {rr:+.2f}")
        fig.suptitle(f"Test 9 – Real reconstructions ({len(real)} sets)", fontsize=12)
        save(fig, out, "5_real_mvs")

    r, ci = bootRho(df.cost.to_numpy(), df.sim_missRmse_m.to_numpy(), brng)
    print(f"Cost vs simulated error-or-miss RMSE: Spearman ρ = {r:+.2f} (95% CI {ci[0]:+.2f} to {ci[1]:+.2f})")


if __name__ == "__main__":
    main()
