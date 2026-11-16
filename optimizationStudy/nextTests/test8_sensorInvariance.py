"""
Test 8 - Do the conclusions hold across sensors and noise settings?
====================================================================
Question: is "the optimal geometry" a property of geometry alone (so it carries
over to any satellite), or does it change with the sensor (GSD, field of view,
altitude) and with the cost's noise parameters sigmaPix and eMax?

What it does
  1. Builds one fixed benchmark library of geometries: ring sets (k views at
     off-nadir theta) plus random sets with 3..15 images.
  2. Scores the whole library under several configurations:
       sensors     SkySat 400 km, SkySat 500 km, Landsat-9 705 km, WorldView-3-like 617 km,
                   Pleiades Neo 620 km
                   (each with the default sigmaPix = 0.5, eMax = 25 m)
       scaled eMax the same sensors with eMax = --emax-gsd-factor x nadir GSD, so the
                   error scale means the same thing for a 0.3 m and a 30 m sensor
       noise grid  SkySat with sigmaPix in {0.25, 0.5, 1} x eMax in {1, 5, 25, 100} m
  3. Compares configurations by
       - rank agreement (Spearman) of the library's costs: 1 = same ranking
       - the best ring off-nadir angle for each k
       - which cost terms carry the variance

Outputs (in --out, default ./results/test8_sensorInvariance)
  benchmark_costs.csv              every geometry x configuration: terms and cost
  1_rank_agreement.png             Spearman between every pair of configurations
  2_ring_curves.png                ring (k = 8) cost vs theta for each sensor
  3_best_theta.png                 optimal ring angle per k, per configuration
  4_noise_grid.png                 SkySat: effect of sigmaPix and eMax
  5_variance_share.png             which term drives the ranking in each configuration

Run
  python test8_sensorInvariance.py --quick       # ~1 min
  python test8_sensorInvariance.py               # ~3 min
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from studyCommon import (CostEvaluator, COST_TERMS, TERM_LABELS, TERM_COLORS, SERIES, SEQ, DIV, INK2,
                         sensArray, save, style, commonArgs, Timer, randomSet, polarToAngles,
                         makeEvaluator, resolveArgs)


def library(rng, ks, thetas, nRandom):
    geoms = []
    for k in ks:
        for t in thetas:
            geoms.append(dict(kind="ring", k=k, theta=t,
                              angles=polarToAngles(np.full(k, t), 360.0 * np.arange(k) / k)))
    for i in range(nRandom):
        N = int(rng.integers(3, 16))
        geoms.append(dict(kind="random", k=N, theta=np.nan, angles=randomSet(rng, N, 30)))
    return geoms


def varianceShare(df, weights):
    """Share of the variance in total cost carried by each weighted term (feasible sets)."""
    d = df[df.cost.notna()]
    E = d[list(COST_TERMS)].to_numpy()
    w = np.array([weights[k] for k in COST_TERMS])
    c = E @ w
    v = c.var(ddof=1)
    return np.array([np.cov(E[:, i] * w[i], c)[0, 1] / v if v > 0 else np.nan
                     for i in range(len(COST_TERMS))])


def main():
    p = commonArgs(__doc__.split("\n")[1], "test8_sensorInvariance")
    p.add_argument("--n-random", type=int, default=80)
    p.add_argument("--emax-gsd-factor", type=float, default=10.0)
    args = resolveArgs(p.parse_args())
    style()
    out = args.out
    rng = np.random.default_rng(args.seed)
    ks = [3, 5, 8, 12, 15]
    thetas = np.arange(2.5, 30.1, 2.5)
    sig, emx = [0.25, 0.5, 1.0], [1.0, 5.0, 25.0, 100.0]
    if args.quick:
        args.n_random, ks, thetas = 20, [3, 8], np.arange(5, 30.1, 5)
        sig, emx = [0.5, 1.0], [5.0, 25.0]
    geoms = library(rng, ks, thetas, args.n_random)

    # ------------------------------------------------------------------ configurations
    sensorCfgs = [("SkySat 400 km", "SkySat", 400), ("SkySat 500 km", "SkySat", 500),
                  ("Landsat-9 705 km", "Landsat9", 705), ("WorldView-3 617 km", "WorldView3", 617),
                  ("Pléiades Neo 620 km", "PleiadesNeo", 620)]
    configs = []        # (label, group, sensor, alt, sigmaPix, eMax)
    for lab, s, a in sensorCfgs:
        configs.append((lab, "sensor", s, a, 0.5, 25.0))
    for lab, s, a in sensorCfgs:
        gsd = sensArray(s, a)[3]
        configs.append((f"{lab}, eMax={args.emax_gsd_factor:g}×GSD", "scaled", s, a, 0.5,
                        args.emax_gsd_factor * gsd))
    for sp in sig:
        for em in emx:
            configs.append((f"σ={sp:g} px, eMax={em:g} m", "noise", "SkySat", 400, sp, em))

    rows = []
    for lab, grp, s, a, sp, em in configs:
        with Timer(f"{lab}"):
            # same agility limit / normalization range (--max-off) for every sensor so the
            # benchmark library is scored on equal terms
            ev = makeEvaluator(args, sensor=s, alt_km=a, sigmaPix=sp, eMax=em)
            for gi, g in enumerate(geoms):
                r = ev.evaluate(g["angles"])
                rows.append({"config": lab, "group": grp, "sensor": s, "alt": a, "sigmaPix": sp, "eMax": em,
                             "geom": gi, "kind": g["kind"], "k": g["k"], "theta": g["theta"],
                             **{k: r[k] for k in COST_TERMS}, "cost": r["cost"]})
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "benchmark_costs.csv", index=False)

    C = df.pivot(index="geom", columns="config", values="cost")
    labels = [c[0] for c in configs]
    C = C[labels]

    # ------------------------------------------------------------------ figure 1: rank agreement
    main_labels = [c[0] for c in configs if c[1] in ("sensor", "scaled")]
    R = spearmanr(C[main_labels].to_numpy(), nan_policy="omit").statistic
    fig, ax = plt.subplots(figsize=(10, 8.5), layout="constrained")
    im = ax.imshow(R, cmap=DIV, vmin=-1, vmax=1)
    for i in range(len(main_labels)):
        for j in range(len(main_labels)):
            ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(R[i, j]) > 0.6 else "k")
    ax.set_xticks(range(len(main_labels)), main_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(main_labels)), main_labels, fontsize=8); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Spearman ρ between the configurations' rankings")
    ax.set_title("Test 8 – Do different sensors rank the same geometries the same way?\n"
                 "(1 = identical ranking → the optimal geometry carries over)")
    save(fig, out, "1_rank_agreement")

    # ------------------------------------------------------------------ figure 2: ring curves
    kShow = 8 if 8 in ks else ks[-1]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), layout="constrained", sharey=False)
    for ax, grp, title in ((axes[0], "sensor", "(a) eMax = 25 m for every sensor"),
                           (axes[1], "scaled", f"(b) eMax = {args.emax_gsd_factor:g} × nadir GSD")):
        cfgs = [c for c in configs if c[1] == grp]
        for col, c in zip(SERIES, cfgs):
            d = df[(df.config == c[0]) & (df.kind == "ring") & (df.k == kShow)].sort_values("theta")
            ax.plot(d.theta, d.cost, "-o", ms=4, color=col, label=c[0].split(",")[0])
            ib = d.cost.idxmin()
            ax.scatter(d.theta[ib], d.cost[ib], marker="*", s=180, color=col, edgecolor="k", zorder=4)
        ax.set_xlabel("ring off-nadir angle θ (°)"); ax.set_ylabel("total cost")
        ax.set_title(title); ax.legend(fontsize=8)
    fig.suptitle(f"Test 8 – Ring of {kShow} views: does the best angle (★) depend on the sensor?", fontsize=12)
    save(fig, out, "2_ring_curves")

    # ------------------------------------------------------------------ figure 3: best theta
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    cfgs = [c for c in configs if c[1] in ("sensor", "scaled")]
    width = 0.8 / len(cfgs)
    for i, (col, c) in enumerate(zip(SERIES, cfgs)):
        d = df[(df.config == c[0]) & (df.kind == "ring")]
        bt = [d[d.k == k].nsmallest(1, "cost").theta.iloc[0] for k in ks]
        ax.bar(np.arange(len(ks)) + (i - len(cfgs) / 2 + 0.5) * width, bt, width * 0.9, color=col,
               edgecolor="white", lw=1, label=c[0], hatch="//" if c[1] == "scaled" else None)
    ax.set_xticks(range(len(ks)), [f"k = {k}" for k in ks])
    ax.set_ylabel("best ring off-nadir angle θ (°)")
    ax.set_title("Test 8 – Optimal ring angle for each number of views (hatched = eMax scaled to GSD)")
    ax.legend(fontsize=7, ncols=2, loc="upper left")
    save(fig, out, "3_best_theta")

    # ------------------------------------------------------------------ figure 4: noise grid
    base = C["σ=0.5 px, eMax=25 m"] if "σ=0.5 px, eMax=25 m" in C else C[labels[0]]
    A = np.full((len(sig), len(emx)), np.nan)
    Bt = np.full_like(A, np.nan)
    Lv = np.full_like(A, np.nan)
    for i, sp in enumerate(sig):
        for j, em in enumerate(emx):
            lab = f"σ={sp:g} px, eMax={em:g} m"
            A[i, j] = spearmanr(C[lab], base, nan_policy="omit").statistic
            d = df[(df.config == lab) & (df.kind == "ring") & (df.k == kShow)]
            Bt[i, j] = d.nsmallest(1, "cost").theta.iloc[0]
            Lv[i, j] = df[df.config == lab].errCostMap.mean()
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), layout="constrained")
    for ax, Z, title, fmt, cmap, lim in (
            (axes[0], Lv, "(a) mean MVS-error-map term\n(≈0 = term is switched off)", "{:.3f}", SEQ, (0, 1)),
            (axes[1], A, "(b) rank agreement with σ=0.5, eMax=25", "{:.2f}", DIV, (-1, 1)),
            (axes[2], Bt, f"(c) best ring angle, k = {kShow} (°)", "{:.1f}", SEQ, (None, None))):
        im = ax.imshow(Z, cmap=cmap, vmin=lim[0], vmax=lim[1], aspect="auto")
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                ax.text(j, i, fmt.format(Z[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if (cmap is SEQ and np.nanmax(Z) and Z[i, j] > np.nanmean(Z) * 1.2) else "k")
        ax.set_xticks(range(len(emx)), [f"{e:g}" for e in emx]); ax.set_xlabel("eMax (m)")
        ax.set_yticks(range(len(sig)), [f"{s:g}" for s in sig]); ax.set_ylabel("σ (pixels)")
        ax.set_title(title, fontsize=10); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Test 8 – SkySat 400 km: sensitivity to the MVS noise parameters", fontsize=12)
    save(fig, out, "4_noise_grid")

    # ------------------------------------------------------------------ figure 5: variance share
    cfgs = [c[0] for c in configs if c[1] in ("sensor", "scaled")]
    from studyCommon import imageGeometry
    S = np.array([varianceShare(df[df.config == c], imageGeometry.costWeights) for c in cfgs])
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    # Shares sum to 1 but can be negative (a term that moves against the total);
    # positives stack right of zero, negatives left of it.
    pos, neg = np.zeros(len(cfgs)), np.zeros(len(cfgs))
    for i, k in enumerate(COST_TERMS):
        v = np.nan_to_num(S[:, i])
        vp, vn = np.clip(v, 0, None), np.clip(v, None, 0)
        ax.barh(range(len(cfgs)), vp, left=pos, color=TERM_COLORS[k], edgecolor="white", lw=1.5,
                height=0.6, label=TERM_LABELS[k])
        ax.barh(range(len(cfgs)), vn, left=neg, color=TERM_COLORS[k], edgecolor="white", lw=1.5, height=0.6)
        pos += vp
        neg += vn
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(cfgs)), cfgs, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("share of the variance in total cost (left of 0 = term moves against the total)")
    ax.set_title("Test 8 – Which term drives the ranking for each sensor")
    ax.legend(ncols=len(COST_TERMS), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, out, "5_variance_share")

    print("\nRank agreement with SkySat 400 km (default noise):")
    for lab, r in zip(main_labels, R[0]):
        print(f"   {lab:<42s} ρ = {r:.3f}")


if __name__ == "__main__":
    main()
