"""
Test 5 - How many images are enough?
====================================
Question: how does the achievable cost fall as the number of images N grows, and
where does adding images stop paying off? This is the practical design answer
(how many collections to task per target).

For each N it computes three things
  random      the spread of random N-image sets (what you get with no planning)
  ring best   the best ring geometry with N images (N on a ring, or nadir + N-1),
              searching the ring's off-nadir angle - a cheap "planned" baseline
  optimized   simulated annealing with N images (studyCommon.anneal)

The "knee" is the smallest N whose optimized cost is within --knee-tol (default 2%)
of the best cost over all N.

Outputs (in --out, default ./results/test5_imageCountSweep)
  image_count_sweep.csv          per N: random percentiles, ring best, optimized best + terms
  optimized_sets.npz             optimized set for each N
  1_cost_vs_n.png                cost vs N for all three, knee marked
  2_marginal_gain.png            cost reduction from each extra image (optimized)
  3_components_vs_n.png          each weighted term vs N (optimized)
  4_geometry_vs_n.png            optimized geometry for each N
  5_features_vs_n.png            off-nadir, convergence, azimuth gap vs N

Run
  python test5_imageCountSweep.py --quick        # ~2 min
  python test5_imageCountSweep.py                # N = 2..20, ~20 min
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from studyCommon import (CostEvaluator, COST_TERMS, TERM_LABELS, TERM_COLORS, SERIES, INK2,
                         FEATURE_LABELS, angleAxes, save, style, commonArgs, Timer, anneal,
                         randomSet, setFeatures, polarToAngles)


def bestRing(ev, N, thetas):
    """Best ring-type geometry with exactly N images: N on a ring, or nadir + N-1."""
    best = (np.inf, None)
    for t in thetas:
        cands = [polarToAngles(np.full(N, t), 360.0 * np.arange(N) / N)]
        if N - 1 >= 2:
            cands.append(np.vstack([[0, 0], polarToAngles(np.full(N - 1, t), 360.0 * np.arange(N - 1) / (N - 1))]))
        for s in cands:
            c = ev.cost(s, failCost=np.inf)
            if c < best[0]:
                best = (c, s)
    return best


def main():
    p = commonArgs(__doc__.split("\n")[1], "test5_imageCountSweep")
    p.add_argument("--ns", default="2,3,4,5,6,7,8,10,12,15,20", help="image counts to test")
    p.add_argument("--n-random", type=int, default=100, help="random sets per N")
    p.add_argument("--iters", type=int, default=1200, help="annealing evaluations per restart")
    p.add_argument("--restarts", type=int, default=2, help="annealing restarts per N")
    p.add_argument("--max-off", type=int, default=30)
    p.add_argument("--knee-tol", type=float, default=0.02)
    args = p.parse_args()
    if args.quick:
        args.ns, args.n_random, args.iters, args.restarts = "2,3,5,8", 15, 120, 1
    style()
    out = args.out
    ev = CostEvaluator(args.sensor, maxOffNadir=args.max_off)
    Ns = [int(v) for v in args.ns.split(",")]
    thetas = np.arange(2.5, args.max_off + 0.1, 2.5)

    rows, optSets = [], {}
    for N in Ns:
        with Timer(f"N = {N}"):
            rng = np.random.default_rng([args.seed, N])
            rc = np.array([ev.cost(randomSet(rng, N, args.max_off), failCost=np.nan)
                           for _ in range(args.n_random)])
            ringC, ringS = bestRing(ev, N, thetas)
            # Start one restart from the ring answer so the optimizer is never worse than it
            best = None
            for r in range(args.restarts):
                init = None
                if r == 0 and ringS is not None:
                    init = np.unique(np.round(ringS), axis=0)
                    if len(init) != N:
                        init = None
                res = anneal(ev.cost, N, np.random.default_rng([args.seed, N, r]), iters=args.iters,
                             maxOff=args.max_off, init=init)
                if best is None or res["bestCost"] < best["bestCost"]:
                    best = res
            terms = ev.evaluate(best["best"])
            optSets[N] = best["best"]
            rows.append({"N": N, "rand_p10": np.nanpercentile(rc, 10), "rand_p50": np.nanpercentile(rc, 50),
                         "rand_p90": np.nanpercentile(rc, 90), "rand_min": np.nanmin(rc),
                         "ring_best": ringC, "opt_best": best["bestCost"],
                         **{f"opt_{k}": terms[k] for k in COST_TERMS},
                         **{f"feat_{k}": v for k, v in setFeatures(best["best"]).items()}})
            print(f"   random median {rows[-1]['rand_p50']:.4f}   ring {ringC:.4f}   "
                  f"optimized {best['bestCost']:.4f}")

    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "image_count_sweep.csv", index=False)
    np.savez(out / "optimized_sets.npz", **{f"N{N}": s for N, s in optSets.items()})

    overall = df.opt_best.min()
    within = df.N[df.opt_best <= overall * (1 + args.knee_tol)]
    knee = int(within.min())
    print(f"\nKnee: N = {knee} is within {args.knee_tol:.0%} of the best cost over all N ({overall:.4f})")

    # ------------------------------------------------------------------ figure 1
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.fill_between(df.N, df.rand_p10, df.rand_p90, color=SERIES[1], alpha=0.15,
                    label="random sets, 10–90th percentile")
    ax.plot(df.N, df.rand_p50, "--", color=SERIES[1], lw=1.5, label="random sets, median")
    ax.plot(df.N, df.rand_min, ":", color=SERIES[1], lw=1.5, label="best random set")
    ax.plot(df.N, df.ring_best, "-s", color=SERIES[2], ms=5, label="best ring geometry")
    ax.plot(df.N, df.opt_best, "-o", color=SERIES[0], label="optimized (annealing)")
    ax.axhspan(overall, overall * (1 + args.knee_tol), color=SERIES[0], alpha=0.08)
    ax.axvline(knee, color=INK2, ls="--", lw=1)
    ax.annotate(f"knee: N = {knee}\n(within {args.knee_tol:.0%} of best)", (knee, ax.get_ylim()[1]),
                xytext=(6, -6), textcoords="offset points", va="top", fontsize=9)
    ax.set_xlabel("number of images N"); ax.set_ylabel("total cost")
    ax.set_xticks(Ns)
    ax.set_title("Test 5 – How cost falls with the number of images")
    ax.legend(loc="upper right")
    save(fig, out, "1_cost_vs_n")

    # ------------------------------------------------------------------ figure 2
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    gain = -np.diff(df.opt_best.to_numpy()) / np.diff(df.N.to_numpy())
    labels = [f"{a}→{b}" for a, b in zip(Ns[:-1], Ns[1:])]
    ax.bar(range(len(gain)), gain, color=SERIES[0], width=0.6)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(gain)), labels)
    ax.set_xlabel("going from N to N'"); ax.set_ylabel("cost reduction per extra image")
    ax.set_title("Test 5 – Marginal value of extra images (optimized sets)")
    save(fig, out, "2_marginal_gain")

    # ------------------------------------------------------------------ figure 3
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    W = ev.geo.costWeights
    for k in COST_TERMS:
        ax.plot(df.N, W[k] * df[f"opt_{k}"], "-o", ms=4, color=TERM_COLORS[k], label=TERM_LABELS[k])
    ax.set_xticks(Ns)
    ax.set_xlabel("number of images N"); ax.set_ylabel("weighted contribution to total cost")
    ax.set_title("Test 5 – Which terms improve as images are added (optimized sets)")
    ax.legend(ncols=3, fontsize=8)
    save(fig, out, "3_components_vs_n")

    # ------------------------------------------------------------------ figure 4
    ncol = min(len(Ns), 4)
    nrow = int(np.ceil(len(Ns) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.4 * nrow), layout="constrained",
                             squeeze=False)
    for ax, N in zip(axes.ravel(), Ns):
        s = optSets[N]
        ax.scatter(s[:, 0], s[:, 1], s=45, color=SERIES[0], edgecolor="k", lw=0.5, zorder=3)
        angleAxes(ax, args.max_off, title=f"N = {N}, cost {df.opt_best[df.N == N].iloc[0]:.4f}")
    for ax in axes.ravel()[len(Ns):]:
        ax.set_visible(False)
    fig.suptitle("Test 5 – Optimized geometry for each number of images", fontsize=12)
    save(fig, out, "4_geometry_vs_n")

    # ------------------------------------------------------------------ figure 5
    feats = ["offNadirMean", "convMedian", "azMaxGap", "bhMean"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), layout="constrained")
    for ax, f in zip(axes, feats):
        ax.plot(df.N, df[f"feat_{f}"], "-o", color=SERIES[0])
        ax.set_xticks(Ns); ax.set_xlabel("number of images N")
        ax.set_title(FEATURE_LABELS[f])
    fig.suptitle("Test 5 – How the optimal geometry changes with N", fontsize=12)
    save(fig, out, "5_features_vs_n")


if __name__ == "__main__":
    main()
