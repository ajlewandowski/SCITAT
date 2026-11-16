"""
Test 2 - Search for the optimum instead of sampling at random
=============================================================
Question: what is the best 15-image geometry the cost function can find, how much
better is it than the best of N random sets, and do independent searches agree on
the SAME shape (a real optimum) or scatter (a flat cost surface)?

What it does
  1. Random baseline: cost of N_RANDOM random sets (as in randomImageGroupStudy).
  2. Simulated annealing (studyCommon.anneal) from N_RESTARTS different random
     starts, each with ITERS cost evaluations. Moves nudge one image a few degrees
     or teleport it to a random pointing.
  3. Equal-budget comparison: best-so-far of annealing vs best-so-far of random
     sampling for the same number of cost evaluations.
  4. Agreement between restarts: geometry features of each restart's answer
     (off-nadir, convergence, azimuth spread, ...) side by side.
  5. Tolerance: jitter every image of the overall best set by +/-d degrees and
     see how fast the cost rises. A flat curve means pointing errors of d degrees
     do not matter; a steep one means the optimum is sharp.

Outputs (in --out, default ./results/test2_optimizerSearch)
  best_sets.npz / best_sets.csv     each restart's best set and its terms
  1_convergence.png                 annealing vs random search, same budget
  2_cost_distribution.png           random-set costs vs optimized costs
  3_optimized_geometries.png        each restart's best set in angle space
  4_restart_agreement.png           features of each restart's answer
  5_components.png                  cost terms: best random vs optimized
  6_tolerance.png                   cost increase vs pointing jitter

Run
  python test2_optimizerSearch.py --quick       # ~1 min
  python test2_optimizerSearch.py               # 6 restarts x 2000 evals, ~10 min
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from studyCommon import (CostEvaluator, COST_TERMS, TERM_LABELS, TERM_COLORS, SERIES, SEQ, INK2,
                         FEATURE_LABELS, angleAxes, save, style, commonArgs, Timer, anneal,
                         randomSet, setFeatures)


def main():
    p = commonArgs(__doc__.split("\n")[1], "test2_optimizerSearch")
    p.add_argument("--n-images", type=int, default=15)
    p.add_argument("--n-random", type=int, default=2000, help="random sets for the baseline")
    p.add_argument("--restarts", type=int, default=6)
    p.add_argument("--iters", type=int, default=2000, help="cost evaluations per restart")
    p.add_argument("--max-off", type=int, default=30, help="max |along| and |cross| (deg)")
    args = p.parse_args()
    if args.quick:
        args.n_random, args.restarts, args.iters = 150, 2, 150
    style()
    out = args.out
    ev = CostEvaluator(args.sensor, maxOffNadir=args.max_off)
    N = args.n_images

    # ------------------------------------------------------------------ 1. random baseline
    with Timer(f"Random baseline: {args.n_random} sets"):
        rng = np.random.default_rng([args.seed, 0])
        randSets = [randomSet(rng, N, args.max_off) for _ in range(args.n_random)]
        randRes = [ev.evaluate(s) for s in randSets]
        randCost = np.array([r["cost"] if r["ok"] else np.nan for r in randRes])
        good = np.isfinite(randCost)
        print(f"   best {np.nanmin(randCost):.4f}   median {np.nanmedian(randCost):.4f}")

    # ------------------------------------------------------------------ 2. annealing restarts
    runs = []
    for r in range(args.restarts):
        with Timer(f"Annealing restart {r + 1}/{args.restarts} ({args.iters} evaluations)"):
            res = anneal(ev.cost, N, np.random.default_rng([args.seed, 1, r]),
                         iters=args.iters, maxOff=args.max_off)
            res["terms"] = ev.evaluate(res["best"])
            runs.append(res)
            print(f"   best cost {res['bestCost']:.4f}  (start temperature {res['T0']:.2g})")

    allBest = min(runs, key=lambda d: d["bestCost"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "best_sets.npz", sets=np.stack([d["best"] for d in runs]),
             costs=np.array([d["bestCost"] for d in runs]),
             traces=np.stack([d["trace"] for d in runs]))
    pd.DataFrame([{"restart": i + 1, **{k: d["terms"][k] for k in COST_TERMS},
                   "cost": d["bestCost"],
                   "angles": ";".join(f"{a:g},{c:g}" for a, c in d["best"])}
                  for i, d in enumerate(runs)]).to_csv(out / "best_sets.csv", index=False)

    # ------------------------------------------------------------------ figure 1: convergence
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    evals = np.arange(1, args.iters + 1)
    for i, d in enumerate(runs):
        ax.plot(evals, d["trace"], color=SERIES[0], alpha=0.75, lw=1.5,
                label="simulated annealing (each restart)" if i == 0 else None)
    # random search with the same budget, using the baseline costs in random order
    rc = np.where(good, randCost, np.inf)
    nRS = min(args.iters, len(rc))
    for i in range(min(args.restarts, max(1, len(rc) // nRS))):
        seg = rc[i * nRS:(i + 1) * nRS]
        ax.plot(np.arange(1, len(seg) + 1), np.minimum.accumulate(seg), color=SERIES[1], alpha=0.8,
                lw=1.5, label="random sampling, same budget" if i == 0 else None)
    ax.axhline(np.nanmin(randCost), color=SERIES[1], ls=":", lw=1.2,
               label=f"best of all {good.sum()} random sets")
    ax.set_xscale("log")
    ax.set_xlabel("number of cost evaluations"); ax.set_ylabel("best cost found so far")
    ax.set_title("Test 2 – Optimizer vs random sampling at equal budget")
    ax.legend()
    save(fig, out, "1_convergence")

    # ------------------------------------------------------------------ figure 2: distribution
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.hist(randCost[good], bins=50, color=SERIES[0], alpha=0.75, label=f"{good.sum()} random sets")
    for i, d in enumerate(runs):
        ax.axvline(d["bestCost"], color=SERIES[1], lw=1.5,
                   label="annealing result (one line per restart)" if i == 0 else None)
    frac = np.mean(randCost[good] <= allBest["bestCost"])
    ax.set_xlabel("total cost"); ax.set_ylabel("number of sets")
    ax.set_title(f"Test 2 – Optimized sets vs the random distribution "
                 f"({frac:.1%} of random sets are as good as the best optimized set)")
    ax.legend()
    save(fig, out, "2_cost_distribution")

    # ------------------------------------------------------------------ figure 3: geometries
    nR = len(runs)
    ncol = min(nR, 3) + 1
    nrow = int(np.ceil((nR + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.6 * nrow), layout="constrained",
                             squeeze=False)
    axes = axes.ravel()
    ib = np.nanargmin(randCost)
    s = randSets[ib]
    axes[0].scatter(s[:, 0], s[:, 1], s=50, color=SERIES[1], edgecolor="k", lw=0.5, zorder=3)
    angleAxes(axes[0], args.max_off, title=f"best random set\ncost {randCost[ib]:.4f}")
    for i, d in enumerate(runs):
        b = d["best"]
        axes[i + 1].scatter(b[:, 0], b[:, 1], s=50, color=SERIES[0], edgecolor="k", lw=0.5, zorder=3)
        angleAxes(axes[i + 1], args.max_off, title=f"annealing restart {i + 1}\ncost {d['bestCost']:.4f}")
    for ax in axes[nR + 1:]:
        ax.set_visible(False)
    fig.suptitle("Test 2 – Where the optimized sets put their images "
                 "(similar shapes across restarts = a real optimum)", fontsize=12)
    save(fig, out, "3_optimized_geometries")

    # ------------------------------------------------------------------ figure 4: agreement
    feats = ["offNadirMean", "offNadirMin", "offNadirMax", "convMedian", "convMax",
             "fracGoodConv", "azMaxGap", "bhMean", "pointingImbalance", "nNearNadir"]
    F_rand = pd.DataFrame([setFeatures(s) for s, g in zip(randSets, good) if g])
    F_opt = pd.DataFrame([setFeatures(d["best"]) for d in runs])
    fig, axes = plt.subplots(2, 5, figsize=(20, 7), layout="constrained")
    for ax, f in zip(axes.ravel(), feats):
        lo, hi = np.percentile(F_rand[f], [5, 95])
        ax.axhspan(lo, hi, color=SERIES[1], alpha=0.12, label="random sets, 5–95%")
        ax.axhline(F_rand[f].median(), color=SERIES[1], lw=1, ls="--", label="random median")
        ax.scatter(np.arange(1, nR + 1), F_opt[f], s=60, color=SERIES[0], edgecolor="k", lw=0.5,
                   zorder=3, label="annealing restarts")
        ax.set_xticks(np.arange(1, nR + 1)); ax.set_xlabel("restart")
        ax.set_title(FEATURE_LABELS[f], fontsize=10)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("Test 2 – Do independent restarts land on the same kind of geometry?", fontsize=12)
    save(fig, out, "4_restart_agreement")

    # ------------------------------------------------------------------ figure 5: components
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    labels = ["best random"] + [f"restart {i + 1}" for i in range(nR)]
    T = [randRes[ib]] + [d["terms"] for d in runs]
    W = ev.geo.costWeights
    bottom = np.zeros(len(T))
    for k in COST_TERMS:
        v = np.array([W[k] * t[k] for t in T])
        ax.bar(range(len(T)), v, bottom=bottom, color=TERM_COLORS[k], width=0.6,
               edgecolor="white", lw=1.5, label=TERM_LABELS[k])
        bottom += v
    for i, b in enumerate(bottom):
        ax.annotate(f"{b:.4f}", (i, b), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(range(len(T)), labels, rotation=20)
    ax.set_ylabel("weighted contribution to total cost")
    ax.set_title("Test 2 – Which terms the optimizer improved")
    ax.legend(ncols=5, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
    save(fig, out, "5_components")

    # ------------------------------------------------------------------ figure 6: tolerance
    with Timer("Tolerance of the best set to pointing jitter"):
        rng = np.random.default_rng([args.seed, 2])
        mags = [1, 2, 3, 5, 8]
        nTrials = 8 if args.quick else 40
        best = allBest["best"]
        base = allBest["bestCost"]
        inc = []
        for m in mags:
            d = []
            for _ in range(nTrials):
                x = np.clip(best + rng.integers(-m, m + 1, size=best.shape), -args.max_off, args.max_off)
                d.append(ev.cost(x) - base)
            inc.append(d)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    bp = ax.boxplot(inc, positions=mags, widths=0.6, showfliers=False, patch_artist=True,
                    medianprops=dict(color="k"))
    for b in bp["boxes"]:
        b.set_facecolor(SERIES[0]); b.set_alpha(0.35)
    for m, d in zip(mags, inc):
        ax.scatter(np.full(len(d), m) + np.random.default_rng(0).uniform(-0.15, 0.15, len(d)), d,
                   s=10, color=SERIES[0], lw=0)
    spread = np.nanpercentile(randCost, 75) - np.nanpercentile(randCost, 25)
    ax.axhline(spread, color=SERIES[1], ls="--", lw=1)
    ax.annotate("interquartile spread of random-set costs", (mags[0], spread), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("pointing jitter per image (± deg, uniform, both axes)")
    ax.set_ylabel("cost increase over the best set")
    ax.set_title("Test 2 – How precisely do the optimal pointings need to be hit?")
    save(fig, out, "6_tolerance")

    print(f"\nBest optimized cost {allBest['bestCost']:.4f} vs best random {np.nanmin(randCost):.4f}"
          f" vs median random {np.nanmedian(randCost):.4f}")
    print("Best optimized set (along, cross):")
    print(np.array2string(allBest["best"][np.argsort(allBest["best"][:, 0])].astype(int)))


if __name__ == "__main__":
    main()
