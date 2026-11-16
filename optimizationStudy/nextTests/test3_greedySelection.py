"""
Test 3 - Greedy "next best view" selection
==========================================
Question: if you could only add one image at a time, where should each next image
go, and how much does each one buy you? This is the tasking version of the
optimization: it gives an ORDER of pointings and the marginal value of each.

What it does
  1. Starts from a seed set (default: one nadir image; change with --seed-angles).
  2. At every step, evaluates the cost of (current set + candidate) for every
     candidate pointing on a grid (default every 2 deg in [-30, 30]^2) and keeps
     the candidate with the lowest cost.
  3. Records the whole "cost if the next image goes here" map at every step - these
     maps show which regions of angle space are worth tasking next.
  4. For reference, the median and 10th percentile of random sets of the same size.

Outputs (in --out, default ./results/test3_greedySelection)
  greedy_sequence.csv               order, pointing, cost and terms after each step
  next_image_maps.npz               the cost map at every step
  1_cost_vs_images.png              greedy cost and marginal gain per added image
  2_greedy_order.png                the chosen pointings, numbered in order
  3_next_image_maps.png             "cost if the next image goes here" maps
  4_components_vs_images.png        how each cost term evolves as images are added

Run
  python test3_greedySelection.py --quick         # ~1 min (6 deg grid, 6 images)
  python test3_greedySelection.py                 # 2 deg grid, 15 images, ~6 min
  python test3_greedySelection.py --seed-angles "-20,0;20,0"   # start from a stereo pair
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from studyCommon import (CostEvaluator, COST_TERMS, TERM_LABELS, TERM_COLORS, SERIES, SEQ, INK2,
                         angleAxes, save, style, commonArgs, Timer, randomSet, randomFeasibleCosts,
                         imageGeometry, makeEvaluator, resolveArgs)


def parseAngles(text):
    """'a,c;a,c' -> (n, 2) array"""
    return np.array([[float(v) for v in pair.split(",")] for pair in text.split(";") if pair.strip()])


def main():
    p = commonArgs(__doc__.split("\n")[1], "test3_greedySelection")
    p.add_argument("--n-images", type=int, default=15, help="final set size")
    p.add_argument("--grid-step", type=float, default=2.0, help="candidate spacing (deg)")
    p.add_argument("--seed-angles", default="0,0", help="starting set, e.g. '0,0' or '-20,0;20,0'")
    p.add_argument("--n-random", type=int, default=60, help="random sets per size for the reference")
    args = resolveArgs(p.parse_args())
    if args.quick:
        args.n_images, args.grid_step, args.n_random = 6, 6.0, 10
    style()
    out = args.out
    ev = makeEvaluator(args)

    gm = np.floor(args.max_off / args.grid_step) * args.grid_step
    g = np.arange(-gm, gm + 1e-9, args.grid_step)
    cand = np.array(np.meshgrid(g, g, indexing="ij")).reshape(2, -1).T   # (M, 2)
    nG = len(g)
    chosen = parseAngles(args.seed_angles)
    nSeed = len(chosen)
    inCone = list(imageGeometry.offNadir(cand) <= args.max_off + 1e-9)   # agility limit
    print(f"{sum(inCone)} candidate pointings within {args.max_off:g}° off-nadir, "
          f"starting from {nSeed} seed image(s)")

    seq, maps = [], {}
    if nSeed >= 2:
        r = ev.evaluate(chosen)
        seq.append({"n": nSeed, "along": np.nan, "cross": np.nan, "cost": r["cost"],
                    **{k: r[k] for k in COST_TERMS}})
    # ------------------------------------------------------------------ greedy loop
    while len(chosen) < args.n_images:
        n = len(chosen) + 1
        with Timer(f"Choosing image {n} of {args.n_images}"):
            costs = np.full(len(cand), np.nan)
            used = {tuple(c) for c in chosen}
            for i, c in enumerate(cand):
                if tuple(c) in used or not inCone[i]:
                    continue
                r = ev.evaluate(np.vstack([chosen, c]))
                if r["ok"]:
                    costs[i] = r["cost"]
            ib = np.nanargmin(costs)
            chosen = np.vstack([chosen, cand[ib]])
            r = ev.evaluate(chosen)
            seq.append({"n": n, "along": cand[ib, 0], "cross": cand[ib, 1], "cost": r["cost"],
                        **{k: r[k] for k in COST_TERMS}})
            maps[n] = costs.reshape(nG, nG)
            print(f"   added ({cand[ib, 0]:+.0f}, {cand[ib, 1]:+.0f})  ->  cost {r['cost']:.4f}")

    df = pd.DataFrame(seq)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "greedy_sequence.csv", index=False)
    np.savez(out / "next_image_maps.npz", grid=g, chosen=chosen,
             **{f"step{n}": m for n, m in maps.items()})

    # ------------------------------------------------------------------ random reference
    with Timer("Random reference sets"):
        sizes = df.n.to_numpy()
        ref = []
        for n in sizes:
            rng = np.random.default_rng([args.seed, int(n)])
            c, _ = randomFeasibleCosts(ev, rng, int(n), args.n_random, int(args.max_off))
            c = c if len(c) else np.array([np.nan])
            ref.append(np.nanpercentile(c, [10, 50, 90]))
        ref = np.array(ref)

    # ------------------------------------------------------------------ figure 1
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), layout="constrained")
    ax = axes[0]
    ax.fill_between(sizes, ref[:, 0], ref[:, 2], color=SERIES[1], alpha=0.15,
                    label="random feasible sets, 10–90th percentile")
    ax.plot(sizes, ref[:, 1], color=SERIES[1], ls="--", lw=1.5, label="random feasible sets, median")
    ax.plot(sizes, df.cost, "-o", color=SERIES[0], label="greedy")
    ax.set_xlabel("number of images"); ax.set_ylabel("total cost")
    ax.set_title("(a) Cost as images are added one at a time")
    ax.legend()
    ax = axes[1]
    gain = -np.diff(df.cost.to_numpy())
    ax.bar(sizes[1:], gain, color=SERIES[0], width=0.6)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("image number added"); ax.set_ylabel("cost reduction from that image")
    ax.set_title("(b) Marginal value of each added image")
    fig.suptitle("Test 3 – Greedy next-best-view selection", fontsize=12)
    save(fig, out, "1_cost_vs_images")

    # ------------------------------------------------------------------ figure 2
    fig, ax = plt.subplots(figsize=(7.5, 7), layout="constrained")
    ax.plot(chosen[:, 0], chosen[:, 1], color=GRIDLINE, lw=0.8, zorder=2)
    sc = ax.scatter(chosen[:, 0], chosen[:, 1], c=np.arange(1, len(chosen) + 1), cmap=SEQ,
                    s=180, edgecolor="k", lw=0.6, zorder=3, vmin=0)
    for i, (a, c) in enumerate(chosen):
        ax.annotate(str(i + 1), (a, c), ha="center", va="center", fontsize=8,
                    color="white" if i > len(chosen) * 0.45 else "k", zorder=4)
    angleAxes(ax, args.max_off, title="Test 3 – Greedy order of pointings (1 = first)")
    fig.colorbar(sc, ax=ax, fraction=0.046, label="order added")
    save(fig, out, "2_greedy_order")

    # ------------------------------------------------------------------ figure 3
    steps = sorted(maps)
    show = steps if len(steps) <= 6 else [steps[i] for i in np.linspace(0, len(steps) - 1, 6).round().astype(int)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout="constrained")
    for ax, n in zip(axes.ravel(), show):
        m = maps[n]
        ext = (g[0] - args.grid_step / 2, g[-1] + args.grid_step / 2) * 2
        im = ax.imshow(m.T, origin="lower", extent=ext, cmap=SEQ, interpolation="nearest")
        prev = chosen[:n - 1]
        ax.scatter(prev[:, 0], prev[:, 1], s=40, color="white", edgecolor="k", lw=0.8, zorder=3,
                   label="already chosen")
        ax.scatter(*chosen[n - 1], marker="*", s=260, color=SERIES[3], edgecolor="k", zorder=4,
                   label="best next image")
        angleAxes(ax, args.max_off, title=f"choosing image {n}", zeroLines=False)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, label="cost with image here (darker = higher)")
    for ax in axes.ravel()[len(show):]:
        ax.set_visible(False)
    axes.ravel()[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Test 3 – Where should the next image go? (light = good place for it)", fontsize=12)
    save(fig, out, "3_next_image_maps")

    # ------------------------------------------------------------------ figure 4
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    W = ev.geo.costWeights
    for k in COST_TERMS:
        y = W[k] * df[k].to_numpy()
        ax.plot(sizes, y, "-o", ms=4, color=TERM_COLORS[k], label=TERM_LABELS[k])
    ax.set_xlabel("number of images"); ax.set_ylabel("weighted contribution to total cost")
    ax.set_title("Test 3 – Which terms each added image improves")
    ax.legend(ncols=3, fontsize=8)
    save(fig, out, "4_components_vs_images")

    print("\nGreedy order (along, cross):")
    print(np.array2string(chosen.astype(int)))


GRIDLINE = "#9a9994"

if __name__ == "__main__":
    main()
