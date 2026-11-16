"""
Test 4 - Structured geometry families
=====================================
Question: instead of random sets, what do simple, taskable patterns score, and
which parameters (off-nadir angle, number of views, pass offset) are optimal?
Each family maps directly onto how a satellite would actually collect.

Families
  A. Ring          k views at the same off-nadir angle theta, evenly spaced in
                   azimuth (first view straight forward), with or without an extra
                   nadir view. Models multi-pass collection from different azimuths.
  B. Single pass   n views along-track from -theta_a to +theta_a at a fixed
                   cross-track offset c0. Models one agile pass (fore/nadir/aft...).
  C. Two passes    two passes at cross-track +c and -c, each with m views from
                   -theta_a to +theta_a along-track. Models ascending + descending or
                   two neighboring passes on opposite sides of the target.

What it does
  Evaluates every member of each family on a parameter grid (members that fail the
  overlap requirement or the agility limit are left blank), then compares the
  best member of each family (and the best of the random sets, if you pass
  --random-best) and shows the champion geometries and their cost terms.

Outputs (in --out, default ./results/test4_geometryFamilies)
  families.csv                    every evaluated member: family, parameters, terms
  1_ring_heatmaps.png             ring cost vs (k, theta), with and without nadir
  2_ring_curves.png               ring cost vs theta for a few k
  3_single_pass_heatmaps.png      single-pass cost vs (theta_a, c0) for each n
  4_two_pass_heatmaps.png         two-pass cost vs (theta_a, c) for each m
  5_family_champions.png          best member of each family: geometry + terms

Run
  python test4_geometryFamilies.py --quick       # ~20 s
  python test4_geometryFamilies.py               # ~1 min
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from studyCommon import (CostEvaluator, COST_TERMS, TERM_LABELS, TERM_COLORS, SERIES, SEQ, INK2,
                         angleAxes, save, style, commonArgs, Timer, polarToAngles, makeEvaluator, resolveArgs)


# ------------------------------------------------------------------ family generators
def ringSet(k, theta, nadir=False, phase=0.0):
    az = phase + 360.0 * np.arange(k) / k
    s = polarToAngles(np.full(k, theta), az)
    return np.vstack([[0.0, 0.0], s]) if nadir else s


def singlePass(n, thetaA, c0):
    return np.column_stack([np.linspace(-thetaA, thetaA, n), np.full(n, c0)])


def twoPass(m, thetaA, c):
    a = np.linspace(-thetaA, thetaA, m)
    return np.vstack([np.column_stack([a, np.full(m, c)]), np.column_stack([a, np.full(m, -c)])])


def heat(ax, Z, xv, yv, xlabel, ylabel, title, fig, vmin=None, vmax=None):
    """Heatmap with the best cell starred. Z is (len(yv), len(xv))."""
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=SEQ, vmin=vmin, vmax=vmax,
                   extent=(xv[0] - (xv[1] - xv[0]) / 2, xv[-1] + (xv[1] - xv[0]) / 2,
                           -0.5, len(yv) - 0.5))
    ax.set_yticks(range(len(yv)), [f"{v:g}" for v in yv])
    if np.isfinite(Z).any():
        iy, ix = np.unravel_index(np.nanargmin(Z), Z.shape)
        ax.scatter(xv[ix], iy, marker="*", s=200, color=SERIES[3], edgecolor="k", zorder=3)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="cost (darker = higher)")


def main():
    p = commonArgs(__doc__.split("\n")[1], "test4_geometryFamilies")
    p.add_argument("--random-best", type=float, default=None,
                   help="best random-set cost to draw as a reference line (e.g. from test 2)")
    args = resolveArgs(p.parse_args())
    style()
    out = args.out
    ev = makeEvaluator(args)
    M = args.max_off

    if args.quick:
        ks, thetas = [3, 5, 8], np.arange(5, M + 0.1, 5.0)
        ns, thetaAs, c0s = [3, 5], np.arange(5, M + 0.1, 5.0), np.arange(0, M + 0.1, 10.0)
        ms, cs = [2, 3], np.arange(5, M + 0.1, 5.0)
    else:
        ks, thetas = [2, 3, 4, 5, 6, 8, 10, 12, 15], np.arange(2.5, M + 0.1, 2.5)
        ns, thetaAs, c0s = [3, 5, 7, 9], np.arange(2.5, M + 0.1, 2.5), np.arange(0, M + 0.1, 5.0)
        ms, cs = [2, 3, 5, 7], np.arange(2.5, M + 0.1, 2.5)

    rows = []

    def add(family, params, angles):
        r = ev.evaluate(angles)
        rows.append({"family": family, **params, "nImages": len(angles),
                     **{k: r[k] for k in COST_TERMS}, "cost": r["cost"],
                     "angles": ";".join(f"{a:.2f},{c:.2f}" for a, c in angles)})
        return r["cost"]

    with Timer("Family A: rings"):
        ringZ = {False: np.full((len(ks), len(thetas)), np.nan), True: np.full((len(ks), len(thetas)), np.nan)}
        for nad in (False, True):
            for i, k in enumerate(ks):
                for j, t in enumerate(thetas):
                    ringZ[nad][i, j] = add("ring" + ("+nadir" if nad else ""),
                                           dict(k=k, theta=t, nadir=nad), ringSet(k, t, nad))
    with Timer("Family B: single pass"):
        spZ = {}
        for n in ns:
            Z = np.full((len(c0s), len(thetaAs)), np.nan)
            for i, c0 in enumerate(c0s):
                for j, ta in enumerate(thetaAs):
                    Z[i, j] = add("single pass", dict(n=n, thetaA=ta, c0=c0), singlePass(n, ta, c0))
            spZ[n] = Z
    with Timer("Family C: two passes"):
        tpZ = {}
        for m in ms:
            Z = np.full((len(cs), len(thetaAs)), np.nan)
            for i, c in enumerate(cs):
                for j, ta in enumerate(thetaAs):
                    Z[i, j] = add("two passes", dict(m=m, thetaA=ta, c=c), twoPass(m, ta, c))
            tpZ[m] = Z

    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "families.csv", index=False)
    print(f"   {len(df)} geometries evaluated, {df.cost.isna().sum()} infeasible (overlap or agility limit)")

    # ------------------------------------------------------------------ figure 1: ring heatmaps
    allRing = np.concatenate([ringZ[False].ravel(), ringZ[True].ravel()])
    vmin, vmax = np.nanmin(allRing), np.nanmax(allRing)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), layout="constrained")
    for ax, nad in zip(axes, (False, True)):
        heat(ax, ringZ[nad], thetas, ks, "off-nadir angle of the ring θ (°)", "views on the ring k",
             f"Ring {'+ one nadir view' if nad else 'only'}", fig, vmin, vmax)
        # best theta for every k
        bestT = thetas[np.nanargmin(ringZ[nad], axis=1)]
        ax.plot(bestT, range(len(ks)), "o", color="white", mec="k", ms=5, label="best θ for each k")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Test 4 – Ring family: k views at off-nadir θ, evenly spaced in azimuth "
                 "(★ = best overall, blank = infeasible: overlap or agility limit)", fontsize=12)
    save(fig, out, "1_ring_heatmaps")

    # ------------------------------------------------------------------ figure 2: ring curves
    pick = [k for k in (3, 5, 8, 15) if k in ks][:4]
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    for c, k in zip(SERIES, pick):
        i = ks.index(k)
        ax.plot(thetas, ringZ[False][i], "-", color=c, label=f"k = {k}")
        ax.plot(thetas, ringZ[True][i], "--", color=c, lw=1.4)
        ax.annotate(f"k={k}", (thetas[-1], ringZ[False][i, -1]), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.plot([], [], "k-", label="ring only"); ax.plot([], [], "k--", lw=1.4, label="ring + nadir")
    if args.random_best:
        ax.axhline(args.random_best, color=INK2, ls=":", lw=1.2, label="best random 15-image set")
    ax.set_xlabel("off-nadir angle of the ring θ (°)"); ax.set_ylabel("total cost")
    ax.set_title("Test 4 – Ring cost vs off-nadir angle")
    ax.legend(ncols=2, fontsize=8)
    save(fig, out, "2_ring_curves")

    # ------------------------------------------------------------------ figure 3/4: pass heatmaps
    for name, Zd, keyName, yv, ylab, fname, title in (
            ("single pass", spZ, "n", c0s, "cross-track offset c₀ (°)", "3_single_pass_heatmaps",
             "Single pass: n views from −θa to +θa along-track at cross-track c₀"),
            ("two passes", tpZ, "m", cs, "pass offsets ±c (°)", "4_two_pass_heatmaps",
             "Two passes at ±c, each with m views from −θa to +θa along-track")):
        keys = list(Zd)
        allZ = np.concatenate([z.ravel() for z in Zd.values()])
        fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.8), layout="constrained",
                                 squeeze=False)
        for ax, kk in zip(axes[0], keys):
            nImg = kk if name == "single pass" else 2 * kk
            heat(ax, Zd[kk], thetaAs, yv, "along-track half-span θa (°)", ylab,
                 f"{keyName} = {kk}  ({nImg} images)", fig, np.nanmin(allZ), np.nanmax(allZ))
        fig.suptitle(f"Test 4 – {title}  (★ = best in panel, blank = infeasible: overlap or agility limit)",
                     fontsize=12)
        save(fig, out, fname)

    # ------------------------------------------------------------------ figure 5: champions
    fams = ["ring", "ring+nadir", "single pass", "two passes"]
    champs = [df[df.family == f].nsmallest(1, "cost").iloc[0] for f in fams if (df.family == f).any()]
    fig = plt.figure(figsize=(5 * len(champs), 10), layout="constrained")
    gs = fig.add_gridspec(2, len(champs), height_ratios=[1.15, 1])
    W = ev.geo.costWeights
    for i, ch in enumerate(champs):
        ax = fig.add_subplot(gs[0, i])
        A = np.array([[float(v) for v in s.split(",")] for s in ch.angles.split(";")])
        ax.scatter(A[:, 0], A[:, 1], s=60, color=SERIES[i], edgecolor="k", lw=0.5, zorder=3)
        params = {k: ch[k] for k in ("k", "theta", "n", "thetaA", "c0", "m", "c") if pd.notna(ch.get(k))}
        ptxt = ", ".join(f"{k}={v:g}" for k, v in params.items())
        angleAxes(ax, M, title=f"{ch.family}: {ptxt}\n{int(ch.nImages)} images, cost {ch.cost:.4f}")
        ax.title.set_fontsize(9)
    ax = fig.add_subplot(gs[1, :])
    bottom = np.zeros(len(champs))
    for k in COST_TERMS:
        v = np.array([W[k] * ch[k] for ch in champs])
        ax.bar(range(len(champs)), v, bottom=bottom, color=TERM_COLORS[k], width=0.5,
               edgecolor="white", lw=1.5, label=TERM_LABELS[k])
        bottom += v
    for i, b in enumerate(bottom):
        ax.annotate(f"{b:.4f}", (i, b), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    if args.random_best:
        ax.axhline(args.random_best, color=INK2, ls=":", lw=1.2, label="best random 15-image set")
    ax.set_xticks(range(len(champs)), [f"{ch.family}\n({int(ch.nImages)} images)" for ch in champs])
    ax.set_ylabel("weighted contribution to total cost")
    ax.legend(ncols=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.suptitle("Test 4 – Best member of each family. Note the image counts differ; "
                 "test 5 compares at equal image count.", fontsize=12)
    save(fig, out, "5_family_champions")

    print("\nBest member of each family:")
    for ch in champs:
        print(f"   {ch.family:<12s} {int(ch.nImages):>2d} images  cost {ch.cost:.4f}")


if __name__ == "__main__":
    main()
