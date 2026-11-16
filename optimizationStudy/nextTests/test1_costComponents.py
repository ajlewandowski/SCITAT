"""
Test 1 - Cost-term decomposition and weight sensitivity
========================================================
Question: which of the cost terms actually decides which geometry "wins",
and does the winner change if the 0.2/0.2/0.2/0.2/0.2 weights change?

What it does
  1. Draws random 15-image sets (integer along/cross pointings inside the sensor's
     off-nadir cone, seeded by [seed, j]) - or loads your saved sets with
     --from-dir - and stores every term per set, not just the total.
     It also reports how many sets meet the overlap requirement (feasible). The
     term analysis uses every set; rankings use feasible sets only.
  2. Variance decomposition: how much of the spread in total cost each weighted
     term is responsible for. A term with ~0% share cannot change any ranking.
  3. Correlation between terms (are two terms measuring the same thing?).
  4. Weight sensitivity:
       a. one-at-a-time: raise one weight by +0.1 (renormalize) -> Spearman rank
          correlation with the baseline ranking and overlap of the top-10% sets
       b. Monte Carlo: 2000 random weight vectors (Dirichlet) -> how often each set
          lands in the top 10%  ("robustly good" vs "good only for these weights")
  5. Pareto front: sets that no other set beats on every term at once.
  6. Geometry of the winner when one term dominates the weights.
  7. Resolution check: decimated (~320 px) vs full-resolution images on a few sets,
     so you know the faster setting used by every other test is safe.

Outputs (in --out, default ./results/test1_costComponents)
  components.csv                   every set: angles, terms, total cost, feasibility
  1_term_distributions.png         raw terms and weighted contributions
  2_variance_share_and_corr.png    variance share + term correlation matrix
  3_weight_sensitivity.png         one-at-a-time and Monte Carlo weight tests
  4_pareto.png                     Pareto front on the two most influential terms
  5_winner_by_weighting.png        best geometry under each term-dominant weighting
  6_resolution_check.png           decimated vs full-res cost
  7_usability_and_incidence_maps.png usable-pixel share and ground incidence angle per pointing

Run
  python test1_costComponents.py --quick          # ~1 min smoke test
  python test1_costComponents.py                  # 500 sets, ~1 min + resolution check
  python test1_costComponents.py --from-dir /exports/.../SS_R15_Data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from studyCommon import (imageGeometry, CostEvaluator, COST_TERMS, TERM_LABELS, TERM_COLORS, SERIES, SEQ, DIV,
                         INK2, angleAxes, save, style, commonArgs, Timer, loadRandomStudy,
                         weightedCost, randomSet, incidenceAngle, coneGrid, makeEvaluator,
                         resolveArgs)


def paretoMask(F):
    """True for rows of F (n, m) that no other row beats on every column (minimization)."""
    n = len(F)
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if np.any(dominated):
            keep[i] = False
    return keep


def topOverlap(costA, costB, frac=0.10):
    """Fraction of the top-frac sets under costA that are also top-frac under costB."""
    k = max(1, int(round(frac * len(costA))))
    a = set(np.argsort(costA)[:k])
    b = set(np.argsort(costB)[:k])
    return len(a & b) / k


def main():
    p = commonArgs(__doc__.split("\n")[1], "test1_costComponents")
    p.add_argument("--n-sets", type=int, default=500, help="random sets to draw")
    p.add_argument("--from-dir", default=None,
                   help="load angles from random_15_images_*.npy instead of drawing new ones")
    p.add_argument("--n-res-check", type=int, default=8,
                   help="sets to re-evaluate at full resolution (slow: ~2 s each)")
    args = resolveArgs(p.parse_args())
    if args.quick:
        args.n_sets, args.n_res_check = 40, 2
    style()
    out = args.out
    ev = makeEvaluator(args)
    W0 = dict(ev.geo.costWeights)

    # ------------------------------------------------------------------ 1. evaluate
    with Timer("Evaluating random sets"):
        if args.from_dir:
            sets, _ = loadRandomStudy(args.from_dir)
        else:
            sets = np.stack([randomSet(np.random.default_rng([args.seed, j]), 15, args.max_off)
                             for j in range(1, args.n_sets + 1)])
        rows = []
        for j, s in enumerate(sets):
            r = ev.evaluate(s)
            rows.append({"set": j + 1, **{k: r[k] for k in COST_TERMS},
                         "cost": r["costUnconstrained"], "feasible": r["feasible"],
                         "overlapMin": r["overlapMin"], "usableMean": r["usableMean"],
                         "computed": r["computed"], "angles": ";".join(f"{a:g},{c:g}" for a, c in s)})
        df = pd.DataFrame(rows)
        keep = df.computed.to_numpy()
        print(f"   {len(df)} sets, {(~keep).sum()} could not be computed and were dropped")
        df = df[keep].reset_index(drop=True)
        sets = sets[keep]
        feas = df.feasible.to_numpy()
        print(f"   {feas.sum()} of {len(df)} sets ({feas.mean():.0%}) meet the "
              f"{ev.geo.minOverlap:.0%} overlap requirement; median worst-image overlap "
              f"{df.overlapMin.median():.0%}")
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "components.csv", index=False)

    E = df[list(COST_TERMS)].to_numpy()             # (n, nT) raw terms
    nT = len(COST_TERMS)
    w0 = np.array([W0[k] for k in COST_TERMS])
    active = w0 > 0                                 # zero-weight terms are reported only
    cost0 = E @ w0

    # ------------------------------------------------------------------ 2. variance share
    # Var(total) = sum_i Cov(w_i e_i, total), so each share is that term's contribution
    contrib = E * w0
    share = np.array([np.cov(contrib[:, i], cost0)[0, 1] for i in range(nT)]) / cost0.var(ddof=1)
    for k in COST_TERMS:
        if W0[k] == 0:
            print(f"   note: {TERM_LABELS[k]} has weight 0 (reported, not scored)")
        elif np.ptp(df[k]) < 1e-6:
            print(f"   WARNING: {TERM_LABELS[k]} is constant ({df[k].iloc[0]:.4f}) over every set -"
                  " it shifts the total but can never change which geometry wins.")
        elif df[k].abs().max() < 1e-3:
            print(f"   WARNING: {TERM_LABELS[k]} is ~0 everywhere (max {df[k].abs().max():.1e}).")
    print("\nShare of total-cost variance per term:")
    for k, s in zip(COST_TERMS, share):
        print(f"   {TERM_LABELS[k]:<18s} {s:+6.1%}   (raw range {df[k].min():.3f} – {df[k].max():.3f})")

    # ------------------------------------------------------------------ figure 1
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), layout="constrained")
    for ax, data, title, ylab in ((axes[0], E, "(a) Raw cost terms (each 0–1)", "term value"),
                                  (axes[1], contrib, "(b) Weighted contribution to total cost",
                                   "weight × term")):
        bp = ax.boxplot([data[:, i] for i in range(nT)], widths=0.5, showfliers=False,
                        patch_artist=True, medianprops=dict(color="k"))
        for patch, k in zip(bp["boxes"], COST_TERMS):
            patch.set_facecolor(TERM_COLORS[k]); patch.set_alpha(0.35)
        jit = np.random.default_rng(0)
        for i, k in enumerate(COST_TERMS):
            ax.scatter(i + 1 + jit.uniform(-0.15, 0.15, len(data)), data[:, i], s=6,
                       color=TERM_COLORS[k], alpha=0.5, lw=0)
        ax.set_xticks(range(1, nT + 1), [TERM_LABELS[k] for k in COST_TERMS], rotation=15)
        ax.set_ylabel(ylab); ax.set_title(title)
    fig.suptitle(f"Test 1 – What the cost terms look like over {len(df)} random 15-image sets "
                 f"({args.sensor})", fontsize=12)
    save(fig, out, "1_term_distributions")

    # ------------------------------------------------------------------ figure 2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1, 1.1]))
    ax = axes[0]
    ax.barh(range(nT), share, color=[TERM_COLORS[k] for k in COST_TERMS], height=0.6)
    for i, s in enumerate(share):
        s = 0.0 if abs(s) < 5e-4 else s          # avoid a "-0.0%" label
        # labels always sit just right of max(bar end, 0) so negative bars don't
        # collide with the term names on the axis
        ax.annotate(f"{s:.1%}", (max(s, 0), i), xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=9)
    ax.set_yticks(range(nT), [TERM_LABELS[k] for k in COST_TERMS]); ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("share of the variance in total cost")
    ax.set_title("(a) Which terms move the total? (near 0 = cannot change rankings)")

    ax = axes[1]
    names = list(COST_TERMS) + ["cost"]
    M = np.column_stack([E, cost0])
    const = np.ptp(M, axis=0) < 1e-9                 # a constant term has no ranking
    rho = np.full((M.shape[1], M.shape[1]), np.nan)
    v = ~const
    rho[np.ix_(v, v)] = spearmanr(M[:, v]).statistic if v.sum() > 2 else np.corrcoef(M[:, v].T)
    im = ax.imshow(rho, cmap=DIV, vmin=-1, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            txt = "const" if np.isnan(rho[i, j]) else f"{rho[i, j]:+.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if abs(np.nan_to_num(rho[i, j])) > 0.6 else "k")
    lab = [TERM_LABELS.get(n, "Total cost") for n in names]
    ax.set_xticks(range(len(names)), lab, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), lab); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Spearman ρ")
    ax.set_title("(b) Rank correlation between terms (|ρ| near 1 = redundant)")
    fig.suptitle("Test 1 – Variance decomposition and term redundancy", fontsize=12)
    save(fig, out, "2_variance_share_and_corr")

    # ------------------------------------------------------------------ 4. weight sensitivity
    with Timer("Weight sensitivity"):
        oat = []
        oatTerms = [k for k, a in zip(COST_TERMS, active) if a]
        for k in oatTerms:
            i = COST_TERMS.index(k)
            w = w0.copy(); w[i] += 0.1; w /= w.sum()
            c = E @ w
            oat.append((spearmanr(cost0, c).statistic, topOverlap(cost0, c)))
        oat = np.array(oat)

        rng = np.random.default_rng(args.seed)
        nMC = 2000
        Wmc = np.zeros((nMC, nT))                          # uniform over the active terms' weights
        Wmc[:, active] = rng.dirichlet(np.ones(active.sum()), size=nMC)
        Cmc = E @ Wmc.T                                    # (n, nMC)
        kTop = max(1, int(round(0.10 * len(df))))
        inTop = np.zeros(len(df))
        rhoMC = np.empty(nMC)
        for m in range(nMC):
            inTop[np.argsort(Cmc[:, m])[:kTop]] += 1
            rhoMC[m] = spearmanr(cost0, Cmc[:, m]).statistic
        inTop /= nMC

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")
    ax = axes[0]
    x = np.arange(len(oatTerms)); wbar = 0.38
    ax.bar(x - wbar / 2, oat[:, 0], wbar, color=SERIES[0], label="Spearman ρ with baseline ranking")
    ax.bar(x + wbar / 2, oat[:, 1], wbar, color=SERIES[1], label="top-10% overlap with baseline")
    ax.set_xticks(x, [f"+0.1 on\n{TERM_LABELS[k]}" for k in oatTerms], fontsize=8)
    ax.set_ylim(0, 1.05); ax.legend(loc="lower left", fontsize=8)
    ax.set_title("(a) Nudge one weight by +0.1")

    ax = axes[1]
    ax.hist(rhoMC, bins=40, color=SERIES[0], alpha=0.8)
    ax.axvline(np.median(rhoMC), color="k", ls="--", lw=1)
    ax.annotate(f"median ρ = {np.median(rhoMC):.2f}", (np.median(rhoMC), ax.get_ylim()[1] * 0.9),
                xytext=(-6, 0), textcoords="offset points", ha="right", fontsize=9)
    ax.set_xlabel("Spearman ρ with baseline ranking"); ax.set_ylabel("number of weight vectors")
    ax.set_title(f"(b) {nMC} random weight vectors")

    ax = axes[2]
    rank0 = np.argsort(np.argsort(cost0))
    ax.scatter(rank0 + 1, inTop, s=10, color=SERIES[0], alpha=0.7, lw=0)
    ax.axvline(kTop + 0.5, color=INK2, ls="--", lw=0.8)
    ax.annotate("baseline top 10%", (kTop, 0.95), xytext=(4, 0), textcoords="offset points", fontsize=8)
    ax.set_xlabel("rank under baseline weights (1 = best)")
    ax.set_ylabel("fraction of weight vectors that put this set in the top 10%")
    ax.set_title("(c) Is a good set good for any weighting?")
    fig.suptitle("Test 1 – How sensitive is the ranking to the cost weights?", fontsize=12)
    save(fig, out, "3_weight_sensitivity")

    # ------------------------------------------------------------------ 5. Pareto front
    # Only terms that actually vary are useful objectives
    useful = [i for i in range(nT) if active[i] and np.ptp(E[:, i]) > 1e-3]
    par = paretoMask(E[:, useful])
    print(f"\nPareto front (on {', '.join(TERM_LABELS[COST_TERMS[i]] for i in useful)}): "
          f"{par.sum()} of {len(df)} sets")
    df["pareto"] = par
    df.to_csv(out / "components.csv", index=False)
    top2 = np.argsort(-np.abs(share))[:2]
    a, b = top2
    fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
    sc = ax.scatter(E[:, a], E[:, b], c=cost0, cmap=SEQ, s=18, edgecolor="none")
    ax.scatter(E[~feas, a], E[~feas, b], marker="x", s=14, color=INK2, lw=0.8,
               label=f"fails the {ev.geo.minOverlap:.0%} overlap requirement")
    ax.scatter(E[par, a], E[par, b], s=60, facecolor="none", edgecolor=SERIES[1], lw=1.4,
               label=f"Pareto set ({par.sum()} sets, all {len(useful)} varying terms)")
    i0 = np.argmin(np.where(feas, cost0, np.inf)) if feas.any() else np.argmin(cost0)
    ax.scatter(E[i0, a], E[i0, b], marker="*", s=220, color=SERIES[3], edgecolor="k", zorder=5,
               label="best feasible set, current weights")
    fig.colorbar(sc, ax=ax, label="total cost (darker = higher)")
    ax.set_xlabel(f"{TERM_LABELS[COST_TERMS[a]]} term"); ax.set_ylabel(f"{TERM_LABELS[COST_TERMS[b]]} term")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Test 1 – Trade-off between the two most influential terms")
    save(fig, out, "4_pareto")

    # ------------------------------------------------------------------ 6. winner by weighting
    cases = [("current weights", w0)]
    for i, k in enumerate(COST_TERMS):
        if not active[i]:
            continue
        w = np.where(active, 0.1, 0.0); w[i] = 0.6; w /= w.sum()
        cases.append((f"{TERM_LABELS[k]} dominant", w))
    fig, axes = plt.subplots(1, len(cases), figsize=(4 * len(cases), 4.6), layout="constrained")
    for ax, (name, w) in zip(axes, cases):
        c = np.where(feas, E @ w, np.inf) if feas.any() else E @ w
        best = np.argmin(c)
        s = sets[best]
        ax.scatter(s[:, 0], s[:, 1], s=55, color=SERIES[0], edgecolor="k", lw=0.5, zorder=3)
        angleAxes(ax, args.max_off, title=f"{name}\nset {df.set[best]}, cost {cost0[best]:.3f} (current wts)")
        ax.title.set_fontsize(9)
    fig.suptitle("Test 1 – Best feasible random set when one term dominates the weights. "
                 "Same set in every panel = weights don't matter.", fontsize=12)
    save(fig, out, "5_winner_by_weighting")

    # ------------------------------------------------------------------ 7. resolution check
    with Timer(f"Resolution check on {args.n_res_check} sets (full-res is slow)"):
        # True full resolution for SkySat-size images; large sensors (Landsat 6196 px,
        # WorldView 42k px) would need several GB per set, so cap the reference at ~2048 px
        pix = int(ev.sens[2])
        evFull = makeEvaluator(args, target_px=None if pix <= 2600 else 2048)
        idx = np.linspace(0, len(df) - 1, args.n_res_check).round().astype(int)
        fast = np.array([cost0[i] for i in idx])
        full = np.array([evFull.evaluate(sets[i])["costUnconstrained"] for i in idx])
        err = (fast - full) / full
        rhoRes = spearmanr(fast, full).statistic if len(idx) > 2 else np.nan
        print(f"   decimated vs full-res total cost: mean |diff| = {np.mean(np.abs(err)):.2%}, "
              f"max = {np.max(np.abs(err)):.2%}, rank agreement ρ = {rhoRes:.3f}")
        print("   (a constant offset does not matter; the rank agreement is what counts)")
    fig, ax = plt.subplots(figsize=(6, 5.5), layout="constrained")
    ax.scatter(full, fast, s=40, color=SERIES[0], edgecolor="k", lw=0.4, zorder=3)
    lo, hi = min(full.min(), fast.min()), max(full.max(), fast.max())
    ax.plot([lo, hi], [lo, hi], color=INK2, ls="--", lw=1, label="1:1")
    ax.set_xlabel("total cost, full resolution"); ax.set_ylabel(f"total cost, decimated (step {ev.step})")
    ax.set_title(f"Test 1 – Resolution check\nmax diff {np.max(np.abs(err)):.1%}, rank ρ = {rhoRes:.3f}")
    ax.legend()
    save(fig, out, "6_resolution_check")

    # ------------------------------------------------------------------ 8. grade maps
    with Timer("Usability and incidence maps"):
        M = args.max_off
        g = np.arange(-np.ceil(M), np.ceil(M) + 1, 2.0)
        U = np.full((len(g), len(g)), np.nan)
        I = np.full_like(U, np.nan)
        for i, a_ in enumerate(g):
            for j, c_ in enumerate(g):
                off = imageGeometry.offNadir([[a_, c_]])[0]
                if off > M + 1e-9:
                    continue                          # outside the agility limit
                U[i, j] = ev.geo.getUsableFraction(ev.image(a_, c_))
                I[i, j] = incidenceAngle(off, None if args.flat_earth else ev.alt)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    ext = (g[0] - 1, g[-1] + 1, g[0] - 1, g[-1] + 1)
    for ax, Z, title, lab, lim in (
            (axes[0], U, f"(a) Usability grade: share of pixels with GSD ≤ {ev.geo.gsdLimit():.3g} m",
             "usable share (darker = more usable)", (0, 1)),
            (axes[1], I, "(b) Incidence angle at the target "
                         + ("(flat Earth = off-nadir)" if args.flat_earth else "(spherical Earth)"),
             "incidence angle (°)", (0, None))):
        im = ax.imshow(Z.T, origin="lower", extent=ext, cmap=SEQ, vmin=lim[0], vmax=lim[1],
                       interpolation="nearest")
        angleAxes(ax, M, title=title, zeroLines=False)
        ax.title.set_fontsize(10); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, label=lab)
    fig.suptitle(f"Test 1 – Per-image limits for {args.sensor} at {ev.alt:g} km "
                 f"(solid ring = {M:g}° agility limit; blank = beyond it)", fontsize=12)
    save(fig, out, "7_usability_and_incidence_maps")

    print("\nTop 5 feasible sets under the current weights:")
    for i in np.argsort(np.where(feas, cost0, np.inf))[:min(5, feas.sum())]:
        print(f"   set {df.set[i]:>4d}  cost {cost0[i]:.4f}  Monte-Carlo top-10% rate {inTop[i]:.0%}")


if __name__ == "__main__":
    main()
