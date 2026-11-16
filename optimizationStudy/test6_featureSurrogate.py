"""
Test 6 - Which geometry features drive the cost? (surrogate model)
==================================================================
Question: in the language of satellite stereo (convergence angle, bisector
elevation, asymmetry, azimuth spread, B/H, off-nadir), what makes a set good?
Per-image correlations can't answer this because the images in a set have no
order. Here every set is described by order-independent features and a
machine-learning model learns cost = f(features).

What it does
  1. Builds a diverse library of sets: random N (3..20), random maximum
     off-nadir (10..30 deg), drawn either on the integer grid or uniformly in
     (off-nadir, azimuth). Optionally adds your saved random_15_images files.
  2. Computes ~18 features per set (studyCommon.setFeatures) and the five terms.
  3. Fits a gradient-boosted tree model and a standardized linear model, scored
     with 5-fold cross-validation (R^2 on held-out sets).
  4. Permutation importance on held-out folds: how much worse the model gets if
     one feature is scrambled. Done for the total cost AND for each term, so you
     see which features drive which term.
  5. Partial dependence: the model's average cost as one feature is varied,
     with the others held at their observed values - the "shape" of each effect.
  6. Feature-feature correlation, so you know which features are interchangeable.
  If the 'shap' package is installed, also writes a SHAP summary plot.

Outputs (in --out, default ./results/test6_featureSurrogate)
  features.csv                      every set: features, terms, cost
  1_model_fit.png                   cross-validated predicted vs actual
  2_feature_importance.png          permutation importance for the total cost
  3_importance_by_term.png          importance heatmap: features x cost terms
  4_partial_dependence.png          effect shapes of the top 6 features
  5_feature_correlation.png         which features carry the same information
  6_shap_summary.png                (only if shap is installed)

Run
  python test6_featureSurrogate.py --quick        # ~1 min
  python test6_featureSurrogate.py                # 3000 sets, ~4 min
  python test6_featureSurrogate.py --from-dir /exports/.../SS_R15_Data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance, partial_dependence
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import spearmanr

from studyCommon import (CostEvaluator, COST_TERMS, TERM_LABELS, SERIES, SEQ, DIV, INK2,
                         FEATURE_LABELS, save, style, commonArgs, Timer, randomSet, setFeatures,
                         polarToAngles, loadRandomStudy)


def drawSet(rng, maxOffCap):
    """One set from a deliberately varied distribution so the features span a wide range."""
    N = int(rng.integers(3, 21))
    maxOff = int(rng.integers(10, maxOffCap + 1))
    if rng.random() < 0.5:
        return randomSet(rng, N, maxOff)                      # integer grid, as before
    # uniform in solid angle within maxOff, rounded to the 1-deg grid
    th = np.degrees(np.arccos(rng.uniform(np.cos(np.radians(maxOff)), 1, N)))
    s = np.round(polarToAngles(th, rng.uniform(0, 360, N)))
    s = np.unique(s, axis=0)
    return s if len(s) >= 3 else randomSet(rng, N, maxOff)


def model():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=10, l2_regularization=1.0, random_state=0)


def main():
    p = commonArgs(__doc__.split("\n")[1], "test6_featureSurrogate")
    p.add_argument("--n-sets", type=int, default=3000)
    p.add_argument("--max-off", type=int, default=30)
    p.add_argument("--from-dir", default=None, help="also include random_15_images_*.npy sets")
    args = p.parse_args()
    if args.quick:
        args.n_sets = 250
    style()
    out = args.out
    ev = CostEvaluator(args.sensor, maxOffNadir=args.max_off)

    # ------------------------------------------------------------------ 1-2. library
    with Timer(f"Building and evaluating {args.n_sets} sets"):
        rng = np.random.default_rng(args.seed)
        sets = [drawSet(rng, args.max_off) for _ in range(args.n_sets)]
        if args.from_dir:
            extra, _ = loadRandomStudy(args.from_dir)
            sets += list(extra)
            print(f"   + {len(extra)} sets from {args.from_dir}")
        rows = []
        for s in sets:
            r = ev.evaluate(s)
            if r["ok"]:
                rows.append({**setFeatures(s), **{k: r[k] for k in COST_TERMS}, "cost": r["cost"]})
        df = pd.DataFrame(rows)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "features.csv", index=False)
        print(f"   {len(df)} usable sets")

    feats = list(FEATURE_LABELS)
    X, y = df[feats].to_numpy(float), df.cost.to_numpy()
    flab = [FEATURE_LABELS[f] for f in feats]
    cv = KFold(5, shuffle=True, random_state=0)

    # ------------------------------------------------------------------ 3. model fit
    with Timer("Cross-validated model fit"):
        predGB = cross_val_predict(model(), X, y, cv=cv)
        predLin = cross_val_predict(make_pipeline(StandardScaler(), LinearRegression()), X, y, cv=cv)
        r2 = lambda p_: 1 - np.sum((y - p_) ** 2) / np.sum((y - y.mean()) ** 2)
        r2GB, r2Lin = r2(predGB), r2(predLin)
        print(f"   held-out R²: boosted trees {r2GB:.3f}, linear {r2Lin:.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    for ax, pr, name, rr in ((axes[0], predGB, "gradient-boosted trees", r2GB),
                             (axes[1], predLin, "linear (standardized features)", r2Lin)):
        ax.scatter(y, pr, s=6, color=SERIES[0], alpha=0.5, lw=0)
        lo, hi = y.min(), y.max()
        ax.plot([lo, hi], [lo, hi], color=INK2, ls="--", lw=1)
        ax.set_xlabel("actual cost"); ax.set_ylabel("predicted cost (held-out fold)")
        ax.set_title(f"{name}: R² = {rr:.3f}")
    fig.suptitle("Test 6 – How well do geometry features alone predict the cost?", fontsize=12)
    save(fig, out, "1_model_fit")

    # ------------------------------------------------------------------ 4. importance
    def heldOutImportance(target):
        """Permutation importance averaged over the 5 held-out folds (R^2 drop)."""
        imp = np.zeros(len(feats))
        for tr, te in cv.split(X):
            m = model().fit(X[tr], target[tr])
            pi = permutation_importance(m, X[te], target[te], n_repeats=5, random_state=0)
            imp += pi.importances_mean / cv.get_n_splits()
        return imp

    with Timer("Permutation importance (total cost and each term)"):
        impTotal = heldOutImportance(y)
        impTerms = {}
        for k in COST_TERMS:
            t = df[k].to_numpy()
            impTerms[k] = heldOutImportance(t) if np.ptp(t) > 1e-6 else np.full(len(feats), np.nan)

    order = np.argsort(-impTotal)
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    ax.barh(range(len(feats)), impTotal[order], color=SERIES[0], height=0.65)
    ax.set_yticks(range(len(feats)), [flab[i] for i in order]); ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("drop in held-out R² when the feature is scrambled")
    ax.set_title("Test 6 – Which geometry features the cost depends on")
    save(fig, out, "2_feature_importance")

    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    H = np.column_stack([impTerms[k] for k in COST_TERMS] + [impTotal])[order]
    colMax = np.array([np.nanmax(c) if np.isfinite(c).any() else np.nan for c in H.T])
    Hn = H / colMax                                            # scale each column to its max
    im = ax.imshow(np.clip(Hn, 0, 1), cmap=SEQ, aspect="auto", vmin=0, vmax=1)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            txt = "const" if np.isnan(H[i, j]) else f"{H[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if np.nan_to_num(Hn[i, j]) > 0.6 else "k")
    ax.set_xticks(range(6), [TERM_LABELS[k] for k in COST_TERMS] + ["Total cost"], rotation=25, ha="right")
    ax.set_yticks(range(len(feats)), [flab[i] for i in order]); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="importance, scaled to each column's max")
    ax.set_title("Test 6 – Which features drive which cost term\n(numbers = drop in held-out R²)")
    save(fig, out, "3_importance_by_term")

    # ------------------------------------------------------------------ 5. partial dependence
    full = model().fit(X, y)
    top = order[:6]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for ax, i in zip(axes.ravel(), top):
        pdp = partial_dependence(full, X, [i], grid_resolution=40, kind="average", method="brute")
        xs, ys = pdp["grid_values"][0], pdp["average"][0]
        ax.scatter(X[:, i], y, s=4, color=INK2, alpha=0.15, lw=0, label="sets")
        ax.plot(xs, ys, color=SERIES[0], lw=2.5, label="model: average effect")
        lo, hi = np.percentile(X[:, i], [5, 95])
        ax.axvspan(X[:, i].min(), lo, color="white", alpha=0.6, lw=0)
        ax.axvspan(hi, X[:, i].max(), color="white", alpha=0.6, lw=0)
        ax.set_xlabel(flab[i]); ax.set_ylabel("cost")
        ax.set_title(f"importance {impTotal[i]:.3f}")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Test 6 – Shape of each top feature's effect (faded = outside the 5–95% data range)",
                 fontsize=12)
    save(fig, out, "4_partial_dependence")

    # ------------------------------------------------------------------ 6. feature correlation
    rho = spearmanr(X).statistic
    o = leaves_list(linkage(1 - np.abs(rho[np.triu_indices(len(feats), 1)]), "average"))
    fig, ax = plt.subplots(figsize=(11, 9.5), layout="constrained")
    im = ax.imshow(rho[np.ix_(o, o)], cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats)), [flab[i] for i in o], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(feats)), [flab[i] for i in o], fontsize=8); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Spearman ρ")
    ax.set_title("Test 6 – Feature correlation (clustered). Strongly correlated features share importance.")
    save(fig, out, "5_feature_correlation")

    # ------------------------------------------------------------------ optional SHAP
    try:
        import shap
        expl = shap.TreeExplainer(full)
        sv = expl.shap_values(X[:2000])
        shap.summary_plot(sv, X[:2000], feature_names=flab, show=False, max_display=12)
        fig = plt.gcf(); fig.suptitle("Test 6 – SHAP summary", fontsize=12)
        save(fig, out, "6_shap_summary")
    except ImportError:
        print("   (shap not installed - skipping the SHAP plot: pip install shap)")

    print("\nTop features for the total cost:")
    for i in order[:8]:
        print(f"   {flab[i]:<34s} {impTotal[i]:.3f}")


if __name__ == "__main__":
    main()
