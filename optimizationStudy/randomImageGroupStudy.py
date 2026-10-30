"""
This file will run the random 15 image study
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.stats import spearmanr
from imageGridding import imageGridding
from imageGeometry import imageGeometry
from tatc import utils
from tatc.schemas import Instrument, Satellite, TwoLineElements
import pandas as pd
from getTLEData import getTLEData
from tatc.analysis import collect_orbit_track, OrbitCoordinate, OrbitOutput
"""
Visualize trends between 15-step angle profiles and cost.

Expected data: 75 matrices, each 16 x 2
    rows 0-14 : [along_track_angle, cross_track_angle]
    row 15    : [cost, 0]

Reads random_15_images_1.npy ... random_15_images_75.npy from pathCost.
PNGs are saved to the current working directory.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# 1. Load data  ->  angles (N, 15, 2), cost (N,)
# ---------------------------------------------------------------------------
pathCost = Path("/exports/nfs_share/Users/al62352/ResolutionHeatmaps/LS9_Random15_OptimizationData")
pathResult = Path("/exports/nfs_share/Users/al62352/ResolutionHeatmaps/LS9_CostResults")   
N_FILES = 250


def load_data():
    """Read random_15_images_{j}.npy for j = 1..75 -> (N, 16, 2) array + file numbers."""
    runs, file_nums = [], []
    for j in range(1, N_FILES + 1):
        matFileName = f"random_15_images_{j}"
        f = pathCost / f"{matFileName}.npy"
        if not f.exists():
            f = pathCost / matFileName          # in case saved without extension
        if not f.exists():
            print(f"Missing: {matFileName}, skipping")
            continue
        m = np.asarray(np.load(f, allow_pickle=True), dtype=float)
        if m.shape == (2, 16):                  # tolerate transposed storage
            m = m.T
        if m.shape != (16, 2):
            print(f"{f.name}: unexpected shape {m.shape}, skipping")
            continue
        runs.append(m)
        file_nums.append(j)
    if not runs:
        raise FileNotFoundError(f"No usable .npy files found in {pathCost}")
    print(f"Loaded {len(runs)} runs from {pathCost}")
    return np.stack(runs), np.array(file_nums)


data, file_nums = load_data()
angles = data[:, :15, :]          # (N, 15, 2)
cost = data[:, 15, 0]             # (N,)
along, cross = angles[..., 0], angles[..., 1]
N, T = along.shape
step = np.arange(1, T + 1)

order = np.argsort(cost)[::-1]    # draw worst first, best on top
norm = colors.Normalize(cost.min(), cost.max())
cmap = cm.viridis_r               # bright = low cost
n_best = max(5, N // 10)          # "best" = lowest ~10%
best = np.argsort(cost)[:n_best]
rest = np.argsort(cost)[n_best:]

# ---------------------------------------------------------------------------
# Plot 1: Path in angle space (along vs cross), colored by cost
#   -> Do good solutions trace a similar trajectory?
# ---------------------------------------------------------------------------
pts_al = along.ravel()
pts_cr = cross.ravel()
pts_cost = np.repeat(cost, T)                   # each image gets its set's cost
pts_r = np.hypot(pts_al, pts_cr)
 
rank = np.empty(N)
rank[np.argsort(cost)] = np.arange(N)           # 0 = lowest cost
rank_norm = colors.Normalize(0, N - 1)
 
 
def binned_mean(x, y, edges):
    """Mean and standard error of y within bins of x."""
    idx = np.digitize(x, edges) - 1
    centers, means, sems, counts = [], [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() >= 5:
            centers.append((edges[b] + edges[b + 1]) / 2)
            means.append(y[sel].mean())
            sems.append(y[sel].std(ddof=1) / np.sqrt(sel.sum()))
            counts.append(sel.sum())
    return map(np.array, (centers, means, sems, counts))
 
 
fig, axes = plt.subplots(2, 2, figsize=(17, 13), layout="constrained")
 
# (a) Scatter, random draw order, one shade per set (by cost rank)
ax = axes[0, 0]
rng_draw = np.random.default_rng(0)
for i in rng_draw.permutation(N):
    ax.scatter(along[i], cross[i], color=cmap(rank_norm(rank[i])),
               s=22, alpha=0.75, edgecolor="k", linewidth=0.2)
cb = fig.colorbar(cm.ScalarMappable(rank_norm, cmap), ax=ax, fraction=0.046)
tick_ranks = np.linspace(0, N - 1, 6).round().astype(int)
cb.set_ticks(tick_ranks)
cb.set_ticklabels([f"{np.sort(cost)[r]:.4f}" for r in tick_ranks])
cb.set_label("Set cost (bright = low)")
ax.set_title(f"(a) All {N * T} images, one shade per set, random draw order")
 
# (b) Mean set cost per cell
ax = axes[0, 1]
lim = np.ceil(max(np.abs(pts_al).max(), np.abs(pts_cr).max()))
hb = ax.hexbin(pts_al, pts_cr, C=pts_cost, reduce_C_function=np.mean,
               gridsize=10, extent=(-lim, lim, -lim, lim), mincnt=5,
               cmap=cmap, edgecolors="white", linewidths=0.5)
fig.colorbar(hb, ax=ax, fraction=0.046, label="Mean set cost in cell (bright = low)")
for rr in (10, 20, 30):                         # reference rings of off-nadir angle
    ax.add_patch(plt.Circle((0, 0), rr, fill=False, ls="--", lw=0.8, color="k", alpha=0.5))
ax.set_title("(b) Mean cost of sets that placed an image in each cell")
 
for ax in axes[0]:
    ax.set_xlabel("Along-track angle (deg)")
    ax.set_ylabel("Cross-track angle (deg)")
    ax.set_aspect("equal")
    ax.set_xlim(-lim - 1, lim + 1); ax.set_ylim(-lim - 1, lim + 1)
    ax.grid(alpha=0.3)
 
# (c) Mean set cost vs off-nadir distance
ax = axes[1, 0]
c_, m_, s_, n_ = binned_mean(pts_r, pts_cost, np.arange(0, pts_r.max() + 2.5, 2.5))
ax.errorbar(c_, m_, yerr=s_, fmt="-o", color="tab:purple", capsize=3)
for x, y, n in zip(c_, m_, n_):
    ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=7, color="gray")
ax.set_xlabel("Off-nadir angle √(along² + cross²)  (deg)")
ax.set_ylabel("Mean cost of sets containing an image here")
ax.set_title(f"(c) Cost vs distance from nadir   (ρ = {spearmanr(pts_r, pts_cost)[0]:+.2f})")
ax.grid(alpha=0.3)
 
# (d) Mean set cost vs |along| and |cross| separately
ax = axes[1, 1]
edges = np.arange(0, lim + 2.5, 2.5)
for vals, label, c in ((np.abs(pts_al), "|along-track|", "tab:blue"),
                       (np.abs(pts_cr), "|cross-track|", "tab:orange")):
    c_, m_, s_, _ = binned_mean(vals, pts_cost, edges)
    r = spearmanr(vals, pts_cost)[0]
    ax.errorbar(c_, m_, yerr=s_, fmt="-o", color=c, capsize=3, label=f"{label}  (ρ = {r:+.2f})")
ax.set_xlabel("|angle| (deg)")
ax.set_ylabel("Mean cost of sets containing an image here")
ax.set_title("(d) Cost vs along-track and cross-track magnitude")
ax.legend(); ax.grid(alpha=0.3)
 
fig.suptitle("Where do low-cost sets place their images?   "
             "(error bars = ±1 standard error)", fontsize=13)
fig.savefig(pathResult / "1_angle_space_regions.png", dpi=150, bbox_inches="tight")
 
 
# ---------------------------------------------------------------------------
# Plot 2: Same as 3(a) and 3(b), but only the TOP_K lowest-cost sets
# ---------------------------------------------------------------------------
TOP_K = 50
top = np.argsort(cost)[:TOP_K]                  # indices of the K lowest-cost sets
top_cost = cost[top]
top_norm = colors.Normalize(0, TOP_K - 1)       # shade by rank within the top K
 
fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), layout="constrained")
 
# (a) Scatter: each of the K sets gets its own shade; random draw order
ax = axes[0]
handles = []
for k in np.random.default_rng(0).permutation(TOP_K):
    i = top[k]
    h = ax.scatter(along[i], cross[i], color=cmap(top_norm(k)), s=45, alpha=0.9,
                   edgecolor="k", linewidth=0.4,
                   label=f"#{k + 1}  file {file_nums[i]}: {cost[i]:.4f}")
    handles.append((k, h))
handles = [h for _, h in sorted(handles, key=lambda t: t[0])]   # legend in rank order
ax.set_title(f"(a) Images from the {TOP_K} lowest-cost sets "
             f"({TOP_K * T} images, one shade per set)")
 
# (b) Mean set cost per cell, using only the top-K sets' images
ax = axes[1]
t_al, t_cr = along[top].ravel(), cross[top].ravel()
t_cost = np.repeat(top_cost, T)
hb = ax.hexbin(t_al, t_cr, C=t_cost, reduce_C_function=np.mean,
               gridsize=10, extent=(-lim, lim, -lim, lim), mincnt=1,
               cmap=cmap, edgecolors="white", linewidths=0.5)
fig.colorbar(hb, ax=ax, fraction=0.046, label="Mean set cost in cell (bright = low)")
ax.set_title(f"(b) Mean cost per cell, top {TOP_K} sets only (empty = no images)")
 
for ax in axes:
    for rr in (10, 20, 30):
        ax.add_patch(plt.Circle((0, 0), rr, fill=False, ls="--", lw=0.8,
                                color="k", alpha=0.5))
    ax.set_xlabel("Along-track angle (deg)")
    ax.set_ylabel("Cross-track angle (deg)")
    ax.set_aspect("equal")
    ax.set_xlim(-lim - 1, lim + 1); ax.set_ylim(-lim - 1, lim + 1)
    ax.grid(alpha=0.3)
 
fig.suptitle(f"Where do the {TOP_K} best sets place their images?   "
             f"(cost {top_cost.min():.4f} – {top_cost.max():.4f})", fontsize=13)
fig.savefig(pathResult / "2_angle_space_top_15.png", dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------------------
# Plot 3: Rank correlation of each angle with cost
#   -> Which steps/axes actually drive cost? (|rho| large = influential)
# ---------------------------------------------------------------------------
rho = np.array([[spearmanr(arr[:, k], cost)[0] for k in range(T)]
                for arr in (along, cross)])
fig, ax = plt.subplots(figsize=(10, 4))
w = 0.4
ax.bar(step - w / 2, rho[0], w, label="Along-track")
ax.bar(step + w / 2, rho[1], w, label="Cross-track")
ax.axhline(0, color="k", lw=0.8)
thr = 2 / np.sqrt(N)  # rough ~95% significance level
ax.axhspan(-thr, thr, color="gray", alpha=0.15, label="≈ noise level")
ax.set_xlabel("Step"); ax.set_ylabel("Spearman ρ with cost")
ax.set_xticks(step); ax.legend(); ax.grid(alpha=0.3, axis="y")
ax.set_title("Correlation of each angle with cost")
fig.savefig(pathResult / "3_correlation_with_cost.png", dpi=150, bbox_inches="tight")

# Same idea for summary features (often more revealing than per-step values)
feats = {
    "mean |along|": np.abs(along).mean(1),
    "mean |cross|": np.abs(cross).mean(1),
    "max |along|": np.abs(along).max(1),
    "max |cross|": np.abs(cross).max(1),
    "along roughness": np.abs(np.diff(along, axis=1)).sum(1),
    "cross roughness": np.abs(np.diff(cross, axis=1)).sum(1),
    "final along": along[:, -1],
    "final cross": cross[:, -1],
}
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for ax, (name, f) in zip(axes.ravel(), feats.items()):
    r = spearmanr(f, cost)[0]
    ax.scatter(f, cost, c=cost, cmap=cmap, norm=norm, s=20)
    ax.set_xlabel(name); ax.set_ylabel("Cost")
    ax.set_title(f"ρ = {r:+.2f}"); ax.grid(alpha=0.3)
fig.suptitle("Summary features vs cost")
fig.tight_layout()
fig.savefig(pathResult / "4_summary_features_vs_cost.png", dpi=150, bbox_inches="tight")

# ===========================================================================
# ORDER-INDEPENDENT ANALYSIS
# The 15 images are an unordered set, so plots 1-5 and 7 (which compare the
# same step across runs) mostly show noise. Plots 8-11 treat each run as a set.
# ===========================================================================
abs_al, abs_cr = np.abs(along), np.abs(cross)
offnadir = np.hypot(along, cross)                       # total off-nadir per image


def pairwise_spread(a, c):
    """Mean distance between every pair of images in (along, cross) space."""
    p = np.stack([a, c], -1)                            # (T, 2)
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    return d[np.triu_indices(len(a), 1)].mean()


spread = np.array([pairwise_spread(along[i], cross[i]) for i in range(N)])


def scatter_fit(ax, x, y, xlabel):
    r = spearmanr(x, y)[0]
    ax.scatter(x, y, c=y, cmap=cmap, norm=norm, s=25, edgecolor="k", lw=0.2)
    k, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, k * xs + b, "r--", lw=1.5)
    ax.scatter(x[best], y[best], s=110, facecolor="none", edgecolor="red", lw=1.2)
    ax.set_xlabel(xlabel); ax.set_ylabel("Cost")
    ax.set_title(f"ρ = {r:+.2f}"); ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# Plot 5: Cost vs mean |along| and mean |cross| (+ regression printed below)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
scatter_fit(axes[0], abs_al.mean(1), cost, "mean |along-track| (deg)")
scatter_fit(axes[1], abs_cr.mean(1), cost, "mean |cross-track| (deg)")
# Joint view: both magnitudes, colored by cost
sc = axes[2].scatter(abs_al.mean(1), abs_cr.mean(1), c=cost, cmap=cmap, norm=norm,
                     s=45, edgecolor="k", lw=0.3)
axes[2].scatter(abs_al.mean(1)[best], abs_cr.mean(1)[best], s=150,
                facecolor="none", edgecolor="red", lw=1.5, label=f"Best {n_best}")
axes[2].set_xlabel("mean |along-track|"); axes[2].set_ylabel("mean |cross-track|")
axes[2].set_title("Both magnitudes (color = cost)")
axes[2].legend(); axes[2].grid(alpha=0.3)
fig.colorbar(sc, ax=axes[2], label="Cost")
fig.suptitle("Angle magnitude vs cost (red circles = best runs)")
fig.tight_layout()
fig.savefig(pathResult / "5_magnitude_vs_cost.png", dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Plot 6: Off-nadir angle sqrt(along² + cross²) and set spread vs cost
# ---------------------------------------------------------------------------
on_feats = {
    "mean off-nadir (deg)": offnadir.mean(1),
    "min off-nadir (deg)": offnadir.min(1),
    "median off-nadir (deg)": np.median(offnadir, 1),
    "std of off-nadir (deg)": offnadir.std(1),
    "# images with off-nadir > 20°": (offnadir > 20).sum(1),
    "mean pairwise spread (deg)": spread,
}
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (name, f) in zip(axes.ravel(), on_feats.items()):
    scatter_fit(ax, f.astype(float), cost, name)
fig.suptitle("Total off-nadir angle and geometric diversity vs cost")
fig.tight_layout()
fig.savefig(pathResult / "6_offnadir_vs_cost.png", dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------------------
# Regression: cost ~ mean|along| + mean|cross|  (printed)
# ---------------------------------------------------------------------------
from scipy.stats import t as t_dist

def ols(X, y, names):
    """Least-squares fit with standardized coefficients and p-values."""
    Xs = (X - X.mean(0)) / X.std(0)                     # standardize -> comparable betas
    A = np.column_stack([np.ones(len(y)), Xs])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = len(y) - A.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    p = 2 * t_dist.sf(np.abs(beta / se), dof)
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    print(f"  R² = {r2:.3f}")
    for n, bb, pp in zip(["intercept"] + names, beta, p):
        if n != "intercept":
            print(f"    {n:<22s} Δcost per 1 std = {bb:+.5f}   p = {pp:.2g}")
    return resid


print("\n=== Regression: cost ~ mean|along| + mean|cross| ===")
ols(np.column_stack([abs_al.mean(1), abs_cr.mean(1)]), cost,
    ["mean|along|", "mean|cross|"])

print("\n=== Regression: cost ~ mean off-nadir ===")
ols(offnadir.mean(1)[:, None], cost, ["mean off-nadir"])

print("\n=== Does roughness matter beyond magnitude? ===")
print("  (If roughness p-values are large, it was only a side effect of bigger angles.)")
rough_al = np.abs(np.diff(along, axis=1)).sum(1)
rough_cr = np.abs(np.diff(cross, axis=1)).sum(1)
ols(np.column_stack([abs_al.mean(1), abs_cr.mean(1), rough_al, rough_cr]), cost,
    ["mean|along|", "mean|cross|", "along roughness", "cross roughness"])
print("  NOTE: with unordered sets, 'roughness' depends on the arbitrary row order,"
      " so treat it as noise.")
print()

print(f"Best {n_best} runs (file number: cost):")
for i in best:
    print(f"  random_15_images_{file_nums[i]}: {cost[i]:.4g}")

























# def runTest():
#     print()
#     print("###################################################################################################")
#     print()

#     # This step is defining parameters of Landsat-8 and its orbit, as well as finding the closest epoch to the desired
#     # image time (imTime). This section will collect the ground track of the orbit.
#     imTime = ["2025-12-30T17:00:00", "2025-12-30T17:03:24"]
#     noradID = 49260

#     tlePath = Path(r"/exports/nfs_share/Users/al62352/SpaceTrackTLE/tle49260.txt")
#     tles = getTLEData(tlePath)
#     line1, line2, closeTime = tles.printTLE(imTime[0])
#     print()

#     #                  FOV, Focal, Pixel #, GSD, Swath, Alt
#     sens = np.array([np.deg2rad(15.0), 0.886, 6196, 30, 185000, 705000])
#     target = np.array([30.30617, -97.88121])
    
#     oli = Instrument(name="OLI",field_of_regard=utils.swath_width_to_field_of_regard(705000, 185000),req_target_sunlit=True)
#     L9 = Satellite(name="Landsat-9",
#         orbit=TwoLineElements(tle=[line1,line2]), instruments=[oli])
#     times = pd.date_range(start=closeTime, end=imTime[1], freq="1s", tz="UTC")
#     state84 = collect_orbit_track(L9, times, coordinates = OrbitCoordinate.WGS84, sat_sunlit = True, solar_altaz = True, solar_beta = True)
#     stateECEF = collect_orbit_track(L9, times, orbit_output = OrbitOutput.POSITION_VELOCITY, coordinates = OrbitCoordinate.ECEF) 

#     aeKM = 6378.137
#     altitude = sens[5] / 1e3
#     maxLook = np.arcsin(aeKM / (aeKM + altitude))*(180/np.pi)
#     maxOffNad = round(0.75*0.75*maxLook)

#     gridded = imageGridding("49260", state84, stateECEF, times, sens, target)
#     angVec = np.array([0, maxOffNad])
#     minGSD, maxGSD = gridded.getColorbarMinMax(angVec, altitude)

#     sigmaPix = 0.5
#     eMax = 25.0
    
#     imageGeo = imageGeometry(noradID, state84, times, sens, target, angVec, minGSD, maxGSD, altitude, sigmaPix, eMax)

#     for j in range(1,251,1):
#         rng = np.random.default_rng()
#         nImages = 15
#         imAngles = rng.integers(-maxOffNad, maxOffNad+1, size=(nImages,2))

#         ims = []
#         imData = np.zeros((imAngles.shape[0],3))
#         for k in range(imAngles.shape[0]):
#             im = np.load(Path(f"/exports/nfs_share/Users/al62352/ResolutionHeatmaps/LS9_705km/GSDarray_matrix_705_km_offnadir_({imAngles[k,0]},{imAngles[k,1]}).npy"))
#             ims.append(im)
#             imData[k,:] = np.array([imAngles[k,0], imAngles[k,1], altitude])

#         cost = imageGeo.getCostFunction(ims, imData, imAngles)

#         pathCost = Path("/exports/nfs_share/Users/al62352/ResolutionHeatmaps/LS9_Random15_OptimizationData_ThreeQuarter")   

#         imsAngs = np.array(imAngles)
#         costArray = np.array([cost, 0.0])
#         imsNow = np.vstack([imsAngs, costArray])
#         matFileName = f'random_15_images_{j}'
#         fullMatPath = pathCost / matFileName
#         np.save(fullMatPath, imsNow)

#         print(f"Ran simulation {j} for the 15 random image optimization study.")


# if __name__ == "__main__":
#     runTest()
    