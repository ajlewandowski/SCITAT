"""
This file will run a ringed study, 0-10deg, 10-20deg, 20-30deg, 30-40deg, 40-50deg
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import spearmanr, kruskal, mannwhitneyu
from imageGriddingClass import imageGridding
from imageGeometryClass import imageGeometry
from tatc import utils
from tatc.schemas import Instrument, Satellite, TwoLineElements
import pandas as pd
from processingTLEs.getTLEData import getTLEData
from tatc.analysis import collect_orbit_track, OrbitCoordinate, OrbitOutput

def runTest(pathCost):
    print()
    print("###################################################################################################")
    print()

    ringLimits = np.array([0, 10, 20, 30, 40, 48])

    # This step is defining parameters of Landsat-8 and its orbit, as well as finding the closest epoch to the desired
    # image time (imTime). This section will collect the ground track of the orbit.
    imTime = ["2025-12-30T17:00:00", "2025-12-30T17:03:24"]
    noradID = 49260

    tlePath = Path(r"/exports/nfs_share/Users/al62352/SpaceTrackTLE/tle49260.txt")
    tles = getTLEData(tlePath)
    line1, line2, closeTime = tles.printTLE(imTime[0])
    print()

    #                  FOV, Focal, Pixel #, GSD, Swath, Alt
    sens = np.array([np.deg2rad(15.0), 0.886, 6196, 30, 185000, 705000])
    target = np.array([30.30617, -97.88121])
    
    oli = Instrument(name="OLI",field_of_regard=utils.swath_width_to_field_of_regard(705000, 185000),req_target_sunlit=True)
    L9 = Satellite(name="Landsat-9",
        orbit=TwoLineElements(tle=[line1,line2]), instruments=[oli])
    times = pd.date_range(start=closeTime, end=imTime[1], freq="1s", tz="UTC")
    state84 = collect_orbit_track(L9, times, coordinates = OrbitCoordinate.WGS84, sat_sunlit = True, solar_altaz = True, solar_beta = True)
    stateECEF = collect_orbit_track(L9, times, orbit_output = OrbitOutput.POSITION_VELOCITY, coordinates = OrbitCoordinate.ECEF) 

    aeKM = 6378.137
    altitude = sens[5] / 1e3
    maxLook = np.arcsin(aeKM / (aeKM + altitude))*(180/np.pi)
    maxOffNad = round(ringLimits[-1])

    gridded = imageGridding("49260", state84, stateECEF, times, sens, target)
    angVec = np.array([0, maxOffNad])
    minGSD, maxGSD = gridded.getColorbarMinMax(angVec, altitude)

    sigmaPix = 0.5
    eMax = 25.0
    
    imageGeo = imageGeometry(noradID, state84, times, sens, target, angVec, minGSD, maxGSD, altitude, sigmaPix, eMax)

    # Every integer (along, cross) pair we have a heatmap for, and its true off-nadir angle
    axisMax = int(ringLimits[-1])
    grid = np.arange(-axisMax, axisMax + 1)
    allAngles = np.array(np.meshgrid(grid, grid, indexing="ij")).reshape(2, -1).T
    allOffNadir = imageGeometry.offNadir(allAngles)

    BASE_SEED = 20261113            # change to draw a new, still reproducible, batch
    nImages = 15

    for i in range(len(ringLimits) - 1):
        angMax = ringLimits[i+1]
        angMin = ringLimits[i]

        # Pointings whose TRUE off-nadir angle lies in [angMin, angMax)
        inRing = allAngles[(allOffNadir >= angMin) & (allOffNadir < angMax)]
        print(f"Ring {angMin}-{angMax} deg: {len(inRing)} candidate pointings")

        for j in range(25,101,1):
            # Seed from (base, ring, set) so any single set can be regenerated
            rng = np.random.default_rng([BASE_SEED, i, j])
            # Uniform over the ring's grid points, no repeated pointing within a set
            imAngles = inRing[rng.choice(len(inRing), size=nImages, replace=False)]


            ims = []
            imData = np.zeros((imAngles.shape[0],3))
            for k in range(imAngles.shape[0]):
                im = np.load(Path(f"/exports/nfs_share/Users/al62352/ResolutionHeatmaps/LS9/LS9_705km/GSDarray_matrix_705_km_offnadir_({imAngles[k,0]},{imAngles[k,1]}).npy"))
                ims.append(im)
                imData[k,:] = np.array([imAngles[k,0], imAngles[k,1], altitude])

            cost = imageGeo.getCostFunction(ims, imData, imAngles)

            imsAngs = np.array(imAngles)
            costArray = np.array([cost, 0.0])
            imsNow = np.vstack([imsAngs, costArray])
            matFileName = f'ring_{angMin}_to_{angMax}_image_set_{j}'
            fullMatPath = pathCost / matFileName
            np.save(fullMatPath, imsNow)

            print(f"Ran simulation {j} for the {angMin} to {angMax} ring.")

 
def plotRingResults(pathCost, pathResult):
    RING_EDGES = np.array([0, 10, 20, 30, 40, 48])   # 0-10, 10-20, 20-30, 30-40, 40-48 deg 

    pathCost, pathResult = Path(pathCost), Path(pathResult)
    pathResult.mkdir(parents=True, exist_ok=True)
    n_rings = len(RING_EDGES) - 1
    ring_names = [f"{a}–{b}°" for a, b in zip(RING_EDGES[:-1], RING_EDGES[1:])]
 
    # ---------------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------------
    runs, names = [], []
    for f in sorted(pathCost.rglob("*.npy")):
        m = np.asarray(np.load(f, allow_pickle=True), dtype=float)
        if m.shape == (2, 16):
            m = m.T
        if m.shape != (16, 2):
            print(f"{f.name}: unexpected shape {m.shape}, skipping")
            continue
        runs.append(m)
        names.append(f.name)
    if not runs:
        raise FileNotFoundError(f"No usable .npy files found in {pathCost}")
    data, names = np.stack(runs), np.array(names)
    along, cross = data[:, :15, 0], data[:, :15, 1]
    cost = data[:, 15, 0]
    print(f"Loaded {len(cost)} sets from {pathCost}")
 
    # ---------------------------------------------------------------------------
    # Assign each set to a ring (majority ring of its 15 images)
    # ---------------------------------------------------------------------------
    # True off-nadir angle, same definition the sampler uses
    off_nadir = np.degrees(np.arctan(np.hypot(np.tan(np.radians(along)),
                                              np.tan(np.radians(cross)))))   # (N, 15)
    img_ring = np.clip(np.digitize(off_nadir, RING_EDGES) - 1, 0, n_rings - 1)
    ring = np.array([np.bincount(r, minlength=n_rings).argmax() for r in img_ring])
    mean_off = off_nadir.mean(1)
    present = np.unique(ring)
    groups = [cost[ring == k] for k in present]
 
    # ---------------------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------------------
    print(f"\n{'Ring':<9s}{'n':>5s}{'mean':>11s}{'median':>11s}{'std':>11s}{'min':>11s}   best file")
    for k, g in zip(present, groups):
        i_best = np.where(ring == k)[0][np.argmin(g)]
        print(f"{ring_names[k]:<9s}{g.size:>5d}{g.mean():>11.4g}{np.median(g):>11.4g}"
              f"{g.std(ddof=1):>11.4g}{g.min():>11.4g}   {names[i_best]}")
 
    best_ring = present[np.argmin([np.median(g) for g in groups])]
    print(f"\nBest ring (lowest median cost): {ring_names[best_ring]}")
 
    kw_p = kruskal(*groups).pvalue
    rho_ring, p_ring = spearmanr(ring, cost)
    rho_off, p_off = spearmanr(mean_off, cost)
    ss_tot = ((cost - cost.mean()) ** 2).sum()
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
 
    print(f"\nDo the rings differ?  Kruskal–Wallis p = {kw_p:.2g}"
          f"  ({'yes' if kw_p < 0.05 else 'no clear difference'})")
    print(f"Ring explains {1 - ss_within / ss_tot:.1%} of the variance in cost")
    print(f"Spearman ρ(ring number, cost)        = {rho_ring:+.2f}   p = {p_ring:.2g}")
    print(f"Spearman ρ(mean off-nadir angle, cost) = {rho_off:+.2f}   p = {p_off:.2g}")
    print("  (ρ > 0: cost rises with angle, ρ < 0: falls, ρ ≈ 0: no steady trend."
          " A U-shape shows up as a small ρ but a small Kruskal–Wallis p.)")
 
    print(f"\nBest ring vs each other ring (Mann–Whitney U):")
    g_best = cost[ring == best_ring]
    for k, g in zip(present, groups):
        if k != best_ring:
            print(f"  {ring_names[best_ring]} vs {ring_names[k]}: median "
                  f"{np.median(g_best):.4g} vs {np.median(g):.4g}   "
                  f"p = {mannwhitneyu(g_best, g).pvalue:.2g}")
 
    print("\nBest 15 sets overall:")
    print(f"  {'#':>3s}  {'cost':>10s}  {'ring':<8s}{'mean off-nadir':>16s}   file")
    for n, i in enumerate(np.argsort(cost)[:15], 1):
        print(f"  {n:>3d}  {cost[i]:>10.4g}  {ring_names[ring[i]]:<8s}"
              f"{mean_off[i]:>15.1f}°   {names[i]}")
 
    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    ring_col = {k: cm.plasma(0.1 + 0.75 * k / (n_rings - 1)) for k in range(n_rings)}
    pos = np.arange(len(present))
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), layout="constrained")
 
    # (a) Cost by ring
    ax = axes[0]
    bp = ax.boxplot(groups, positions=pos, widths=0.55, showfliers=False, patch_artist=True)
    for patch, k in zip(bp["boxes"], present):
        patch.set_facecolor(ring_col[k]); patch.set_alpha(0.35)
        if k == best_ring:
            patch.set_edgecolor("red"); patch.set_linewidth(2.5)
    jit = np.random.default_rng(0)
    for j, (k, g) in enumerate(zip(present, groups)):
        ax.scatter(j + jit.uniform(-0.18, 0.18, g.size), g, s=14, color=ring_col[k],
                   edgecolor="k", lw=0.2, alpha=0.8, zorder=3)
    ax.plot(pos, [g.mean() for g in groups], "D-", color="k", ms=6, zorder=4, label="Mean")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{ring_names[k]}\nn={g.size}" for k, g in zip(present, groups)])
    ax.set_xlabel("Off-nadir ring"); ax.set_ylabel("Cost")
    ax.set_title(f"(a) Cost by ring (red = best)   Kruskal–Wallis p = {kw_p:.2g}", fontsize=10)
    ax.legend(); ax.grid(alpha=0.3, axis="y")
 
    # (b) Cost vs each set's mean off-nadir angle
    ax = axes[1]
    for k in present:
        sel = ring == k
        ax.scatter(mean_off[sel], cost[sel], s=18, color=ring_col[k], alpha=0.7,
                   edgecolor="k", lw=0.2, label=ring_names[k])
    for e in RING_EDGES[1:-1]:
        ax.axvline(e, color="k", ls="--", lw=0.8, alpha=0.4)
    ax.set_xlabel("Set mean off-nadir angle (deg)"); ax.set_ylabel("Cost")
    ax.set_title(f"(b) Cost vs mean off-nadir angle   ρ = {rho_off:+.2f}", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
 
    # (c) Share of each ring among the best 10% of sets
    ax = axes[2]
    n_best = max(5, len(cost) // 10)
    best = np.argsort(cost)[:n_best]
    w = 0.38
    ax.bar(pos - w / 2, [np.mean(ring == k) for k in present], w, color="lightgray",
           edgecolor="k", label="Share of all sets")
    ax.bar(pos + w / 2, [np.mean(ring[best] == k) for k in present], w,
           color=[ring_col[k] for k in present], edgecolor="k",
           label=f"Share of best {n_best} sets")
    ax.set_xticks(pos); ax.set_xticklabels([ring_names[k] for k in present])
    ax.set_ylabel("Fraction")
    ax.set_title("(c) Which rings the lowest-cost sets come from", fontsize=10)
    ax.legend(); ax.grid(alpha=0.3, axis="y")
 
    fig.suptitle(f"Cost by off-nadir ring   (best ring: {ring_names[best_ring]})", fontsize=13)
    fig.savefig(pathResult / "ring_cost_summary.png", dpi=150, bbox_inches="tight")
 
 
if __name__ == "__main__":
    pathCost = Path("/exports/nfs_share/Users/al62352/ResolutionHeatmaps/LS9/LS9_Random15_RingsData")   
    runTest(pathCost)
    
    pathCost = Path("/exports/nfs_share/Users/al62352/ResolutionHeatmaps/LS9/LS9_Random15_RingsData")
    pathResult = pathCost.parent / "LS9_Random15_RingsResults"
    plotRingResults(pathCost, pathResult)
