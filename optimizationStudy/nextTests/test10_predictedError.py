"""
Test 10 - Optimal geometry in metres (predicted DSM error)
==========================================================
Question: when geometries are scored by predicted DSM error in METRES
(imageGeometry.predictedErrorM) instead of the weighted cost, what geometry is
best - and how does that depend on the scene (open terrain vs suburbs vs a dense
city) and on the sensor?

The predicted error combines
  height precision  closed-form triangulation covariance of the views that can
                    match each other (spherical Earth, slant range, per-pair
                    matching precision and match probability)
  completeness      share of the scene some matchable pair can see past the
                    buildings (ray-traced on a canonical DSM per scene)
  systematic floor  orbit-model / bundle-adjustment error that no geometry removes
    E_rms = sqrt( C (sigma_z^2 + sigma_sys^2) + (1 - C) e_miss^2 )
Parameters are typical starting values, not yet calibrated (see test 9).

What it does
  1. Ring family (k views at off-nadir theta, evenly spaced in azimuth) for every
     scene type: E_rms, sigma_z and completeness on a (k, theta) grid.
  2. The best theta for each k, per scene - the scene-dependent answer.
  3. Precision vs completeness for an 8-view ring: why the optimum moves.
  4. Best achievable error per sensor (each with its own agility limit), in metres -
     comparable across sensors, which the weighted cost was not.

Outputs (in --out, default ./results/test10_predictedError)
  ring_predicted_error.csv          every ring, scene and sensor: E, sigma_z, completeness
  1_error_heatmaps_by_scene.png     E_rms (m) over (k, theta) for each scene
  2_best_angle_by_scene.png         best theta per k, per scene, and the error there
  3_precision_vs_completeness.png   sigma_z and completeness vs theta (8 views)
  4_best_error_by_sensor.png        best ring error per sensor and scene, in metres

Run
  python test10_predictedError.py --quick
  python test10_predictedError.py --sensor PleiadesNeo --gsd-limit-m 0.9
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from studyCommon import (SENSORS, SERIES, SEQ, INK2, save, style, commonArgs, Timer, polarToAngles,
                         makeEvaluator, resolveArgs)

SCENES = ["open", "suburban", "urban"]


def ringErrors(ev, ks, thetas):
    """Predicted error for every ring (k, theta); NaN where the set is infeasible."""
    E, S, C = (np.full((len(ks), len(thetas)), np.nan) for _ in range(3))
    for i, k in enumerate(ks):
        for j, t in enumerate(thetas):
            r = ev.predict(polarToAngles(np.full(k, t), 360.0 * np.arange(k) / k))
            if r["ok"]:
                E[i, j], S[i, j], C[i, j] = r["predErrM"], r["predSigmaZM"], r["predCompleteness"]
    return E, S, C


def main():
    p = commonArgs(__doc__.split("\n")[1], "test10_predictedError")
    p.add_argument("--sensors", default="SkySat,WorldView3,PleiadesNeo",
                   help="sensors for the comparison in figure 4")
    args = resolveArgs(p.parse_args())
    style()
    out = args.out
    if args.quick:
        ks, step = [2, 4, 8], 5.0
    else:
        ks, step = [2, 3, 4, 6, 8, 12], 2.5
    thetas = np.arange(step, args.max_off + 1e-9, step)

    # ------------------------------------------------------------------ 1. scenes, main sensor
    rows, grids = [], {}
    for sc in SCENES:
        with Timer(f"{args.sensor}, {sc} scene: {len(ks)} x {len(thetas)} rings"):
            ev = makeEvaluator(args, scene=sc)
            E, S, C = ringErrors(ev, ks, thetas)
            grids[sc] = (E, S, C)
            for i, k in enumerate(ks):
                for j, t in enumerate(thetas):
                    rows.append(dict(sensor=args.sensor, scene=sc, k=k, theta=t,
                                     err_m=E[i, j], sigma_m=S[i, j], completeness=C[i, j]))
            ib = np.unravel_index(np.nanargmin(E), E.shape)
            print(f"   best: {ks[ib[0]]} views at {thetas[ib[1]]:g}°  ->  {np.nanmin(E):.3f} m "
                  f"(height precision {S[ib]:.3f} m, completeness {C[ib]:.0%})")

    # ------------------------------------------------------------------ figure 1
    allE = np.concatenate([g[0].ravel() for g in grids.values()])
    norm = LogNorm(vmin=np.nanmin(allE), vmax=np.nanmax(allE))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), layout="constrained")
    for ax, sc in zip(axes, SCENES):
        E = grids[sc][0]
        im = ax.imshow(E, origin="lower", aspect="auto", cmap=SEQ, norm=norm,
                       extent=(thetas[0] - step / 2, thetas[-1] + step / 2, -0.5, len(ks) - 0.5))
        ax.set_yticks(range(len(ks)), [str(k) for k in ks])
        ib = np.unravel_index(np.nanargmin(E), E.shape)
        ax.scatter(thetas[ib[1]], ib[0], marker="*", s=220, color=SERIES[3], edgecolor="k", zorder=3)
        ax.plot(thetas[np.nanargmin(E, axis=1)], range(len(ks)), "o", color="white", mec="k", ms=5)
        ax.set_xlabel("ring off-nadir angle θ (°)"); ax.set_ylabel("views on the ring k")
        ax.set_title(f"{sc}: best {np.nanmin(E):.2f} m"); ax.grid(False)
    fig.colorbar(im, ax=axes, fraction=0.025, label="predicted DSM error E_rms (m, darker = worse)")
    fig.suptitle(f"Test 10 – Predicted DSM error of ring geometries, {args.sensor} "
                 f"(○ = best θ per k, ★ = best overall)", fontsize=12)
    save(fig, out, "1_error_heatmaps_by_scene")

    # ------------------------------------------------------------------ figure 2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for col, sc in zip(SERIES, SCENES):
        E = grids[sc][0]
        jb = np.nanargmin(E, axis=1)
        axes[0].plot(ks, thetas[jb], "-o", color=col, label=sc)
        axes[1].plot(ks, E[np.arange(len(ks)), jb], "-o", color=col, label=sc)
    axes[0].set_ylabel("best ring off-nadir angle θ (°)")
    axes[1].set_ylabel("predicted error at that θ (m)"); axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("views on the ring k"); ax.set_xticks(ks); ax.legend()
    axes[0].set_title("(a) The best angle depends on the scene")
    axes[1].set_title("(b) Error at the best angle")
    fig.suptitle(f"Test 10 – Scene-dependent optimum, {args.sensor}", fontsize=12)
    save(fig, out, "2_best_angle_by_scene")

    # ------------------------------------------------------------------ figure 3
    k8 = ks.index(8) if 8 in ks else len(ks) - 1
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), layout="constrained")
    for col, sc in zip(SERIES, SCENES):
        E, S, C = grids[sc]
        axes[0].plot(thetas, S[k8], color=col, label=sc)
        axes[1].plot(thetas, C[k8], color=col, label=sc)
        axes[2].plot(thetas, E[k8], color=col, label=sc)
    axes[0].set_ylabel("height precision σ_z (m)"); axes[0].set_title("(a) Precision improves with angle")
    axes[1].set_ylabel("completeness"); axes[1].set_title("(b) Buildings hide more at steep angles")
    axes[2].set_ylabel("E_rms (m)"); axes[2].set_title("(c) Combined: the trade-off sets the optimum")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("ring off-nadir angle θ (°)"); ax.legend(fontsize=8)
    fig.suptitle(f"Test 10 – Why the optimum moves: {ks[k8]}-view ring, {args.sensor}", fontsize=12)
    save(fig, out, "3_precision_vs_completeness")

    # ------------------------------------------------------------------ 4. sensors in metres
    sensors = [s.strip() for s in args.sensors.split(",") if s.strip()]
    best = np.full((len(sensors), len(SCENES)), np.nan)
    for a, sn in enumerate(sensors):
        lim = SENSORS[sn]["maxOffNadir"]
        th = np.arange(step, lim + 1e-9, step)
        for b, sc in enumerate(SCENES):
            with Timer(f"{sn}, {sc} scene (limit {lim:g}°)"):
                # each sensor with its own agility limit; same GSD rule and model parameters
                ev = makeEvaluator(args, sensor=sn, scene=sc, maxOffNadir=lim)
                E, S, C = ringErrors(ev, ks, th)
                best[a, b] = np.nanmin(E)
                for i, k in enumerate(ks):
                    for j, t in enumerate(th):
                        rows.append(dict(sensor=sn, scene=sc, k=k, theta=t, err_m=E[i, j],
                                         sigma_m=S[i, j], completeness=C[i, j]))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates().to_csv(out / "ring_predicted_error.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.2), layout="constrained")
    width = 0.8 / len(SCENES)
    for b, (col, sc) in enumerate(zip(SERIES, SCENES)):
        x = np.arange(len(sensors)) + (b - len(SCENES) / 2 + 0.5) * width
        ax.bar(x, best[:, b], width * 0.92, color=col, label=sc)
        for xi, v in zip(x, best[:, b]):
            ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8)
    ax.set_xticks(range(len(sensors)), [f"{s}\n(limit {SENSORS[s]['maxOffNadir']:g}°)" for s in sensors])
    ax.set_ylabel("best predicted DSM error (m)")
    ax.set_title("Test 10 – Best achievable error per sensor and scene, in metres")
    ax.legend()
    save(fig, out, "4_best_error_by_sensor")
    print("\nBest predicted error (m):")
    print(pd.DataFrame(best, index=sensors, columns=SCENES).round(3).to_string())


if __name__ == "__main__":
    main()
