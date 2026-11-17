# Next-step geometry studies (tests 1–10)

Ten scripts that build on `randomImageGroupStudy.py` and `ringedImageStudy.py`.
Together they go from "is the cost function measuring the right things?" to
"which orbit can actually deliver a low-error multi-view stereo (MVS) set?"

| # | Script | Question | Full run* |
|---|--------|----------|-----------|
| 1 | `test1_costComponents.py` | Which cost terms decide the ranking, and does the winner change with the weights? | ~1 min |
| 2 | `test2_optimizerSearch.py` | What is the best 15-image set, and do independent searches agree? | ~8 min |
| 3 | `test3_greedySelection.py` | If images are added one at a time, where should each go? | ~8 min |
| 4 | `test4_geometryFamilies.py` | What do simple, taskable patterns (rings, single pass, two passes) score? | ~1 min |
| 5 | `test5_imageCountSweep.py` | How many images are enough? | ~18 min |
| 6 | `test6_featureSurrogate.py` | Which stereo-geometry features (convergence, azimuth spread, B/H, ...) drive the cost? | ~2 min |
| 7 | `test7_orbitConstrained.py` | With real passes, sun and slew limits, what can an orbit deliver? | ~18 min |
| 8 | `test8_sensorInvariance.py` | Do the conclusions hold across sensors and noise settings? | ~2 min |
| 9 | `test9_mvsValidation.py` | Does a low cost mean a low 3-D error? Does the predicted error in metres match the simulator? (+ sets to reconstruct for real) | ~1 min |
| 10 | `test10_predictedError.py` | Scored by predicted DSM error in metres, which geometry is best for open, suburban and urban scenes, and per sensor? | ~1 min |

\*Measured on a laptop-class CPU. Every script accepts `--quick` (a tiny run in
well under a minute) — run that first to check your setup.

## Requirements
Everything in `requirements.txt` plus `scikit-learn` (test 6). `shap` is optional
(test 6 adds a SHAP plot if it is installed). The scripts do **not** need `tatc` or
SPICE: test 7 propagates orbits with `sgp4` directly.

These tests also need the change to `imageGeometryClass.py` that adds
`getCostComponents()` (the five unweighted terms) and the `costWeights` attribute.
`getCostFunction()` returns the same weighted total it did before.

Target placement: `imageGeometry.targetMode` is `"center"` by default (the boresight is
on the target and the images are compared on a common ground grid around it). The
old behavior, where the target moves to the lowest-GSD part of each frame, is
`targetMode = "bestGSD"`. In the tests, use `CostEvaluator(..., targetMode="bestGSD")`
to compare. `aoiScale` sets the size of the ground area being reconstructed (1.0 = one
nadir scene); larger values make the overlap term matter.

Geometry requirements (shared options for every test):

| Option | Default | Meaning |
|---|---|---|
| `--max-off` | sensor's limit | agility limit: largest TRUE off-nadir angle (deg). Tests search the cone inside it, not a ±box |
| `--min-overlap` | 0.75 | share of the target area every image must cover |
| `--pointing-err` | 0 | pointing error (deg) allowed for in the overlap requirement |
| `--aoi-scale` | 1.0 | target-area half-width as a multiple of the nadir half-swath |
| `--gsd-limit-m` | none | usable-pixel GSD requirement in metres (e.g. 0.6) |
| `--gsd-limit-factor` | 1.25 | used when `--gsd-limit-m` is not given: usable if GSD ≤ factor × nadir GSD |
| `--target-bh` | 1.0 | target mean B/H for the B/H term |
| `--flat-earth` | off | flat-Earth geometry (spherical is the default) |

Overlap is measured on the target area, so off-nadir viewing alone does not reduce
it: with no pointing error it is 100 % at any angle. It falls with `--pointing-err`
or `--aoi-scale` > 1. Sets that fail the overlap requirement or the agility limit are
infeasible (the tests drop them; optimizers see a graded penalty).

Sensors: `--sensor SkySat | Landsat9 | WorldView3 | PleiadesNeo`. Pléiades Neo uses
Airbus's published 620 km orbit, 0.3 m nadir GSD and 14 km swath, with a 48° limit
(the satellite can point to ±52°).

Spherical Earth: lines of sight, convergence angles, B/H, footprints, the ground grid
and test 9's simulator use the incidence angle at the target, which is larger than
the off-nadir angle (54.6° at 48° from 620 km).

Example — Pléiades Neo out to 48°, 60 cm usable GSD, 0.02° pointing error:
```
python runAll.py --quick --sensor PleiadesNeo --gsd-limit-m 0.6 --pointing-err 0.02
```

Usability grade: each image's usable-pixel share is the `errUsable` term (weight 0.2);
unusable pixels are dropped before the GSD and MVS error-map terms. Test 1 figure 7 maps the usable share and the incidence angle over every pointing.

## Predicted error in metres (`imageGeometry.predictedErrorM`)
Alongside the weighted cost, `predictedErrorM(ims, imData, imAngles)` returns a predicted
DSM height error in metres:

    E_rms = sqrt( C (sigma_z^2 + sigma_sys^2) + (1 - C) e_miss^2 )

- **sigma_z** – closed-form triangulation covariance of all views together. Each view's
  ray noise is matching precision (pixels) × its own ground pixel size × slant range.
  Matching precision grows with convergence angle and with the GSD ratio of the pair;
  match probability falls off past ~45° convergence (logistic). Each view is weighted
  by the chance it matches at least one other view, so pairs are **not** double counted
  the way summing pair-by-pair information would. `errFusionEff` (0.7) is the
  loss from real fusion compared with ideal least squares.
- **C (completeness)** – share of the scene where some matchable pair can see past the
  buildings, ray-traced on a canonical DSM (`errScene`: `open`, `suburban`, `urban`, or
  your own via `imageGeometry.registerScene(name, dsm, cellM)`).
- **sigma_sys** – systematic floor (orbit model / bundle adjustment), `errSysGSD` × nadir GSD
  (0.5) or an absolute `errSysM`.
- **e_miss** – error charged where nothing is reconstructed (`errMissM`, 10 m).

| Parameter | Default | Meaning |
|---|---|---|
| `errSigma0Pix` | 0.3 | matching precision (px) of a small-convergence, equal-GSD pair |
| `errConvGrowth`, `errConvRefDeg` | 0.5, 30 | precision worsens by (1 + growth (conv/ref)^2) |
| `errScaleExp` | 1.0 | precision worsens by (GSD ratio)^exp |
| `errMatch50Deg`, `errMatchWidthDeg` | 45, 3 | convergence where matching succeeds half the time; width of the fall-off |
| `errFusionEff` | 0.7 | fusion efficiency |
| `errSysM` / `errSysGSD` | none / 0.5 | systematic floor |
| `errMissM` | 10 | error charged for unreconstructed area |
| `errScene` | suburban | occlusion scene |

**These are starting values, not calibrated.** Calibrate them by reconstructing the
`mvs_validation_sets.csv` sets from test 9 (or your Austin data) and fitting the
parameters to the measured error; the model's form stays the same.

Test options: `--objective cost|error|sigma` (what the optimizers minimise: the weighted
cost, predicted E_rms, or predicted sigma_z only) and `--scene open|suburban|urban`.
Tests 1 and 8 always analyse the weighted cost. `CostEvaluator.predict(angles)` returns
`predErrM`, `predSigmaZM` and `predCompleteness` for any set.

## Running
```
cd optimizationStudy/nextTests
python runAll.py --quick          # every test, tiny sizes
python runAll.py                  # every test, full sizes (~1 h)
python test7_orbitConstrained.py --help     # each script documents its options
```
Common options (all scripts): `--out DIR` (default `./results/<test>`),
`--sensor {SkySat, Landsat9, WorldView3, PleiadesNeo}`, `--seed N`, `--quick`,
`--objective {cost, error, sigma}`, `--scene {open, suburban, urban}`.

Useful extras:
- `test1 --from-dir /exports/.../SS_R15_Data` and `test6 --from-dir ...` load your existing `random_15_images_*.npy` sets.
- `test4 --random-best 0.327` draws the best random-set cost as a reference line.
- `test7 --tle ../../processingTLEs/tle49070.txt --alts 0` uses a real satellite's TLE.
- `test7 --target LAT,LON --slew-rate 2 --settle 3 --min-sun 20` changes the scenario.
- `test9 --mvs-results my_results.csv` plots real reconstruction error against the cost.

## Shared code
`studyCommon.py` holds everything the tests share:
- the sensor presets
- `CostEvaluator`, which caches GSD images and decimates them to ~320 px so one evaluation takes ~0.03 s instead of ~1.8 s; test 1 checks that rankings are unchanged (ρ = 1.00)
- the angle conversions and stereo pair descriptors (convergence, bisector elevation, asymmetry)
- the order-independent set features
- the simulated-annealing optimizer
- the plotting style

## Reading the results
Each script prints a short summary and writes numbered PNGs plus CSV/NPZ data
to its results folder. The docstring at the top of every script explains each
output file.
