# Next-step geometry studies (tests 1–9)

Nine scripts that build on `randomImageGroupStudy.py` and `ringedImageStudy.py`.
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
| 9 | `test9_mvsValidation.py` | Does a low cost mean a low 3-D error? (+ sets to reconstruct for real) | <1 min |

\*Measured on a laptop-class CPU. Every script accepts `--quick` (a tiny run in
well under a minute) — run that first to check your setup.

## Requirements
Everything in `requirements.txt` plus `scikit-learn` (test 6). `shap` is optional
(test 6 adds a SHAP plot if it is installed). The scripts do **not** need `tatc` or
SPICE: test 7 propagates orbits with `sgp4` directly.

These tests also need the change to `imageGeometryClass.py` that adds
`getCostComponents()` (the five unweighted terms) and the `costWeights` attribute.
`getCostFunction()` returns the same weighted total it did before.

## Running
```
cd optimizationStudy/nextTests
python runAll.py --quick          # every test, tiny sizes
python runAll.py                  # every test, full sizes (~1 h)
python test7_orbitConstrained.py --help     # each script documents its options
```
Common options (all scripts): `--out DIR` (default `./results/<test>`),
`--sensor {SkySat, Landsat9, WorldView3}`, `--seed N`, `--quick`.

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
