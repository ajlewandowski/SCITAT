# Calibrating the predicted-error model on DFC2019

SCITAT's `imageGeometry.predictedErrorM` predicts DSM error in metres from imaging
geometry. Its triangulation and occlusion geometry is exact, but its matching and
systematic parameters are assumed starting values. These scripts fit those
parameters to real reconstructions.

The real data is the 2019 IEEE GRSS Data Fusion Contest (DFC2019) Track 3:
- WorldView-3 images with RPC camera models, over Jacksonville (JAX) and Omaha (OMA)
- airborne-lidar DSMs as ground truth

## 1. Get the data

You need a free IEEE account. IEEE membership is not required, but DataPort may ask you
to sign up for a subscription; IEEE Society members get it free.

1. Go to IEEE DataPort, **"Data Fusion Contest 2019 (DFC2019)"**:
   https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019
   Log in, and accept the contest terms.
2. Download only these Track 3 files:

   | File | Size | Why |
   |---|---|---|
   | `Train-Track3-RGB-1` and `Train-Track3-RGB-2` | 7.6 GB + 12.5 GB | the image tiles, RPCs in the TIFFs |
   | `Train-Track3-Truth` | 37 MB | lidar DSMs (`*_DSM.tif`) and classes (`*_CLS.tif`) |
   | `Track3-Metadata` | 0.1 MB | image metadata (dates, if the TIFF tags lack them) |

   Skip the multispectral (MSI) parts (about 80 GB) and the validation data, whose lidar
   truth is withheld.
3. Unzip them into one data folder with `images/`, `truth/` and `metadata/` inside.
   The scripts look in
   `/exports/nfs_share/Users/al62352/ResolutionHeatmaps/{images,truth,metadata}` by
   default (`DATA_ROOT` at the top of `dfc2019Common.py`). To use another folder, change
   `DATA_ROOT`, set the `SCITAT_DFC_ROOT` environment variable, or pass
   `--images/--truth/--metadata`.

   Working files go to `ResolutionHeatmaps/calibration_work/`, next to the data and
   outside the repo: single-band image copies, pair DSMs, CSVs and the calibration
   result. On a computer without that folder they go to `calibration/work/`, which git
   ignores. `--work` overrides both.

**Terms.** Redistribution is not allowed, so keep the data out of the GitHub repo.
Publications using it need approval from the contest organisers, must cite the data
as `grss_dfc_2019`, and must include their acknowledgement text. Both are on the
DataPort page.

## 2. Install

```
pip install rasterio pyproj scipy pandas matplotlib     # the scripts
pip install s2p                                         # the stereo pipeline
```

s2p also needs GDAL, fftw and libtiff. On Ubuntu:
```
sudo apt install libgdal-dev gdal-bin libfftw3-dev libgeotiff-dev libtiff5-dev
```
Without root on the server, a conda environment works:
```
conda install -c conda-forge gdal fftw libtiff
```
Ames Stereo Pipeline or your own pipeline also works through `--engine-cmd`
(see `runReconstructions.py`).

Check the scripts before the data arrives:
```
cd optimizationStudy/calibration
python selfTest.py                         # fake tiles with RPCs: steps 1-5, ~10 s
python calibrateErrorModel.py --synthetic  # recovers known parameters, ~3 min
```

## 3. Run

```
cd optimizationStudy/calibration

# Step 1 - look at the files first
python dfc2019Geometry.py --inspect

# Step 1 - image geometry (angles from the RPCs, dates)
python dfc2019Geometry.py

# Step 2 - choose 12 tiles of varied building density, a pool of 10 images per tile
python dfc2019Subsets.py --tiles 12 --pool 10

# Step 3 - stereo pairs (45 per tile; resumable)
python runReconstructions.py --only JAX_004 --jobs 2 --threads 4   # one tile first
python runReconstructions.py --jobs 4 --threads 4                 # then all (overnight)

# Step 4 - fuse each subset's pairs and score it against the lidar
python scoreReconstructions.py

# Step 5 - fit, with the leave-one-tile-out check
python calibrateErrorModel.py --cv
```

**Where the files come from.** The scripts expect DFC-style names: images
`JAX_004_006_RGB.tif` and truth `JAX_004_DSM.tif` / `JAX_004_CLS.tif`. If the files are
named differently, change `--image-regex` / `--dsm-glob`.

**If `--inspect` finds no RPC or no dates.**
- No RPC: the RPC is read from the GeoTIFF RPC tags, then from sidecar files
  (`_RPC.TXT`, `.RPB`).
- No dates: make a two-column CSV `image_id,date` and pass `--dates`. The
  calibration also runs without dates; it then leaves out the date-gap term.

## 4. Use the result

`ResolutionHeatmaps/calibration_work/calibration/calibration.json` holds the fitted parameters. To use them:
- **In every test:** add `--calibration path/to/calibration.json`. It applies to the
  predicted error, `--objective error|sigma`, and test 10.
- **In your own code:** `geo.loadErrorCalibration("calibration.json")`.

## What the steps do

| Step | Script | Output |
|---|---|---|
| 1 | `dfc2019Geometry.py` | `images.csv`: per image, satellite azimuth and incidence (from the RPC at the tile centre), off-nadir angle, SCITAT (along, cross), GSD, date |
| 2 | `dfc2019Subsets.py` | `pairs.csv` (jobs) and `subsets.csv`: pairs, triples, random / spread / narrow sets of 3–8 views from a geometry-spanning pool |
| 3 | `runReconstructions.py` | `pairs/<pair>/dsm.tif`: one s2p reconstruction per pair, on the lidar tile's UTM grid |
| 4 | `scoreReconstructions.py` | `measured.csv`: per subset, the median fusion of its pair DSMs vs lidar |
| 5 | `calibrateErrorModel.py` | `calibration/calibration.json`, `predictions.csv`, figures |

**Step 4 scoring.** The measured quantities follow `predictedErrorM`'s own definitions:
- completeness: the share of cells within 5 m of the lidar
- σz: the RMS error of those cells
- E_rms: √(C σz² + (1−C)·10²)

Water and a 20-cell edge are left out. The median vertical bias is removed, because
the Track 3 RPCs were already adjusted to the lidar. Use `--no-bias` to keep it, so
that `errSysM` absorbs it.

**Step 5 fitting.**
- Fitted: matching precision, its growth with convergence and GSD mismatch, the
  match cut-off, the systematic floor, and (with dates) the date-gap term.
- Held fixed: `errFusionEff` at 1. It scales every set the same way as the matching
  precision, so the two cannot be separated, and the matching precision absorbs it.
- Scene: each tile's own lidar is the occlusion scene, so the buildings are real.

**Reading the result.**
- `2_residuals.png`: a trend against convergence, view count or date gap means the
  model's form is missing an effect, not just a parameter value. For example, residuals
  that grow with the number of views mean median fusion gains less from extra views
  than ideal least squares predicts.
- `3_leave_tile_out.png`: this is the honest test. Trust the calibrated metres only to
  the extent held-out tiles agree.

## Scope

The calibration is for **this pipeline on WorldView-3-class imagery** (0.3 m, 2014–2016
images). It transfers as follows:
- **Geometry, slant range, GSD and occlusion:** computed for any sensor, so they need
  no recalibration.
- **Matching parameters:** in pixels and degrees. They should carry over to Pléiades
  Neo; check SkySat separately.
- **Systematic floor:** set by each satellite's pointing knowledge and by whether you
  use ground control. It is the most sensor-specific parameter.
- **New MVS pipeline:** recalibrate.
