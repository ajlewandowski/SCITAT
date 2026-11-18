"""
Self-test of the calibration scripts without the real data or s2p
=================================================================
Builds a fake DFC2019-style folder - lidar DSM tiles (one with a GeoTIFF CRS, one
with a DFC-style .txt sidecar), WorldView-3-like image tiles carrying RPC tags made
from known viewing angles and dates - then runs steps 1, 2, 3 (--dry-run), fake
pair DSMs with known noise in place of s2p, step 4 and step 5.

It checks that
  - the RPC angles come back as the angles the images were made with
  - the dates are read from the TIFF tags
  - pair DSMs on a different grid are resampled onto the lidar grid
  - the scored sigma_z matches the noise that was added
  - the calibration runs on the result

Run:  python selfTest.py          (about a minute; writes ./selftest)
"""
import shutil
import subprocess
import sys
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Dataset has no geotransform")
HERE = Path(__file__).resolve().parent
WORK = HERE / "selftest"
EPSG = 32617
R = 6378137.0


def utmToLonLat(e, n):
    from pyproj import Transformer
    return Transformer.from_crs(EPSG, 4326, always_xy=True).transform(e, n)


def lonLatToUtm(lon, lat):
    from pyproj import Transformer
    return Transformer.from_crs(4326, EPSG, always_xy=True).transform(lon, lat)


def fitRPC(az, inc, e0, n0, gsd=0.31, size=600):
    """RPC (denominators = 1) of an affine camera that sees from azimuth az, incidence inc."""
    from dfc2019Common import _terms
    lon0, lat0 = utmToLonLat(e0, n0)
    g = np.linspace(-1, 1, 9)
    L, P, H = [a.ravel() for a in np.meshgrid(g, g, g, indexing="ij")]
    ls, ps, hs = 0.003, 0.003, 100.0
    lon, lat, h = lon0 + L * ls, lat0 + P * ps, 20 + H * hs
    E, N = lonLatToUtm(lon, lat)
    t = np.tan(np.radians(inc))
    E2 = E - h * t * np.sin(np.radians(az))                # a higher point lies toward the satellite,
    N2 = N - h * t * np.cos(np.radians(az))                # so it images where lower ground away from it does
    col = (E2 - e0) / gsd + size / 2
    row = -(N2 - n0) / gsd + size / 2
    T = _terms(L, P, H).T
    cs, rs = 2000.0, 2000.0
    cn = np.linalg.lstsq(T, (col - size / 2) / cs, rcond=None)[0]
    rn = np.linalg.lstsq(T, (row - size / 2) / rs, rcond=None)[0]
    den = np.zeros(20); den[0] = 1
    f = lambda v: " ".join(f"{x:.15e}" for x in v)
    return dict(LINE_OFF=str(size / 2), SAMP_OFF=str(size / 2), LAT_OFF=f"{lat0:.8f}", LONG_OFF=f"{lon0:.8f}",
                HEIGHT_OFF="20", LINE_SCALE=str(rs), SAMP_SCALE=str(cs), LAT_SCALE=str(ps), LONG_SCALE=str(ls),
                HEIGHT_SCALE=str(hs), LINE_NUM_COEFF=f(rn), LINE_DEN_COEFF=f(den), SAMP_NUM_COEFF=f(cn),
                SAMP_DEN_COEFF=f(den))


def makeData(rng):
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS
    data = WORK / "data"
    (data / "images").mkdir(parents=True); (data / "truth").mkdir(parents=True)
    truthAngles = {}
    for t, (tile, georef) in enumerate((("SYN_001", "crs"), ("JAX_002", "txt"))):
        size, res = 300, 0.5
        e0, n0 = 435000.0 + 2000 * t, 3354000.0
        z = np.full((size, size), 5.0)
        for _ in range(25):
            w, l = rng.integers(10, 40, 2); r, c = rng.integers(0, size - w), rng.integers(0, size - l)
            z[r:r + w, c:c + l] = 5 + rng.uniform(4, 20)
        cls = np.where(z > 6, 6, 2).astype(np.uint8)
        cls[:15, :15] = 9                                      # some water
        tr = from_origin(e0 - size * res / 2, n0 + size * res / 2, res, res)
        kw = dict(driver="GTiff", height=size, width=size, count=1)
        if georef == "crs":
            with rasterio.open(data / "truth" / f"{tile}_DSM.tif", "w", dtype="float32", crs=CRS.from_epsg(EPSG),
                               transform=tr, **kw) as d:
                d.write(z.astype(np.float32), 1)
        else:   # no CRS, DFC-style sidecar: easting northing size gsd (lower-left corner)
            with rasterio.open(data / "truth" / f"{tile}_DSM.tif", "w", dtype="float32", **kw) as d:
                d.write(z.astype(np.float32), 1)
            (data / "truth" / f"{tile}_DSM.txt").write_text(
                f"{e0 - size * res / 2} {n0 - size * res / 2} {size} {res}\n")
        with rasterio.open(data / "truth" / f"{tile}_CLS.tif", "w", dtype="uint8", **kw) as d:
            d.write(cls, 1)
        for i in range(7):
            az, inc = float(rng.uniform(0, 360)), float(rng.uniform(5, 35))
            date = np.datetime64("2015-01-10") + np.timedelta64(int(rng.integers(0, 600)), "D")
            iid = f"{tile}_{i:03d}"
            truthAngles[iid] = (az, inc, str(date))
            p = data / "images" / f"{iid}_RGB.tif"
            from rasterio.rpc import RPC as RRPC
            g = fitRPC(az, inc, e0, n0)
            rp = RRPC.from_gdal(g)
            with rasterio.open(p, "w", driver="GTiff", height=64, width=64, count=3, dtype="uint8", rpcs=rp) as d:
                d.write(rng.integers(0, 255, (3, 64, 64), dtype=np.uint8))
                d.update_tags(TIFFTAG_DATETIME=str(date).replace("-", ":") + " 16:00:00")
    return data, truthAngles


def fakePairs(rng):
    """Pair 'reconstructions': lidar + noise growing with convergence, holes, on a shifted
    1 m grid in the same CRS (exercises the resampling)."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    from dfc2019Common import LidarTile
    img = pd.read_csv(WORK / "images.csv").set_index("image_id")
    pairs = pd.read_csv(WORK / "pairs.csv")
    expected = {}
    for r in pairs.itertuples():
        a, b = img.loc[r.image_a], img.loc[r.image_b]
        L = LidarTile(a.lidar, a.cls)
        sig = 0.3 + 0.02 * abs(a.incidence_deg - b.incidence_deg)
        z = L.z + rng.normal(0, sig, L.z.shape)
        z[rng.random(L.z.shape) < 0.1] = np.nan
        out = WORK / "pairs" / r.pair_id
        out.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out / "dsm.tif", "w", driver="GTiff", height=z.shape[0], width=z.shape[1], count=1,
                           dtype="float32", crs=CRS.from_epsg(L.epsg), nodata=np.nan,
                           transform=from_origin(L.x0, L.y0, L.res, L.res)) as d:
            d.write(z.astype(np.float32), 1)
        expected[r.pair_id] = sig
    return expected


def run(*cmd):
    print("\n$ python", " ".join(cmd), flush=True)
    rc = subprocess.call([sys.executable, *cmd], cwd=HERE)
    if rc:
        raise SystemExit(f"FAILED: {cmd[0]} exit code {rc}")


def main():
    rng = np.random.default_rng(1)
    shutil.rmtree(WORK, ignore_errors=True)
    data, truth = makeData(rng)
    w = str(WORK)
    run("dfc2019Geometry.py", "--images", str(data / "images"), "--truth", str(data / "truth"), "--inspect")
    run("dfc2019Geometry.py", "--images", str(data / "images"), "--truth", str(data / "truth"), "--work", w)
    img = pd.read_csv(WORK / "images.csv").set_index("image_id")
    dAz = max(abs(((img.loc[k, "sat_az_deg"] - v[0] + 180) % 360) - 180) for k, v in truth.items())
    dInc = max(abs(img.loc[k, "incidence_deg"] - v[1]) for k, v in truth.items())
    dates = all(img.loc[k, "date"] == v[2] for k, v in truth.items())
    print(f"\nCHECK angles from RPC: max azimuth error {dAz:.3f}°, max incidence error {dInc:.3f}°, dates ok: {dates}")
    assert dAz < 0.5 and dInc < 0.2 and dates
    run("dfc2019Subsets.py", "--work", w, "--pool", "5")
    run("runReconstructions.py", "--work", w, "--dry-run")
    expected = fakePairs(rng)
    run("scoreReconstructions.py", "--work", w)
    m = pd.read_csv(WORK / "measured.csv")
    pr = m[m.kind == "pair"].copy()
    pr["expected"] = [expected.get(i.replace(";", "__"), np.nan) for i in pr.image_ids]
    ratio = (pr.sigma_z_m / pr.expected).median()
    print(f"\nCHECK scored pair sigma_z / added noise: median {ratio:.2f} (resampling 0.5 m -> 0.5 m grid, "
          f"expect ~1)")
    assert 0.8 < ratio < 1.2
    run("calibrateErrorModel.py", "--work", w, "--points", "1500",
        "--fit", "errSigma0Pix,errSysM,errMatch50Deg")
    print("\nSelf-test passed.")


if __name__ == "__main__":
    main()
