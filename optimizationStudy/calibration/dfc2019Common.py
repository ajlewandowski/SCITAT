"""
Shared code for calibrating SCITAT's predicted-error model on the 2019 IEEE GRSS
Data Fusion Contest (DFC2019) Track 3 data: WorldView-3 images with RPC camera
models over Jacksonville (JAX) and Omaha (OMA), and airborne-lidar DSMs.

Contents
  RPC                RPC00B camera model: ground -> image, image -> ground, look direction
  readRPC            RPC from GeoTIFF tags (rasterio or GDAL) or an _RPC.TXT / .RPB sidecar
  findTiles          discover image tiles and lidar truth by file name
  LidarTile          a lidar DSM with its ground grid (UTM), class mask and margins
  toGrid             resample any GeoTIFF DSM onto a lidar tile's grid
  scitatAngles       satellite azimuth / elevation -> SCITAT (along, cross) pointing angles
  makeGeometry       an imageGeometry configured for WorldView-3 (for predictedErrorM)
  designSubsets      the image pool and subsets (pairs, triples, spread / narrow / random sets)

Conventions
  Lidar rasters are north-up: rows run south, columns east. SCITAT scenes use rows =
  along-track, columns = cross-track, so the calibration puts along = south and
  cross = east. The predicted error does not depend on that choice (only relative
  geometry and the buildings matter), as long as images and scene use the same one.
"""
import os
import re
import sys
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # repo root (imageGeometryClass.py)
NEXT = HERE.parent / "nextTests"             # studyCommon.py (sensor presets)
for p in (ROOT, NEXT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from imageGeometryClass import imageGeometry          # noqa: E402
from studyCommon import SENSORS, sensArray            # noqa: E402

# ---- Where the DFC2019 data lives -------------------------------------------------
# DATA_ROOT holds the unzipped images/, truth/ and metadata/ folders. Change it here,
# set the environment variable SCITAT_DFC_ROOT, or pass --images/--truth/--metadata.
DATA_ROOT = Path(os.environ.get("SCITAT_DFC_ROOT", "/exports/nfs_share/Users/al62352/ResolutionHeatmaps"))
IMAGES_DIR = DATA_ROOT / "images"
TRUTH_DIR = DATA_ROOT / "truth"
METADATA_DIR = DATA_ROOT / "metadata"
# Working files (single-band image copies, pair DSMs, CSVs, calibration results) go next
# to the data, keeping the large files out of the repo; if that folder is not there
# (another computer), they go to calibration/work instead.
WORK_DIR = DATA_ROOT / "calibration_work" if DATA_ROOT.exists() else HERE / "work"

EARTH_R = 6378137.0
WV3_ALT_KM = SENSORS["WorldView3"]["alt_km"]
# UTM zones of the two DFC2019 cities (used when a lidar file carries no CRS)
CITY_EPSG = {"JAX": 32617, "OMA": 32615}

# Default file-name patterns (DFC2019 Track 3). Override with --image-regex / --dsm-glob.
IMAGE_REGEX = r"(?P<tile>[A-Z]{3}_\d{3})_(?P<img>\d{3})_(?P<band>RGB|MSI|PAN)\.tif$"
DSM_GLOB = "{tile}_DSM.tif"
CLS_GLOB = "{tile}_CLS.tif"


# =====================================================================================
# RPC camera model
# =====================================================================================
def _terms(L, P, H):
    """RPC00B polynomial terms (the order GDAL and NITF use). L = lon, P = lat, H = height,
    all normalised."""
    one = np.ones_like(L)
    return np.stack([one, L, P, H, L * P, L * H, P * H, L * L, P * P, H * H,
                     P * L * H, L ** 3, L * P * P, L * H * H, L * L * P, P ** 3, P * H * H,
                     L * L * H, P * P * H, H ** 3])


class RPC:
    """Rational polynomial camera: (lon, lat, h) -> (col, row) and back."""
    KEYS = ("LINE_OFF", "SAMP_OFF", "LAT_OFF", "LONG_OFF", "HEIGHT_OFF", "LINE_SCALE",
            "SAMP_SCALE", "LAT_SCALE", "LONG_SCALE", "HEIGHT_SCALE")

    def __init__(self, d):
        for k in self.KEYS:
            setattr(self, k.lower(), float(str(d[k]).split()[0]))
        def coeffs(v):
            v = v.split() if isinstance(v, str) else v
            return np.asarray([float(x) for x in v][:20], float)
        self.lineNum, self.lineDen, self.sampNum, self.sampDen = (
            coeffs(d[k]) for k in ("LINE_NUM_COEFF", "LINE_DEN_COEFF", "SAMP_NUM_COEFF", "SAMP_DEN_COEFF"))

    def projection(self, lon, lat, h):
        """Ground (deg, deg, m) -> image (col, row) in pixels."""
        L = (np.asarray(lon, float) - self.long_off) / self.long_scale
        P = (np.asarray(lat, float) - self.lat_off) / self.lat_scale
        H = (np.asarray(h, float) - self.height_off) / self.height_scale
        T = _terms(L, P, H)
        row = (self.lineNum @ T) / (self.lineDen @ T)
        col = (self.sampNum @ T) / (self.sampDen @ T)
        return col * self.samp_scale + self.samp_off, row * self.line_scale + self.line_off

    def localization(self, col, row, h, iters=20):
        """Image (col, row) at height h -> ground (lon, lat), by Newton iteration."""
        lon, lat = self.long_off, self.lat_off
        for _ in range(iters):
            c0, r0 = self.projection(lon, lat, h)
            e = 1e-6 * self.long_scale
            c1, r1 = self.projection(lon + e, lat, h)
            c2, r2 = self.projection(lon, lat + e, h)
            J = np.array([[(c1 - c0) / e, (c2 - c0) / e], [(r1 - r0) / e, (r2 - r0) / e]])
            d = np.linalg.solve(J, [col - c0, row - r0])
            lon, lat = lon + d[0], lat + d[1]
            if np.abs(d).max() < 1e-11:
                break
        return float(lon), float(lat)

    def viewAngles(self, lon=None, lat=None, h=None, dh=100.0):
        """Satellite azimuth (deg clockwise from north, seen from the ground point) and
        incidence (deg from vertical) at a ground point (default: the RPC centre).
        Method: the same pixel at two heights; the higher point lies toward the satellite."""
        lon = self.long_off if lon is None else lon
        lat = self.lat_off if lat is None else lat
        h = self.height_off if h is None else h
        col, row = self.projection(lon, lat, h)
        lo1, la1 = self.localization(col, row, h)
        lo2, la2 = self.localization(col, row, h + dh)
        dE = np.radians(lo2 - lo1) * EARTH_R * np.cos(np.radians(lat))
        dN = np.radians(la2 - la1) * EARTH_R
        inc = np.degrees(np.arctan2(np.hypot(dE, dN), dh))
        az = np.degrees(np.arctan2(dE, dN)) % 360
        return float(az), float(inc)

    def gsd(self, lon=None, lat=None, h=None):
        """Ground size (m) of one pixel at a point: geometric mean of the two directions."""
        lon = self.long_off if lon is None else lon
        lat = self.lat_off if lat is None else lat
        h = self.height_off if h is None else h
        col, row = self.projection(lon, lat, h)
        p0 = np.array(self.localization(col, row, h))
        p1 = np.array(self.localization(col + 1, row, h))
        p2 = np.array(self.localization(col, row + 1, h))
        k = np.array([np.cos(np.radians(lat)), 1.0]) * np.radians(1) * EARTH_R
        return float(np.sqrt(np.linalg.norm((p1 - p0) * k) * np.linalg.norm((p2 - p0) * k)))


def _parseRPCText(text):
    """Key/value RPC text (GDAL _RPC.TXT, IKONOS-style or .RPB)."""
    d, coef = {}, {"LINE_NUM_COEFF": {}, "LINE_DEN_COEFF": {}, "SAMP_NUM_COEFF": {}, "SAMP_DEN_COEFF": {}}
    rpb = {"lineOffset": "LINE_OFF", "sampOffset": "SAMP_OFF", "latOffset": "LAT_OFF",
           "longOffset": "LONG_OFF", "heightOffset": "HEIGHT_OFF", "lineScale": "LINE_SCALE",
           "sampScale": "SAMP_SCALE", "latScale": "LAT_SCALE", "longScale": "LONG_SCALE",
           "heightScale": "HEIGHT_SCALE", "lineNumCoef": "LINE_NUM_COEFF",
           "lineDenCoef": "LINE_DEN_COEFF", "sampNumCoef": "SAMP_NUM_COEFF", "sampDenCoef": "SAMP_DEN_COEFF"}
    t = re.sub(r"\s+", " ", text)
    for k, K in rpb.items():                                # .RPB: key = value; / key = ( a, b, ... );
        m = re.search(rf"\b{k}\s*=\s*\(?([^;)]*)\)?\s*;", t)
        if m:
            d[K] = " ".join(m.group(1).replace(",", " ").split())
    for line in text.splitlines():                          # _RPC.TXT: KEY: value [units]
        m = re.match(r"\s*([A-Z_]+?)(?:_(\d+))?\s*[:=]\s*([-+0-9.eE ]+)", line)
        if not m:
            continue
        k, i, v = m.group(1), m.group(2), m.group(3).split()
        if k in coef and i:
            coef[k][int(i)] = float(v[0])
        elif k in coef:
            d[k] = " ".join(v)
        elif k in RPC.KEYS:
            d[k] = v[0]
    for k, c in coef.items():
        if c:
            d[k] = " ".join(str(c[i]) for i in sorted(c))
    return d


def readRPC(path):
    """RPC of an image: GeoTIFF RPC tags (rasterio, then GDAL), then a sidecar
    (<stem>_RPC.TXT, <stem>.RPB, <stem>.rpc, <stem>.txt). Returns RPC or None."""
    path = Path(path)
    d = None
    try:
        import rasterio
        with rasterio.open(path) as src:
            t = src.tags(ns="RPC")
            if not t and src.rpcs is not None:
                t = src.rpcs.to_gdal()
        d = {k: (" ".join(map(str, v)) if isinstance(v, (list, tuple)) else v) for k, v in t.items()} if t else None
    except Exception:
        pass
    if d is None:
        try:
            from osgeo import gdal
            ds = gdal.Open(str(path))
            t = ds.GetMetadata("RPC") if ds is not None else None
            d = t or None
        except Exception:
            pass
    if d is None:
        for side in (path.with_name(path.stem + "_RPC.TXT"), path.with_suffix(".RPB"),
                     path.with_suffix(".rpc"), path.with_suffix(".txt")):
            if side.exists():
                d = _parseRPCText(side.read_text(errors="ignore"))
                if all(k in d for k in RPC.KEYS):
                    break
                d = None
    try:
        return RPC(d) if d else None
    except (KeyError, ValueError):
        return None


def readTags(path):
    """All plain TIFF tags of an image (for dates and inspection)."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            return dict(src.tags())
    except Exception:
        try:
            from osgeo import gdal
            return dict(gdal.Open(str(path)).GetMetadata() or {})
        except Exception:
            return {}


_MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG",
                                    "SEP", "OCT", "NOV", "DEC"], 1)}


def parseDate(text):
    """First acquisition-like date in a string: 2015-09-01, 2015:09:01, 20150901,
    or the DigitalGlobe 01SEP15 form. Returns numpy datetime64[D] or None."""
    s = str(text)
    m = re.search(r"(20\d\d)[-:/_]?([01]\d)[-:/_]?([0-3]\d)", s)
    if m:
        y, mo, d = map(int, m.groups())
    else:
        m = re.search(r"([0-3]\d)(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d\d)", s.upper())
        if not m:
            return None
        d, mo, y = int(m.group(1)), _MON[m.group(2)], 2000 + int(m.group(3))
    try:
        return np.datetime64(f"{y:04d}-{mo:02d}-{d:02d}", "D")
    except ValueError:
        return None


# =====================================================================================
# Files
# =====================================================================================
def findTiles(imageDir, truthDir, imageRegex=IMAGE_REGEX, dsmGlob=DSM_GLOB, band=None):
    """{tile: {"images": {imgId: path}, "dsm": path or None, "cls": path or None}}.
    imgId is "<tile>_<img>". band keeps one band type (RGB / MSI / PAN) if several exist."""
    rx = re.compile(imageRegex)
    tiles = {}
    for p in sorted(Path(imageDir).rglob("*")):
        m = rx.search(p.name)
        if not m or not p.is_file():
            continue
        g = m.groupdict()
        if band and g.get("band") and g["band"] != band:
            continue
        t = tiles.setdefault(g["tile"], {"images": {}, "dsm": None, "cls": None})
        t["images"].setdefault(f'{g["tile"]}_{g["img"]}', p)
    for tile, t in tiles.items():
        for key, pat in (("dsm", dsmGlob), ("cls", CLS_GLOB)):
            hits = sorted(Path(truthDir).rglob(pat.format(tile=tile))) if truthDir else []
            t[key] = hits[0] if hits else None
    return tiles


# =====================================================================================
# Lidar truth and grids
# =====================================================================================
class LidarTile:
    """Lidar DSM of one tile with its UTM grid.
    Georeferencing: the GeoTIFF's own transform and CRS if it has them; otherwise a
    sidecar <name>.txt holding "easting northing size gsd" (DFC2019 style), where
    (easting, northing) is the lower-left corner unless origin="upperleft"."""

    def __init__(self, dsmPath, clsPath=None, epsg=None, origin="lowerleft", nodata=-9999):
        import rasterio
        self.path = Path(dsmPath)
        with rasterio.open(self.path) as src:
            z = src.read(1).astype(float)
            tr, crs = src.transform, src.crs
            nd = src.nodata
        z[(z == nodata) | ~np.isfinite(z)] = np.nan
        if nd is not None:
            z[z == nd] = np.nan
        z[z < -1000] = np.nan
        self.z = z
        self.nrows, self.ncols = z.shape
        side = self.path.with_suffix(".txt")
        if crs is not None and not tr.is_identity:
            self.res = float(tr.a)
            self.x0, self.y0 = float(tr.c), float(tr.f)              # upper-left corner
            self.epsg = crs.to_epsg() or epsg
        elif side.exists():
            e, n, size, gsd = [float(v) for v in side.read_text().split()[:4]]
            self.res = gsd
            self.x0 = e
            self.y0 = n + self.nrows * gsd if origin == "lowerleft" else n
            self.epsg = epsg
        else:
            raise ValueError(f"{self.path.name}: no georeferencing (no CRS in the TIFF and no {side.name})")
        if self.epsg is None:
            self.epsg = CITY_EPSG.get(self.path.name[:3])
        self.cls = None
        if clsPath is not None and Path(clsPath).exists():
            with rasterio.open(clsPath) as src:
                self.cls = src.read(1)

    @property
    def transform(self):
        from rasterio.transform import from_origin
        return from_origin(self.x0, self.y0, self.res, self.res)

    def centerLonLat(self):
        from pyproj import Transformer
        tf = Transformer.from_crs(self.epsg, 4326, always_xy=True)
        x = self.x0 + self.ncols * self.res / 2
        y = self.y0 - self.nrows * self.res / 2
        lon, lat = tf.transform(x, y)
        return float(lon), float(lat), float(np.nanmedian(self.z))

    def lonLatPolygon(self, pad=0.0):
        """GeoJSON polygon (lon, lat) of the tile, padded by `pad` metres (for s2p)."""
        from pyproj import Transformer
        tf = Transformer.from_crs(self.epsg, 4326, always_xy=True)
        x1, x2 = self.x0 - pad, self.x0 + self.ncols * self.res + pad
        y1, y2 = self.y0 - self.nrows * self.res - pad, self.y0 + pad
        ring = [tf.transform(x, y) for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1))]
        return {"type": "Polygon", "coordinates": [[list(map(float, p)) for p in ring]]}

    def evalMask(self, margin=10, excludeClasses=(9,)):
        """Cells used for scoring: valid lidar, away from the edge, not excluded classes
        (DFC2019 classes: 2 ground, 5 trees, 6 building, 9 water, 17 bridge, 65 unlabeled)."""
        m = np.isfinite(self.z)
        m[:margin] = m[-margin:] = False
        m[:, :margin] = m[:, -margin:] = False
        if self.cls is not None and excludeClasses:
            m &= ~np.isin(self.cls, list(excludeClasses))
        return m

    def sceneForModel(self, cellM=1.0):
        """(dsm, factor) for imageGeometry.registerScene: the lidar block-averaged to
        about cellM, gaps filled with low ground so they never block a ray."""
        f = max(1, int(round(cellM / self.res)))
        z = self.z.copy()
        fill = np.nanpercentile(z, 5) if np.isfinite(z).any() else 0.0
        z[~np.isfinite(z)] = fill
        r, c = (self.nrows // f) * f, (self.ncols // f) * f
        z = z[:r, :c].reshape(r // f, f, c // f, f).mean(axis=(1, 3))
        return z - np.percentile(z, 5), f        # heights relative to the ground


def toGrid(path, tile, resampling="average"):
    """Resample a GeoTIFF DSM (any CRS) onto a LidarTile's grid. NaN where empty."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS
    out = np.full((tile.nrows, tile.ncols), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        src_arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            src_arr[src_arr == src.nodata] = np.nan
        reproject(src_arr, out, src_transform=src.transform, src_crs=src.crs or CRS.from_epsg(tile.epsg),
                  src_nodata=np.nan, dst_transform=tile.transform, dst_crs=CRS.from_epsg(tile.epsg),
                  dst_nodata=np.nan, resampling=getattr(Resampling, resampling))
    return out


def saveGrid(arr, tile, path):
    """Write an array on a tile's grid as a GeoTIFF."""
    import rasterio
    from rasterio.crs import CRS
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs=CRS.from_epsg(tile.epsg), transform=tile.transform,
                       nodata=np.nan, compress="deflate") as dst:
        dst.write(arr.astype(np.float32), 1)


# =====================================================================================
# Geometry conversions and the model
# =====================================================================================
def offNadirFromIncidence(incDeg, altKm=WV3_ALT_KM):
    """Spherical Earth: off-nadir angle at the satellite for an incidence angle at the ground."""
    k = (EARTH_R + altKm * 1e3) / EARTH_R
    return np.degrees(np.arcsin(np.sin(np.radians(incDeg)) / k))


def scitatAngles(satAzDeg, incDeg, altKm=WV3_ALT_KM):
    """Satellite azimuth (from the ground, clockwise from north) and incidence ->
    SCITAT (along, cross) pointing angles, with along = south and cross = east (the
    row / column directions of a north-up raster; see the module docstring)."""
    th = np.radians(offNadirFromIncidence(incDeg, altKm))
    az = -np.radians(satAzDeg)                    # look direction in (south, east)
    along = np.degrees(np.arctan(np.tan(th) * np.cos(az)))
    cross = np.degrees(np.arctan(np.tan(th) * np.sin(az)))
    return along, cross


def makeGeometry(sensor="WorldView3", altKm=None):
    """imageGeometry set up for predictedErrorM (no GSD images needed: pass
    availability=1 when calling it)."""
    alt = SENSORS[sensor]["alt_km"] if altKm is None else altKm
    vec = np.zeros(1)
    geo = imageGeometry(0, vec, vec, sensArray(sensor, alt), vec,
                        np.array([0.0, SENSORS[sensor]["maxOffNadir"]]), 0.0, 1.0, alt, 0.5, 1.0)
    geo.alt = alt
    return geo


def predict(geo, scene, angles, dates=None, altKm=WV3_ALT_KM):
    """predictedErrorM for one set on a registered scene."""
    geo.errScene = scene
    angles = np.asarray(angles, float).reshape(-1, 2)
    imData = np.column_stack([np.zeros((len(angles), 2)), np.full(len(angles), altKm)])
    return geo.predictedErrorM(None, imData, angles, availability=1.0, dates=dates)


# =====================================================================================
# Subset design
# =====================================================================================
def _features(inc, az, doy):
    """Feature vector per image for spreading choices: the unit vector toward the
    satellite (so azimuth only matters off nadir) plus season on a circle."""
    i, a, s = np.radians(inc), np.radians(az), 2 * np.pi * np.asarray(doy, float) / 365.25
    return np.column_stack([np.sin(i) * np.sin(a), np.sin(i) * np.cos(a), np.cos(i),
                            0.3 * np.cos(s), 0.3 * np.sin(s)])


def farthestPoint(F, k, start=None, rng=None):
    """Indices of k rows of F spread out greedily (farthest-point sampling)."""
    rng = rng or np.random.default_rng(0)
    idx = [int(rng.integers(len(F))) if start is None else int(start)]
    d = np.linalg.norm(F - F[idx[0]], axis=1)
    while len(idx) < min(k, len(F)):
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, np.linalg.norm(F - F[j], axis=1))
    return idx


def designSubsets(img, poolSize=10, nRandom=6, nSpread=4, nNarrow=4, sizes=(3, 4, 5, 6, 8),
                  nTriples=10, maxIncidence=45.0, rng=None):
    """Subsets for one tile.
    img: DataFrame rows for that tile with columns image_id, incidence_deg, sat_az_deg, doy.
    Returns (pool image ids, list of (kind, [image ids])). Every pair inside the pool is a
    subset ("pair"), so every pair reconstruction gets used on its own as well."""
    from itertools import combinations
    rng = rng or np.random.default_rng(0)
    img = img[img.incidence_deg <= maxIncidence].reset_index(drop=True)
    if len(img) < 2:
        return [], []
    doy = img["doy"].fillna(0).to_numpy()
    F = _features(img.incidence_deg.to_numpy(), img.sat_az_deg.to_numpy(), doy)
    near = int(np.argmin(img.incidence_deg.to_numpy()))
    pool = farthestPoint(F, poolSize, start=near, rng=rng)
    ids = img.image_id.to_numpy()[pool]
    Fp = F[pool]
    subs = [("pair", [ids[a], ids[b]]) for a, b in combinations(range(len(pool)), 2)]
    trip = list(combinations(range(len(pool)), 3))
    for t in rng.permutation(len(trip))[:nTriples]:
        subs.append(("triple", [ids[i] for i in trip[t]]))
    for k in sizes:
        if k > len(pool):
            continue
        for _ in range(nRandom):
            subs.append(("random", list(ids[rng.choice(len(pool), k, replace=False)])))
        for _ in range(nSpread):
            subs.append(("spread", list(ids[farthestPoint(Fp, k, rng=rng)])))
        for _ in range(nNarrow):
            s = int(rng.integers(len(pool)))
            order = np.argsort(np.linalg.norm(Fp - Fp[s], axis=1))[:k]
            subs.append(("narrow", list(ids[order])))
    seen, out = set(), []
    for kind, s in subs:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            out.append((kind, list(key)))
    return list(ids), out


def readJSON(path):
    return json.loads(Path(path).read_text())
