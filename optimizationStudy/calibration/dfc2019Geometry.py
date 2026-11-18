"""
Step 1 - Image geometry of the DFC2019 tiles
=============================================
For every Track 3 image tile: read its RPC camera model and work out, at the
centre of the lidar tile,
    satellite azimuth and incidence angle   (from the RPC: one pixel at two heights)
    off-nadir angle                          (spherical Earth, WorldView-3 at 617 km)
    SCITAT (along, cross) pointing angles    (along = south, cross = east)
    ground pixel size                        (from the RPC)
    acquisition date                         (TIFF tags, --dates CSV, or file names)

Output: <work>/images.csv, one row per image (work defaults to DATA_ROOT/calibration_work).

Run
  python dfc2019Geometry.py --inspect
  python dfc2019Geometry.py
  python dfc2019Geometry.py --dates my_dates.csv           # columns image_id,date (optional)
The data folders default to DATA_ROOT in dfc2019Common.py
(/exports/nfs_share/Users/al62352/ResolutionHeatmaps/{images,truth,metadata});
--images / --truth / --metadata override them.

--inspect prints what the first files contain (tags, RPC, georeferencing) and
stops. Run it first: if it finds no RPC or no dates, it says where it looked.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dfc2019Common import (IMAGE_REGEX, DSM_GLOB, IMAGES_DIR, TRUTH_DIR, METADATA_DIR, WORK_DIR, LidarTile, findTiles, readRPC, readTags, parseDate,
                           offNadirFromIncidence, scitatAngles)


# Tags holding the acquisition time, best first. NITF_IDATIM is the image capture time
# in DFC2019 / DigitalGlobe NITF-derived tiles; NITF_FDT is the FILE date (when the tile
# was made, 2017) and must not be used.
DATE_TAGS = ("NITF_IDATIM", "NITF_STDIDC_ACQUISITION_DATE", "ACQUISITION_DATE", "TIFFTAG_DATETIME")
NOT_DATE_TAGS = ("NITF_FDT",)
TITLE_TAGS = ("NITF_IID2", "NITF_FTITLE")      # e.g. 14DEC14WV031100014DEC14160402-M1BS-...


def findDate(imgPath, tags, metaIndex):
    """Acquisition date: known capture-time tags, other tags with DATE/TIME in the key,
    the DigitalGlobe image title (14DEC14WV03...), the metadata index, the file name."""
    for k in DATE_TAGS:
        if k in tags and parseDate(tags[k]) is not None:
            return parseDate(tags[k]), f"tag {k}"
    for k, v in tags.items():
        if (("DATE" in k.upper() or "TIME" in k.upper() or k.upper().endswith("DATIM"))
                and k not in NOT_DATE_TAGS):
            d = parseDate(v)
            if d is not None:
                return d, f"tag {k}"
    for k in TITLE_TAGS:
        if k in tags:
            d = parseDate(str(tags[k])[:7])          # leading DDMONYY
            if d is not None:
                return d, f"tag {k}"
    for key in (imgPath.stem, imgPath.stem.rsplit("_", 1)[0]):
        if key in metaIndex:
            return metaIndex[key], "metadata"
    d = parseDate(imgPath.name)
    return (d, "file name") if d is not None else (None, "not found")


def indexMetadata(metaDir):
    """Look through every text file under the metadata folder for lines that hold an
    image name (e.g. JAX_004_006) and a date; returns {name: date}."""
    import re
    out = {}
    if not metaDir:
        return out
    for p in Path(metaDir).rglob("*"):
        if not p.is_file() or p.stat().st_size > 50e6:
            continue
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            continue
        for line in text.splitlines():
            names = re.findall(r"[A-Z]{3}_\d{3}_\d{3}", line)
            d = parseDate(line)
            if names and d is not None:
                for n in names:
                    out.setdefault(n, d)
        # a per-image metadata file named after the image (e.g. JAX_004_006.txt / .IMD)
        m = re.search(r"[A-Z]{3}_\d{3}_\d{3}", p.name)
        if m and m.group(0) not in out:
            d = parseDate(text)
            if d is not None:
                out[m.group(0)] = d
    return out


def footprintCheck(rpc, imgPath, t, args):
    """Where the lidar tile lands in the image, under both readings of the lidar .txt
    corner. The DFC2019 image tiles are cut around the lidar tile, so the right reading
    puts the tile inside the image, roughly centred, with its lowest points matching."""
    import rasterio
    with rasterio.open(imgPath) as s:
        W, H = s.width, s.height
    for origin in ("lowerleft", "upperleft"):
        try:
            L = LidarTile(t["dsm"], t["cls"], origin=origin)
        except Exception as e:
            print(f"    lidar ({origin}): {e}"); return
        lon, lat, h = L.centerLonLat()
        c, r = rpc.projection(lon, lat, h)
        inside = 0 <= c < W and 0 <= r < H
        print(f"    lidar centre with .txt corner = {origin:<9s}: pixel ({c:7.0f}, {r:7.0f}) in a {W}x{H} image"
              f" -> {'inside' if inside else 'OUTSIDE'}, {np.hypot(c - W / 2, r - H / 2):.0f} px from its centre")
        if L.epsg and L.res and not (L.path.with_suffix('.txt').exists()):
            break


def inspect(tiles, args, n=2):
    print(f"Found {len(tiles)} tiles, {sum(len(t['images']) for t in tiles.values())} images.")
    for tile, t in list(tiles.items())[:n]:
        print(f"\n=== tile {tile}: {len(t['images'])} images, DSM: {t['dsm']}, CLS: {t['cls']}")
        for iid, p in list(t["images"].items())[:2]:
            print(f"--- {p}")
            tags = readTags(p)
            print("    TIFF tags:", {k: (v[:60] + "...") if len(v) > 60 else v for k, v in tags.items()} or "none")
            d, src = findDate(p, tags, {})
            print(f"    acquisition date: {d} ({src})")
            rpc = readRPC(p)
            if rpc is None:
                print("    RPC: NOT FOUND (looked in GeoTIFF RPC tags and _RPC.TXT/.RPB/.rpc/.txt sidecars)")
            else:
                az, inc = rpc.viewAngles()
                print(f"    RPC: ok. centre {rpc.lat_off:.4f}N {rpc.long_off:.4f}E; satellite azimuth "
                      f"{az:.1f}°, incidence {inc:.1f}°, GSD {rpc.gsd():.2f} m")
                if t["dsm"]:
                    footprintCheck(rpc, p, t, args)
        if t["dsm"]:
            side = Path(t["dsm"]).with_suffix(".txt")
            if side.exists():
                print(f"    lidar .txt: {side.read_text().split()}")
            try:
                L = LidarTile(t["dsm"], t["cls"], origin=args.origin)
                lon, lat, h = L.centerLonLat()
                print(f"    lidar: {L.nrows}x{L.ncols} at {L.res} m, EPSG {L.epsg}, centre {lat:.4f}N {lon:.4f}E, "
                      f"height {np.nanmin(L.z):.1f}–{np.nanmax(L.z):.1f} m")
            except Exception as e:
                print(f"    lidar: could not read georeferencing: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, default=IMAGES_DIR,
                   help=f"folder with the Track 3 image tiles, searched recursively (default {IMAGES_DIR})")
    p.add_argument("--truth", type=Path, default=TRUTH_DIR,
                   help=f"folder with the Track 3 lidar truth: DSM, CLS (default {TRUTH_DIR})")
    p.add_argument("--metadata", type=Path, default=METADATA_DIR,
                   help=f"unzipped Track3-Metadata folder, for dates (default {METADATA_DIR})")
    p.add_argument("--dates", type=Path, default=None, help="CSV image_id,date overriding everything else")
    p.add_argument("--work", type=Path, default=WORK_DIR, help=f"working folder (default {WORK_DIR})")
    p.add_argument("--band", default="RGB", help="image type to use if several exist (RGB, MSI, PAN)")
    p.add_argument("--image-regex", default=IMAGE_REGEX)
    p.add_argument("--dsm-glob", default=DSM_GLOB)
    p.add_argument("--origin", default="lowerleft", choices=["lowerleft", "upperleft"],
                   help="corner given in the lidar .txt sidecar (only used if the DSM TIFF has no transform)")
    p.add_argument("--inspect", action="store_true", help="print what the files contain and stop")
    args = p.parse_args()

    for name, d in (("--images", args.images), ("--truth", args.truth)):
        if not d.exists():
            raise SystemExit(f"{name} folder not found: {d}")
    if args.metadata is not None and not args.metadata.exists():
        args.metadata = None
    tiles = findTiles(args.images, args.truth, args.image_regex, args.dsm_glob, args.band)
    if not tiles:
        raise SystemExit(f"No images matching {args.image_regex!r} under {args.images}")
    if args.inspect:
        inspect(tiles, args)
        return

    metaIndex = indexMetadata(args.metadata)
    override = {}
    if args.dates:
        d = pd.read_csv(args.dates)
        override = {r.image_id: parseDate(r.date) for r in d.itertuples()}

    rows, noRPC, noDSM = [], 0, []
    for tile, t in tiles.items():
        if t["dsm"] is None:
            noDSM.append(tile)
            continue
        L = LidarTile(t["dsm"], t["cls"], origin=args.origin)
        lon, lat, h = L.centerLonLat()
        for iid, path in t["images"].items():
            rpc = readRPC(path)
            if rpc is None:
                noRPC += 1
                continue
            az, inc = rpc.viewAngles(lon, lat, h)
            along, cross = scitatAngles(az, inc)
            if iid in override and override[iid] is not None:
                date, src = override[iid], "dates CSV"
            else:
                date, src = findDate(path, readTags(path), metaIndex)
            rows.append(dict(tile=tile, image_id=iid, path=str(path), sat_az_deg=az, incidence_deg=inc,
                             offnadir_deg=float(offNadirFromIncidence(inc)), along_deg=float(along),
                             cross_deg=float(cross), sat_elev_deg=90 - inc, gsd_m=rpc.gsd(lon, lat, h),
                             date=None if date is None else str(date), date_source=src,
                             doy=None if date is None else int((date - date.astype("datetime64[Y]")).astype(int)) + 1,
                             lidar=str(t["dsm"]), cls=str(t["cls"]) if t["cls"] else ""))
    df = pd.DataFrame(rows)
    args.work.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.work / "images.csv", index=False)
    print(f"{len(df)} images on {df.tile.nunique() if len(df) else 0} tiles -> {args.work / 'images.csv'}")
    if noRPC:
        print(f"  {noRPC} images skipped: no RPC (run --inspect)")
    if noDSM:
        print(f"  {len(noDSM)} tiles skipped: no lidar DSM (validation/test tiles have none): {', '.join(noDSM[:8])}...")
    if len(df):
        print(f"  incidence {df.incidence_deg.min():.1f}–{df.incidence_deg.max():.1f}°, "
              f"GSD {df.gsd_m.min():.2f}–{df.gsd_m.max():.2f} m, "
              f"dates found for {df.date.notna().sum()} of {len(df)}")
        if df.date.isna().any():
            print("  Without dates the calibration skips the date-gap term. Give dates with --dates "
                  "(CSV image_id,date) or --metadata.")


if __name__ == "__main__":
    main()
