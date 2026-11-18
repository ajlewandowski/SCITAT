"""
Step 3 - Run the stereo reconstructions
=======================================
Runs one stereo reconstruction per pair in pairs.csv, over the lidar tile (plus a
small pad), on the lidar grid's resolution and UTM zone. Multi-view subsets are
made later (step 4) by fusing these pair DSMs, so each pair runs once however many
subsets use it. Finished pairs (<work>/pairs/<pair_id>/dsm.tif) are skipped, so the
script can be stopped and restarted.

Default engine: s2p (CNES / ENS Paris-Saclay, https://github.com/centreborelli/s2p,
`pip install s2p`, needs GDAL, fftw and libtiff). It reads the RPCs from the GeoTIFFs.
Multi-band tiles are first averaged to one band (RPC tags kept) in <work>/gray.

Any other pipeline: --engine-cmd with a command template; placeholders
  {left} {right}   single-band image paths      {out_dir}  folder; must produce dsm.tif
  {roi}            GeoJSON polygon file (lon/lat) {epsg}    output UTM EPSG code
  {res}            DSM resolution (m)
e.g. --engine-cmd "bash asp_pair.sh {left} {right} {out_dir} {epsg} {res}"

Run
  python runReconstructions.py --dry-run          # write configs, print commands
  python runReconstructions.py --only JAX_004     # one tile first, to check
  python runReconstructions.py --jobs 4 --threads 4
"""
import argparse
import json
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from dfc2019Common import LidarTile, WORK_DIR


def singleBand(src, dst):
    """Average the bands of a GeoTIFF into one float32 band, keeping the RPC tags."""
    import rasterio
    if dst.exists():
        return dst
    with rasterio.open(src) as s:
        if s.count == 1:
            return Path(src)
        arr = s.read().astype(np.float32).mean(axis=0)
        prof = s.profile.copy()
        rpcs = s.rpcs
        tags = s.tags()
    prof.update(count=1, dtype="float32", compress="deflate", photometric=None)
    prof.pop("photometric", None)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst, "w", rpcs=rpcs, **prof) as d:
        d.write(arr, 1)
        d.update_tags(**tags)
    return dst


def jobFor(row, args, tiles):
    tile = tiles[row.tile]
    out = args.work / "pairs" / row.pair_id
    left = singleBand(Path(row.path_a), args.work / "gray" / f"{row.image_a}.tif")
    right = singleBand(Path(row.path_b), args.work / "gray" / f"{row.image_b}.tif")
    out.mkdir(parents=True, exist_ok=True)
    roi = tile.lonLatPolygon(pad=args.pad)
    (out / "roi.geojson").write_text(json.dumps(roi))
    if args.engine_cmd:
        cmd = shlex.split(args.engine_cmd.format(left=left, right=right, out_dir=out, roi=out / "roi.geojson",
                                                 epsg=tile.epsg, res=tile.res))
    else:
        cfg = {"out_dir": str(out), "temporary_dir": str(out / "tmp"),
               "images": [{"img": str(left)}, {"img": str(right)}],
               "roi_geojson": roi, "dsm_resolution": float(tile.res),
               "out_crs": f"epsg:{tile.epsg}", "max_processes": args.threads,
               "clean_intermediate": True, "clean_tmp": True}
        cfg.update(json.loads(args.s2p_extra) if args.s2p_extra else {})
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
        cmd = ["s2p", str(out / "config.json")]
    return out, cmd


def run(out, cmd, timeout):
    t = time.time()
    with open(out / "log.txt", "w") as log:
        try:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = "timeout"
        except FileNotFoundError:
            rc = "engine not found"
    ok = (out / "dsm.tif").exists()
    return out.name, ok, rc, time.time() - t


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", type=Path, default=WORK_DIR, help=f"working folder (default {WORK_DIR})")
    p.add_argument("--jobs", type=int, default=1, help="pairs run at the same time")
    p.add_argument("--threads", type=int, default=4, help="processes each s2p run may use")
    p.add_argument("--pad", type=float, default=20.0, help="metres added around the tile")
    p.add_argument("--only", default=None, help="comma-separated tiles to run")
    p.add_argument("--engine-cmd", default=None, help="command template for another pipeline (see above)")
    p.add_argument("--s2p-extra", default=None, help='extra s2p settings as JSON, e.g. \'{"matching_algorithm": "mgm_multi"}\'')
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    pairs = pd.read_csv(args.work / "pairs.csv")
    if args.only:
        pairs = pairs[pairs.tile.isin([t.strip() for t in args.only.split(",")])]
    img = pd.read_csv(args.work / "images.csv").groupby("tile").first()
    tiles = {t: LidarTile(img.loc[t, "lidar"], img.loc[t, "cls"] if isinstance(img.loc[t, "cls"], str) else None)
             for t in pairs.tile.unique()}
    todo = [r for r in pairs.itertuples() if not (args.work / "pairs" / r.pair_id / "dsm.tif").exists()]
    print(f"{len(pairs)} pairs, {len(pairs) - len(todo)} done, {len(todo)} to run")
    jobs = [jobFor(r, args, tiles) for r in todo]
    if args.dry_run:
        for out, cmd in jobs[:5]:
            print("  ", " ".join(map(str, cmd)))
        print(f"  ... configs written for {len(jobs)} pairs. Remove --dry-run to run them.")
        return
    t0, nOk = time.time(), 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run, out, cmd, args.timeout) for out, cmd in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            name, ok, rc, dt = f.result()
            nOk += ok
            el = time.time() - t0
            print(f"[{i}/{len(jobs)}] {name}: {'ok' if ok else f'FAILED ({rc}), see log.txt'} in {dt:.0f} s; "
                  f"about {el / i * (len(jobs) - i) / 60:.0f} min left", flush=True)
    print(f"\n{nOk} of {len(jobs)} pairs reconstructed.")


if __name__ == "__main__":
    main()
