"""
Which pairs failed, and do they share images?
=============================================
Lists every pair in pairs.csv without a dsm.tif, with its geometry (incidence,
convergence, date gap) and the error at the end of its s2p log, and counts how
often each image appears among the failures relative to how often it was used.
An image that fails with most of its partners is probably the problem (cloud, snow,
haze, a bad camera model); a pair whose images both work elsewhere failed because of
the pair itself (very wide convergence, long date gap, season change).

It also writes a small preview PNG of every image involved (<work>/failed/previews),
so you can look for clouds or snow.

Run:  python checkFailedPairs.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dfc2019Common import WORK_DIR, makeGeometry, WV3_ALT_KM


def preview(src, dst, size=512):
    import rasterio
    from rasterio.enums import Resampling
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with rasterio.open(src) as s:
        f = max(1, max(s.width, s.height) // size)
        a = s.read(out_shape=(s.count, s.height // f, s.width // f), resampling=Resampling.average).astype(float)
    a = a[:3] if a.shape[0] >= 3 else a[:1].repeat(3, 0)
    lo, hi = np.percentile(a, [1, 99])
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1).transpose(1, 2, 0)
    plt.imsave(dst, a)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--work", type=Path, default=WORK_DIR)
    p.add_argument("--no-previews", action="store_true")
    args = p.parse_args()

    img = pd.read_csv(args.work / "images.csv").set_index("image_id")
    pairs = pd.read_csv(args.work / "pairs.csv")
    pairs["ok"] = [(args.work / "pairs" / r.pair_id / "dsm.tif").exists() for r in pairs.itertuples()]
    geo = makeGeometry("WorldView3")
    rows = []
    for r in pairs.itertuples():
        a, b = img.loc[r.image_a], img.loc[r.image_b]
        u = geo.lookVectors(np.array([[a.along_deg, a.cross_deg], [b.along_deg, b.cross_deg]]), WV3_ALT_KM)
        conv = float(np.degrees(np.arccos(np.clip(u[0] @ u[1], -1, 1))))
        gap = abs((np.datetime64(a.date) - np.datetime64(b.date)).astype(int)) if isinstance(a.date, str) else np.nan
        err = ""
        log = args.work / "pairs" / r.pair_id / "log.txt"
        if not r.ok and log.exists():
            lines = [l.strip() for l in log.read_text(errors="ignore").splitlines() if l.strip()]
            err = lines[-1][:90] if lines else ""
        rows.append(dict(pair_id=r.pair_id, tile=r.tile, ok=r.ok, image_a=r.image_a, image_b=r.image_b,
                         inc_a=a.incidence_deg, inc_b=b.incidence_deg, convergence_deg=conv,
                         date_a=a.date, date_b=b.date, gap_days=gap, error=err))
    df = pd.DataFrame(rows)
    fail = df[~df.ok]
    out = args.work / "failed"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "all_pairs_geometry.csv", index=False)
    print(f"{len(fail)} of {len(df)} pairs failed\n")
    if fail.empty:
        return
    pd.set_option("display.width", 200)
    print(fail[["pair_id", "inc_a", "inc_b", "convergence_deg", "date_a", "date_b", "gap_days"]]
          .round(1).to_string(index=False))
    print("\nLast log line of the failures:")
    print(fail.error.value_counts().to_string())

    print("\nOK vs failed pairs (medians):")
    print(df.groupby("ok")[["convergence_deg", "gap_days", "inc_a", "inc_b"]].median().round(1).to_string())

    used = pd.concat([df.image_a, df.image_b]).value_counts()
    failed = pd.concat([fail.image_a, fail.image_b]).value_counts()
    share = (failed / used.reindex(failed.index)).sort_values(ascending=False)
    t = pd.DataFrame({"failed_pairs": failed, "pairs_used": used.reindex(failed.index), "fail_share": share.round(2)})
    t["date"] = [img.loc[i, "date"] for i in t.index]
    t["incidence"] = [round(img.loc[i, "incidence_deg"], 1) for i in t.index]
    print("\nImages among the failures (fail_share near 1 = the image is probably the problem):")
    print(t.sort_values("fail_share", ascending=False).to_string())
    # the same acquisition across tiles: OMA_xxx_010 in two tiles is the same satellite image
    acq = pd.Series([i.split("_")[0] + "_*_" + i.split("_")[2] for i in pd.concat([fail.image_a, fail.image_b])])
    print("\nSame acquisition across tiles (city_*_image):")
    print(acq.value_counts().head(10).to_string())
    fail.to_csv(out / "failed_pairs.csv", index=False)
    if not args.no_previews:
        (out / "previews").mkdir(exist_ok=True)
        for i in sorted(set(fail.image_a) | set(fail.image_b)):
            try:
                preview(img.loc[i, "path"], out / "previews" / f"{i}.png")
            except Exception as e:
                print(f"  preview {i}: {e}")
        print(f"\nPreviews of the images involved: {out / 'previews'} (look for cloud, snow, haze)")
    print(f"Tables: {out}")


if __name__ == "__main__":
    main()
