"""
Step 2 - Choose the image subsets to reconstruct
================================================
Per tile, picks a pool of images that spans the geometry (incidence, azimuth,
season), then subsets of the pool:
    pair     every pair in the pool (each is one stereo reconstruction)
    triple   random triples
    random   k random pool images (k = 3, 4, 5, 6, 8)
    spread   k images spread as far apart as possible (wide azimuth / angle / season)
    narrow   k images clustered together (small convergence)
Multi-view subsets are built by fusing the pair reconstructions inside them, so the
only reconstructions to run are the pool pairs: poolSize*(poolSize-1)/2 per tile
(45 for a pool of 10).

Outputs (in --work)
  pairs.csv     pair_id, tile, image_a, image_b, paths   (the jobs for step 3)
  subsets.csv   subset_id, tile, kind, n, image_ids (';'-separated)

Run
  python dfc2019Subsets.py --tiles 12 --pool 10
  python dfc2019Subsets.py --tile-list JAX_004,JAX_068,OMA_212
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dfc2019Common import LidarTile, designSubsets, WORK_DIR


def builtShare(row):
    """Share of the tile covered by buildings (class 6) or, without classes, above 3 m."""
    try:
        L = LidarTile(row.lidar, row.cls or None)
        if L.cls is not None:
            return float(np.mean(L.cls == 6))
        z = L.z - np.nanpercentile(L.z, 5)
        return float(np.nanmean(z > 3))
    except Exception:
        return np.nan


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", type=Path, default=WORK_DIR, help=f"working folder (default {WORK_DIR})")
    p.add_argument("--tiles", type=int, default=12, help="number of tiles (chosen to span building density)")
    p.add_argument("--tile-list", default=None, help="comma-separated tiles to use instead")
    p.add_argument("--pool", type=int, default=10, help="images per tile; pairs to run = pool*(pool-1)/2")
    p.add_argument("--max-incidence", type=float, default=45.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    img = pd.read_csv(args.work / "images.csv")
    tiles = img.groupby("tile").first().reset_index()
    if args.tile_list:
        keep = [t.strip() for t in args.tile_list.split(",")]
        tiles = tiles[tiles.tile.isin(keep)]
    else:
        tiles["built"] = [builtShare(r) for r in tiles.itertuples()]
        tiles = tiles.sort_values("built").reset_index(drop=True)
        # spread over building density, both cities
        pick = np.unique(np.linspace(0, len(tiles) - 1, min(args.tiles, len(tiles))).round().astype(int))
        tiles = tiles.iloc[pick]
        print(tiles[["tile", "built"]].to_string(index=False))

    pairs, subsets = [], []
    for tile in tiles.tile:
        ti = img[img.tile == tile]
        pool, subs = designSubsets(ti, poolSize=args.pool, maxIncidence=args.max_incidence, rng=rng)
        path = dict(zip(ti.image_id, ti.path))
        for kind, s in subs:
            subsets.append(dict(subset_id=f"{tile}_s{len(subsets):05d}", tile=tile, kind=kind, n=len(s),
                                image_ids=";".join(s)))
            if kind == "pair":
                pairs.append(dict(pair_id=f"{s[0]}__{s[1]}", tile=tile, image_a=s[0], image_b=s[1],
                                  path_a=path[s[0]], path_b=path[s[1]]))
        print(f"{tile}: pool {len(pool)} images -> {sum(1 for k, _ in subs if k == 'pair')} pairs, "
              f"{len(subs)} subsets")
    pd.DataFrame(pairs).to_csv(args.work / "pairs.csv", index=False)
    sd = pd.DataFrame(subsets)
    sd.to_csv(args.work / "subsets.csv", index=False)
    print(f"\n{len(pairs)} pair reconstructions to run, {len(sd)} subsets "
          f"({sd.kind.value_counts().to_dict()}) -> {args.work}")


if __name__ == "__main__":
    main()
