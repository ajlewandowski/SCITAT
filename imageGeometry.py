"""
This file contains the class that accounts for image geometry analysis and optimization
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.ndimage import label, maximum_filter


# This class will import system parameters and produce a resolution "heat map"

class imageGeometry:
    def __init__(self, noradID, state84, times, sens, target, angVec, minGSD, maxGSD, altitude, sigmaPix, eMax):
        self.norad = noradID
        self.state84 = state84
        self.sens = sens
        self.target = target
        self.times = times
        self.IFOVbase = sens[0] / sens[2]
        self.angVec = angVec
        self.minGSD = minGSD
        self.maxGSD = maxGSD
        self.alt = altitude
        self.sigmaPix = sigmaPix
        self.eMax = eMax

        self.wgsf = 1 / 298.257223563
        self.ae = 6378137.0
        self.wgsSMA = self.ae

        self.targetBHCanopy = 0.25
        self.targetBH = 1.0
        self.targetOverlap = 75.0
        self.pixGSDlimFactor = 0.25

    def getCostFunction(self, ims, imData, imAngles):

        overlapMean, overlapSlices, overlap = self.getOverlapRadius(ims, imData)
        overlapRad = overlap["centered"]["rGeo"]
        errOverlap = (((ims[0].shape[0] // 2) - overlapRad) / (ims[0].shape[0] // 2))

        errGSD = self.getGSDError(overlapMean)
        
        bhRatio = self.getBHRatio(imData)
        errbhRatio = (abs(self.targetBH - bhRatio) / self.targetBH)
        
        errCov = 1.0 - self.getCoverage(ims, imData)
        
        costMap = self.computeMVSCostMap(overlapSlices, imAngles, self.sigmaPix, self.eMax)
        errCostMap = np.mean(costMap)
        
        cost = (0.2*errOverlap + 0.2*errGSD + 0.2*errbhRatio + 0.2*errCov + 0.2*errCostMap)

        return cost

    def getOverlapRadius(self, ims, imData, window="centered",
                        nan_policy="propagate", centerRef="floor",
                        dtype=np.float64):
        """
        Align N images on their targets and measure the common region.

        ims        : sequence of N 2-D arrays (shapes may differ)
        imData     : (N, ...) per-image data passed to self.getImCenterCoords
        window     : "centered" -> cutout symmetric about the target
                    "max"      -> largest common region (target off-center)
                    which window gets averaged; both are always measured
        nan_policy : "propagate" -> a NaN in any frame gives NaN in the mean
                    "omit"      -> average over the valid frames only
        centerRef  : "floor" -> center pixel is shape // 2
                    "exact" -> center is (shape - 1) / 2
                    must match the convention used by getImCenterCoords

        Returns (mean, stack, overlap):
            mean    : (nRows, nCols) averaged cutout for the chosen window
            stack   : (N, nRows, nCols) the aligned cutouts
            overlap : dict with "centered" and "max" sub-dicts, each holding
                    the window dims, slices, target index, and scalar sizes.
                    "max" additionally holds the four directional margins.

        Directional margins are in ARRAY orientation: "up" means toward lower
        row index (the top of the array as displayed by imshow with the default
        origin="upper"). Flip up/down if your row axis is y-up.
        """
        imgs = list(ims)
        n = len(imgs)
        if n == 0:
            raise ValueError("no images given")

        # --- target offsets relative to each image's center pixel ---
        offsets = np.empty((n, 2), dtype=float)
        for i, im in enumerate(imgs):
            imRow, imCol = self.getImCenterCoords(im, imData[i, :])
            offsets[i] = (imRow, imCol)

        shapes = np.array([im.shape[:2] for im in imgs])

        if centerRef == "floor":
            centers = shapes // 2
        elif centerRef == "exact":
            centers = (shapes - 1) / 2.0
        else:
            raise ValueError("centerRef must be 'floor' or 'exact'")

        anchors = np.rint(centers + offsets).astype(int)

        outside = np.any((anchors < 0) | (anchors > shapes - 1), axis=1)
        if np.any(outside):
            raise ValueError(f"target(s) outside the image: {np.flatnonzero(outside).tolist()}")

        # --- per-axis margins around each anchor, reduced over images ---
        before = anchors                                # (N, 2) above / left
        after = shapes - 1 - anchors                    # (N, 2) below / right

        mBefore = before.min(axis=0)                    # (up, left)
        mAfter = after.min(axis=0)                      # (down, right)

        # centered window: tighter side wins on each axis
        half = np.minimum(mBefore, mAfter)
        hRow, hCol = int(half[0]), int(half[1])

        def _build(back, fwd):
            back = np.asarray(back, dtype=int)
            fwd = np.asarray(fwd, dtype=int)
            dims = back + fwd + 1
            slices = [(slice(r0 - back[0], r0 + fwd[0] + 1),
                    slice(c0 - back[1], c0 + fwd[1] + 1)) for r0, c0 in anchors]
            return {
                "nRows": int(dims[0]),
                "nCols": int(dims[1]),
                "slices": slices,
                "targetIndex": (int(back[0]), int(back[1])),
                "rGeo": float(np.sqrt(back[0] * fwd[0] * back[1] * fwd[1]) ** 0.5)
                        if False else float(np.sqrt(
                            ((back[0] + fwd[0]) / 2.0) * ((back[1] + fwd[1]) / 2.0))),
                "sideEq": float(np.sqrt(dims[0] * dims[1])),
            }

        centered = _build(half, half)
        maximal = _build(mBefore, mAfter)

        maximal.update({
            "up": int(mBefore[0]),
            "down": int(mAfter[0]),
            "left": int(mBefore[1]),
            "right": int(mAfter[1]),
        })

        overlap = {"centered": centered, "max": maximal}

        # --- cut and average the requested window ---
        if window not in overlap:
            raise ValueError("window must be 'centered' or 'max'")
        chosen = overlap[window]

        if chosen["nRows"] < 2 or chosen["nCols"] < 2:
            raise ValueError(
                f"degenerate {window} overlap: {chosen['nRows']}x{chosen['nCols']}")

        stack = np.stack([im[s] for im, s in zip(imgs, chosen["slices"])]
                        ).astype(dtype, copy=False)

        if nan_policy == "propagate":
            mean = stack.mean(axis=0)
        elif nan_policy == "omit":
            valid = np.isfinite(stack)
            count = valid.sum(axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(np.where(valid, stack, np.nan), axis=0)
            mean[count == 0] = np.nan
        else:
            raise ValueError("nan_policy must be 'propagate' or 'omit'")

        return mean, stack, overlap

    def getImCenterCoords(self, im, imData):
        """ The output coordinate is relative to the center of image, so where is target at optimal GSD relative to image center """

        # imdata will be: fore angle sat, starboard angle sat, target location row, target location col
        # im is the matrix
        fore = imData[0]
        star = imData[1]

        row, col = self.getTargetPlacement(im)

        targCenRow = (row - im.shape[0] // 2)
        targCenCol = (col - im.shape[1] // 2)

        return targCenRow, targCenCol

    def getBHRatio(self, imData):

        BHs = []

        for i in range(imData.shape[0]):
            for j in range(i+1, imData.shape[0]):
                imFore1 = imData[i,0]
                imStar1 = imData[i,1]
                imAlt1 = imData[i,2]

                imFore2 = imData[j,0]
                imStar2 = imData[j,1]
                imAlt2 = imData[j,2]

                dBF = np.abs(imAlt1 * np.tan(np.deg2rad(imFore1)) - imAlt2 * np.tan(np.deg2rad(imFore2)))
                dBS = np.abs(imAlt1 * np.tan(np.deg2rad(imStar1)) - imAlt2 * np.tan(np.deg2rad(imStar2)))
                dB = np.sqrt(dBF**2 + dBS**2)
                BH = dB / ((imAlt1 + imAlt2)/2)
                BHs.append(BH)

        BHRatio = np.mean(np.array(BHs))

        return BHRatio

    def getCoverage(self, ims, imData):

        fov = self.sens[0]
        halfFov = np.degrees(fov) / 2

        dataIm = imData[~np.all(imData[:, 0:2] == 0.0, axis=1)]

        imSpan = np.zeros((dataIm.shape[0],2))
        for k in range(dataIm.shape[0]):
            foreAng = np.radians(abs(dataIm[k,0]))
            starAng = np.radians(abs(dataIm[k,1]))

            signFore = np.sign(dataIm[k,0])
            signStar = np.sign(dataIm[k,1])

            ang = np.arctan2(np.tan(-signFore * foreAng), np.tan(-signStar * starAng))
            ang = np.degrees(ang) % 360

            imSpanEnd = (ang + (halfFov)) % 360
            imSpanStart = (ang - (halfFov)) % 360

            imSpan[k,:] = np.array([imSpanStart, imSpanEnd])

        segs = []
        for s, e in np.asarray(imSpan, dtype=float) % 360:
            if e > s:
                segs.append((s, e))
            else:                       # arc wraps past 0
                segs.append((s, 360))
                segs.append((0.0, e))
        segs.sort()
        covered, cur_s, cur_e = 0.0, *segs[0]
        for s, e in segs[1:]:
            if s > cur_e:               # gap between merged blocks
                covered += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        covered += cur_e - cur_s

        return covered / 360

    def getGSDError(self, overlapMean):

        meanGSD = np.mean(overlapMean)

        # Max GSD and Min GSD are calculated from nadir and the max look angle based off altitude
        errGSD = (meanGSD - self.minGSD) / (self.maxGSD - self.minGSD)

        return errGSD

    def getUsablePix(self, im):

        gsdLim = self.pixGSDlimFactor*(self.maxGSD - self.minGSD) + self.minGSD

        numPix = np.count_nonzero(im < gsdLim)

        errPix = 1.0 - (numPix / (im.shape[0] * im.shape[1]))

        return errPix

    def computeMVSCostMap(self, gsdMats, imAngles, sigmaPix, eMax):
        """
        Computes MVS cost map where unusable pixels have been set to np.nan.
        All input matrices must still be the same dimensional shape (e.g., 1000x1000).
        """
        offNadAngles = np.array(imAngles.flatten())
        # 1. 3D Vectorization
        gsdCube = np.stack(gsdMats, axis=0)
        
        # 2. Dynamic View Counting
        # Count how many valid (non-NaN) views exist FOR EACH PIXEL
        validViews = np.sum(~np.isnan(gsdCube), axis=0)
        
        # Create a mask for pixels that don't meet the stereo requirement (< 2 views)
        invalidStereoMask = validViews < 2
        
        # 3. Base Uncertainty (Ignoring NaNs)
        # Convert NaNs to infinity temporarily just to avoid math warnings, 
        # though np.nansum handles NaNs automatically.
        safeGSDCube = np.where(np.isnan(gsdCube), np.inf, np.clip(gsdCube, a_min=1e-6, a_max=None))
        wCube = 1.0 / ((sigmaPix * safeGSDCube) ** 2)
        
        # Use np.nansum to add up the weights, completely ignoring the "cut" NaN pixels
        wSumMat = np.nansum(wCube, axis=0)
        
        # Prevent divide-by-zero where weight_sum is 0 (invalid pixels)
        with np.errstate(divide='ignore'):
            baseErrMap = np.where(wSumMat > 0, np.sqrt(1.0 / wSumMat), np.inf)
        
        # 4. Geometry Penalty (Assumes global pass geometry)
        convergenceDeg = np.max(offNadAngles) - np.min(offNadAngles)
        theta = np.radians(np.clip(abs(convergenceDeg), 1.0, 179.0))
        geometricPenalty = 1.0 / np.sin(theta)
        
        # 5. Redundancy Penalty (Now calculated PER PIXEL)
        alpha = 0.5
        # Pixels with 5 views get a much better score than pixels with only 2 views!
        redundancyPenalty = np.where(
            validViews >= 2, 
            1.0 / (1.0 - np.exp(-alpha * (validViews - 1))), 
            np.inf
        )
        
        # 6. Total Error and Normalization
        totalErrorMap = baseErrMap * geometricPenalty * redundancyPenalty
        costMap = 1.0 - np.exp(-((totalErrorMap / eMax) ** 2))
        
        # Force any pixel with < 2 views to immediately fail with a cost of 1.0
        costMap[invalidStereoMask] = 1.0
        
        return np.clip(costMap, 0.0, 1.0)

    def getTargetPlacement(self, gsd, target_radius=0, edge_margin=40, tolerance=0.05,
                          max_gsd=None, decimate=1, dtype=np.float64,
                          tie_rtol=None, return_info=False):
        """Find where in the image frame to place the target.
    
        Minimises the worst Ground Sampling Distance anywhere on the target
        footprint, subject to keeping the aim point at least `edge_margin` pixels
        from the frame boundary. The margin matters because GSD fields from an
        oblique camera are usually monotone: the unconstrained minimum sits on the
        near edge of the frame, which is the worst place to aim, since any pointing
        error drops the target out of view.
    
        Parameters
        ----------
        gsd : 2D array
            GSD in image coordinates, metres/pixel, lower is better. Non-finite and
            non-positive entries are treated as unusable.
        target_radius : int
            Half-extent of the target footprint in image pixels. 0 treats the
            target as a point.
        edge_margin : int
            Minimum distance from the aim point to the frame edge; should cover
            pointing and geolocation uncertainty, not just target size. The
            effective margin is max(edge_margin, target_radius), so the footprint
            can never hang off the frame.
        tolerance : float
            Defines the near-optimal region reported in `info`: all placements
            whose worst-case GSD is within this fraction of the best.
        max_gsd : float, optional
            Pixels above this are treated as unusable. Useful for cutting off
            horizon-grazing rays that are finite but useless.
        decimate : int
            If >1, search every `decimate`-th pixel, then refine at full
            resolution near the coarse answer. GSD fields are smooth, so 4-8 is
            usually safe and much faster on large frames. Makes `info` region
            statistics approximate.
        return_info : bool
            If True, also return a diagnostics dict.
    
        Returns
        -------
        (row, col), or ((row, col), info) if return_info is True.
        """
        g = np.asarray(gsd)
        if g.ndim != 2:
            raise ValueError(f"expected a 2D GSD array, got shape {g.shape}")
        if target_radius < 0 or edge_margin < 0:
            raise ValueError("target_radius and edge_margin must be non-negative")
    
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("dtype must be np.float32 or np.float64")
    
        r = int(target_radius)
        m = max(int(edge_margin), r)
        h, w = g.shape
        if 2 * m >= min(h, w):
            raise ValueError(f"margin {m} leaves no interior in a {h}x{w} frame")
    
        # Invalid pixels become +inf so that maximum_filter propagates them
        # deterministically; NaN would not, it poisons neighbourhoods according to
        # the filter's comparison order.
        g = g.astype(dtype, copy=True)
        bad = ~np.isfinite(g) | (g <= 0)
        if max_gsd is not None:
            bad |= g > max_gsd
        g[bad] = np.inf
        del bad
    
        rtol = 64 * np.finfo(dtype).eps if tie_rtol is None else float(tie_rtol)
    
        def plateauCenter(cost):
            """Centroid of the best connected tied plateau in `cost`."""
            idx = np.unravel_index(int(np.argmin(cost)), cost.shape)
            best = float(cost[idx])
            if not np.isfinite(best):
                raise ValueError("no valid placement: every candidate pixel is invalid")
            tie = cost <= (best * (1.0 + rtol) if best > 0 else best + rtol)
            # Restricting to one connected component keeps the centroid from
            # landing in the warm gap between two separate basins.
            lab, _ = label(tie)
            comp = lab == lab[idx]
            ys, xs = np.nonzero(comp)
            cy, cx = float(ys.mean()), float(xs.mean())
            # A plateau can be non-convex or have a hole punched in it by an
            # invalid pixel, in which case its centroid is not itself a legal
            # placement. Snap to the nearest pixel that is.
            iy, ix = int(round(cy)), int(round(cx))
            if not (0 <= iy < cost.shape[0] and 0 <= ix < cost.shape[1] and comp[iy, ix]):
                k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
                cy, cx = float(ys[k]), float(xs[k])
            return cy, cx, best
    
        d = max(1, int(decimate))
        work = g if d == 1 else np.ascontiguousarray(g[::d, ::d])
        rw = r if d == 1 else max(0, int(round(r / d)))
        mw = m if d == 1 else max(int(np.ceil(m / d)), rw)
        hw, ww = work.shape
        if 2 * mw >= min(hw, ww):
            raise ValueError(f"decimate={d} leaves no interior after margin")
    
        # Scalar `size` decomposes into two O(n) 1D passes; a disc `footprint`
        # would be O(n*k^2) and crawl on a 38M-pixel frame.
        worst = maximum_filter(work, size=2 * rw + 1, mode="nearest") if rw else work
        free_r, free_c, _ = plateauCenter(worst)
        free_r, free_c = free_r * d, free_c * d
    
        cost = np.full(worst.shape, np.inf, dtype=dtype)
        keep = (slice(mw, hw - mw), slice(mw, ww - mw))
        cost[keep] = worst[keep]
        cbest = float(cost.min())
        row, col, _ = plateauCenter(cost)
        row, col = row * d, col * d
    
        if d > 1:
            # Refine at full resolution around the coarse answer. The window is
            # expanded by r before filtering so the footprint maximum is correct at
            # the window's own edges, then cropped back.
            pad = 2 * d
            lo_r, hi_r = max(m, int(row) - pad), min(h - m, int(row) + pad + 1)
            lo_c, hi_c = max(m, int(col) - pad), min(w - m, int(col) + pad + 1)
            sr, sc = lo_r - r, lo_c - r
            sub = g[sr:hi_r + r, sc:hi_c + r]
            sub_worst = maximum_filter(sub, size=2 * r + 1, mode="nearest") if r else sub
            win = sub_worst[lo_r - sr:lo_r - sr + hi_r - lo_r,
                            lo_c - sc:lo_c - sc + hi_c - lo_c]
            wr, wc, _ = plateauCenter(win)
            row, col = lo_r + wr, lo_c + wc
    
        # Round half to even, so a plateau centred between two pixels does not
        # drift consistently one way across repeated calls.
        irow = min(max(int(np.round(row)), m), h - m - 1)
        icol = min(max(int(np.round(col)), m), w - m - 1)
        if not return_info:
            return irow, icol
    
        ok = cost <= cbest * (1.0 + tolerance)
        ys, xs = np.nonzero(ok)
        info = {
            "subpixel_rowcol": (row, col),
            # Exact full-resolution footprint maximum at the chosen aim point.
            "worst_gsd": float(g[irow - r:irow + r + 1, icol - r:icol + r + 1].max()),
            "center_gsd": float(g[irow, icol]),
            "edge_margin": m,
            "target_radius": r,
            # True if the unconstrained optimum lies outside the margin band, i.e.
            # the margin is what set the answer rather than the GSD field. Expect
            # this on any monotone (oblique) field.
            "boundary_limited": not (m <= free_r <= h - m - 1
                                    and m <= free_c <= w - m - 1),
            "free_rowcol": (free_r, free_c),
            # Bounding box of placements within `tolerance` of optimal. If this is
            # large you have slack to spend on margin rather than sitting at the
            # mathematical optimum with no robustness.
            "region_bbox": (int(ys.min()) * d, int(ys.max()) * d,
                            int(xs.min()) * d, int(xs.max()) * d),
            "region_frac": float(ok.sum()) / float(np.isfinite(cost).sum()),
            "approximate": d > 1,
        }
        return irow, icol, info

    def plotCostMap(self, costMap, imAngles, sigmaPix, eMax, pathCost, cost):

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(costMap, origin='lower', cmap='coolwarm', interpolation='nearest')
        cbar = fig.colorbar(im)
        fmt = ScalarFormatter(useOffset=False, useMathText=False)
        fmt.set_scientific(False)
        cbar.ax.yaxis.set_major_formatter(fmt)

        cbar.set_label('Multi-View Stereo Cost')
        ax.set_title(f'MVS Cost Map {imAngles.shape[0]} images, sigmaPix: {sigmaPix}, eMax: {eMax}')
        per_row = 4
        items = [f"({a:+5.1f}, {b:+5.1f})" for a, b in imAngles]
        rows = [ "   ".join(items[i:i+per_row]) for i in range(0, len(items), per_row) ]
        note = "Angle sets:\n" + "\n".join(rows)
        n_lines = note.count("\n") + 1
        line_h = 0.022                      # ~fraction of figure height per line at fontsize 9
        fig.subplots_adjust(bottom=0.10 + n_lines * line_h)
        fig.text(0.5, 0.02, note, ha='center', va='bottom',
                fontsize=9, family='monospace')

        fileName = f'costmap_{imAngles.shape[0]:02d}_images_{self.alt}_km_{cost}.png'
        fullPath = pathCost / fileName
        plt.savefig(fullPath)
        print(f'costmap_{imAngles.shape[0]}_images_{self.alt}_km_{cost}.png')

    def plotRandomCostMap(self, costMap, imAngles, sigmaPix, eMax, pathCost, cost, randNum):
    
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(costMap, origin='lower', cmap='coolwarm', interpolation='nearest')
        cbar = fig.colorbar(im)
        fmt = ScalarFormatter(useOffset=False, useMathText=False)
        fmt.set_scientific(False)
        cbar.ax.yaxis.set_major_formatter(fmt)

        cbar.set_label('Multi-View Stereo Cost')
        ax.set_title(f'MVS Cost Map {imAngles.shape[0]} images, sigmaPix: {sigmaPix}, eMax: {eMax}')
        per_row = 4
        items = [f"({a:+5.1f}, {b:+5.1f})" for a, b in imAngles]
        rows = [ "   ".join(items[i:i+per_row]) for i in range(0, len(items), per_row) ]
        note = "Angle sets:\n" + "\n".join(rows)
        n_lines = note.count("\n") + 1
        line_h = 0.022                      # ~fraction of figure height per line at fontsize 9
        fig.subplots_adjust(bottom=0.10 + n_lines * line_h)
        fig.text(0.5, 0.02, note, ha='center', va='bottom',
                fontsize=9, family='monospace')

        fileName = f'costmap_{imAngles.shape[0]:02d}_images_{self.alt}_km_{cost}_{randNum}.png'
        fullPath = pathCost / fileName
        plt.savefig(fullPath)
        print(f'costmap_{imAngles.shape[0]}_images_{self.alt}_km_{cost}_{randNum}.png')

