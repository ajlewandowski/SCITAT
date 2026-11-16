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

        # Azimuth coverage: each view "covers" +/- azHalfWidth degrees of azimuth
        # around the target. 22.5 deg means 8 evenly spaced views fill the circle.
        # This is a geometry setting, independent of the sensor's FOV.
        self.azHalfWidth = 22.5
        # Views closer to nadir than this have no meaningful azimuth and are skipped
        self.minAzOffNadir = 2.0

        # Where the target sits in each image:
        #   "center"  - the satellite points its boresight at the target, so the target
        #               is the center pixel of every image (what an agile satellite does)
        #   "bestGSD" - legacy: the target is moved to the lowest-GSD part of the frame
        #               (getTargetPlacement), which a real pointing system would not do
        self.targetMode = "center"
        # Area of interest (AOI) around the target, as a multiple of the nadir half-swath
        # (1.0 = one nadir scene centered on the target). Used only when targetMode is
        # "center", to put every image on the same ground grid.
        self.aoiScale = 1.0

        # ---- Earth model -----------------------------------------------------------
        # True: lines of sight, incidence angles, B/H, footprints and the ground grid
        # use a spherical Earth (needed above ~30 deg off-nadir: at 48 deg from 620 km
        # the incidence angle at the ground is ~55 deg). False: flat-Earth formulas.
        self.sphericalEarth = True

        # ---- Overlap requirement (a hard constraint, not a weighted term) ----------
        # Overlap = share of the target area (the AOI above) that EVERY image is
        # guaranteed to cover, allowing each image to miss the target by up to
        # pointingErrDeg on each axis. With target-centered pointing and no pointing
        # error this is 100 % at any off-nadir angle; it drops with pointing error or
        # when the AOI is larger than a frame (aoiScale > 1).
        # Sets below minOverlap are infeasible.
        self.minOverlap = 0.75
        self.pointingErrDeg = 0.0

        # ---- Agility limit (a hard constraint) -------------------------------------
        # Largest true off-nadir angle the satellite can point to (deg). Defaults to
        # the largest angle in angVec; set per sensor (e.g. 48 for Pleiades Neo).
        self.maxOffNadirDeg = float(np.max(np.abs(angVec)))

        # Infeasible sets get getCostFunction = infeasibleCost + how far they miss
        # (overlap shortfall + relative agility excess), so optimizers can recover.
        self.infeasibleCost = 1.0

        # ---- Image usability grade ------------------------------------------------
        # A pixel is usable if its GSD <= the GSD limit:
        #   gsdLimitM (metres) if set - a mission requirement, e.g. 0.6 for 60 cm
        #   otherwise gsdLimitFactor x nadir GSD (self.minGSD)
        # Each image's grade is its usable pixel share; unusable pixels are dropped
        # (set to NaN) before the GSD and MVS error-map terms are computed.
        self.gsdLimitM = None
        self.gsdLimitFactor = 1.25

    # ---- Geometry helpers ------------------------------------------------------------
    # Pointing angles (along, cross) are measured AT THE SATELLITE. On a spherical
    # Earth the angle at the GROUND is larger: incidence = asin(k sin(offNadir)),
    # k = (R + h) / R. All helpers take altKm; altKm=None means flat Earth.
    EARTH_R = 6378137.0

    @staticmethod
    def pointingVectors(angles):
        """Unit boresight vectors in the SATELLITE frame (x along, y cross, z nadir).
        Exact at the satellite for any Earth model - use for slew angles."""
        a = np.radians(np.asarray(angles, dtype=float).reshape(-1, 2))
        v = np.column_stack([np.tan(a[:, 0]), np.tan(a[:, 1]), np.ones(len(a))])
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    @staticmethod
    def offNadir(angles):
        """True off-nadir angle (deg) of each (along, cross) pointing, at the satellite."""
        a = np.radians(np.asarray(angles, dtype=float).reshape(-1, 2))
        return np.degrees(np.arctan(np.hypot(np.tan(a[:, 0]), np.tan(a[:, 1]))))

    @staticmethod
    def incidenceAngle(offNadirDeg, altKm=None):
        """Angle (deg) between the line of sight and the local vertical AT THE TARGET.
        Equals the off-nadir angle on a flat Earth (altKm=None)."""
        th = np.radians(np.asarray(offNadirDeg, dtype=float))
        if altKm is None:
            return np.degrees(th)
        k = (imageGeometry.EARTH_R + np.asarray(altKm, float) * 1e3) / imageGeometry.EARTH_R
        return np.degrees(np.arcsin(np.clip(k * np.sin(th), -1.0, 1.0)))

    @staticmethod
    def lookVectors(angles, altKm=None):
        """
        Unit line-of-sight vectors (satellite -> target) in the TARGET's local frame:
        x = along-track, y = cross-track, z = down. The horizontal direction is the
        pointing azimuth; the angle from vertical is the incidence angle (spherical
        Earth when altKm is given, otherwise flat Earth where it equals off-nadir).
        altKm may be a scalar or one value per pointing.
        """
        angles = np.asarray(angles, dtype=float).reshape(-1, 2)
        a = np.radians(angles)
        az = np.arctan2(np.tan(a[:, 1]), np.tan(a[:, 0]))
        inc = np.radians(imageGeometry.incidenceAngle(imageGeometry.offNadir(angles), altKm))
        return np.column_stack([np.sin(inc) * np.cos(az), np.sin(inc) * np.sin(az), np.cos(inc)])

    @staticmethod
    def groundOffset(angRad, altM, spherical=True):
        """Signed ground distance (m) from the sub-satellite point to where a ray at
        angle angRad (one axis) meets the ground: R * central angle, or h tan(ang)."""
        if not spherical:
            return altM * np.tan(angRad)
        R = imageGeometry.EARTH_R
        k = (R + altM) / R
        return R * (np.arcsin(np.clip(k * np.sin(angRad), -1.0, 1.0)) - angRad)

    @staticmethod
    def groundToAngle(distM, altM, spherical=True):
        """Inverse of groundOffset: the look angle (rad) that hits ground distance distM."""
        if not spherical:
            return np.arctan(distM / altM)
        R = imageGeometry.EARTH_R
        lam = distM / R
        return np.arctan2(np.sin(lam), (R + altM) / R - np.cos(lam))

    def _altKm(self, n, imData=None):
        """Altitudes (km) for spherical formulas, or None for flat Earth."""
        if not self.sphericalEarth:
            return None
        if imData is not None and np.asarray(imData).shape[1] >= 3:
            return np.asarray(imData, float)[:, 2]
        return np.full(n, float(self.alt))

    def getConvergenceAngles(self, imAngles, imData=None):
        """Convergence angle (deg) at the target between every pair of lines of sight."""
        v = self.lookVectors(imAngles, self._altKm(len(imAngles), imData))
        cosAng = np.clip(v @ v.T, -1.0, 1.0)
        iu = np.triu_indices(len(v), 1)
        return np.degrees(np.arccos(cosAng[iu]))

    # Order and weights of the five cost terms. Change the weights here (they should
    # sum to 1) and every study that calls getCostFunction picks them up.
    # errOverlap is still reported (1 - worst image's overlap share) but has weight 0:
    # overlap is enforced as a constraint (minOverlap) instead.
    COST_TERMS = ("errOverlap", "errGSD", "errBH", "errCov", "errCostMap", "errUsable")
    costWeights = {"errOverlap": 0.0, "errGSD": 0.2, "errBH": 0.2, "errCov": 0.2,
                   "errCostMap": 0.2, "errUsable": 0.2}

    def getCostComponents(self, ims, imData, imAngles):
        """
        Returns the unweighted error terms (each roughly 0..1) plus:
            feasible      overlapOK and agilityOK
            overlapOK     overlapMin >= minOverlap
            overlapMin    share of the target area every image covers (with pointing error)
            overlapShares each image's own coverage of the target area
            agilityOK     every image's true off-nadir angle <= maxOffNadirDeg
            maxOffNadir   the largest true off-nadir angle in the set (deg)
            penalty       how far an infeasible set misses (0 if feasible)
            usable        each image's usable-pixel share (GSD <= gsdLimit())
        getCostFunction is the weighted sum of the COST_TERMS.
        """
        overlapMean, overlapSlices, overlap = self.getOverlapRadius(ims, imData)

        # Overlap: target-area coverage (target-centered) or the legacy pixel window
        if "overlapSet" in overlap:
            shares = np.asarray(overlap["overlapShares"], float)
            overlapMin = float(overlap["overlapSet"])
        else:
            overlapRad = overlap["centered"]["rGeo"]
            shares = np.full(len(ims), overlapRad / (ims[0].shape[0] // 2))
            overlapMin = float(shares.min())
        errOverlap = 1.0 - overlapMin
        overlapOK = overlapMin >= self.minOverlap - 1e-9      # tolerance for round-off

        # Agility: no image may point further off-nadir than the satellite can
        maxOff = float(np.max(self.offNadir(imAngles)))
        agilityOK = maxOff <= self.maxOffNadirDeg + 1e-9
        feasible = overlapOK and agilityOK
        penalty = (max(0.0, self.minOverlap - overlapMin)
                   + max(0.0, maxOff - self.maxOffNadirDeg) / max(self.maxOffNadirDeg, 1e-9))

        # Usability: grade every whole image, then keep only usable pixels in the stack
        usable = np.array([self.getUsableFraction(im) for im in ims])
        errUsable = 1.0 - float(usable.mean())
        lim = self.gsdLimit()
        overlapSlices = [np.where(s <= lim, s, np.nan) for s in overlapSlices]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            overlapMean = np.nanmean(np.stack(overlapSlices), axis=0)

        # GSD of the usable pixels only; if nothing is usable, the worst GSD error
        errGSD = self.getGSDError(overlapMean) if np.isfinite(overlapMean).any() else 1.0

        bhRatio = self.getBHRatio(imData)
        errbhRatio = (abs(self.targetBH - bhRatio) / self.targetBH)

        errCov = 1.0 - self.getCoverage(ims, imData)

        costMap = self.computeMVSCostMap(overlapSlices, imAngles, self.sigmaPix, self.eMax)
        errCostMap = np.mean(costMap)

        return {"errOverlap": float(errOverlap), "errGSD": float(errGSD),
                "errBH": float(errbhRatio), "errCov": float(errCov),
                "errCostMap": float(errCostMap), "errUsable": float(errUsable),
                "feasible": bool(feasible), "overlapOK": bool(overlapOK),
                "agilityOK": bool(agilityOK), "overlapMin": overlapMin,
                "overlapShares": shares.tolist(), "maxOffNadir": maxOff,
                "penalty": float(penalty), "usable": usable.tolist()}

    def getCostFunction(self, ims, imData, imAngles, weights=None):
        """Weighted sum of the cost terms (weights default to self.costWeights).
        A set that fails the overlap or agility requirement returns
            infeasibleCost + penalty   (overlap shortfall + relative agility excess)
        which is worse than any feasible set but still tells an optimizer which
        infeasible sets are closer to feasible."""
        comps = self.getCostComponents(ims, imData, imAngles)
        if not comps["feasible"]:
            return self.infeasibleCost + comps["penalty"]
        w = self.costWeights if weights is None else weights
        return sum(w[k] * comps[k] for k in self.COST_TERMS)

    def gsdLimit(self):
        """GSD (m) above which a pixel is unusable: gsdLimitM if set, else
        gsdLimitFactor x nadir GSD."""
        if self.gsdLimitM is not None:
            return float(self.gsdLimitM)
        return self.gsdLimitFactor * self.minGSD

    def getUsableFraction(self, im):
        """Usability grade of one image: share of its pixels with GSD <= gsdLimit()."""
        return float(np.mean(im <= self.gsdLimit()))

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

        # Target-centered pointing: register the images on the ground instead of by pixel
        if self.targetMode == "center":
            return self.getGroundOverlap(imgs, imData, dtype=dtype)

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

    def getGroundOverlap(self, ims, imData, dtype=np.float64):
        """
        Put every image on one ground grid centered on the target and compare the
        images point by point on the ground.

        Each image is pointed at the target, so the target is its center pixel. The
        AOI is a square of half-width aoiScale * h * tan(FOV/2) (one nadir scene for
        aoiScale = 1), sampled with the same number of points as the images. For every
        image, each ground point is mapped back to the pixel that sees it:

            look angle to ground point x (along) = groundToAngle(groundOffset(fore) + x)
            pixel row = (look angle - (fore - FOV/2)) / (FOV / nRows)

        and the same for cross-track / columns. Spherical Earth when sphericalEarth is
        True (per-axis, the same separable model as the GSD vectors). Ground points outside an image's
        footprint get NaN, which computeMVSCostMap already treats as "not seen".

        Returns (mean, stack, overlap) like getOverlapRadius:
            mean    : (nRows, nCols) GSD averaged over the views that see each point
            stack   : (N, nRows, nCols) each view's GSD on the common ground grid
            overlap : dict with overlapSet and overlapShares (getAOIOverlap),
                      coverageFraction (AOI share seen by every image on the grid),
                      fracTwoViews (share seen by at least two), aoiHalfWidth_m,
                      and a "centered" entry kept for older callers.
        """
        fov = float(self.sens[0])
        nR, nC = ims[0].shape[:2]
        sph = self.sphericalEarth
        aoiHalf = self.aoiHalfWidth()
        # cell-center samples, so a nadir image covers the nadir-sized AOI exactly
        u = ((np.arange(nR) + 0.5) / nR * 2 - 1) * aoiHalf    # along-track ground offsets (m)
        v = ((np.arange(nC) + 0.5) / nC * 2 - 1) * aoiHalf    # cross-track ground offsets (m)

        stack = np.full((len(ims), nR, nC), np.nan, dtype=dtype)
        for k, im in enumerate(ims):
            fore, star = np.radians(imData[k, 0]), np.radians(imData[k, 1])
            h = float(imData[k, 2]) * 1e3
            r, c = im.shape[:2]
            # ground point -> look angle (per axis), measured from the sub-satellite point
            angR = self.groundToAngle(self.groundOffset(fore, h, sph) + u, h, sph)
            angC = self.groundToAngle(self.groundOffset(star, h, sph) + v, h, sph)
            rows = np.floor((angR - (fore - fov / 2)) / (fov / r)).astype(int)
            cols = np.floor((angC - (star - fov / 2)) / (fov / c)).astype(int)
            okR = (rows >= 0) & (rows < r)
            okC = (cols >= 0) & (cols < c)
            g = im[np.ix_(np.clip(rows, 0, r - 1), np.clip(cols, 0, c - 1))].astype(dtype)
            g[~okR, :] = np.nan
            g[:, ~okC] = np.nan
            stack[k] = g

        nSeen = np.sum(np.isfinite(stack), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(stack, axis=0)
        coverage = float(np.mean(nSeen == len(ims)))
        overlapSet, shares = self.getAOIOverlap(imData)

        overlap = {
            "mode": "center",
            "overlapSet": overlapSet,
            "overlapShares": shares,
            "coverageFraction": coverage,
            "fracTwoViews": float(np.mean(nSeen >= 2)),
            "aoiHalfWidth_m": aoiHalf,
            # equivalent-radius entry so code written for the pixel method still runs
            "centered": {"nRows": nR, "nCols": nC, "targetIndex": (nR // 2, nC // 2),
                         "rGeo": float((nR // 2) * np.sqrt(coverage)),
                         "slices": None, "sideEq": float(np.sqrt(nR * nC * coverage))},
        }
        return mean, list(stack), overlap

    def aoiHalfWidth(self):
        """Half-width (m) of the target area: aoiScale x the nadir half-swath on the ground."""
        return self.aoiScale * abs(self.groundOffset(float(self.sens[0]) / 2, float(self.sens[5]),
                                                     self.sphericalEarth))

    def getAOIOverlap(self, imData):
        """
        Overlap of an image set measured on the TARGET AREA (AOI).

        Each image is pointed at the target. On each axis, its footprint on the ground,
        measured from the target, runs from
            groundOffset(ang - FOV/2) - groundOffset(ang)  to  groundOffset(ang + FOV/2) - groundOffset(ang)
        (spherical Earth if sphericalEarth). With a pointing error of up to e
        (pointingErrDeg) the boresight lands anywhere in ang +/- e, so the ground the
        image covers for ANY such error is the footprint shrunk on each side:
            [ g(ang + e - FOV/2) - g(ang),  g(ang - e + FOV/2) - g(ang) ]

        Returns
            overlapSet : share of the AOI (square of half-width aoiHalfWidth()) that
                         EVERY image is guaranteed to cover - the number the 75 %
                         requirement applies to
            shares     : each image's own guaranteed coverage of the AOI

        An oblique image covers MORE ground than a nadir one, so with no pointing error
        and aoiScale <= 1 the overlap is 100 % at any off-nadir angle: off-nadir
        viewing alone does not reduce overlap. Overlap drops when the images may miss
        the target (pointingErrDeg) or the AOI is larger than one frame (aoiScale > 1).
        """
        fov = float(self.sens[0])
        e = np.radians(self.pointingErrDeg)
        sph = self.sphericalEarth
        A = self.aoiHalfWidth()
        ang = np.radians(np.asarray(imData[:, 0:2], float))          # (N, 2) along, cross
        h = np.asarray(imData[:, 2], float)[:, None] * 1e3           # (N, 1) metres
        g0 = self.groundOffset(ang, h, sph)
        gLo = self.groundOffset(ang + e - fov / 2, h, sph) - g0       # guaranteed footprint
        gHi = self.groundOffset(ang - e + fov / 2, h, sph) - g0
        lo = np.maximum(gLo, -A)                                      # clipped to the AOI
        hi = np.minimum(gHi, A)
        perImage = np.prod(np.clip(hi - lo, 0, None) / (2 * A), axis=1)
        common = np.clip(np.minimum(gHi.min(0), A) - np.maximum(gLo.max(0), -A), 0, None)
        return float(np.prod(common / (2 * A))), perImage.tolist()

    def getImCenterCoords(self, im, imData):
        """ Target position relative to the image center pixel (row, col).
            targetMode "center": the boresight is on the target, so (0, 0).
            targetMode "bestGSD": legacy, where the GSD is lowest in the frame. """

        if self.targetMode == "center":
            return 0, 0

        # imdata will be: fore angle sat, starboard angle sat, target location row, target location col
        # im is the matrix
        fore = imData[0]
        star = imData[1]

        row, col = self.getTargetPlacement(im, edge_margin = 40//8)

        targCenRow = (row - im.shape[0] // 2)
        targCenCol = (col - im.shape[1] // 2)

        return targCenRow, targCenCol

    def getBHRatio(self, imData):
        """
        Mean base-to-height ratio over all pairs of views.

        For each view, the line of sight from the target reaches unit height at a
        horizontal offset tan(incidence) * (cos az, sin az). A pair's B/H is the distance
        between those two offsets. On a flat Earth tan(incidence) * cos az = tan(along),
        which is the original  hypot(tan(fore1) - tan(fore2), tan(star1) - tan(star2)).
        With sphericalEarth the incidence angle at the target is used, which is larger
        than the off-nadir angle (e.g. ~55 deg vs 48 deg from 620 km).
        """
        angles = np.asarray(imData[:, 0:2], float)
        v = self.lookVectors(angles, self._altKm(len(angles), imData))
        p = v[:, :2] / v[:, 2:3]                       # offsets at unit height
        i, j = np.triu_indices(len(p), 1)
        return float(np.mean(np.linalg.norm(p[i] - p[j], axis=1)))

    def getCoverage(self, ims, imData, return_spans=False):
        """
        Fraction of the 360 deg of azimuth around the target that is seen by the views.
        Each view (off-nadir >= minAzOffNadir) covers +/- azHalfWidth around the
        azimuth of its line of sight as seen from the target.
        """
        angles = np.asarray(imData[:, 0:2], dtype=float)
        dataIm = angles[self.offNadir(angles) >= self.minAzOffNadir]
        if len(dataIm) == 0:
            return (0.0, np.zeros((0, 2))) if return_spans else 0.0

        # Azimuth of the satellite as seen from the target (target -> satellite
        # is the reverse of the line of sight), measured from +along-track toward
        # +cross-track.
        v = self.lookVectors(dataIm)
        az = np.degrees(np.arctan2(-v[:, 1], -v[:, 0])) % 360

        imSpan = np.column_stack([(az - self.azHalfWidth) % 360,
                                  (az + self.azHalfWidth) % 360])

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

        if return_spans:
            return covered / 360, imSpan
        return covered / 360

    def getGSDError(self, overlapMean):

        meanGSD = np.nanmean(overlapMean)     # NaN = ground point no image sees

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
        # Largest convergence angle between any two lines of sight (true 3D angle,
        # not the range of the raw along/cross numbers)
        convergenceDeg = np.max(self.getConvergenceAngles(imAngles))
        theta = np.radians(np.clip(convergenceDeg, 1.0, 179.0))
        geometricPenalty = 1.0 / np.sin(theta)
        
        # 5. Redundancy Penalty (Now calculated PER PIXEL)
        alpha = 0.5
        # Pixels with 5 views get a much better score than pixels with only 2 views!
        # (np.maximum keeps the unused branch finite: np.where evaluates both sides)
        redundancyPenalty = np.where(
            validViews >= 2,
            1.0 / (1.0 - np.exp(-alpha * np.maximum(validViews - 1, 1))),
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

