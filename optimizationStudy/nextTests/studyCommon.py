"""
studyCommon.py
==============
Shared pieces for the nine "next step" geometry studies in this folder.

Everything that more than one test needs lives here so each test file can stay
focused on its own question:

    SENSORS            - sensor presets (SkySat, Landsat-9, WorldView-3-like)
    CostEvaluator      - wraps imageGridding + imageGeometry, caches GSD images,
                         returns every cost term, the weighted total and feasibility
    geometry helpers   - (off-nadir, azimuth) <-> (along, cross) conversions,
                         convergence / bisector-elevation / asymmetry angles,
                         a standard set of per-set "features"
    random sets        - the same sampling the original random study used
    anneal()           - a simulated-annealing optimizer over image sets
    plotting helpers   - one consistent look for every figure
    commonArgs()       - --quick / --out / --sensor / --seed for every script

Angle conventions (same as the rest of SCITAT):
    along  (fore)      + = pointing forward of nadir along the velocity vector
    cross  (starboard) + = pointing to the right of the ground track
    off-nadir angle    = true angle between boresight and nadir:
                         atan( sqrt( tan(along)^2 + tan(cross)^2 ) )
    pointing azimuth   = direction the boresight is tilted, measured from +along
                         toward +cross (0 = straight forward, 90 = starboard)

    tan(along) = tan(offNadir) * cos(azimuth)
    tan(cross) = tan(offNadir) * sin(azimuth)

Speed: the GSD images are decimated so they are ~320 x 320 pixels
(CostEvaluator(target_px=...)). One cost evaluation then takes ~0.03 s instead of
~1.8 s at full resolution, and the total cost changes by ~1% (test 1 checks this
for your setup). Use target_px=None for full resolution.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # write PNGs, never open windows
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# The repo root holds imageGeometryClass.py and imageGriddingClass.py
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from imageGriddingClass import imageGridding      # noqa: E402
from imageGeometryClass import imageGeometry      # noqa: E402

COST_TERMS = imageGeometry.COST_TERMS
TERM_LABELS = {"errOverlap": "Overlap", "errGSD": "GSD", "errBH": "B/H ratio",
               "errCov": "Azimuth coverage", "errCostMap": "MVS error map",
               "errUsable": "Usable pixels"}


# =====================================================================================
# Sensors
# =====================================================================================
# fov_deg   : full field of view of the detector line (deg)
# focal_m   : focal length (m) - carried through to the sens array, not used by the cost
# pix       : detector pixels across the swath
# swath_m   : nadir swath width (m)
# alt_km    : nominal altitude (km)
# maxOffNadir : largest true off-nadir angle the tests search and allow (deg) - the
#             agility limit. 30 deg for the presets that reproduce the original studies;
#             override with --max-off.
SENSORS = {
    # Matches randomImageGroupStudy.py
    "SkySat": dict(fov_deg=0.7878045564, focal_m=3.6, pix=2560, swath_m=5500, alt_km=400,
                   maxOffNadir=30.0),
    # Matches ringedImageStudy.py
    "Landsat9": dict(fov_deg=15.0, focal_m=0.886, pix=6196, swath_m=185000, alt_km=705,
                     maxOffNadir=30.0),
    # Approximate public numbers: 0.31 m GSD, 13.1 km swath, 617 km altitude
    "WorldView3": dict(fov_deg=float(np.degrees(2 * np.arctan(6550 / 617000))),
                       focal_m=13.3, pix=42258, swath_m=13100, alt_km=617, maxOffNadir=30.0),
    # Airbus Pleiades Neo: 620 km sun-synchronous, 0.3 m panchromatic GSD and 14 km swath
    # at nadir, viewing angle +/-52 deg (agility); 48 deg used here as the working limit.
    # Focal length is not used by the cost, so it is left as NaN.
    "PleiadesNeo": dict(fov_deg=float(np.degrees(2 * np.arctan(7000 / 620000))),
                        focal_m=np.nan, pix=46667, swath_m=14000, alt_km=620, maxOffNadir=48.0),
}


def sensArray(sensor, alt_km=None):
    """Build the 6-element sens array SCITAT classes expect: FOV, focal, pix, GSD, swath, alt(m)."""
    s = SENSORS[sensor]
    alt = s["alt_km"] if alt_km is None else alt_km
    # Swath and GSD scale with altitude for a fixed FOV
    swath = s["swath_m"] * alt / s["alt_km"]
    return np.array([np.deg2rad(s["fov_deg"]), s["focal_m"], s["pix"],
                     swath / s["pix"], swath, alt * 1e3])


# =====================================================================================
# Cost evaluation
# =====================================================================================
class CostEvaluator:
    """
    Evaluate the imageGeometry cost for a set of pointing angles.

    ev = CostEvaluator("SkySat")
    r  = ev.evaluate([[0, 0], [20, 0], [-20, 0]])
    r["cost"], r["errBH"], ...

    Parameters
    ----------
    sensor       : key of SENSORS
    alt_km       : orbit altitude used to build the GSD images (default: sensor nominal)
    sigmaPix     : matching precision in pixels (imageGeometry sigmaPix)
    eMax         : error scale (m) in the MVS cost map (imageGeometry eMax)
    maxOffNadir  : agility limit (true off-nadir, deg) AND the GSD normalization range
                   [nadir, maxOffNadir]; default = the sensor's maxOffNadir
    target_px    : decimate images to about this many pixels per side (None = full res)
    norm_alt_km  : altitude used for the GSD normalization (minGSD/maxGSD). Keep this
                   fixed when comparing altitudes so higher orbits are penalized for
                   their coarser GSD instead of being re-normalized away.
    targetMode   : "center" (boresight on the target, default) or "bestGSD" (legacy
                   placement in the lowest-GSD part of the frame)
    aoiScale     : area of interest half-width as a multiple of the nadir half-swath
                   (only used with targetMode="center")
    minOverlap   : overlap requirement (worst image's share of footprint in the common area)
    pointingErrDeg : pointing error allowed for in the overlap requirement (deg)
    gsdLimitFactor : a pixel is usable if GSD <= this x nadir GSD
    gsdLimitM    : absolute GSD requirement (m); overrides gsdLimitFactor when given
    sphericalEarth : spherical (default) or flat-Earth geometry
    """

    def __init__(self, sensor="SkySat", alt_km=None, sigmaPix=0.5, eMax=25.0,
                 maxOffNadir=None, target_px=320, norm_alt_km=None, targetMode="center",
                 aoiScale=1.0, minOverlap=0.75, pointingErrDeg=0.0, gsdLimitFactor=1.25,
                 gsdLimitM=None, sphericalEarth=True):
        self.sensor = sensor
        self.alt = SENSORS[sensor]["alt_km"] if alt_km is None else alt_km
        self.sens = sensArray(sensor, self.alt)
        pix = int(self.sens[2])
        self.step = 1 if target_px is None else max(1, pix // target_px)
        maxOffNadir = SENSORS[sensor]["maxOffNadir"] if maxOffNadir is None else float(maxOffNadir)
        self.sigmaPix, self.eMax, self.maxOffNadir = sigmaPix, eMax, maxOffNadir

        vec = np.zeros(10)
        self.gridded = imageGridding(sensor, vec, vec, vec, self.sens, vec)
        normAlt = self.alt if norm_alt_km is None else norm_alt_km
        normGrid = imageGridding(sensor, vec, vec, vec, sensArray(sensor, normAlt), vec)
        self.minGSD, self.maxGSD = normGrid.getColorbarMinMax(np.array([0, maxOffNadir]), normAlt)
        self.geo = imageGeometry(0, vec, vec, self.sens, vec, np.array([0, maxOffNadir]),
                                 self.minGSD, self.maxGSD, self.alt, sigmaPix, eMax)
        self.geo.targetMode = targetMode
        self.geo.aoiScale = aoiScale
        self.geo.minOverlap = minOverlap
        self.geo.pointingErrDeg = pointingErrDeg
        self.geo.gsdLimitFactor = gsdLimitFactor
        self.geo.gsdLimitM = gsdLimitM
        self.geo.maxOffNadirDeg = maxOffNadir
        self.geo.sphericalEarth = sphericalEarth

        self._cache = {}            # (alt, along, cross) -> (gsdFore, gsdStar) decimated
        self.nEvals = 0
        self.nFailed = 0

    # ---- images -------------------------------------------------------------------
    def image(self, along, cross, alt=None):
        """GSD matrix for one pointing. Only the two 1-D GSD vectors are cached
        (a few kB each); the outer product is rebuilt on demand (~0.1 ms)."""
        alt = self.alt if alt is None else alt
        key = (round(float(alt), 1), round(float(along), 3), round(float(cross), 3))
        if key not in self._cache:
            g = self.gridded
            spherical = g.useSphericalModel(key[0], key[1], key[2])
            gF, gS, _, _ = g.getGSDVectors(key[0], key[1], key[2], spherical)
            self._cache[key] = (gF[::self.step].copy(), gS[::self.step].copy())
        gF, gS = self._cache[key]
        return np.outer(gF, gS)

    # ---- cost ---------------------------------------------------------------------
    def evaluate(self, angles, alt=None, weights=None):
        """
        angles : (N, 2) along/cross pointing in degrees (N >= 2)
        alt    : None (nominal), a scalar, or a length-N array of altitudes in km
        Returns a dict with
            the COST_TERMS          each unweighted term
            costUnconstrained       weighted total, ignoring the overlap requirement
            cost                    the same total, but NaN if the set is infeasible
            feasible                worst image's overlap share >= geo.minOverlap
            overlapMin              share of the target area every image covers
            usableMean              mean usable-pixel share
            maxOffNadir, penalty    largest off-nadir angle; how far an infeasible set misses
            computed                the cost function ran (False = degenerate geometry)
            ok                      computed AND feasible - the sets to rank and keep
        """
        angles = np.asarray(angles, dtype=float).reshape(-1, 2)
        n = len(angles)
        alts = (np.full(n, float(self.alt)) if alt is None
                else np.broadcast_to(np.asarray(alt, float), (n,)).copy())
        self.nEvals += 1
        out = {k: np.nan for k in COST_TERMS}
        out.update(cost=np.nan, costUnconstrained=np.nan, feasible=False, overlapMin=np.nan,
                   usableMean=np.nan, maxOffNadir=np.nan, penalty=np.nan, computed=False, ok=False)
        if n < 2:
            return out
        try:
            ims = [self.image(a, c, h) for (a, c), h in zip(angles, alts)]
            imData = np.column_stack([angles, alts])
            comps = self.geo.getCostComponents(ims, imData, angles)
        except (ValueError, IndexError, FloatingPointError):
            self.nFailed += 1
            return out
        w = self.geo.costWeights if weights is None else weights
        out.update({k: comps[k] for k in COST_TERMS})
        total = sum(w[k] * comps[k] for k in COST_TERMS)
        out.update(costUnconstrained=total, feasible=comps["feasible"], overlapMin=comps["overlapMin"],
                   usableMean=float(np.mean(comps["usable"])), maxOffNadir=comps["maxOffNadir"],
                   penalty=comps["penalty"], computed=bool(np.isfinite(total)))
        out["ok"] = out["computed"] and out["feasible"]
        out["cost"] = total if out["ok"] else np.nan
        return out

    def cost(self, angles, alt=None, weights=None, failCost=1.0, infeasible="penalty"):
        """
        Just the total, for optimizers.
          feasible set      -> its cost
          infeasible set    -> infeasible="penalty": geo.infeasibleCost + penalty
                               (overlap shortfall + relative agility excess)
                               (worse than any feasible set, but graded so an optimizer
                               can climb back toward feasibility)
                               infeasible="fail": failCost
          could not compute -> failCost
        Use failCost=np.nan, infeasible="fail" to EXCLUDE bad sets from statistics.
        """
        r = self.evaluate(angles, alt, weights)
        if r["ok"]:
            return r["cost"]
        if r["computed"] and infeasible == "penalty":
            return self.geo.infeasibleCost + r["penalty"]
        return failCost


def weightedCost(components, weights):
    """Recombine stored components with new weights (no re-evaluation needed).
    components: dict of arrays or a structured table with the COST_TERMS columns."""
    return sum(weights[k] * np.asarray(components[k]) for k in COST_TERMS)


# =====================================================================================
# Geometry helpers
# =====================================================================================
def polarToAngles(offNadir_deg, azimuth_deg):
    """(off-nadir, pointing azimuth) -> (along, cross) in degrees. Works on arrays."""
    t = np.tan(np.radians(offNadir_deg))
    az = np.radians(azimuth_deg)
    along = np.degrees(np.arctan(t * np.cos(az)))
    cross = np.degrees(np.arctan(t * np.sin(az)))
    return np.column_stack([np.atleast_1d(along), np.atleast_1d(cross)])


def anglesToPolar(angles):
    """(along, cross) -> (off-nadir, pointing azimuth in [0, 360)) in degrees."""
    a = np.radians(np.asarray(angles, float).reshape(-1, 2))
    ta, tc = np.tan(a[:, 0]), np.tan(a[:, 1])
    off = np.degrees(np.arctan(np.hypot(ta, tc)))
    az = np.degrees(np.arctan2(tc, ta)) % 360
    return off, az


def lookVectors(angles, altKm=None):
    """Unit line-of-sight vectors (satellite -> ground) in the target's frame: x along,
    y cross, z down. Spherical Earth when altKm is given (incidence > off-nadir)."""
    return imageGeometry.lookVectors(np.asarray(angles, float).reshape(-1, 2), altKm)


def pointingVectors(angles):
    """Unit boresight vectors in the satellite frame - use for slew angles."""
    return imageGeometry.pointingVectors(np.asarray(angles, float).reshape(-1, 2))


def incidenceAngle(offNadirDeg, altKm=None):
    """Angle from the local vertical at the target (deg); = off-nadir on a flat Earth."""
    return imageGeometry.incidenceAngle(offNadirDeg, altKm)


def pairGeometry(angles, altKm=None):
    """
    For every pair of views returns (convergence, bisectorElevation, asymmetry) in deg.

    convergence        : angle between the two lines of sight
    bisector elevation : elevation above the ground plane of the bisector of the two
                         target->satellite rays (90 = the pair is centered on nadir)
    asymmetry          : angle between that bisector and the local vertical projected
                         into the epipolar plane (0 = the two views lean equally)
    These are the standard satellite-stereo pair descriptors.
    altKm: spherical Earth (angles measured at the target) when given.
    """
    u = -lookVectors(angles, altKm)          # target -> satellite rays
    up = np.array([0.0, 0.0, -1.0])          # z is down in this frame
    i, j = np.triu_indices(len(u), 1)
    u1, u2 = u[i], u[j]
    conv = np.degrees(np.arccos(np.clip(np.sum(u1 * u2, 1), -1, 1)))
    b = u1 + u2
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    bie = np.degrees(np.arcsin(np.clip(b @ up, -1, 1)))
    nrm = np.cross(u1, u2)
    nn = np.linalg.norm(nrm, axis=1, keepdims=True)
    asym = np.zeros(len(b))
    good = nn[:, 0] > 1e-9
    if np.any(good):
        nrm = nrm[good] / nn[good]
        p = up - (nrm @ up)[:, None] * nrm   # vertical projected into the epipolar plane
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        asym[good] = np.degrees(np.arccos(np.clip(np.sum(p * b[good], 1), -1, 1)))
    return conv, bie, asym


def setFeatures(angles, altKm=None, minAzOffNadir=2.0, goodConv=(10.0, 40.0)):
    """
    Order-independent numbers that describe one image set. Used by tests 2, 5 and 6.
    altKm: spherical Earth when given (pass the evaluator's ev.alt).
    goodConv: convergence band usually reported as matchable for satellite MVS.
    """
    angles = np.asarray(angles, float).reshape(-1, 2)
    off, az = anglesToPolar(angles)
    conv, bie, asym = pairGeometry(angles, altKm)
    ta, tc = np.tan(np.radians(angles[:, 0])), np.tan(np.radians(angles[:, 1]))

    # Azimuth diversity of the views that are far enough from nadir to have one
    azUse = np.radians(az[off >= minAzOffNadir])
    if len(azUse) >= 1:
        R = np.hypot(np.cos(azUse).mean(), np.sin(azUse).mean())   # 1 = all same azimuth
        s = np.sort(np.degrees(azUse))
        gaps = np.diff(np.concatenate([s, [s[0] + 360]]))
        maxGap = gaps.max()
    else:
        R, maxGap = 1.0, 360.0

    # Mean B/H over pairs, same formula as imageGeometry.getBHRatio
    v = lookVectors(angles, altKm)
    pts = v[:, :2] / v[:, 2:3]
    i, j = np.triu_indices(len(angles), 1)
    bh = np.linalg.norm(pts[i] - pts[j], axis=1)

    return {
        "nImages": len(angles),
        "offNadirMean": off.mean(), "offNadirMin": off.min(),
        "offNadirMax": off.max(), "offNadirStd": off.std(),
        "nNearNadir": int(np.sum(off < 5)),
        "convMin": conv.min(), "convMedian": np.median(conv), "convMax": conv.max(),
        "fracGoodConv": np.mean((conv >= goodConv[0]) & (conv <= goodConv[1])),
        "azCircVar": 1 - R, "azMaxGap": maxGap,
        "bieMedian": np.median(bie), "asymMedian": np.median(asym),
        "bhMean": bh.mean(),
        # How far the "center of mass" of the pointings is from nadir (0 = balanced)
        "pointingImbalance": np.degrees(np.arctan(np.hypot(ta.mean(), tc.mean()))),
        "alongSpan": np.ptp(angles[:, 0]), "crossSpan": np.ptp(angles[:, 1]),
    }


FEATURE_LABELS = {
    "nImages": "number of images", "offNadirMean": "mean off-nadir (°)",
    "offNadirMin": "min off-nadir (°)", "offNadirMax": "max off-nadir (°)",
    "offNadirStd": "std of off-nadir (°)", "nNearNadir": "# images < 5° off-nadir",
    "convMin": "min convergence (°)", "convMedian": "median convergence (°)",
    "convMax": "max convergence (°)", "fracGoodConv": "fraction of pairs 10–40° conv.",
    "azCircVar": "azimuth circular variance", "azMaxGap": "largest azimuth gap (°)",
    "bieMedian": "median bisector elevation (°)", "asymMedian": "median asymmetry (°)",
    "bhMean": "mean B/H", "pointingImbalance": "pointing imbalance (°)",
    "alongSpan": "along-track span (°)", "crossSpan": "cross-track span (°)",
}


# =====================================================================================
# Sampling
# =====================================================================================
_CONE_CACHE = {}


def coneGrid(maxOff):
    """Every integer (along, cross) pointing whose TRUE off-nadir angle <= maxOff."""
    key = float(maxOff)
    if key not in _CONE_CACHE:
        m = int(np.ceil(maxOff))
        g = np.arange(-m, m + 1)
        pts = np.array(np.meshgrid(g, g, indexing="ij")).reshape(2, -1).T.astype(float)
        _CONE_CACHE[key] = pts[imageGeometry.offNadir(pts) <= maxOff + 1e-9]
    return _CONE_CACHE[key]


def randomSet(rng, nImages, maxOff=30, unique=True):
    """nImages integer (along, cross) pointings drawn uniformly from the cone of TRUE
    off-nadir angle <= maxOff (the satellite's agility limit). unique=True avoids
    imaging the exact same pointing twice. (The original randomImageGroupStudy drew
    from the box |along|, |cross| <= 30, whose corners reach ~39 deg off-nadir.)"""
    pts = coneGrid(maxOff)
    idx = rng.choice(len(pts), size=nImages, replace=not unique)
    return pts[idx].copy()


def randomFeasibleCosts(ev, rng, nImages, nSets, maxOff=30, maxTries=None):
    """
    Costs of nSets random sets that MEET the overlap requirement (rejection sampling),
    plus the share of draws that were feasible. Random sets in a +/-30 deg box mostly
    fail a 75 % overlap requirement, so this is the fair "no planning" baseline.
    Stops after maxTries draws (default 40 x nSets) and returns what it found.
    """
    maxTries = 40 * nSets if maxTries is None else maxTries
    costs, tries = [], 0
    while len(costs) < nSets and tries < maxTries:
        tries += 1
        r = ev.evaluate(randomSet(rng, nImages, maxOff))
        if r["ok"]:
            costs.append(r["cost"])
    return np.array(costs), (len(costs) / tries if tries else np.nan)


# =====================================================================================
# Simulated annealing over image sets
# =====================================================================================
def anneal(costFn, nImages, rng, iters=1500, maxOff=30, init=None, initMaxOff=None,
           T0=None, T1=1e-4, maxNudge=3, pTeleport=0.2, callback=None):
    """
    Minimize costFn(angles) over sets of nImages integer pointings inside the cone of
    true off-nadir angle <= maxOff.

    Moves (one image at a time):
      nudge    - shift one image by up to +/-maxNudge degrees on each axis
      teleport - move one image to a random grid point (probability pTeleport)
    Temperature falls geometrically from T0 to T1. If T0 is None it is set from the
    median cost change of 20 random moves, so it adapts to the cost's scale.
    initMaxOff: draw the random starting set inside +/-initMaxOff (default maxOff).
    With a 75 % overlap requirement, +/-20 deg starts are always feasible, which saves
    the optimizer from spending its budget climbing out of infeasible sets.

    Returns dict(best=angles, bestCost=float, trace=(iters,) best-so-far,
                 current=(iters,) current cost).
    """
    cur = (randomSet(rng, nImages, maxOff if initMaxOff is None else initMaxOff)
           if init is None else np.array(init, float))
    curC = costFn(cur)

    cone = coneGrid(maxOff)

    def propose(x):
        y = x.copy()
        k = rng.integers(len(y))
        if rng.random() < pTeleport:
            y[k] = cone[rng.integers(len(cone))]
        else:
            y[k] += rng.integers(-maxNudge, maxNudge + 1, size=2)
            if imageGeometry.offNadir(y[k])[0] > maxOff + 1e-9:    # outside the agility cone
                return x.copy()
        # keep pointings distinct; if not, undo
        if len(np.unique(y, axis=0)) < len(y):
            return x.copy()
        return y

    if T0 is None:
        d = [abs(costFn(propose(cur)) - curC) for _ in range(20)]
        T0 = max(np.median(d), 1e-3)
    temps = T0 * (T1 / T0) ** (np.arange(iters) / max(iters - 1, 1))

    best, bestC = cur.copy(), curC
    trace, current = np.empty(iters), np.empty(iters)
    for it, T in enumerate(temps):
        cand = propose(cur)
        c = costFn(cand)
        if c <= curC or rng.random() < np.exp(-(c - curC) / T):
            cur, curC = cand, c
            if c < bestC:
                best, bestC = cand.copy(), c
        trace[it], current[it] = bestC, curC
        if callback is not None:
            callback(it, bestC)
    return dict(best=best, bestCost=bestC, trace=trace, current=current, T0=T0)


# =====================================================================================
# Plotting
# =====================================================================================
# Fixed categorical order (never cycled). Text stays in ink colors, not series colors.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
SURFACE = "#fcfcfb"
# Sequential (one hue, light -> dark) for magnitudes such as cost
SEQ = LinearSegmentedColormap.from_list(
    "seqBlue", ["#eef5fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
# Diverging (blue <-> red, gray midpoint) for signed values such as correlations
DIV = LinearSegmentedColormap.from_list(
    "divBlueRed", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4a7a4", "#e34948", "#a32a2a"])
TERM_COLORS = dict(zip(COST_TERMS, SERIES))


def style():
    """One look for every figure: light surface, recessive grid, ink-colored text."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "lines.linewidth": 2, "lines.markersize": 6,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "figure.dpi": 110,
    })


def angleAxes(ax, lim=30, rings=None, title=None, zeroLines=True):
    """Square along/cross axes with dashed true-off-nadir rings every 10 deg and a solid
    ring at lim (the agility limit). zeroLines=False for heatmaps."""
    if rings is None:
        rings = [r for r in range(10, int(np.ceil(lim)), 10)] + [lim]
    for r in rings:
        pts = polarToAngles(np.full(361, r), np.arange(361))
        isLim = np.isclose(r, lim)
        ax.plot(pts[:, 0], pts[:, 1], ls="-" if isLim else "--", lw=1.2 if isLim else 0.8,
                color=INK2, alpha=0.8 if isLim else 0.6, zorder=1)
        ax.annotate(f"{r:g}°", polarToAngles(r, 45)[0], fontsize=7, color=INK2)
    if zeroLines:
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax.set_xlim(-lim - 2, lim + 2)
    ax.set_ylim(-lim - 2, lim + 2)
    ax.set_aspect("equal")
    ax.set_xlabel("Along-track angle (°, + fore)")
    ax.set_ylabel("Cross-track angle (°, + starboard)")
    if title:
        ax.set_title(title)


def save(fig, outDir, name):
    """Save a figure as PNG into outDir and close it."""
    outDir = Path(outDir)
    outDir.mkdir(parents=True, exist_ok=True)
    p = outDir / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")
    return p


# =====================================================================================
# Command line + bookkeeping
# =====================================================================================
def commonArgs(description, defaultOut):
    """Arguments every test accepts. Returns the argparse parser so tests can add more."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results" / defaultOut,
                   help="folder for figures and data (default: ./results/<test>)")
    p.add_argument("--sensor", default="SkySat", choices=list(SENSORS),
                   help="sensor preset (default SkySat, as in randomImageGroupStudy)")
    p.add_argument("--seed", type=int, default=20261113, help="base random seed")
    p.add_argument("--quick", action="store_true",
                   help="tiny run (minutes) to check everything works before the real one")
    g = p.add_argument_group("geometry requirements (shared by every test)")
    g.add_argument("--max-off", type=float, default=None,
                   help="agility limit: largest true off-nadir angle (deg); default = sensor's")
    g.add_argument("--min-overlap", type=float, default=0.75,
                   help="overlap requirement on the target area (0-1), default 0.75")
    g.add_argument("--pointing-err", type=float, default=0.0,
                   help="pointing error allowed for in the overlap requirement (deg)")
    g.add_argument("--aoi-scale", type=float, default=1.0,
                   help="target-area half-width as a multiple of the nadir half-swath")
    g.add_argument("--gsd-limit-m", type=float, default=None,
                   help="usable-pixel GSD requirement in metres (overrides --gsd-limit-factor)")
    g.add_argument("--gsd-limit-factor", type=float, default=1.25,
                   help="usable if GSD <= this x nadir GSD (default 1.25)")
    g.add_argument("--flat-earth", action="store_true", help="use flat-Earth geometry")
    g.add_argument("--target-bh", type=float, default=None,
                   help="target mean B/H for the B/H term (class default 1.0)")
    return p


def resolveArgs(args):
    """Fill defaults that depend on the sensor (call right after parse_args)."""
    if args.max_off is None:
        args.max_off = SENSORS[args.sensor]["maxOffNadir"]
    return args


def makeEvaluator(args, **kw):
    """CostEvaluator built from the shared command-line options; kw overrides them."""
    opts = dict(maxOffNadir=args.max_off, minOverlap=args.min_overlap,
                pointingErrDeg=args.pointing_err, aoiScale=args.aoi_scale,
                gsdLimitM=args.gsd_limit_m, gsdLimitFactor=args.gsd_limit_factor,
                sphericalEarth=not args.flat_earth)
    opts.update(kw)
    sensor = opts.pop("sensor", args.sensor)
    ev = CostEvaluator(sensor, **opts)
    if getattr(args, "target_bh", None) is not None:
        ev.geo.targetBH = args.target_bh
    return ev


class Timer:
    """with Timer('label'): ...  prints how long the block took."""
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.time()
        print(f"\n== {self.label} ==")
        return self

    def __exit__(self, *exc):
        print(f"   ({time.time() - self.t:.1f} s)")


def loadRandomStudy(pathCost, nFiles=500):
    """Load angles from the original random study's random_15_images_{j}.npy files.
    Returns (angles (N,15,2), oldCost (N,)). The old costs used the old cost
    function, so re-evaluate the angles before comparing with new results."""
    pathCost = Path(pathCost)
    angs, costs = [], []
    for j in range(1, nFiles + 1):
        f = pathCost / f"random_15_images_{j}.npy"
        if not f.exists():
            continue
        m = np.asarray(np.load(f, allow_pickle=True), float)
        if m.shape == (2, 16):
            m = m.T
        if m.shape == (16, 2):
            angs.append(m[:15])
            costs.append(m[15, 0])
    if not angs:
        raise FileNotFoundError(f"no random_15_images_*.npy files in {pathCost}")
    return np.stack(angs), np.array(costs)
