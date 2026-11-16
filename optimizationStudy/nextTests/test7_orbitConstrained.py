"""
Test 7 - Orbit-constrained geometry: what can an orbit actually deliver?
========================================================================
Question: the earlier tests let every (along, cross) pointing be available. A real
satellite only sees the target on certain passes, each pass fixes the cross-track
angle, and slew time limits how many along-track looks fit in one pass. Given an
orbit, a collection window and an agility limit, what is the best image set you
can actually collect - and which orbit choices (altitude, window length, number
of satellites) get you closest to the unconstrained optimum?
This is the step that turns "optimal geometry" into "optimal orbit / tasking".

What it does
  1. Propagates each satellite with SGP4 (circular sun-synchronous orbits built
     from an altitude, or a real TLE with --tle) at --dt second steps.
  2. Finds every time the target is within the agility limit (true off-nadir angle
     <= --max-off, default the sensor's limit, e.g. 48 deg for Pleiades Neo), with
     the sun at least --min-sun deg above the target's horizon (optical imaging).
     Consecutive samples form a pass. Each sample is a candidate image with its
     own along/cross angle and altitude.
  3. Chooses images with simulated annealing over the candidates. Feasibility:
     inside one pass, two images need  dt >= slew angle / slew rate + settle time.
     The set size is free between 2 and --max-images (moves add, remove, swap, or
     shift an image in time), so the optimizer also decides how many to take.
  4. Repeats for several collection windows (days) and orbit configurations.
  5. Reference: the best unconstrained ring geometry (test 4/5 result).

Look-angle convention: frame at the satellite with z = nadir (toward Earth's
center), x = ground-relative velocity (yaw-steered along the ground track),
y = z cross x (starboard). along = atan2(LOS.x, LOS.z), cross = atan2(LOS.y, LOS.z),
the same convention as lookVectors() in imageGeometryClass.

Outputs (in --out, default ./results/test7_orbitConstrained)
  results.csv                      per configuration and window: best cost, terms, #images, #passes
  best_sets.csv                    the chosen images (time, pass, angles, sun) for every case
  1_ground_tracks.png              where the accessible passes are, by day
  2_candidates_and_choice.png      every candidate pointing and the chosen set
  3_cost_vs_window.png             best achievable cost vs collection window, per orbit
  4_collection_timeline.png        when the chosen images are taken, and at what sun elevation
  5_components.png                 cost terms: unconstrained optimum vs orbit-constrained
  6_access_vs_window.png           how many passes each orbit gets

Run
  python test7_orbitConstrained.py --quick
  python test7_orbitConstrained.py                                  # ~20 min
  python test7_orbitConstrained.py --tle ../../processingTLEs/tle49070.txt --alts 0
  python test7_orbitConstrained.py --target 30.30617,-97.88121 --slew-rate 2 --settle 3
"""
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sgp4.api import Satrec, WGS72, jday

from studyCommon import (makeEvaluator, resolveArgs, pointingVectors, CostEvaluator, COST_TERMS, TERM_LABELS, TERM_COLORS, SERIES, SEQ, INK2,
                         SENSORS, angleAxes, anglesToPolar, lookVectors, polarToAngles, save,
                         style, commonArgs, Timer, REPO_ROOT)

MU = 398600.4418          # km^3/s^2
RE = 6378.137             # km
J2 = 1.08262668e-3
F_WGS84 = 1 / 298.257223563
OMEGA_E = 7.2921150e-5    # rad/s
SSO_RATE = 2 * np.pi / (365.2422 * 86400)


# =====================================================================================
# Orbit and Earth geometry
# =====================================================================================
def ssoInclination(alt_km):
    """Inclination (rad) that makes a circular orbit sun-synchronous."""
    a = RE + alt_km
    n = np.sqrt(MU / a ** 3)
    return np.arccos(-SSO_RATE / (1.5 * n * J2 * (RE / a) ** 2))


def gmst(jd):
    """Greenwich mean sidereal time (rad), adequate for pointing geometry."""
    return np.radians((280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360)


def sunVectorEci(jd):
    """Unit vector to the Sun (low-precision almanac, ~0.01 deg)."""
    n = jd - 2451545.0
    L = np.radians(280.460 + 0.9856474 * n)
    g = np.radians(357.528 + 0.9856003 * n)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 4e-7 * n)
    return np.column_stack([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)])


def eciToEcef(v, th):
    """Rotate (n,3) vectors from TEME/ECI to ECEF by the sidereal angle th (n,)."""
    c, s = np.cos(th), np.sin(th)
    return np.column_stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]])


def geodeticToEcef(lat, lon, h_km=0.0):
    lat, lon = np.radians(lat), np.radians(lon)
    e2 = F_WGS84 * (2 - F_WGS84)
    N = RE / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.array([(N + h_km) * np.cos(lat) * np.cos(lon), (N + h_km) * np.cos(lat) * np.sin(lon),
                     (N * (1 - e2) + h_km) * np.sin(lat)])


def makeSatellite(alt_km, epoch, raanOffset_deg=0.0, phase_deg=0.0, ltdn_hours=10.5, satnum=90000):
    """Circular sun-synchronous orbit with the given local time of descending node."""
    jd, fr = jday(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second)
    sun = sunVectorEci(np.array([jd + fr]))[0]
    raSun = np.arctan2(sun[1], sun[0])
    # ascending node is 12 h after the descending node in local time
    raan = raSun + np.radians((ltdn_hours + 12 - 12) * 15.0) + np.radians(raanOffset_deg)
    a = RE + alt_km
    n_radmin = np.sqrt(MU / a ** 3) * 60.0
    sat = Satrec()
    sat.sgp4init(WGS72, "i", satnum, (jd + fr) - 2433281.5, 0.0, 0.0, 0.0, 1e-4,
                 0.0, ssoInclination(alt_km), np.radians(phase_deg), n_radmin, raan % (2 * np.pi))
    return sat


def loadTle(path, epoch):
    """Latest TLE at or before epoch, using SCITAT's getTLEData."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "processingTLEs"))
    from getTLEData import getTLEData
    tles = getTLEData(str(path))
    l1, l2, _ = tles.closestTLE(epoch)
    return Satrec.twoline2rv(l1, l2)


# =====================================================================================
# Access and candidates
# =====================================================================================
def candidates(sat, satId, start, days, target, maxOff, minSun, dtSec):
    """Every sample where the target is within maxOff deg off-nadir and sunlit."""
    t = np.arange(0, days * 86400, dtSec, dtype=float)
    jd0, fr0 = jday(start.year, start.month, start.day, start.hour, start.minute, start.second)
    fr = fr0 + t / 86400.0
    jd = np.full_like(fr, jd0)
    err, r, v = sat.sgp4_array(jd, fr)
    ok = err == 0
    th = gmst(jd + fr)
    rE = eciToEcef(r, th)
    vE = eciToEcef(v, th) - np.cross(np.array([0, 0, OMEGA_E]), rE)     # ground-relative velocity

    T = geodeticToEcef(*target)
    up = T / np.linalg.norm(T)
    los = T - rE
    z = -rE / np.linalg.norm(rE, axis=1, keepdims=True)
    x = vE - np.sum(vE * z, 1, keepdims=True) * z
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(z, x)
    lz, lx, ly = np.sum(los * z, 1), np.sum(los * x, 1), np.sum(los * y, 1)
    along = np.degrees(np.arctan2(lx, lz))
    cross = np.degrees(np.arctan2(ly, lz))
    sunE = eciToEcef(sunVectorEci(jd + fr), th)
    sunEl = np.degrees(np.arcsin(np.clip(sunE @ up, -1, 1)))
    elevSat = np.degrees(np.arcsin(np.clip(np.sum(-los / np.linalg.norm(los, axis=1, keepdims=True) * up, 1), -1, 1)))
    lat = np.degrees(np.arcsin(rE[:, 2] / np.linalg.norm(rE, axis=1)))
    alt = np.linalg.norm(rE, axis=1) - RE * (1 - F_WGS84 * np.sin(np.radians(lat)) ** 2)

    offNad = np.degrees(np.arctan(np.hypot(lx, ly) / np.where(lz > 0, lz, np.nan)))
    vis = ok & (lz > 0) & (offNad <= maxOff) & (sunEl >= minSun) & (elevSat > 0)
    idx = np.flatnonzero(vis)
    if len(idx) == 0:
        return pd.DataFrame()
    passNo = np.concatenate([[0], np.cumsum(np.diff(t[idx]) > 3 * dtSec)])
    lon = np.degrees(np.arctan2(rE[idx, 1], rE[idx, 0]))
    return pd.DataFrame({"sat": satId, "pass": [f"{satId}-{p}" for p in passNo], "t": t[idx],
                         "along": along[idx], "cross": cross[idx], "alt": alt[idx],
                         "sunEl": sunEl[idx], "lat": lat[idx], "lon": lon})


# =====================================================================================
# Constrained selection
# =====================================================================================
class Selector:
    """Simulated annealing over subsets of candidate images with slew feasibility."""

    def __init__(self, cand, ev, slewRate, settle, maxImages, rng):
        self.c = cand.reset_index(drop=True)
        self.ev, self.rng = ev, rng
        self.slew, self.settle, self.maxN = slewRate, settle, maxImages
        self.A = self.c[["along", "cross"]].to_numpy()
        self.L = pointingVectors(self.A)       # satellite frame: slew angles
        self.t = self.c.t.to_numpy()
        self.p = self.c["pass"].to_numpy()
        self.alt = self.c.alt.to_numpy()
        self.byPass = {k: np.flatnonzero(self.p == k) for k in np.unique(self.p)}
        self.memo = {}

    def feasible(self, sel):
        if len(set(sel)) < len(sel):
            return False
        for k in set(self.p[sel]):
            s = sorted((i for i in sel if self.p[i] == k), key=lambda i: self.t[i])
            for i, j in zip(s[:-1], s[1:]):
                ang = np.degrees(np.arccos(np.clip(self.L[i] @ self.L[j], -1, 1)))
                if self.t[j] - self.t[i] < ang / self.slew + self.settle:
                    return False
        return True

    def cost(self, sel):
        key = tuple(sorted(sel))
        if key not in self.memo:
            self.memo[key] = self.ev.cost(self.A[list(key)], alt=self.alt[list(key)]) if len(key) >= 2 else 1.0
        return self.memo[key]

    def propose(self, sel):
        s = list(sel)
        u = self.rng.random()
        n = len(self.c)
        if (u < 0.15 and len(s) < self.maxN) or len(s) < 2:
            s.append(int(self.rng.integers(n)))                     # add
        elif u < 0.25 and len(s) > 2:
            s.pop(int(self.rng.integers(len(s))))                   # remove
        elif u < 0.6:
            s[int(self.rng.integers(len(s)))] = int(self.rng.integers(n))   # swap for any candidate
        else:                                                       # shift in time within the pass
            k = int(self.rng.integers(len(s)))
            members = self.byPass[self.p[s[k]]]
            pos = int(np.searchsorted(members, s[k]))
            s[k] = int(members[np.clip(pos + self.rng.integers(-3, 4), 0, len(members) - 1)])
        return s if self.feasible(s) else list(sel)

    def greedy(self, nTry=150):
        """Feasible starting set: best pair from a sample, then keep adding the best
        feasible image up to maxN (even if the cost briefly rises) and return the
        best prefix - adding one image can hurt while adding two helps."""
        n = len(self.c)
        cur, curC = None, np.inf
        for _ in range(nTry):
            s = [int(v) for v in self.rng.choice(n, 2, replace=False)]
            if self.feasible(s):
                c = self.cost(s)
                if c < curC:
                    cur, curC = s, c
        if cur is None:
            return None, np.inf
        best, bestC = list(cur), curC
        while len(cur) < self.maxN:
            trial = self.rng.choice(n, min(nTry, n), replace=False)
            opts = [(self.cost(cur + [int(i)]), int(i)) for i in trial
                    if int(i) not in cur and self.feasible(cur + [int(i)])]
            if not opts:
                break
            c, i = min(opts)
            cur, curC = cur + [i], c
            if c < bestC:
                best, bestC = list(cur), c
        return best, bestC

    def spread(self):
        """Heuristic starting set: the first and last sample of every pass (or just the
        middle one when the slew limit doesn't allow both), capped at maxN images."""
        s = []
        for k, members in self.byPass.items():
            ends = [int(members[0]), int(members[-1])]
            s += ends if self.feasible(ends) else [int(members[len(members) // 2])]
        s = s[:self.maxN]
        return (s, self.cost(s)) if len(s) >= 2 and self.feasible(s) else (None, np.inf)

    def fromChosen(self, chosen):
        """Map a previously chosen set (rows with sat and t) onto this candidate list."""
        if chosen is None:
            return None, np.inf
        key = {(sa, tt): i for i, (sa, tt) in enumerate(zip(self.c.sat, self.c.t))}
        s = [key[(sa, tt)] for sa, tt in zip(chosen.sat, chosen.t) if (sa, tt) in key]
        return (s, self.cost(s)) if len(s) >= 2 and self.feasible(s) else (None, np.inf)

    def run(self, iters, warm=None):
        # Start from the best of: greedy, one-or-two-per-pass, and (for a longer window)
        # the shorter window's answer - so a longer window can never score worse.
        starts = [self.greedy(), self.spread(), self.fromChosen(warm)]
        cur, curC = min(starts, key=lambda sc: sc[1])
        if cur is None:
            return None, np.inf
        best, bestC = list(cur), curC
        d = [abs(self.cost(self.propose(cur)) - curC) for _ in range(15)]
        T0 = max(np.median(d), 1e-3)
        for T in T0 * (1e-4 / T0) ** (np.arange(iters) / max(iters - 1, 1)):
            cand = self.propose(cur)
            c = self.cost(cand)
            if c <= curC or self.rng.random() < np.exp(-(c - curC) / T):
                cur, curC = cand, c
                if c < bestC:
                    best, bestC = list(cand), c
        return sorted(best, key=lambda i: self.t[i]), bestC


def bestUnconstrained(ev, maxN, maxOff):
    """Best ring geometry with 2..maxN images (N on a ring or nadir + N-1)."""
    best = (np.inf, None)
    for N in range(2, maxN + 1):
        for th in np.arange(2.5, maxOff + 0.1, 2.5):
            for s in (polarToAngles(np.full(N, th), 360.0 * np.arange(N) / N),
                      np.vstack([[0, 0], polarToAngles(np.full(N - 1, th), 360.0 * np.arange(N - 1) / (N - 1))])
                      if N >= 3 else None):
                if s is not None:
                    c = ev.cost(s, failCost=np.inf, infeasible="fail")
                    if c < best[0]:
                        best = (c, s)
    return best


# =====================================================================================
def main():
    p = commonArgs(__doc__.split("\n")[1], "test7_orbitConstrained")
    p.add_argument("--target", default="30.30617,-97.88121", help="lat,lon in deg (default: Austin test site)")
    p.add_argument("--start", default="2025-12-30T00:00:00", help="UTC start of the collection window")
    p.add_argument("--windows", default="1,2,3,5,7,10,14", help="collection windows to test (days)")
    p.add_argument("--alts", default=None,
                   help="single-satellite SSO altitudes (km), comma separated; 0 = skip "
                        "(use with --tle). Default: the sensor's altitude and +50/+100/+150 km")
    p.add_argument("--constellations", action="store_true", default=True,
                   help="also test two 2-satellite constellations at the sensor's nominal altitude")
    p.add_argument("--no-constellations", dest="constellations", action="store_false")
    p.add_argument("--tle", default=None, help="TLE file: add this real satellite as a configuration")
    p.add_argument("--min-sun", type=float, default=15.0, help="min sun elevation at the target (deg)")
    p.add_argument("--slew-rate", type=float, default=1.0, help="deg/s")
    p.add_argument("--settle", type=float, default=5.0, help="s, per re-point")
    p.add_argument("--dt", type=float, default=5.0, help="propagation / candidate step (s)")
    p.add_argument("--max-images", type=int, default=15)
    p.add_argument("--iters", type=int, default=800, help="annealing iterations per case")
    args = resolveArgs(p.parse_args())
    if args.alts is None:
        a0 = SENSORS[args.sensor]["alt_km"]
        args.alts = ",".join(str(a0 + d) for d in (0, 50, 100, 150))
    if args.quick:
        args.windows, args.alts, args.constellations, args.iters = "3,7", args.alts.split(",")[0], False, 150
    style()
    out = args.out
    target = tuple(float(v) for v in args.target.split(","))
    start = dt.datetime.fromisoformat(args.start)
    windows = [float(v) for v in args.windows.split(",")]
    nomAlt = SENSORS[args.sensor]["alt_km"]

    # ------------------------------------------------------------------ configurations
    configs = []        # (name, list of Satrec, altitude for the GSD images)
    for a in [float(v) for v in args.alts.split(",") if float(v) > 0]:
        configs.append((f"1 sat, {a:.0f} km", [makeSatellite(a, start)], a))
    if args.constellations:
        configs.append((f"2 sats same plane, {nomAlt} km",
                        [makeSatellite(nomAlt, start), makeSatellite(nomAlt, start, phase_deg=180, satnum=90001)], nomAlt))
        configs.append((f"2 sats, planes 30° apart, {nomAlt} km",
                        [makeSatellite(nomAlt, start), makeSatellite(nomAlt, start, raanOffset_deg=30, satnum=90001)], nomAlt))
    if args.tle:
        s = loadTle(args.tle, start)
        a = (MU / (s.no_kozai / 60) ** 2) ** (1 / 3) - RE
        configs.append((f"TLE {s.satnum}", [s], round(a)))
    if not configs:
        raise SystemExit("no configurations: give --alts or --tle")

    # Reference geometry (unconstrained) at the nominal altitude
    evNom = makeEvaluator(args)
    with Timer("Unconstrained reference (best ring geometry)"):
        refCost, refSet = bestUnconstrained(evNom, args.max_images, args.max_off)
        refTerms = evNom.evaluate(refSet)
        print(f"   {len(refSet)} images, cost {refCost:.4f}")

    rows, chosenRows, keep = [], [], {}
    for ci, (name, sats, altGSD) in enumerate(configs):
        ev = makeEvaluator(args, alt_km=altGSD, norm_alt_km=nomAlt)
        with Timer(f"{name}: propagating {max(windows):g} days"):
            allCand = pd.concat([candidates(s, k, start, max(windows), target, args.max_off, args.min_sun, args.dt)
                                 for k, s in enumerate(sats)], ignore_index=True)
            print(f"   {allCand['pass'].nunique() if len(allCand) else 0} sunlit passes within the "
                  f"{args.max_off:g}° off-nadir limit, {len(allCand)} candidate samples")
        prevChosen = None
        for w in sorted(windows):
            cand = allCand[allCand.t < w * 86400] if len(allCand) else allCand
            nPass = cand["pass"].nunique() if len(cand) else 0
            with Timer(f"{name}: {w:g}-day window ({nPass} passes)"):
                if len(cand) < 2:
                    rows.append({"config": name, "window_days": w, "passes": nPass, "nImages": 0,
                                 "cost": np.nan})
                    print("   not enough access")
                    continue
                sel = Selector(cand, ev, args.slew_rate, args.settle, args.max_images,
                               np.random.default_rng([args.seed, ci, int(w * 10)]))
                best, bestC = sel.run(args.iters, warm=prevChosen)
                if best is None:
                    rows.append({"config": name, "window_days": w, "passes": nPass, "nImages": 0, "cost": np.nan})
                    print("   no feasible pair")
                    continue
                chosen = sel.c.iloc[best]
                r = ev.evaluate(chosen[["along", "cross"]].to_numpy(), alt=chosen.alt.to_numpy())
                rows.append({"config": name, "window_days": w, "passes": nPass, "nImages": len(best),
                             "passesUsed": chosen["pass"].nunique(), "cost": bestC,
                             **{k: r[k] for k in COST_TERMS}})
                for _, row in chosen.iterrows():
                    chosenRows.append({"config": name, "window_days": w, **row.to_dict()})
                keep[(name, w)] = (sel.c, chosen)
                prevChosen = chosen
                print(f"   cost {bestC:.4f} with {len(best)} images from {chosen['pass'].nunique()} passes")

    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    pd.DataFrame(chosenRows).to_csv(out / "best_sets.csv", index=False)

    main_cfg = configs[0][0]
    showW = max(w for w in windows if (main_cfg, w) in keep) if any(k[0] == main_cfg for k in keep) else None

    # ------------------------------------------------------------------ figure 1: ground tracks
    if showW is not None:
        cand, chosen = keep[(main_cfg, showW)]
        fig, ax = plt.subplots(figsize=(8.5, 7), layout="constrained")
        day = cand.t / 86400
        sc = ax.scatter(cand.lon, cand.lat, c=day, cmap=SEQ, s=6, vmin=0, vmax=showW)
        ax.scatter(chosen.lon, chosen.lat, s=70, facecolor="none", edgecolor=SERIES[1], lw=1.5,
                   label="sub-satellite point of chosen images", zorder=3)
        ax.scatter(target[1], target[0], marker="*", s=300, color=SERIES[3], edgecolor="k", zorder=4,
                   label="target")
        fig.colorbar(sc, ax=ax, label="day of window")
        ax.set_xlabel("longitude (°)"); ax.set_ylabel("latitude (°)")
        ax.set_aspect(1 / np.cos(np.radians(target[0])))
        ax.legend(loc="lower left", fontsize=8)
        ax.set_title(f"Test 7 – Accessible sunlit pass segments, {main_cfg}, {showW:g} days")
        save(fig, out, "1_ground_tracks")

        # ------------------------------------------------------------------ figure 2: candidates
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), layout="constrained")
        ax = axes[0]
        ax.scatter(cand.along, cand.cross, s=6, color="#bdbcb6", label="every candidate sample")
        sc = ax.scatter(chosen.along, chosen.cross, c=chosen.t / 86400, cmap=SEQ, vmin=0, vmax=showW,
                        s=110, edgecolor="k", lw=0.6, zorder=3, label="chosen images")
        for i, (_, r) in enumerate(chosen.iterrows()):
            ax.annotate(str(i + 1), (r.along, r.cross), xytext=(6, 4), textcoords="offset points", fontsize=8)
        ax.scatter(refSet[:, 0], refSet[:, 1], marker="x", s=60, color=SERIES[1], zorder=2,
                   label="unconstrained optimum")
        angleAxes(ax, args.max_off, title=f"{main_cfg}, {showW:g}-day window")
        fig.colorbar(sc, ax=ax, fraction=0.046, label="day taken")
        ax.legend(fontsize=8, loc="lower left")
        ax = fig.add_subplot(1, 2, 2, projection="polar")
        axes[1].set_visible(False)
        off, az = anglesToPolar(chosen[["along", "cross"]].to_numpy())
        offR, azR = anglesToPolar(refSet)
        ax.scatter(np.radians(azR), offR, marker="x", s=60, color=SERIES[1], label="unconstrained optimum")
        ax.scatter(np.radians(az), off, s=90, color=SERIES[0], edgecolor="k", lw=0.5, label="orbit-constrained")
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_rlim(0, args.max_off * 1.35)
        ax.set_title("Pointing azimuth (0° = fore, 90° = starboard) vs off-nadir", fontsize=10)
        ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(-0.1, -0.12))
        fig.suptitle("Test 7 – What the orbit offers vs what the optimizer took", fontsize=12)
        save(fig, out, "2_candidates_and_choice")

    # ------------------------------------------------------------------ figure 3: cost vs window
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for c, (name, _, _) in zip(SERIES, configs):
        d = df[df.config == name]
        ax.plot(d.window_days, d.cost, "-o", color=c, label=name)
    ax.axhline(refCost, color=INK2, ls="--", lw=1.2, label=f"unconstrained optimum ({len(refSet)} images)")
    ax.set_xlabel("collection window (days)"); ax.set_ylabel("best achievable cost")
    ax.set_xticks(windows)
    ax.set_title("Test 7 – How close can each orbit get to the optimal geometry?")
    ax.legend(fontsize=8)
    save(fig, out, "3_cost_vs_window")

    # ------------------------------------------------------------------ figure 4: timeline
    if showW is not None:
        cand, chosen = keep[(main_cfg, showW)]
        fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
        passStart = cand.groupby("pass").t.min() / 86400
        for x in passStart:
            ax.axvline(x, color="#d9d8d3", lw=1, zorder=0)
        off, _ = anglesToPolar(chosen[["along", "cross"]].to_numpy())
        sc = ax.scatter(chosen.t / 86400, off, c=chosen.sunEl, cmap=SEQ, s=90, edgecolor="k", lw=0.5, zorder=3)
        fig.colorbar(sc, ax=ax, label="sun elevation at target (°)")
        ax.set_xlim(0, showW)
        ax.set_xlabel("day of window (thin lines = every accessible pass)")
        ax.set_ylabel("off-nadir angle (°)")
        ax.set_title(f"Test 7 – When the chosen images are taken ({main_cfg}). Images days apart see "
                     "different sun and scene conditions – the cost does not model that yet.", fontsize=10)
        save(fig, out, "4_collection_timeline")

    # ------------------------------------------------------------------ figure 5: components
    d = df[(df.config == main_cfg) & df.cost.notna()]
    if len(d):
        fig, ax = plt.subplots(figsize=(11, 5.2), layout="constrained")
        labels = ["unconstrained\noptimum"] + [f"{w:g} d" for w in d.window_days]
        T = [refTerms] + [r for _, r in d.iterrows()]
        W = evNom.geo.costWeights
        bottom = np.zeros(len(T))
        for k in COST_TERMS:
            v = np.array([W[k] * t[k] for t in T])
            ax.bar(range(len(T)), v, bottom=bottom, color=TERM_COLORS[k], width=0.6, edgecolor="white",
                   lw=1.5, label=TERM_LABELS[k])
            bottom += v
        ax.set_xticks(range(len(T)), labels)
        ax.set_ylabel("weighted contribution to total cost")
        ax.set_title(f"Test 7 – Which terms the orbit constraint costs you ({main_cfg})")
        ax.legend(ncols=len(COST_TERMS), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        save(fig, out, "5_components")

    # ------------------------------------------------------------------ figure 6: access
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for c, (name, _, _) in zip(SERIES, configs):
        dd = df[df.config == name]
        ax.plot(dd.window_days, dd.passes, "-o", color=c, label=name)
    ax.set_xlabel("collection window (days)"); ax.set_ylabel("sunlit passes with access")
    ax.set_xticks(windows)
    ax.set_title(f"Test 7 – Access to the target (off-nadir ≤ {args.max_off:g}°, sun ≥ {args.min_sun:g}°)")
    ax.legend(fontsize=8)
    save(fig, out, "6_access_vs_window")

    print("\nSummary (best achievable cost):")
    print(df.pivot(index="window_days", columns="config", values="cost").round(4).to_string())
    print(f"Unconstrained optimum: {refCost:.4f}")


if __name__ == "__main__":
    main()
