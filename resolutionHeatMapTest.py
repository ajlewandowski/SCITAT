"""
Gridding and resolution heat map test file
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from imageGridding import imageGridding
from tatc import utils
from tatc.schemas import Instrument, Satellite, TwoLineElements
import pandas as pd
from getTLEData import getTLEData
from tatc.analysis import collect_orbit_track, OrbitCoordinate, OrbitOutput

wgsSMA = 6378137.0
wgsf = 1 / 298.257223563
ae = 6378137

def runTest():
    print()
    print("###################################################################################################")
    print()

    # This step is defining parameters of Landsat-8 and its orbit, as well as finding the closest epoch to the desired
    # image time (imTime). This section will collect the ground track of the orbit.
    imTime = ["2025-12-30T17:00:00", "2025-12-30T17:03:24"]

    tlePath = Path(r"D:\Users\al62352\SpaceTrackTLE\tle49260.txt")
    tles = getTLEData(tlePath)
    line1, line2, closeTime = tles.printTLE(imTime[0])
    print()

    #                  FOV, Focal, Pixel #, GSD, Swath, Alt
    sens = np.array([np.deg2rad(15.0), 0.886, 6196, 30, 185000, 705000])
    target = np.array([30.30617, -97.88121])
    
    oli = Instrument(name="OLI",field_of_regard=utils.swath_width_to_field_of_regard(705000, 185000),req_target_sunlit=True)
    L9 = Satellite(name="Landsat-9",
        orbit=TwoLineElements(tle=[line1,line2]), instruments=[oli])
    times = pd.date_range(start=closeTime, end=imTime[1], freq="1s", tz="UTC")
    state84 = collect_orbit_track(L9, times, coordinates = OrbitCoordinate.WGS84, sat_sunlit = True, solar_altaz = True, solar_beta = True)
    stateECEF = collect_orbit_track(L9, times, orbit_output = OrbitOutput.POSITION_VELOCITY, coordinates = OrbitCoordinate.ECEF)    

    startAng = 0.0
    stepAng = 1.0
    stopAng = 25.0
    altitude = 705

    # print(f"Producing Ground Sampling Distance (GSD) Heatmaps from {startAng}° to {stopAng}° in steps of {stepAng}°.")
    gridded = imageGridding("49260", state84, stateECEF, times, sens, target)
    angVecs = np.arange(startAng, stopAng + stepAng, stepAng)
    minGSD, maxGSD = gridded.getColorbarMinMax(angVecs, altitude)

    pathHeatmap = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_705km")

    # flips = np.array([1.0,2.0,3.0,4.0])

    # for i in range(len(flips)):
    #     for j in range(len(flips)):
    #         imData = np.load(Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9_705km/GSDarray_matrix_705_km_offnadir_({flips[i]},{flips[j]}).npy"))
    #         gridded.flipOne2Two(pathHeatmap, flips[i], flips[j], altitude, minGSD, maxGSD, imData)
    #         gridded.flipOne2Three(pathHeatmap, flips[i], flips[j], altitude, minGSD, maxGSD, imData)
    #         gridded.flipOne2Four(pathHeatmap, flips[i], flips[j], altitude, minGSD, maxGSD, imData)

    fore = 1.0
    star = 1.0
    imData = np.load(Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9_705km/GSDarray_matrix_705_km_offnadir_({fore},{star}).npy"))
    gridded.flipOne2Two(pathHeatmap, fore, star, altitude, minGSD, maxGSD, imData)
    gridded.flipOne2Three(pathHeatmap, fore, star, altitude, minGSD, maxGSD, imData)
    gridded.flipOne2Four(pathHeatmap, fore, star, altitude, minGSD, maxGSD, imData)

    fore = 6.0
    star = 6.0
    imData = np.load(Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9_705km/GSDarray_matrix_705_km_offnadir_({fore},{star}).npy"))
    gridded.flipOne2Two(pathHeatmap, fore, star, altitude, minGSD, maxGSD, imData)
    gridded.flipOne2Three(pathHeatmap, fore, star, altitude, minGSD, maxGSD, imData)
    gridded.flipOne2Four(pathHeatmap, fore, star, altitude, minGSD, maxGSD, imData)

    fore = 11.0
    star = 11.0
    imData = np.load(Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9_705km/GSDarray_matrix_705_km_offnadir_({fore},{star}).npy"))
    gridded.flipOne2Two(pathHeatmap, fore, star, altitude, minGSD, maxGSD, imData)
    gridded.flipOne2Three(pathHeatmap, fore, star, altitude, minGSD, maxGSD, imData)
    gridded.flipOne2Four(pathHeatmap, fore, star, altitude, minGSD, maxGSD, imData)

    fore = 16.0
    star = 16.0
    imData = np.load(Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9_705km/GSDarray_matrix_705_km_offnadir_({fore},{star}).npy"))
    gridded.flipOne2Two(pathHeatmap, fore, star, altitude, minGSD, maxGSD, imData)
    gridded.flipOne2Three(pathHeatmap, fore, star, altitude, minGSD, maxGSD, imData)
    gridded.flipOne2Four(pathHeatmap, fore, star, altitude, minGSD, maxGSD, imData)

    fore = 21.0
    star = 21.0
    imData = np.load(Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9_705km/GSDarray_matrix_705_km_offnadir_({fore},{star}).npy"))
    gridded.flipOne2Two(pathHeatmap, fore, star, altitude, minGSD, maxGSD, imData)
    gridded.flipOne2Three(pathHeatmap, fore, star, altitude, minGSD, maxGSD, imData)
    gridded.flipOne2Four(pathHeatmap, fore, star, altitude, minGSD, maxGSD, imData)

    print()



if __name__ == "__main__":
    runTest()