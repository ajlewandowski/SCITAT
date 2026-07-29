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

    print(f"Producing Ground Sampling Distance (GSD) Heatmaps from {startAng}° to {stopAng}° in steps of {stepAng}°.")
    gridded = imageGridding("49260", state84, stateECEF, times, sens, target)
    angVecs = np.arange(startAng, stopAng + stepAng, stepAng)
    minGSD, maxGSD = gridded.getColorbarMinMax(angVecs, altitude)

    pathHeatmap = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_705km")

    testVecs = np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    for j in range(len(testVecs)):
        for k in range(len(angVecs)):
            _ = gridded.getGSDArray(altitude, testVecs[j], angVecs[k], minGSD, maxGSD, saveMat=True, saveFig=True, path=pathHeatmap)

    print()



if __name__ == "__main__":
    runTest()