"""
Gridding and resolution heat map test file for changing altitudes in the cost trials
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

    alt = 705

    # This step is defining parameters of Landsat-8 and its orbit, as well as finding the closest epoch to the desired
    # image time (imTime). This section will collect the ground track of the orbit.
    imTime = ["2025-12-30T17:00:00", "2025-12-30T17:03:24"]

    tlePath = Path("/exports/nfs_share/Users/al62352/SpaceTrackTLE/tle49260.txt")
    tles = getTLEData(tlePath)
    line1, line2, closeTime = tles.printTLE(imTime[0])
    print()

    #                  FOV, Focal, Pixel #, GSD, Swath, Alt
    swath = 2*alt*1e3*np.tan(np.deg2rad(15.0/2))
    sens = np.array([np.deg2rad(15.0), 0.886, 6196, swath/6196, swath, alt*1e3])
    target = np.array([30.30617, -97.88121])
    
    oli = Instrument(name="OLI",field_of_regard=utils.swath_width_to_field_of_regard(alt*1e3, swath),req_target_sunlit=True)
    L9 = Satellite(name="Landsat-9",
        orbit=TwoLineElements(tle=[line1,line2]), instruments=[oli])
    times = pd.date_range(start=closeTime, end=imTime[1], freq="1s", tz="UTC")
    state84 = collect_orbit_track(L9, times, coordinates = OrbitCoordinate.WGS84, sat_sunlit = True, solar_altaz = True, solar_beta = True)
    stateECEF = collect_orbit_track(L9, times, orbit_output = OrbitOutput.POSITION_VELOCITY, coordinates = OrbitCoordinate.ECEF)    

    #### Last image generated was (32, 9)
    angVecsNeg = np.arange(-40, 1, 1)
    altitude = alt

    # print(f"Producing Ground Sampling Distance (GSD) Heatmaps from {startAng}° to {stopAng}° in steps of {stepAng}°.")
    gridded = imageGridding("49260", state84, stateECEF, times, sens, target)
    minGSD, maxGSD = gridded.getColorbarMinMax(angVecsNeg, altitude)

    angVecsPos = np.arange(0, 41, 1)
    altitude = alt
    
    pathHeatmap = Path(f"/exports/nfs_share/Users/al62352/ResolutionHeatmaps/LS9_705km")

    for i in range(30,len(angVecsPos),1):
        for j in range(len(angVecsPos)):
            fig = gridded.getGSDArray(altitude, angVecsPos[i], angVecsPos[j], minGSD, maxGSD, True, True, pathHeatmap)
            plt.close()

    for i in range(len(angVecsPos)):
        for j in range(len(angVecsPos)):
            fig = gridded.getGSDArray(altitude, angVecsNeg[i], angVecsPos[j], minGSD, maxGSD, True, True, pathHeatmap)
            plt.close()

    for i in range(len(angVecsPos)):
        for j in range(len(angVecsPos)):
            fig = gridded.getGSDArray(altitude, angVecsPos[i], angVecsNeg[j], minGSD, maxGSD, True, True, pathHeatmap)
            plt.close()

    for i in range(len(angVecsPos)):
        for j in range(len(angVecsPos)):
            fig = gridded.getGSDArray(altitude, angVecsNeg[i], angVecsNeg[j], minGSD, maxGSD, True, True, pathHeatmap)
            plt.close()


    print()








if __name__ == "__main__":
    runTest()