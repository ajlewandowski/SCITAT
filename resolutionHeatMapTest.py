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
    noradID = 49260

    tlePath = Path(r"D:\Users\al62352\SpaceTrackTLE\tle49260.txt")
    tles = getTLEData(tlePath)
    line1, line2, closeTime = tles.printTLE(imTime[0])
    print()

    #                  FOV, Focal, Pixel #, GSD, Swath, Alt
    sens = np.array([np.deg2rad(15.0), 0.886, 6196, 30, 185000, 705000])
    solarAngles = np.array([32.30026603, 155.16461405])
    target = np.array([30.30617, -97.88121])
    imCorner = np.array([[31.35457, -99.06834],[31.33346, -96.67196],[29.23974, -96.72091],[29.25915, -99.06690]])

    oli = Instrument(name="OLI",field_of_regard=utils.swath_width_to_field_of_regard(705000, 185000),req_target_sunlit=True)
    L8 = Satellite(name="Landsat-8",
        orbit=TwoLineElements(tle=[line1,line2]), instruments=[oli])
    times = pd.date_range(start=closeTime, end=imTime[1], freq="1s", tz="UTC")
    state84 = collect_orbit_track(L8, times, coordinates = OrbitCoordinate.WGS84, sat_sunlit = True, solar_altaz = True, solar_beta = True)
    # stateECI = collect_orbit_track(L8, times, orbit_output = OrbitOutput.POSITION_VELOCITY, coordinates = OrbitCoordinate.ECI)
    stateECEF = collect_orbit_track(L8, times, orbit_output = OrbitOutput.POSITION_VELOCITY, coordinates = OrbitCoordinate.ECEF)    
    
    # gridded = imageGridding("49260", state84, stateECEF, times, sens, target, 5.0)
    # gridded.getOffNad2DArrayFlat(705000, 5.0, 5.0)
    # gridded.plotNadirBaseArray(705)

    # gridded.analyze2DArrayModels(0.0, 0.0)

    angVecs = np.array([0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0])
    # minMat, maxMat = gridded.getColorbarMinMax(angVecs)
    # print(minMat)
    # print(maxMat)

    simpleMin = 29.78834222052943
    # sphereMin = 29.788342227426124
    # simpleMax = 37.860686691020085
    sphereMax = 39.720874420117006

    # for i in range(len(angVecs)):
    #     for k in range(len(angVecs)):
    #         gridded.getOffNad2DArraySpherical(705000, angVecs[i], angVecs[k], simpleMin, sphereMax)

    # for i in range(len(angVecs)):
    #     for k in range(len(angVecs)):
    #         fileName = f'fov_skew_2Darray_matrix_offnadir_({angVecs[i]},{angVecs[k]}).npy'
    #         pathNowSimple = Path("D:/Users/al62352/SimpleHeatmap")
    #         fullPathSimple = pathNowSimple / fileName
    #         dataSimple = np.load(fullPathSimple)

    #         fileName = f'fov_skew_2Darray_spherical_matrix_offnadir_({angVecs[i]},{angVecs[k]}).npy'
    #         pathNowSpherical = Path("D:/Users/al62352/SphericalHeatmap")
    #         fullPathSpherical = pathNowSpherical / fileName
    #         dataSpherical = np.load(fullPathSpherical)

    #         dataDiff = dataSpherical - dataSimple

    #         matFileName = f'fov_skew_2Darray_diff_matrix_offnadir_({angVecs[i]},{angVecs[k]}).npy'
    #         fullMatPath = Path("D:/Users/al62352/HeatmapDiffs") / matFileName
    #         np.save(fullMatPath, dataDiff)


    print()


if __name__ == "__main__":
    runTest()