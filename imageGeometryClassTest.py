""" This file will test and ensure the imageGeometry class is working correctly """

import numpy as np
from imageGeometry import imageGeometry
from pathlib import Path
import matplotlib.pyplot as plt
from imageGridding import imageGridding
from tatc import utils
from tatc.schemas import Instrument, Satellite, TwoLineElements
import pandas as pd
from getTLEData import getTLEData
from tatc.analysis import collect_orbit_track, OrbitCoordinate, OrbitOutput


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

    gridded = imageGridding("49260", state84, stateECEF, times, sens, target)
    angVec = np.arange(startAng, stopAng + stepAng, stepAng)
    minGSD, maxGSD = gridded.getColorbarMinMax(angVec, altitude)

    sigmaPix = 0.5
    eMax = 25.0
 
    imageGeo = imageGeometry(noradID, state84, times, sens, target, angVec, minGSD, maxGSD, altitude, sigmaPix, eMax)

    for j in range(90,251,1):
        rng = np.random.default_rng()
        nImages = rng.integers(2, 30)
        imAngles = rng.integers(-25, 26, size=(nImages,2)).astype(np.float64)

        print()
        print(f"This run has {nImages} images from the following angles:")
        print(imAngles)
        print()


        ims = []
        imData = np.zeros((imAngles.shape[0],3))
        for k in range(imAngles.shape[0]):
            im = np.load(Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9_705km/GSDarray_matrix_705_km_offnadir_({imAngles[k,0]},{imAngles[k,1]}).npy"))
            ims.append(im)
            imData[k,:] = np.array([imAngles[k,0], imAngles[k,1], altitude])

        cost = imageGeo.getCostFunction(ims, imData, imAngles)
        print()
        print(f"The cost for {imAngles.shape[0]} images of:")
        print()
        print(imAngles)
        print()
        print(cost)
        print()

        pathCost = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_CostMaps_705km")   

        imsAngs = np.array(imAngles)
        matFileName = f'random_full_705km_angles_{j}'
        fullMatPath = pathCost / matFileName
        np.save(fullMatPath, imsAngs)
        
        imsCosts = np.array(cost)
        matFileName = f'random_full_705km_costs_{j}'   
        fullMatPath = pathCost / matFileName
        np.save(fullMatPath, imsCosts)









    
if __name__ == "__main__":
    runTest()
