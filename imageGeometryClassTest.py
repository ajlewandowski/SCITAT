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

    print(f"Producing Ground Sampling Distance (GSD) Heatmaps from {startAng}° to {stopAng}° in steps of {stepAng}°.")
    gridded = imageGridding("49260", state84, stateECEF, times, sens, target)
    angVec = np.arange(startAng, stopAng + stepAng, stepAng)
    minGSD, maxGSD = gridded.getColorbarMinMax(angVec, altitude)

    pathHeatmap = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_705km")   
    imageGeo = imageGeometry(noradID, state84, times, sens, target, angVec, minGSD, maxGSD, altitude)
    imAngles = np.array([[-25.0, 25.0], [25.0, 25.0], [-25.0, -25.0]])
    im1 = np.load(Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9_705km/GSDarray_matrix_705_km_offnadir_({imAngles[0,0]},{imAngles[0,1]}).npy"))
    im2 = np.load(Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9_705km/GSDarray_matrix_705_km_offnadir_({imAngles[1,0]},{imAngles[1,1]}).npy"))
    im3 = np.load(Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9_705km/GSDarray_matrix_705_km_offnadir_({imAngles[2,0]},{imAngles[2,1]}).npy"))

    im1Data = np.array([imAngles[0,0], imAngles[0,1], altitude])
    im2Data = np.array([imAngles[1,0], imAngles[1,1], altitude])
    im3Data = np.array([imAngles[2,0], imAngles[2,1], altitude])

    cost = imageGeo.getCostFunction(im1, im1Data, im2, im2Data, im3, im3Data, imAngles)
    print()
    print(f"The cost for the following images of ({imAngles[0,0]},{imAngles[0,1]}), ({imAngles[1,0]},{imAngles[1,1]}), ({imAngles[2,0]},{imAngles[2,1]})")
    print(cost)

    overlapMean, overlapSlices, overlapRad = imageGeo.getOverlapRadius(im1, im1Data, im1, im2Data, im1, im3Data)
    print("Overlap Radius")
    print(overlapRad)

    # errGSD = imageGeo.getGSDError(overlapMean)
    # print()
    # print("Average GSD Error normalized in the overlap region")
    # print(errGSD)

    # bhRatio = imageGeo.getBHRatio(imData1, imData2, imData3)
    # print()
    # print("Average Base to Height Ratio between all the images")
    # print(bhRatio)

    # orthoAvg = imageGeo.getOrthogonality(imLoad1, imData1, imLoad2, imData2, imLoad3, imData3)
    # print()
    # print("Average dot product between images, closer to zero more orthogonal")
    # print(orthoAvg)

    # errPix = imageGeo.getUsablePix(imLoad1)
    # print()
    # print("Get usable pixels in the image that are sub-1.25 times the min GSD")
    # print(errPix)

    costMap = imageGeo.computeMVSCostMap(overlapSlices, imAngles, 0.5, 35.0)
    print()
    print("MVS cost map")
    print(costMap)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(costMap, origin='lower', cmap='coolwarm', interpolation='nearest')
    cbar = fig.colorbar(im)
    cbar.set_label('MVS Cost (m)')
    ax.set_title(f'Cost map')
    plt.show()









    
if __name__ == "__main__":
    runTest()
