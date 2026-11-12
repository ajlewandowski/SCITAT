""" This file will test and ensure the overlap component of the cost function is working correctly """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from imageGeometryClass import imageGeometry
from pathlib import Path
import matplotlib.pyplot as plt
from imageGriddingClass import imageGridding
from tatc import utils
from tatc.schemas import Instrument, Satellite, TwoLineElements
import pandas as pd
from processingTLEs.getTLEData import getTLEData
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

    target = np.array([30.30617, -97.88121])

    sens = np.array([np.deg2rad(15.0), 0.886, 6196, 30, 185000, 705000])    
    oli = Instrument(name="OLI",field_of_regard=utils.swath_width_to_field_of_regard(705000, 185000),req_target_sunlit=True)
    L9 = Satellite(name="Landsat-9",
        orbit=TwoLineElements(tle=[line1,line2]), instruments=[oli])
    times = pd.date_range(start=closeTime, end=imTime[1], freq="1s", tz="UTC")
    state84 = collect_orbit_track(L9, times, coordinates = OrbitCoordinate.WGS84, sat_sunlit = True, solar_altaz = True, solar_beta = True)
    stateECEF = collect_orbit_track(L9, times, orbit_output = OrbitOutput.POSITION_VELOCITY, coordinates = OrbitCoordinate.ECEF) 
    gridded = imageGridding("49260", state84, stateECEF, times, sens, target)

    startAng = 1.0
    stepAng = 1.0
    stopAng = 25.0
    altitude = 705

    angVec = np.arange(startAng, stopAng + stepAng, stepAng)
    minGSD, maxGSD = gridded.getColorbarMinMax(angVec, altitude)
    sigmaPix = 0.5
    eMax = 25.0
    imageGeo = imageGeometry(noradID, state84, times, sens, target, angVec, minGSD, maxGSD, altitude, sigmaPix, eMax)

    for k in range(len(angVec)):
        ang = angVec[k]
        imAngles = np.array([[0.0,0.0], [ang,ang], [ang,-ang], [-ang,ang], [-ang,-ang]])

        ims = []
        imData = np.zeros((imAngles.shape[0],3))
        for j in range(imAngles.shape[0]):
            im = np.load(Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9/LS9_705km/GSDarray_matrix_705_km_offnadir_({imAngles[j,0]},{imAngles[j,1]}).npy"))
            ims.append(im)
            imData[j,:] = np.array([imAngles[j,0], imAngles[j,1], 705])

        cost = imageGeo.getCostFunction(ims, imData, imAngles)

        imsCosts = np.array(cost)
        matFileName = f'overlap_error_simulation_705km_costs_{k+1}'
        pathCost = Path("D:/Users/al62352/ResolutionHeatmaps/LS9/LS9_Overlap_Test")   
        fullMatPath = pathCost / matFileName
        np.save(fullMatPath, imsCosts)

def analyzeOverlapTest():
    startAng = 12.0
    stepAng = 1.0
    stopAng = 25.0
    altitude = 705
    
    angVec = np.arange(startAng, stopAng + stepAng, stepAng)

    path = Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9/LS9_Overlap_Test")
    costs = []

    for k in range(len(angVec)):
        matFileName = f'overlap_error_simulation_705km_costs_{k+12}.npy'
        fullMatPath = path / matFileName
        data = np.load(fullMatPath)
        costs.append(data)
    costs = np.array(costs)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(angVec, costs, "o-", color="crimson", linewidth=2, markersize=7, zorder=3)
    ax.set_xlabel("Angles for the overlap set")
    ax.set_ylabel("Costs")
    ax.set_title("Cost vs. Decreasing Overlap Pct")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Save the graphic and data file
    fileName = 'cost_changing_overlap_simulation_graphic.png'
    pathNew = Path("D:/Users/al62352/ResolutionHeatmaps/LS9/LS9_CostResults")   
    fullPath = pathNew / fileName
    plt.savefig(fullPath)


if __name__ == "__main__":
    # runTest()
    analyzeOverlapTest()