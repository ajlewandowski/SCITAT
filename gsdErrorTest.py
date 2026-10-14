""" This file will test and ensure the GSD component of the cost function is working correctly """

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

    target = np.array([30.30617, -97.88121])

    alts = np.array([400, 500, 600, 700, 800, 900, 1000])

    for k in range(len(alts)):
        #                  FOV, Focal, Pixel #, GSD, Swath, Alt
        swath = alts[k]*1e3*np.tan(np.deg2rad(15.0))
        sens = np.array([np.deg2rad(15.0), 0.886, 6196, swath/6196, swath, alts[k]*1e3])
        oli = Instrument(name="OLI",field_of_regard=utils.swath_width_to_field_of_regard(alts[k]*1e3, swath),req_target_sunlit=True)
        L9 = Satellite(name="Landsat-9",
            orbit=TwoLineElements(tle=[line1,line2]), instruments=[oli])
        times = pd.date_range(start=closeTime, end=imTime[1], freq="1s", tz="UTC")
        state84 = collect_orbit_track(L9, times, coordinates = OrbitCoordinate.WGS84, sat_sunlit = True, solar_altaz = True, solar_beta = True)
        stateECEF = collect_orbit_track(L9, times, orbit_output = OrbitOutput.POSITION_VELOCITY, coordinates = OrbitCoordinate.ECEF) 

        angVec = np.array([-25.0, -12.5, 0.0, 12.5, 25.0])
        altitude = alts[k]

        gridded = imageGridding("49260", state84, stateECEF, times, sens, target)
        minGSD, maxGSD = gridded.getColorbarMinMax(angVec, altitude)

        sigmaPix = 0.5
        eMax = 25.0
    
        imageGeo = imageGeometry(noradID, state84, times, sens, target, angVec, minGSD, maxGSD, altitude, sigmaPix, eMax)

        imAngles = np.array([[-25.0,-25.0],[-25.0,-12.5],[-25.0,0.0],[-25.0,12.5],[-25.0,25.0],
                             [-12.5,-25.0],[-12.5,-12.5],[-12.5,0.0],[-12.5,12.5],[-12.5,25.0],
                             [0.0,-25.0],[0.0,-12.5],[0.0,0.0],[0.0,12.5],[0.0,25.0],
                             [12.5,-25.0],[12.5,-12.5],[12.5,0.0],[12.5,12.5],[12.5,25.0],
                             [25.0,-25.0],[25.0,-12.5],[25.0,0.0],[25.0,12.5],[25.0,25.0],])

        ims = []
        imData = np.zeros((imAngles.shape[0],3))
        for j in range(imAngles.shape[0]):
            im = np.load(Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9_{alts[k]}km/GSDarray_matrix_{alts[k]}_km_offnadir_({imAngles[j,0]},{imAngles[j,1]}).npy"))
            ims.append(im)
            imData[j,:] = np.array([imAngles[j,0], imAngles[j,1], altitude])

        cost = imageGeo.getCostFunction(ims, imData, imAngles)

        imsCosts = np.array(cost)
        matFileName = f'gsd_error_simulation_{alts[k]}km_costs'
        pathCost = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_Alts_GSD")   
        fullMatPath = pathCost / matFileName
        np.save(fullMatPath, imsCosts)

def analyzeGSDTest():
    alts = np.array([400, 500, 600, 700, 800, 900, 1000])

    swath = alts*1e3*np.tan(np.deg2rad(15.0))
    gsds = swath/6196

    path = Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9_Alts_GSD")
    costs = []

    for k in range(len(alts)):
        matFileName = f'gsd_error_simulation_{alts[k]}km_costs.npy'
        fullMatPath = path / matFileName
        data = np.load(fullMatPath)
        costs.append(data)
    costs = np.array(costs)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gsds, costs, "o-", color="crimson", linewidth=2, markersize=7, zorder=3)
    ax.set_xlabel("Ground Sampling Distances")
    ax.set_ylabel("Costs")
    ax.set_title("Cost vs. Ground Sampling Distance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Save the graphic and data file
    fileName = 'cost_changing_meanGSD_simulation_graphic.png'
    pathNew = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_CostResults")   
    fullPath = pathNew / fileName
    plt.savefig(fullPath)

    


if __name__ == "__main__":
    runTest()
    analyzeGSDTest()