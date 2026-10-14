""" This file will test and ensure the GSD component of the cost function is working correctly """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    alt = 705

    #                  FOV, Focal, Pixel #, GSD, Swath, Alt
    swath = 705*1e3*np.tan(np.deg2rad(15.0))
    sens = np.array([np.deg2rad(15.0), 0.886, 6196, swath/6196, swath, 705*1e3])
    oli = Instrument(name="OLI",field_of_regard=utils.swath_width_to_field_of_regard(705*1e3, swath),req_target_sunlit=True)
    L9 = Satellite(name="Landsat-9",
        orbit=TwoLineElements(tle=[line1,line2]), instruments=[oli])
    times = pd.date_range(start=closeTime, end=imTime[1], freq="1s", tz="UTC")
    state84 = collect_orbit_track(L9, times, coordinates = OrbitCoordinate.WGS84, sat_sunlit = True, solar_altaz = True, solar_beta = True)
    stateECEF = collect_orbit_track(L9, times, orbit_output = OrbitOutput.POSITION_VELOCITY, coordinates = OrbitCoordinate.ECEF) 

    angVec = np.array([-25.0, 25.0])

    gridded = imageGridding("49260", state84, stateECEF, times, sens, target)
    minGSD, maxGSD = gridded.getColorbarMinMax(angVec, alt)

    sigmaPix = 0.5
    eMax = 25.0

    angs = np.linspace(1.0, 25.0, 25)
    imageGeo = imageGeometry(noradID, state84, times, sens, target, angs, minGSD, maxGSD, alt, sigmaPix, eMax)

    for k in range(len(angs)):
        imAngles = np.array([[0.0,-angs[k]],[0.0,angs[k]]])

        ims = []
        imData = np.zeros((imAngles.shape[0],3))
        for j in range(imAngles.shape[0]):
            im = np.load(Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9_705km/GSDarray_matrix_705_km_offnadir_({imAngles[j,0]},{imAngles[j,1]}).npy"))
            ims.append(im)
            imData[j,:] = np.array([imAngles[j,0], imAngles[j,1], 705])

        cost = imageGeo.getCostFunction(ims, imData, imAngles)

        imsCosts = np.array(cost)
        matFileName = f'bh_ratio_error_simulation_705km_costs_{k+1}'
        pathCost = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_BHRatio_Test")   
        fullMatPath = pathCost / matFileName
        np.save(fullMatPath, imsCosts)

def analyzeBHTest():
    angs = np.linspace(1.0, 25.0, 25)
    alt = 705
    bases = alt*np.tan(np.radians(angs))
    bhRatios = 2*bases / alt

    path = Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9_BHRatio_Test")
    costs = []

    for k in range(len(angs)):
        matFileName = f'bh_ratio_error_simulation_705km_costs_{k+1}.npy'
        fullMatPath = path / matFileName
        data = np.load(fullMatPath)
        costs.append(data)
    costs = np.array(costs)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bhRatios, costs, "o-", color="crimson", linewidth=2, markersize=7, zorder=3)
    ax.set_xlabel("BH Ratio")
    ax.set_ylabel("Costs")
    ax.set_title("Cost vs. Base-to-Height Ratio")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Save the graphic and data file
    fileName = 'cost_changing_bhRatio_simulation_graphic.png'
    pathNew = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_CostResults")   
    fullPath = pathNew / fileName
    plt.savefig(fullPath)

if __name__ == "__main__":
    runTest()
    analyzeBHTest()