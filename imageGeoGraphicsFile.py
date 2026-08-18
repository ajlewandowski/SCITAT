# This file is going to create graphics about the trends in the cost map/image geometry study


import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt
from imageGeometry import imageGeometry
from imageGridding import imageGridding
from tatc import utils
from tatc.schemas import Instrument, Satellite, TwoLineElements
import pandas as pd
from getTLEData import getTLEData
from tatc.analysis import collect_orbit_track, OrbitCoordinate, OrbitOutput

def runCostVsNumImages():

    folder = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_CostMaps_705km")   

    IMG_EXT = {".png"}

    # group 1 = the two-digit index, group 2 = the long decimal number
    pattern = re.compile(r"costmap_(\d{2})_images_705_km_(\d+\.\d+)")

    ids, values = [], []
    for f in sorted(folder.iterdir()):
        if not f.is_file() or f.suffix.lower() not in IMG_EXT:
            continue
        m = pattern.search(f.name)
        if m is None:
            print(f"skipped (no match): {f.name}")
            continue
        ids.append(int(m.group(1)))
        values.append(float(m.group(2)))

    ids = np.array(ids, dtype=int)
    values = np.array(values, dtype=float)

    print(f"{len(ids)} graphics found")

    # per-ID means
    unique_ids = np.unique(ids)
    mean_values = np.array([values[ids == i].mean() for i in unique_ids])
    counts = np.array([(ids == i).sum() for i in unique_ids])

    fig, ax = plt.subplots(figsize=(8, 5))

    # raw points (no connecting line — several values can share one ID)
    ax.scatter(ids, values, s=25, alpha=0.5, label="Individual Costs")

    # the average on top
    ax.plot(unique_ids, mean_values, "o-", color="crimson",
            linewidth=2, markersize=7, zorder=3, label="Mean per Number of Images")

    ax.set_xlabel("Number of Images")
    ax.set_ylabel("Cost")
    ax.set_title("Cost vs. Number of Images in MVS Set")
    ax.set_xticks(unique_ids)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    # Save the graphic and data file
    fileName = 'cost_versus_numberofimages_graphic.png'
    pathNew = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_CostResults")   
    fullPath = pathNew / fileName
    plt.savefig(fullPath)
    print('Figure saved as cost_versus_numberofimages_graphic.png')


def runBigSymmetricSim():

    sigmaPix = 0.5
    eMax = 25.0

    imAngles = np.array([[0.0,0.0],[0.0,5.0],[0.0,-5.0],[5.0,0.0],[-5.0,0.0],
                      [5.0,5.0],[-5.0,5.0],[-5.0,-5.0],[5.0,-5.0],
                      [0.0,15.0],[0.0,-15.0],[15.0,0.0],[-15.0,0.0],
                      [15.0,15.0],[-15.0,15.0],[-15.0,-15.0],[15.0,-15.0],
                      [0.0,25.0],[0.0,-25.0],[25.0,0.0],[-25.0,0.0],
                      [25.0,25.0],[-25.0,25.0],[-25.0,-25.0],[25.0,-25.0]])

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

    overlapMean, overlapSlices, overlapRad = imageGeo.getOverlapRadius(ims, imData)
    costMap = imageGeo.computeMVSCostMap(overlapSlices, imAngles, sigmaPix, eMax)
    pathCost = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_CostMaps_705km_Symmetry")   
    imageGeo.plotCostMap(costMap, imAngles, sigmaPix, eMax, pathCost, cost)
    print()

def runRandomSymmetryTest():

    sigmaPix = 0.5
    eMax = 25.0

    for j in range(25, 30, 1):
        imAngles = np.array([[0.0,0.0],[5.0,5.0],[-5.0,5.0],[-5.0,-5.0],[5.0,-5.0],
                            [15.0,15.0],[-15.0,15.0],[-15.0,-15.0],[15.0,-15.0],
                            [25.0,25.0],[-25.0,25.0],[-25.0,-25.0],[25.0,-25.0]])

        imSets = []
        costs = []

        # This is the actual code to run symmetry test, rest is basic stuff
        totSize = imAngles.size

        for i in range(0, totSize+1, 5):
            rng = np.random.default_rng()

            imCopy = imAngles.copy()
            flatIDX = rng.choice(totSize, size=i, replace=False)

            imCopy.flat[flatIDX] = rng.integers(-25, 26, size=i).astype(np.float64)

            print()
            print(f"This run has {i} angle values randomized")
            print()

            ims = []
            imData = np.zeros((imCopy.shape[0],3))
            for k in range(imCopy.shape[0]):
                im = np.load(Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9_705km/GSDarray_matrix_705_km_offnadir_({imCopy[k,0]},{imCopy[k,1]}).npy"))
                ims.append(im)
                imData[k,:] = np.array([imCopy[k,0], imCopy[k,1], altitude])

            cost = imageGeo.getCostFunction(ims, imData, imCopy)
            saveCost = np.array([cost, i])

            costs.append(saveCost)
            imSets.append(imCopy)

        numRun = j

        imsAngs = np.array(imSets)
        matFileName = f'random_symmetry_{imAngles.shape[0]}_images_705km_angles_{numRun}'
        pathCost = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_CostMaps_705km_Symmetry")   
        fullMatPath = pathCost / matFileName
        np.save(fullMatPath, imsAngs)

        imsCosts = np.array(costs)
        matFileName = f'random_symmetry_{imAngles.shape[0]}_images_705km_costs_{numRun}'
        pathCost = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_CostMaps_705km_Symmetry")   
        fullMatPath = pathCost / matFileName
        np.save(fullMatPath, imsCosts)

        print(imsCosts)

def analyzeSymmetrySim():

    folder = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_CostMaps_705km_Symmetry")   

    data = []
    for j in range(1, 26, 1):
        fileName = f"random_symmetry_13_images_705km_costs_{j}.npy"
        path = folder / fileName
        dataMat = np.load(path)
        data.append(dataMat)

    data = np.vstack(data)
    angle = data[:,1]
    costs = data[:,0]

    keys = np.unique(angle)
    means = np.array([costs[angle == k].mean() for k in keys])
    stds = np.array([costs[angle == k].std(ddof=1) for k in keys])
    counts = np.array([np.count_nonzero(angle == k) for k in keys])

    fig, ax = plt.subplots(figsize=(8, 5))
 
    x_scatter = angle
    ax.scatter(x_scatter, costs, s=36, alpha=0.67, color="darkblue", label=f"All Points (N={len(costs)})")
    ax.errorbar(keys, means, yerr=stds, color="tab:red", marker="o", capsize=4, linewidth=2, zorder=3, label="Mean ± 1 SD")
    ax.plot(keys, means, color="tab:red", marker="o", linewidth=2, zorder=3, label="Mean")
    ax.set_xlabel("Angles Randomized")
    ax.set_ylabel("Cost")
    ax.set_xticks(keys)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"Cost of Symmetric and Off-Symmetric Cases for {j} Trials")
    fig.tight_layout()


    # Save the graphic and data file
    fileName = 'Off_Symmetry_Randomization' + fileName[25:29] + 'km_offnadir.png'
    pathNew = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_CostResults")   
    fullPath = pathNew / fileName
    plt.savefig(fullPath)
    print('Figure saved as Off_Symmetry_Randomization' + fileName[25:30] + 'km_offnadir.png')



    
if __name__ == "__main__":
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
    
    runCostVsNumImages()
    # runBigSymmetricSim()
    # runRandomSymmetryTest()
    analyzeSymmetrySim()