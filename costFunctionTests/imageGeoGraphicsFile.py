# This file is going to create graphics about the trends in the cost map/image geometry study
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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

    # folder = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_CostMaps_705km")   

    # IMG_EXT = {".png"}

    # # group 1 = the two-digit index, group 2 = the long decimal number
    # pattern = re.compile(r"costmap_(\d{2})_images_705_km_(\d+\.\d+)")

    # ids, values = [], []
    # for f in sorted(folder.iterdir()):
    #     if not f.is_file() or f.suffix.lower() not in IMG_EXT:
    #         continue
    #     m = pattern.search(f.name)
    #     if m is None:
    #         print(f"skipped (no match): {f.name}")
    #         continue
    #     ids.append(int(m.group(1)))
    #     values.append(float(m.group(2)))

    # ids = np.array(ids, dtype=int)
    # values = np.array(values, dtype=float)

    # print(f"{len(ids)} graphics found")

    # # per-ID means
    # unique_ids = np.unique(ids)
    # mean_values = np.array([values[ids == i].mean() for i in unique_ids])
    # counts = np.array([(ids == i).sum() for i in unique_ids])

    folder = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_CostMaps_705km")   
    
    data = []
    imNum = []
    for j in range(1, 251, 1):
        fileName = f"random_full_705km_costs_{j}.npy"
        path = folder / fileName
        dataMat = np.load(path)
        data.append(dataMat)

        fileName = f"random_full_705km_angles_{j}.npy"
        path = folder / fileName
        dataMat = np.load(path)
        row, col = dataMat.shape
        imNum.append(row)

    costs = np.concatenate([np.atleast_1d(d).ravel() for d in data])
    nums  = np.concatenate([np.full(np.atleast_1d(d).size, n) for d, n in zip(data, imNum)])

    # Group by number of images and take the mean of each group
    uniq  = np.unique(nums)
    means = np.array([costs[nums == n].mean() for n in uniq])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(nums, costs, color="tab:blue", alpha=0.6, label="Individual Costs")
    ax.plot(uniq, means, "o-", color="crimson", lw=2, markersize=8,
            label="Mean per Number of Images")

    ax.set_xticks(uniq)
    ax.set_xlabel("Number of Images")
    ax.set_ylabel("Cost")
    ax.set_title("Cost vs. Number of Images in MVS Set")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

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
    pathCost = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_CostResults")   
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

    mask = costs < 0.4
    costs = costs[mask]
    angle = angle[mask]

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

def runFullQuadrantSim():

    for j in range(51):
        rng = np.random.default_rng()
        nImages = 5
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
        print(cost)
        print()

        pathCost = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_CostMaps_705km")   

        imsAngs = np.array(imAngles)
        matFileName = f'random_full_705km_angles_{j+1}'
        fullMatPath = pathCost / matFileName
        np.save(fullMatPath, imsAngs)
        
        imsCosts = np.array(cost)
        matFileName = f'random_full_705km_costs_{j+1}'   
        fullMatPath = pathCost / matFileName
        np.save(fullMatPath, imsCosts)


def runThreeQuadrantSim():

    for j in range(1, 51, 1):
        rng = np.random.default_rng()
        nImages = 5
        imAngles = rng.integers(-25, 26, size=(nImages,2)).astype(np.float64)

        for i in range(imAngles.shape[0]):
            if imAngles[i,0] < 0 and imAngles[i,1] < 0:
                ind = rng.integers(0,3)
                if ind == 0:
                    imAngles[i,1] = -imAngles[i,1]
                    imAngles[i,0] = -imAngles[i,0]
                elif ind == 1:
                    imAngles[i,1] = -imAngles[i,1]
                else:
                    imAngles[i,0] = -imAngles[i,0]

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
        print(cost)
        print()

        pathCost = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_CostMaps_705km_ThreeQuad")   

        imsAngs = np.array(imAngles)
        matFileName = f'random_three_quadrant_images_705km_angles_{j}'
        fullMatPath = pathCost / matFileName
        np.save(fullMatPath, imsAngs)

        imsCosts = np.array(cost)
        matFileName = f'random_three_quadrant_images_705km_costs_{j}'   
        fullMatPath = pathCost / matFileName
        np.save(fullMatPath, imsCosts)
        
        print()

def runTwoQuadrantSim():
    for j in range(1, 51, 1):
        rng = np.random.default_rng()
        nImages = 5
        imAngles = rng.integers(-25, 26, size=(nImages,2)).astype(np.float64)

        for i in range(imAngles.shape[0]):
            if imAngles[i,0] < 0 and imAngles[i,1] < 0:
                ind = rng.integers(0,2)
                if ind == 0:
                    imAngles[i,1] = -imAngles[i,1]
                    imAngles[i,0] = -imAngles[i,0]
                elif ind == 1:
                    imAngles[i,0] = -imAngles[i,0]


            if imAngles[i,0] < 0 and imAngles[i,1] > 0:
                ind = rng.integers(0,2)
                if ind == 0:
                    imAngles[i,0] = -imAngles[i,0]
                elif ind == 1:
                    imAngles[i,1] = -imAngles[i,1]
                    imAngles[i,0] = -imAngles[i,0]

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
        print(cost)
        print()

        pathCost = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_CostMaps_705km_TwoQuad")   

        imsAngs = np.array(imAngles)
        matFileName = f'random_two_quadrant_705km_angles_{j}'
        fullMatPath = pathCost / matFileName
        np.save(fullMatPath, imsAngs)

        imsCosts = np.array(cost)
        matFileName = f'random_two_quadrant_705km_costs_{j}'   
        fullMatPath = pathCost / matFileName
        np.save(fullMatPath, imsCosts)
        
        print()

def runOneQuadrantSim():
    for j in range(1, 51, 1):
        rng = np.random.default_rng()
        nImages = 5
        imAngles = rng.integers(0, 26, size=(nImages,2)).astype(np.float64)

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
        print(cost)
        print()

        pathCost = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_CostMaps_705km_OneQuad")   

        imsAngs = np.array(imAngles)
        matFileName = f'random_one_quadrant_705km_angles_{j}'
        fullMatPath = pathCost / matFileName
        np.save(fullMatPath, imsAngs)

        imsCosts = np.array(cost)
        matFileName = f'random_one_quadrant_705km_costs_{j}'   
        fullMatPath = pathCost / matFileName
        np.save(fullMatPath, imsCosts)
        
        print()

def analyzeQuadrantSim():
    pathFull = Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9_CostMaps_705km")
    fullCost = []
    for j in range(1, 51, 1):
        for f in pathFull.glob(f"*costs_{j}.*"):
            data = np.load(f)
            fullCost.append(data)
    full = np.array(fullCost)
    fullMean = np.mean(full)

    pathQuad1 = Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9_CostMaps_705km_OneQuad")
    quad1cost = []
    for j in range(1, 51, 1):
        for f in pathQuad1.glob(f"*costs_{j}.*"):
            data = np.load(f)
            quad1cost.append(data)
    quad1 = np.array(quad1cost)
    quad1Mean = np.mean(quad1)

    pathQuad2 = Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9_CostMaps_705km_TwoQuad")
    quad2cost = []
    for j in range(1, 51, 1):
        for f in pathQuad2.glob(f"*costs_{j}.*"):
            data = np.load(f)
            quad2cost.append(data)
    quad2 = np.array(quad2cost)
    quad2Mean = np.mean(quad2)

    pathQuad3 = Path(f"D:/Users/al62352/ResolutionHeatmaps/LS9_CostMaps_705km_ThreeQuad")
    quad3cost = []
    for j in range(1, 51, 1):
        for f in pathQuad3.glob(f"*costs_{j}.*"):
            data = np.load(f)
            quad3cost.append(data)
    quad3 = np.array(quad3cost)
    quad3Mean = np.mean(quad3)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(np.ones(len(full)), full, s=25, alpha=0.5, label="All-Quadrant")
    ax.scatter(2.0*np.ones(len(quad3)), quad3, s=25, alpha=0.5, label="Three-Quadrant")
    ax.scatter(3.0*np.ones(len(quad2)), quad2, s=25, alpha=0.5, label="Two-Quadrant")
    ax.scatter(4.0*np.ones(len(quad1)), quad1, s=25, alpha=0.5, label="One-Quadrant")

    ax.plot(np.array([1.0, 2.0, 3.0, 4.0]), np.array([fullMean, quad3Mean, quad2Mean, quad1Mean]), "o-", color="crimson",
            linewidth=2, markersize=7, zorder=3, label="Mean")
    ax.set_xlabel("Image Quadrants")
    ax.set_ylabel("Cost")
    ax.set_title("Cost vs. Quadrants Covered in image")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    # Save the graphic and data file
    fileName = 'cost_quadrant_simulation_graphic.png'
    pathNew = Path("D:/Users/al62352/ResolutionHeatmaps/LS9_CostResults")   
    fullPath = pathNew / fileName
    plt.savefig(fullPath)
    print('Figure saved as cost_quadrant_simulation_graphic.png')

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

    # runFullQuadrantSim()
    # runThreeQuadrantSim()
    # runTwoQuadrantSim()
    # runOneQuadrantSim()
    # analyzeQuadrantSim()
    
    # runCostVsNumImages()
    # runBigSymmetricSim()
    # runRandomSymmetryTest()
    # analyzeSymmetrySim()