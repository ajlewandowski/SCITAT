""" Generating the heatmaps for Pleadis Neo """
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
aeKM = 6378.137

def runTest():
    print()
    print("###################################################################################################")
    print()

    alt = 620

    #                  FOV, Focal, Pixel #, GSD, Swath, Alt
    swath = 14
    gsd = 0.30
    sens = np.array([np.deg2rad(1.29378), 25.2, swath*1e3/gsd, gsd, swath, alt*1e3])
    target = np.array([30.30617, -97.88121])
    vec = np.ones(1000)
  
    #### Finding the maximum satellite look angle given the altitude, and studying 75% of it
    maxLook = np.arcsin(aeKM / (aeKM + alt))*(180/np.pi)
    fracMaxLook = round(0.75*maxLook)
    maxLookPN = 52
    pctMaxLook = maxLookPN
    print(f"The max look angle for altitude {alt} km: {pctMaxLook}°")

    angVecsPos = np.arange(0, pctMaxLook + 1, 1)
    angVecsNeg = -angVecsPos

    # print(f"Producing Ground Sampling Distance (GSD) Heatmaps from {startAng}° to {stopAng}° in steps of {stepAng}°.")
    gridded = imageGridding("49070", vec, vec, vec, sens, target)
    minGSD, maxGSD = gridded.getColorbarMinMax(angVecsNeg, alt)
    
    pathHeatmap = Path(f"/exports/nfs_share/Users/al62352/ResolutionHeatmaps/PleadisNeo/PN_620km")

    for i in range(len(angVecsPos)):
        for j in range(len(angVecsPos)):
            fig = gridded.getGSDArray(alt, angVecsPos[i], angVecsPos[j], minGSD, maxGSD, True, True, pathHeatmap)
            plt.close()

    for i in range(len(angVecsNeg)):
        for j in range(len(angVecsPos)):
            fig = gridded.getGSDArray(alt, angVecsNeg[i], angVecsPos[j], minGSD, maxGSD, True, True, pathHeatmap)
            plt.close()

    for i in range(len(angVecsPos)):
        for j in range(len(angVecsNeg)):
            fig = gridded.getGSDArray(alt, angVecsPos[i], angVecsNeg[j], minGSD, maxGSD, True, True, pathHeatmap)
            plt.close()

    for i in range(len(angVecsNeg)):
        for j in range(len(angVecsNeg)):
            fig = gridded.getGSDArray(alt, angVecsNeg[i], angVecsNeg[j], minGSD, maxGSD, True, True, pathHeatmap)
            plt.close()


    print()

if __name__ == "__main__":
    runTest()