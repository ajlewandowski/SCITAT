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

    #                  FOV, Focal, Pixel #, GSD (m), Swath (m), Alt (m)
    swath = 14000
    pixNum = 48096
    sens = np.array([np.deg2rad(1.29372), 25.2, pixNum, swath/pixNum, swath, alt*1e3])
    vec = np.zeros(10)
  
    #### Finding the maximum satellite look angle given the altitude, and studying 75% of it
    maxLook = np.arcsin(aeKM / (aeKM + alt))*(180/np.pi)
    pctMaxLook = 46
    print(f"The max look angle for altitude {alt} km: {pctMaxLook}°")

    angVecsPos = np.arange(0, pctMaxLook + 1, 1)
    angVecsNeg = -angVecsPos

    # print(f"Producing Ground Sampling Distance (GSD) Heatmaps from {startAng}° to {stopAng}° in steps of {stepAng}°.")
    gridded = imageGridding("49070", vec, vec, vec, sens, vec)
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