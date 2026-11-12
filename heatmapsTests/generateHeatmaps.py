"""
Gridding and resolution heat map test file for changing altitudes in the cost trials
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from imageGriddingClass import imageGridding
from tatc import utils
from tatc.schemas import Instrument, Satellite, TwoLineElements
import pandas as pd
from processingTLEs.getTLEData import getTLEData
from tatc.analysis import collect_orbit_track, OrbitCoordinate, OrbitOutput

wgsSMA = 6378137.0
wgsf = 1 / 298.257223563
ae = 6378137
aeKM = 6378.137

def runTest():
    print()
    print("###################################################################################################")
    print()

    noradID = 00000
        
    alt = 400
    
    #                  FOV, Focal, Pixel #, GSD (m), Swath (m), Alt (m)
    swath = 5500
    pixNum = 2560
    sens = np.array([np.deg2rad(0.7878045564), 3.6, pixNum, swath/pixNum, swath, alt*1e3])
    vec = np.zeros(10) 

    #### Finding the maximum satellite look angle given the altitude, and studying 75% of it
    maxLook = np.arcsin(aeKM / (aeKM + alt))*(180/np.pi)
    pctMaxLook = 30 # round(0.75*maxLook)
    print(f"The max look angle for altitude {alt} km: {pctMaxLook}°")

    angVecsPos = np.arange(0, pctMaxLook + 1, 1)
    angVecsNeg = -angVecsPos

    altitude = alt

    # print(f"Producing Ground Sampling Distance (GSD) Heatmaps from {startAng}° to {stopAng}° in steps of {stepAng}°.")
    gridded = imageGridding("49260", vec, vec, vec, sens, vec)
    minGSD, maxGSD = gridded.getColorbarMinMax(angVecsNeg, altitude)
    
    pathHeatmap = Path(f"/exports/nfs_share/Users/al62352/ResolutionHeatmaps/SkySat/SS_400km")

    for i in range(len(angVecsPos)):
        for j in range(len(angVecsPos)):
            fig = gridded.getGSDArray(altitude, angVecsPos[i], angVecsPos[j], minGSD, maxGSD, True, True, pathHeatmap)
            plt.close()

    for i in range(len(angVecsNeg)):
        for j in range(len(angVecsPos)):
            fig = gridded.getGSDArray(altitude, angVecsNeg[i], angVecsPos[j], minGSD, maxGSD, True, True, pathHeatmap)
            plt.close()

    for i in range(len(angVecsPos)):
        for j in range(len(angVecsNeg)):
            fig = gridded.getGSDArray(altitude, angVecsPos[i], angVecsNeg[j], minGSD, maxGSD, True, True, pathHeatmap)
            plt.close()

    for i in range(len(angVecsNeg)):
        for j in range(len(angVecsNeg)):
            fig = gridded.getGSDArray(altitude, angVecsNeg[i], angVecsNeg[j], minGSD, maxGSD, True, True, pathHeatmap)
            plt.close()


    print()








if __name__ == "__main__":
    runTest()