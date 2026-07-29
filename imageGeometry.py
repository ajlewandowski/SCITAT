"""
This file contains the class that accounts for image geometry analysis and optimization
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# This class will import system parameters and produce a resolution "heat map"

class imageGeometry:
    def __init__(self, noradID, state84, times, sens, target, angVec, minGSD, maxGSD, altitude):
        self.norad = noradID
        self.state84 = state84
        self.sens = sens
        self.target = target
        self.times = times
        self.IFOVbase = sens[0] / sens[2]
        self.angVec = angVec
        self.minGSD = minGSD
        self.maxGSD = maxGSD
        self.alt = altitude

        self.wgsSMA = 6378137.0
        self.wgsf = 1 / 298.257223563
        self.ae = 6378137

        self.targetBHCanopy = 0.25
        self.targetBH = 1.0
        self.targetOverlap = 75.0
        self.pixGSDlimFactor = 1.25

    def getCostFunction(self, im1, im2):

        cost = 1.0

        return cost

    def getOverlapPct(self, im1, im2):

        overlapPct = 1.0

        return overlapPct

    def getBHRatio(self, im1, im2):

        BHRatio = 1.0

        return BHRatio

    def getOrthogonality(self, im1, im2):

        sumDotProds = 1.0

        return sumDotProds

    def getGSDError(self, im1, im2):

        errGSD = 1.0

        return errGSD

    def getUsablePix(self, im1, im2):

        errPix = 1.0

        return errPix

    def computeMVSCostMap(gsdMats, offNadAngles, sigmaPix, eMax):
        """
        Computes MVS cost map where unusable pixels have been set to np.nan.
        All input matrices must still be the same dimensional shape (e.g., 1000x1000).
        """
        # 1. 3D Vectorization
        gsdCube = np.stack(gsdMats, axis=0)
        
        # 2. Dynamic View Counting
        # Count how many valid (non-NaN) views exist FOR EACH PIXEL
        validViews = np.sum(~np.isnan(gsdCube), axis=0)
        
        # Create a mask for pixels that don't meet the stereo requirement (< 2 views)
        invalidStereoMask = validViews < 2
        
        # 3. Base Uncertainty (Ignoring NaNs)
        # Convert NaNs to infinity temporarily just to avoid math warnings, 
        # though np.nansum handles NaNs automatically.
        safeGSDCube = np.where(np.isnan(gsdCube), np.inf, np.clip(gsdCube, a_min=1e-6, a_max=None))
        wCube = 1.0 / ((sigmaPix * safeGSDCube) ** 2)
        
        # Use np.nansum to add up the weights, completely ignoring the "cut" NaN pixels
        wSumMat = np.nansum(wCube, axis=0)
        
        # Prevent divide-by-zero where weight_sum is 0 (invalid pixels)
        with np.errstate(divide='ignore'):
            baseErrMap = np.where(wSumMat > 0, np.sqrt(1.0 / wSumMat), np.inf)
        
        # 4. Geometry Penalty (Assumes global pass geometry)
        convergenceDeg = np.max(offNadAngles) - np.min(offNadAngles)
        theta = np.radians(np.clip(abs(convergenceDeg), 1.0, 179.0))
        geometricPenalty = 1.0 / np.sin(theta)
        
        # 5. Redundancy Penalty (Now calculated PER PIXEL)
        alpha = 0.5
        # Pixels with 5 views get a much better score than pixels with only 2 views!
        redundancyPenalty = np.where(
            validViews >= 2, 
            1.0 / (1.0 - np.exp(-alpha * (validViews - 1))), 
            np.inf
        )
        
        # 6. Total Error and Normalization
        totalErrorMap = baseErrMap * geometricPenalty * redundancyPenalty
        costMap = 1.0 - np.exp(-((totalErrorMap / eMax) ** 2))
        
        # Force any pixel with < 2 views to immediately fail with a cost of 1.0
        costMap[invalidStereoMask] = 1.0
        
        return np.clip(costMap, 0.0, 1.0)
