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
