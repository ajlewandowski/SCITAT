"""
This file contains the class that accounts for image gridding and creating the heat maps
"""

import numpy as np

# This class will import system parameters and produce a resolution "heat map"

class imageGridding:
    def __init__(self, noradID, latlon, sens, solar, target, imCorner, offNadAngle):
        self.norad = noradID
        self.lat, self.lon = latlon
        self.sens = sens
        self.solar = solar
        self.target = target
        self.imCorner = imCorner
        self.offNad = offNadAngle

    