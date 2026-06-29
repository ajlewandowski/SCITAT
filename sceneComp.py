"""
This file contains the class that accounts for image scene comparisons
"""

import numpy as np
from orbitPropClass import orbitPropClass
from groundCovClass import groundCovClass
from imageClass import imageClass
from SpaceTrackAPIClass import SpaceTrackAPI

# The structure of the image embedding is an array where each row is designated to a specific image:
# each row looks like: lat, lon, alt, latspan, lonspan, footprint width, footprint length, GSD width, GSD length, Overlap% of image set 9-n
#
# So if set is 5 images, overlap pct is 9-13 of im row vector
# Each image will have one image that is 100% overlap (indicates the image in row)

class sceneComp:
    def __init__(self, noradID, epochs, sens, solar, target, imCorner, capTime, user, password, caseNum):
        self.norad = noradID
        self.epochs = epochs
        self.sens = sens
        self.solar = solar
        self.target = target
        self.imCorner = imCorner
        self.capTime = capTime

        self.user = user
        self.password = password

        self.caseNum = caseNum

    def runComp(self):
        spaceDataAPI = SpaceTrackAPI(self.user, self.password)
        tle, _ = spaceDataAPI.getTLEatEpoch(self.norad, self.epochs)
        oe0, epochUTC = spaceDataAPI.tle2oe(tle[0], tle[1])
        _, _, _, tSpan = spaceDataAPI.getTimeData(epochUTC, self.epochs[1])
        spaceDataAPI.printTLE(tle)

        # Creating the orbit propagator class
        prop = orbitPropClass(epochUTC, tSpan, self.sens, incJ2 = True, incMoon = True, incSun = True)

        # Propagate for X seconds
        t, oe = prop.propagate(oe0, dtSec=self.capTime)
        cov = groundCovClass(epochUTC, self.epochs[1], t, oe, self.sens)

        # Getting latitude, longitude, and altitude
        latlonalt = cov.getLatLonHist(t)
        lat, lon, alt = latlonalt
        print(lat)
        print(lon)
        import matplotlib.pyplot as plt
        plt.plot(lon, lat)
        plt.xlabel("lon")
        plt.ylabel("lat")
        plt.show()


        fL, fR, bL, bR, mid, betas = cov.getlatlonFOVnadirHist(latlonalt)
        timeImage, instFOVs, numImages, _ = cov.imageLatLonNadir(t, self.target, fL, fR, bL, bR, mid, betas, alt)
        # diffsLat = abs(instFOVs[:,8] - self.target[0])
        # diffsLon = abs(instFOVs[:,9] - self.target[1])
        # diffsLon = np.where(diffsLon > 180, 360 - diffsLon, diffsLon)
        # diffsMin = np.sqrt(diffsLat**2 + diffsLon**2).argmin()
        # timeind = np.abs((lat - instFOVs[diffsMin,8])**2 + (lon - instFOVs[diffsMin,9])**2).argmin()

        offNadAngles = np.zeros(2)
        _, _, _, swathAlong, swathCross = cov.getlatlonFOVnadir(latlonalt[:,-1], -1)
        images = imageClass(instFOVs, offNadAngles, self.sens, numImages, timeImage, epochUTC, swathAlong, swathCross)
        imageChar, _ = images.imageFeatures()

        # images.printFeatures(imageChar, imTitle, timeind)

        # The USGS images include black in the scene so here is real comparison
        frontLat = (self.imCorner[0,0] + self.imCorner[1,0]) * 0.5
        backLat = (self.imCorner[2,0] + self.imCorner[3,0]) * 0.5
        rightLon = (self.imCorner[1,1] + self.imCorner[2,1]) * 0.5
        leftLon = (self.imCorner[0,1] + self.imCorner[3,1]) * 0.5

        print()
        print(f"Image scene comparison for Case {self.caseNum}")
        print(f"    Scene Center Latitude Error: {abs(imageChar[-1,0] - self.target[0]):.6f}°")
        print(f"    Scene Center Longitude Error: {abs(imageChar[-1,1] - self.target[1]):.6f}°")
        # print(f"    Scene Front Latitude Error: {abs(instFOVs[diffsMin,0] - frontLat):.6f}°")
        # print(f"    Scene Back Latitude Error: {abs(instFOVs[diffsMin,4] - backLat):.6f}°")
        # print(f"    Scene Left Longitude Error: {abs(instFOVs[diffsMin,7] - leftLon):.6f}°")
        # print(f"    Scene Right Longitude Error: {abs(instFOVs[diffsMin,3] - rightLon):.6f}°")
        print(f"    Solar Azimuth Error: {abs(imageChar[-1,8] - self.solar[1]):.6f}")
        print(f"    Solar Elevation Error: {abs(imageChar[-1,9] - self.solar[0]):.6f}")
        print()