"""
This file contains the class that accounts for image embeddings and imaging data
"""

import numpy as np
from datetime import timedelta, datetime
import spiceypy as spice
# The structure of the image embedding is an array where each row is designated to a specific image:
# each row looks like: lat, lon, alt, latspan, lonspan, footprint width, footprint length, GSD width, GSD length, Overlap% of image set 9-n
#
# So if set is 5 images, overlap pct is 9-13 of im row vector
# Each image will have one image that is 100% overlap (indicates the image in row)

class imageClass:
    def __init__(self, instFOVs, offNadirAngles, sens, numImages, timeImage, epochUTC, swathAlong, swathCross):
        self.loadSpice()

        self.FOVs = instFOVs
        self.sens = sens
        self.numImage = numImages
        self.epoch = epochUTC
        self.t = timeImage
        self.offNad = offNadirAngles

        self.swathAlong = swathAlong
        self.swathCross = swathCross
    
    def loadSpice(self):
        from pathlib import Path
        import spiceypy as spice
        if spice.ktotal('ALL') == 0:
            metakernel = Path.home() / ".spice_kernels" / "orbit_prop.tm"
            if not metakernel.exists():
                raise FileNotFoundError(f"SPICE kernels not found at {metakernel}")
            spice.furnsh(str(metakernel))

    def imageFeatures(self):
        imOverlaps = self.getOverlapPCTnadir()

        featTitleList = ['Image Center Latitude (°)',
                          'Image Center Longitude (°)',
                          'Image Altitude (km)',
                          'Image Time UTC',
                          'Swath - Along Track (km)',
                          'Swath - Cross Track (km)',
                          'GSD - Along Track (m)',
                          'GSD - Cross Track (m)',
                          'Solar Azimuth - Satellite (°)',
                          'Solar Elevation - Satellite (°)',
                          'Solar Azimuth - Ground POI (°)',
                          'Solar Elevation - Ground POI (°)',
                          'Off-Nadir Angle  (°)',
                          'Off-Nadir Azimuth  (°)',
                          'Image Range (km)']
        
        imageVecLength = (self.numImage) + len(featTitleList)
        image = np.empty((self.numImage, imageVecLength), dtype=object)
        
        for k in range(self.numImage):
            image[k,0] = self.FOVs[k,8]
            image[k,1] = self.FOVs[k,9]
            image[k,2] = self.FOVs[k,10]

            imTime = self.getImageTime(k)
            image[k,3] = imTime.strftime("%Y-%m-%d %H:%M:%S")

            image[k,4], image[k,5] = self.swathAlong, self.swathCross

            image[k,6], image[k,7] = self.getGSD(k)

            image[k,9], image[k,8] = self.getSolarAngles(self.FOVs[k,8], self.FOVs[k,9], self.FOVs[k,10], image[k,3])

            image[k,11], image[k,10] = self.getSolarAngles(self.FOVs[k,8], self.FOVs[k,9], 0.0, image[k,3])

            image[k,12] = self.offNad[0]
            image[k,13] = self.offNad[1]

            image[k,14] = self.FOVs[k,10] / np.cos(np.deg2rad(self.offNad[0]))

            image[k,15:] = imOverlaps[k,:]

            featTitleList.append(f'Overlap % with Image {k+1}')
    
        return image, featTitleList
    
    def getGSD(self, imInd):

        GSDalong = 1000 * self.FOVs[imInd,10] * self.sens[3] / self.sens[0]
        GSDcross = 1000 * self.FOVs[imInd,10] * self.sens[3] / self.sens[0]

        return GSDalong, GSDcross

    def getOverlapPCTnadir(self):
        fov = self.FOVs
        im = np.zeros((self.numImage, self.numImage))
        for j in range(self.numImage):
            imNow = np.zeros(self.numImage)
            latNow = abs(fov[j,0] - fov[j,4])
            lonNow = abs(fov[j,1] - fov[j,3])
            for k in range(self.numImage):
                if k != j:
                    dLat = abs(fov[k,0] - fov[j,0])
                    dLon = abs(fov[k,1] - fov[j,1])
                    imNow[k] = 100 * ((latNow - dLat)*(lonNow - dLon) / (latNow * lonNow))    
                if k == j:
                    imNow[k] = 100
            im[j,:] = imNow

        return im
    
    def getImageTime(self, imInd):
        
        imTime = datetime.fromisoformat(self.epoch) + timedelta(seconds=self.t[imInd])

        return imTime
    
    def printFeatures(self, imageEmbed, imTitle, indIm):

        print()
        print(f"Features for Image {indIm + 1}:")
        for k in range(len(imTitle)):
            value = imageEmbed[indIm, k]
            formatted = f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
            print(f"     {imTitle[k]}: {formatted}")
        print()
    
    def getSolarAngles(self, lat, lon, alt, timeUTC):
        et = spice.str2et(timeUTC) # converting UTC time to ephemeris time
        state, _ = spice.spkezr('SUN', et, 'IAU_EARTH', 'LT+S','EARTH') # getting the suns position in IAU earth with the light-time second delay
        sunR = state[:3] # extracted position-vector of Sun relative to IAU

        earthR = spice.bodvrd("EARTH", "RADII", 3)[1]
        eqEarthR = earthR[0] # earth equatorial radius
        poleEarthR = earthR[2] # earth polar radius
        flatCoef = abs(eqEarthR - poleEarthR) / eqEarthR # flattening coefficient for spice georec

        latRad = np.radians(lat)
        lonRad = np.radians(lon)

        obs = spice.georec(lonRad, latRad, alt, eqEarthR, flatCoef) # Get the vector from the center of the earth to location of interest
        # input is lon in rad, lat in rad, alt in km, earth equatorial rad in km, flattening coeff for earth ellipsoid shape

        sunVec = sunR - obs # obtain vector of sun relative to observer

        zenIAU = np.array([np.cos(latRad)*np.cos(lonRad), np.cos(latRad)*np.sin(lonRad), np.sin(latRad)]) # normalized zenith vector which is normal to the ellipsoid surface at POI

        cZ = np.dot(sunVec, zenIAU) / np.linalg.norm(sunVec) # getting cosine of zenith
        elevation = 90 - np.degrees(np.arccos(cZ))

        east = np.array([-np.sin(lonRad), np.cos(lonRad), 0])
        north = np.array([ -np.sin(latRad)*np.cos(lonRad), -np.sin(latRad)*np.sin(lonRad), np.cos(latRad)])

        sunEast = np.dot(sunVec, east)
        sunNorth = np.dot(sunVec, north)

        azimuth = np.degrees(np.arctan2(sunEast, sunNorth))
        if azimuth < 0:
            azimuth += 360

        return elevation, azimuth