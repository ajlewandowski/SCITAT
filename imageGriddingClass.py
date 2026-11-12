"""
This file contains the class that accounts for image gridding and creating the heat maps
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# This class will import system parameters and produce a resolution "heat map"

class imageGridding:
    def __init__(self, noradID, state84, stateECEF, times, sens, target):
        self.norad = noradID
        self.state84 = state84
        self.stateECEF = stateECEF
        self.sens = sens
        self.target = target
        self.times = times
        self.IFOVbase = sens[0] / sens[2]

        self.wgsSMA = 6378137.0
        self.wgsf = 1 / 298.257223563
        self.ae = 6378137

    def useSphericalModel(self, altitude, fore, starboard):
        """ This function will analyze the satellite's altitude and determine whether to use flat-earth (quicker) or spherical earth (slower) model """
        """ Data was taken in skewModelTest.py and the following fit was determined to correlate altitude to the maximum off-nadir angle the field of view can subtend, 
            before the flat model becomes inaccurate (1% threshold). y = a * x^b      a = 563.0276124576732      b = -0.5638225633766288    RSquared = 0.9941159055261909"""
        degLimitFlat = 563.0276124576732 * np.power(altitude, -0.5638225633766288)
        fovHalf = np.rad2deg(self.sens[0]/2)
        checkVal = np.max(abs(np.array([fore + fovHalf, fore - fovHalf, starboard + fovHalf, starboard - fovHalf])))

        return checkVal >= degLimitFlat

    def getAngleVector(self, angDeg):
        # Look angles at the pixel boundaries (N + 1 values) for a boresight offset of angDeg
        ang = np.deg2rad(angDeg)
        return np.linspace(-self.sens[0]/2 + ang, self.sens[0]/2 + ang, int(self.sens[2]) + 1)

    def getGSDVectors(self, altitude, fore, starboard, spherical):
        """ Returns (gsdFore, gsdStar, angVecFore, angVecStar).
 
            The full GSD array is np.outer(gsdFore, gsdStar).
            Flat earth:      GSD = sqrt(outer(a, b)) = outer(sqrt(a), sqrt(b))                  (exact)
            Spherical earth: GSD = sqrt(4 ae^2 arcsin(outer(sin(dLf/2), sin(dLs/2))))
                             arcsin(z) == z in float64 here (z ~ 1e-15), so
                             GSD = outer(sqrt(2 ae sin(dLf/2)), sqrt(2 ae sin(dLs/2)))      """
        alt = altitude*1e3
        angVecFore = self.getAngleVector(fore)
        angVecStar = self.getAngleVector(starboard)
 
        if spherical:
            # Earth central angles at the pixel boundaries, then per-pixel central-angle spans
            dLamFore = np.abs(np.diff(self.getEarthCentralAngle(alt, angVecFore)))
            dLamStar = np.abs(np.diff(self.getEarthCentralAngle(alt, angVecStar)))
            gsdFore = np.sqrt(2 * self.ae * np.sin(dLamFore/2))
            gsdStar = np.sqrt(2 * self.ae * np.sin(dLamStar/2))
        else:
            # Range = alt / cos, GSD skewed by a further 1 / cos, evaluated at the pixel centers
            cosFore = np.cos(angVecFore[:-1] + self.IFOVbase*0.5)
            cosStar = np.cos(angVecStar[:-1] + self.IFOVbase*0.5)
            gsdFore = np.sqrt(self.IFOVbase * alt / cosFore**2)
            gsdStar = np.sqrt(self.IFOVbase * alt / cosStar**2)
 
        return gsdFore, gsdStar, angVecFore, angVecStar

    def getGSDArray(self, altitude, fore, starboard, matMin, matMax, saveMat, saveFig, path, maxPlotPix=1000):
        """ This function will analyze the satellite's altitude and determine whether to use flat-earth (quicker) or spherical earth (slower) model """
        print()
        if self.useSphericalModel(altitude, fore, starboard):
            fig = self.getOffNad2DArraySpherical(altitude, fore, starboard, matMin, matMax, saveMat, saveFig, path, maxPlotPix)
        else:
            fig = self.getOffNad2DArrayFlat(altitude, fore, starboard, matMin, matMax, saveMat, saveFig, path, maxPlotPix)
        return fig

    def getOffNad2DArrayFlat(self, altitude, fore, starboard, matMin, matMax, saveMat, saveFig, path, maxPlotPix=1000):
        # Flat-Earth GSD array. Fore is positive forward / aft negative, starboard positive right / port negative left.
        return self.makeGSDArray(altitude, fore, starboard, matMin, matMax, saveMat, saveFig, path, maxPlotPix, spherical=False)
 
    def getOffNad2DArraySpherical(self, altitude, fore, starboard, matMin, matMax, saveMat, saveFig, path, maxPlotPix=1000):
        # Spherical-Earth GSD array. Fore is positive forward / aft negative, starboard positive right / port negative left.
        return self.makeGSDArray(altitude, fore, starboard, matMin, matMax, saveMat, saveFig, path, maxPlotPix, spherical=True)
 
    def makeGSDArray(self, altitude, fore, starboard, matMin, matMax, saveMat, saveFig, path, maxPlotPix, spherical):
        gsdFore, gsdStar, angVecFore, angVecStar = self.getGSDVectors(altitude, fore, starboard, spherical)
 
        fig = self.plotGSD(gsdFore, gsdStar, angVecFore, angVecStar, fore, starboard,
                            matMin, matMax, cmapDefault='jet', maxPlotPix=maxPlotPix)
 
        model = 'Spherical-Earth' if spherical else 'Flat-Earth'
        print(f'GSD Array for ({fore}°,{starboard}°) utilized the {model} Approximation')
 
        self.saveGSD(fig, path, altitude, fore, starboard, gsdFore, gsdStar, angVecFore, angVecStar, saveFig, saveMat)
        return fig
 
    def plotGSD(self, gsdFore, gsdStar, angVecFore, angVecStar, fore, starboard, matMin, matMax,
                 cmapDefault='jet', maxPlotPix=1000):
        # Only a decimated grid is drawn; the figure cannot show more pixels than this anyway
        step = max(1, len(gsdFore) // maxPlotPix)
        GSDsPlot = np.outer(gsdFore[::step], gsdStar[::step])
 
        extent = np.rad2deg([angVecStar[0], angVecStar[-1], angVecFore[0], angVecFore[-1]])
 
        fig, ax = plt.subplots(figsize=(10, 8))
        if matMin > 0 and matMax > 0:
            im = ax.imshow(GSDsPlot, vmin=matMin, vmax=matMax, origin='lower', cmap='coolwarm',
                           interpolation='nearest', extent=extent)
        else:
            im = ax.imshow(GSDsPlot, origin='lower', cmap=cmapDefault, interpolation='nearest', extent=extent)
        cbar = fig.colorbar(im)
        cbar.set_label('Ground Sampling Distance (m)')
 
        # First, quarter, half, three-quarter, and end boundaries for labeling, in degrees
        n = len(angVecFore) - 1
        selectInd = [0, n // 4, n // 2, n - (n // 4), n]
        ticksStar = np.rad2deg(angVecStar[selectInd])
        ticksFore = np.rad2deg(angVecFore[selectInd])
        ax.set_xticks(ticksStar)
        ax.set_yticks(ticksFore)
        ax.set_xticklabels(np.round(ticksStar, decimals=2))
        ax.set_yticklabels(np.round(ticksFore, decimals=2))
        ax.set_xlabel('Starboard/Port Direction (°)')
        ax.set_ylabel('Fore/Aft Direction (°)')
        ax.set_title(f'Pixel-by-Pixel GSD Off-Nadir ({fore}°,{starboard}°)')
        return fig
 
    def saveGSD(self, fig, path, altitude, fore, starboard, gsdFore, gsdStar, angVecFore, angVecStar, saveFig, saveMat):
        tag = f'{altitude}_km_offnadir_({fore},{starboard})'
        path = Path(path)
 
        if saveFig:
            fileName = f'GSDarray_{tag}.png'
            fig.savefig(path / fileName)
            print(f"Plot saved as {fileName}")
 
        if saveMat:
            # Two vectors (~0.8 MB at 48096 px) instead of the full matrix (~18.5 GB)
            matFileName = f'GSDarray_vectors_{tag}.npz'
            np.savez(path / matFileName, gsdFore=gsdFore, gsdStar=gsdStar,
                     angVecFore=angVecFore, angVecStar=angVecStar)
            print(f"Vectors saved as {matFileName}")

    def getGSDMatrix(self, altitude, fore, starboard, step=1):
        # Computes the GSD matrix directly (no files needed), decimated by step
        spherical = self.useSphericalModel(altitude, fore, starboard)
        gsdFore, gsdStar, _, _ = self.getGSDVectors(altitude, fore, starboard, spherical)
        return np.outer(gsdFore[::step], gsdStar[::step])

    @staticmethod
    def loadGSDVectors(path, altitude, fore, starboard):
        # Returns a dict with gsdFore, gsdStar, angVecFore, angVecStar
        tag = f'{altitude}_km_offnadir_({fore},{starboard})'
        with np.load(Path(path) / f'GSDarray_vectors_{tag}.npz') as d:
            return {k: d[k] for k in d.files}
 
    @staticmethod
    def loadGSD(path, altitude, fore, starboard, step=1):
        """ Rebuilds the GSD matrix for one look angle. step > 1 keeps every step-th pixel in
            each direction (step=8 turns 48096x48096 into 6012x6012).
            Falls back to the old GSDarray_matrix_*.npy files so existing data still loads. """
        path = Path(path)
        tag = f'{altitude}_km_offnadir_({fore},{starboard})'
 
        vecFile = path / f'GSDarray_vectors_{tag}.npz'
        if vecFile.exists():
            with np.load(vecFile) as d:
                return np.outer(d['gsdFore'][::step], d['gsdStar'][::step])
 
        legacyFile = path / f'GSDarray_matrix_{tag}.npy'
        if legacyFile.exists():
            # Memory-map so only the requested pixels are read from disk
            return np.array(np.load(legacyFile, mmap_mode='r')[::step, ::step])
 
        raise FileNotFoundError(f"No GSD vectors or matrix for {tag} in {path}")
 
    def getEarthCentralAngle(self, alt, offNadAngle):
 
        # Calculate the earth-central angle
        return np.arcsin(np.sin(offNadAngle) * (self.ae + alt) / self.ae) - offNadAngle
 
    def getOffNadOLIArrayFlat(self, angle):
 
        # Find the IFOV and define the altitude time-series from the state output from the WGS84 TAT-C model
        alts84 = self.state84['geometry'].apply(lambda p: p.z).values
        time84 = pd.Series(self.times)
 
        # Creating a vector of satellite look angles
        offNadAngle = np.deg2rad(angle)
        angVec = np.linspace(-self.sens[0]/2 + offNadAngle, self.sens[0]/2 + offNadAngle, int(self.sens[2]) + 1)
        
        # Creating an array of GSDs based of the altitude time-series assuming a flat earth model
        cosTerm = np.cos(angVec[:-1] + 0.5*self.IFOVbase)
        rangeVec = alts84 / cosTerm
        GSDbase = self.IFOVbase * rangeVec
        GSDsNow = GSDbase / cosTerm
 
        # Indexing the first, quarter, half, three-quarter, and end for graphic labeling purposes
        nRow = GSDsNow.shape[0]
        nCol = GSDsNow.shape[1]
        selectIndRow = [0, nRow // 4, nRow // 2, nRow - 1 - (nRow // 4), nRow - 1]
        selectIndCol = [0, nCol // 4, nCol // 2, nCol - 1 - (nCol // 4), nCol - 1]
 
        # Creating vectors of the selected indices
        selectTimes = time84.iloc[selectIndRow].dt.strftime('%H:%M:%S')
        dateTime = time84.iloc[0].strftime('%Y-%m-%d')
        selectAlts = alts84[selectIndRow]
 
        # Plot GSD over array 
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(GSDsNow, origin='lower', cmap='coolwarm', interpolation='nearest')
        cbar = fig.colorbar(im)
        cbar.set_label('Ground Sampling Distance (m)')
        combLabels = [f"{cat}\n({met:.3f} m)" for cat, met in zip(selectTimes, selectAlts)]
        stepRow = (nRow - 1) / (len(selectIndRow) - 1)
        stepCol = (nCol - 1) / (len(selectIndCol) - 1)
        ax.set_yticks(np.arange(0, nRow, stepRow))
        ax.set_xticks(np.arange(0, nCol, stepCol))
        ax.set_xticklabels(np.round(np.rad2deg(angVec[selectIndCol]), decimals=2))
        ax.set_yticklabels(combLabels)
        ax.set_xlabel('FOV Span (degrees)')
        ax.set_ylabel('Along Track Line Integration Time-series (Time/Altitude (m))')
        ax.set_title(f'Pixel-by-Pixel GSD Landsat-8 OLI (Nadir): {dateTime}')
        plt.savefig(f'fov_skew_array_offnadir_{angle}_timeseries.png')
        print(f"Plot saved as fov_skew_array_offnadir_{angle}_timeseries.png")
 
    def getSatSSPHeading(self):
        # This function will calculate the satellite's heading angle in the plane tangent to the surface at the SSP
 
        # Extract state vectors into individual components in the ECEF frame, in km and km/s
        xecef = self.stateECEF['geometry'].apply(lambda p: p.x).values / 1e3
        yecef = self.stateECEF['geometry'].apply(lambda p: p.y).values / 1e3
        zecef = self.stateECEF['geometry'].apply(lambda p: p.z).values / 1e3
        vxecef = self.stateECEF['velocity'].apply(lambda v: v.x).values / 1e3
        vyecef = self.stateECEF['velocity'].apply(lambda v: v.y).values / 1e3
        vzecef = self.stateECEF['velocity'].apply(lambda v: v.z).values / 1e3
 
        # Extract the latitudes, longitudes, and altitudes
        lats = self.state84['geometry'].apply(lambda p: p.x).values
        lons = self.state84['geometry'].apply(lambda p: p.y).values
        alts = self.state84['geometry'].apply(lambda p: p.z).values / 1e3
 
        # Calculate the heading angle, the angle between the north pole and velocity vector
        head = np.zeros(len(xecef))
        for k in range(len(head)):
            rECEF = np.array([xecef[k], yecef[k], zecef[k]]) * 1e3
            vECEF = np.array([vxecef[k], vyecef[k], vzecef[k]])
 
            # Using this function to get the vector pointing north from the latitude longitude position in ECEF
            northECEF = self.getTrueNorthInECEF(lats[k], lons[k])
 
            # This function projects the ECEF velocity onto the Earth's surface at the latitude longitude position of the SSP
            vSurface = self.projectVelo2Surface(rECEF, vECEF)
 
            # Calculating the heading angle and wrapping to 360
            head[k] = np.mod(np.degrees(np.arctan2(vSurface[1], vSurface[0]) - np.arctan2(northECEF[1], northECEF[0])), 360)
 
        # Plot the figure of the satellite heading over time
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(head)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Heading Angle (degrees)')
        ax.set_title('Satellite Heading Angle Over Time')
        plt.savefig('landsat8_heading.png')
        print("Plot saved as landsat8_heading.png")
 
        return head
 
    def projectVelo2Surface(self, rECEF, vECEF):
        # This function takes the satellite's position and velocity and projects it on the Earth's surface
        # tangent to the surface at the SSP
 
        rNorm = rECEF / np.linalg.norm(rECEF)
        vRad = np.dot(vECEF, rNorm) * rNorm
 
        return vECEF - vRad
 
    def getTrueNorthInECEF(self, lat, lon):
        # This function calculates the north direction from the SSP
 
        lat = np.radians(lat)
        lon = np.radians(lon)
 
        # WGS84 geometric model to create position vector
        N = self.wgsSMA / np.sqrt(1 - (2 * self.wgsf - self.wgsf**2) * np.sin(lat)**2)
        x = -(N * np.cos(lat) * np.sin(lon))
        y = N * np.cos(lat) * np.cos(lon)
        z = (N * (1 - (2 * self.wgsf - self.wgsf**2)) * np.sin(lat))
        
        # Normalize the position vector
        rECEF = np.array([x, y, z])
        rNorm = rECEF / np.linalg.norm(rECEF)
        
        # True north direction in ECEF frame
        east = np.array([-np.sin(lon), np.cos(lon), 0.0])
 
        # Crossing the normed-position vector with the east direction to get true north from SSP
        north = np.cross(rNorm, east)
        north = north / np.linalg.norm(north)
 
        return north
 
    def getSatLVLHHeading(self):
        # This function will calculate the satellite's heading angle in the Local Vertical/Local Horizontal frame of the satellite
 
        # Extract state vectors into individual components in the ECEF frame, in km and km/s
        xecef = self.stateECEF['geometry'].apply(lambda p: p.x).values / 1e3
        yecef = self.stateECEF['geometry'].apply(lambda p: p.y).values / 1e3
        zecef = self.stateECEF['geometry'].apply(lambda p: p.z).values / 1e3
        vxecef = self.stateECEF['velocity'].apply(lambda v: v.x).values / 1e3
        vyecef = self.stateECEF['velocity'].apply(lambda v: v.y).values / 1e3
        vzecef = self.stateECEF['velocity'].apply(lambda v: v.z).values / 1e3
 
        head = np.zeros(len(xecef))
 
        # Loop that will calculate the heading angle at each time step
        for k in range(len(head)):
 
            # Create position and velocity vectors in ECEF
            rECEF = np.array([xecef[k], yecef[k], zecef[k]])
            vECEF = np.array([vxecef[k], vyecef[k], vzecef[k]])
 
            # Get transformation matrix from ECEF to the LVLH frame
            T = self.ECEF2LVLH(rECEF, vECEF)
 
            # Find local north transformation matrix in the LVLH frame
            northLVLH = self.getLocalNorthInLVLH(rECEF)
 
            # Transform the velocity vector to LVLH and normalize for velocity direction
            vLVLH = T @ vECEF
            vLVLHnorm = vLVLH / np.linalg.norm(vLVLH)
        
            # Find the angle between satellite velo direction in LVLH and north in LVLH
            angle = np.degrees(np.arctan2(northLVLH[1], northLVLH[0]))
            velAngle = np.degrees(np.arctan2(vLVLHnorm[1], vLVLHnorm[0]))
 
            head[k] = np.degrees(velAngle - angle) % 360
 
        # Plot heading angle versus time 
        fig, ax = plt.subplots(figsize=(8, 5))
        xvals = np.linspace(0, len(head), len(head))
        ax.plot(xvals, head, linewidth=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Heading Angle (degrees)')
        ax.set_title('Satellite Heading Over Time')
        ax.grid(True, alpha=0.3)
        plt.savefig('satellite_heading_graphic.png')
        print("Plot saved as satellite_heading_graphic.png")
 
        return head
 
    def ECEF2LVLH(self, rECEF, vECEF):
        # Determine the transformation matrix to convert the ECEF frame to the LVLH frame
 
        ez = -rECEF / np.linalg.norm(rECEF)
 
        ey = -np.cross(rECEF, vECEF)
        ey /= np.linalg.norm(ey)
 
        ex = np.cross(ey, ez)
        ex /= np.linalg.norm(ex)
 
        return np.array([ex, ey, ez])
 
    def getLocalNorthInLVLH(self, rECEF):
        # Determine direction to the north pole, due north, in the LVLH frame local to the satellite
 
        k = np.array([0, 0, 1])
 
        ez = -rECEF / np.linalg.norm(rECEF)
 
        northECEF = k - np.dot(k, ez) * ez
        northECEF /= np.linalg.norm(northECEF)
 
        T = self.ECEF2LVLH(rECEF, np.array([1, 0, 0]))
 
        northLVLH = T @ northECEF
 
        return northLVLH
 
    def plotNadirBaseArray(self, altitude, plotSave):
        # This function plots a single line-integration imager array to look at a single dispersion over the sensor
 
        alt = altitude*1e3
 
        GSDbase = self.IFOVbase * alt
        angVecFlat = np.linspace(-self.sens[0]/2, self.sens[0]/2, int(self.sens[2]))
 
        rangeSat = alt / np.cos(angVecFlat)
        GSDSat = self.IFOVbase * rangeSat
        GSDsNow = GSDSat / np.cos(angVecFlat)
 
        angVecSphere = np.linspace(-self.sens[0]/2, self.sens[0]/2, int(self.sens[2]) + 1)
 
        lam = self.getEarthCentralAngle(alt, angVecSphere)
 
        # GSD array being created 
        dLam = np.diff(lam)
        GSDs = self.ae * dLam
 
        if plotSave:
            # Plot GSD over array 
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(np.rad2deg(angVecFlat), GSDs, linewidth=2, color='r', label='Spherical')
            ax.plot(np.rad2deg(angVecFlat), GSDsNow, linewidth=2, color='b', label='Simple')
            ax.set_xlabel('FOV Span (degrees)')
            ax.set_ylabel('GSD (m)')
            ax.set_title(f'Single Line Array Pixel-by-Pixel GSD Landsat-8 OLI (Nadir) at Altitude: {altitude} km')
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.savefig(f'fov_skew_array_base_{altitude}.png')
            print(f"Plot saved as fov_skew_array_base_{altitude}.png")
 
 
            # Plot GSD over array 
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(np.rad2deg(angVecFlat), abs(GSDs - GSDsNow)*100 / GSDs, linewidth=2, color='r')
            ax.set_xlabel('FOV Span (degrees)')
            ax.set_ylabel('Percent Error in GSD (m)')
            ax.set_title(f'Pixel-by-Pixel % Difference of Base GSD Landsat-8 OLI (Nadir) at Altitude: {altitude} km')
            ax.grid(True, alpha=0.3)
            plt.savefig(f'fov_skew_array_base_percerror_{altitude}.png')
            print(f"Plot saved as fov_skew_array_base_percerror_{altitude}.png")
 
        return abs(GSDs - GSDsNow)*100 / GSDs, np.rad2deg(angVecFlat)
 
    def plotAltitudeTimeseries(self):
        # This function plots the altitude time-series 
 
        alts84 = self.state84['geometry'].apply(lambda p: p.z).values
        time84 = pd.Series(self.times)
 
        n = len(alts84)
        selectIndRow = [0, n // 4, n // 2, n - 1 - (n // 4), n - 1]
        selectTimes = time84.iloc[selectIndRow].dt.strftime('%H:%M:%S')
        dateTime = time84.iloc[0].strftime('%Y-%m-%d')
 
        # Plot Altitude over Timespan
        fig, ax = plt.subplots(figsize=(8, 5))
        xvals = np.linspace(0, len(alts84), len(alts84))
        ax.plot(xvals, alts84/1e3, linewidth=2)
        stepRow = (n - 1) / (len(selectIndRow) - 1)
        ax.set_xticks(np.arange(0, n, stepRow))
        ax.set_xticklabels(selectTimes)
        ax.set_xlabel('Time-Series of Altitude')
        ax.set_ylabel('Altitude (km)')
        ax.set_title(f'Time-Series of Landsat-8 Altitudes: {dateTime}')
        ax.grid(True, alpha=0.3)
        plt.savefig('landsat8_altitude.png')
        print("Plot saved as landsat8_altitude.png")
 
    def getColorbarMinMax(self, angVec, altitude):
 
        alt = altitude * 1e3
 
        minAng = np.deg2rad(min(abs(angVec)))
        maxAng = np.deg2rad(max(abs(angVec)))
 
        rangeVec = alt / np.cos(minAng + self.IFOVbase*0.5)
        GSDbase = self.IFOVbase * rangeVec
        GSDsFore = GSDbase / np.cos(minAng + self.IFOVbase*0.5)
        rangeVec = alt / np.cos(minAng + self.IFOVbase*0.5)
        GSDbase = self.IFOVbase * rangeVec
        GSDsStar = GSDbase / np.cos(minAng + self.IFOVbase*0.5)
        minGSD = np.sqrt(GSDsFore * GSDsStar)
 
        lam2 = self.getEarthCentralAngle(alt, maxAng + self.sens[0]/2)
        lam1 = self.getEarthCentralAngle(alt, maxAng + self.sens[0]/2 - self.IFOVbase)
        dLam = abs(lam2 - lam1)
        A = abs(4 * self.ae**2 * np.arcsin(np.sin(dLam/2)*np.sin(dLam/2)))
        maxGSD = np.sqrt(A)
      
        return minGSD, maxGSD
 
    # ------------------------------------------------------------------------------------------
    # Quadrant flips: mirroring the matrix is the same as reversing a vector.
    #   np.fliplr(outer(u, v)) == outer(u, v[::-1])
    #   np.flipud(outer(u, v)) == outer(u[::-1], v)
    #   np.flip(outer(u, v))   == outer(u[::-1], v[::-1])
    # gsdVecs is (gsdFore, gsdStar) for the quadrant-one angles (fore, starboard), e.g. from
    # getGSDVectors(...)[:2] or loadGSDVectors(...). Calling getGSDArray with the negated
    # angles gives the same result and is just as fast.
    # ------------------------------------------------------------------------------------------
 
    def _flipQuadrant(self, path, fore, starboard, altitude, matMin, matMax, gsdVecs, flipFore, flipStar):
        gsdFore, gsdStar = gsdVecs[0], gsdVecs[1]
 
        foreNew = -fore if flipFore else fore
        starNew = -starboard if flipStar else starboard
        gsdForeNew = gsdFore[::-1] if flipFore else gsdFore
        gsdStarNew = gsdStar[::-1] if flipStar else gsdStar
        angVecForeNew = self.getAngleVector(foreNew)
        angVecStarNew = self.getAngleVector(starNew)
 
        fig = self._plotGSD(gsdForeNew, gsdStarNew, angVecForeNew, angVecStarNew, foreNew, starNew,
                            matMin, matMax, cmapDefault='coolwarm')
        self._saveGSD(fig, path, altitude, foreNew, starNew, gsdForeNew, gsdStarNew,
                      angVecForeNew, angVecStarNew, saveFig=True, saveMat=True)
        return fig
 
    def flipOne2Two(self, path, fore, starboard, altitude, matMin, matMax, gsdVecs):
        # Quadrant one to two: starboard -> port
        return self._flipQuadrant(path, fore, starboard, altitude, matMin, matMax, gsdVecs, flipFore=False, flipStar=True)
 
    def flipOne2Three(self, path, fore, starboard, altitude, matMin, matMax, gsdVecs):
        # Quadrant one to three: fore -> aft and starboard -> port
        return self._flipQuadrant(path, fore, starboard, altitude, matMin, matMax, gsdVecs, flipFore=True, flipStar=True)
 
    def flipOne2Four(self, path, fore, starboard, altitude, matMin, matMax, gsdVecs):
        # Quadrant one to four: fore -> aft
        return self._flipQuadrant(path, fore, starboard, altitude, matMin, matMax, gsdVecs, flipFore=True, flipStar=False)