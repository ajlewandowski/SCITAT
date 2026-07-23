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

        self.wgsSMA = 6378137.0
        self.wgsf = 1 / 298.257223563
        self.ae = 6378137

    def getOffNad2DArrayFlat(self, altitude, fore, starboard, matMin, matMax):
        # This function looks at altitude, satellite look angles, and sensor parameters to create 2D arrays of
        # a satellite image's GSD. Assuming the Earth's surface is flat, this will break down the greater
        # the look angle. Takes shorter time to run compared to spherical Earth model.

        # Fore is forward/backward angle - fore is the positive and aft negative
        # Starboard is right/left angle - starboard is the positive to the right, port negative to the left

        # Calculate the instantaneous field-of-view (IFOV), which is FOV / # of Pix
        IFOVbase = self.sens[0] / self.sens[2]
        alt = altitude

        # Creating a series of satellite look angles 
        angFore = np.deg2rad(fore)
        angVecFore = np.linspace(-self.sens[0]/2 + angFore, self.sens[0]/2 + angFore, int(self.sens[2]) + 1)
        angStar = np.deg2rad(starboard)
        angVecStar = np.linspace(-self.sens[0]/2 + angStar, self.sens[0]/2 + angStar, int(self.sens[2]) + 1)

        # Initializing the 2D array and spreads of GSDs
        GSDsFore = np.zeros(len(angVecFore) - 1)
        GSDsStar = np.zeros(len(angVecStar) - 1)

        # Iterate through the satellite look angles creating the GSD arrays
        for k in range(len(angVecFore) - 1):
            # Get the range vector assuming a flat earth model
            rangeVec = alt / np.cos(angVecFore[k] + IFOVbase*0.5)
            # Get the base GSD from the range vector and IFOV
            GSDbase = IFOVbase * rangeVec
            # Skew the IFOV based on the angle between the range and ground
            GSDsFore[k] = GSDbase / np.cos(angVecFore[k] + IFOVbase*0.5)

            rangeVec = alt / np.cos(angVecStar[k] + IFOVbase*0.5)
            GSDbase = IFOVbase * rangeVec
            GSDsStar[k] = GSDbase / np.cos(angVecStar[k] + IFOVbase*0.5)

        # Creating the 2D array with the skewed multi-directional angles
        GSDs = np.zeros((len(GSDsFore), len(GSDsStar)))
        for k in range(len(GSDsFore)):
            for j in range(len(GSDsStar)):
                # Calculate the geometric mean, which is sqrt(length * width)
                GSDs[k,j] = np.sqrt(GSDsFore[k] * GSDsStar[j])

        # Indexing the first, quarter, half, three-quarter, and end points for labeling graphic
        nRow = GSDs.shape[0]
        nCol = GSDs.shape[1]
        selectIndRow = [0, nRow // 4, nRow // 2, nRow - 1 - (nRow // 4), nRow - 1]
        selectIndCol = [0, nCol // 4, nCol // 2, nCol - 1 - (nCol // 4), nCol - 1]

        # Plot GSD over array 
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(GSDs, vmin = matMin, vmax = matMax, origin='lower', cmap='coolwarm', interpolation='nearest')
        cbar = fig.colorbar(im)
        cbar.set_label('Ground Sampling Distance (m)')
        stepRow = (nRow - 1) / (len(selectIndRow) - 1)
        stepCol = (nCol - 1) / (len(selectIndCol) - 1)
        ax.set_yticks(np.arange(0, nRow, stepRow))
        ax.set_xticks(np.arange(0, nCol, stepCol))
        ax.set_xticklabels(np.round(np.rad2deg(angVecStar[selectIndRow]), decimals=2))
        ax.set_yticklabels(np.round(np.rad2deg(angVecFore[selectIndCol]), decimals=2))
        ax.set_xlabel('Starboard/Port Direction (°)')
        ax.set_ylabel('Fore/Aft Direction (°)')
        ax.set_title(f'Pixel-by-Pixel GSD Off-Nadir ({fore}°,{starboard}°)')

        # Save the graphic and data file
        fileName = f'fov_skew_2Darray_offnadir_({fore},{starboard}).png'
        pathNow = Path("D:/Users/al62352/SimpleHeatmap")
        fullPath = pathNow / fileName

        plt.savefig(fullPath)
        print(f"Plot saved as fov_skew_2Darray_offnadir_({fore},{starboard}).png")

        # matFileName = f'fov_skew_2Darray_matrix_offnadir_({fore},{starboard}).npy'
        # fullMatPath = self.path / matFileName
        # np.save(fullMatPath, GSDs)

    def getOffNad2DArraySpherical(self, altitude, fore, starboard, matMin, matMax):
        # This function looks at altitude, satellite look angles, and sensor parameters to create 2D arrays of
        # a satellite image's GSD. Assuming the Earth's surface is spherical, takes longer to run.

        # Fore is forward/backward angle - fore is the positive and aft negative
        # Starboard is right/left angle - starboard is the positive to the right, port negative to the left

        # Calculate the instantaneous field-of-view (IFOV), which is FOV / # of Pix
        IFOVbase = self.sens[0] / self.sens[2]
        alt = altitude

        # Creating a series of off-nadir angles for Landsat-8 (15 deg fov)
        angFore = np.deg2rad(fore)
        angVecFore = np.linspace(-self.sens[0]/2 + angFore, self.sens[0]/2 + angFore, int(self.sens[2]) + 1)
        angStar = np.deg2rad(starboard)
        angVecStar = np.linspace(-self.sens[0]/2 + angStar, self.sens[0]/2 + angStar, int(self.sens[2]) + 1)

        # lamFore/lamStar is the earth central angle given a spherical earth, the points are 
        # defined between the pixels not to the center of the pixels
        lamFore = np.zeros(len(angVecFore))
        lamStar = np.zeros(len(angVecStar))
        for k in range(len(angVecFore)):
            lamFore[k] = self.getEarthCentralAngle(altitude, angVecFore[k])
            lamStar[k] = self.getEarthCentralAngle(altitude, angVecStar[k])
            # rho = (altitude + ae*(1 - np.cos(lam))) / np.cos(angVecFore[k] + IFOVbase*0.5)
            # print(rho)
        # maximum look angle - sin(eta) = ae / (altitude + ae)

        # arcs is the GSD array being created 
        arcs = np.zeros((len(lamFore)-1,len(lamStar) - 1))
        for j in range(len(lamFore) - 1):
            # Find the difference in the earth-centered angles
            dLamFore = lamFore[j+1] - lamFore[j]
            for k in range(len(lamStar) - 1):
                dLamStar = lamStar[k+1] - lamStar[k]
                # Using the solid angle patch formulas to get surface area, then get geometric mean for a pseudo-GSD
                A = abs(4 * self.ae**2 * np.arcsin(np.sin(dLamFore/2)*np.sin(dLamStar/2)))
                arcs[j,k] = np.sqrt(A)

        # Indexing the first, quarter, half, three-quarter, and end points for labeling graphic
        nRow = arcs.shape[0]
        nCol = arcs.shape[1]
        selectIndRow = [0, nRow // 4, nRow // 2, nRow - 1 - (nRow // 4), nRow - 1]
        selectIndCol = [0, nCol // 4, nCol // 2, nCol - 1 - (nCol // 4), nCol - 1]

        # Plot GSD over array 
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(arcs, vmin = matMin, vmax = matMax, origin='lower', cmap='coolwarm', interpolation='nearest')
        cbar = fig.colorbar(im)
        cbar.set_label('Ground Sampling Distance (m)')
        stepRow = (nRow - 1) / (len(selectIndRow) - 1)
        stepCol = (nCol - 1) / (len(selectIndCol) - 1)
        ax.set_yticks(np.arange(0, nRow, stepRow))
        ax.set_xticks(np.arange(0, nCol, stepCol))
        ax.set_xticklabels(np.round(np.rad2deg(angVecStar[selectIndRow]), decimals=2))
        ax.set_yticklabels(np.round(np.rad2deg(angVecFore[selectIndCol]), decimals=2))
        ax.set_xlabel('Starboard/Port Direction (°)')
        ax.set_ylabel('Fore/Aft Direction (°)')
        ax.set_title(f'Pixel-by-Pixel GSD Off-Nadir ({fore}°,{starboard}°)')
        
        # Save the graphic and data file
        fileName = f'fov_skew_2Darray_spherical_offnadir_({fore},{starboard}).png'
        pathNow = Path("D:/Users/al62352/SphericalHeatmap")
        fullPath = pathNow / fileName

        plt.savefig(fullPath)
        print(f"Plot saved as fov_skew_2Darray_spherical_offnadir_({fore},{starboard}).png")

        # matFileName = f'fov_skew_2Darray_spherical_matrix_offnadir_({fore},{starboard}).npy'
        # fullMatPath = self.path / matFileName
        # np.save(fullMatPath, arcs)
        # print(f"Array saved as fov_skew_2Darray_spherical_matrix_offnadir_({fore},{starboard}).npy")

    def getEarthCentralAngle(self, altitude, offNadAngle):

        # Calculate the earth-central angle
        return np.arcsin(np.sin(offNadAngle) * (self.ae + altitude) / self.ae) - offNadAngle

    def getOffNadOLIArrayFlat(self, angle):

        # Find the IFOV and define the altitude time-series from the state output from the WGS84 TAT-C model
        IFOVbase = self.sens[0] / self.sens[2]
        alts84 = self.state84['geometry'].apply(lambda p: p.z).values
        time84 = pd.Series(self.times)

        # Creating a vector of satellite look angles
        offNadAngle = np.deg2rad(angle)
        angVec = np.linspace(-self.sens[0]/2 + offNadAngle, self.sens[0]/2 + offNadAngle, int(self.sens[2]) + 1)
        
        # Creating an array of GSDs based of the altitude time-series assuming a flat earth model
        GSDsNow = np.zeros((len(alts84), len(angVec) - 1))
        for k in range(len(alts84)):
            rangeVec = alts84[k] / np.cos(angVec[:-1] + 0.5*IFOVbase)
            GSDbase = IFOVbase * rangeVec
            for j in range(len(angVec) - 1):
                GSDsNow[k,j] = GSDbase[j] / np.cos(angVec[j] + 0.5*IFOVbase)

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

        IFOVbase = self.sens[0] / self.sens[2]
        GSDbase = IFOVbase * alt
        angVecFlat = np.linspace(-self.sens[0]/2, self.sens[0]/2, int(self.sens[2]))

        GSDsNow = np.zeros(len(angVecFlat))
        for k in range(len(angVecFlat)):
            rangeSat = alt / np.cos(angVecFlat[k])
            GSDSat = IFOVbase * rangeSat
            GSDsNow[k] = GSDSat / np.cos(angVecFlat[k])

        angVecSphere = np.linspace(-self.sens[0]/2, self.sens[0]/2, int(self.sens[2]) + 1)

        lam = np.zeros(len(angVecSphere))
        for k in range(len(angVecSphere)):
            lam[k] = self.getEarthCentralAngle(alt, angVecSphere[k])

        # arcs is the GSD array being created 
        arcs = np.zeros(len(lam) - 1)
        for j in range(len(lam) - 1):
            # Find the difference in the earth-centered angles
            dLam = lam[j+1] - lam[j]
            s = self.ae * dLam
            arcs[j] = s

        if plotSave:
            # Plot GSD over array 
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(np.rad2deg(angVecFlat), arcs, linewidth=2, color='r', label='Spherical')
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
            ax.plot(np.rad2deg(angVecFlat), abs(arcs - GSDsNow)*100 / arcs, linewidth=2, color='r')
            ax.set_xlabel('FOV Span (degrees)')
            ax.set_ylabel('Percent Error in GSD (m)')
            ax.set_title(f'Pixel-by-Pixel % Difference of Base GSD Landsat-8 OLI (Nadir) at Altitude: {altitude} km')
            ax.grid(True, alpha=0.3)
            plt.savefig(f'fov_skew_array_base_percerror_{altitude}.png')
            print(f"Plot saved as fov_skew_array_base_percerror_{altitude}.png")

        return abs(arcs - GSDsNow)*100 / arcs, np.rad2deg(angVecFlat)

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

    def analyze2DArrayModels(self, fore, star):

        flatPath = Path(f"D:/Users/al62352/SimpleHeatmap/fov_skew_2Darray_matrix_offnadir_({fore},{star}).npy")
        spherePath = Path(f"D:/Users/al62352/SphericalHeatmap/fov_skew_2Darray_spherical_matrix_offnadir_({fore},{star}).npy")

        flatData = np.load(flatPath)
        sphereData = np.load(spherePath)

        diffData = sphereData - flatData

        nRow = diffData.shape[0]
        nCol = diffData.shape[1]
        selectIndRow = [0, nRow // 4, nRow // 2, nRow - 1 - (nRow // 4), nRow - 1]
        selectIndCol = [0, nCol // 4, nCol // 2, nCol - 1 - (nCol // 4), nCol - 1]

        angFore = np.deg2rad(fore)
        angVecFore = np.linspace(-self.sens[0]/2 + angFore, self.sens[0]/2 + angFore, int(self.sens[2]) + 1)
        angStar = np.deg2rad(star)
        angVecStar = np.linspace(-self.sens[0]/2 + angStar, self.sens[0]/2 + angStar, int(self.sens[2]) + 1)

        # Plot GSD over array 
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(diffData, origin='lower', cmap='coolwarm', interpolation='nearest')
        cbar = fig.colorbar(im)
        cbar.set_label('Ground Sampling Distance (m)')
        stepRow = (nRow - 1) / (len(selectIndRow) - 1)
        stepCol = (nCol - 1) / (len(selectIndCol) - 1)
        ax.set_yticks(np.arange(0, nRow, stepRow))
        ax.set_xticks(np.arange(0, nCol, stepCol))
        ax.set_xticklabels(np.round(np.rad2deg(angVecStar[selectIndRow]), decimals=2))
        ax.set_yticklabels(np.round(np.rad2deg(angVecFore[selectIndCol]), decimals=2))
        ax.set_xlabel('Starboard/Port Direction (°)')
        ax.set_ylabel('Fore/Aft Direction (°)')
        ax.set_title(f'Pixel-by-Pixel GSD Off-Nadir ({fore}°,{star}°)')
        plt.savefig(f'difference_skew_array_({fore},{star}).png')
        print(f'Plot saved as difference_skew_array_({fore},{star}).png')

    def getColorbarMinMax(self, angVec):

        mats = []
        for i in range(len(angVec)):
            for j in range(len(angVec)):
                matPath = Path(f"D:/Users/al62352/SphericalHeatmap/fov_skew_2Darray_spherical_matrix_offnadir_({angVec[i]},{angVec[j]}).npy")
                matData = np.load(matPath)
                mats.append(matData)

        matMin = min(np.nanmin(m) for m in mats)
        matMax = max(np.nanmax(m) for m in mats)

        return matMin, matMax

