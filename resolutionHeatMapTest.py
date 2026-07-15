"""
Gridding and resolution heat map test file
"""

import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tatc import utils
from tatc.schemas import Instrument, Satellite, TwoLineElements
import pandas as pd
from SpaceTrackAPIClass import SpaceTrackAPI
from tatc.analysis import collect_orbit_track, OrbitCoordinate, OrbitOutput

wgsSMA = 6378137.0
wgsf = 1 / 298.257223563

def runTest():
    print()
    print("###################################################################################################")
    print()

    # This step is defining parameters of Landsat-8 and its orbit, as well as finding the closest epoch to the desired
    # image time (imTime). This section will collect the ground track of the orbit.
    username = "alec.lewandowski@utexas.edu"
    password = "!Lew3dglatutaustin"
    imTime = ["2025-12-30T17:00:00", "2025-12-30T17:03:24"]
    noradID = 49260

    #                  FOV, Focal, Pixel #, GSD, Swath, Alt
    sens = np.array([np.deg2rad(15.0), 0.886, 6196, 30, 185000, 705000])
    solarAngles = np.array([32.30026603, 155.16461405])
    target = np.array([30.30617, -97.88121])
    imCorner = np.array([[31.35457, -99.06834],[31.33346, -96.67196],[29.23974, -96.72091],[29.25915, -99.06690]])

    try:
        data = SpaceTrackAPI(username, password)
        tles, closeTime = data.getTLEatEpoch(noradID, imTime)
    except Exception as e:
        print(f"Error fetching TLEs: {e}")
        return

    oli = Instrument(name="OLI",field_of_regard=utils.swath_width_to_field_of_regard(705000, 185000),req_target_sunlit=True)
    L8 = Satellite(name="Landsat-8",
        orbit=TwoLineElements(tle=[tles[1],tles[2]]), instruments=[oli])
    times = pd.date_range(start=closeTime, end=imTime[1], freq="1s", tz="UTC")
    state84 = collect_orbit_track(L8, times, coordinates = OrbitCoordinate.WGS84, sat_sunlit = True, solar_altaz = True, solar_beta = True)
    stateECI = collect_orbit_track(L8, times, orbit_output = OrbitOutput.POSITION_VELOCITY, coordinates = OrbitCoordinate.ECI)
    stateECEF = collect_orbit_track(L8, times, orbit_output = OrbitOutput.POSITION_VELOCITY, coordinates = OrbitCoordinate.ECEF)    

    # # Obtain the heading angle (angle between satellite heading and north in LVLH) 
    # head = getSatLVLHHeading(stateECEF)

    # # Get a time series of Geocentric latitude and longitude 
    # geoc = getGeocentricLatLon(stateECEF)

    # # getGSDArray(stateECEF, state84, head, sens, geoc, times)

    # plotAltitudeTimeseries(state84, times)

    # headAng = getSatSSPHeading(stateECEF, state84)

    # getOffNadOLIArray(state84, sens, times, 5)
    # getOffNadOLIArray(state84, sens, times, 15)
    # getOffNadOLIArray(state84, sens, times, 10)
    # getOffNadOLIArray(state84, sens, times, 0)

    # plotNadirBaseArray(state84, sens, times, 705)

    # The following 4 lines of code runs a very long simulation to create heatmaps for 81 different viewing scenarios
    # offNadVec = np.array([0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0])
    # for k in range(len(offNadVec)):
    #     for j in range(len(offNadVec)):
    #         getOffNad2DArray(sens, 705000, offNadVec[k], offNadVec[j])

    mat = np.load(Path('C:/Users/alecl/OneDrive/Desktop/Graphics/fov_skew_2Darray_matrix_offnadir_(0.0,10.0).npy'))
    print(mat)
    print(mat.shape)
    

def getOffNad2DArray(sens, altitude, fore, starboard):
    # fore is forward/backward angle - fore is the positive and aft negative
    # starboard is right/left angle - starboard is the positive to the right, port negative to the left

    IFOVbase = sens[0] / sens[2]
    alt = altitude

    # Creating a series of off-nadir angles for Landsat-8 (15 deg fov)
    angFore = fore * (np.pi/180)
    angVecFore = np.linspace(-sens[0]/2 + angFore, sens[0]/2 + angFore, int(sens[2]))

    angStar = starboard * (np.pi/180)
    angVecStar = np.linspace(-sens[0]/2 + angStar, sens[0]/2 + angStar, int(sens[2]))

    # Initializing the 2D array and spreads of GSDs
    GSDsFore = np.zeros(len(angVecFore))
    GSDsStar = np.zeros(len(angVecStar))
    for k in range(len(angVecFore)):
        rangeVec = alt / np.cos(angVecFore[k])
        GSDbase = IFOVbase * rangeVec
        GSDsFore[k] = GSDbase / np.cos(angVecFore[k])

        rangeVec = alt / np.cos(angVecStar[k])
        GSDbase = IFOVbase * rangeVec
        GSDsStar[k] = GSDbase / np.cos(angVecStar[k])

    # Creating the 2D array with the skewed multi-directional angles
    GSDs = np.zeros((len(angVecFore), len(angVecStar)))
    for k in range(len(angVecFore)):
        for j in range(len(angVecStar)):
            GSDs[k,j] = np.sqrt(GSDsFore[k] * GSDsStar[j])

    nRow = GSDs.shape[0]
    nCol = GSDs.shape[1]
    selectIndRow = [0, nRow // 4, nRow // 2, nRow - 1 - (nRow // 4), nRow - 1]
    selectIndCol = [0, nCol // 4, nCol // 2, nCol - 1 - (nCol // 4), nCol - 1]

    # Plot GSD over array 
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(GSDs, origin='lower', cmap='coolwarm', interpolation='nearest')
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

    outDir = Path("C:/Users/alecl/OneDrive/Desktop/Graphics")
    fileName = f'fov_skew_2Darray_offnadir_({fore},{starboard}).png'
    fullPath = outDir / fileName

    plt.savefig(fullPath)
    print(f"Plot saved as fov_skew_2Darray_offnadir_({fore},{starboard}).png")

    matFileName = f'fov_skew_2Darray_matrix_offnadir_({fore},{starboard}).npy'
    fullMatPath = outDir / matFileName
    np.save(fullMatPath, GSDs)
    

def getOffNadOLIArray(state84, sens, times, angle):

    IFOVbase = sens[0] / sens[2]
    alts84 = state84['geometry'].apply(lambda p: p.z).values
    time84 = pd.Series(times)

    # This test is going to look at an off-nadir angle, only sideways of 10 degrees
    offNadAngle = angle * (np.pi/180)
    angVec = np.linspace(-sens[0]/2 + offNadAngle, sens[0]/2 + offNadAngle, int(sens[2]))
    GSDsNow = np.zeros((len(alts84), len(angVec)))
    for k in range(len(alts84)):
        rangeVec = alts84[k] / np.cos(angVec)
        GSDbase = IFOVbase * rangeVec
        for j in range(len(angVec)):
            GSDsNow[k,j] = GSDbase[j] / np.cos(angVec[j])

    nRow = GSDsNow.shape[0]
    nCol = GSDsNow.shape[1]
    selectIndRow = [0, nRow // 4, nRow // 2, nRow - 1 - (nRow // 4), nRow - 1]
    selectIndCol = [0, nCol // 4, nCol // 2, nCol - 1 - (nCol // 4), nCol - 1]
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

def getSatSSPHeading(stateECEF, state84):
    
    # Extract state vectors into individual components in the ECEF frame, in km and km/s
    xecef = stateECEF['geometry'].apply(lambda p: p.x).values / 1e3
    yecef = stateECEF['geometry'].apply(lambda p: p.y).values / 1e3
    zecef = stateECEF['geometry'].apply(lambda p: p.z).values / 1e3
    vxecef = stateECEF['velocity'].apply(lambda v: v.x).values / 1e3
    vyecef = stateECEF['velocity'].apply(lambda v: v.y).values / 1e3
    vzecef = stateECEF['velocity'].apply(lambda v: v.z).values / 1e3

    lats = state84['geometry'].apply(lambda p: p.x).values
    lons = state84['geometry'].apply(lambda p: p.y).values
    alts = state84['geometry'].apply(lambda p: p.z).values / 1e3


    head = np.zeros(len(xecef))
    for k in range(len(head)):
        rECEF = np.array([xecef[k], yecef[k], zecef[k]]) * 1e3
        vECEF = np.array([vxecef[k], vyecef[k], vzecef[k]])

        northECEF = getTrueNorthInECEF(lats[k], lons[k])

        vSurface = projectVelo2Surface(rECEF, vECEF)

        head[k] = np.mod(np.degrees(np.arctan2(vSurface[1], vSurface[0]) - np.arctan2(northECEF[1], northECEF[0])),360)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(head)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Heading Angle (degrees)')
    ax.set_title('Satellite Heading Angle Over Time')
    plt.savefig('landsat8_heading.png')
    print("Plot saved as landsat8_heading.png")

    return head

def projectVelo2Surface(rECEF, vECEF):

    rNorm = rECEF / np.linalg.norm(rECEF)
    vRad = np.dot(vECEF, rNorm) * rNorm

    return vECEF - vRad


def getTrueNorthInECEF(lat, lon):
    lat = np.radians(lat)
    lon = np.radians(lon)

    N = wgsSMA / np.sqrt(1 - (2 * wgsf - wgsf**2) * np.sin(lat)**2)

    x = -(N * np.cos(lat) * np.sin(lon))
    y = N * np.cos(lat) * np.cos(lon)
    z = (N * (1 - (2 * wgsf - wgsf**2)) * np.sin(lat))
    
    # Normalize the position vector
    rECEF = np.array([x, y, z])
    rNorm = rECEF / np.linalg.norm(rECEF)
    
    # True north direction in ECEF frame
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])

    north = np.cross(rNorm, east)
    north = north / np.linalg.norm(north)

    return north

def getSatLVLHHeading(stateECEF):

    # Extract state vectors into individual components in the ECEF frame, in km and km/s
    xecef = stateECEF['geometry'].apply(lambda p: p.x).values / 1e3
    yecef = stateECEF['geometry'].apply(lambda p: p.y).values / 1e3
    zecef = stateECEF['geometry'].apply(lambda p: p.z).values / 1e3
    vxecef = stateECEF['velocity'].apply(lambda v: v.x).values / 1e3
    vyecef = stateECEF['velocity'].apply(lambda v: v.y).values / 1e3
    vzecef = stateECEF['velocity'].apply(lambda v: v.z).values / 1e3

    head = np.zeros(len(xecef))

    # Loop that will calculate the heading angle at each time step
    for k in range(len(head)):

        # Create position and velocity vectors in ECEF
        rECEF = np.array([xecef[k], yecef[k], zecef[k]])
        vECEF = np.array([vxecef[k], vyecef[k], vzecef[k]])

        # Get transformation matrix from ECEF to the LVLH frame
        T = ECEF2LVLH(rECEF, vECEF)

        # Find local north transformation matrix in the LVLH frame
        northLVLH = getLocalNorthInLVLH(rECEF)

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

def ECEF2LVLH(rECEF, vECEF):
        
    # Determine the transformation matrix to convert the ECEF frame to the LVLH frame
    ez = -rECEF / np.linalg.norm(rECEF)

    ey = -np.cross(rECEF, vECEF)
    ey /= np.linalg.norm(ey)

    ex = np.cross(ey, ez)
    ex /= np.linalg.norm(ex)

    return np.array([ex, ey, ez])

def getLocalNorthInLVLH(rECEF):
     
    # Determine direction to the north pole, due north, in the LVLH frame local to the satellite
    k = np.array([0, 0, 1])

    ez = -rECEF / np.linalg.norm(rECEF)

    northECEF = k - np.dot(k, ez) * ez
    northECEF /= np.linalg.norm(northECEF)

    T = ECEF2LVLH(rECEF, np.array([1, 0, 0]))

    northLVLH = T @ northECEF

    return northLVLH

def getGeocentricLatLon(stateECEF):

    xecef = stateECEF['geometry'].apply(lambda p: p.x).values / 1e3
    yecef = stateECEF['geometry'].apply(lambda p: p.y).values / 1e3
    zecef = stateECEF['geometry'].apply(lambda p: p.z).values / 1e3
    
    lats = np.zeros((len(xecef),1))
    lons = np.zeros((len(xecef),1))

    for k in range(len(lats)):
        rTemp = np.array([xecef[k], yecef[k], zecef[k]])
        rnorm = np.linalg.norm(rTemp)
        lat = np.degrees(np.arcsin(zecef[k] / rnorm))
        xyvec = rTemp[:2]
        ecix = np.array([1, 0])
        lon = np.degrees(np.arccos(np.dot(ecix,xyvec) / np.linalg.norm(xyvec)))
        lats[k] = lat
        lons[k] = lon
    
    geoc = np.array([lats, lons])

    return geoc

def plotNadirBaseArray(state84, sens, times, alt):

    alts84 = state84['geometry'].apply(lambda p: p.z).values
    time84 = pd.Series(times)

    IFOVbase = sens[0] / sens[2]
    angVec = np.linspace(-sens[0]/2, sens[0]/2, int(sens[2]))

    GSDsNow = np.zeros(len(angVec))
    for k in range(len(angVec)):
        rangeSat = alt / np.cos(angVec[k])
        GSDSat = IFOVbase * rangeSat
        GSDsNow[k] = GSDSat / np.cos(angVec[k])

    # Plot GSD over array 
    fig, ax = plt.subplots(figsize=(8, 5))
    xvals = np.linspace(0, len(GSDsNow), len(GSDsNow))
    ax.plot(np.rad2deg(angVec), GSDsNow*1e3, linewidth=2)
    ax.set_xlabel('FOV Span (degrees)')
    ax.set_ylabel('GSD (m)')
    ax.set_title(f'Pixel-by-Pixel GSD of Landsat-8 OLI (Nadir) at Altitude: {alt} km')
    ax.grid(True, alpha=0.3)
    plt.savefig('fov_skew_array_complex.png')
    print("Plot saved as fov_skew_array_complex.png")


def plotAltitudeTimeseries(state84, times):

    alts84 = state84['geometry'].apply(lambda p: p.z).values
    time84 = pd.Series(times)

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

def getCamCorners(self, fovCross, fovAlong):
    hx = (fovCross / 2)
    hy = (fovAlong / 2)
    cornersGeo = np.array([[-np.tan(hx), -np.tan(hy), 1.0],
                        [np.tan(hx), -np.tan(hy), 1.0],
                        [np.tan(hx), np.tan(hy), 1.0],
                        [-np.tan(hx), np.tan(hy), 1.0]])
    corners = cornersGeo / np.linalg.norm(cornersGeo, axis = 1, keepdims = True)
    return np.array(corners)

def getRayIntersect(self, oECEF, dECEF):
    ox, oy, oz = oECEF
    dx, dy, dz = dECEF
    a2 = (self.eqAx)**2
    b2 = (self.poleAx)**2
    qa = dx**2/a2 + dy**2/a2 + dz**2/b2
    qb = 2*(ox*dx/a2 + oy*dy/a2 + oz*dz/b2)
    qc = ox**2/a2 + oy**2/a2 + oz**2/b2 - 1
    disc = qb**2 - 4*qa*qc
    if disc < 0:
        return None
    t = (-qb - np.sqrt(disc)) / (2*qa)
    if t < 0:
        t = (-qb + np.sqrt(disc)) / (2*qa)
    if t < 0:
        return None
    return oECEF + t*dECEF

def angular2km(self, latlon1, latlon2):
    lat1, lon1 = np.radians(latlon1)
    lat2, lon2 = np.radians(latlon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * self.earthR * np.arcsin(np.sqrt(a))

if __name__ == "__main__":
    #   Corners of Image defined in imCorner below in this order [Up Left Lat, Up Left Lon, Up Right Lat, Up Right Lon, Bot Right Lat, Bot Right Lon, Bot Left Lat, Bot Left Lon]
    runTest()