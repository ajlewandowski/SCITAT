"""
Gridding adn resolution heat map test file
"""

import numpy as np
import matplotlib.pyplot as plt
from tatc import utils
from tatc.schemas import Instrument, Satellite, TwoLineElements
import pandas as pd
from SpaceTrackAPIClass import SpaceTrackAPI
from tatc.analysis import collect_orbit_track, OrbitCoordinate, OrbitOutput

def runTest():
    print()
    print("###################################################################################################")
    print()
    username = "alec.lewandowski@utexas.edu"
    password = "!Lew3dglatutaustin"
    imTime = ["2025-12-30T17:00:00", "2025-12-30T17:03:24"]
    noradID = 49260
    sens = np.array([0.886, 6029, 6196, 30*0.886/705000])
    solarAngles = np.array([32.30026603, 155.16461405])
    target = np.array([30.30617, -97.88121])
    imCorner = np.array([[31.35457, -99.06834],[31.33346, -96.67196],[29.23974, -96.72091],[29.25915, -99.06690]])

    data = SpaceTrackAPI(username, password)
    tles, closeTime = data.getTLEatEpoch(noradID, imTime)

    oli = Instrument(name="OLI",field_of_regard=utils.swath_width_to_field_of_regard(705000, 185000),req_target_sunlit=True)
    L8 = Satellite(name="Landsat-8",
        orbit=TwoLineElements(tle=[tles[1],tles[2]]), instruments=[oli])
    times = pd.date_range(start=closeTime, end=imTime[1], freq="1s", tz="UTC")
    state84 = collect_orbit_track(L8, times, coordinates = OrbitCoordinate.WGS84, sat_sunlit = True, solar_altaz = True, solar_beta = True)
    stateECI = collect_orbit_track(L8, times, orbit_output = OrbitOutput.POSITION_VELOCITY, coordinates = OrbitCoordinate.ECI)
    stateECEF = collect_orbit_track(L8, times, orbit_output = OrbitOutput.POSITION_VELOCITY, coordinates = OrbitCoordinate.ECEF)    
    
    head = getSatLVLHHeading(stateECEF)
    lonCen, latCen = getGeocentricLatLon(stateECEF)

    for j in range(len(head)):
        print(head[j])
        print(lonCen[j])
        print(latCen[j])

    fig, ax = plt.subplots(figsize=(8, 5))
    xvals = np.linspace(0, len(times), len(times))
    ax.plot(xvals, head, linewidth=2)
    ax.grid(True, alpha=0.3)
    plt.savefig('satellite_heading_graphic.png')
    print("Plot saved as satellite_heading_graphic.png")

def getSatLVLHHeading(stateECEF):

    xecef = stateECEF['geometry'].apply(lambda p: p.x).values / 1e3
    yecef = stateECEF['geometry'].apply(lambda p: p.y).values / 1e3
    zecef = stateECEF['geometry'].apply(lambda p: p.z).values / 1e3
    vxecef = stateECEF['velocity'].apply(lambda v: v.x).values / 1e3
    vyecef = stateECEF['velocity'].apply(lambda v: v.y).values / 1e3
    vzecef = stateECEF['velocity'].apply(lambda v: v.z).values / 1e3
    
    head = np.zeros(len(xecef))

    for k in range(len(head)):

        rECEF = np.array([xecef[k], yecef[k], zecef[k]])
        vECEF = np.array([vxecef[k], vyecef[k], vzecef[k]])

        T = ECEF2LVLH(rECEF, vECEF)

        northLVLH = getLocalNorthInLVLH(rECEF)

        vLVLH = T @ vECEF
        vLVLHnorm = vLVLH / np.linalg.norm(vLVLH)
    
        angle = np.degrees(np.arctan2(northLVLH[1], northLVLH[0]))
        velAngle = np.degrees(np.arctan2(vLVLHnorm[1], vLVLHnorm[0]))

        head[k] = np.degrees(velAngle - angle) % 360

    return head

def ECEF2LVLH(rECEF, vECEF):
        
    ez = -rECEF / np.linalg.norm(rECEF)

    ey = -np.cross(rECEF, vECEF)
    ey /= np.linalg.norm(ey)

    vNorm = np.linalg.norm(vECEF)
    if vNorm == 0:
        ex = np.array([1, 0, 0])
    else:
        ex = np.cross(ey, ez)
        ex /= np.linalg.norm(ex)
    
    # ey = np.cross(ez, ex)
    # eyNorm = np.linalg.norm(ey)
    # if eyNorm == 0:
    #     ex = np.array([1, 0, 0])
    #     ey = np.cross(ez, ex)

    # ey /= np.linalg.norm(ey)

    return np.array([ex, ey, ez])

def getLocalNorthInLVLH(rECEF):
     
    k = np.array([0, 0, 1])

    ez = -rECEF / np.linalg.norm(rECEF)

    northECEF = k - np.dot(k, ez) * ez
    northECEF /= np.linalg.norm(northECEF)

    T = ECEF2LVLH(rECEF, np.zeros_like(rECEF))

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

    return lons, lats


if __name__ == "__main__":
    #   Corners of Image defined in imCorner below in this order [Up Left Lat, Up Left Lon, Up Right Lat, Up Right Lon, Bot Right Lat, Bot Right Lon, Bot Left Lat, Bot Left Lon]
    runTest()