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

    lon84 = state84.geometry.x
    lat84 = state84.geometry.y
    alt = state84.geometry.z / 1e3
    azi = state84.solar_az
    elev = state84.solar_alt
    
    head = getSatLVLHHeading(stateECI)
    lonCen, latCen = getGeocentricLatLon(stateECEF)

    # fig, ax = plt.subplots(figsize=(8, 5))
    # xvals = np.linspace(0, len(lat), len(lat))
    # ax.plot(xvals, head, linewidth=2)
    # ax.legend()
    # ax.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()

def getSatLVLHHeading(stateECI):

    xeci = stateECI.geometry.x / 1e3
    yeci = stateECI.geometry.y / 1e3
    zeci = stateECI.geometry.z / 1e3
    vxeci = stateECI['velocity'].apply(lambda v: v.x) / 1e3
    vyeci = stateECI['velocity'].apply(lambda v: v.y) / 1e3
    vzeci = stateECI['velocity'].apply(lambda v: v.z) / 1e3
    state = np.vstack([xeci, yeci, zeci, vxeci, vyeci, vzeci])
    head = np.zeros((len(state[0,:]),1))

    for k in range(len(head)):
        rECI = state[:3,k]
        vECI = state[3:,k]

        rNPsat = np.array([0 - rECI[0], 0 - rECI[1], 6356.7523 - rECI[2]])

        eci2lvlh = ECI2LVLH(rECI, vECI)

        rNP = eci2lvlh @ rNPsat
        vSat = eci2lvlh @ vECI
        vSatNorm = np.linalg.norm(vSat)
        rNPNorm = np.linalg.norm(rNP)

        theta = np.arccos(np.dot(rNP, (vSat / vSatNorm)) / (vSatNorm * rNPNorm))

        head[k] = np.degrees(theta) % 360

    return head

def ECI2LVLH(rECI, vECI):
        r = rECI / np.linalg.norm(rECI)
        v = vECI / np.linalg.norm(vECI)
        
        ez = -r
        ex = v - np.dot(v, ez) * ez
        ex /= np.linalg.norm(ex)
        ey = np.cross(ez, ex)
        ey /= np.linalg.norm(ey)

        return np.array([ex, ey, ez])

def getGeocentricLatLon(coords):

    xecef = coords.geometry.x / 1e3
    yecef = coords.geometry.y / 1e3
    zecef = coords.geometry.z / 1e3

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