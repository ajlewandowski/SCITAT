"""
USGS Test Files
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tatc import utils
from tatc.schemas import Instrument, Satellite, TwoLineElements
import pandas as pd
from getTLEData import getTLEData
from tatc.analysis import collect_orbit_track

def runTest1():
    print()
    print("###################################################################################################")
    print()
    username = "alec.lewandowski@utexas.edu"
    password = "!Lew3dglatutaustin"
    imTime = ["2025-12-22T17:00:00", "2025-12-22T17:03:15"]
    noradID = 39084
    sens = np.array([0.886, 6029, 6196, 30*0.886/705000])
    solarAngles = np.array([32.34603894, 156.26970520])
    target = np.array([30.30635, -97.86956])
    imCorner = np.array([[31.35457, -99.05888], [31.33324, -96.65936], [29.23683, -96.70863],[29.25645, -99.05764]])

    tlePath = Path(r"D:\Users\al62352\SpaceTrackTLE\tle39084.txt")
    tles = getTLEData(tlePath)
    line1, line2, closeTime = tles.printTLE(imTime[0])
    print()

    oli = Instrument(name="OLI",field_of_regard=utils.swath_width_to_field_of_regard(705000, 185000),req_target_sunlit=True)
    L8 = Satellite(name="Landsat-8",
        orbit=TwoLineElements(tle=[line1,line2]),instruments=[oli])
    times = pd.date_range(start=closeTime, end=imTime[1], freq="1s", tz="UTC")
    results = collect_orbit_track(L8, times, sat_sunlit = True, solar_altaz = True, solar_beta = True)
    lon = results.geometry.x
    lat = results.geometry.y
    print()
    print(f"Image scene comparison:")
    print(f"    Propagated Final Lat: {lat.iloc[-1]:.6f}°  and Target Lat: {target[0]:.6f}°")
    print(f"    Propagated Final Lon: {lon.iloc[-1]:.6f}°  and Target Lat: {target[1]:.6f}°")
    print(f"    Scene Center Latitude Error: {abs(lat.iloc[-1] - target[0]):.6f}°")
    print(f"    Scene Center Longitude Error: {abs(lon.iloc[-1] - target[1]):.6f}°")
    azi = results.solar_az
    elev = results.solar_alt
    print(f"    Solar azimuth angle error: {abs(azi.iloc[-1] - solarAngles[1]):.6f}°")
    print(f"    Solar azimuth angle error: {abs(elev.iloc[-1] - solarAngles[0]):.6f}°")
    errsLat.append(abs(lat.iloc[-1] - target[0]))
    errsLon.append(abs(lon.iloc[-1] - target[1]))
    errsAzi.append(abs(azi.iloc[-1] - solarAngles[1]))
    errsElev.append(abs(elev.iloc[-1] - solarAngles[0]))

def runTest2():
    print()
    print("###################################################################################################")
    print()
    username = "alec.lewandowski@utexas.edu"
    password = "!Lew3dglatutaustin"
    imTime = ["2025-12-06T17:00:00", "2025-12-06T17:03:16"]
    noradID = 39084
    sens = np.array([0.886, 6029, 6196, 30*0.886/705000])
    solarAngles = np.array([33.78604596, 157.84150300])
    target = np.array([30.30633, -97.86138])
    imCorner = np.array([[31.35728, -99.04942],[31.33577, -96.64984],[29.23667, -96.69938],[29.25645, -99.04837]])

    tlePath = Path(r"D:\Users\al62352\SpaceTrackTLE\tle39084.txt")
    tles = getTLEData(tlePath)
    line1, line2, closeTime = tles.printTLE(imTime[0])
    print()

    oli = Instrument(name="OLI",field_of_regard=utils.swath_width_to_field_of_regard(705000, 185000),req_target_sunlit=True)
    L8 = Satellite(name="Landsat-8",
        orbit=TwoLineElements(tle=[line1,line2]),instruments=[oli])
    times = pd.date_range(start=closeTime, end=imTime[1], freq="1s", tz="UTC")
    results = collect_orbit_track(L8, times, sat_sunlit = True, solar_altaz = True, solar_beta = True)
    lon = results.geometry.x
    lat = results.geometry.y
    print()
    print(f"Image scene comparison:")
    print(f"    Propagated Final Lat: {lat.iloc[-1]:.6f}°  and Target Lat: {target[0]:.6f}°")
    print(f"    Propagated Final Lon: {lon.iloc[-1]:.6f}°  and Target Lat: {target[1]:.6f}°")
    print(f"    Scene Center Latitude Error: {abs(lat.iloc[-1] - target[0]):.6f}°")
    print(f"    Scene Center Longitude Error: {abs(lon.iloc[-1] - target[1]):.6f}°")
    azi = results.solar_az
    elev = results.solar_alt
    print(f"    Solar azimuth angle error: {abs(azi.iloc[-1] - solarAngles[1]):.6f}°")
    print(f"    Solar azimuth angle error: {abs(elev.iloc[-1] - solarAngles[0]):.6f}°")
    errsLat.append(abs(lat.iloc[-1] - target[0]))
    errsLon.append(abs(lon.iloc[-1] - target[1]))
    errsAzi.append(abs(azi.iloc[-1] - solarAngles[1]))
    errsElev.append(abs(elev.iloc[-1] - solarAngles[0]))

def runTest3():
    print()
    print("###################################################################################################")
    print()
    username = "alec.lewandowski@utexas.edu"
    password = "!Lew3dglatutaustin"
    imTime = ["2025-11-20T17:00:00", "2025-11-20T17:03:18"]
    noradID = 39084
    sens = np.array([0.886, 6029, 6196, 30*0.886/705000])
    solarAngles = np.array([36.78359000, 158.15761834])
    target = np.array([30.30615, -97.86711])
    imCorner = np.array([[31.35457, -99.05572],[31.33318, -96.65620],[29.23677, -96.70554],[29.25645, -99.05455]])

    tlePath = Path(r"D:\Users\al62352\SpaceTrackTLE\tle39084.txt")
    tles = getTLEData(tlePath)
    line1, line2, closeTime = tles.printTLE(imTime[0])
    print()

    oli = Instrument(name="OLI",field_of_regard=utils.swath_width_to_field_of_regard(705000, 185000),req_target_sunlit=True)
    L8 = Satellite(name="Landsat-8",
        orbit=TwoLineElements(tle=[line1,line2]),instruments=[oli])
    times = pd.date_range(start=closeTime, end=imTime[1], freq="1s", tz="UTC")
    results = collect_orbit_track(L8, times, sat_sunlit = True, solar_altaz = True, solar_beta = True)
    lon = results.geometry.x
    lat = results.geometry.y
    print()
    print(f"Image scene comparison:")
    print(f"    Propagated Final Lat: {lat.iloc[-1]:.6f}°  and Target Lat: {target[0]:.6f}°")
    print(f"    Propagated Final Lon: {lon.iloc[-1]:.6f}°  and Target Lat: {target[1]:.6f}°")
    print(f"    Scene Center Latitude Error: {abs(lat.iloc[-1] - target[0]):.6f}°")
    print(f"    Scene Center Longitude Error: {abs(lon.iloc[-1] - target[1]):.6f}°")
    azi = results.solar_az
    elev = results.solar_alt
    print(f"    Solar azimuth angle error: {abs(azi.iloc[-1] - solarAngles[1]):.6f}°")
    print(f"    Solar azimuth angle error: {abs(elev.iloc[-1] - solarAngles[0]):.6f}°")
    errsLat.append(abs(lat.iloc[-1] - target[0]))
    errsLon.append(abs(lon.iloc[-1] - target[1]))
    errsAzi.append(abs(azi.iloc[-1] - solarAngles[1]))
    errsElev.append(abs(elev.iloc[-1] - solarAngles[0]))

def runTest4():
    print()
    print("###################################################################################################")
    print()
    username = "alec.lewandowski@utexas.edu"
    password = "!Lew3dglatutaustin"
    imTime = ["2025-11-04T17:00:00", "2025-11-04T17:03:18"]
    noradID = 39084
    sens = np.array([0.886, 6029, 6196, 30*0.886/705000])
    solarAngles = np.array([40.95210647, 156.88012216])
    target = np.array([30.30640, -97.86433])
    imCorner = np.array([[31.35728, -99.05257],[31.33582, -96.65299],[29.23672, -96.70246],[29.25645, -99.05146]])

    tlePath = Path(r"D:\Users\al62352\SpaceTrackTLE\tle39084.txt")
    tles = getTLEData(tlePath)
    line1, line2, closeTime = tles.printTLE(imTime[0])
    print()

    oli = Instrument(name="OLI",field_of_regard=utils.swath_width_to_field_of_regard(705000, 185000),req_target_sunlit=True)
    L8 = Satellite(name="Landsat-8",
        orbit=TwoLineElements(tle=[line1,line2]),instruments=[oli])
    times = pd.date_range(start=closeTime, end=imTime[1], freq="1s", tz="UTC")
    results = collect_orbit_track(L8, times, sat_sunlit = True, solar_altaz = True, solar_beta = True)
    lon = results.geometry.x
    lat = results.geometry.y
    print()
    print(f"Image scene comparison:")
    print(f"    Propagated Final Lat: {lat.iloc[-1]:.6f}°  and Target Lat: {target[0]:.6f}°")
    print(f"    Propagated Final Lon: {lon.iloc[-1]:.6f}°  and Target Lat: {target[1]:.6f}°")
    print(f"    Scene Center Latitude Error: {abs(lat.iloc[-1] - target[0]):.6f}°")
    print(f"    Scene Center Longitude Error: {abs(lon.iloc[-1] - target[1]):.6f}°")
    azi = results.solar_az
    elev = results.solar_alt
    print(f"    Solar azimuth angle error: {abs(azi.iloc[-1] - solarAngles[1]):.6f}°")
    print(f"    Solar azimuth angle error: {abs(elev.iloc[-1] - solarAngles[0]):.6f}°")
    errsLat.append(abs(lat.iloc[-1] - target[0]))
    errsLon.append(abs(lon.iloc[-1] - target[1]))
    errsAzi.append(abs(azi.iloc[-1] - solarAngles[1]))
    errsElev.append(abs(elev.iloc[-1] - solarAngles[0]))

def runTest5():
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

    tlePath = Path(r"D:\Users\al62352\SpaceTrackTLE\tle49260.txt")
    tles = getTLEData(tlePath)
    line1, line2, closeTime = tles.printTLE(imTime[0])
    print()

    oli = Instrument(name="OLI",field_of_regard=utils.swath_width_to_field_of_regard(705000, 185000),req_target_sunlit=True)
    L8 = Satellite(name="Landsat-8",
        orbit=TwoLineElements(tle=[line1,line2]),instruments=[oli])
    times = pd.date_range(start=closeTime, end=imTime[1], freq="1s", tz="UTC")
    results = collect_orbit_track(L8, times, sat_sunlit = True, solar_altaz = True, solar_beta = True)
    lon = results.geometry.x
    lat = results.geometry.y
    print()
    print(f"Image scene comparison:")
    print(f"    Propagated Final Lat: {lat.iloc[-1]:.6f}°  and Target Lat: {target[0]:.6f}°")
    print(f"    Propagated Final Lon: {lon.iloc[-1]:.6f}°  and Target Lat: {target[1]:.6f}°")
    print(f"    Scene Center Latitude Error: {abs(lat.iloc[-1] - target[0]):.6f}°")
    print(f"    Scene Center Longitude Error: {abs(lon.iloc[-1] - target[1]):.6f}°")
    azi = results.solar_az
    elev = results.solar_alt
    print(f"    Solar azimuth angle error: {abs(azi.iloc[-1] - solarAngles[1]):.6f}°")
    print(f"    Solar azimuth angle error: {abs(elev.iloc[-1] - solarAngles[0]):.6f}°")
    errsLat.append(abs(lat.iloc[-1] - target[0]))
    errsLon.append(abs(lon.iloc[-1] - target[1]))
    errsAzi.append(abs(azi.iloc[-1] - solarAngles[1]))
    errsElev.append(abs(elev.iloc[-1] - solarAngles[0]))
    print()

def plotResults(errsLat, errsLon, errsAzi, errsElev):
    # All data parsed into LandSat8 first 5, LandSat9 second 5
    propTimes = np.array([141.83, 636.05, 141.63, 636.02, 141.60, 735.07, 141.72, 42.77, 339.73, 734.78])
    cenLatsP = np.array([0.007459, 0.204940, 0.008185, 0.205400, 0.043606, 0.448584, 0.011274, 0.015619, 0.056479, 0.306703])
    cenLonsP = np.array([0.125697, 0.715103, 0.052271, 0.669167, 0.088114, 0.817786, 0.012238, 0.031745, 0.266943, 0.843618])
    frontLatErrsP = np.array([0.149062, 0.201937, 0.034970, 0.264121, 0.030009, 0.279967, 0.151926, 0.119385, 0.135958, 0.284863])
    backLatErrsP = np.array([0.021371, 0.370838, 0.206818, 0.432597, 0.202031, 0.446354, 0.015914, 0.048022, 0.031700, 0.450888])
    leftLonErrsP = np.array([0.289655, 0.865063, 0.108020, 0.848396, 0.107649, 0.993459, 0.145028, 0.176910, 0.459262, 0.993141])
    rightLonErrsP = np.array([0.488207, 1.053902, 0.300905, 1.035322, 0.300839, 1.180767, 0.340678, 0.374453, 0.657482, 1.176787])
    aziP = np.array([0.256134, 0.856826, 0.059544, 0.898192, 0.044632, 0.961142, 0.099620, 0.129209, 0.474340, 0.969874])
    elevP = np.array([0.118166, 0.505794, 0.024687, 0.472951, 0.033165, 0.899597, 0.030157, 0.036706, 0.267983, 0.665377])

    errsLat = np.array(errsLat)
    errsLon = np.array(errsLon)
    errsAzi = np.array(errsAzi)
    errsElev = np.array(errsElev)

    plt.scatter(propTimes, cenLatsP, label='Old Center Lat Errs')
    plt.scatter(propTimes, cenLonsP, label='Old Center Lon Errs')
    plt.scatter(propTimes, aziP, label='Old Solar Azimuth Errs')
    plt.scatter(propTimes, elevP, label='Old Solar Elevation Errs')
    plt.scatter(propTimes, errsLat, label='New Center Lat Errs')
    plt.scatter(propTimes, errsLon, label='New Center Lon Errs')
    plt.scatter(propTimes, errsAzi, label ='New Solar Azimuth Errs')
    plt.scatter(propTimes, errsElev, label ='New Solar Elev Errs')
    plt.xlabel("Propagation Time")
    plt.ylabel("Error (deg)")
    plt.tight_layout()
    plt.title('Projected FOV test on USGS Imagery')
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    #   Corners of Image defined in imCorner below in this order [Up Left Lat, Up Left Lon, Up Right Lat, Up Right Lon, Bot Right Lat, Bot Right Lon, Bot Left Lat, Bot Left Lon]
    errsAzi = []
    errsElev = []
    errsLat = []
    errsLon = []
    runTest1()
    runTest2()
    runTest3()
    runTest4()
    runTest5()
    plotResults(errsLat, errsLon, errsAzi, errsElev)