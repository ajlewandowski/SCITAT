"""
This file tests the flat-earth assumption versus the spherical earth assumption
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from imageGridding import imageGridding
from tatc import utils
from tatc.schemas import Instrument, Satellite, TwoLineElements
import pandas as pd
from getTLEData import getTLEData
from tatc.analysis import collect_orbit_track, OrbitCoordinate, OrbitOutput

def getRSquared(actual, predicted):
    ssRes = np.sum((actual - predicted) ** 2)
    ssTot = np.sum((actual - np.mean(predicted)) ** 2)
    return 1 - ssRes/ssTot

def runTest():
    print()
    print("###################################################################################################")
    print()

    # This step is defining parameters of Landsat-8 and its orbit, as well as finding the closest epoch to the desired
    # image time (imTime). This section will collect the ground track of the orbit.
    imTime = ["2025-12-30T17:00:00", "2025-12-30T17:03:24"]
    tlePath = Path(r"D:\Users\al62352\SpaceTrackTLE\tle49260.txt")
    tles = getTLEData(tlePath)
    line1, line2, closeTime = tles.printTLE(imTime[0])
    print()

    #                  FOV, Focal, Pixel #, GSD, Swath, Alt
    sens = np.array([np.deg2rad(15.0), 0.886, 6196, 30, 185000, 705000])
    target = np.array([30.30617, -97.88121])

    oli = Instrument(name="OLI",field_of_regard=utils.swath_width_to_field_of_regard(705000, 185000),req_target_sunlit=True)
    L8 = Satellite(name="Landsat-8",
        orbit=TwoLineElements(tle=[line1,line2]), instruments=[oli])
    times = pd.date_range(start=closeTime, end=imTime[1], freq="1s", tz="UTC")
    state84 = collect_orbit_track(L8, times, coordinates = OrbitCoordinate.WGS84, sat_sunlit = True, solar_altaz = True, solar_beta = True)
    stateECEF = collect_orbit_track(L8, times, orbit_output = OrbitOutput.POSITION_VELOCITY, coordinates = OrbitCoordinate.ECEF)    
    angVecs = np.array([0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0])

    sens = np.array([np.deg2rad(90.0), 0.886, 6*6196, 30, 185000, 705000])
    gridded = imageGridding("49260", state84, stateECEF, times, sens, target)

    altVec = np.linspace(300, 10000, 5000)
    percentLimit = 1.0
    degVals = np.zeros(len(altVec))
    for i in range(len(altVec)):
        errVec, degVec = gridded.plotNadirBaseArray(altVec[i], plotSave=False)
        half = len(errVec) // 2
        aboveInd = np.where(errVec[half:] > percentLimit)[0]

        degVals[i] = degVec[half + aboveInd[0]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(altVec, degVals, linewidth=2, color='r')
    ax.set_xlabel('Altitude (km)')
    ax.set_ylabel(f'Satellite Look-Angle Where % Error Crosses {percentLimit}%')
    ax.set_title(f'Percent Error Threshold Crossing versus Altitude')
    ax.grid(True, alpha=0.3)
    plt.savefig(f'threshold_crossings_of_flat_model_vs_alt.png')

    coeffsLog = np.polyfit(np.log(altVec), degVals, 1)
    yFitLog = coeffsLog[0] * np.log(altVec) + coeffsLog[1]
    print('For the log fit y = a ln(x) + b:')
    print(f'a = {coeffsLog[0]}')
    print(f'b = {coeffsLog[1]}')
    print()

    coeffsPower = np.polyfit(np.log(altVec), np.log(degVals), 1)
    yFitPower = np.exp(coeffsPower[1]) * np.power(altVec, coeffsPower[0])
    print('For the power fit y = a * x^b')
    print(f'a = exp(coeff) = {np.exp(coeffsPower[1])}')
    print(f'b = {coeffsPower[0]}')
    print()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(altVec, degVals, linewidth=2, color='r', label='Data')
    ax.plot(altVec, yFitLog, linewidth=2, color='b', label='Log Curve-Fit')
    ax.plot(altVec, yFitPower, linewidth=2, color='k', label='Power Law Curve-Fit')
    ax.set_xlabel('Altitude (km)')
    ax.set_ylabel(f'Satellite Look-Angle Where % Error Crosses {percentLimit}%')
    ax.set_title(f'Percent Error Threshold Crossing versus Altitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('data_fits_to_threshold_limit_model_log_and_power_fit.png')

    logRS = getRSquared(degVals, yFitLog)
    powerRS = getRSquared(degVals, yFitPower)
    print(f'R-Squared Value for the Log Fit: {logRS}')
    print(f'R-Squared Value for the Power Fit: {powerRS}')

    print()


if __name__ == "__main__":
    runTest()