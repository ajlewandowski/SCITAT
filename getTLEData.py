from datetime import datetime, timedelta, timezone
from sgp4.api import Satrec
from pathlib import Path


class getTLEData:
    def __init__(self, time, noradID):
        self.time = time
        self.noradID = noradID

    def tle2oe(self, line1, line2):
        """Convert TLE to classical orbital elements array"""
        sat = Satrec.twoline2rv(line1, line2)

        year = sat.epochyr
        if year < 57:
            year += 2000
        else:
            year += 1900
        
        # epochdays is day of year with fractional component
        epoch = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=sat.epochdays - 1)

        coe = (sat.a * 6378.137, sat.ecco, sat.inclo, sat.nodeo, sat.argpo, sat.mo)

        return coe, epoch.strftime('%Y-%m-%dT%H:%M:%S')
    
    def getTimeData(self, epochUTC, endEpochUTC):

        ep1 = datetime.fromisoformat(epochUTC).replace(tzinfo=timezone.utc)
        ep2 = datetime.fromisoformat(endEpochUTC).replace(tzinfo=timezone.utc)
        deltaT = ep2 - ep1
        tSpan = deltaT.total_seconds()

        return ep1, ep2, deltaT, tSpan
    
    def printTLE(self, tle):

        print()
        print("TLE Obtained from Space-Tracker")
        print(tle[1])  # Line 1
        print(tle[2])  # Line 2  
