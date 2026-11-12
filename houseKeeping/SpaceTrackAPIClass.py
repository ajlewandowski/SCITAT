import requests
from datetime import datetime, timedelta, timezone
import numpy as np
from sgp4.api import Satrec

class SpaceTrackAPI:
    def __init__(self, username, password):
        self.baseURL = "https://www.space-track.org"
        self.session = requests.Session()
        self.login(username, password)

    def login(self, username, password):
        loginURL = f"{self.baseURL}/ajaxauth/login"
        data = {'identity': username, 'password': password}
        response = self.session.post(loginURL, data = data)
        if response.status_code != 200:
            raise Exception("Login Failed")
        
    def getTLEatEpoch(self, noradID, epochs):

        epochUTC = epochs[0]
        endEpoch = epochs[1]
        end_dt = datetime.fromisoformat(endEpoch)

        # Parse datetime if string and ensure UTC
        if isinstance(epochUTC, str):
            epoch_dt = datetime.fromisoformat(epochUTC)
            if epoch_dt.tzinfo is None:
                epoch_dt = epoch_dt.replace(tzinfo=timezone.utc)
        else:
            epoch_dt = epochUTC
            if epoch_dt.tzinfo is None:
                epoch_dt = epoch_dt.replace(tzinfo=timezone.utc)
        
        # Create search window
        start_dt = epoch_dt - timedelta(hours=24)
        
        # Format for Space-Track API (they use YYYY-MM-DD HH:MM:SS format in UTC)
        start_str = start_dt.strftime('%Y-%m-%d %H:%M:%S')
        end_str = end_dt.strftime('%Y-%m-%d %H:%M:%S')
        
        query_url = (
            f"{self.baseURL}/basicspacedata/query/class/gp_history/"
            f"NORAD_CAT_ID/{noradID}/"
            f"EPOCH/{start_str}--{end_str}/"
            f"orderby/EPOCH asc/format/tle"
        )
        
        api_response = self.session.get(query_url)
    
        if api_response.status_code == 200:
            tle_data = api_response.text.strip()
                        
            tle_lines = [line.strip() for line in tle_data.split('\n') if line.strip()]
            # ALWAYS find closest TLE
            if len(tle_lines) >= 3:
                return self.findClosestTLE(tle_lines, epoch_dt)
            elif len(tle_lines) == 2:
                tle_lines = [f"NORAD {noradID}"] + tle_lines
                return tle_lines
            else:
                raise Exception(f"Invalid TLE data: only {len(tle_lines)} lines")
        else:
            raise Exception(f"Query failed: {api_response.status_code}")
        
       
    def findClosestTLE(self, tleLines, targetEpoch):
        from skyfield.api import EarthSatellite, load
        
        ts = load.timescale()
        closest_tle = None
        min_diff = float('inf')
        
        if targetEpoch.tzinfo is None:
            targetEpoch = targetEpoch.replace(tzinfo=timezone.utc)
        
        # Parse into (name, line1, line2) groups by detecting actual TLE lines
        groups = []
        name = None
        line1 = None

        for line in tleLines:
            if line.startswith('1 ') and len(line) >= 69:
                line1 = line
                name = name or "UNKNOWN"
            elif line.startswith('2 ') and len(line) >= 69 and line1 is not None:
                groups.append((name, line1, line))
                name = None
                line1 = None
            else:
                # It's a name line — only update name if we haven't seen a line1 yet
                if line1 is None:
                    name = line

        for name, line1, line2 in groups:
            try:
                sat = EarthSatellite(line1, line2, name, ts)
                tle_epoch = sat.epoch.utc_datetime()
                
                if tle_epoch.tzinfo is None:
                    tle_epoch = tle_epoch.replace(tzinfo=timezone.utc)
                
                diff = abs((tle_epoch - targetEpoch).total_seconds())
                
                if diff < min_diff:
                    min_diff = diff
                    closest_tle = [name, line1, line2]
                    closest_epoch = tle_epoch
            
            except Exception as e:
                print(f"Warning: Could not parse TLE: {e}")
                continue
        
        if closest_tle is None:
            raise Exception("No valid TLE found in response")
        
        closeTime = closest_epoch.strftime('%Y-%m-%d %H:%M:%S')
        print(f"Closest TLE epoch: {closeTime} UTC")
        print(f"Time difference: {min_diff/3600:.2f} hours from target epoch")
        return closest_tle, closeTime
    
    def getCurrentTLE(self, noradID):
        query_url = (
            f"{self.baseURL}/basicspacedata/query/class/gp/"
            f"NORAD_CAT_ID/{noradID}/"
            f"orderby/EPOCH desc/limit/1/format/tle"
        )
        
        response = self.session.get(query_url)
        
        if response.status_code == 200:
            tle_data = response.text.strip()
            if not tle_data:
                raise Exception(f"No TLE found for NORAD ID {noradID}")
            return tle_data.split('\n')
        else:
            raise Exception(f"Query failed: {response.status_code}")

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
