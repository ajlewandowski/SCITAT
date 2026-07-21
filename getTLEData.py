import os
from datetime import datetime, timedelta

class getTLEData:
    def __init__(self, tle_file_path):
        self.tle_file_path = tle_file_path
        if not os.path.exists(self.tle_file_path):
            raise FileNotFoundError(f"The file {tle_file_path} does not exist.")
        
        # Preload all TLE data into memory
        self.tles = list(self.readFile())
        self.tles.sort(key=lambda tle: self.parseTLE(tle[0]))  # Sort by epoch time
    
    def readFile(self):
        """Yields TLE data in chunks of two lines (line1, line2)."""
        with open(self.tle_file_path, 'r') as file:
            while True:
                line1 = file.readline().strip()
                if not line1:
                    break
                line2 = file.readline().strip()
                yield (line1, line2)
    
    def parseTLE(self, line1):
        """Parses the epoch from the first TLE line."""
        # The epoch is in the format YYYYDDD.DDDDDD
        year = int(line1[18:20]) + 2000  # Assuming 20xx for years 00-99
        day_of_year = float(line1[20:32])
        return datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    
    def parseTarget(self, target_time_str):
        """Parses the target time string to a datetime object."""
        return datetime.strptime(target_time_str, "%Y-%m-%dT%H:%M:%S")
    
    def closestTLE(self, target_time):
        """Finds the TLE with the closest epoch to the target time that is before or equal to the target time using binary search."""
        left, right = 0, len(self.tles) - 1
        closest_tle = None
        
        while left <= right:
            mid = (left + right) // 2
            line1, line2 = self.tles[mid]
            epoch_time = self.parseTLE(line1)
            
            if epoch_time <= target_time:
                closest_tle = (line1, line2, epoch_time)
                left = mid + 1  # Continue searching to the right for a closer match
            else:
                right = mid - 1
        
        return closest_tle
    
    def getTLE(self, target_time_str):
        """Returns the TLE data for the given target time that is before or equal to the target time."""
        target_time = self.parseTarget(target_time_str)
        tle = self.closestTLE(target_time)
        if tle is None:
            raise ValueError(f"No TLE found at or before the specified time.")
        return tle
    
    def printTLE(self, target_time_str):
        """Prints the TLE data for the given target time that is before or equal to the target time."""
        line1, line2, epoch_time = self.getTLE(target_time_str)
        print(line1)
        print(line2)
        closeTime = epoch_time.strftime("%Y-%m-%dT%H:%M:%S")
        return line1, line2, closeTime
    
    def outputTLE(self, target_time_str):
        """Outputs the TLE data for the given target time that is before or equal to the target time as a string."""
        line1, line2, _ = self.getTLE(target_time_str)
        return f"{line1}\n{line2}"
    
    def getClosestEpoch(self, target_time_str):
        """Returns the closest TLE epoch that is before or equal to the target time as a string in the format '%Y-%m-%dT%H:%M:%S'."""
        _, _, epoch_time = self.getTLE(target_time_str)
        return epoch_time.strftime("%Y-%m-%dT%H:%M:%S")
