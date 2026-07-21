# getTLEData.py
import os
from datetime import datetime, timedelta

class getTLEData:
    def __init__(self, tle_file_path):
        self.tle_file_path = tle_file_path
        if not os.path.exists(self.tle_file_path):
            raise FileNotFoundError(f"The file {tle_file_path} does not exist.")
        
        # Preload all TLE data into memory
        self.tles = list(self._read_tle_file_line_by_line())
        self.tles.sort(key=lambda tle: self._parse_tle_epoch(tle[0]))  # Sort by epoch time
    
    def _read_tle_file_line_by_line(self):
        """Yields TLE data in chunks of two lines (line1, line2)."""
        with open(self.tle_file_path, 'r') as file:
            while True:
                line1 = file.readline().strip()
                if not line1:
                    break
                line2 = file.readline().strip()
                yield (line1, line2)
    
    def _parse_tle_epoch(self, line1):
        """Parses the epoch from the first TLE line."""
        # The epoch is in the format YYYYDDD.DDDDDD
        year = int(line1[18:20]) + 2000  # Assuming 20xx for years 00-99
        day_of_year = float(line1[20:32])
        return datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    
    def _parse_target_time(self, target_time_str):
        """Parses the target time string to a datetime object."""
        return datetime.strptime(target_time_str, "%Y-%m-%dT%H:%M:%S")
    
    def binary_search_closest_tle_before(self, target_time):
        """Finds the TLE with the closest epoch to the target time that is before or equal to the target time using binary search."""
        left, right = 0, len(self.tles) - 1
        closest_tle = None
        
        while left <= right:
            mid = (left + right) // 2
            line1, line2 = self.tles[mid]
            epoch_time = self._parse_tle_epoch(line1)
            
            if epoch_time <= target_time:
                closest_tle = (line1, line2, epoch_time)
                left = mid + 1  # Continue searching to the right for a closer match
            else:
                right = mid - 1
        
        return closest_tle
    
    def get_tle(self, target_time_str):
        """Returns the TLE data for the given target time that is before or equal to the target time."""
        target_time = self._parse_target_time(target_time_str)
        tle = self.binary_search_closest_tle_before(target_time)
        if tle is None:
            raise ValueError(f"No TLE found at or before the specified time.")
        return tle
    
    def print_tle(self, target_time_str):
        """Prints the TLE data for the given target time that is before or equal to the target time."""
        line1, line2, epoch_time = self.get_tle(target_time_str)
        print(line1)
        print(line2)
        closeTime = epoch_time.strftime("%Y-%m-%dT%H:%M:%S")
        return line1, line2, closeTime
    
    def output_tle(self, target_time_str):
        """Outputs the TLE data for the given target time that is before or equal to the target time as a string."""
        line1, line2, _ = self.get_tle(target_time_str)
        return f"{line1}\n{line2}"
    
    def get_closest_epoch(self, target_time_str):
        """Returns the closest TLE epoch that is before or equal to the target time as a string in the format '%Y-%m-%dT%H:%M:%S'."""
        _, _, epoch_time = self.get_tle(target_time_str)
        return epoch_time.strftime("%Y-%m-%dT%H:%M:%S")
