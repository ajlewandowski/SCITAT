from datetime import datetime, timedelta
import os

class getTLEData:
    def __init__(self, fileName, tlePath):
        self.tlePath = tlePath
        self.fileName = fileName
        if not os.path.exists(tlePath):
            raise FileNotFoundError(f"The file {fileName} was not found or does not exist.")

    def readTLEFile(self):
        """ Reads the TLE file and stores the data """
        with open(self.tlePath, 'r') as file:
            while True:
                line1 = file.readline().strip()
                line2 = file.readline().strip()
                yield (line1, line2)

    def parseTLEEpoch(self, line1):
        """ Parses the epoch from the first TLE line. """
        year = int(line1[18:20]) + 2000
        day = float(line1[20:32])
        return datetime(year, 1, 1) + timedelta(days=day - 1)
    
    def parseTargetTime(self, target):
        return datetime.strptime(target, "%Y-%m-%dT%H:%M:%S")
    
    def findClosestTLE(self, target, noradID):
        """ Finds the closest TLE to the desired target epoch """
        targTime = self.parseTargetTime(target)
        closestTLE = None
        smallDiff = float('inf')
        for line1, line2 in self.readTLEFile():
            if f' {noradID}' in line1 and line2:
                epochTime = self.parseTLEEpoch(line1)
                timeDiff = abs((epochTime - targTime).total_seconds())
                if timeDiff < smallDiff:
                    smallDiff = timeDiff
                    closestTLE = (line1, line2, epochTime)
        return closestTLE
    
    def getTLE(self, target, noradID):
        """ Returns the TLE data for the given NORAD ID and target time """
        tle = self.findClosestTLE(target, noradID)
        if tle is None:
            raise ValueError(f"No TLE found for NORAD ID {noradID} at the specified time.")
        return tle
    
    def printTLE(self, target, noradID):
        """ Prints the TLE data for the given NORAD ID and target time. """
        line1, line2, epochTime = self.getTLE(target, noradID)
        print()
        print("TLE Obtained from Space-Tracker")
        print(line1)  # Line 1
        print(line2)  # Line 2  
        return line1, line2, epochTime.strftime("%Y-%m-%dT%H:%M:%S")

    def outputTLE(self, target, noradID):
        """ Outputs the TLE data for the given NORAD ID and target time as a string. """
        line1, line2, epochTime = self.getTLE(target, noradID)
        return f"{line1}\n{line2}\n\n"
    
    def outputCloseTLETime(self, target, noradID):
        _, _, epochTime = self.getTLE(target, noradID)
        return epochTime.strftime("%Y-%m-%dT%H:%M:%S")