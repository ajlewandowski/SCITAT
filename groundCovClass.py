"""
This file contains the class that accounts for ground coverage, FOV projections, and its associated functions
"""

import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
from skyfield.api import wgs84, load, Distance
from skyfield.positionlib import Geocentric
from skyfield.constants import AU_KM
from datetime import timedelta, datetime, timezone
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class groundCovClass:
    def __init__(self, times, data, state, sens):
        from pathlib import Path
        if spice.ktotal('ALL') == 0:
            metakernel = Path.home() / ".spice_kernels" / "orbit_prop.tm"
            spice.furnsh(str(metakernel))

        self.moonGM = spice.bodvrd("MOON", "GM", 1)[1][0]
        self.earthGM = spice.bodvrd("EARTH", "GM", 1)[1][0]
        self.sunGM = spice.bodvrd("SUN", "GM", 1)[1][0]
        self.earthR = spice.bodvrd("EARTH", "RADII", 3)[1][0]

        self.focalLength = sens[0]
        self.baseGSD = sens[1]
        self.baseAlt = sens[2]
        self.baseFOV = sens[3]
        # Assuming square pixels
        self.acrossFOV = np.rad2deg(2*np.tan(self.baseGSD / self.baseAlt))

        self.eqAx = wgs84.radius.km
        self.poleAx = wgs84.polar_radius.km
        self.E2 = 1 - (self.poleAx/self.eqAx)**2

        self.pos = state.position.km # Earth inertial
        self.vel = state.velocity.km_per_s # Earth inertial

        self.times = times
        self.endEpoch = self.times[-1]
        self.startEpoch = self.times[0]
        self.lats = data.geometry.x
        self.lons = data.geometry.y
        self.alts = data.geometry.z
        self.lit = data.sat_sunlit
        self.swaths = data.swath_width
        self.azis = data.solar_az
        self.elevs = data.solar_alt
        self.betas = data.solar_beta

        oes = np.zeros((6, len(self.pos[0,:])))
        bearings = np.zeros((1, len(self.pos[0,:])))
        for k in range(len(self.pos[0,:])):
            oes[:,k] = self.rv2oe(self.pos[:,k], self.vel[:,k])
            bearings[:,k] = self.getBearingAngle(oes[:,k], self.lats[k])
        self.oes = oes
        self.bearings = bearings

        self.ts = load.timescale()
        # self.epochUTCdt = datetime.fromisoformat(startEpoch.replace('Z','+00:00'))
        # if self.epochUTCdt.tzinfo is None:
        #     self.epochUTCdt = self.epochUTCdt.replace(tzinfo=timezone.utc)

    # def getLatLonOE(self, oes, tSec):
    #     """ Converting the time orbital elements to latitude, longitude, and altitude """

    #     r, _ = self.oe2rv(oes)

    #     timeNow = self.epochUTCdt + timedelta(seconds=tSec)
    #     tNow = self.ts.from_datetime(timeNow)

    #     position = Geocentric(position_au=Distance(km=r).au, t=tNow)
    #     geographic = wgs84.geographic_position_of(position)

    #     lat = geographic.latitude.degrees
    #     lon = geographic.longitude.degrees
    #     alt = geographic.elevation.km

    #     return np.array([lat, lon, alt])
    
    # def getLatLonR(self, r, tSec):
    #     """ Converting r to latitude, longitude, and altitude """

    #     timeNow = self.epochUTCdt + timedelta(seconds=tSec)
    #     tNow = self.ts.from_datetime(timeNow)

    #     position = Geocentric(position_au=Distance(km=r).au, t=tNow)
    #     geographic = wgs84.geographic_position_of(position)

    #     lat = geographic.latitude.degrees
    #     lon = geographic.longitude.degrees
    #     alt = geographic.elevation.km

    #     return np.array([lat, lon, alt])
    
    # def getLatLonHist(self, t):
    #     """ Get an array time history of latitude, longitude, and altitude """

    #     nP = len(t)
    #     latlonalt = np.zeros((3, nP))

    #     for i in range(nP):
    #         lat, lon, alt = self.getLatLonOE(self.state[:,i], t[i])
    #         latlonalt[0,i] = lat
    #         latlonalt[1,i] = lon
    #         latlonalt[2,i] = alt

    #     return latlonalt
    
    def getBearingAngle(self, oe, lat):

        E = oe[5]
        for _ in range(50):
            f = E - oe[1] * np.sin(E) - oe[5]
            fp = 1 - oe[1] * np.cos(E)
            Enew = E - f / fp
            if abs(Enew - E) < 1e-10:
                E = Enew
                break
            E = Enew

        nu = 2 * np.arctan2(np.sqrt(1 + oe[1]) * np.sin(E / 2), np.sqrt(1 - oe[1]) * np.cos(E / 2))

        # Getting the direction of satellite
        u = oe[4] + nu

        if np.cos(u) > 0:
            dir = 1
        else:
            dir = 0
        
        arg = np.cos(oe[2]) / np.cos(np.deg2rad(lat))
        arg = np.clip(arg, -1.0, 1.0)
        beta0 = np.arcsin(arg)

        if dir == 1:
            beta = beta0
        else:
            beta = np.pi - beta0

        beta = beta % (2 * np.pi)

        return beta
    
    def getLineFOV(self, timeind):
        """ Get an instantaneous FOV in lat/lon coordinates """
         
        rays = self.getCamCorners(np.deg2rad(self.baseFOV), np.deg2rad(self.acrossFOV))
        tNow = self.ts.from_datetime(self.times[timeind].to_pydatetime())

        rECEF, vECEF = self.eci2ecef(self.pos[:,timeind], self.vel[:,timeind], tNow)

        rLVLH2ECEF = self.ecef2lvlh(rECEF, vECEF)

        cornersGeo = []
        for k in range(int(np.size(rays)/3)):
            rayECEF = rLVLH2ECEF @ rays[k,:]
            rayECEF /= np.linalg.norm(rayECEF)
            pt = self.getRayIntersect(rECEF, rayECEF) / AU_KM
            if pt is None:
                raise ValueError("Corner ray missed earth - FOV")
            ptPos = Geocentric(position_au=pt, t=tNow)
            intPt = wgs84.subpoint(ptPos)
            cornersGeo.append((intPt.latitude.degrees, intPt.longitude.degrees))

        corners = np.array(cornersGeo)

        print()
        print(corners)
        print()

        lonsAntiMeridCheck = corners[:,1]
        lonRange = np.max(lonsAntiMeridCheck) - np.min(lonsAntiMeridCheck)
        if lonRange > 180:
            lonsUnwrap = np.where(lonsAntiMeridCheck < 0, lonsAntiMeridCheck + 360, lonsAntiMeridCheck)
            cornersWork = corners.copy()
            cornersWork[:,1] = lonsUnwrap
        else:
            cornersWork = corners

        latSort = corners[np.argsort(corners[:,0])]
        north2 = latSort[2:]
        south2 = latSort[:2]

        tl = north2[np.argmin(north2[:,1])]
        tr = north2[np.argmax(north2[:,1])]
        bl = south2[np.argmin(south2[:,1])]
        br = south2[np.argmax(south2[:,1])]

        if lonRange > 180:
            tl = self.unwrapLon(tl)
            tr = self.unwrapLon(tr)
            bl = self.unwrapLon(bl)
            br = self.unwrapLon(br)
                
        corners = np.array([tl, tr, br, bl])

        crossLatSpan = abs(corners[0,0] - corners[1,0])
        crossLonSpan = abs(corners[0,1] - corners[1,1])
        alongLatSpan = abs(corners[0,0] - corners[3,0])
        alongLonSpan = abs(corners[0,1] - corners[3,1])

        swathAlong = self.angular2km(corners[0], corners[1])
        swathCross = self.angular2km(corners[0], corners[3])
        
        return corners, np.array([alongLatSpan, alongLonSpan]), np.array([crossLatSpan, crossLonSpan]), swathAlong, swathCross
    
    # def getlatlonFOVnadirHist(self, latlonalt):
    #     """ Get an FOV in lat/lon coordinates """
    #     lat, lon, _ = latlonalt

    #     tL = np.zeros((len(lat),2))
    #     tR = np.zeros((len(lat),2))
    #     bL = np.zeros((len(lat),2))
    #     bR = np.zeros((len(lat),2))
    #     mid = np.zeros((len(lat),2))
    #     betas = np.zeros((len(lat),1))

    #     for k in range(len(lat)):

    #         t = self.time[k]
    #         timeNow = self.epochUTCdt + timedelta(seconds=t)
    #         tNow = self.ts.from_datetime(timeNow)

    #         crossFOV = (self.crossPix * self.pixPitch / self.focalLength)
    #         alongFOV = (self.alongPix * self.pixPitch / self.focalLength)

    #         rays = self.getCamCorners(crossFOV, alongFOV)

    #         oes = self.state[:,k]
    #         r, v = self.oe2rv(oes)

    #         beta = self.getBearingAngle(oes, lat[k])

    #         rECEF, vECEF = self.eci2ecef(r, v, tNow)

    #         rLVLH2ECEF = self.ecef2lvlh(rECEF, vECEF)

    #         cornersGeo = []
    #         for j in range(int(np.size(rays)/3)):
    #             rayECEF = rLVLH2ECEF @ rays[j,:]
    #             rayECEF /= np.linalg.norm(rayECEF)
    #             pt = self.getRayIntersect(rECEF, rayECEF)
    #             if pt is None:
    #                 raise ValueError("Corner ray missed earth - FOV")
    #             latN, lonN, _ = self.ecef2geodetic(pt)
    #             cornersGeo.append((latN, lonN))

    #         corners = np.array(cornersGeo)
    #         lonsAntiMeridCheck = corners[:,1]
    #         lonRange = np.max(lonsAntiMeridCheck) - np.min(lonsAntiMeridCheck)
    #         if lonRange > 180:
    #             lonsUnwrap = np.where(lonsAntiMeridCheck < 0, lonsAntiMeridCheck + 360, lonsAntiMeridCheck)
    #             cornersWork = corners.copy()
    #             cornersWork[:,1] = lonsUnwrap
    #         else:
    #             cornersWork = corners

    #         latSort = corners[np.argsort(corners[:,0])]
    #         north2 = latSort[2:]
    #         south2 = latSort[:2]

    #         tl = north2[np.argmin(north2[:,1])]
    #         tr = north2[np.argmax(north2[:,1])]
    #         bl = south2[np.argmin(south2[:,1])]
    #         br = south2[np.argmax(south2[:,1])]

    #         if lonRange > 180:
    #             tl = self.unwrapLon(tl)
    #             tr = self.unwrapLon(tr)
    #             bl = self.unwrapLon(bl)
    #             br = self.unwrapLon(br)
                    
    #         corners = np.array([tl, tr, br, bl])
            
    #         if any(x is None for x in [tl, tr, br, bl]):
    #             raise ValueError(f"Corner classification failed at timestep {k} — corners: {corners}")

    #         tL[k,:] = np.array(tl)
    #         tR[k,:] = np.array(tr)
    #         bL[k,:] = np.array(bl)
    #         bR[k,:] = np.array(br)
    #         mid[k,:] = np.array([lat[k], lon[k]])
    #         betas[k] = beta

    #     return tL, tR, bL, bR, mid, betas
    
    # def imageLatLonNadir(self, t, targetLatLon, tL, tR, bL, bR, mid, betas, alt):
    #     tLat, tLon = targetLatLon

    #     tImage = []
    #     tLIlat = []
    #     tLIlon = []
    #     tRIlat = []
    #     tRIlon = []
    #     bLIlat = []
    #     bLIlon = []
    #     bRIlat = []
    #     bRIlon = []
    #     midIlat = []
    #     midIlon = []
    #     beta = []
    #     altI = []
    #     a = []
    #     e = []
    #     i = []
    #     RAAN = []
    #     aop = []
    #     M = []

    #     print(f"Target latitude: {tLat:.4f}° and target longitude: {tLon:.4f}°")
    #     print()

    #     for k in range(len(tL)):
    #         if (min(tL[k,0],bL[k,0]) <= tLat <= max(tL[k,0],bL[k,0])) and (min(tL[k,1],tR[k,1]) <= tLon <= max(tL[k,1],tR[k,1])):
    #             tImage.append(t[k])

    #             tLIlat.append(tL[k,0])
    #             tLIlon.append(tL[k,1])

    #             tRIlat.append(tR[k,0])
    #             tRIlon.append(tR[k,1])

    #             bLIlat.append(bL[k,0])
    #             bLIlon.append(bL[k,1])

    #             bRIlat.append(bR[k,0])
    #             bRIlon.append(bR[k,1])

    #             midIlat.append(mid[k,0])
    #             midIlon.append(mid[k,1])

    #             beta.append(betas[k])

    #             altI.append(alt[k])

    #             a.append(self.state[0,k])
    #             e.append(self.state[1,k])
    #             i.append(self.state[2,k])
    #             RAAN.append(self.state[3,k])
    #             aop.append(self.state[4,k])
    #             M.append(self.state[5,k])


    #     if not tImage:
    #         print()
    #         print("Imaging of target latitude and longitude not possible nadir!")
    #         return 0, 0, 0, 0
    #     else:
    #         numImages = len(tImage)
    #         print()
    #         print("Target latitude and longitude is within view!")
        
    #     timeImage = np.array(tImage)
    #     instFOVs = np.column_stack([np.array(tLIlat),np.array(tLIlon), 
    #                                 np.array(tRIlat),np.array(tRIlon),
    #                                 np.array(bRIlat),np.array(bRIlon), 
    #                                 np.array(bLIlat),np.array(bLIlon),
    #                                 np.array(midIlat),np.array(midIlon),
    #                                 np.array(altI), np.array(beta)])
        
    #     oes = np.column_stack([np.array(a), np.array(e), np.array(i), np.array(RAAN), np.array(aop), np.array(M)])
        
    #     tImFirst = datetime.fromisoformat(self.startEpoch) + timedelta(seconds=timeImage[0])
    #     tImLast = datetime.fromisoformat(self.startEpoch) + timedelta(seconds=timeImage[-1])

    #     print(f"{numImages:.0f} images can be taken of this location given a nadir view")
    #     print(f"The timespan of the images is from: {tImFirst} to {tImLast}")
    #     print()

    #     return timeImage, instFOVs, numImages, oes

    def oe2rv(self, oe):
        """ Converting classical orbital elements to position and velocity """
        
        a, e, i, RAAN, aop, M = oe

        # True anomaly
        nu, _ = self.mean2nu(M, e)

        # Perifocal frame
        p = a * (1 - e**2)
        rPQW = (p / (1 + e * np.cos(nu))) * np.array([np.cos(nu), np.sin(nu), 0])
        vPQW = np.sqrt(self.earthGM / p) * np.array([-np.sin(nu), e + np.cos(nu), 0])

        # Rotation to ECI
        R = self.pqw2eci(RAAN, aop, i)
        r = R @ rPQW
        v = R @ vPQW

        return r, v
    
    def rv2oe(self, r, v):
        r_norm = np.linalg.norm(r)
        v_norm = np.linalg.norm(v)
        h = np.cross(r, v)
        h_norm = np.linalg.norm(h)
        mu = self.earthGM
        
        # Semi-major axis and eccentricity
        energy = v_norm**2 / 2 - mu / r_norm
        a = -mu / (2*energy)
        e_vec = ((v_norm**2 - mu/r_norm)*r - np.dot(r,v)*v)/mu
        e = np.linalg.norm(e_vec)
        
        # Inclination
        i = np.arccos(h[2]/h_norm)
        
        # RAAN
        n = np.cross([0,0,1], h)
        n_norm = np.linalg.norm(n)
        RAAN = np.arccos(n[0]/n_norm)
        if n[1] < 0: RAAN = 2*np.pi - RAAN
        
        # Argument of periapsis
        aop = np.arccos(np.dot(n,e_vec)/(n_norm*e))
        if e_vec[2] < 0: aop = 2*np.pi - aop
        
        # True anomaly
        nu = np.arccos(np.dot(e_vec,r)/(e*r_norm))
        if np.dot(r,v) < 0: nu = 2*np.pi - nu
        
        # Mean anomaly
        E = 2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(nu/2))
        M = E - e*np.sin(E)

        M = M % (2*np.pi)
        
        return [a, e, i, RAAN, aop, M]

    def eci2ecef(self, rECI, vECI, time):
        gmst = np.deg2rad(time.gmst * 15)

        rot = np.array([[np.cos(gmst), np.sin(gmst), 0],
                        [-np.sin(gmst), np.cos(gmst), 0],
                        [0, 0, 1]])
        
        rECEF = np.dot(rot, rECI)
        vECEF = np.dot(rot, vECI)

        return rECEF, vECEF
    
    def unwrapLon(self, c):
        return np.array([c[0], c[1] - 360 if c[1] > 180 else c[1]])
    
    def ecef2geodetic(self, rECEF):
        x, y, z = rECEF
        a = self.eqAx
        e2 = self.E2

        lon = np.degrees(np.arctan2(y,x))
        p = np.sqrt(x**2 + y**2)
        lat = np.arctan2(z, p*(1 - e2))

        for _ in range(10):
            sinLat = np.sin(lat)
            N = a / np.sqrt(1 - e2*sinLat**2)
            lat = np.arctan2(z + e2*N*sinLat, p)

        cosLat = np.cos(lat)
        if abs(cosLat) > 1e-10:
            alt = p / cosLat - N
        else:
            alt = abs(z) / abs(np.sin(lat)) - N * (1 - e2)

        return np.degrees(lat), lon, alt
    
    def pqw2eci(self, RAAN, aop, i):
        """ Rotation matrix from perifocal frame to ECI frame """

        r3RAAN = np.array([[np.cos(RAAN), -np.sin(RAAN), 0], [np.sin(RAAN), np.cos(RAAN), 0], [0, 0, 1]])
        r1I = np.array([[1, 0, 0], [0, np.cos(i), -np.sin(i)], [0, np.sin(i), np.cos(i)]])
        r3aop = np.array([[np.cos(aop), -np.sin(aop), 0], [np.sin(aop), np.cos(aop), 0], [0, 0, 1]])

        return r3RAAN @ r1I @ r3aop

    def mean2nu(self, M, e, tol=1e-10):
        """ Converting mean anomaly to true anomaly """

        E = M
        for _ in range(50):
            f = E - e * np.sin(E) - M
            fp = 1 - e * np.cos(E)
            Enew = E - f / fp
            if abs(Enew - E) < tol:
                E = Enew
                break
            E = Enew

        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))

        return nu, E
    
    def ecef2lvlh(self, rECEF, vECEF):
        r = rECEF / np.linalg.norm(rECEF)
        v = vECEF / np.linalg.norm(vECEF)
        
        ez = -r
        ex = v - np.dot(v, ez) * ez
        ex /= np.linalg.norm(ex)
        ey = np.cross(ez, ex)
        ey /= np.linalg.norm(ey)

        return np.array([ex, -ey, ez]).T
    
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
    
    def plotGroundTrack(self, latlonalt):
        """ Plot the ground track from the latlonalt coordinates """

        # Get lat/lon
        lats = latlonalt[0, :]
        lons = latlonalt[1, :]
        
        # Create figure with map projection
        fig = plt.figure(figsize=(14, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        
        # Plot ground track
        ax.plot(lons, lats, 'r-', linewidth=2, transform=ccrs.Geodetic(), label='Ground track')
        ax.plot(lons[0], lats[0], 'go', markersize=15, transform=ccrs.Geodetic(), label='Start')
        ax.plot(lons[-1], lats[-1], 'r*', markersize=15, transform=ccrs.Geodetic(), label='End')
        
        # Set extent to show full track
        margin = 10
        ax.set_extent([lons.min()-margin, lons.max()+margin, 
                    lats.min()-margin, lats.max()+margin], 
                    crs=ccrs.PlateCarree())
        
        plt.title(f'Ground Track {self.startEpoch} to {self.endEpoch}', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        return fig
    
    def plotInstFOV(self, coordsFOV, satlatlon, time):
        """ Plot the instantaneous FOV - optimized version """

        # Importing the coordinates 
        lats = np.append(coordsFOV[:,0], coordsFOV[0,0])
        lons = np.append(coordsFOV[:,1], coordsFOV[0,1])
        
        # Get satellite position (center of FOV)
        sat_lat = satlatlon[0]
        sat_lon = satlatlon[1]
        
        # Calculate FOV size
        fov_lat_size = lats.max() - lats.min()
        fov_lon_size = lons.max() - lons.min()
        
        # Margin: FOV size or at least 2 degrees
        margin_lat = max(fov_lat_size, 2)
        margin_lon = max(fov_lon_size, 2)
        
        # Calculate extent
        lat_min = max(sat_lat - margin_lat, -90)
        lat_max = min(sat_lat + margin_lat, 90)
        lon_min = max(sat_lon - margin_lon, -180)
        lon_max = min(sat_lon + margin_lon, 180)
        
        # Handle wraparound
        if lon_max - lon_min > 180:
            margin_lon = min(margin_lon, 20)
            lon_min = sat_lon - margin_lon
            lon_max = sat_lon + margin_lon
        
        # Create figure
        fig = plt.figure(figsize=(14, 9))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Set extent FIRST (before adding features) - this is key for performance
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Add map features with lower resolution for speed
        ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray', edgecolor='black', linewidth=0.3)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5, edgecolor='black')
        
        # Simplified gridlines (fewer lines)
        gl = ax.gridlines(draw_labels=True, linewidth=0.8, color='gray', alpha=0.4, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        # Fill FOV polygon first (draws faster as base layer)
        from matplotlib.patches import Polygon
        fov_polygon = Polygon(list(zip(lons[:-1], lats[:-1])), 
                            facecolor='red', alpha=0.25, edgecolor='red', linewidth=3,
                            transform=ccrs.PlateCarree(), zorder=4)
        ax.add_patch(fov_polygon)
    
        # Plot satellite nadir
        ax.plot(sat_lon, sat_lat, 'bo', transform=ccrs.PlateCarree(), 
                markersize=10, markeredgecolor='darkblue', markeredgewidth=1.5, label='Nadir', zorder=10)
        
        # Title
        plt.title(f'Instantaneous FOV', fontsize=13, fontweight='bold', pad=12)
        
        plt.legend(loc='upper right', fontsize=10, framealpha=0.95)
        plt.tight_layout()

        return fig
    
    def plotMultipleFOVsClose(self, instFOVs, times, target_lat, target_lon):

        coordsMFOV = []
        latlonlist = []
        timelist = []

        # Compiling data to pass into plotting mulitple FOVs function
        for i in range(len(times)):
            coordsMFOV.append(np.array([[instFOVs[i,0], instFOVs[i,1]], 
                                        [instFOVs[i,2], instFOVs[i,3]], 
                                        [instFOVs[i,4], instFOVs[i,5]], 
                                        [instFOVs[i,6], instFOVs[i,7]]]))
            latlonlist.append([instFOVs[i,8],instFOVs[i,9]])
            timelist.append(times[i])

        # Calculate overall extent from all FOVs and target
        all_lats = [target_lat]
        all_lons = [target_lon]
        
        for coordsFOV in coordsMFOV:
            all_lats.extend(coordsFOV[:,0])
            all_lons.extend(coordsFOV[:,1])
        
        all_lats = np.array(all_lats)
        all_lons = np.array(all_lons)
        
        # Calculate extent with margin
        lat_range = all_lats.max() - all_lats.min()
        lon_range = all_lons.max() - all_lons.min()
        
        margin_lat = max(lat_range * 0.6, 6)
        margin_lon = max(lon_range * 0.6, 6)
        
        lat_min = max(all_lats.min() - margin_lat, -90)
        lat_max = min(all_lats.max() + margin_lat, 90)
        lon_min = max(all_lons.min() - margin_lon, -180)
        lon_max = min(all_lons.max() + margin_lon, 180)
        
        # Handle wraparound
        if lon_max - lon_min > 180:
            lon_center = np.mean(all_lons)
            margin_lon = min(margin_lon, 20)
            lon_min = lon_center - margin_lon
            lon_max = lon_center + margin_lon
        
        # Create figure
        fig = plt.figure(figsize=(14, 9))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Set extent
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray', edgecolor='black', linewidth=0.3)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5, edgecolor='black')
        
        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.8, color='gray', alpha=0.4, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        
        # Plot each FOV
        from matplotlib.patches import Polygon
        
        for i, (coordsFOV, satlatlon, time) in enumerate(zip(coordsMFOV, latlonlist, times)):
            # Prepare coordinates
            lats = np.append(coordsFOV[:,0], coordsFOV[0,0])
            lons = np.append(coordsFOV[:,1], coordsFOV[0,1])
            
            # Plot FOV polygon
            fov_polygon = Polygon(list(zip(lons[:-1], lats[:-1])), 
                                facecolor='red', alpha=0.25, edgecolor='red', linewidth=2,
                                transform=ccrs.PlateCarree(), zorder=4)
            ax.add_patch(fov_polygon)
        
        # Plot target location (green dot)
        ax.plot(target_lon, target_lat, 'go', transform=ccrs.PlateCarree(), 
                markersize=12, markeredgecolor='darkgreen', markeredgewidth=2, label='Target', zorder=10)
        
        # Title
        plt.title(f'Multiple FOVs - {len(coordsMFOV)} positions\n' + 
                f'Target: ({target_lat:.2f}°, {target_lon:.2f}°) | Time range: {times[0]:.4f} to {times[-1]:.4f} sec', 
                fontsize=13, fontweight='bold', pad=12)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                alpha=0.25, markersize=12, markeredgecolor='red', markeredgewidth=2, label='FOV'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                markersize=12, markeredgecolor='darkgreen', markeredgewidth=2, label='Target')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)
        
        plt.tight_layout()
        
        return fig
    
    def plotMultipleFOVsFar(self, instFOVs, times, target_lat, target_lon):

        coordsMFOV = []
        latlonlist = []
        timelist = []

        # Compiling data to pass into plotting mulitple FOVs function
        for i in range(len(times)):
            coordsMFOV.append(np.array([[instFOVs[i,0], instFOVs[i,1]], 
                                        [instFOVs[i,2], instFOVs[i,3]], 
                                        [instFOVs[i,4], instFOVs[i,5]], 
                                        [instFOVs[i,6], instFOVs[i,7]]]))
            latlonlist.append([instFOVs[i,8],instFOVs[i,9]])
            timelist.append(times[i])
                
        # Calculate overall extent from all FOVs and target
        all_lats = [target_lat]
        all_lons = [target_lon]
        
        for coordsFOV in coordsMFOV:
            all_lats.extend(coordsFOV[:,0])
            all_lons.extend(coordsFOV[:,1])
        
        all_lats = np.array(all_lats)
        all_lons = np.array(all_lons)
        
        # Calculate extent with margin
        lat_range = all_lats.max() - all_lats.min()
        lon_range = all_lons.max() - all_lons.min()
        
        margin_lat = max(lat_range * 4, 40)
        margin_lon = max(lon_range * 4, 40)
        
        lat_min = max(all_lats.min() - margin_lat, -90)
        lat_max = min(all_lats.max() + margin_lat, 90)
        lon_min = max(all_lons.min() - margin_lon, -180)
        lon_max = min(all_lons.max() + margin_lon, 180)
        
        # Handle wraparound
        if lon_max - lon_min > 180:
            lon_center = np.mean(all_lons)
            margin_lon = min(margin_lon, 20)
            lon_min = lon_center - margin_lon
            lon_max = lon_center + margin_lon
        
        # Create figure
        fig = plt.figure(figsize=(14, 9))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Set extent
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray', edgecolor='black', linewidth=0.3)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5, edgecolor='black')
        
        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.8, color='gray', alpha=0.4, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        
        # Plot each FOV
        from matplotlib.patches import Polygon
        
        for i, (coordsFOV, satlatlon, time) in enumerate(zip(coordsMFOV, latlonlist, times)):
            # Prepare coordinates
            lats = np.append(coordsFOV[:,0], coordsFOV[0,0])
            lons = np.append(coordsFOV[:,1], coordsFOV[0,1])
            
            # Plot FOV polygon
            fov_polygon = Polygon(list(zip(lons[:-1], lats[:-1])), 
                                facecolor='red', alpha=0.25, edgecolor='red', linewidth=2,
                                transform=ccrs.PlateCarree(), zorder=4)
            ax.add_patch(fov_polygon)
        
        # Title
        plt.title(f'Multiple FOVs - {len(coordsMFOV)} positions\n' + 
                f'Target: ({target_lat:.2f}°, {target_lon:.2f}°) | Time range: {times[0]:.4f} to {times[-1]:.4f} sec', 
                fontsize=13, fontweight='bold', pad=12)
        
        plt.tight_layout()
        
        return fig
    
    def printFOVCoords(self, latlonNadirInst, latlonalt, t, timeind):

        print()
        print(f'Instantaneous FOV at t = {(t[timeind]/60):.3f} min')
        print(f"  Upper Left (lat, lon) = ({latlonNadirInst[0,0]:.3f}°, {latlonNadirInst[0,1]:.3f}°)")
        print(f"  Upper Right (lat, lon) = ({latlonNadirInst[1,0]:.3f}°, {latlonNadirInst[1,1]:.3f}°)")
        print(f"  Bottom Left (lat, lon) = ({latlonNadirInst[2,0]:.3f}°, {latlonNadirInst[2,1]:.3f}°)")
        print(f"  Bottom Right (lat, lon) = ({latlonNadirInst[3,0]:.3f}°, {latlonNadirInst[3,1]:.3f}°)")
        print(f"  Satellite Pos. (lat, lon) = ({latlonalt[0,timeind]:.3f}°, {latlonalt[1,timeind]:.3f}°)")
        print(f"  Satellite Altitude = {latlonalt[2,timeind]:.3f} km")
        print()