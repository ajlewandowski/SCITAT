"""
This file contains an orbit propagator class using the Lagrange Planetary Equations

The orbit propagation is adding in the J2 perturbation and the Sun and Moon's N body perturbation

INPUT: Classical orbital elements with mean anomaly as fast variable
OUTPUT: Propagated orbit with 3D visualization
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path
import spiceypy as spice
from skyfield.api import load
from datetime import datetime, timezone

class orbitPropClass:
    """
    orbitPropClass uses the Lagrange Planetary Equations to propagate including J2, lunar, and solar perturbations 
    """

    def __init__(self, epochUTC, tSpan, sens, incJ2=True, incMoon=True, incSun=True):
        """
        Initializing propagator and parameters
        
        eopchUTC: Epoch time in UTC format YYYY-MM-DDTHH:MM:SS
        incJ2: bool including J2 or not (T/F)
        incMoon: bool including Moon pert or not (T/F)
        incSun: bool including Sun pert or not (T/F)
        """
        self.loadSpice()

        self.moonGM = spice.bodvrd("MOON", "GM", 1)[1][0]
        self.earthGM = spice.bodvrd("EARTH", "GM", 1)[1][0]
        self.sunGM = spice.bodvrd("SUN", "GM", 1)[1][0]
        self.J2coeff = 1.08263e-3
        
        self.earthR = spice.bodvrd("EARTH", "RADII", 3)[1][0]

        self.epochUTC = epochUTC
        self.incJ2 = incJ2
        self.incMoon = incMoon
        self.incSun = incSun

        self.epochET = spice.str2et(epochUTC)
        self.epochETFinal = self.epochET + tSpan
        self.tSpan = tSpan

        self.ts = load.timescale()
        self.epochUTCdt = datetime.fromisoformat(epochUTC.replace('Z','+00:00'))
        if self.epochUTCdt.tzinfo is None:
            self.epochUTCdt = self.epochUTCdt.replace(tzinfo=timezone.utc)

        self.focalLength = sens[0]
        self.sensorWidth = sens[1]*sens[3]
        self.sensorLength = sens[2]*sens[3]

        print()
        print(f"Orbit Propagator initialized")
        print(f"  Epoch: {epochUTC}")
        print(f"  J2: {incJ2}, Moon: {incMoon}, Sun: {incSun}")
        print(f"  μEarth = {self.earthGM:.4f} km³/s²")
        print()

    def loadSpice(self):
    # Only load if not already loaded
        if spice.ktotal('ALL') == 0:
            metakernel = Path.home() / ".spice_kernels" / "orbit_prop.tm"
            spice.furnsh(str(metakernel))

    def __del__(self):
        pass

    def getBodyPosition(self, bodyName, t):
        """ Getting position of a body relative to Earth at time t (since epoch) with bodyName "MOON" or "SUN" """
        if spice.ktotal("ALL") == 0:  # kernel pool is empty
            for k in self.kernelPaths:
                spice.furnsh(k)
        
        et = self.epochET + t
        state, _ = spice.spkezr(bodyName, et, "J2000","NONE","EARTH")
        
        return np.array(state[:3])

    def getJ2(self, oe):
        """ Calculating the J2 pertubation using LPEs """
        a, e, i, _, _, _ = oe

        # Avoiding singularities
        e = max(e, 1e-10)
        sinI = max(np.sin(i),1e-10)
        cosI = max(np.cos(i),1e-10)

        # Mean motion
        n = np.sqrt(self.earthGM / a**3)
        ae = self.earthR

        # J2 coefficient and LPE calculation - Secularly precessing ellipse model
        dadt = 0.0
        dedt = 0.0
        didt = 0.0
        dRAANdt = -1.5 * n * self.J2coeff * (1/(2*(1-e**2)**2)) * (2.5*sinI**2 - 2) * (ae / a)**2
        daopdt = -1.5 * n * self.J2coeff * (1/(2*(1-e**2)**2)) * cosI * (ae / a)**2
        dMdt = n * (1 - ((3*self.J2coeff)/(2*(1-e**2)**1.5))*(1.5*sinI**2 - 1)*(ae/a)**2)

        return np.array([dadt, dedt, didt, dRAANdt, daopdt, dMdt])

    def getThirdBodyPert(self, oe, rBody, muBody):
        """ Calculating the third body perturbation using Gauss variational equations """
        
        r, v = self.oe2rv(oe)

        rSB = rBody - r
        rBodyNorm = np.linalg.norm(rBody)
        rSBNorm = np.linalg.norm(rSB)

        # Third-body pert acceleration
        aPert = muBody * (rSB / rSBNorm**3 - rBody / rBodyNorm**3)

        return aPert
        
    def gaussEqs(self, oe, aPert):
        """ Gauss's variational equations to get orbital element rates """
        
        a, e, i, _, omega, M = oe
        r, v = self.oe2rv(oe)
        
        # Orbital parameters
        rNorm = np.linalg.norm(r)
        n = np.sqrt(self.earthGM / a**3)
        p = a * (1 - e**2)
        h = np.cross(r, v)
        hNorm = np.linalg.norm(h)
        
        # True anomaly
        nu, _ = self.mean2nu(M, e)
        
        # RSW frame (radial, along-track, cross-track)
        r_hat = r / rNorm
        h_hat = h / hNorm
        theta_hat = np.cross(h_hat, r_hat)  # Transverse direction
        
        # Acceleration components in RTN frame
        aR = np.dot(aPert, r_hat)      # Radial
        aT = np.dot(aPert, theta_hat)  # Transverse
        aN = np.dot(aPert, h_hat)      # Cross-track (normal)

        dadt = (2 / (n*np.sqrt(1-e**2))) * (e*np.sin(nu)*aR + (p/rNorm)*aT)
        dedt = ((np.sqrt(1-e**2))/(n*a))*(np.sin(nu)*aR + aT*(np.cos(nu) + ((e+np.cos(nu))/(1+e*np.cos(nu)))))
        didt = (rNorm / (n*np.sqrt(1-e**2)*a**2)) * np.cos(omega + nu) * aN
        dwdt = ((np.sqrt(1-e**2))/(n*a*e))*(-np.cos(nu)*aR + np.sin(nu)*aT*(1 + (rNorm/p))) - ((rNorm*np.cos(i)*np.sin(omega+nu))/(hNorm*np.sin(i)))*aN
        dOdt = aN * ((rNorm*np.sin(omega+nu))/(n*np.sin(i)*np.sqrt(1-e**2)*a**2))
        dMdt = (1/(n*e*a**2))*((p*np.cos(nu) - 2*e*rNorm)*aR - (p + rNorm)*np.sin(nu)*aT) + n
        
        return np.array([dadt, dedt, didt, dOdt, dwdt, dMdt])
    
    def getODE(self, t, oe):
        """ This function combines all the rates of change or obital elements for integration """
        
        doe = np.zeros(6)
        aPert = np.zeros(3)
        
        # J2 pert
        if self.incJ2:
            doe += self.getJ2(oe)

        # Lunar pert
        if self.incMoon:
            rMoon = self.getBodyPosition("MOON", t)
            tempPert = self.getThirdBodyPert(oe, rMoon, self.moonGM)
            aPert = aPert + tempPert

        
        # Solar pert
        if self.incSun:
            rSun = self.getBodyPosition("SUN", t)
            tempPert = self.getThirdBodyPert(oe, rSun, self.sunGM)
            aPert = aPert + tempPert

        doe += self.gaussEqs(oe, aPert)

        return doe
    
    def propagate(self, oe0, dtSec):
        """ Propagating orbit """

        print()
        print(f"Propagating orbit")
        print(f"    Duration: {self.tSpan/60} min")
        print(f"    Time step: {dtSec} seconds")
        print()

        # print()
        # print("Initial Orbital Elements:")
        # print(f"  a = {oe0[0]:.1f} km")
        # print(f"  e = {oe0[1]:.6f}")
        # print(f"  i = {np.degrees(oe0[2]):.2f}°")
        # print(f"  Ω = {np.degrees(oe0[3]):.2f}°")
        # print(f"  ω = {np.degrees(oe0[4]):.2f}°")
        # print(f"  M = {np.degrees(oe0[5]):.2f}°")
        # print()

        # Time spans
        tSpanSec = self.tSpan

        print(tSpanSec)

        nP = int(tSpanSec / dtSec) + 1
        tSpan = np.linspace(0, tSpanSec, nP)

        # Integrate
        sol = solve_ivp(self.getODE, [0, tSpanSec], oe0, method='RK45', t_eval=tSpan, rtol=1e-9, atol=1e-12)

        # print()
        # print("Final Orbital Elements:")
        # print(f"  a = {sol.y[0, -1]:.1f} km (Δ = {sol.y[0, -1] - oe0[0]:.6f} km)")
        # print(f"  e = {sol.y[1, -1]:.6f} (Δ = {sol.y[1, -1] - oe0[1]:.6e})")
        # print(f"  i = {np.rad2deg(sol.y[2, -1]):.6f}° (Δ = {np.rad2deg(sol.y[2, -1]) - np.rad2deg(oe0[2]):.6f}°)")
        # print(f"  Ω = {np.rad2deg(sol.y[3, -1]):.2f}° (Δ = {np.rad2deg(sol.y[3, -1]) - np.rad2deg(oe0[3]):.6f}°)")
        # print(f"  ω = {np.rad2deg(sol.y[4, -1]):.2f}° (Δ = {np.rad2deg(sol.y[4, -1]) - np.rad2deg(oe0[4]):.6f}°)")
        # print()

        yNew = np.zeros((6,len(sol.t)))
        yNew[0,:] = sol.y[0,:]
        yNew[1,:] = sol.y[1,:]
        yNew[2,:] = sol.y[2,:] % (2*np.pi)
        yNew[3,:] = sol.y[3,:] % (2*np.pi)
        yNew[4,:] = sol.y[4,:] % (2*np.pi)
        yNew[5,:] = sol.y[5,:] % (2*np.pi)

        return sol.t, yNew
    
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
    
    def getRHist(self, oe, t):

        r = np.zeros((3,len(t)))
        for k in range(len(t)):
            r[:,k], _ = self.oe2rv(oe[:,k])

        return r
    
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
    
    def plot3d(self, t, oe):
        """" Plotting 3D orbit trajectory around Earth """

        # Subsample for plotting
        idx = np.linspace(0, len(t)-1, len(t), dtype=int)
        
        # Convert to positions
        positions = np.zeros((3, len(idx)))
        for i, j in enumerate(idx):
            r, _ = self.oe2rv(oe[:, j])
            positions[:, i] = r
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Earth
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_earth = self.earthR * np.outer(np.cos(u), np.sin(v))
        y_earth = self.earthR * np.outer(np.sin(u), np.sin(v))
        z_earth = self.earthR * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.4)
        
        # Plot orbit
        ax.plot(positions[0, :], positions[1, :], positions[2, :], 
                'r-', linewidth=1, label='Orbit')
        ax.plot([positions[0, 0]], [positions[1, 0]], [positions[2, 0]], 
                'go', markersize=12, label='Start')
        ax.plot([positions[0, -1]], [positions[1, -1]], [positions[2, -1]], 
                'r*', markersize=15, label='End')
        
        epochUTCfinal = spice.et2utc(self.epochETFinal, "ISOC", 0)

        # Labels
        ax.set_xlabel('X [km]', fontsize=12)
        ax.set_ylabel('Y [km]', fontsize=12)
        ax.set_zlabel('Z [km]', fontsize=12)
        ax.set_title(f'Orbit Propagation - {self.epochUTC} to {epochUTCfinal}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        
        # Equal aspect ratio
        max_range = np.array([
            positions[0, :].max() - positions[0, :].min(),
            positions[1, :].max() - positions[1, :].min(),
            positions[2, :].max() - positions[2, :].min()
        ]).max() / 2.0
        
        mid = np.array([
            (positions[0, :].max() + positions[0, :].min()) * 0.5,
            (positions[1, :].max() + positions[1, :].min()) * 0.5,
            (positions[2, :].max() + positions[2, :].min()) * 0.5
        ])
        
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        plt.tight_layout()
        return fig
    
    def plotOEs(self, t, oe):
        """ Plotting the orbital elements """

        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('Orbital Element Evolution', fontsize=14, fontweight='bold')
        
        # Semi-major axis
        axes[0, 0].plot(t, oe[0, :], 'b-', linewidth=2)
        axes[0, 0].set_ylabel('Semi-major Axis [km]')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title('Semi-major Axis')
        
        # Eccentricity
        axes[0, 1].plot(t, oe[1, :], 'r-', linewidth=2)
        axes[0, 1].set_ylabel('Eccentricity')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title('Eccentricity')
        
        # Inclination
        axes[1, 0].plot(t, np.rad2deg(oe[2, :]), 'g-', linewidth=2)
        axes[1, 0].set_ylabel('Inclination [deg]')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title('Inclination')
        
        # RAAN
        axes[1, 1].plot(t, np.rad2deg(oe[3, :]), 'm-', linewidth=2)
        axes[1, 1].set_ylabel('RAAN [deg]')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title('Right Ascension of Ascending Node')
        
        # Argument of periapsis
        axes[2, 0].plot(t/60, np.rad2deg(oe[4, :]), 'c-', linewidth=2)
        axes[2, 0].set_ylabel('Arg. of Periapsis [deg]')
        axes[2, 0].set_xlabel('Time [min]')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_title('Argument of Periapsis')
        
        # Mean anomaly
        axes[2, 1].plot(t/60, np.rad2deg(np.mod(oe[5, :], 2*np.pi)), 'orange', linewidth=2)
        axes[2, 1].set_ylabel('Mean Anomaly [deg]')
        axes[2, 1].set_xlabel('Time [min]')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_title('Mean Anomaly')
        
        plt.tight_layout()
        return fig
    
    def __del__(self):
        """ Clean up SPICE kernels """
        try:
            spice.kclear()
        except:
            pass











   

