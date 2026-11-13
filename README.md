# SCITAT - Satellite Constellation Imaging Trade Analysis Tool
This repository is a part of a greater effort to provide extended tradespace analysis and design tools for satellite constellation architectures. The intent is to extend the capabilities of NASA TAT-C (which is integrated into this repository) for advanced distributed spacecraft mission design that includes end-to-end systematic uncertainties. This work is part of a doctoral program at The University of Texas at Austin for Alec Lewandowski, under primary investigator Dr. Lori Magruder. The lab is the 3D Geospatial Laboratory at UT's Center for Space Research: https://magruder3dgl.com/. 

Citations:
P. T. Grogan et al., “Tradespace Analysis Toolkit for Constellations (TAT-C),” version X.X.X, Python package, 20XX. [Online]. Available: https://github.com/code-lab-org/tatc, doi: 10.5281/zenodo.17363628.
Annex et al., (2020). SpiceyPy: a Pythonic Wrapper for the SPICE Toolkit. Journal of Open Source Software, 5(46), 2050, https://doi.org/10.21105/joss.02050.

Run this command to install all requirements in bash: pip install -r requirements.txt
Download the SPICE kernels needed to pull ephemeris data in cmd: python downloadSpiceKernels.py
Test if SPICE is working correctly by running this file in cmd: python checkSPICE.py
For live TLE data
    pip install skyfield
    pip install cartopy
