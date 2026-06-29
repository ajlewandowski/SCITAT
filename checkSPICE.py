"""
This file will be a test case to ensure the NASA SPICE (SpiceyPy) package is working correctly

If this script runs completely, and no errors occur with correct outputs, you are good to go!
"""

import os
import sys
import urllib.request
from pathlib import Path
import spiceypy as spice

def checkSpiceyPy():
    """Check if spiceypy is installed."""
    try: 
        version = spice.tkvrsn('TOOLKIT')
        print("spiceypy is installed")
        print(f"  Version: {version}")
        return True
    except Exception as e:
        print("spiceypy is NOT installed")
        return False
    
def checkExample():
    """Check spicepy files working properly with quick examples"""
    # Load the meta-kernel from home directory
    metakernel = Path.home() / ".spice_kernels" / "orbit_prop.tm"
    
    if not metakernel.exists():
        print(f"Meta-kernel not found at: {metakernel}")
        print("\nKernels not downloaded yet. Run this first:")
        print("  python download_spice_kernels.py")
        return False
    
    try:
        spice.furnsh(str(metakernel))
    except Exception as e:
        print(f"Failed to load kernels: {e}")
        return False
    
    # Test 1: Time conversion

    print("="*70)
    print("Example 1: Time conversion")
    print("="*70)
    try:
        et = spice.str2et("2025-01-01T00:00:00")
        print(f"  '2025-01-01T00:00:00' → {et:.2f} seconds past J2000")
        print("  Time conversion works if output is 788961669.18 sec")
    except Exception as e:
        print(f"  Time conversion failed: {e}")
        return False
    print()
    print()

    # Test 2: Moon position
    print("="*70)
    print("Example 2: Moon ephemeris")
    print("="*70)
    try:
        moon_state, lt = spice.spkezr("MOON", et, "J2000", "NONE", "EARTH")
        moon_pos = moon_state[:3]
        moon_dist = (moon_pos[0]**2 + moon_pos[1]**2 + moon_pos[2]**2)**0.5
        print(f"  Moon position (2025-01-01):")
        print(f"    X: {moon_pos[0]:>12.3f} km")
        print(f"    Y: {moon_pos[1]:>12.3f} km")
        print(f"    Z: {moon_pos[2]:>12.3f} km")
        print(f"  Distance: {moon_dist:.1f} km")
        print("  Moon ephemeris works if distance output is 381735.7 km")
    except Exception as e:
        print(f"  Moon ephemeris failed: {e}")
        return False
    print()
    print()
    
    # Test 3: Sun position
    print("="*70)
    print("Example 3: Sun ephemeris")
    print("="*70)
    try:
        sun_state, lt = spice.spkezr("SUN", et, "J2000", "NONE", "EARTH")
        sun_pos = sun_state[:3]
        sun_dist = (sun_pos[0]**2 + sun_pos[1]**2 + sun_pos[2]**2)**0.5
        print(f"  Sun distance: {sun_dist:.1f} km ({sun_dist/1.496e8:.3f} AU)")
        print("   Sun ephemeris works if distance output is 147107542.1 km")
    except Exception as e:
        print(f"  Sun ephemeris failed: {e}")
        return False
    print()
    print()
    
    # Test 4: Planetary constants
    print("="*70)
    print("Example 4: Planetary constants")
    print("="*70)
    try:
        moon_gm = spice.bodvrd("MOON", "GM", 1)[1][0]
        earth_gm = spice.bodvrd("EARTH", "GM", 1)[1][0]
        print(f"  Earth GM: {earth_gm:.4f} km³/s²")
        print(f"  Moon GM:  {moon_gm:.4f} km³/s²")
        print("  Planetary constants work if outputs for Earth and Moon are 398600.4355 and 4902.8001 km³/s² respectively")
    except Exception as e:
        print(f"  Planetary constants failed: {e}")
        return False
    print()
    print()
    
    # Clean up
    spice.kclear()
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED - SPICE is working correctly")
    print("="*70)
    
    return True

def main():
    """Main verification routine."""
    print()
    print()
    print("="*70)
    print("SPICE Installation and Verification")
    print("="*70)
    
    # Step 1: Check if spiceypy is installed
    if not checkSpiceyPy():
        print("\nPlease install spiceypy:")
        print("  pip install spiceypy")
        sys.exit(1)
    
    print()
    print()

    # Step 2: Check if kernels work
    if not checkExample():
        print("\n✗ SPICE tests failed")
        print("\nMake sure you've run:")
        print("  python download_spice_kernels.py")
        sys.exit(1)

    print()
    print()
    print("You're all set! SPICE is working perfectly.")
    print()

if __name__ == "__main__":
    main()