"""
Standalone SPICE Kernel Downloader with ISS Support
Copy this entire file and save as: download_spice_kernels.py
Then run: python download_spice_kernels.py
"""

import os
import urllib.request
from pathlib import Path

def download_kernel(url, filepath):
    """Download a file from URL to filepath."""
    try:
        print(f"  Downloading: {filepath.name}...", end=" ", flush=True)
        urllib.request.urlretrieve(url, filepath)
        print("✓")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def main():
    print("="*70)
    print("SPICE Kernel Setup for Orbit Propagator")
    print("="*70)
    
    # Create kernel directory
    kernel_dir = Path.home() / ".spice_kernels"
    kernel_dir.mkdir(exist_ok=True)
    print(f"\nKernel directory: {kernel_dir}")
    
    # Define kernels to download
    base_url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"
    
    kernels = {
        "naif0012.tls": f"{base_url}/lsk/naif0012.tls",
        "gm_de440.tpc": f"{base_url}/pck/gm_de440.tpc",
        "pck00011.tpc": f"{base_url}/pck/pck00011.tpc",
        "de440s.bsp": f"{base_url}/spk/planets/de440s.bsp",
        "earth_000101_260125_251230.bpc": f"{base_url}/pck/earth_000101_260125_251230.bpc",
    }
    
    print("\nDownloading SPICE kernels...")
    print("(This may take a few minutes - de440s.bsp is ~100 MB)")
    print()
    
    success_count = 0
    for filename, url in kernels.items():
        filepath = kernel_dir / filename
        if filepath.exists():
            print(f"  ✓ Already exists: {filename}")
            success_count += 1
        else:
            if download_kernel(url, filepath):
                success_count += 1
    
    print(f"\n✓ Downloaded {success_count}/{len(kernels)} kernels")
    
    # Download ISS kernel
    print("\n" + "="*70)
    print("Downloading ISS Ephemeris Kernel")
    print("="*70)
    print()
    
    iss_url = f"{base_url}/spk/stations/iss.bsp"
    iss_filepath = kernel_dir / "iss.bsp"
    
    if iss_filepath.exists():
        print(f"  ✓ Already exists: iss.bsp")
        iss_success = True
    else:
        print("  Note: ISS kernel is ~50-100 MB and may take a minute...")
        iss_success = download_kernel(iss_url, iss_filepath)
    
    if iss_success:
        print("\n✓ ISS kernel downloaded successfully!")
    else:
        print("\n⚠ ISS kernel download failed")
        print("  You can:")
        print("  1. Try downloading manually from:")
        print(f"     {iss_url}")
        print(f"  2. Save it to: {iss_filepath}")
        print("  3. Or use TLE data instead (see get_iss_elements.py)")
    
    # Create meta-kernel
    metakernel_path = kernel_dir / "orbit_prop.tm"
    print(f"\nCreating meta-kernel: {metakernel_path}")
    
    # Include ISS kernel if available
    kernel_list = [
        "'$KERNELS/naif0012.tls'",
        "'$KERNELS/gm_de440.tpc'",
        "'$KERNELS/pck00011.tpc'",
        "'$KERNELS/de440s.bsp'"
    ]
    
    if iss_success:
        kernel_list.append("'$KERNELS/iss.bsp'")
    
    metakernel_content = f"""\\begindata
    PATH_VALUES = ( '{kernel_dir}' )
    PATH_SYMBOLS = ( 'KERNELS' )
    KERNELS_TO_LOAD = (
    {chr(10).join('  ' + k for k in kernel_list)}
    )
    \\begintext
    """
    
    with open(metakernel_path, 'w') as f:
        f.write(metakernel_content)
    
    print("✓ Meta-kernel created")
    
    # Test SPICE
    print("\n" + "="*70)
    print("Testing SPICE")
    print("="*70)
    
    try:
        import spiceypy as spice
        print("✓ SpiceyPy imported")
        
        spice.furnsh(str(metakernel_path))
        print("✓ Kernels loaded successfully")
        
        et = spice.str2et("2024-01-01T00:00:00")
        print(f"✓ Time conversion works: {et:.2f} seconds past J2000")
        
        moon_state, lt = spice.spkezr("MOON", et, "J2000", "NONE", "EARTH")
        moon_dist = (moon_state[0]**2 + moon_state[1]**2 + moon_state[2]**2)**0.5
        print(f"✓ Moon ephemeris works: distance = {moon_dist:.1f} km")
        
        # Test ISS if kernel was downloaded
        if iss_success:
            try:
                # ISS NAIF ID is -125544
                iss_state, lt = spice.spkezr("-125544", et, "J2000", "NONE", "EARTH")
                iss_r = (iss_state[0]**2 + iss_state[1]**2 + iss_state[2]**2)**0.5
                iss_alt = iss_r - 6378.137  # Altitude above Earth
                print(f"✓ ISS ephemeris works: altitude = {iss_alt:.1f} km")
            except Exception as e:
                print(f"⚠ ISS test failed: {e}")
                print("  ISS data may not be available for this date")
        
        spice.kclear()
        
        print("\n" + "="*70)
        print("✓✓✓ SUCCESS - SPICE is ready to use! ✓✓✓")
        print("="*70)
        print(f"\nYour meta-kernel path:")
        print(f"  {metakernel_path}")
        print("\nKernels loaded:")
        print("  • Planetary ephemerides (Moon, Sun, planets)")
        print("  • Leap seconds")
        print("  • Physical constants")
        if iss_success:
            print("  • ISS ephemeris")
        
        print("\nTo use in your code:")
        print("  import spiceypy as spice")
        print("  from pathlib import Path")
        print()
        print("  # Load kernels")
        print(f"  metakernel = Path.home() / '.spice_kernels' / 'orbit_prop.tm'")
        print("  spice.furnsh(str(metakernel))")
        print()
        print("  # Get ISS state")
        print("  et = spice.str2et('2024-06-01T00:00:00')")
        print("  state, _ = spice.spkezr('-125544', et, 'J2000', 'NONE', 'EARTH')")
        print("  r = state[:3]  # Position [km]")
        print("  v = state[3:6]  # Velocity [km/s]")
        
        if not iss_success:
            print("\n" + "="*70)
            print("ISS KERNEL NOT AVAILABLE")
            print("="*70)
            print("\nFor real-time ISS tracking, use TLE data instead:")
            print("  pip install skyfield")
            print("  # See get_iss_elements.py for example code")
        
    except ImportError:
        print("✗ SpiceyPy not installed")
        print("\nInstall it with:")
        print("  pip install spiceypy")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("\nKernels downloaded but test failed.")
        print("You may still be able to use them.")

if __name__ == "__main__":
    main()