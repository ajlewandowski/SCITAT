import numpy as np
from advancedRPCClass import advancedRPCClass

# Example usage
if __name__ == "__main__":
    # Initialize RPCs with example coefficients (these should be obtained from your satellite imagery metadata)
    line_off = 3096.0
    samp_off = 3096.0
    lat_off = 45.0
    lon_off = -120.0
    height_off = 0.0
    line_scale = 3096.0
    samp_scale = 3096.0
    lat_scale = 180.0 / np.pi
    lon_scale = 180.0 / np.pi
    height_scale = 25000.0

    row_coeffs_a = [
        # Coefficients for row polynomial (Rij) for image A
        1.39574746e-07, -1.43651660e-08, -1.75164925e-09, -2.74619369e-11,
        2.23766616e-08, 8.82818623e-11, 1.34375269e-11, -4.22863635e-14,
        # Coefficients for row polynomial (Rij') for image A
        -1.05505382e-12, 4.29640683e-14, 7.09961682e-15, 4.45343957e-17,
        -3.27374943e-14, -1.03774360e-16, -3.45440088e-17, 1.38679685e-19,
        # Add more coefficients as needed
    ]

    col_coeffs_a = [
        # Coefficients for column polynomial (Cij) for image A
        1.23146087e-07, -1.23169674e-08, -1.59703710e-09, -2.44116198e-11,
        2.17194730e-08, 7.89613888e-11, 1.15639281e-11, -3.74545011e-14,
        # Coefficients for column polynomial (Cij') for image A
        -1.04960615e-12, 4.13945461e-14, 6.84701116e-15, 4.38984634e-17,
        -3.23464637e-14, -9.99832885e-17, -3.34608872e-17, 1.34638973e-19,
        # Add more coefficients as needed
    ]

    row_coeffs_b = [
        # Coefficients for row polynomial (Rij) for image B
        1.38574746e-07, -1.42651660e-08, -1.74164925e-09, -2.73619369e-11,
        2.22766616e-08, 8.81818623e-11, 1.33375269e-11, -4.21863635e-14,
        # Coefficients for row polynomial (Rij') for image B
        -1.04505382e-12, 4.28640683e-14, 7.08961682e-15, 4.44343957e-17,
        -3.26374943e-14, -1.04774360e-16, -3.44440088e-17, 1.37679685e-19,
        # Add more coefficients as needed
    ]

    col_coeffs_b = [
        # Coefficients for column polynomial (Cij) for image B
        1.22146087e-07, -1.22169674e-08, -1.58703710e-09, -2.43116198e-11,
        2.16194730e-08, 7.88613888e-11, 1.14639281e-11, -3.73545011e-14,
        # Coefficients for column polynomial (Cij') for image B
        -1.03960615e-12, 4.12945461e-14, 6.83701116e-15, 4.37984634e-17,
        -3.22464637e-14, -9.98832885e-17, -3.33608872e-17, 1.33638973e-19,
        # Add more coefficients as needed
    ]

    # Ensure row_coeffs and col_coeffs have the correct number of elements
    assert len(row_coeffs_a) == 4 * 8, "Row coefficients for image A must have 32 elements."
    assert len(col_coeffs_a) == 4 * 8, "Column coefficients for image A must have 32 elements."
    assert len(row_coeffs_b) == 4 * 8, "Row coefficients for image B must have 32 elements."
    assert len(col_coeffs_b) == 4 * 8, "Column coefficients for image B must have 32 elements."

    rpc_a = advancedRPCClass(
        line_off=line_off,
        samp_off=samp_off,
        lat_off=lat_off,
        lon_off=lon_off,
        height_off=height_off,
        line_scale=line_scale,
        samp_scale=samp_scale,
        lat_scale=lat_scale,
        lon_scale=lon_scale,
        height_scale=height_scale,
        row_coeffs=row_coeffs_a,
        col_coeffs=col_coeffs_a
    )

    rpc_b = advancedRPCClass(
        line_off=line_off,
        samp_off=samp_off,
        lat_off=lat_off,
        lon_off=lon_off,
        height_off=height_off,
        line_scale=line_scale,
        samp_scale=samp_scale,
        lat_scale=lat_scale,
        lon_scale=lon_scale,
        height_scale=height_scale,
        row_coeffs=row_coeffs_b,
        col_coeffs=col_coeffs_b
    )

    # Example GSD matrices (6196x6196)
    gsd_matrix_a = np.random.uniform(0.5, 2.0, (6196, 6196))  # Replace with actual data
    gsd_matrix_b = np.random.uniform(0.5, 2.0, (6196, 6196))  # Replace with actual data

    # Example image coordinates (line, sample)
    line = 3096
    sample = 3096

    # Calculate stereo cost at the given pixel
    cost = advancedRPCClass.calculate_stereo_cost(rpc_a, rpc_b, gsd_matrix_a, gsd_matrix_b, line, sample)
    print(f"Stereo Cost at Line={line}, Sample={sample}: {cost}")

    # Optionally, compute costs for a region or entire image
    total_cost = 0.0
    num_pixels = 100  # Example: Compute cost for 100x100 region around the center
    for i in range(line - num_pixels // 2, line + num_pixels // 2):
        for j in range(sample - num_pixels // 2, sample + num_pixels // 2):
            total_cost += advancedRPCClass.calculate_stereo_cost(rpc_a, rpc_b, gsd_matrix_a, gsd_matrix_b, i, j)

    average_cost = total_cost / (num_pixels * num_pixels)
    print(f"Average Stereo Cost in Region: {average_cost}")