import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from pathlib import Path

def computeMVSCostMap(gsdMats, offNadAngles, sigmaPix, eMax):
    """
    Computes MVS cost map where unusable pixels have been set to np.nan.
    All input matrices must still be the same dimensional shape (e.g., 1000x1000).
    """
    # 1. 3D Vectorization
    gsdCube = np.stack(gsdMats, axis=0)
    
    # 2. Dynamic View Counting
    # Count how many valid (non-NaN) views exist FOR EACH PIXEL
    validViews = np.sum(~np.isnan(gsdCube), axis=0)
    
    # Create a mask for pixels that don't meet the stereo requirement (< 2 views)
    invalidStereoMask = validViews < 2
    
    # 3. Base Uncertainty (Ignoring NaNs)
    # Convert NaNs to infinity temporarily just to avoid math warnings, 
    # though np.nansum handles NaNs automatically.
    safeGSDCube = np.where(np.isnan(gsdCube), np.inf, np.clip(gsdCube, a_min=1e-6, a_max=None))
    wCube = 1.0 / ((sigmaPix * safeGSDCube) ** 2)
    
    # Use np.nansum to add up the weights, completely ignoring the "cut" NaN pixels
    wSumMat = np.nansum(wCube, axis=0)
    
    # Prevent divide-by-zero where weight_sum is 0 (invalid pixels)
    with np.errstate(divide='ignore'):
        baseErrMap = np.where(wSumMat > 0, np.sqrt(1.0 / wSumMat), np.inf)
    
    # 4. Geometry Penalty (Assumes global pass geometry)
    convergenceDeg = np.max(offNadAngles) - np.min(offNadAngles)
    theta = np.radians(np.clip(abs(convergenceDeg), 1.0, 179.0))
    geometricPenalty = 1.0 / np.sin(theta)
    
    # 5. Redundancy Penalty (Now calculated PER PIXEL)
    alpha = 0.5
    # Pixels with 5 views get a much better score than pixels with only 2 views!
    redundancyPenalty = np.where(
        validViews >= 2, 
        1.0 / (1.0 - np.exp(-alpha * (validViews - 1))), 
        np.inf
    )
    
    # 6. Total Error and Normalization
    totalErrorMap = baseErrMap * geometricPenalty * redundancyPenalty
    costMap = 1.0 - np.exp(-((totalErrorMap / eMax) ** 2))
    
    # Force any pixel with < 2 views to immediately fail with a cost of 1.0
    costMap[invalidStereoMask] = 1.0
    
    return np.clip(costMap, 0.0, 1.0)


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Generate 10 mock GSD matrices (1000x1000 pixels each)
    # Using random GSDs between 0.3m and 0.6m to simulate different satellite passes
    data1 = np.load(Path("D:/Users/al62352/ResolutionHeatmaps/LS9_705km/GSDarray_matrix_705_km_offnadir_(3.0,15.0).npy"))
    data2 = np.load(Path("D:/Users/al62352/ResolutionHeatmaps/LS9_705km/GSDarray_matrix_705_km_offnadir_(0.0,15.0).npy"))
    data3 = np.load(Path("D:/Users/al62352/ResolutionHeatmaps/LS9_705km/GSDarray_matrix_705_km_offnadir_(0.0,0.0).npy"))
    data4 = np.load(Path("D:/Users/al62352/ResolutionHeatmaps/LS9_705km/GSDarray_matrix_705_km_offnadir_(1.0,20.0).npy"))
    data5 = np.load(Path("D:/Users/al62352/ResolutionHeatmaps/LS9_705km/GSDarray_matrix_705_km_offnadir_(2.0,25.0).npy"))

    dataMats = [data1, data2, data3, data4, data5]
    
    # 10 corresponding off-nadir angles tracking the target
    # Ranging from looking backward (-25) to looking forward (+30)
    angles = [16.0, 15.0, 0.0, 20.5, 25.0]
    
    final_cost_matrix = computeMVSCostMap(
        gsdMats=dataMats,
        offNadAngles=angles,
        sigmaPix=0.5,
        eMax=35.0
    )
    
    print(f"Cost processed for {5} views!")
    print(f"Sample output cost at [3098, 3098]: {final_cost_matrix[3098, 3098]:.8f}")