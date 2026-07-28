import numpy as np
from rpcClass import rpcClass

class advancedRPCClass(rpcClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_look_vector(self, line, sample, height=0.0):
        """
        Calculate the unit vector from the ground point 
        towards the satellite using numerical differentiation.
        """
        eps = 1e-6
        
        # Perturb line and sample to see how lon/lat changes
        lon_l, lat_l = self.image_to_ground(line + eps, sample, height)
        lon_s, lat_s = self.image_to_ground(line, sample + eps, height)
        
        d_lon_d_line = (lon_l - lon) / eps
        d_lat_d_line = (lat_l - lat) / eps
        d_lon_d_samp = (lon_s - lon) / eps
        d_lat_d_samp = (lat_s - lat) / eps
        
        # Construct the vector in ground coordinates
        vec = np.array([d_lon_d_line, d_lat_d_line, 1.0])
        
        # Normalize to get unit vector
        return vec / np.linalg.norm(vec)

def calculate_intersection_angle(look_vector_a, look_vector_b):
    """
    Calculate the intersection angle between two look vectors.
    """
    dot_product = np.dot(look_vector_a, look_vector_b)
    cos_theta = np.clip(dot_product, -1.0, 1.0)  # Ensure within valid range for arccos
    theta = np.arccos(cos_theta)
    return theta

def normalize_intersection_angle(theta):
    """
    Normalize the intersection angle to a value between 0 and 1.
    
    Intersection angles close to 5 degrees or 85 degrees are penalized more.
    """
    # Convert radians to degrees
    theta_degrees = np.degrees(theta)
    
    # Define minimum and maximum acceptable angles in degrees
    min_angle = 5.0
    max_angle = 85.0
    
    # Normalize the angle using a sigmoid-like function
    if theta_degrees < min_angle:
        return 1.0 - (theta_degrees / min_angle)
    elif theta_degrees > max_angle:
        return 1.0 - ((90 - theta_degrees) / (90 - max_angle))
    else:
        # Linear scaling within the acceptable range
        return (theta_degrees - min_angle) / (max_angle - min_angle)

def normalize_gsd(gsd, max_expected_gsd):
    """
    Normalize GSD value to a value between 0 and 1.
    
    Higher GSD values contribute more to the cost.
    """
    if gsd <= 0:
        return 1.0  # Highly penalize zero or negative GSD
    normalized_gsd = min(gsd / max_expected_gsd, 1.0)  # Ensure it doesn't exceed 1
    return normalized_gsd

def calculate_stereo_cost(rpc_a, rpc_b, gsd_matrix_a, gsd_matrix_b, line, sample, max_expected_gsd=2.0):
    """
    Calculate the stereo cost at a given pixel using RPCs and GSD matrices.
    
    Parameters:
    - rpc_a: AdvancedRPC object for image A
    - rpc_b: AdvancedRPC object for image B
    - gsd_matrix_a: Ground Sampling Distance matrix for image A (6196x6196)
    - gsd_matrix_b: Ground Sampling Distance matrix for image B (6196x6196)
    - line: Image row coordinate (pixel y-coordinate)
    - sample: Image column coordinate (pixel x-coordinate)
    - max_expected_gsd: Maximum expected GSD value for normalization
    
    Returns:
    - cost: Stereo cost at the given pixel, normalized to [0, 1]
    """
    # Get look vectors from both RPCs
    look_vector_a = rpc_a.get_look_vector(line, sample)
    look_vector_b = rpc_b.get_look_vector(line, sample)
    
    # Calculate intersection angle
    theta = calculate_intersection_angle(look_vector_a, look_vector_b)
    
    # Normalize intersection angle to [0, 1]
    normalized_theta_cost = normalize_intersection_angle(theta)
    
    # Get GSD values at the given pixel from both matrices
    gsd_a = gsd_matrix_a[line, sample]
    gsd_b = gsd_matrix_b[line, sample]
    
    # Normalize GSD values to [0, 1]
    normalized_gsd_cost_a = normalize_gsd(gsd_a, max_expected_gsd)
    normalized_gsd_cost_b = normalize_gsd(gsd_b, max_expected_gsd)
    
    # Average the GSD costs
    average_gsd_cost = (normalized_gsd_cost_a + normalized_gsd_cost_b) / 2.0
    
    # Define weights for each component
    weight_intersection = 0.7
    weight_gsd = 0.3
    
    # Combine costs with weights
    total_cost = (weight_intersection * normalized_theta_cost +
                  weight_gsd * average_gsd_cost)
    
    # Ensure the total cost is between 0 and 1
    total_cost = np.clip(total_cost, 0.0, 1.0)
    
    return total_cost
