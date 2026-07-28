import numpy as np

class rpcClass:
    def __init__(self, line_off, samp_off, lat_off, lon_off, height_off,
                 line_scale, samp_scale, lat_scale, lon_scale, height_scale,
                 row_coeffs, col_coeffs):
        self.line_off = line_off
        self.samp_off = samp_off
        self.lat_off = lat_off
        self.lon_off = lon_off
        self.height_off = height_off
        self.line_scale = line_scale
        self.samp_scale = samp_scale
        self.lat_scale = lat_scale
        self.lon_scale = lon_scale
        self.height_scale = height_scale
        self.row_coeffs = np.array(row_coeffs).reshape((4, 8))
        self.col_coeffs = np.array(col_coeffs).reshape((4, 8))

    def _evaluate_polynomial(self, coeffs, line_norm, samp_norm):
        result = 0.0
        for i in range(4):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        for m in range(2):
                            index = i * 8 + j * 4 + k * 2 + l
                            result += coeffs[i, index] * (line_norm ** i) * (samp_norm ** j)
        return result

    def _evaluate_rational_polynomial(self, line_norm, samp_norm, coeffs):
        num = self._evaluate_polynomial(coeffs[:, :8], line_norm, samp_norm)
        den = 1 + self._evaluate_polynomial(coeffs[:, 8:], line_norm, samp_norm)
        return num / den

    def image_to_ground(self, line, sample, height=0.0):
        """
        Convert image coordinates to ground coordinates using RPC.
        
        Parameters:
        - line: Image row coordinate
        - sample: Image column coordinate
        - height: Ground height (default is 0.0)
        
        Returns:
        - lon: Ground longitude
        - lat: Ground latitude
        """
        line_norm = (line - self.line_off) / self.line_scale
        samp_norm = (sample - self.samp_off) / self.samp_scale
        height_norm = (height - self.height_off) / self.height_scale

        # Evaluate the rational polynomials
        lon_denorm = self._evaluate_rational_polynomial(line_norm, samp_norm, self.col_coeffs)
        lat_denorm = self._evaluate_rational_polynomial(line_norm, samp_norm, self.row_coeffs)

        lon = lon_denorm * self.lon_scale + self.lon_off
        lat = lat_denorm * self.lat_scale + self.lat_off

        return lon, lat

    def ground_to_image(self, lon, lat, height=0.0):
        """
        Convert ground coordinates to image coordinates using RPC.
        
        Parameters:
        - lon: Ground longitude
        - lat: Ground latitude
        - height: Ground height (default is 0.0)
        
        Returns:
        - line: Image row coordinate
        - sample: Image column coordinate
        """
        # This is a simplified version and may not converge for all cases.
        # A more robust implementation would use numerical optimization techniques like Levenberg-Marquardt.
        raise NotImplementedError("Ground to image conversion using RPC is complex and requires iterative methods.")