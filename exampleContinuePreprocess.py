# preprocessTLE.py
import os

def preprocess_tle_file(input_file_path, output_file_path, norad_id):
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"The file {input_file_path} does not exist.")
    
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        while True:
            name = infile.readline().strip()
            line1 = infile.readline().strip()
            line2 = infile.readline().strip()
            
            if not name or not line1 or not line2:
                break
            
            # Check if the NORAD ID is in the first TLE line
            if f' {norad_id} ' in line1:
                outfile.write(f"{line1}\n")
                outfile.write(f"{line2}\n")

# Example usage
if __name__ == "__main__":
    input_file_path = '/Users/al62352/SCITAT/data/tle2025.txt'  # Update this path to the correct location of tle2025.txt on your macOS system
    output_file_path = f'/Users/al62352/SCITAT/data/tle49260.txt'  # Ensure the output path is also correct
    norad_id = 49260  # Example NORAD ID for ISS