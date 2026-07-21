import os
from pathlib import Path

def preprocess_tle_file(input_file_path, output_file_path, norad_id):
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"The file {input_file_path} does not exist.")
    
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        while True:
            line1 = infile.readline().strip()
            line2 = infile.readline().strip()
            
            # Check if the NORAD ID is in the name line
            if f' {norad_id} ' or f' {norad_id}U' in line1 and line2:
                outfile.write(f"{line1}\n")
                outfile.write(f"{line2}\n")

# Example usage
if __name__ == "__main__":
    norad_id = 39084  # Example NORAD ID for L9
    input_file_path = Path(r"D:\Users\al62352\SpaceTrackTLE\data\exports\tle2025.txt")  # Path to the full TLE file
    output_file_path = f'tle{norad_id}_new.txt'  # Path to the filtered TLE file

    
    preprocess_tle_file(input_file_path, output_file_path, norad_id)
    print(f"Filtered TLEs saved to {output_file_path}")