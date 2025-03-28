import os
import subprocess
import sys

def convert_flv_to_wav(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate through files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".flv"):
            input_path = os.path.join(input_folder, filename)
            # Construct output file path with the same filename but with .wav extension
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".wav")
            # Run ffmpeg command to convert FLV to WAV
            subprocess.run(['ffmpeg', '-i', input_path, output_path])

if __name__ == "__main__":
    # Check if input arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python3 flv_to_wav.py input_folder output_folder")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print("Input folder does not exist.")
        sys.exit(1)
    
    convert_flv_to_wav(input_folder, output_folder)
