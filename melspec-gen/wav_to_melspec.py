import os
import librosa
import numpy as np

def generate_mel_spectrogram(input_folder, output_folder, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate through files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_folder, filename)
            # Load audio file
            y, sr = librosa.load(input_path, sr=sr)
            # Generate mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            # Convert to decibels
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            # Save mel spectrogram
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
            plt.figure(figsize=(10, 10))
            librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=hop_length, x_axis=None, y_axis=None)
            plt.axis('off')  # Hide axes
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    # Check if input arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python wav_to_melspec.py input_folder output_folder")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print("Input folder does not exist.")
        sys.exit(1)
    
    generate_mel_spectrogram(input_folder, output_folder)
