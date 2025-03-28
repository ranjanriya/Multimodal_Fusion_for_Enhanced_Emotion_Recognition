import cv2
import os
import numpy as np
import time
import sys
from helper_functions import get_all_file_names

def convert_video_to_frames(video_path, output_folder):
    """
    Function to extract and save frames as a numpy array from CREMA-D videos.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video information (width, height, frames per second, etc.)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    print(f"Video Information - Width: {width}, Height: {height}, FPS: {fps}, Reduced FPS: {fps*0.8}, Total Frames: {total_frames}, Reduced Total Frames: {int(total_frames*0.8)}")

    video = []

    # Loop through each frame and save it as an image
    frame_number = 0
    while True:
        ret, frame = cap.read()

        # Break the loop if we have reached the end of the video
        if not ret:
            break

        # Delay frame capture rate to reduce memory size. Capture at 24 FPS instead of 30
        if (frame_number % 5 != 0):
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            video.append(frame)

        frame_number += 1

    cap.release()

    for i in range(len(video)):
        video[i] = cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB)

    video = np.array(video)
    video = np.transpose(video, (3, 0, 1, 2))

    np.save(f"{output_folder}/video.npy", video)

    print(f"Frames saved to: {output_folder}")
    print(f"Video shape: {video.shape}")

    return video.shape[1]

def convert_spect_to_chunks(spect_path, output_folder, length):
    """
    Function to divide mel spectrograms into chunks along the temporal axis.
    The number of chunks, length, is the number of frames the corresponding video has.
    Chunks are saved as a numpy array.
    """
    img = cv2.imread(spect_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width = img.shape[:2]
    split_width = width // length

    audio = []

    for i in range(length):
        left = i * split_width
        right = min((i+1) * split_width, width)

        split_img = img[:, left:right]
        split_img = cv2.resize(split_img, (224, 224), interpolation=cv2.INTER_LINEAR)

        audio.append(split_img)

    audio = np.array(audio)
    audio = np.transpose(audio, (3, 0, 1, 2))
    np.save(f"{output_folder}/audio.npy", audio)

    print(f"Audio shape: {audio.shape}")

def get_subject_files(file_names, subject):
    """
    Helper function to get all files of a subject
    """
    files = [file for file in file_names if f"{subject:03d}" in file]
    return files

def convert(video_names, spect_names, video_folder, audio_folder, output_folder):
    """
    Function to create 3D vision and audio data. Data is saved to an output folder/
    """
    video_files = sorted(video_names)
    spect_files = sorted(spect_names)
    max_length = 0

    for i in range(91):
        subject_video_files = get_subject_files(video_files, i+1)
        subject_spect_files = get_subject_files(spect_files, i+1)
        for j in range(len(subject_video_files)):
            length = convert_video_to_frames(f"{video_folder}/{subject_video_files[j]}", f"{output_folder}/{os.path.splitext(subject_video_files[j])[0]}")
            if (length > max_length):
                max_length = length
            convert_spect_to_chunks(f"{audio_folder}/{subject_spect_files[j]}", f"{output_folder}/{os.path.splitext(subject_video_files[j])[0]}", length)

    print(f"MAX VIDEO FRAMES: {max_length}")

if __name__ == "__main__":
    video_folder = sys.argv[1]
    audio_folder = sys.argv[2]
    output_folder = sys.argv[3]
    video_files_temp = get_all_file_names(video_folder)
    audio_files = get_all_file_names(audio_folder)

    video_files = []

    for f in video_files_temp:
        if ("1064_IEO_DIS_MD" in f or "1064_TIE_SAD_XX" in f or "1076_MTI_NEU_XX" in f):
            print("SKIPPING.")
            continue
        video_files.append(f)

    convert(video_files, audio_files, video_folder, audio_folder, output_folder)
