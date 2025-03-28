import numpy as np
import cv2
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

class UnimodalDataset(Dataset):
    """
    A custom unimodal dataloader for the 3D CNN models
    """
    def __init__(self, data, labels, folder_path, modality):
        self.data = data
        self.labels = labels
        self.folder_path = folder_path
        self.modality = modality

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file = self.data[idx]

        if (self.modality == "vision"):
            X = np.load(f"{self.folder_path}/{file}/video.npy")
        elif (self.modality == "audio"):
            X = np.load(f"{self.folder_path}/{file}/audio.npy")

        return X, self.labels[idx]

class MultimodalDataset(Dataset):
    """
    A custom multimodal dataloader for the 3D CNN models
    """
    def __init__(self, data, labels, folder_path):
        self.data = data
        self.labels = labels
        self.folder_path = folder_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file = self.data[idx]

        video = np.load(f"{self.folder_path}/{file}/video.npy")
        audio = np.load(f"{self.folder_path}/{file}/audio.npy")

        return np.concatenate((video, audio), axis=3), self.labels[idx]

class AblatedDataset(Dataset):
    """
    A custom dataloader that masks a defined modality to test/analyze the 3D CNN
    models' multimodal interaction.
    """
    def __init__(self, data, labels, folder_path, modality):
        self.data = data
        self.labels = labels
        self.folder_path = folder_path
        self.modality = modality

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file = self.data[idx]

        if (self.modality == "vision"):
            X = np.load(f"{self.folder_path}/{file}/video.npy")
            pad = np.zeros(shape=X.shape, dtype=np.uint8)

            X = np.concatenate((X, pad), axis=3)
        elif (self.modality == "audio"):
            X = np.load(f"{self.folder_path}/{file}/audio.npy")
            pad = np.zeros(shape=X.shape, dtype=np.uint8)

            X = np.concatenate((pad, X), axis=3)

        return X, self.labels[idx]

def collate_fn(batch, max_length, device):
    """
    A custom collate_fn for 3D CNN dataloaders that pads videos with blank frames
    until they reach max_length
    """
    videos, labels = zip(*batch)
    CHANNELS = videos[0].shape[0]
    WIDTH, HEIGHT = videos[0].shape[2:]

    padded_videos = None
    for video in videos:
        pad = np.zeros((CHANNELS, max_length-video.shape[1], WIDTH, HEIGHT), dtype=np.int8)
        padded_video = np.concatenate((video, pad), axis=1)

        if (device == "cpu"):
            padded_video = torch.tensor(padded_video).unsqueeze(0).to(device).type(torch.uint8)
        else:
            padded_video = torch.tensor(padded_video).unsqueeze(0).to(device).type(torch.cuda.ByteTensor)
        if (padded_videos == None):
            padded_videos = padded_video
        else:
            padded_videos = torch.vstack((padded_videos, padded_video))

    return(padded_videos, labels)

def get_all_file_names(folder_path):
    """
    Read all file names in a folder
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return []

    # Get a list of all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    return files

def get_all_dir_names(folder_path):
    """
    Read all directory names in a folder
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return []

    # Get a list of all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    return files

def create_labels(file_names):
    """
    A helper function that returns a list of labels from a list of files
    """
    labels = []

    for file in file_names:
        if ("HAP" in file):
            labels.append(0)
        elif ("SAD" in file):
            labels.append(1)
        elif ("ANG" in file):
            labels.append(2)
        elif ("DIS" in file):
            labels.append(3)
        elif ("NEU" in file):
            labels.append(4)
        elif ("FEA" in file):
            labels.append(5)

    return labels

def select_train_test(data_folder):
    """
    Randomly create training and testing data
    """
    file_names = get_all_dir_names(data_folder)
    labels = create_labels(file_names)

    X_train, X_test, y_train, y_test = train_test_split(file_names, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
