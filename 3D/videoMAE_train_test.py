from transformers import VideoMAEForVideoClassification, VideoMAEConfig
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch import optim
from torch.utils.data import DataLoader, Dataset
from helper_functions import select_train_test
import sys

class UnimodalDataset(Dataset):
    """
    Custom unimodal dataloader for VideoMAE.
    """
    def __init__(self, data, labels, folder_path, modality, num_frames):
        self.data = data
        self.labels = labels
        self.folder_path = folder_path
        self.modality = modality
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file = self.data[idx]

        if (self.modality == "vision"):
            X = np.load(f"{self.folder_path}/{file}/video.npy")
        elif (self.modality == "audio"):
            X = np.load(f"{self.folder_path}/{file}/audio.npy")

        # Sample num_frames evenly across the video
        indices = np.linspace(0, X.shape[1] - 1, self.num_frames, dtype=int)

        X = X[:, indices]

        return X, self.labels[idx]

class MultimodalDataset(Dataset):
    """
    Custom multimodal dataloader for VideoMAE.
    """
    def __init__(self, data, labels, folder_path, num_frames):
        self.data = data
        self.labels = labels
        self.folder_path = folder_path
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file = self.data[idx]

        video = np.load(f"{self.folder_path}/{file}/video.npy")
        audio = np.load(f"{self.folder_path}/{file}/audio.npy")

        new_video = []
        new_audio = []

        # Sample num_frames evenly across the video
        indices = np.linspace(0, video.shape[1] - 1, self.num_frames, dtype=int)

        # Further resize frames and spectrogram chunks
        for i in indices:
            frame = cv2.resize(video[:, i, :, :].transpose(2, 1, 0), (208, 224), interpolation=cv2.INTER_LINEAR)
            spect = cv2.resize(audio[:, i, :, :].transpose(2, 1, 0), (16, 224), interpolation=cv2.INTER_LINEAR)
            new_video.append(frame.transpose(2, 1, 0))
            new_audio.append(spect.transpose(2, 1, 0))

        fused = np.concatenate((np.array(new_video), np.array(new_audio)), axis=2)
        fused = fused.transpose(1, 0, 2, 3)

        return fused, self.labels[idx]

def train_step(model, dataloader, loss_fn, optimizer, device):
    """
    A helper function that defines the training step for VideoMAE.
    """
    model.train()

    num, train_loss, train_acc = 0, 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.type(torch.FloatTensor).to(device), y.to(device)
        X = X.permute(0, 2, 1, 3, 4)

        y_pred = model(X).logits

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class==y).sum().item()
        num += len(y_pred_class)

    train_loss /= batch+1
    train_acc /= num

    return(train_loss, train_acc)

def test_step(model, dataloader, loss_fn, device):
    """
    A helper function that defines the test step for VideoMAE.
    """
    model.eval()

    num, test_loss, test_acc = 0, 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.type(torch.FloatTensor).to(device), y.to(device)
            X = X.permute(0, 2, 1, 3, 4)

            test_pred_logits = model(X).logits

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels==y).sum().item()
            num += len(test_pred_labels)

    test_loss /= batch+1
    test_acc /= num

    return(test_loss, test_acc)

def train(model,
          model_name,
          epochs,
          loss_fn,
          optimizer,
          data_folder,
          output_folder,
          modality,
          max_length,
          device,
          checkpoint=None):

    """
    The training loop for VideoMAE. Saves the model and optimizer state dicts,
    and the training and testing results after every epoch.
    """

    # Load in checkpoint if defined
    if (checkpoint != None):
        states = torch.load(checkpoint)
        model.load_state_dict(states["model_state_dict"])
        optimizer.load_state_dict(states["optimizer_state_dict"])
        results = states["results"]
    else:
        results = {"train_loss": [],
                   "train_acc": [],
                   "test_loss": [],
                   "test_acc": []}

    length = len(results["train_loss"])
    print(f"Continuing from Epoch: {length}")
    X_train, X_test, y_train, y_test = select_train_test(data_folder)

    if (modality == "multi"):
        train_dataset = MultimodalDataset(X_train, y_train, data_folder, max_length)
        test_dataset = MultimodalDataset(X_test, y_test, data_folder, max_length)
    else:
        train_dataset = UnimodalDataset(X_train, y_train, data_folder, modality, max_length)
        test_dataset = UnimodalDataset(X_test, y_test, data_folder, modality, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        total_train_loss, total_train_acc = 0, 0
        total_test_loss, total_test_acc = 0, 0
        print("-----------")
        print(f"Epoch: {epoch+1}")
        print("Training")

        train_loss, train_acc = train_step(model=model,
                   dataloader=train_dataloader,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   device=device)

        print("Testing")

        test_loss, test_acc = test_step(model=model,
                  dataloader=test_dataloader,
                  loss_fn=loss_fn,
                  device=device)

        print(f"Epoch: {epoch+1}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # Save current model and results
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "results": results
        }, f"{output_folder}/{model_name}_{epoch+length}.pt")

    return (results)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(10)
    torch.cuda.manual_seed(10)

    EPOCHS = 50
    MAX_LENGTH = 32
    modality = sys.argv[1]

    print("Creating model...")

    config = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    config.num_frames = MAX_LENGTH
    config.num_labels = 6

    model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics", config=config, ignore_mismatched_sizes=True)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    print("Train and testing...")

    res = train(model=model,
                model_name=f"videomae_{modality}",
                epochs=EPOCHS,
                loss_fn=loss_fn,
                optimizer=optimizer,
                data_folder=sys.argv[2],
                output_folder=sys.argv[3],
                modality=modality,
                max_length=MAX_LENGTH,
                device=device,
                checkpoint=None)
