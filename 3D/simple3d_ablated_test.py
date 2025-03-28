import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from helper_functions import select_train_test, collate_fn, AblatedDataset
from simple3d_train_test import Simple3DCNN, test_step
import sys

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(10)
    torch.cuda.manual_seed(10)

    # Define data folder path and max length of videos
    data_folder = sys.argv[2]
    MAX_LENGTH = 135

    print("Loading data...")
    X_train, X_test, y_train, y_test = select_train_test(data_folder)

    # Create ablated dataloader. Keeps modality sys.argv[1] and masks the other
    dataset = AblatedDataset(X_test, y_test, data_folder, sys.argv[1])
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: collate_fn(x, MAX_LENGTH, device))

    print("Creating model...")

    model = Simple3DCNN(num_classes=6, modality="multi")

    try:
        model.load_state_dict(torch.load(sys.argv[3])["model_state_dict"])
    except:
        print("Missing or invalid checkpoint. Using untrained Simple3D model...")

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    print("Testing...")
    test_loss, test_acc = test_step(model=model, dataloader=dataloader, loss_fn=loss_fn, device=device)

    print(f"Loss: {test_loss}, Accuracy: {test_acc}")
