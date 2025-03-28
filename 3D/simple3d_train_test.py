import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from helper_functions import select_train_test, collate_fn, UnimodalDataset, MultimodalDataset
import sys

class Simple3DCNN(nn.Module):
    """
    A small 3D CNN model with 6 3D CNN layers followed by a final fully connected
    classification layer. The primary purpose is to serve as a baseline for 3D
    emotion recognition.
    """
    def __init__(self, num_classes, modality):
        super(Simple3DCNN, self).__init__()
        self.modality = modality

        self.conv1 = nn.Conv3d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(8, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(144, num_classes)
        self.fc2 = nn.Linear(336, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.flatten(x)
        if (self.modality == "multi"):
            x = self.fc2(x)
        else:
            x = self.fc1(x)
        return x

def train_step(model, dataloader, loss_fn, optimizer, device):
    """
    A helper function that defines the training step for Simple3D.
    """
    model.train()

    num, train_loss, train_acc = 0, 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X, torch.tensor(y).to(device)
        if (device == "cpu"):
            X = X.type(torch.FloatTensor)
        else:
            X = X.type(torch.cuda.FloatTensor)

        y_pred = model(X)

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
    A helper function that defines the test step for Simple3D.
    """
    model.eval()

    num, test_loss, test_acc = 0, 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X, torch.tensor(y).to(device)
            if (device == "cpu"):
                X = X.type(torch.FloatTensor)
            else:
                X = X.type(torch.cuda.FloatTensor)

            test_pred_logits = model(X)

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
          MAX_LENGTH,
          modality,
          device,
          checkpoint=None):

    """
    The training loop for Simple3D. Saves the model and optimizer state dicts,
    and the training and testing results after every epoch.
    """

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
        train_dataset = MultimodalDataset(X_train, y_train, data_folder)
        test_dataset = MultimodalDataset(X_test, y_test, data_folder)
    else:
        train_dataset = UnimodalDataset(X_train, y_train, data_folder, modality)
        test_dataset = UnimodalDataset(X_test, y_test, data_folder, modality)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: collate_fn(x, MAX_LENGTH, device))
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: collate_fn(x, MAX_LENGTH, device))

    # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        total_train_loss, total_train_acc, total_train_num = 0, 0, 0
        total_test_loss, total_test_acc, total_test_num = 0, 0, 0
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

        print(f"Epoch: {epoch+1}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}" )

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

    # Define number of EPOCHS, MAX_LENGTH of videos, modality to train
    EPOCHS = 50
    MAX_LENGTH = 135
    modality = sys.argv[1]

    print("Creating model...")

    model = Simple3DCNN(num_classes=6, modality=modality).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    print("Train and testing...")

    res = train(model=model,
                model_name=f"simple3d_{modality}",
                epochs=EPOCHS,
                loss_fn=loss_fn,
                optimizer=optimizer,
                data_folder=sys.argv[2],
                output_folder=sys.argv[3],
                MAX_LENGTH=MAX_LENGTH,
                modality=modality,
                device=device,
                checkpoint=None)
