import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from pytorch_i3d import InceptionI3d
from helper_functions import select_train_test, collate_fn, UnimodalDataset, MultimodalDataset
import sys

def train_step(model, dataloader, loss_fn, optimizer, device):
    """
    A helper function that defines the training step for I3D.
    """
    model.train()

    num, train_loss, train_acc = 0, 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.type(torch.FloatTensor).to(device), torch.tensor(y).to(device)

        y_pred = model(X)
        y_pred = F.adaptive_avg_pool3d(y_pred, output_size=(1, 1, 1)).squeeze()

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
    A helper function that defines the test step for I3D.
    """
    model.eval()

    num, test_loss, test_acc = 0, 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.type(torch.FloatTensor).to(device), torch.tensor(y).to(device)


            test_pred_logits = model(X)
            test_pred_logits = F.adaptive_avg_pool3d(test_pred_logits, output_size=(1, 1, 1)).squeeze()

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
          max_length,
          modality,
          device,
          checkpoint=None):

    """
    The training loop for I3D. Saves the model and optimizer state dicts,
    and the training and testing results after every epoch.
    """

    # Load checkpoint if defined
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

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: collate_fn(x, max_length, device))
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: collate_fn(x, max_length, device))

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

    EPOCHS = 50
    MAX_LENGTH = 135
    modality = sys.argv[1]


    print("Creating model...")

    model = InceptionI3d(num_classes=400, spatial_squeeze=False)

    try:
        model.load_state_dict(torch.load(sys.argv[4]))
    except:
        print("Missing or invalid pretraining checkpoint. Using untrained I3D")

    model.replace_logits(num_classes=6)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    print("Train and testing...")

    res = train(model=model,
                model_name=f"i3d_{modality}",
                epochs=EPOCHS,
                loss_fn=loss_fn,
                optimizer=optimizer,
                data_folder=sys.argv[2],
                output_folder=sys.argv[3],
                max_length=MAX_LENGTH,
                modality=modality,
                device=device,
                checkpoint=None)
