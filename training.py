import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import h5py
import io

import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim

def get_statistics(name, set, cases=["Benign", "Malignant"]):
    size = len(set)
    print(f"\n{name} Size: {size}")
    unique, unique_counts = np.unique(set[:, 1], return_counts=True)
    for _, element_count, case in zip(unique, unique_counts, cases):
        print(f"Number of {case} Cases: {element_count}")
        print(f"% of {case} Cases: {100 * element_count / size}%")
    return size

def preload(load_path, save_path, val_split=0.1):
    # Whole Dataset
    gt = pd.read_csv(load_path)[['isic_id', 'target']].to_numpy()
    n = get_statistics("Data Set", gt)
    # Create Train/Val Splits
    np.random.shuffle(gt)
    val_size = int(n * val_split)
    val_gt, train_gt = gt[0:val_size], gt[val_size::]
    # Train/Val Set Summary
    train_n = get_statistics("Training Set", train_gt)
    val_n = get_statistics("Validation Set", val_gt)
    # Sanity Check
    if (train_n + val_n) != n:
        raise Exception("Individual Dataset Sizes do not add up Whole Dataset")
    np.savez(save_path, train=train_gt, val=val_gt)
    return (train_gt, val_gt), (train_n, val_n)

def reload(path):
    sets = np.load(path, allow_pickle=True)
    train_gt, val_gt = sets['train'], sets['val']
    train_n = get_statistics("Training Set", train_gt)
    val_n = get_statistics("Validation Set", val_gt)
    return (train_gt, val_gt), (train_n, val_n)

def load_batch(hdf5, gt_set, i, size=256):
    gt = gt_set[i:i+size]
    keys, y = list(gt[:, 0]), np.array(gt[:, 1], dtype=int)
    with h5py.File(hdf5, "r") as f:
        imgs = [transforms(Image.open(io.BytesIO(f.get(key)[()]))) for key in keys]
    X = torch.stack(imgs).to(device)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    return X, y

def main():
    #  Dataset
    if not os.path.isfile("Data/split.npz"):
        print("Creating New Train/Val Split...saves to Data/split.npz")
        (train_gt, val_gt), (train_n, val_n) = preload("Data/train-metadata.csv", "Data/split.npz")
    else:
        print("Loading Previously Created Train/Val Split...from Data/split.npz")
        (train_gt, val_gt), (train_n, val_n) = reload("Data/split.npz")

    # Model
    if os.path.isfile("best_model.pth"):
        print("Loading Previously Trained Model...from best_model.pth")
        model.load_state_dict(torch.load("best_model.pth"))
    else:
        print("Using Pretrained ResNet18 from PyTorch")

    # Training Setup
    n_benign, n_malignant = 360600, 354
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_benign/n_malignant))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs, batch_size = 2, 1024
    losst, lossv, best_val_loss = [], [], float('inf')

    # Training Loop
    for epoch in range(epochs):
        # Shuffle Sets
        print(f"Epoch {epoch + 1} of {epochs}")
        np.random.shuffle(train_gt)
        np.random.shuffle(val_gt)

        for i in tqdm(range(0, train_n, batch_size)):
            # Standard Training
            model.train()
            X, y = load_batch("Data/train-image.hdf5", train_gt, i)
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

            # Check Loss on Validation Set
            if ((i / batch_size) % 10) == 0:
                losst.append(loss.item())
                model.eval()
                np.random.shuffle(val_gt)
                with torch.no_grad():
                    X, y = load_batch("Data/train-image.hdf5", val_gt, 0)
                    outputs = model(X)
                    lossv.append(loss_fn(outputs, y).item())

                # Save model if new loss is better
                if lossv[-1] < best_val_loss:
                    best_val_loss = lossv[-1]
                    print("Saving model...")
                    torch.save(model.state_dict(), "best_model.pth")

                print(f"Train Loss: {losst[-1]}, Val Loss: {lossv[-1]}")

                # Create Loss History plot
                plt.figure()
                plt.plot(losst, label="Train Loss")
                plt.plot(lossv, label="Val Loss")
                plt.xlabel("Iterations")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig("loss.png")
                plt.close()

if __name__ == "__main__":
    transforms = ResNet18_Weights.IMAGENET1K_V1.transforms()

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)

    device = torch.device("mps")
    model = model.to(device)

    main()
