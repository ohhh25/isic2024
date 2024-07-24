import numpy as np
import pandas as pd

import os

import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import v2

device = torch.device("mps")
model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)

transforms = v2.Compose([ResNet18_Weights.IMAGENET1K_V1.transforms])

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

if not os.path.isfile("Data/split.npz"):
    print("Creating New Train/Val Split...saves to Data/split.npz")
    (train_gt, val_gt), (train_n, val_n) = preload("Data/train-metadata.csv", "Data/split.npz")
else:
    print("Loading Previously Created Train/Val Split...from Data/split.npz")
    (train_gt, val_gt), (train_n, val_n) = reload("Data/split.npz")

print(model)
