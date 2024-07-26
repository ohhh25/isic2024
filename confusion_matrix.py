import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

from training import reload, load_batch

def get_cmatrix(gt_set, threshold=0.5, batch_size=1024):
    cmatrix = np.zeros((2, 2), dtype=int)    # Start with zeros
    for i in tqdm(range(0, len(gt_set), batch_size)):
        # Get Predictions and Ground Truth Labels
        with torch.no_grad():
            X, y = load_batch(hdf5, transforms, gt_set, device, i, size=batch_size)
            outputs = (model(X) >= threshold).float()
        
        # Add up TN, FP, FN, TP counts
        pos, neg, posp, negp = (y == 1), (y == 0), (outputs == 1), (outputs == 0)
        tn, fp = int(torch.sum(neg & negp)), int(torch.sum(neg & posp))
        fn, tp = int(torch.sum(pos & negp)), int(torch.sum(pos & posp))
        cmatrix += np.array([[tn, fp], [fn, tp]])    # Add counts to Matrix

        del X, y, outputs, pos, neg, posp, negp, tn, fp, fn, tp
        torch.mps.empty_cache()

    return cmatrix

def vis_cmatrix(cmatrix, title, save_path):
    plt.figure()
    labels = ["Benign", "Malignant"]
    sns.heatmap(cmatrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Data
    hdf5 = "Data/train-image.hdf5"
    (train_gt, val_gt), (train_n, val_n) = reload("Data/split.npz")
    transforms = ResNet18_Weights.IMAGENET1K_V1.transforms()

    # Model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load("best_model.pth"))

    # Inference
    device = torch.device("mps")
    model = model.to(device)
    model.eval();

    train_cmatrix = get_cmatrix(train_gt, threshold=0.5, batch_size=1024)
    vis_cmatrix(train_cmatrix, "Training Set Confusion Matrix", "train_cmatrix.png")
    Image.open("train_cmatrix.png").show()

    val_cmatrix = get_cmatrix(val_gt, threshold=0.5, batch_size=1024)
    vis_cmatrix(val_cmatrix, "Validation Set Confusion Matrix", "val_cmatrix.png")
    Image.open("val_cmatrix.png").show()
