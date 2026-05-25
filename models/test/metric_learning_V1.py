import os
import sys
import numpy as np
import pandas as pd
import cv2 as cv
import random
import psutil
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
# Adds the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.preprocessing import preprocess_image

"""NOTE
- Triplet lost method: the image (anchor) is compared to the same identity (positive sample) and a negative sample.
- Model will learn to minimise distances between the anchor and the positive whilst maximising the distance between the anchor and the negative.
- Model output: vectors (embeddings), not class labels -> no training needed for new users (in theory)
- might experiment with mahalanabis distance

Sample training hyperparameters:
epochs = 20
batch_size = 32
lr = 1e-3
optimizer = Adam
margin = 0.3
embedding_dim = 512

"""

DATASET_PATH = 'data/' # ! alter path according to your structure -> temporary fix and must decide on a common structure afterwards

CLASSIFICATION_PATH = os.path.join(DATASET_PATH, 'classification_data/')
VERIFICATION_PATH = os.path.join(DATASET_PATH, 'verification_data/')

TRAIN_PATH = os.path.join(CLASSIFICATION_PATH, 'train_data/')
TEST_PATH = os.path.join(CLASSIFICATION_PATH, 'test_data/')
VAL_PATH = os.path.join(CLASSIFICATION_PATH, 'val_data/')

IMG_SIZE = (224, 224)

EPOCHS = 20

paths = {
    "Dataset": DATASET_PATH,
    "Classification": CLASSIFICATION_PATH,
    "Verification": VERIFICATION_PATH,
    "Train Path": TRAIN_PATH,
    "Test Path": TEST_PATH,
    "Val Path": VAL_PATH
}

for name, path in paths.items():
    if os.path.exists(path):
        continue
    else:
        print(f"path {name} not found with path {path}")


# ! NEEDS TO CLARIFY WITH TUTOR IF FACE NET IS ALLOWED - ELSE, GO WITH RESNET
# FaceNet: https://medium.com/analytics-vidhya/introduction-to-facenet-a-unified-embedding-for-face-recognition-and-clustering-dbdac8e6f02

class MetricLearningModel(nn.Module):
    def __init__(self, embedding_dim=128): # may change to a smaller number
        super().__init__()
        self.features = nn.Sequential( # 4 layers

            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1,1))
        )
        self.embedding = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1) # L2 normalisation for FaceNet

        return x


class TripletDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir

        self.people = {}
        self.ids = []

        for person in os.listdir(root_dir):
            folder = os.path.join(root_dir, person)

            if not os.path.isdir(folder):
                continue

            imgs = [
                os.path.join(folder, x)
                for x in os.listdir(folder)
            ]

            if len(imgs) >= 2:
                self.people[person] = imgs
                self.ids.append(person)

    def __len__(self):
        return sum(len(imgs) for imgs in self.people.values())

    def __getitem__(self, idx):

        anchor_id = random.choice(self.ids)

        negative_id = random.choice(self.ids)
        while negative_id == anchor_id:
            negative_id = random.choice(self.ids)

        anchor_path, positive_path = random.sample(
            self.people[anchor_id], 2
        )

        negative_path = random.choice(
            self.people[negative_id]
        )

        anchor = preprocess_image(anchor_path)
        positive = preprocess_image(positive_path)
        negative = preprocess_image(negative_path)

        # anchor = anchor.to(device, non_blocking=True)
        # positive = positive.to(device, non_blocking=True)
        # negative = negative.to(device, non_blocking=True)

        return anchor, positive, negative

def evaluate_auc(model, pairs_file, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        with open(pairs_file) as f:
            for line in tqdm(f, desc="Evaluating AUC", leave=False):
                path_a, path_b, label = line.strip().split()
                emb_a = model(preprocess_image(path_a).unsqueeze(0).to(device))
                emb_b = model(preprocess_image(path_b).unsqueeze(0).to(device))
                sim = F.cosine_similarity(emb_a, emb_b).item()
                scores.append(sim)
                labels.append(int(label))
    return roc_auc_score(labels, scores)


def cosine_similarity_score(emb_a, emb_b):
    """Returns score in [-1, 1]. Higher = more similar."""
    return F.cosine_similarity(emb_a, emb_b, dim=1)

def euclidean_distance_score(emb_a, emb_b):
    """Returns distance. Lower = more similar. Negated for ROC consistency."""
    dist = torch.norm(emb_a - emb_b, p=2, dim=1)
    return -dist  # negate so higher = more similar, matching cosine convention

def log_memory():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 ** 3  # GB
    print(f"RAM usage: {mem:.2f} GB")

def main():
    
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    # test
    x = torch.randn(3,3).cuda()
    print(x)

    # * training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("formatting dataset for metrics learning")
    dataset = TripletDataset(TRAIN_PATH)
    # loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print("starting data loading")
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=6, # ! parallel image loading workers -> PLEASE CHANGE IF YOU HAVE A WEAKER GPU
        pin_memory=True, # pages memory for faster CPU->GPU transfer
        prefetch_factor=2, # each worker prefetches 2 batches ahead
        persistent_workers=True # don't kill/respawn workers each epoch
    )
    print("modelling")
    model = MetricLearningModel(128).to(device)

    # https://docs.pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
    criterion = nn.TripletMarginLoss(margin=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    epoch_bar = tqdm(range(EPOCHS), desc="Training", unit="epoch")
    # print("starting training loop")
    for epoch in range(EPOCHS):
        # print(f"Epoch: {epoch+1}")
        log_memory()
        model.train()
        total_loss = 0

        batch_bar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False, unit="batch")

        for i, (anchor, pos, neg) in enumerate(batch_bar):
            # anchor = anchor.to(device)
            # pos = pos.to(device)
            # neg = neg.to(device)

            anchor = anchor.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            neg = neg.to(device, non_blocking=True)

            emb_a = model(anchor)
            emb_p = model(pos)
            emb_n = model(neg)

            loss = criterion(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)

            batch_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "batch_loss": f"{loss.item():.4f}"
            })
        auc = evaluate_auc(model, "verification_pairs_val.txt", device)
        epoch_bar.set_postfix({
            "avg_loss": f"{avg_loss:.4f}",
            "epoch": epoch + 1,
            "AUC": f"{auc:.4f}"
        })

        print(epoch, total_loss)

        if total_loss < best_loss:
            best_loss = total_loss

            torch.save(
                model.state_dict(),
                "models/artifacts/metric_learning_128_best.pth"
            )

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": total_loss
        }, "models/artifacts/metric_learning_128.pth")


if __name__ == "__main__":
    main()



