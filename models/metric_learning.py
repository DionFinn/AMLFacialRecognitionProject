import os
import sys
import json
import random
import psutil
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

# Adds the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.preprocessing import preprocess_image
from logger import TrainingLogger

"""
NOTE
- Triplet loss method: the image (anchor) is compared to the same identity (positive sample) and a negative sample.
- Model will learn to minimise distances between the anchor and the positive whilst maximising the distance between the anchor and the negative.
- Model output: vectors (embeddings), not class labels -> no training needed for new users (in theory)
"""

# file paths
DATASET_PATH = "data/" # ! alter path according to your structure -> temporary fix and must decide on a common structure afterwards

CLASSIFICATION_PATH = os.path.join(DATASET_PATH, "classification_data/")
VERIFICATION_PATH = os.path.join(DATASET_PATH, "verification_data/")

TRAIN_PATH = os.path.join(CLASSIFICATION_PATH, "train_data/")
TEST_PATH = os.path.join(CLASSIFICATION_PATH, "test_data/")
VAL_PATH = os.path.join(CLASSIFICATION_PATH, "val_data/")

PAIRS_FILE = os.path.join(DATASET_PATH, 'verification_pairs_val.txt')
ARTIFACTS_DIR = "models/artifacts/"

IMG_SIZE = (224, 224)

EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-5
MARGIN = 0.3 # TODO EXPERIMENT WITH MARGINS
EMBEDDING_DIM = 128 # 128 dimensional plot
NUM_WORKERS = 8 # specific to my GPU - may change to lower/higher


def depthwise_separable_conv(in_ch, out_ch): # only for readability purposes
    return nn.Sequential(
        # depthwise - one filter per input channel
        nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(),
        # pointwise - mix channels cheaply with 1x1
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )

class MetricLearningModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            depthwise_separable_conv(3, 32),
            nn.MaxPool2d(2),
            depthwise_separable_conv(32, 64),
            nn.MaxPool2d(2),
            depthwise_separable_conv(64, 128),
            nn.MaxPool2d(2),
            depthwise_separable_conv(128, 256),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.embedding = nn.Linear(256, embedding_dim)
        self.scale = nn.Parameter(torch.tensor(10.0)) # learned scale

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.embedding(x)
        x = F.normalize(x, p=2.0, dim=1) # DO NOT CHANGE 2.0 TO INT
        return x * self.scale # scaled unit vector

# transforming datasset for Triplet method for metric learning
# ONLY constructs file paths
class TripletDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.people = {} # gives out {"n000003": ("../img1.jpg", "../img2.jpg", ..)}
        self.ids = [] # gives out ["n000003", ...]

        for person in os.listdir(root_dir):
            folder = os.path.join(root_dir, person)
            if not os.path.isdir(folder):
                continue

            imgs = tuple( # lists every images in this person's id - tuple used to not be immutable
                os.path.join(folder, x)
                for x in os.listdir(folder)
            )
            if len(imgs) >= 2: # metric learning can't work unless an anchor and a positive are present
                self.people[person] = imgs
                self.ids.append(person)

    def __len__(self):
        # true dataset size: total number of images across all identities
        return sum(len(imgs) for imgs in self.people.values())

    def __getitem__(self, idx): # gets the idx from the self.ids above (key)
        # idx is intentionally ignored - triplets are sampled randomly
        anchor_id = random.choice(self.ids)

        negative_id = random.choice(self.ids)
        while negative_id == anchor_id: # ensures negative != anchor
            negative_id = random.choice(self.ids)

        # two distinct images from the same identity
        anchor_path, positive_path = random.sample(self.people[anchor_id], 2)

        # one image from the negative identity
        negative_path = random.choice(self.people[negative_id])

        # TODO modify preprocess_image - needs to verify with supervised learning
        anchor = preprocess_image(anchor_path)
        positive = preprocess_image(positive_path)
        negative = preprocess_image(negative_path)

        return anchor, positive, negative
    
# evaluation metrics
def cosine_similarity_score(emb_a, emb_b):
    return F.cosine_similarity(emb_a, emb_b, dim=1)

def euclidean_distance_score(emb_a, emb_b):
    return -torch.norm(emb_a - emb_b, p=2, dim=1)

# evaluation
def evaluate_verification(model, pairs_file, device, metric="cosine"):

    model.eval()
    scores = []
    labels = []

    pairs_dir = os.path.dirname(os.path.abspath(pairs_file))

    with open(pairs_file, "r") as f:
        pairs = [line.strip() for line in f if line.strip()]

    with torch.no_grad():
        for line in tqdm(pairs, desc=f"Evaluating [{metric}]", leave=False):
            parts = line.split()
            if len(parts) != 3:
                continue

            path_a, path_b, label = parts
            full_a = path_a if os.path.isabs(path_a) else os.path.join(pairs_dir, path_a)
            full_b = path_b if os.path.isabs(path_b) else os.path.join(pairs_dir, path_b)

            if not os.path.exists(full_a) or not os.path.exists(full_b):
                tqdm.write(f"[SKIP] missing file in pair")
                continue

            result_a = preprocess_image(full_a)
            result_b = preprocess_image(full_b)

            # ! convert numpy array -> may modify preprocessing.py
            def to_tensor(x):
                if isinstance(x, torch.Tensor):
                    return x
                if isinstance(x, np.ndarray):
                    # shape is (H, W, C) from cv2 - convert to (C, H, W)
                    t = torch.from_numpy(x).float()
                    if t.ndim == 3 and t.shape[-1] in (1, 3, 4):
                        t = t.permute(2, 0, 1)
                    return t
                return None

            tensor_a = to_tensor(result_a)
            tensor_b = to_tensor(result_b)
                                            
            tensor_a = tensor_a.unsqueeze(0).to(device)
            tensor_b = tensor_b.unsqueeze(0).to(device) 
            emb_a = model(tensor_a)
            emb_b = model(tensor_b)

            if metric == "cosine":
                score = F.cosine_similarity(emb_a, emb_b, dim=1).item()
                # score = cosine_similarity_score(emb_a, emb_b).item()
            else:
                # score = euclidean_distance_score(emb_a, emb_b).item()
                score = -torch.norm(emb_a - emb_b, p=2, dim=1).item()

            scores.append(score)
            labels.append(int(label))

    scores_np = np.array(scores)
    labels_np = np.array(labels)
    auc_score = roc_auc_score(labels_np, scores_np)
    fpr, tpr, thresholds = roc_curve(labels_np, scores_np)

    return {
        "auc": auc_score,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "scores": scores_np.tolist(),
        "labels": labels_np.tolist(),
        "metric": metric,
    }

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # dataset and loader
    print("Indexing dataset to match with Triplet format")
    dataset = TripletDataset(TRAIN_PATH)
    print(f"{len(dataset):,} total images across {len(dataset.ids):,} identities\n")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # model, loss, optimiser 
    model = MetricLearningModel(EMBEDDING_DIM).to(device)
    criterion = nn.TripletMarginLoss(margin=MARGIN) # https://docs.pytorch.org/docs/2.12/generated/torch.nn.TripletMarginLoss.html
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # lgogger
    logger = TrainingLogger()
    logger.set_meta(
        embedding_dim=EMBEDDING_DIM,
        batch_size=BATCH_SIZE,
        optimizer="Adam",
        lr=LR,
        margin=MARGIN,
    )

    # training loop 
    best_cosine_auc = 0.0
    best_loss = float("inf")

    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        batch_bar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False, unit="batch")

        for i, (anchor, pos, neg) in enumerate(batch_bar):
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
                "avg_loss": f"{avg_loss:.4f}",
                "batch_loss": f"{loss.item():.4f}",
            })

        # per-epoch evaluation
        cosine_auc = None
        euclidean_auc = None

        # ! THIS IS WHERE EVALUATION TAKES PLACE
        cos_results = evaluate_verification(model, PAIRS_FILE, device, metric="cosine")
        euc_results = evaluate_verification(model, PAIRS_FILE, device, metric="euclidean")
        cosine_auc = cos_results["auc"]
        euclidean_auc = euc_results["auc"]

        # logging 
        logger.log_epoch(epoch, avg_loss, cosine_auc, euclidean_auc)

        epoch_bar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "cos_auc": f"{cosine_auc:.4f}" if cosine_auc is not None else "-",
            "euc_auc": f"{euclidean_auc:.4f}" if euclidean_auc is not None else "-",
        })

        # always save the latest checkpoint for resume
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "cosine_auc": cosine_auc,
            "euclidean_auc": euclidean_auc,
        }, os.path.join(ARTIFACTS_DIR, "metric_learning_latest.pth"))

        # save best loss model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.state_dict(),
                os.path.join(ARTIFACTS_DIR, "metric_learning_best_loss.pth")
            )
            tqdm.write(f"Best loss model saved with (loss: {avg_loss:.4f})")

        # save AUC model
        if cosine_auc is not None and cosine_auc > best_cosine_auc:
            best_cosine_auc = cosine_auc
            torch.save(
                model.state_dict(),
                os.path.join(ARTIFACTS_DIR, "metric_learning_best_auc.pth")
            )
            # TorchScript for deployment (saved for )
            scripted = torch.jit.script(model)
            scripted.save(
                os.path.join(ARTIFACTS_DIR, "metric_learning_scripted.pt")
            )

            tqdm.write(f"Best AUC model saved with (cosine AUC: {cosine_auc:.4f})")

    print("\nTraining complete.")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Best cosine AUC: {best_cosine_auc:.4f}")
    print(f"History saved : {logger.save_path}")
    print(f"{logger.txt_path}")


if __name__ == "__main__":
    main()