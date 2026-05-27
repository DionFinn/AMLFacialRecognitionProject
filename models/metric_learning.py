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
from torch.optim.lr_scheduler import ReduceLROnPlateau   
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision.models import mobilenet_v3_small

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
LR = 1e-3 # had it at 1e-5 but was too little (score of around 0.63)
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
    def __init__(self, embedding_dim=128, normalize=True):
        super().__init__()
        # self.normalize = normalize # True = cosine model, False = euclidean model
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
        # self.embedding = nn.Linear(256, embedding_dim)
        self.embedding = nn.Sequential(
            nn.Linear(256,512),
            nn.BatchNorm1d(512), # hope to stablise embedding distributions
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,embedding_dim)
        )
        # self.scale = nn.Parameter(torch.tensor(10.0)) # learned scale

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.embedding(x)
        # if self.normalize:
        #     x = F.normalize(x, p=2.0, dim=1) # DO NOT CHANGE 2.0 TO INT
        return x
        
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
        # while negative_id == anchor_id: # ensures negative != anchor
        #     # negative_id = random.choice(self.ids)
        #     candidate_ids = [x for x in self.ids if x != anchor_id]
        #     negative_id = random.choice(candidate_ids)

        candidate_ids = [x for x in self.ids if x != anchor_id]
        negative_id = random.choice(candidate_ids)

        # two distinct images from the same identity
        anchor_path, positive_path = random.sample(self.people[anchor_id], 2)

        # one image from the negative identity
        negative_path = random.choice(self.people[negative_id])

        # TODO modify preprocess_image - needs to verify with supervised learning
        anchor = preprocess_image(anchor_path)
        positive = preprocess_image(positive_path)
        negative = preprocess_image(negative_path)

        return anchor, positive, negative

# evaluation
def evaluate_verification(model, pairs_file, device, metric="cosine"):

    model.eval()

    pairs_dir = os.path.dirname(os.path.abspath(pairs_file))
    pairs = []
    unique_paths = set()

    with open(pairs_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
        # pairs = [line.strip().split() for line in f if line.strip()]
            parts = line.split()
            if len(parts) != 3:
                continue
            path_a, path_b, label = parts

            full_a = path_a if os.path.isabs(path_a) else os.path.join(pairs_dir, path_a)
            full_b = path_b if os.path.isabs(path_b) else os.path.join(pairs_dir, path_b)

            # ! DEBUG - WILL REMOVE
            if not os.path.exists(full_a) or not os.path.exists(full_b):
                tqdm.write(f"[SKIP] Missing file in pair")
                continue

            pairs.append((full_a, full_b, int(label)))

            unique_paths.add(full_a)
            unique_paths.add(full_b)

    embedding_cache = {}

    with torch.no_grad():
        for path in tqdm(unique_paths, desc="Caching embeddings", leave=False):
            img = preprocess_image(path)
            img = torch.from_numpy(img).float()
            img = img.unsqueeze(0).to(device)

            embedding = model(img)
            embedding_cache[path] = embedding.squeeze(0).cpu()

    cosine_scores = []
    euclidean_scores = []
    labels = []

    for full_a, full_b, label in tqdm(pairs,desc="Evaluating pairs",leave=False):

        emb_a = embedding_cache[full_a]
        emb_b = embedding_cache[full_b]

        cos_score = F.cosine_similarity(emb_a.unsqueeze(0),emb_b.unsqueeze(0),dim=1).item()
        euc_score = -torch.norm(emb_a - emb_b,p=2).item()

        cosine_scores.append(cos_score)
        euclidean_scores.append(euc_score)
        labels.append(label)

    labels_np = np.array(labels)
    cosine_scores_np = np.array(cosine_scores)
    euclidean_scores_np = np.array(euclidean_scores)

    cos_auc = roc_auc_score(labels_np,cosine_scores_np)
    cos_fpr, cos_tpr, cos_thresholds = roc_curve(labels_np,cosine_scores_np)

    euc_auc = roc_auc_score(labels_np,euclidean_scores_np)
    euc_fpr, euc_tpr, euc_thresholds = roc_curve(labels_np,euclidean_scores_np)
    
    return {
        "cosine": {
            "auc": cos_auc,
            "fpr": cos_fpr.tolist(),
            "tpr": cos_tpr.tolist(),
            "thresholds": cos_thresholds.tolist(),
            "scores": cosine_scores_np.tolist(),
        },
        "euclidean": {
            "auc": euc_auc,
            "fpr": euc_fpr.tolist(),
            "tpr": euc_tpr.tolist(),
            "thresholds": euc_thresholds.tolist(),
            "scores": euclidean_scores_np.tolist(),
        },
        "labels": labels_np.tolist(),
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
    criterion = nn.TripletMarginLoss(margin=MARGIN, p=2) # https://docs.pytorch.org/docs/2.12/generated/torch.nn.TripletMarginLoss.html
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )
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

        results = evaluate_verification(model,PAIRS_FILE,device)
        with open(os.path.join(ARTIFACTS_DIR, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        cosine_auc = results["cosine"]["auc"]
        euclidean_auc = results["euclidean"]["auc"]

        scheduler.step(cosine_auc)

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

        # ! TEST - REMOVE LATER
        # scripted = torch.jit.script(model)
        # print("saving...")
        # scripted.save(
        #     os.path.join(ARTIFACTS_DIR, "metric_learning_scripted.pt")
        # )
        # print("saved")
        # ! TEST - REMOVE LATER

        with torch.no_grad(): # TEST
            sample = next(iter(loader))[0][:32].to(device)
            emb = model(sample)
            emb_std = emb.std(dim=0).mean().item()
            emb_norm = emb.norm(dim=1).mean().item()
            print(f"Embedding STD: {emb_std:.4f}")
            print(f"Embedding Norm: {emb_norm:.4f}")

    print("\nTraining complete.")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Best cosine AUC: {best_cosine_auc:.4f}")
    print(f"History saved : {logger.save_path}")
    print(f"{logger.txt_path}")


if __name__ == "__main__":
    main()