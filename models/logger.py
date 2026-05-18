# saves everything into the artifacts folder

import os
import sys
import json
from datetime import datetime
# Adds the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ARTIFACTS_DIR = "models/artifacts/"

# training history
class TrainingLogger:
    def __init__(self, save_path=os.path.join(ARTIFACTS_DIR, "training_history.json")):
        self.save_path = save_path
        self.txt_path = save_path.replace(".json", ".txt")
        self.history = {
            "meta":   {
                "started_at": datetime.now().isoformat(),
                "embedding_dim": None,
                "batch_size": None,
                "optimizer": None,
                "lr": None,
                "margin": None,
            },
            "epochs": [],
        }

    def set_meta(self, **kwargs):
        self.history["meta"].update(kwargs)
        self._save()

    def log_epoch(self, epoch, train_loss, cosine_auc=None, euclidean_auc=None):
        entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "train_loss": round(train_loss, 6),
            "cosine_auc": round(cosine_auc, 6),
            "euclidean_auc": round(euclidean_auc, 6),
        }
        self.history["epochs"].append(entry)
        self._save()
        return entry

    def _save(self):
        with open(self.save_path, "w") as f:
            json.dump(self.history, f, indent=2)

        with open(self.txt_path, "w") as f:
            m = self.history["meta"]
            f.write(f"Training started: {m['started_at']}\n")
            f.write(f"Embedding dim: {m['embedding_dim']}\n")
            f.write(f"Batch size: {m['batch_size']}\n")
            f.write(f"Optimizer: {m['optimizer']}\n")
            f.write(f"Learning rate: {m['lr']}\n")
            f.write(f"Triplet margin: {m['margin']}\n")
            f.write("\n")
            f.write(f'{"Epoch":<8} {"Loss":<14} {"Cosine AUC":<16} {"Euclidean AUC"}\n')
            f.write("-" * 54 + "\n")
            for e in self.history["epochs"]:
                cos = f'{e["cosine_auc"]:.6f}' if e["cosine_auc"] is not None else "N/A"
                euc = f'{e["euclidean_auc"]:.6f}' if e["euclidean_auc"] is not None else "N/A"
                f.write(f'{e["epoch"]:<8} {e["train_loss"]:<14.6f} {cos:<16} {euc}\n')
