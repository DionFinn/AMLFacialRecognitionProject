# ! NOT DONE YET

import torch
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def save_checkpoint(model, optimizer, epoch, total_loss, best_loss, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": total_loss,
        "best_loss": best_loss
    }

    torch.save(checkpoint, save_path)

    # optionally also save "best model separately"
    if total_loss < best_loss:
        torch.save(model.state_dict(), save_path.replace(".pth", "_best.pth"))
        best_loss = total_loss

    return best_loss



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MetricLearningModel(128).to(device)

checkpoint_path = "models/artifacts/metric_learning_128.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

# load weights
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']}")

auc = evaluate_auc(model, "models/verification_pairs_val.txt", device)
print("Validation AUC:", auc)