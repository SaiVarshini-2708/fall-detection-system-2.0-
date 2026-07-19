# train.py
# Loads dataset.csv, trains the CNN-LSTM model for 20 epochs, saves checkpoints.
# Logs per-epoch metrics to training_log.csv and saves the best model by val loss.

import sys
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Sibling modules — works when run from ml/src/ or ml/
sys.path.insert(0, str(Path(__file__).parent))
from model   import FallDetectorCNN_LSTM
from augment import augment_window

# ------------------------------------------------------------------ #
# Paths (relative to ml/)                                            #
# ------------------------------------------------------------------ #
DATASET_CSV  = Path(__file__).parent.parent / "data"   / "dataset.csv"
MODELS_DIR   = Path(__file__).parent.parent / "models"
BEST_MODEL   = MODELS_DIR / "best_model.pt"
TRAINING_LOG = MODELS_DIR / "training_log.csv"

# ------------------------------------------------------------------ #
# Label encoding maps                                                #
# ------------------------------------------------------------------ #
FALL_TYPE_ENC: dict[str, int] = {
    "none": 0,   # placeholder for ADL windows — filtered out of fall_type loss
    "slip": 0,   # maps to model class 0
    "trip": 1,   # maps to model class 1
    "faint": 2,  # maps to model class 2
}
PRE_ACTIVITY_ENC: dict[str, int] = {
    "walking": 0, "standing": 1, "bending": 2, "sitting": 3,
}

# ------------------------------------------------------------------ #
# Hyper-parameters                                                   #
# ------------------------------------------------------------------ #
BATCH_SIZE   = 32
N_EPOCHS     = 30   # increased from 20 — larger augmented dataset needs more passes
LR           = 1e-3
LR_PATIENCE  = 5   # increased from 3 — give LR more time before reducing

# Weights for the combined loss
W_FALL_TYPE    = 0.8
W_PRE_ACTIVITY = 0.8

# Train / val / test split fractions (must sum to 1.0)
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC  = 0.15  (the remainder)


class FallWindowDataset(Dataset):
    """
    PyTorch Dataset wrapping the flat feature CSV.

    Each row contains 1200 feature values (f_0 … f_1199) that map to a
    (200, 6) IMU window, plus integer-encoded labels for all three tasks.
    """

    def __init__(self, features: np.ndarray, labels: dict) -> None:
        """
        Parameters
        ----------
        features : np.ndarray
            Shape (N, 1200) float32.
        labels : dict
            Keys: 'fall', 'fall_type', 'pre_activity' — each np.ndarray of shape (N,).
        """
        self.features     = torch.tensor(features, dtype=torch.float32)
        self.fall         = torch.tensor(labels["fall"],         dtype=torch.float32)
        self.fall_type    = torch.tensor(labels["fall_type"],    dtype=torch.long)
        self.pre_activity = torch.tensor(labels["pre_activity"], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        # Reshape flat 1200-vector → (200, 6) window
        window = self.features[idx].reshape(200, 6)
        return window, {
            "fall":         self.fall[idx],
            "fall_type":    self.fall_type[idx],
            "pre_activity": self.pre_activity[idx],
        }

#start
def load_data() -> tuple[np.ndarray, dict]:
    """
    Load and encode the dataset CSV.

    Returns
    -------
    tuple
        (features_np, labels_dict) where labels_dict has keys:
        'fall', 'fall_type', 'pre_activity' — each shape (N,).
    """
    print(f"[TRAIN] Loading dataset from {DATASET_CSV} ...")

    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found at {DATASET_CSV}\n"
            "Run:  python src/dataset_builder.py  first."
        )

    df = pd.read_csv(str(DATASET_CSV))
    print(f"[TRAIN] Loaded {len(df):,} rows, {df.shape[1]} columns")

    # Feature columns are named f_0 … f_1199
    feature_cols = [f"f_{i}" for i in range(1200)]
    X = df[feature_cols].values.astype(np.float32)

    # Encode categorical labels to integers
    fall_enc = df["fall_label"].values.astype(np.int32)

    fall_type_enc = (
        df["fall_type"]
        .map(FALL_TYPE_ENC)
        .fillna(0)
        .astype(np.int64)
        .values
    )
    pre_enc = (
        df["pre_activity"]
        .map(PRE_ACTIVITY_ENC)
        .fillna(0)
        .astype(np.int64)
        .values
    )

    labels = {
        "fall":         fall_enc,
        "fall_type":    fall_type_enc,
        "pre_activity": pre_enc,
    }
    return X, labels


def split_data(X: np.ndarray, labels: dict):
    """
    Stratified 70/15/15 train / val / test split on fall_label.

    Uses integer indices throughout so label arrays stay perfectly in sync
    with the feature matrix without any float-matching hacks.

    Returns
    -------
    tuple of (X_train, X_val, X_test, labels_train, labels_val, labels_test)
    """
    n       = len(X)
    indices = np.arange(n)
    strat   = labels["fall"]  # stratify on binary fall label

    # First split: 70% train vs 30% temp
    idx_train, idx_tmp = train_test_split(
        indices, test_size=(1 - TRAIN_FRAC), stratify=strat, random_state=42
    )

    # Second split: val=50% of temp, test=50% of temp  (15% / 15% of total)
    strat_tmp = strat[idx_tmp]
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=0.5, stratify=strat_tmp, random_state=42
    )

    def _subset(idx):
        return {k: v[idx] for k, v in labels.items()}

    labels_train = _subset(idx_train)
    labels_val   = _subset(idx_val)
    labels_test  = _subset(idx_test)

    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]

    print(
        f"[TRAIN] Split — train: {len(X_train):,}  "
        f"val: {len(X_val):,}  test: {len(X_test):,}"
    )
    return X_train, X_val, X_test, labels_train, labels_val, labels_test, idx_test


def compute_loss(
    outputs: dict,
    targets: dict,
    bce_loss: nn.BCELoss,
    ce_loss: nn.CrossEntropyLoss,
) -> torch.Tensor:
    """
    Compute the combined multi-task loss.

    Loss formula:
      total = loss_fall + 0.8*loss_fall_type + 0.8*loss_pre_activity

    loss_fall_type is only computed on windows that are actual falls
    (fall_label == 1), because ADL windows have fall_type='none' which
    is not a real fall-type category and would confuse the classifier.

    Parameters
    ----------
    outputs : dict
        Model outputs with keys 'fall', 'fall_type', 'pre_activity'.
    targets : dict
        Ground-truth tensors with the same keys.
    bce_loss : nn.BCELoss
    ce_loss  : nn.CrossEntropyLoss

    Returns
    -------
    torch.Tensor
        Scalar combined loss.
    """
    # Fall detection loss (BCE on sigmoid probability)
    fall_pred   = outputs["fall"].squeeze(1)            # (batch,)
    fall_target = targets["fall"].float()               # (batch,)
    loss_fall   = bce_loss(fall_pred, fall_target)

    # Pre-activity classification loss (all windows)
    loss_pre = ce_loss(outputs["pre_activity"], targets["pre_activity"])

    # Fall-type loss only on windows where a fall actually occurred
    fall_mask = targets["fall"].bool()                  # (batch,) boolean
    if fall_mask.sum() > 0:
        ft_pred   = outputs["fall_type"][fall_mask]     # (n_falls, 3)
        ft_target = targets["fall_type"][fall_mask]     # (n_falls,)
        loss_ft   = ce_loss(ft_pred, ft_target)
    else:
        loss_ft = torch.tensor(0.0, device=fall_pred.device)

    total = loss_fall + W_FALL_TYPE * loss_ft + W_PRE_ACTIVITY * loss_pre
    return total


def augment_training_set(X_train: np.ndarray, labels_train: dict) -> tuple[np.ndarray, dict]:
    """Apply 6-way augmentation to every fall window, producing 7x more fall examples."""
    fall_mask = labels_train["fall"] == 1
    fall_idx  = np.where(fall_mask)[0]

    aug_X: list[np.ndarray] = []
    aug_labels: dict = {k: [] for k in labels_train}

    for idx in fall_idx:
        window = X_train[idx].reshape(200, 6)
        for aug_win in augment_window(window):        # returns 6 (200,6) arrays
            aug_X.append(aug_win.flatten())
            for k in labels_train:
                aug_labels[k].append(labels_train[k][idx])

    aug_X_np      = np.array(aug_X, dtype=np.float32)
    aug_labels_np = {k: np.array(v, dtype=labels_train[k].dtype) for k, v in aug_labels.items()}

    X_aug      = np.vstack([X_train, aug_X_np])
    labels_aug = {k: np.concatenate([labels_train[k], aug_labels_np[k]]) for k in labels_train}

    n_fall = len(fall_idx)
    n_adl  = int((~fall_mask).sum())
    print(f"[AUG] Before: {len(X_train):,}  (fall: {n_fall:,}  ADL: {n_adl:,})")
    print(f"[AUG] After : {len(X_aug):,}  (fall: {n_fall * 7:,}  ADL: {n_adl:,})")

    return X_aug, labels_aug


def train_epoch(model, loader, optimizer, bce, ce, device) -> float:
    """Run one training epoch and return mean loss."""
    model.train()
    total_loss = 0.0

    for windows, targets in loader:
        windows = windows.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        optimizer.zero_grad()
        outputs = model(windows)
        loss    = compute_loss(outputs, targets, bce, ce)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, bce, ce, device) -> tuple[float, float]:
    """Run one validation pass, return (mean_loss, fall_accuracy)."""
    model.eval()
    total_loss  = 0.0
    correct     = 0
    total_items = 0

    for windows, targets in loader:
        windows = windows.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        outputs  = model(windows)
        loss     = compute_loss(outputs, targets, bce, ce)
        total_loss += loss.item()

        # Fall accuracy: threshold at 0.5
        preds   = (outputs["fall"].squeeze(1) >= 0.5).long()
        correct += (preds == targets["fall"]).sum().item()
        total_items += len(windows)

    accuracy = correct / total_items if total_items > 0 else 0.0
    return total_loss / len(loader), accuracy


def train() -> None:
    """Main training function — runs for N_EPOCHS and saves checkpoints."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Using device: {device}")

    # ------------------------------------------------------------------ #
    # Load and split data                                                 #
    # ------------------------------------------------------------------ #
    X, labels = load_data()
    X_train, X_val, X_test, l_train, l_val, l_test, idx_test = split_data(X, labels)

    # ------------------------------------------------------------------ #
    # Augment training set (fall windows only)                            #
    # ------------------------------------------------------------------ #
    print("\n[TRAIN] Augmenting fall windows in training set ...")
    X_train, l_train = augment_training_set(X_train, l_train)

    # ------------------------------------------------------------------ #
    # DataLoaders                                                         #
    # ------------------------------------------------------------------ #
    train_ds = FallWindowDataset(X_train, l_train)
    val_ds   = FallWindowDataset(X_val,   l_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"[TRAIN] Batches/epoch — train: {len(train_loader)}  val: {len(val_loader)}")

    # ------------------------------------------------------------------ #
    # Model, losses, optimiser, scheduler                                 #
    # ------------------------------------------------------------------ #
    model = FallDetectorCNN_LSTM().to(device)
    bce   = nn.BCELoss()
    ce    = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=0.5
    )

    # ------------------------------------------------------------------ #
    # Training loop                                                       #
    # ------------------------------------------------------------------ #
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    log_rows: list[dict] = []

    print(f"\n[TRAIN] Starting training for {N_EPOCHS} epochs...\n")

    for epoch in range(1, N_EPOCHS + 1):
        train_loss          = train_epoch(model, train_loader, optimizer, bce, ce, device)
        val_loss, val_acc   = eval_epoch(model, val_loader, bce, ce, device)

        scheduler.step(val_loss)

        # Console output
        print(
            f"  Epoch {epoch:>2}/{N_EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_fall_acc={val_acc:.4f}"
        )

        # Track metrics for CSV log
        log_rows.append({
            "epoch":        epoch,
            "train_loss":   round(train_loss, 6),
            "val_loss":     round(val_loss,   6),
            "val_accuracy": round(val_acc,    6),
        })

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = MODELS_DIR / f"checkpoint_epoch_{epoch}.pt"
            torch.save(model.state_dict(), str(ckpt_path))
            print(f"    [CKPT] Saved checkpoint to {ckpt_path.name}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(BEST_MODEL))
            print(f"    [BEST] New best val_loss={val_loss:.4f} — saved best_model.pt")

    # ------------------------------------------------------------------ #
    # Write training log                                                  #
    # ------------------------------------------------------------------ #
    with open(str(TRAINING_LOG), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\n[TRAIN] Training log saved to {TRAINING_LOG}")
    print(f"[TRAIN] Best val loss: {best_val_loss:.4f}")
    print("[TRAIN] Done.")

    # Save the test split for evaluate.py — features + encoded labels
    test_csv_path = DATASET_CSV.parent / "dataset_test.csv"
    test_df = pd.DataFrame(X_test, columns=[f"f_{i}" for i in range(1200)])
    test_df["fall_label"]       = l_test["fall"]
    test_df["fall_type_enc"]    = l_test["fall_type"]
    test_df["pre_activity_enc"] = l_test["pre_activity"]
    test_df.to_csv(str(test_csv_path), index=False)
    print(f"[TRAIN] Test split ({len(test_df):,} rows) saved to {test_csv_path}")


# ---------------------------------------------------------------------------
# Standalone entry point — run:  python src/train.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train()
