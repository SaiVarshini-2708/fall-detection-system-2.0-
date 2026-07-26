import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import train

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
N_EPOCHS = 30
LR = 1e-3
SEED = 42
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.3


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class FallDetectorCNN_Transformer(nn.Module):
    """Same CNN feature extractor as baseline LSTM, but with a small Transformer encoder head."""

    def __init__(self) -> None:
        super().__init__()
        self.cnn_block1 = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.cnn_block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.positional_encoding = PositionalEncoding(d_model=D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            dim_feedforward=256,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.head_fall = nn.Sequential(nn.Linear(D_MODEL, 1), nn.Sigmoid())
        self.head_fall_type = nn.Linear(D_MODEL, 3)
        self.head_pre_activity = nn.Linear(D_MODEL, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.permute(0, 2, 1)
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return {
            "fall": self.head_fall(x),
            "fall_type": self.head_fall_type(x),
            "pre_activity": self.head_pre_activity(x),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _set_seed() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def train_and_evaluate(epochs: int = N_EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LR, save_metrics: bool = True) -> dict:
    _set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRANSFORMER] Using device: {device}")

    X, labels = train.load_data()
    X_train, X_val, X_test, labels_train, labels_val, labels_test, _ = train.split_data(X, labels)
    X_train, labels_train = train.augment_training_set(X_train, labels_train)

    train_ds = train.FallWindowDataset(X_train, labels_train)
    val_ds = train.FallWindowDataset(X_val, labels_val)
    test_ds = train.FallWindowDataset(X_test, labels_test)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = FallDetectorCNN_Transformer().to(device)
    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    start_time = time.perf_counter()
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train.train_epoch(model, train_loader, optimizer, bce, ce, device)
        val_loss, _ = train.eval_epoch(model, val_loader, bce, ce, device)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        print(f"[TRANSFORMER] epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    training_time = time.perf_counter() - start_time

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for windows, targets in test_loader:
            windows = windows.to(device)
            outputs = model(windows)
            preds = (outputs["fall"].squeeze(1) >= 0.5).cpu().long()
            targets = targets["fall"].cpu().long()
            all_preds.append(preds)
            all_targets.append(targets)

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "model": "cnn_transformer",
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "parameter_count": int(count_parameters(model)),
        "training_time_seconds": round(float(training_time), 6),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "seed": int(SEED),
    }

    if save_metrics:
        output_path = RESULTS_DIR / "transformer_metrics.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"[TRANSFORMER] Metrics saved to {output_path}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate the CNN-Transformer ablation model")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()
    train_and_evaluate(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, save_metrics=True)
