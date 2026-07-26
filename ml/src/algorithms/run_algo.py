import argparse
import json
import subprocess
import sys
from pathlib import Path

ALGOS = ["lstm", "gru", "transformer"]
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def get_python_executable() -> str:
    ml_root = Path(__file__).resolve().parents[2]
    candidates = [
        ml_root / ".venv" / "Scripts" / "python.exe",
        ml_root / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def run_algo(algo: str, force: bool = False, epochs: int = 30, batch_size: int = 32, lr: float = 1e-3) -> Path:
    if algo not in ALGOS:
        raise ValueError(f"Unsupported algo: {algo}")

    metrics_name = {
        "lstm": "lstm_metrics.json",
        "gru": "gru_metrics.json",
        "transformer": "transformer_metrics.json",
    }[algo]
    output_path = RESULTS_DIR / metrics_name

    if output_path.exists() and not force:
        print(f"[RUN] Found existing metrics at {output_path}; skipping training. Use --force to retrain.")
        return output_path

    if algo == "lstm":
        src_root = Path(__file__).resolve().parents[1]
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))

        import train

        class LSTMLoop:
            def __init__(self) -> None:
                self.metrics_path = output_path

            def run(self) -> Path:
                import json
                import time
                import numpy as np
                import torch
                import torch.nn as nn
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                from model import FallDetectorCNN_LSTM

                np.random.seed(42)
                torch.manual_seed(42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(42)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                X, labels = train.load_data()
                X_train, X_val, X_test, labels_train, labels_val, labels_test, _ = train.split_data(X, labels)
                X_train, labels_train = train.augment_training_set(X_train, labels_train)

                train_ds = train.FallWindowDataset(X_train, labels_train)
                val_ds = train.FallWindowDataset(X_val, labels_val)
                test_ds = train.FallWindowDataset(X_test, labels_test)

                train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
                val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
                test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

                model = FallDetectorCNN_LSTM().to(device)
                bce = nn.BCELoss()
                ce = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

                start_time = time.perf_counter()
                for _ in range(epochs):
                    train_loss = train.train_epoch(model, train_loader, optimizer, bce, ce, device)
                    val_loss, _ = train.eval_epoch(model, val_loader, bce, ce, device)
                    scheduler.step(val_loss)
                training_time = time.perf_counter() - start_time

                model.eval()
                preds, targets = [], []
                with torch.no_grad():
                    for windows, batch_targets in test_loader:
                        windows = windows.to(device)
                        outputs = model(windows)
                        preds.append((outputs["fall"].squeeze(1) >= 0.5).cpu().long())
                        targets.append(batch_targets["fall"].cpu().long())
                y_true = torch.cat(targets).numpy()
                y_pred = torch.cat(preds).numpy()

                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                metrics = {
                    "model": "cnn_lstm",
                    "accuracy": round(float(accuracy), 6),
                    "precision": round(float(precision), 6),
                    "recall": round(float(recall), 6),
                    "f1": round(float(f1), 6),
                    "parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                    "training_time_seconds": round(float(training_time), 6),
                    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
                    "epochs": int(epochs),
                    "batch_size": int(batch_size),
                    "seed": 42,
                }
                with open(self.metrics_path, "w", encoding="utf-8") as handle:
                    json.dump(metrics, handle, indent=2)
                return self.metrics_path

        return LSTMLoop().run()

    script_name = {
        "gru": "cnn_gru_model.py",
        "transformer": "cnn_transformer_model.py",
    }[algo]
    script_path = ROOT / script_name
    python_exe = get_python_executable()
    cmd = [python_exe, str(script_path), "--epochs", str(epochs), "--batch-size", str(batch_size), "--lr", str(lr)]
    print(f"[RUN] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single ablation model")
    parser.add_argument("--algo", choices=ALGOS, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    run_algo(args.algo, force=args.force, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
