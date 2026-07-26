import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


ALGO_FILES = {
    "lstm": "lstm_metrics.json",
    "gru": "gru_metrics.json",
    "transformer": "transformer_metrics.json",
}


def load_or_run_metrics(algo: str, force: bool) -> Path:
    output_path = RESULTS_DIR / ALGO_FILES[algo]
    if output_path.exists() and not force:
        print(f"[COMPARE] Using existing metrics for {algo}: {output_path}")
        return output_path

    print(f"[COMPARE] Metrics missing or --force used for {algo}; running training...")
    run_script = ROOT / "run_algo.py"
    python_exe = get_python_executable()
    cmd = [python_exe, str(run_script), "--algo", algo]
    if force:
        cmd.append("--force")
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return output_path


def build_comparison_table(force: bool) -> list[dict]:
    rows = []
    for algo in ["lstm", "gru", "transformer"]:
        metrics_path = load_or_run_metrics(algo, force)
        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append({
            "model": algo,
            "accuracy": metrics.get("accuracy", ""),
            "precision": metrics.get("precision", ""),
            "recall": metrics.get("recall", ""),
            "f1": metrics.get("f1", ""),
            "parameter_count": metrics.get("parameter_count", ""),
            "training_time_seconds": metrics.get("training_time_seconds", ""),
            "tn": metrics.get("confusion_matrix", {}).get("tn", ""),
            "fp": metrics.get("confusion_matrix", {}).get("fp", ""),
            "fn": metrics.get("confusion_matrix", {}).get("fn", ""),
            "tp": metrics.get("confusion_matrix", {}).get("tp", ""),
        })
    return rows


def save_csv(rows: list[dict]) -> Path:
    output_path = RESULTS_DIR / "comparison_table.csv"
    fieldnames = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "parameter_count",
        "training_time_seconds",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def print_table(rows: list[dict]) -> None:
    print("\n=== Model comparison ===")
    for row in rows:
        print(
            f"{row['model']:<12}  acc={row['accuracy']:.6f}  prec={row['precision']:.6f}  rec={row['recall']:.6f}  "
            f"f1={row['f1']:.6f}  params={row['parameter_count']}  train_time={row['training_time_seconds']:.3f}s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare LSTM, GRU, and Transformer ablation results")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = build_comparison_table(force=args.force)
    print_table(rows)
    csv_path = save_csv(rows)
    print(f"\nSaved comparison table to {csv_path}")
