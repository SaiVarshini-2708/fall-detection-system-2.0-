import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "SisFall_dataset"
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "severity_proxy_dataset.csv"

FALL_TYPE_MAP = {
    "F01": "slip",
    "F02": "slip",
    "F03": "slip",
    "F04": "slip",
    "F05": "slip",
    "F06": "trip",
    "F07": "trip",
    "F08": "trip",
    "F09": "trip",
    "F10": "trip",
    "F11": "faint",
    "F12": "faint",
    "F13": "faint",
    "F14": "faint",
    "F15": "faint",
}


def load_clip(filepath: Path) -> np.ndarray | None:
    rows = []
    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip().rstrip(";")
                if not line:
                    continue
                values = [float(x) for x in line.split(",")]
                if len(values) != 9:
                    return None
                rows.append([values[0], values[1], values[2], values[6], values[7], values[8]])
    except Exception:
        return None

    if not rows:
        return None

    data = np.array(rows, dtype=np.float32)
    return data


def compute_peak_and_delta(data: np.ndarray) -> tuple[float, float, float]:
    if data is None or data.ndim != 2 or data.shape[0] < 2:
        return float("nan"), float("nan"), float("nan")

    acc_xyz = data[:, :3]
    magnitude = np.sqrt(np.sum(acc_xyz ** 2, axis=1))
    peak_acc_g = float(np.max(magnitude))

    peak_idx = int(np.argmax(magnitude))
    sample_rate_hz = 100.0
    window_half_samples = int(round(0.15 * sample_rate_hz))
    start_idx = max(0, peak_idx - window_half_samples)
    end_idx = min(len(magnitude), peak_idx + window_half_samples + 1)
    local_window = magnitude[start_idx:end_idx]

    # Numerical integration over a small window around the peak.
    delta_v = float(np.trapezoid(local_window, dx=1.0 / sample_rate_hz))

    impact_duration_s = float(np.sum(magnitude > 2.0) / sample_rate_hz)
    return peak_acc_g, delta_v, impact_duration_s


def build_dataset() -> pd.DataFrame:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"SisFall dataset not found at {DATASET_ROOT}")

    clip_paths = sorted(DATASET_ROOT.rglob("*.txt"))
    records = []

    for path in clip_paths:
        if not path.name.startswith("F"):
            continue

        activity_code = path.stem.split("_")[0]
        if activity_code not in FALL_TYPE_MAP:
            continue

        data = load_clip(path)
        if data is None:
            continue

        subject_id = path.stem.split("_")[1] if len(path.stem.split("_")) > 1 else "UNKNOWN"
        peak_acc_g, delta_v, impact_duration_s = compute_peak_and_delta(data)

        records.append({
            "subject_id": subject_id,
            "activity_code": activity_code,
            "fall_type": FALL_TYPE_MAP[activity_code],
            "peak_acc_g": peak_acc_g,
            "delta_v": delta_v,
            "impact_duration_s": impact_duration_s,
        })

    if not records:
        raise RuntimeError("No fall clips were processed")

    df = pd.DataFrame(records)
    df = df.sort_values("peak_acc_g").reset_index(drop=True)

    n = len(df)
    if n == 0:
        raise RuntimeError("No rows available for severity proxy assignment")

    low_cut = n // 3
    mid_cut = 2 * n // 3

    def _severity(value: float, idx: int) -> str:
        if idx < low_cut:
            return "low"
        if idx < mid_cut:
            return "medium"
        return "high"

    df["severity_proxy"] = [
        _severity(value, idx) for idx, value in enumerate(df["peak_acc_g"].tolist())
    ]
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("=== severity proxy extraction summary ===")
    print(f"Rows processed: {len(df)}")
    print("\nSeverity proxy value counts:")
    print(df["severity_proxy"].value_counts().sort_index().to_string())

    print("\nFall type vs severity proxy:")
    print(pd.crosstab(df["fall_type"], df["severity_proxy"]).to_string())

    print("\npeak_acc_g stats:")
    stats = df["peak_acc_g"].describe()
    print(stats.to_string())

    peak_range = (float(df["peak_acc_g"].min()), float(df["peak_acc_g"].max()))
    print(f"\nSanity check (published fall studies often report ~3-6g):")
    if peak_range[0] >= 3.0 and peak_range[1] <= 6.0:
        print("PASS: computed peak acceleration values fall within the expected rough range.")
    else:
        print("FLAG: computed peak acceleration values are outside the expected rough range; this suggests a units/windowing issue.")
        print(f"Observed peak_acc_g range: {peak_range[0]:.3f} to {peak_range[1]:.3f} g")


def main() -> None:
    df = build_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print_summary(df)
    print(f"\nCSV written to: {OUTPUT_PATH}")
    print(f"CSV rows written: {len(df)}")


if __name__ == "__main__":
    main()
