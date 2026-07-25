# infer.py
# Real-time inference using ONNX Runtime + HTTP alert dispatch to the backend.
# Simulation mode: replays fall windows from dataset.csv to test the full pipeline.

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import os
import onnxruntime as ort

# ------------------------------------------------------------------ #
# Load environment variables from ml/.env                            #
# ------------------------------------------------------------------ #
# Walk up from src/ to find .env in ml/
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=str(_ENV_PATH))

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3001")

# ------------------------------------------------------------------ #
# Label decode tables — index matches model output head ordering      #
# ------------------------------------------------------------------ #
FALL_TYPE_IDX: dict[int, str] = {
    0: "slip",
    1: "trip",
    2: "faint",
}

PRE_ACTIVITY_IDX: dict[int, str] = {
    0: "walking",
    1: "standing",
    2: "bending",
    3: "sitting",
}

# ------------------------------------------------------------------ #
# ONNX model path                                                     #
# ------------------------------------------------------------------ #
ONNX_PATH      = Path(__file__).parent.parent / "models" / "model.onnx"
DATASET_CSV    = Path(__file__).parent.parent / "data"   / "dataset.csv"
THRESHOLD_PATH = Path(__file__).parent.parent / "models" / "threshold.txt"


def _load_threshold() -> float:
    """Load optimal decision threshold saved by evaluate.py; default 0.5."""
    if THRESHOLD_PATH.exists():
        try:
            val = float(THRESHOLD_PATH.read_text().strip())
            print(f"[INFER] Loaded decision threshold: {val:.2f}")
            return val
        except ValueError:
            pass
    print("[INFER] threshold.txt not found — using default 0.5")
    return 0.5


FALL_THRESHOLD = _load_threshold()

# Stillness thresholds (same as labeller.py)
_STILLNESS_UNCONSCIOUS = 0.85
_STILLNESS_STUNNED     = 0.55

# Seconds to sleep between simulated alerts (avoids flooding the backend)
ALERT_SLEEP_SECONDS = 2

# ─── Adaptive Confirmation Window — tunable constants ──────────────────────
#
# WHY three tiers instead of one fixed delay: high-confidence detections are
# statistically less likely to be false positives, so a short re-check suffices.
# Low-confidence detections are ambiguous and benefit from a longer window to
# let transient events (a stumble that self-corrects, a vigorous ADL) naturally
# resolve before we commit to an alert.
#
# Adjust these constants to tune sensitivity vs. alert latency trade-off:
#   - Raising a window duration → fewer false positives, but slower true alerts.
#   - Lowering CONFIRM_TIER_* thresholds → more events use the longer windows.

CONFIRM_TIER_HIGH  = 0.90   # confidence >= this → use the short 1-second window
CONFIRM_TIER_MID   = 0.75   # confidence >= this → use the medium 2.5-second window
                             # confidence  < 0.75 → use the long  4-second window

CONFIRM_WINDOW_HIGH_SECS = 1.0   # seconds — high confidence, confirm fast
CONFIRM_WINDOW_MID_SECS  = 2.5   # seconds — medium confidence
CONFIRM_WINDOW_LOW_SECS  = 4.0   # seconds — low confidence, take more time

# Secondary gate: minimum "stillness" score in the last 0.5 s of the window.
# Stillness = 1 / (1 + variance), same formula as compute_post_state().
# A score below this means vigorous movement continued through the window tail —
# consistent with a transient ADL spike rather than a sustained fall event.
# Tune upward to be stricter (reject more marginal cases); downward to be
# more permissive (let more borderline events through to the post-state check).
CONFIRM_TAIL_STILLNESS_MIN = 0.15

# Where suppressed false-positive events are written for later analysis.
# This is a plain-text append log: one line per suppressed event.
SUPPRESSED_LOG_PATH = Path(__file__).parent.parent / "logs" / "suppressed_alerts.log"

# ─── BLE Localization — tunable constants ──────────────────────────────────────
#
# ROOMS is the ordered patrol sequence. Add, remove, or reorder rooms here —
# the patrol logic (threshold comparison, strongest-signal selection) adapts
# automatically; no other code needs to change.
ROOMS: list[str] = ["Bedroom", "Living Room", "Kitchen", "Bathroom", "Hallway"]

# BLE MAC address of the specific wearable device paired with this system.
# In production this would be read from device config or set during onboarding.
WEARABLE_BLE_ID = "BLE:AA:BB:CC:DD:EE:FF"

# RSSI cutoff (dBm) for confirming same-room proximity.
# Readings at or above this value indicate the wearable is in the scanned room;
# readings below indicate the signal is attenuated by one or more walls.
RSSI_PROXIMITY_THRESHOLD = -65  # dBm

# Simulated RSSI ranges grounded in real-world BLE indoor localization data.
# Source: UCI/Kaggle "BLE RSSI Dataset for Indoor Localization" and related
# literature — same-room readings cluster at -40 to -65 dBm; through-wall
# readings fall to -65 to -100 dBm due to building-material path loss.
RSSI_SAME_ROOM_MIN  = -65    # dBm — weakest plausible same-room reading
RSSI_SAME_ROOM_MAX  = -40    # dBm — strongest plausible same-room reading
RSSI_OTHER_ROOM_MIN = -100   # dBm — most attenuated (many walls) other-room reading
RSSI_OTHER_ROOM_MAX = -66    # dBm — strongest other-room reading (one thin wall)

# Time the robot spends moving to and scanning each room.
# Keep short (0.5 s) for unit tests; raise to 3–5 s for a realistic live demo.
PATROL_DELAY_SECONDS = 0.5   # seconds per room

# Log file recording the full patrol sequence for each confirmed fall event.
PATROL_LOG_PATH = Path(__file__).parent.parent / "logs" / "patrol_log.txt"


def _load_onnx_session(onnx_path: str = None) -> ort.InferenceSession:
    """
    Load the ONNX Runtime inference session.

    Parameters
    ----------
    onnx_path : str, optional
        Path to model.onnx. Defaults to ml/models/model.onnx.

    Returns
    -------
    ort.InferenceSession
    """
    path = Path(onnx_path) if onnx_path else ONNX_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"ONNX model not found at {path}\n"
            "Run:  python src/export_onnx.py  first."
        )

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print(f"[INFER] Loaded ONNX model from {path.name}")
    return session


# Module-level session — loaded once, reused for every window
_session: ort.InferenceSession | None = None


def _get_session() -> ort.InferenceSession:
    """Lazy-load and cache the ONNX session at module level."""
    global _session
    if _session is None:
        _session = _load_onnx_session()
    return _session


def run_inference(window: np.ndarray) -> dict | None:
    """
    Run fall detection inference on a single 2-second IMU window.

    Parameters
    ----------
    window : np.ndarray
        Shape (200, 6) float32 — already z-score normalised.

    Returns
    -------
    dict or None
        None if fall probability < 0.5 (no fall detected).
        Otherwise: {'fall_type': str, 'pre_activity': str, 'confidence': float}
    """
    session = _get_session()

    # Reshape to (1, 200, 6) — ONNX input expects a batch dimension
    x = window.reshape(1, 200, 6).astype(np.float32)

    # Run all three output heads in one call
    ort_outputs = session.run(None, {"imu_window": x})
    fall_raw, fall_type_logits, pre_activity_logits = ort_outputs

    # fall_prob is already a sigmoid probability (from the fall head)
    fall_prob = float(fall_raw[0, 0])

    # No fall detected — skip silently
    if fall_prob < FALL_THRESHOLD:
        return None

    # Determine fall type from argmax of logits
    # Index 0 = 'none', so real types start at index 1
    ft_idx    = int(np.argmax(fall_type_logits[0]))
    fall_type = FALL_TYPE_IDX.get(ft_idx, "slip")

    # Determine pre-activity from argmax
    pre_idx      = int(np.argmax(pre_activity_logits[0]))
    pre_activity = PRE_ACTIVITY_IDX.get(pre_idx, "walking")

    confidence = round(fall_prob, 4)

    return {
        "fall_type":    fall_type,
        "pre_activity": pre_activity,
        "confidence":   confidence,
    }


def compute_post_state(signal_after_fall: np.ndarray) -> str:
    """
    Estimate the physical state of the person using the signal immediately after a fall.

    Uses the variance of the resultant acceleration magnitude to measure
    how much movement is still happening after the impact.

    Parameters
    ----------
    signal_after_fall : np.ndarray
        Shape (300, 6) float32 — 3 seconds at 100Hz immediately after the fall event.
        Columns 0-2 are acc_x, acc_y, acc_z.

    Returns
    -------
    str
        'unconscious' | 'stunned' | 'moving'
    """
    # Resultant accelerometer magnitude at each timestep
    acc            = signal_after_fall[:, :3]   # (300, 3)
    acc_magnitude  = np.sqrt(np.sum(acc ** 2, axis=1))  # (300,)

    # Variance of the full 3-second window
    variance  = float(np.var(acc_magnitude))
    stillness = 1.0 / (1.0 + variance)

    if stillness > _STILLNESS_UNCONSCIOUS:
        return "unconscious"
    elif stillness > _STILLNESS_STUNNED:
        return "stunned"
    else:
        return "moving"


def simulate_ble_scan(room: str, wearable_id: str, true_room: str) -> float:
    """
    Return a simulated BLE RSSI value (dBm) for scanning wearable_id in room.

    In production, replace this function body with a real hardware BLE scan
    (e.g. bleak.BleakScanner or a UART read from the robot's BLE radio).
    The surrounding logic — threshold comparison, strongest-signal selection,
    room registry — stays identical; only this one call changes.

    RSSI ranges are grounded in BLE indoor localization literature:
    same-room: -40 to -65 dBm; through-wall: -65 to -100 dBm.
    (Source: UCI/Kaggle "BLE RSSI Dataset for Indoor Localization")
    """
    if room == true_room:
        return random.uniform(RSSI_SAME_ROOM_MIN, RSSI_SAME_ROOM_MAX)
    return random.uniform(RSSI_OTHER_ROOM_MIN, RSSI_OTHER_ROOM_MAX)


def simulate_robot_patrol(wearable_id: str, true_room: str | None = None) -> str:
    """
    Simulate a robot patrol that scans for wearable_id's BLE signal across ROOMS
    and returns the room where the signal is strongest above RSSI_PROXIMITY_THRESHOLD.

    The robot does NOT detect or classify humans — it has no general presence-
    sensing capability. Its only task is locating the room where THIS specific
    wearable's BLE MAC address produces the strongest RSSI. Human detection and
    fall confirmation already happened upstream in the IMU pipeline and adaptive
    confirmation window.

    Parameters
    ----------
    wearable_id : str
        BLE MAC address of the wearable to locate.
    true_room : str or None
        Ground-truth room for the simulation. Chosen randomly from ROOMS when
        None, representing an unknown real-world scenario.

    Returns
    -------
    str
        Room name from ROOMS, or "location_unknown" if no room cleared
        RSSI_PROXIMITY_THRESHOLD after a full patrol cycle.
    """
    if true_room is None:
        true_room = random.choice(ROOMS)

    PATROL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    log_lines = [
        f"\n[{ts}] PATROL START",
        f"  wearable  : {wearable_id}",
        f"  true_room : {true_room}",
        f"  order     : {' -> '.join(ROOMS)}",
    ]

    best_room: str | None = None
    best_rssi = float("-inf")

    for room in ROOMS:
        # Simulated transit + scan delay per room.
        # In real deployment this represents the robot driving to the room entrance
        # and dwelling long enough for a reliable RSSI average.
        time.sleep(PATROL_DELAY_SECONDS)

        rssi = simulate_ble_scan(room, wearable_id, true_room)
        clears = rssi >= RSSI_PROXIMITY_THRESHOLD
        tag = "ABOVE THRESHOLD ✓" if clears else "below threshold"
        log_lines.append(f"  [{room:<12}] RSSI = {rssi:6.1f} dBm  ({tag})")

        if clears and rssi > best_rssi:
            best_rssi = rssi
            best_room = room

    if best_room is not None:
        result_str = best_room
        log_lines.append(
            f"  RESULT: located in '{best_room}' (best RSSI = {best_rssi:.1f} dBm)"
        )
    else:
        result_str = "location_unknown"
        log_lines.append(
            f"  RESULT: location_unknown — no room cleared threshold "
            f"({RSSI_PROXIMITY_THRESHOLD} dBm)"
        )

    log_lines.append(f"[{ts}] PATROL END\n")

    with PATROL_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n")

    print(f"  [PATROL] Wearable located in: {result_str}")
    return result_str


def _derive_severity_proxy(window: np.ndarray | None) -> str | None:
    """Derive a lightweight impact-based proxy from a 2-second IMU window."""
    if window is None:
        return None

    if window.ndim != 2 or window.shape[0] < 2 or window.shape[1] < 3:
        return None

    acc_xyz = window[:, :3]
    magnitude = np.sqrt(np.sum(acc_xyz ** 2, axis=1))

    peak_acc_g = float(np.max(magnitude))
    if not np.isfinite(peak_acc_g):
        return None

    if peak_acc_g >= 6.0:
        return "high"
    if peak_acc_g >= 3.5:
        return "medium"
    return "low"


def send_alert(
    result: dict,
    post_state: str,
    confirmation_window_ms: int | None = None,
    location: str | None = None,
    severity_proxy: str | None = None,
    window: np.ndarray | None = None,
) -> None:
    """
    POST a fall alert to the backend API.

    The backend (Team B) expects:
      { fall_type, pre_activity, post_state, confidence }
    plus optional fields:
      { confirmation_window_ms }  — only present for confirmed falls that passed
                                    the adaptive confirmation window.
      { location }                — room name from BLE patrol, or "location_unknown".
      { severity_proxy }          — optional impact-based proxy severity label.

    Handles connection errors gracefully — inference continues even
    when the backend is unreachable (e.g. during standalone testing).

    Parameters
    ----------
    result : dict
        Output from run_inference(): keys fall_type, pre_activity, confidence.
    post_state : str
        State estimate from compute_post_state().
    confirmation_window_ms : int or None
        Actual wall-clock duration of the confirmation window in milliseconds.
        Omitted from the payload when None (e.g. if called without the feature).
    location : str or None
        Room name returned by simulate_robot_patrol(), or "location_unknown".
        Omitted when None (e.g. in standalone tests that skip the patrol).
    severity_proxy : str or None
        Optional impact-based proxy severity label such as "low", "medium", or "high".
        Omitted from the payload when None.
    """
    payload = {
        "fall_type":    result["fall_type"],
        "pre_activity": result["pre_activity"],
        "post_state":   post_state,
        "confidence":   result["confidence"],
    }
    if confirmation_window_ms is not None:
        payload["confirmation_window_ms"] = confirmation_window_ms
    if location is not None:
        payload["location"] = location

    if severity_proxy is None:
        severity_proxy = _derive_severity_proxy(window)
    if severity_proxy is not None:
        payload["severity_proxy"] = severity_proxy

    url = f"{BACKEND_URL}/api/alert"

    try:
        response = requests.post(url, json=payload, timeout=5)

        if response.status_code in (200, 201):
            win_str = f" | win={confirmation_window_ms}ms" if confirmation_window_ms is not None else ""
            print(
                f"  [SENT]  fall_type={result['fall_type']:<8} | "
                f"pre={result['pre_activity']:<10} | "
                f"post={post_state:<12} | "
                f"conf={result['confidence']:.4f}"
                f"{win_str}"
            )
        else:
            print(
                f"  [ERROR] HTTP {response.status_code} from {url}: "
                f"{response.text[:200]}"
            )

    except requests.exceptions.ConnectionError:
        # Backend is offline — print and continue, do not crash
        print(
            f"  [ERROR] Backend unreachable at {url} "
            f"(fall_type={result['fall_type']}, conf={result['confidence']:.4f})"
        )
    except requests.exceptions.Timeout:
        print(f"  [ERROR] Request to {url} timed out after 5s")
    except Exception as exc:
        print(f"  [ERROR] Unexpected error sending alert: {exc}")


def _log_suppressed(confidence: float, reason: str) -> None:
    """
    Append a suppressed (rejected) fall event to SUPPRESSED_LOG_PATH.

    Each line is a plain-text record with an ISO timestamp, the original
    confidence score, and a short reason code so false-positive rates can
    be reviewed later without any extra tooling (just open the file).

    Never raises — creates the logs/ directory if it doesn't exist yet.
    """
    SUPPRESSED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = (
        f"{time.strftime('%Y-%m-%dT%H:%M:%S')} "
        f"SUPPRESSED confidence={confidence:.4f} reason={reason}\n"
    )
    with SUPPRESSED_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(entry)
    print(f"  [SUPPRESSED] conf={confidence:.4f} | reason={reason}")


def _adaptive_confirm(window: np.ndarray, confidence: float) -> tuple[bool, int]:
    """
    Adaptive confirmation window: waits a confidence-tiered duration, then
    re-examines the IMU window to decide whether the initial detection is a
    real fall or a false positive.

    Parameters
    ----------
    window : np.ndarray
        Shape (200, 6) float32 — the same window that triggered initial detection.
    confidence : float
        Fall probability from run_inference(), used to select the wait duration.

    Returns
    -------
    (confirmed: bool, elapsed_ms: int)
        confirmed   — True if the fall is verified and should proceed to the
                      post-state check + alert. False if it should be suppressed.
        elapsed_ms  — actual wall-clock duration of the confirmation window in ms,
                      forwarded to the dashboard as confirmation_window_ms.

    SIMULATION NOTE:
        In real-time deployment, fresh IMU samples would arrive during the sleep
        interval and be used for re-checking. In simulation mode we only have the
        original window, so the check uses two heuristics on that data:
          1. Primary gate  — re-run the ONNX model; if it no longer fires, reject.
          2. Secondary gate — check the last 0.5 s of the window for sustained
             stillness (the "settling" period after a real fall). A tail that is
             still highly active indicates the person resumed movement immediately,
             which is inconsistent with a genuine fall.
    """
    # ── Select window duration from confidence tier ──────────────────────────
    # WHY variable duration: high-confidence detections need less re-observation
    # time; ambiguous low-confidence ones need more to distinguish a real fall
    # from a transient ADL spike that happens to clear the detection threshold.
    if confidence >= CONFIRM_TIER_HIGH:
        window_secs = CONFIRM_WINDOW_HIGH_SECS
    elif confidence >= CONFIRM_TIER_MID:
        window_secs = CONFIRM_WINDOW_MID_SECS
    else:
        window_secs = CONFIRM_WINDOW_LOW_SECS

    # ── Wait for the confirmation interval to elapse ─────────────────────────
    t_start = time.monotonic()
    time.sleep(window_secs)
    elapsed_ms = round((time.monotonic() - t_start) * 1000)

    # ── Primary gate: re-run the model on the available signal ───────────────
    # In production this would use a freshly collected 2-second window captured
    # during the sleep interval above. In simulation it's the same window — if
    # the model no longer fires (e.g. after a threshold change or due to
    # borderline probability) we treat this as a false positive.
    re_result = run_inference(window)
    if re_result is None:
        return False, elapsed_ms

    # ── Secondary gate: tail-variance check ──────────────────────────────────
    # A genuine fall leaves the person on the ground for at least 0.5 s,
    # producing low variance in the window tail (the "settling" period).
    # A transient ADL spike that happened to fire the model returns to vigorous
    # movement immediately, keeping the tail variance high.
    # We use the last 50 samples (0.5 s at 100 Hz) as the tail.
    acc_mag = np.sqrt(np.sum(window[:, :3] ** 2, axis=1))   # resultant, shape (200,)
    tail_stillness = 1.0 / (1.0 + float(np.var(acc_mag[150:])))  # same formula as compute_post_state

    if tail_stillness < CONFIRM_TAIL_STILLNESS_MIN:
        # Tail is still too active — motion resumed immediately → likely a false positive.
        return False, elapsed_ms

    return True, elapsed_ms


def run_simulation(dataset_csv: str = None) -> None:
    """
    Simulation mode: replay all fall windows from dataset.csv through the full pipeline.

    For each fall window:
      1. Reconstruct the (200,6) array from the flat f_0…f_1199 columns.
      2. Run inference — skip if no fall detected.
      3. Estimate post-fall state using the same window (approximate).
      4. Send the alert to the backend.
      5. Sleep 2 seconds to simulate real-time cadence.

    Parameters
    ----------
    dataset_csv : str, optional
        Path to the flat feature CSV. Defaults to ml/data/dataset.csv.
    """
    csv_path = Path(dataset_csv) if dataset_csv else DATASET_CSV

    if not csv_path.exists():
        print(f"[INFER] Dataset CSV not found: {csv_path}")
        print("        Run python src/dataset_builder.py first.")
        return

    print(f"\n[INFER] Loading dataset from {csv_path.name} ...")
    df = pd.read_csv(str(csv_path))

    # Keep only fall windows — ADL windows are not used in simulation
    fall_df = df[df["fall_label"] == 1].reset_index(drop=True)
    print(f"[INFER] Fall windows available: {len(fall_df):,}")
    print(f"[INFER] Backend URL: {BACKEND_URL}")
    print(f"[INFER] Starting simulation — {ALERT_SLEEP_SECONDS}s pause between alerts\n")

    alerts_sent       = 0
    alerts_skipped    = 0
    alerts_suppressed = 0

    for row_idx in range(len(fall_df)):
        row = fall_df.iloc[row_idx]

        # Reconstruct (200, 6) window from the 1200 flat feature values
        flat   = row[[f"f_{i}" for i in range(1200)]].values.astype(np.float32)
        window = flat.reshape(200, 6)

        # Run fall detection
        result = run_inference(window)

        if result is None:
            # Model didn't fire on this window (fall_prob < FALL_THRESHOLD)
            alerts_skipped += 1
            continue

        # ── Adaptive confirmation window ──────────────────────────────────
        # Waits a confidence-tiered duration, then re-examines the signal
        # to confirm this is a genuine fall before proceeding to post-state
        # analysis and alerting. Suppresses false positives that resolve on
        # their own (transient ADL spike, person immediately stood back up).
        # Does NOT modify compute_post_state() or send_alert() — only gates
        # whether we call them at all.
        confirmed, conf_window_ms = _adaptive_confirm(window, result["confidence"])

        if not confirmed:
            _log_suppressed(
                confidence=result["confidence"],
                reason="signal_not_sustained",
            )
            alerts_suppressed += 1
            continue

        # ── Post-fall state and alert (only reached for confirmed falls) ──
        # In production this would be 300 samples AFTER the fall event;
        # here we use the available window as an approximation.
        post_signal = np.tile(window, (2, 1))[:300]   # pad to 300 samples
        post_state  = compute_post_state(post_signal)

        # ── BLE localization: patrol rooms to find where the wearable is ──
        # Called AFTER confirmation — does not alter the confirmation logic.
        # true_room=None lets the simulation pick a random room each run,
        # representing a real unknown-location scenario.
        location = simulate_robot_patrol(WEARABLE_BLE_ID)

        send_alert(
            result,
            post_state,
            confirmation_window_ms=conf_window_ms,
            location=location,
            window=window,
        )
        alerts_sent += 1

        # Simulate real-time gap between fall events
        time.sleep(ALERT_SLEEP_SECONDS)

    print(f"\n[INFER] Simulation complete.")
    print(f"  Total fall windows : {len(fall_df):,}")
    print(f"  Alerts sent        : {alerts_sent:,}")
    print(f"  Suppressed (conf window) : {alerts_suppressed:,}")
    print(f"  Skipped (conf<{FALL_THRESHOLD:.2f})      : {alerts_skipped:,}")
    if alerts_suppressed > 0:
        print(f"  Suppression log    : {SUPPRESSED_LOG_PATH}")


# ---------------------------------------------------------------------------
# Standalone entry point — run:  python src/infer.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== infer.py — fall detection simulation ===")
    print(f"Backend URL : {BACKEND_URL}")
    run_simulation()
