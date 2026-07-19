# labeller.py
# Converts SisFall activity codes into structured labels used for training.
# Provides fall type, pre-activity context, and post-fall state estimation.

import numpy as np

# ---------------------------------------------------------------------------
# Activity code → fall type mapping (F01–F15)
# Groups fall codes into three biomechanical categories
# ---------------------------------------------------------------------------
FALL_TYPE_MAP: dict[str, str] = {
    "F01": "slip",  "F02": "slip",  "F03": "slip",  "F04": "slip",  "F05": "slip",
    "F06": "trip",  "F07": "trip",  "F08": "trip",  "F09": "trip",  "F10": "trip",
    "F11": "faint", "F12": "faint", "F13": "faint", "F14": "faint", "F15": "faint",
}

# ---------------------------------------------------------------------------
# ADL activity code → human-readable activity name (D01–D19)
# ---------------------------------------------------------------------------
PRE_ACTIVITY_MAP: dict[str, str] = {
    "D01": "walking",  "D02": "walking",
    "D03": "walking",  "D04": "walking",
    "D05": "standing", "D06": "standing",
    "D07": "sitting",  "D08": "sitting",
    "D09": "bending",  "D10": "bending",
    "D11": "bending",  "D12": "standing",
    "D13": "standing", "D14": "sitting",
    "D15": "walking",  "D16": "walking",
    "D17": "sitting",  "D18": "standing",
    "D19": "walking",
}

# ---------------------------------------------------------------------------
# What was the person doing just before each fall type?
# Used to assign pre-activity to fall clips (they have no D-code)
# ---------------------------------------------------------------------------
FALL_TO_PRE_ACTIVITY: dict[str, str] = {
    "slip":  "walking",   # slips happen while walking on slippery surfaces
    "trip":  "walking",   # trips happen while walking and hitting an obstacle
    "faint": "standing",  # faints happen from a standing/stationary position
}

# Thresholds for post-fall state classification based on continuous stillness
# duration after a fall. The 10s/30s cutoffs are consistent with published
# fall-detection studies that use similar inactivity-duration thresholds to
# distinguish brief ADL movement from prolonged immobility.
_STILLNESS_DURATION_STUNNED_SECONDS = 10.0
_STILLNESS_DURATION_UNCONSCIOUS_SECONDS = 30.0
_SAMPLE_RATE_HZ = 100.0
_MOVEMENT_THRESHOLD = 0.75


def get_fall_label(activity_code: str) -> int:
    """
    Return binary fall label.

    Parameters
    ----------
    activity_code : str
        SisFall activity code e.g. 'F01', 'D05'.

    Returns
    -------
    int
        1 if this is a fall clip, 0 if this is a daily-activity clip.
    """
    return 1 if activity_code.upper().startswith("F") else 0


def get_fall_type(activity_code: str) -> str:
    """
    Return the fall type string for a given activity code.

    Parameters
    ----------
    activity_code : str
        SisFall code — must start with 'F' for a meaningful return value.

    Returns
    -------
    str
        'slip', 'trip', or 'faint' for fall codes.
        'none' for ADL codes (not a fall).
    """
    return FALL_TYPE_MAP.get(activity_code.upper(), "none")


def get_pre_activity(activity_code: str) -> str:
    """
    Return the activity being performed before (or during) the clip.

    For fall clips — the preceding activity is inferred from the fall type
    (e.g. slips and trips happen while walking).
    For ADL clips — the activity is read directly from PRE_ACTIVITY_MAP.

    Parameters
    ----------
    activity_code : str
        SisFall activity code e.g. 'F06' or 'D01'.

    Returns
    -------
    str
        One of: 'walking', 'standing', 'bending', 'sitting', or 'unknown'.
    """
    code = activity_code.upper()

    if code.startswith("F"):
        # Look up what precedes this fall type
        fall_type = get_fall_type(code)
        return FALL_TO_PRE_ACTIVITY.get(fall_type, "unknown")

    # ADL clip — direct lookup
    return PRE_ACTIVITY_MAP.get(code, "unknown")


def compute_post_state(window: np.ndarray, is_fall: bool = True) -> str:
    """
    Estimate the physical state of the person after a fall from the duration of
    continuous stillness in the post-impact signal.

    Instead of looking at only the last 0.3 seconds of the window, this version
    measures how long the person remains motionless before movement resumes.
    Long inactivity durations are treated as increasingly severe post-fall states.

    Parameters
    ----------
    window : np.ndarray
        Shape (n, 6) float32 — IMU signal segment.
        Columns 0-2 are acc_x, acc_y, acc_z.
    is_fall : bool
        If False (ADL window), always returns 'moving' without computation.

    Returns
    -------
    str
        'unconscious' | 'stunned' | 'moving' | 'unknown'
    """
    # ADL windows never classify as unconscious or stunned
    if not is_fall:
        return "moving"

    if window is None or not isinstance(window, np.ndarray):
        return "unknown"

    if window.ndim != 2 or window.shape[0] < 10 or window.shape[1] < 3:
        return "unknown"

    # Extract accelerometer channels (columns 0, 1, 2)
    acc = window[:, :3].astype(np.float32)

    # Normalize each axis so the movement threshold is robust for both raw and
    # z-scored signals.
    acc_norm = np.empty_like(acc, dtype=np.float32)
    for axis in range(acc.shape[1]):
        col = acc[:, axis]
        std = float(np.std(col))
        if std < 1e-8:
            acc_norm[:, axis] = 0.0
        else:
            acc_norm[:, axis] = (col - np.mean(col)) / std

    # Resultant magnitude: sqrt(x² + y² + z²) at each timestep
    acc_magnitude = np.sqrt(np.sum(acc_norm ** 2, axis=1))

    # Smooth with a short moving-average filter to reduce jitter.
    smooth = acc_magnitude
    if len(smooth) >= 5:
        smooth = np.convolve(smooth, np.ones(5, dtype=np.float32) / 5.0, mode="same")

    # Ignore a brief impact buffer so the initial fall spike does not
    # immediately collapse the stillness duration to zero.
    impact_buffer = int(0.5 * _SAMPLE_RATE_HZ)
    start_idx = min(max(impact_buffer, 1), len(smooth) - 1)

    still_samples = 0
    for value in smooth[start_idx:]:
        if value <= _MOVEMENT_THRESHOLD:
            still_samples += 1
        else:
            break

    still_seconds = still_samples / _SAMPLE_RATE_HZ

    if still_seconds > _STILLNESS_DURATION_UNCONSCIOUS_SECONDS:
        return "unconscious"
    elif still_seconds >= _STILLNESS_DURATION_STUNNED_SECONDS:
        return "stunned"
    else:
        return "moving"


# ---------------------------------------------------------------------------
# Standalone demo — run:  python src/labeller.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    from data_loader import load_clip

    print("=== labeller.py demo ===")

    test_codes = ["F01", "F06", "F11", "D01", "D05", "D09", "UNKNOWN"]

    for code in test_codes:
        label = get_fall_label(code)
        ftype = get_fall_type(code)
        pre   = get_pre_activity(code)
        print(f"  {code:<10} | fall={label} | type={ftype:<6} | pre_activity={pre}")

    # Demonstrate post-state detection
    print("\nPost-state examples:")

    # Simulate a 'still' window (small noise after impact)
    rng = np.random.default_rng(99)
    still_window = rng.normal(0, 0.01, (200, 6)).astype(np.float32)
    print(f"  Near-static window → {compute_post_state(still_window, is_fall=True)}")

    # Simulate an active window (large movement throughout)
    active_window = rng.normal(0, 5.0, (200, 6)).astype(np.float32)
    print(f"  Active window      → {compute_post_state(active_window, is_fall=True)}")

    # ADL window always returns 'moving'
    adl_window = rng.normal(0, 1.0, (200, 6)).astype(np.float32)
    print(f"  ADL window         → {compute_post_state(adl_window, is_fall=False)}")

    # Compare the old variance-based logic with the new duration-based logic
    # on a handful of real fall clips from the SisFall dataset.
    def _legacy_compute_post_state(window: np.ndarray, is_fall: bool = True) -> str:
        if not is_fall:
            return "moving"

        if window is None or not isinstance(window, np.ndarray):
            return "unknown"

        if window.ndim != 2 or window.shape[0] < 30 or window.shape[1] < 3:
            return "unknown"

        acc = window[:, :3].astype(np.float32)
        acc_magnitude = np.sqrt(np.sum(acc ** 2, axis=1))
        tail = acc_magnitude[-30:]
        variance = float(np.var(tail))
        stillness = 1.0 / (1.0 + variance)

        if stillness > 0.85:
            return "unconscious"
        elif stillness > 0.55:
            return "stunned"
        else:
            return "moving"

    data_root = Path(__file__).parent.parent / "data" / "SisFall_dataset"
    if data_root.exists():
        clip_paths = []
        for path in sorted(data_root.rglob("*.txt")):
            if path.name.startswith("F"):
                clip_paths.append(path)
                if len(clip_paths) >= 4:
                    break

        print("\nReal SisFall clip comparison (old vs new):")
        for path in clip_paths:
            clip = load_clip(str(path))
            if clip is None:
                continue

            tail = clip[-min(4000, len(clip)):]
            old_state = _legacy_compute_post_state(tail, is_fall=True)
            new_state = compute_post_state(tail, is_fall=True)
            print(f"  {path.name:<20} | old={old_state:<10} | new={new_state:<10}")
    else:
        print("\nSisFall dataset not found — skipped real-clip comparison")
