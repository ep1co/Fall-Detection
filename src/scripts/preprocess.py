import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from utils.pose_features_v2 import (
    FEATURE_NAMES,
    BASE_FEATURE_COUNT,
    extract_features,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

# Expected structure:
# data/kfold/fold_1/fall
# data/kfold/fold_1/ADL
# ...
# data/kfold/fold_5/fall
# data/kfold/fold_5/ADL
DATASET_DIR = DATA_DIR / "kfold"

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FRAME_FEATURES_FILE = PROCESSED_DIR / "frame_features_kfold_v3.csv"
WINDOW_FEATURES_FILE = PROCESSED_DIR / "window_features_kfold_v3.csv"

FALL_SEGMENTS_FILE = DATA_DIR / "fall_segments.csv"

CLASS_MAP = {
    "fall": 1,
    "falls": 1,
    "ADL": 0,
    "adl": 0,
    "no_fall": 0,
    "no_falls": 0,
}

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# Sample every N frames.
# Example: 25 FPS, sample every 5 frames -> 5 samples/second.
FRAME_SAMPLE_RATE = 5

# Window-level model settings.
WINDOW_SEC = 2.0
WINDOW_STEP_SEC = 0.5

# A window is fall if at least 30% of the window overlaps with the annotated fall segment.
FALL_OVERLAP_THRESHOLD = 0.30

# If a fall window overlaps slightly with fall segment but not enough, skip it to avoid noisy labels.
SKIP_AMBIGUOUS_WINDOWS = True

mp_pose = mp.solutions.pose


def iter_video_files(folder: Path):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def read_fall_segments():
    """
    Read fall segment annotations.

    Expected format:
    video_name,fall_start_sec,fall_end_sec

    Example:
    video (7).avi,3.8,5.6
    """
    segments = {}

    if not FALL_SEGMENTS_FILE.exists():
        print(f"[WARN] Missing annotation file: {FALL_SEGMENTS_FILE}")
        print("[WARN] Fall videos will be labeled as fall for all windows. This is not recommended.")
        return segments

    with open(FALL_SEGMENTS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            video_name = row["video_name"].strip()

            try:
                start_sec = float(row["fall_start_sec"])
                end_sec = float(row["fall_end_sec"])
            except Exception:
                print(f"[WARN] Invalid fall segment row: {row}")
                continue

            if end_sec <= start_sec:
                print(f"[WARN] Invalid segment time for {video_name}: {start_sec} -> {end_sec}")
                continue

            segments.setdefault(video_name, []).append((start_sec, end_sec))

    print(f"[INFO] Loaded fall annotations for {len(segments)} videos")
    return segments


def get_segments_for_video(video_path: Path, fall_segments: dict):
    """
    Match annotation by filename.

    If your dataset has duplicate filenames in different folders,
    rename videos or extend this function to use relative path.
    """
    return fall_segments.get(video_path.name, [])


def overlap_duration(a_start, a_end, b_start, b_end):
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0.0, end - start)


def get_window_label(
    video_label,
    win_start,
    win_end,
    fall_intervals,
):
    """
    Return:
    - 1 for fall window
    - 0 for non-fall window
    - None for ambiguous window that should be skipped
    """
    if video_label == 0:
        return 0

    # No annotation: fallback to video-level label.
    # This is less accurate but keeps the pipeline runnable.
    if not fall_intervals:
        return 1

    window_len = max(win_end - win_start, 1e-6)

    max_overlap = 0.0
    for fall_start, fall_end in fall_intervals:
        max_overlap = max(
            max_overlap,
            overlap_duration(win_start, win_end, fall_start, fall_end),
        )

    overlap_ratio = max_overlap / window_len

    if overlap_ratio >= FALL_OVERLAP_THRESHOLD:
        return 1

    if overlap_ratio <= 1e-6:
        return 0

    if SKIP_AMBIGUOUS_WINDOWS:
        return None

    return 0


def summarize_window(frame_rows):
    """
    Convert multiple frame-level rows into one window-level feature vector.

    For each frame feature, compute:
    mean, std, min, max, range, first, last, delta
    """
    arr = np.array(
        [[float(r[name]) for name in FEATURE_NAMES] for r in frame_rows],
        dtype=np.float32,
    )

    stats = {}

    for i, name in enumerate(FEATURE_NAMES):
        values = arr[:, i]

        v_mean = float(np.mean(values))
        v_std = float(np.std(values))
        v_min = float(np.min(values))
        v_max = float(np.max(values))
        v_range = float(v_max - v_min)
        v_first = float(values[0])
        v_last = float(values[-1])
        v_delta = float(v_last - v_first)

        stats[f"{name}_mean"] = v_mean
        stats[f"{name}_std"] = v_std
        stats[f"{name}_min"] = v_min
        stats[f"{name}_max"] = v_max
        stats[f"{name}_range"] = v_range
        stats[f"{name}_first"] = v_first
        stats[f"{name}_last"] = v_last
        stats[f"{name}_delta"] = v_delta

    # Add useful global window quality features.
    stats["window_num_frames"] = int(len(frame_rows))
    stats["window_duration_sec"] = float(
        frame_rows[-1]["timestamp_sec"] - frame_rows[0]["timestamp_sec"]
    )

    return stats


def make_windows_for_video(
    video_frame_rows,
    video_label,
    fall_intervals,
):
    """
    Build window-level samples from frame-level samples of one video.
    """
    if not video_frame_rows:
        return []

    video_frame_rows = sorted(video_frame_rows, key=lambda r: r["timestamp_sec"])

    start_time = float(video_frame_rows[0]["timestamp_sec"])
    end_time = float(video_frame_rows[-1]["timestamp_sec"])

    windows = []
    win_start = start_time

    while win_start + WINDOW_SEC <= end_time + 1e-6:
        win_end = win_start + WINDOW_SEC

        rows_in_window = [
            r for r in video_frame_rows
            if win_start <= float(r["timestamp_sec"]) < win_end
        ]

        # Need at least a few samples to compute meaningful stats.
        if len(rows_in_window) >= 3:
            label = get_window_label(
                video_label=video_label,
                win_start=win_start,
                win_end=win_end,
                fall_intervals=fall_intervals,
            )

            if label is not None:
                stats = summarize_window(rows_in_window)

                first_row = rows_in_window[0]

                window_row = {
                    **stats,
                    "label": label,
                    "video_label": video_label,
                    "fold": first_row["fold"],
                    "activity_type": first_row["activity_type"],
                    "video_name": first_row["video_name"],
                    "window_start_sec": float(win_start),
                    "window_end_sec": float(win_end),
                }

                windows.append(window_row)

        win_start += WINDOW_STEP_SEC

    return windows


def process_video(
    video_path: Path,
    label: int,
    fold_name: str,
    activity_type: str,
    fall_segments: dict,
):
    """
    Return:
    - frame_rows
    - window_rows
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 30.0

    fall_intervals = get_segments_for_video(video_path, fall_segments)

    if label == 1 and not fall_intervals:
        print(f"[WARN] No fall annotation for {video_path.name}. Using video-level label.")

    frame_idx = 0
    prev_base_features = None
    prev_frame_idx = None

    frame_rows = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_SAMPLE_RATE != 0:
                frame_idx += 1
                continue

            timestamp_sec = frame_idx / fps

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if prev_frame_idx is None:
                dt_sec = 1.0
            else:
                dt_sec = (frame_idx - prev_frame_idx) / fps

            feats = extract_features(
                results,
                frame.shape,
                mp_pose,
                prev_base_features=prev_base_features,
                dt_sec=dt_sec,
            )

            if feats is not None:
                row = {
                    name: float(value)
                    for name, value in zip(FEATURE_NAMES, feats)
                }

                row.update(
                    {
                        "label": label,
                        "fold": fold_name,
                        "activity_type": activity_type,
                        "video_name": video_path.name,
                        "frame_idx": int(frame_idx),
                        "timestamp_sec": float(timestamp_sec),
                        "fps": float(fps),
                    }
                )

                frame_rows.append(row)

                prev_base_features = feats[:BASE_FEATURE_COUNT]
                prev_frame_idx = frame_idx

            frame_idx += 1

    cap.release()

    window_rows = make_windows_for_video(
        video_frame_rows=frame_rows,
        video_label=label,
        fall_intervals=fall_intervals,
    )

    return frame_rows, window_rows


def write_csv(path: Path, rows: list, header: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def build_feature_dataset():
    all_frame_rows = []
    all_window_rows = []

    if not DATASET_DIR.exists():
        print(f"[ERROR] Missing dataset directory: {DATASET_DIR}")
        return

    fall_segments = read_fall_segments()

    fold_dirs = sorted(
        p for p in DATASET_DIR.iterdir()
        if p.is_dir() and p.name.startswith("fold_")
    )

    if not fold_dirs:
        print(f"[ERROR] No fold directories found in: {DATASET_DIR}")
        return

    print(f"[INFO] Found {len(fold_dirs)} folds")

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        print(f"\n[INFO] Processing {fold_name}")

        for class_dir in sorted(p for p in fold_dir.iterdir() if p.is_dir()):
            class_name = class_dir.name

            if class_name not in CLASS_MAP:
                print(f"[WARN] Skip unknown class folder: {class_dir}")
                continue

            label = CLASS_MAP[class_name]
            video_paths = iter_video_files(class_dir)

            print(
                f"[INFO] {fold_name}/{class_name}: "
                f"{len(video_paths)} videos, label={label}"
            )

            for video_path in video_paths:
                activity_type = "fall" if label == 1 else "ADL"

                print(
                    f"[INFO] Processing {video_path.name} "
                    f"(fold={fold_name}, label={label}, activity={activity_type})"
                )

                frame_rows, window_rows = process_video(
                    video_path=video_path,
                    label=label,
                    fold_name=fold_name,
                    activity_type=activity_type,
                    fall_segments=fall_segments,
                )

                all_frame_rows.extend(frame_rows)
                all_window_rows.extend(window_rows)

    if not all_frame_rows:
        print("[ERROR] No frame data extracted. Check your paths and videos.")
        return

    if not all_window_rows:
        print("[ERROR] No window data created. Check video length and window settings.")
        return

    frame_header = (
        FEATURE_NAMES
        + [
            "label",
            "fold",
            "activity_type",
            "video_name",
            "frame_idx",
            "timestamp_sec",
            "fps",
        ]
    )

    window_feature_names = [
        key for key in all_window_rows[0].keys()
        if key not in {
            "label",
            "video_label",
            "fold",
            "activity_type",
            "video_name",
            "window_start_sec",
            "window_end_sec",
        }
    ]

    window_header = (
        window_feature_names
        + [
            "label",
            "video_label",
            "fold",
            "activity_type",
            "video_name",
            "window_start_sec",
            "window_end_sec",
        ]
    )

    write_csv(FRAME_FEATURES_FILE, all_frame_rows, frame_header)
    write_csv(WINDOW_FEATURES_FILE, all_window_rows, window_header)

    print(f"\n[OK] Saved {len(all_frame_rows)} frame samples to {FRAME_FEATURES_FILE}")
    print(f"[OK] Saved {len(all_window_rows)} window samples to {WINDOW_FEATURES_FILE}")


if __name__ == "__main__":
    build_feature_dataset()