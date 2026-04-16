import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# --------------------
# Paths & constants
# --------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_FILE = PROCESSED_DIR / "features.csv"

CLASS_MAP = {
    "falls": 1,
    "no_falls": 0,
}

FRAME_SAMPLE_RATE = 5  # use every 5th frame to reduce compute


mp_pose = mp.solutions.pose(model_complexity=0)


# --------------------
# Feature extraction
# --------------------
def extract_features(results, image_shape):
    """
    Extract simple pose-based features from MediaPipe Pose results.
    Returns a list of floats.
    """
    height, width, _ = image_shape
    landmarks = results.pose_landmarks.landmark

    def get_point(idx):
        lm = landmarks[idx]
        return np.array([lm.x * width, lm.y * height], dtype=np.float32)

    # Key joints
    left_shoulder = get_point(mp_pose.PoseLandmark.LEFT_SHOULDER)
    right_shoulder = get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    left_hip = get_point(mp_pose.PoseLandmark.LEFT_HIP)
    right_hip = get_point(mp_pose.PoseLandmark.RIGHT_HIP)

    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    mid_hip = (left_hip + right_hip) / 2.0

    # Vector from hip to shoulder
    vec = mid_shoulder - mid_hip  # [dx, dy]

    # Angle between this vector and vertical axis (0 = upright, 90 = horizontal)
    # vertical axis approx (0, -1); we just look at ratio dx/dy
    angle_rad = np.arctan2(abs(vec[0]), abs(vec[1]) + 1e-6)
    torso_angle_deg = float(np.degrees(angle_rad))

    # Normalised vertical positions
    hip_y_norm = float(mid_hip[1] / height)
    shoulder_y_norm = float(mid_shoulder[1] / height)

    # Bounding box around all landmarks
    xs = np.array([lm.x for lm in landmarks], dtype=np.float32) * width
    ys = np.array([lm.y for lm in landmarks], dtype=np.float32) * height

    bbox_w_norm = float((xs.max() - xs.min()) / width)
    bbox_h_norm = float((ys.max() - ys.min()) / height)

    return [
        torso_angle_deg,
        hip_y_norm,
        shoulder_y_norm,
        bbox_w_norm,
        bbox_h_norm,
    ]


# --------------------
# Video processing
# --------------------
def process_video(video_path: Path, label: int, rows: list):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return

    frame_idx = 0

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

            # Subsample frames
            if frame_idx % FRAME_SAMPLE_RATE != 0:
                frame_idx += 1
                continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                feats = extract_features(results, frame.shape)
                rows.append(
                    feats
                    + [
                        label,
                        video_path.name,
                        frame_idx,
                    ]
                )

            frame_idx += 1

    cap.release()


def build_feature_dataset():
    rows = []

    for class_name, label in CLASS_MAP.items():
        class_dir = RAW_DIR / class_name
        if not class_dir.exists():
            print(f"[WARN] Missing directory: {class_dir}")
            continue

        # Handle different extensions (.avi, .mp4, etc.)
        video_paths = list(class_dir.glob("*.*"))
        print(f"[INFO] Found {len(video_paths)} videos in {class_dir}")

        for video_path in video_paths:
            print(f"[INFO] Processing {video_path.name} (label={label})")
            process_video(video_path, label, rows)

    if not rows:
        print("[ERROR] No data extracted. Check your paths and videos.")
        return

    header = [
        "torso_angle_deg",
        "hip_y_norm",
        "shoulder_y_norm",
        "bbox_w_norm",
        "bbox_h_norm",
        "label",
        "video_name",
        "frame_idx",
    ]

    with open(FEATURES_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[OK] Saved {len(rows)} samples to {FEATURES_FILE}")


if __name__ == "__main__":
    build_feature_dataset()
