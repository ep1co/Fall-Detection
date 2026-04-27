import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from utils.pose_features import extract_features

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

FRAME_SAMPLE_RATE = 5

mp_pose = mp.solutions.pose

def process_video(video_path: Path, label: int, rows: list):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return

    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # similar to realtime.py 
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

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            feats = extract_features(results, frame.shape, mp_pose)
            if feats is not None:
                rows.append(feats + [label, video_path.name, frame_idx])

            frame_idx += 1

    cap.release()

def build_feature_dataset():
    rows = []

    for class_name, label in CLASS_MAP.items():
        class_dir = RAW_DIR / class_name
        if not class_dir.exists():
            print(f"[WARN] Missing directory: {class_dir}")
            continue

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