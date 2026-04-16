from pathlib import Path
from collections import deque
import time

import cv2
import mediapipe as mp
import numpy as np
import joblib


# --------------------
# Paths & constants
# --------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "fall_detector_rf_test1.pkl"

FALL_LABEL = 1
NO_FALL_LABEL = 0

WINDOW_NAME = "AI Fall Detection - Realtime"

# Smoothing: how many recent frames to consider and threshold for fall
PREDICTION_WINDOW = 10
FALL_PROB_THRESHOLD = 0.6  # 60% of last N frames predicted as fall


mp_pose = mp.solutions.pose


# --------------------
# Feature extraction (same as in preprocess.py)
# --------------------
def extract_features(results, image_shape):
    """
    Extract simple pose-based features from MediaPipe Pose results.
    Must match features used in training.
    """
    height, width, _ = image_shape
    landmarks = results.pose_landmarks.landmark

    def get_point(idx):
        lm = landmarks[idx]
        return np.array([lm.x * width, lm.y * height], dtype=np.float32)

    left_shoulder = get_point(mp_pose.PoseLandmark.LEFT_SHOULDER)
    right_shoulder = get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    left_hip = get_point(mp_pose.PoseLandmark.LEFT_HIP)
    right_hip = get_point(mp_pose.PoseLandmark.RIGHT_HIP)

    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    mid_hip = (left_hip + right_hip) / 2.0

    vec = mid_shoulder - mid_hip  # [dx, dy]

    angle_rad = np.arctan2(abs(vec[0]), abs(vec[1]) + 1e-6)
    torso_angle_deg = float(np.degrees(angle_rad))

    hip_y_norm = float(mid_hip[1] / height)
    shoulder_y_norm = float(mid_shoulder[1] / height)

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
# Main realtime loop
# --------------------
def main():
    # Load model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it first.")

    print(f"[INFO] Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # Some sklearn models may not have predict_proba (ours does, but just in case)
    has_proba = hasattr(model, "predict_proba")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions.")

    print("[INFO] Webcam opened. Press 'q' to quit.")

    # For smoothing predictions
    recent_preds = deque(maxlen=PREDICTION_WINDOW)

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
                print("[WARN] Failed to read frame from webcam.")
                break

            # Flip horizontally for natural feel
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            fall_prob_display = 0.0
            fall_flag = False
            status_text = "NO FALL"
            status_color = (0, 255, 0)  # green

            if results.pose_landmarks:
                # Draw pose
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

                features = extract_features(results, frame.shape)
                X = np.array(features, dtype=np.float32).reshape(1, -1)

                if has_proba:
                    proba = model.predict_proba(X)[0]
                    # proba[1] corresponds to label 1 (fall) if classes_ is [0,1]
                    if hasattr(model, "classes_"):
                        # Ensure we pick the probability of FALL_LABEL
                        fall_index = int(np.where(model.classes_ == FALL_LABEL)[0][0])
                    else:
                        fall_index = 1
                    fall_prob = float(proba[fall_index])
                    fall_prob_display = fall_prob
                    recent_preds.append(fall_prob > 0.5)
                else:
                    pred_label = int(model.predict(X)[0])
                    recent_preds.append(pred_label == FALL_LABEL)
                    fall_prob = sum(recent_preds) / max(len(recent_preds), 1)
                    fall_prob_display = fall_prob

                # Smooth decision over last N frames
                if recent_preds:
                    fall_ratio = sum(recent_preds) / len(recent_preds)
                    if fall_ratio >= FALL_PROB_THRESHOLD:
                        fall_flag = True

            # Decide final status text / colour
            if fall_flag:
                status_text = "FALL DETECTED"
                status_color = (0, 0, 255)  # red

            # Overlay UI
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), thickness=-1)

            cv2.putText(
                frame,
                status_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                status_color,
                3,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Fall prob: {fall_prob_display:.2f}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Quitting realtime detection.")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
