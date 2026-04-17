# scripts/run_realtime.py
from pathlib import Path
from collections import deque
import time

import cv2
import mediapipe as mp
import numpy as np
import joblib

from utils.pose_features import extract_features
from alerts.manager import AlertManager
from alerts.buzzer import BuzzerAlert
from alerts.sim_a7680c import SimA7680CAlert

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "fall_detector_rf_test1.pkl"

FALL_LABEL = 1
WINDOW_NAME = "AI Fall Detection - Realtime"

PREDICTION_WINDOW = 10
FALL_RATIO_THRESHOLD = 0.6   # 60% of last N frames predicted as fall
PROBA_FRAME_THRESHOLD = 0.5  # threshold to convert proba->bool in recent_preds

# Giảm false alarm bằng xác nhận theo thời gian
CONFIRM_SEC = 1.5            # phải fall liên tục >= 1.5s mới trigger
ALERT_COOLDOWN_SEC = 60

# ALERT CONFIG
BUZZER_GPIO = 23
SIM_PORT = "/dev/serial0"    # hoặc "/dev/ttyUSB2"
SIM_BAUD = 115200
CALL_NUMBERS = ["+84942826528"]  # sửa số của bạn
RING_SEC = 20

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    print(f"[INFO] Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    has_proba = hasattr(model, "predict_proba")

    # Alert manager (runs in background)
    alert_mgr = AlertManager(
        alerts=[
            BuzzerAlert(pin=BUZZER_GPIO),
            SimA7680CAlert(port=SIM_PORT, baud=SIM_BAUD, numbers=CALL_NUMBERS, ring_sec=RING_SEC),
        ],
        cooldown_sec=ALERT_COOLDOWN_SEC,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("[INFO] Webcam opened. Press 'q' to quit.")
    recent_preds = deque(maxlen=PREDICTION_WINDOW)

    # state for confirm
    fall_started_ts = None
    fall_triggered_this_event = False

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

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            fall_prob_display = 0.0
            fall_flag = False

            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                feats = extract_features(results, frame.shape, mp_pose)
                if feats is not None:
                    X = np.array(feats, dtype=np.float32).reshape(1, -1)

                    if has_proba:
                        proba = model.predict_proba(X)[0]
                        if hasattr(model, "classes_"):
                            fall_index = int(np.where(model.classes_ == FALL_LABEL)[0][0])
                        else:
                            fall_index = 1
                        fall_prob = float(proba[fall_index])
                        fall_prob_display = fall_prob
                        recent_preds.append(fall_prob > PROBA_FRAME_THRESHOLD)
                    else:
                        pred_label = int(model.predict(X)[0])
                        recent_preds.append(pred_label == FALL_LABEL)
                        fall_prob_display = sum(recent_preds) / max(len(recent_preds), 1)

                    if recent_preds:
                        fall_ratio = sum(recent_preds) / len(recent_preds)
                        if fall_ratio >= FALL_RATIO_THRESHOLD:
                            fall_flag = True

            now = time.time()

            # Confirm logic to reduce false alarm:
            # require fall_flag continuously for CONFIRM_SEC before triggering alerts
            if fall_flag:
                if fall_started_ts is None:
                    fall_started_ts = now
                    fall_triggered_this_event = False

                if (not fall_triggered_this_event) and (now - fall_started_ts >= CONFIRM_SEC):
                    fired = alert_mgr.trigger({
                        "ts": now,
                        "fall_prob": fall_prob_display,
                        "fall_ratio": (sum(recent_preds)/len(recent_preds)) if recent_preds else 0.0,
                    })
                    if fired:
                        fall_triggered_this_event = True
            else:
                fall_started_ts = None
                fall_triggered_this_event = False

            # UI
            status_text = "FALL DETECTED" if fall_flag else "NO FALL"
            status_color = (0, 0, 255) if fall_flag else (0, 255, 0)

            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), thickness=-1)

            cv2.putText(frame, status_text, (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3, cv2.LINE_AA)

            cv2.putText(frame, f"Fall prob: {fall_prob_display:.2f}", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Quitting realtime detection.")
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()