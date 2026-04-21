# scripts/run_realtime.py (phần chính)
from pathlib import Path
from collections import deque
import time

import cv2
import mediapipe as mp
import numpy as np
import joblib

from utils.pose_features import extract_features
from alerts.buzzer import ContinuousBuzzer
from alerts.sim_a7680c import SimA7680CAlarm

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "fall_detector_rf_test1.pkl"

WINDOW_NAME = "AI Fall Detection - Realtime"
FALL_LABEL = 1

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# --- smoothing window cho RF ---
PREDICTION_WINDOW = 10
FALL_RATIO_THRESHOLD = 0.6
PROBA_FRAME_THRESHOLD = 0.5

# --- state machine timing ---
CONFIRM_FALL_SEC = 1.2
RECOVER_SEC = 3.0

# --- heuristic thresholds (tune) ---
HIP_V_THR = 0.55
HIP_A_THR = 2.0
ANG_V_THR = 120.0
TORSO_ANGLE_THR = 50.0
ASPECT_THR = 1.2

# --- alert ---
BUZZER_PIN = 23
SIM_PORT = "/dev/serial0"
SIM_BAUD = 115200
CALL_NUMBERS = ["+84942826528"]
RING_SEC = 20

class State:
    NORMAL = "NORMAL"
    FALL_SUSPECT = "FALL_SUSPECT"
    ALARMING = "ALARMING"
    RECOVERING = "RECOVERING"

def main():
    model = joblib.load(MODEL_PATH)
    has_proba = hasattr(model, "predict_proba")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    recent_preds = deque(maxlen=PREDICTION_WINDOW)

    # buffers for dynamics
    prev_t = None
    prev_hip_y = None
    prev_hip_v = None
    prev_angle = None

    # state machine vars
    state = State.NORMAL
    state_ts = time.time()

    # alarms
    buzzer = ContinuousBuzzer(pin=BUZZER_PIN, on_sec=0.15, off_sec=0.15)
    sim_alarm = SimA7680CAlarm(port=SIM_PORT, baud=SIM_BAUD, numbers=CALL_NUMBERS, ring_sec=RING_SEC)

    def start_alarms():
        buzzer.start()
        sim_alarm.start()

    def stop_alarms():
        buzzer.stop()
        sim_alarm.stop()

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

            frame = cv2.flip(frame, 1)
            now = time.time()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            fall_prob = 0.0
            fall_ratio = 0.0
            torso_angle = None
            hip_y = None
            bbox_aspect = None

            fall_candidate = False
            upright_candidate = False

            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                feats = extract_features(results, frame.shape, mp_pose)
                if feats is not None:
                    torso_angle = float(feats[0])
                    hip_y = float(feats[1])
                    bbox_w = float(feats[3])
                    bbox_h = float(feats[4])
                    bbox_aspect = bbox_w / (bbox_h + 1e-6)

                    X = np.array(feats, dtype=np.float32).reshape(1, -1)

                    if has_proba:
                        proba = model.predict_proba(X)[0]
                        fall_index = int(np.where(model.classes_ == FALL_LABEL)[0][0]) if hasattr(model, "classes_") else 1
                        fall_prob = float(proba[fall_index])
                        recent_preds.append(fall_prob > PROBA_FRAME_THRESHOLD)
                    else:
                        pred = int(model.predict(X)[0])
                        recent_preds.append(pred == FALL_LABEL)
                        fall_prob = sum(recent_preds) / max(len(recent_preds), 1)

                    if recent_preds:
                        fall_ratio = sum(recent_preds) / len(recent_preds)

                    # --- dynamics ---
                    if prev_t is not None and hip_y is not None and prev_hip_y is not None:
                        dt = max(1e-3, now - prev_t)
                        hip_v = (hip_y - prev_hip_y) / dt
                        hip_a = 0.0 if prev_hip_v is None else (hip_v - prev_hip_v) / dt

                        ang_v = 0.0
                        if prev_angle is not None and torso_angle is not None:
                            ang_v = (torso_angle - prev_angle) / dt

                        # Fall gating logic (RF + dynamics + posture)
                        dynamic_ok = (hip_v > HIP_V_THR and (hip_a > HIP_A_THR or abs(ang_v) > ANG_V_THR))
                        posture_ok = (torso_angle > TORSO_ANGLE_THR or (bbox_aspect is not None and bbox_aspect > ASPECT_THR))

                        # Kết hợp: RF phải "nghi ngờ", rồi dynamics + posture xác nhận
                        fall_candidate = (fall_prob > 0.5 and posture_ok and dynamic_ok) or (fall_ratio >= FALL_RATIO_THRESHOLD and posture_ok)

                        # Upright candidate (đứng dậy / không còn nằm ngang)
                        # torso nhỏ + bbox cao hơn rộng
                        upright_candidate = (torso_angle < 25.0) and (bbox_aspect is not None and bbox_aspect < 0.8)

                        prev_hip_v = hip_v
                    prev_t = now
                    prev_hip_y = hip_y
                    prev_angle = torso_angle

            # --- state machine ---
            if state == State.NORMAL:
                if fall_candidate:
                    state = State.FALL_SUSPECT
                    state_ts = now

            elif state == State.FALL_SUSPECT:
                if fall_candidate:
                    if now - state_ts >= CONFIRM_FALL_SEC:
                        state = State.ALARMING
                        state_ts = now
                        start_alarms()
                else:
                    state = State.NORMAL
                    state_ts = now

            elif state == State.ALARMING:
                # nếu có dấu hiệu đứng dậy thì chuyển RECOVERING
                if upright_candidate:
                    state = State.RECOVERING
                    state_ts = now

            elif state == State.RECOVERING:
                if upright_candidate:
                    if now - state_ts >= RECOVER_SEC:
                        stop_alarms()
                        state = State.NORMAL
                        state_ts = now
                else:
                    # lại ngã => quay lại alarming ngay
                    state = State.ALARMING
                    state_ts = now

            # --- UI overlay ---
            status_text = state
            status_color = (0, 255, 0)
            if state in (State.FALL_SUSPECT,):
                status_color = (0, 165, 255)
            if state in (State.ALARMING, State.RECOVERING):
                status_color = (0, 0, 255)

            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), thickness=-1)
            cv2.putText(frame, status_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
            cv2.putText(frame, f"p_fall={fall_prob:.2f} ratio={fall_ratio:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow(WINDOW_NAME, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    # cleanup
    stop_alarms()
    buzzer.cleanup()
    cap.release()
    cv2.destroyAllWindows()