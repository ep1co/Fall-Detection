# scripts/run_realtime.py - AI Fall Detection with Cloud & MQTT
from pathlib import Path
from collections import deque
import time
import threading
import os

from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).resolve().parents[2] / ".env"
print(f"[APP] Loading env from: {env_path}")
print(f"[APP] .env exists: {env_path.exists()}")
load_dotenv(env_path)

# Verify critical config
def verify_config():
    required_vars = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "HIVEMQ_HOST", "HIVEMQ_USER"]
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print(f"\n[WARNING] Missing environment variables: {', '.join(missing)}")
        print(f"[WARNING] Please check your .env file at: {env_path}\n")
    else:
        print("[APP] ✓ All required environment variables loaded\n")

verify_config()

import cv2
import mediapipe as mp
import numpy as np
import joblib

from utils.pose_features import extract_features
from alerts.buzzer import ContinuousBuzzer
from alerts.sim_a7680c import SimA7680CAlarm
from utils.cloud_uploader import gen_event_id, upload_image_to_supabase, insert_fall_event
from utils.mqtt_handler import init_mqtt, publish_fall_alarm, publish_safe_status, publish_muted_status

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
RING_SEC = 15

# --- snapshot ---
SNAP_W, SNAP_H = 640, 480
JPEG_QUALITY = 70


class State:
    NORMAL = "NORMAL"
    FALL_SUSPECT = "FALL_SUSPECT"
    ALARMING = "ALARMING"
    RECOVERING = "RECOVERING"


def make_jpeg_bytes(frame_bgr, w=640, h=480, quality=70):
    """Compress frame to JPEG bytes."""
    small = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return buf.tobytes()


def handle_mute_command(mqtt_payload: dict, buzzer: ContinuousBuzzer):
    """Handle MQTT mute command."""
    mute_sec = mqtt_payload.get("mute_sec", 30)
    event_id = mqtt_payload.get("event_id")
    
    print(f"[APP] Mute command: {mute_sec}s (event_id: {event_id})")
    buzzer.mute(mute_sec)
    publish_muted_status(event_id)


def upload_fall_event_async(
    frame: np.ndarray,
    event_id: str,
    fall_prob: float,
    event_time: float = None,
):
    """
    Upload fall event to Supabase in background thread.
    - Capture and upload image
    - Insert event record with proper schema
    - Publish MQTT status
    """
    def _upload():
        try:
            if event_time is None:
                _event_time = time.time()
            else:
                _event_time = event_time
            
            # 1. Capture and upload image
            jpg_bytes = make_jpeg_bytes(frame, w=SNAP_W, h=SNAP_H, quality=JPEG_QUALITY)
            if jpg_bytes is None:
                print("[APP][WARN] Failed to create JPEG")
                return
            
            filename = f"{event_id}.jpg"
            image_url = upload_image_to_supabase(jpg_bytes, filename)
            if not image_url:
                print("[APP][WARN] Failed to upload image")
                return
            
            # 2. Insert event record with all schema fields
            insert_fall_event(
                event_id=event_id,
                image_url=image_url,
                state="ALARMING",
                image_path=filename,  # Local filename
                event_time=_event_time,
            )
            
            # 3. Publish MQTT status
            publish_fall_alarm(event_id)
            
            print(f"[APP] Fall event uploaded: {event_id}")
            
        except Exception as e:
            print(f"[APP][WARN] Upload error: {e}")
    
    thread = threading.Thread(target=_upload, daemon=True)
    thread.start()


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
    event_sent = False
    current_event_id = None

    # alarms
    buzzer = ContinuousBuzzer(pin=BUZZER_PIN, on_sec=0.15, off_sec=0.15)
    sim_alarm = SimA7680CAlarm(
        port=SIM_PORT,
        baud=SIM_BAUD,
        numbers=CALL_NUMBERS,
        ring_sec=RING_SEC,
        retry_pause_sec=5,
        send_sms_after_first_call=False,
        sms_text="Canh bao: Phat hien te nga!",
    )

    # MQTT handler
    def on_mute_callback(payload):
        handle_mute_command(payload, buzzer)
    
    mqtt_handler = init_mqtt(on_mute_callback)

    def start_alarms(event_id: str, frame_for_upload: np.ndarray, fall_prob: float, event_time: float):
        buzzer.start()
        sim_alarm.send_sms_to_all_once()
        sim_alarm.start()
        # Upload event in background (non-blocking)
        upload_fall_event_async(frame_for_upload, event_id, fall_prob, event_time)

    def stop_alarms():
        buzzer.stop()
        sim_alarm.stop()
        publish_safe_status()

    try:
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

                            fall_candidate = (fall_prob > 0.5 and posture_ok and dynamic_ok) or (fall_ratio >= FALL_RATIO_THRESHOLD and posture_ok)
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
                            event_sent = False
                            current_event_id = gen_event_id()
                            start_alarms(current_event_id, frame, fall_prob, now)
                    else:
                        state = State.NORMAL
                        state_ts = now

                elif state == State.ALARMING:
                    if upright_candidate:
                        state = State.RECOVERING
                        state_ts = now

                elif state == State.RECOVERING:
                    if upright_candidate:
                        if now - state_ts >= RECOVER_SEC:
                            stop_alarms()
                            state = State.NORMAL
                            state_ts = now
                            event_sent = False
                    else:
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

    finally:
        # cleanup
        stop_alarms()
        buzzer.cleanup()
        if mqtt_handler:
            mqtt_handler.stop()
        cap.release()
        cv2.destroyAllWindows()


#if __name__ == "__main__":
#    main()