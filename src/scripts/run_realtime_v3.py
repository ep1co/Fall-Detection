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


def verify_config():
    required_vars = [
        "SUPABASE_URL",
        "SUPABASE_SERVICE_KEY",
        "HIVEMQ_HOST",
        "HIVEMQ_USER",
    ]

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

from utils.pose_features_v3 import (
    extract_features,
    BASE_FEATURE_COUNT,
    FEATURE_NAMES,
    motion_score,
)

from alerts.buzzer import ContinuousBuzzer
from alerts.sim_a7680c import SimA7680CAlarm
from utils.cloud_uploader import (
    gen_event_id,
    upload_image_to_supabase,
    insert_fall_event,
)
from utils.mqtt_handler import (
    init_mqtt,
    publish_fall_alarm,
    publish_safe_status,
    publish_muted_status,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "fall_detector_rf_window_kfold_v4.pkl"

WINDOW_NAME = "AI Fall Detection - Realtime"
DISPLAY_W, DISPLAY_H = 640, 480
FALL_LABEL = 1

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


# ============================================================
# Prediction smoothing
# ============================================================
PREDICTION_WINDOW = 10
MOTION_WINDOW = 30

# ============================================================
# Window-level inference settings
# ============================================================
WINDOW_SEC = 2.0
WINDOW_MIN_FRAMES = 3

# Pose lost handling
POSE_LOST_GRACE_SEC = 1.0
POSE_LOST_AFTER_UPRIGHT_SEC = 2.5
POSE_REAPPEAR_LYING_SEC = 1.2
POSE_LOST_FLAG_MAX_SEC = 3.0

# Ngưỡng cho nhánh ngã nhanh
PROBA_FRAME_THRESHOLD = 0.55
FALL_RATIO_THRESHOLD = 0.60
MOTION_SCORE_THRESHOLD = 0.20

# Ngưỡng cho nhánh ngã chậm
SLOW_FALL_PROBA_THRESHOLD = 0.40
SLOW_FALL_RATIO_THRESHOLD = 0.40

# ============================================================
# Posture thresholds
# ============================================================
TORSO_ANGLE_THR = 50.0
ASPECT_THR = 1.20

# Dùng để ghi nhớ trạng thái đứng/ngồi trước đó
UPRIGHT_ANGLE_THR = 35.0
UPRIGHT_ASPECT_THR = 0.90

# Dùng để nhận biết trạng thái nằm/ngang hiện tại
LYING_ANGLE_THR = 55.0
LYING_ASPECT_THR = 1.20

# Nếu trong khoảng này từng thấy người đứng/ngồi,
# sau đó chuyển sang nằm/ngang thì có thể là ngã chậm
UPRIGHT_MEMORY_SEC = 4.0
LYING_CONFIRM_SEC = 1.2


# ============================================================
# Dynamic thresholds from new features
# ============================================================
HIP_V_THR = 0.55
SHOULDER_V_THR = 0.55
ANG_V_THR = 120.0
BBOX_H_V_THR = 0.50


# ============================================================
# State machine timing
# ============================================================
FAST_CONFIRM_SEC = 0.8
SLOW_CONFIRM_SEC = 0.5
RECOVER_SEC = 3.0


# ============================================================
# Alert
# ============================================================
BUZZER_PIN = 23
SIM_PORT = "/dev/serial0"
SIM_BAUD = 115200
CALL_NUMBERS = ["+84942826528"]
RING_SEC = 15


# ============================================================
# Snapshot
# ============================================================
SNAP_W, SNAP_H = 1280, 720  # thu thay cho 640x480
JPEG_QUALITY = 70


class State:
    NORMAL = "NORMAL"
    FALL_SUSPECT = "FALL_SUSPECT"
    ALARMING = "ALARMING"
    RECOVERING = "RECOVERING"


def make_jpeg_bytes(frame_bgr, w= SNAP_W, h= SNAP_H, quality=70): 
    """Compress frame to JPEG bytes."""
    small = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(
        ".jpg",
        small,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )

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
    """

    def _upload():
        try:
            _event_time = time.time() if event_time is None else event_time

            jpg_bytes = make_jpeg_bytes(
                frame,
                w=SNAP_W,
                h=SNAP_H,
                quality=JPEG_QUALITY,
            )

            if jpg_bytes is None:
                print("[APP][WARN] Failed to create JPEG")
                return

            filename = f"{event_id}.jpg"
            image_url = upload_image_to_supabase(jpg_bytes, filename)

            if not image_url:
                print("[APP][WARN] Failed to upload image")
                return

            insert_fall_event(
                event_id=event_id,
                image_url=image_url,
                state="ALARMING",
                image_path=filename,
                event_time=_event_time,
            )

            publish_fall_alarm(event_id)

            print(f"[APP] Fall event uploaded: {event_id}")

        except Exception as e:
            print(f"[APP][WARN] Upload error: {e}")

    thread = threading.Thread(target=_upload, daemon=True)
    thread.start()


def get_fall_probability(model, X):
    """
    Return probability of fall class.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]

        if hasattr(model, "classes_"):
            fall_indices = np.where(model.classes_ == FALL_LABEL)[0]
            if len(fall_indices) == 0:
                return 0.0
            fall_index = int(fall_indices[0])
        else:
            fall_index = 1

        return float(proba[fall_index])

    pred = int(model.predict(X)[0])
    return 1.0 if pred == FALL_LABEL else 0.0

META_COLUMNS = {
    "label",
    "video_label",
    "fold",
    "activity_type",
    "video_name",
    "window_start_sec",
    "window_end_sec",
}


def summarize_realtime_window(frame_feature_rows):
    """
    Convert recent frame-level features into one window-level feature vector.

    This must match summarize_window() in preprocess_v2.py:
    mean, std, min, max, range, first, last, delta for every frame feature.
    """
    arr = np.array(
        [[float(r[name]) for name in FEATURE_NAMES] for r in frame_feature_rows],
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

    stats["window_num_frames"] = int(len(frame_feature_rows))
    stats["window_duration_sec"] = float(
        frame_feature_rows[-1]["timestamp_sec"] - frame_feature_rows[0]["timestamp_sec"]
    )

    return stats


def make_model_input_from_window(frame_feature_rows, model_feature_names):
    """
    Build X with the exact feature order saved during training.
    """
    stats = summarize_realtime_window(frame_feature_rows)

    missing = [name for name in model_feature_names if name not in stats]
    if missing:
        raise RuntimeError(
            f"Missing realtime window features: {missing[:10]} "
            f"total_missing={len(missing)}"
        )

    X = np.array(
        [float(stats[name]) for name in model_feature_names],
        dtype=np.float32,
    ).reshape(1, -1)

    return X


def prune_old_realtime_rows(rows, now, window_sec):
    """
    Keep only rows inside the recent realtime window.
    """
    while rows and (now - rows[0]["timestamp_sec"]) > window_sec:
        rows.popleft()

def main():
    artifact = joblib.load(MODEL_PATH)

    if isinstance(artifact, dict):
        model = artifact["model"]
        model_feature_names = artifact["feature_names"]
        model_threshold = float(artifact.get("threshold", 0.50))
    else:
        # Fallback for old saved model format.
        model = artifact
        model_feature_names = None
        model_threshold = 0.50

    expected_features = getattr(model, "n_features_in_", None)

    print(f"[APP] Loaded model artifact: {MODEL_PATH}")
    print(f"[APP] Model expected window features: {expected_features}")
    print(f"[APP] Model threshold: {model_threshold}")
    print(f"[APP] Frame FEATURE_NAMES ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

    if model_feature_names is None:
        raise RuntimeError(
            "The loaded model does not contain feature_names. "
            "Please train with train_model.py v3 and save artifact = "
            "{'model': clf, 'feature_names': feature_names, 'threshold': threshold}."
        )

    print(f"[APP] Window feature count from artifact: {len(model_feature_names)}")

    if expected_features is not None and expected_features != len(model_feature_names):
        raise RuntimeError(
            f"Feature mismatch: model expects {expected_features}, "
            f"but artifact has {len(model_feature_names)} feature names."
        )
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, DISPLAY_H) 

    recent_probs = deque(maxlen=PREDICTION_WINDOW)
    recent_features = deque(maxlen=MOTION_WINDOW)
    recent_frame_rows = deque()

    # For temporal features
    prev_base_features = None
    prev_feature_time = None

    # For slow fall detection
    last_upright_time = None
    lying_start_time = None

    last_pose_seen_time = None
    pose_lost_start_time = None
    pose_lost_after_upright = False
    pose_lost_flag_time = None

    # State machine vars
    state = State.NORMAL
    state_ts = time.time()
    current_event_id = None
    suspect_mode = "NONE"

    # Alarms
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

    def start_alarms(
        event_id: str,
        frame_for_upload: np.ndarray,
        fall_prob: float,
        event_time: float,
    ):
        buzzer.start()

        sim_alarm.send_sms_to_all_once()
        sim_alarm.start()

        upload_fall_event_async(
            frame_for_upload,
            event_id,
            fall_prob,
            event_time,
        )

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
                fall_ratio_fast = 0.0
                fall_ratio_slow = 0.0
                max_motion = 0.0
                lying_duration = 0.0

                torso_angle = None
                bbox_aspect = None

                upright_now = False
                lying_now = False

                fast_fall_candidate = False
                slow_fall_candidate = False
                fall_candidate = False
                candidate_mode = "NONE"
                upright_candidate = False

                if results.pose_landmarks:
                    last_pose_seen_time = now
                    pose_lost_start_time = None
                    mp_draw.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )

                    if prev_feature_time is None:
                        dt_sec = 1.0 / 30.0
                    else:
                        dt_sec = max(1e-3, now - prev_feature_time)

                    feats = extract_features(
                        results=results,
                        image_shape=frame.shape,
                        mp_pose_module=mp_pose,
                        prev_base_features=prev_base_features,
                        dt_sec=dt_sec,
                    )

                    if feats is not None:
                        prev_base_features = feats[:BASE_FEATURE_COUNT]
                        prev_feature_time = now

                        # Store current frame features for window-level inference.
                        frame_feature_row = {
                            name: float(value)
                            for name, value in zip(FEATURE_NAMES, feats)
                        }
                        frame_feature_row["timestamp_sec"] = float(now)

                        recent_frame_rows.append(frame_feature_row)
                        prune_old_realtime_rows(
                            rows=recent_frame_rows,
                            now=now,
                            window_sec=WINDOW_SEC,
                        )

                        # Only predict when we have enough samples in the realtime window.
                        if len(recent_frame_rows) >= WINDOW_MIN_FRAMES:
                            X = make_model_input_from_window(
                                frame_feature_rows=list(recent_frame_rows),
                                model_feature_names=model_feature_names,
                            )

                            if expected_features is not None and X.shape[1] != expected_features:
                                raise RuntimeError(
                                    f"Realtime window feature mismatch: X has {X.shape[1]} features, "
                                    f"model expects {expected_features}."
                                )

                            fall_prob = get_fall_probability(model, X)
                        else:
                            fall_prob = 0.0

                        recent_probs.append(fall_prob)
                        recent_features.append(feats)

                        probs_arr = np.array(recent_probs, dtype=np.float32)

                        fall_ratio_fast = float(
                            np.mean(probs_arr >= max(PROBA_FRAME_THRESHOLD, model_threshold))
                        )

                        fall_ratio_slow = float(
                            np.mean(probs_arr >= SLOW_FALL_PROBA_THRESHOLD)
                        )

                        max_motion = max(motion_score(f) for f in recent_features)

                        # --------------------
                        # Current features
                        # --------------------
                        f = dict(zip(FEATURE_NAMES, feats))

                        torso_angle = float(f["torso_angle_deg"])
                        bbox_aspect = float(f["bbox_aspect"])

                        hip_v = float(f["hip_v_norm"])
                        shoulder_v = float(f["shoulder_v_norm"])
                        angle_v = float(f["torso_angle_v_deg"])
                        bbox_h_v = float(f["bbox_h_v_norm"])

                        # --------------------
                        # Posture states
                        # --------------------
                        upright_now = (
                            torso_angle < UPRIGHT_ANGLE_THR
                            and bbox_aspect < UPRIGHT_ASPECT_THR
                        )

                        lying_now = (
                            torso_angle > LYING_ANGLE_THR
                            or bbox_aspect > LYING_ASPECT_THR
                        )

                        if upright_now:
                            last_upright_time = now
                            lying_start_time = None
                            pose_lost_after_upright = False
                            pose_lost_flag_time = None

                        elif lying_now:
                            if lying_start_time is None:
                                lying_start_time = now

                        else:
                            lying_start_time = None

                        if lying_start_time is not None:
                            lying_duration = now - lying_start_time

                        had_recent_upright = (
                            last_upright_time is not None
                            and (now - last_upright_time) <= UPRIGHT_MEMORY_SEC
                        )

                        # --------------------
                        # Fast fall branch
                        # --------------------
                        posture_ok = (
                            torso_angle > TORSO_ANGLE_THR
                            or bbox_aspect > ASPECT_THR
                        )

                        dynamic_ok = (
                            abs(hip_v) > HIP_V_THR
                            or abs(shoulder_v) > SHOULDER_V_THR
                            or abs(angle_v) > ANG_V_THR
                            or abs(bbox_h_v) > BBOX_H_V_THR
                            or max_motion > MOTION_SCORE_THRESHOLD
                        )

                        fast_fall_candidate = (
                            fall_ratio_fast >= FALL_RATIO_THRESHOLD
                            and posture_ok
                            and dynamic_ok
                        )

                        # --------------------
                        # Slow fall branch
                        # --------------------
                        slow_prob_ok = (
                            fall_prob >= SLOW_FALL_PROBA_THRESHOLD
                            or fall_ratio_slow >= SLOW_FALL_RATIO_THRESHOLD
                        )

                        normal_slow_fall = (
                            had_recent_upright
                            and lying_now
                            and lying_duration >= LYING_CONFIRM_SEC
                            and slow_prob_ok
                        )

                        pose_lost_flag_valid = (
                            pose_lost_after_upright
                            and pose_lost_flag_time is not None
                            and (now - pose_lost_flag_time) <= POSE_LOST_FLAG_MAX_SEC
                        )

                        pose_lost_then_lying = (
                            pose_lost_after_upright
                            and lying_now
                            and lying_duration >= POSE_REAPPEAR_LYING_SEC
                        )

                        slow_fall_candidate = normal_slow_fall or pose_lost_then_lying

                        if fast_fall_candidate:
                            fall_candidate = True
                            candidate_mode = "FAST"

                        elif slow_fall_candidate:
                            fall_candidate = True
                            candidate_mode = "SLOW"

                        # Recover only when body is clearly upright again
                        upright_candidate = (
                            torso_angle < 25.0
                            and bbox_aspect < 0.8
                        )
                else:
                    # Pose lost: this can happen during a fast fall
                    # due to motion blur, occlusion, or body moving out of view.
                    if pose_lost_start_time is None:
                        pose_lost_start_time = now

                    pose_lost_duration = now - pose_lost_start_time

                    had_recent_upright_before_lost = (
                        last_upright_time is not None
                        and (pose_lost_start_time - last_upright_time) <= POSE_LOST_AFTER_UPRIGHT_SEC
                    )

                    if had_recent_upright_before_lost and pose_lost_duration <= POSE_LOST_GRACE_SEC:
                        pose_lost_after_upright = True
                        pose_lost_flag_time = now


                # ============================================================
                # State machine
                # ============================================================
                if state == State.NORMAL:
                    if fall_candidate:
                        state = State.FALL_SUSPECT
                        state_ts = now
                        suspect_mode = candidate_mode

                elif state == State.FALL_SUSPECT:
                    if fall_candidate:
                        suspect_mode = candidate_mode

                        confirm_sec = (
                            FAST_CONFIRM_SEC
                            if suspect_mode == "FAST"
                            else SLOW_CONFIRM_SEC
                        )

                        if now - state_ts >= confirm_sec:
                            state = State.ALARMING
                            state_ts = now
                            current_event_id = gen_event_id()
                            pose_lost_after_upright = False

                            start_alarms(
                                current_event_id,
                                frame.copy(),
                                fall_prob,
                                now,
                            )
                    else:
                        state = State.NORMAL
                        state_ts = now
                        suspect_mode = "NONE"

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
                            current_event_id = None
                            suspect_mode = "NONE"

                            recent_probs.clear()
                            recent_features.clear()
                            recent_frame_rows.clear()
                            pose_lost_after_upright = False
                            pose_lost_start_time = None
                    else:
                        state = State.ALARMING
                        state_ts = now

                # ============================================================
                # UI overlay
                # ============================================================
                status_text = state
                status_color = (0, 255, 0)

                if state == State.FALL_SUSPECT:
                    status_color = (0, 165, 255)

                if state in (State.ALARMING, State.RECOVERING):
                    status_color = (0, 0, 255)

                h, w, _ = frame.shape
                cv2.rectangle(frame, (0, 0), (w, 135), (0, 0, 0), thickness=-1)

                cv2.putText(
                    frame,
                    f"{status_text} mode={suspect_mode}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    status_color,
                    2,
                )

                cv2.putText(
                    frame,
                    f"p={fall_prob:.2f} fast_ratio={fall_ratio_fast:.2f} slow_ratio={fall_ratio_slow:.2f}",
                    (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"motion={max_motion:.2f} lying_dur={lying_duration:.1f}s",
                    (10, 98),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    2,
                )

                if torso_angle is not None and bbox_aspect is not None:
                    cv2.putText(
                        frame,
                        f"angle={torso_angle:.1f} aspect={bbox_aspect:.2f} upright={upright_now} lying={lying_now}",
                        (10, 126),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (255, 255, 255),
                        2,
                    )

                cv2.imshow(WINDOW_NAME, frame)

                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    finally:
        stop_alarms()
        buzzer.cleanup()

        if mqtt_handler:
            mqtt_handler.stop()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()