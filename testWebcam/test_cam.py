import cv2
import numpy as np
import time
import mediapipe as mp
import tensorflow as tf
from collections import deque
#from ai_edge_litert.compiled_model import CompiledModel

# =========================
# CONFIG
# =========================
MODEL_PATH = "fall_multitask_builtins_fp32.tflite"
POSE_TASK_PATH = "pose_landmarker_lite.task"

WINDOW_SEC = 10
FPS_TARGET = 20               # fps "mong muốn" cho buffer; thực tế sẽ đo và cập nhật
INFER_HZ = 5                  # infer 5 lần/giây (giảm tải). Tăng lên nếu máy mạnh
THRESH_FALL = 0.8

T = WINDOW_SEC * FPS_TARGET
F = 99  # 33 * (x,y,visibility)

# =========================
# Load TFlite model (CPU fallback is OK)
# =========================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =========================
# MediaPipe Pose Landmarker (VIDEO mode - synchronous)
# =========================
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_TASK_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
)

# =========================
# Sliding buffer
# =========================
buffer = deque(maxlen=T)
zero_vec = np.zeros((F,), dtype=np.float32)

def normalize_landmarks(lm):
    """lm: list of 33 landmarks"""
    L_SHO, R_SHO = 11, 12
    L_HIP, R_HIP = 23, 24

    feats = []
    for p in lm:
        feats.extend([p.x, p.y, getattr(p, "visibility", 0.0)])
    feats = np.array(feats, dtype=np.float32).reshape(33, 3)

    center = 0.5 * (feats[L_HIP, :2] + feats[R_HIP, :2])
    scale = np.linalg.norm(feats[L_SHO, :2] - feats[R_SHO, :2])
    if scale < 1e-6:
        # fallback: hip width
        scale = np.linalg.norm(feats[L_HIP, :2] - feats[R_HIP, :2])
    if scale < 1e-6:
        scale = 1.0

    feats[:, :2] = (feats[:, :2] - center) / scale
    return feats.reshape(-1)

# =========================
# Webcam
# =========================

cap = cv2.VideoCapture(0)

# (Khuyến nghị) đặt độ phân giải thấp để tăng tốc
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# FPS đo thực tế
fps_est = FPS_TARGET
last_frame_t = time.time()
fps_smooth = 0.9  # EMA smoothing

# inference scheduling
infer_period = 1.0 / INFER_HZ
last_infer_t = 0.0

# timestamp cho VIDEO mode (phải tăng dần)
t0 = time.time()

p_fall_last = 0.0
fall_time_last = None

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read webcam frame.")
            break

        now = time.time()

        # Update FPS estimate (EMA)
        dt = now - last_frame_t
        last_frame_t = now
        if dt > 1e-6:
            inst_fps = 1.0 / dt
            fps_est = fps_smooth * fps_est + (1.0 - fps_smooth) * inst_fps

        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # VIDEO mode timestamp in ms must be monotonic increasing
        timestamp_ms = int((now - t0) * 1000)

        # Pose inference (synchronous)
        res = landmarker.detect_for_video(mp_image, timestamp_ms)

        if res.pose_landmarks:
            vec = normalize_landmarks(res.pose_landmarks[0])
            buffer.append(vec)
        else:
            buffer.append(zero_vec)

        # LiteRT inference only when buffer full and enough time passed
        if len(buffer) == T and (now - last_infer_t) >= infer_period:
            last_infer_t = now

            X = np.array(buffer, dtype=np.float32)          # (T,F)
            X = np.expand_dims(X, axis=0)                   # (1,T,F)

            interpreter.allocate_tensors()
            interpreter.set_tensor(input_details[0]['index'], X)
            interpreter.invoke()

            p_fall_last = float(interpreter.get_tensor(output_details[0]['index']).flatten()[0])
            p_time = interpreter.get_tensor(output_details[1]['index']).flatten()

            fall_frame = int(np.argmax(p_time))

            # Convert to seconds using measured fps (more robust than FPS_TARGET)
            fps_used = max(1.0, float(fps_est))
            fall_time_last = fall_frame / fps_used

        # Overlay info
        text1 = f"FPS ~ {fps_est:.1f} | p_fall={p_fall_last:.2f}"
        cv2.putText(frame, text1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if fall_time_last is not None and p_fall_last >= THRESH_FALL:
            text2 = f"FALL at ~{fall_time_last:.2f}s (in window)"
            cv2.putText(frame, text2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif fall_time_last is not None:
            text2 = f"time_peak ~{fall_time_last:.2f}s"
            cv2.putText(frame, text2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Fall Detection (Webcam)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()