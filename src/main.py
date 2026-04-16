import threading
import cv2
from config import *
from modules.camera import Camera
from modules.pose_estimator import PoseEstimator
from modules.fall_detector import FallDetector
from modules.buzzer import Buzzer
from modules.sim_alert import SimAlert

def on_fall_detected():
    """Chạy trong thread riêng để không block inference"""
    def alert():
        print("[ALERT] Fall detected!")
        buzzer.beep(times=5)
        sim.send_sms(PHONE_NUMBER, "CANH BAO: Phat hien te nga!")
        sim.make_call(PHONE_NUMBER, duration=20)

    threading.Thread(target=alert).start()

# Khởi tạo các module
camera    = Camera(CAMERA_WIDTH, CAMERA_HEIGHT)
estimator = PoseEstimator(MODEL_PATH)
detector  = FallDetector()
buzzer    = Buzzer(BUZZER_PIN)
sim       = SimAlert()

print("System started. Press Q to quit.")

while True:
    frame = camera.read()
    if frame is None:
        continue

    # Pose estimation
    keypoints = estimator.get_keypoints(frame)

    # Fall detection
    if detector.update(keypoints):
        on_fall_detected()

    # Hiển thị (tùy chọn)
    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
camera.release()
buzzer.cleanup()
cv2.destroyAllWindows()