# Tập trung toàn bộ cấu hình vào đây
# Khi cần chỉnh thì chỉ sửa 1 file

PHONE_NUMBER = "+84332785126"
BUZZER_PIN = 23
FRAME_BUFFER_SIZE = 15
FALL_VELOCITY_THRESHOLD = 0.08
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "fall_detector_rf_test1.pkl"