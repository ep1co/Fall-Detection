import numpy as np

FEATURE_NAMES = [
    "torso_angle_deg",
    "hip_y_norm",
    "shoulder_y_norm",
    "bbox_w_norm",
    "bbox_h_norm",
]

def extract_features(results, image_shape, mp_pose_module):
    """
    Extract features from MediaPipe Pose results.
    mp_pose_module = mp.solutions.pose (module)
    """
    if not results.pose_landmarks:
        return None

    height, width, _ = image_shape
    landmarks = results.pose_landmarks.landmark

    def get_point(idx):
        lm = landmarks[idx]
        return np.array([lm.x * width, lm.y * height], dtype=np.float32)

    # Key joints
    left_shoulder = get_point(mp_pose_module.PoseLandmark.LEFT_SHOULDER)
    right_shoulder = get_point(mp_pose_module.PoseLandmark.RIGHT_SHOULDER)
    left_hip = get_point(mp_pose_module.PoseLandmark.LEFT_HIP)
    right_hip = get_point(mp_pose_module.PoseLandmark.RIGHT_HIP)

    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    mid_hip = (left_hip + right_hip) / 2.0

    # Vector hip -> shoulder
    vec = mid_shoulder - mid_hip  # [dx, dy]

    # Angle vs vertical (0 upright, ~90 horizontal)
    angle_rad = np.arctan2(abs(vec[0]), abs(vec[1]) + 1e-6)
    torso_angle_deg = float(np.degrees(angle_rad))

    hip_y_norm = float(mid_hip[1] / height)
    shoulder_y_norm = float(mid_shoulder[1] / height)

    # Bounding box of all landmarks
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