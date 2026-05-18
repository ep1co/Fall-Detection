import numpy as np

BASE_FEATURE_NAMES = [
    "torso_angle_deg",
    "hip_y_norm",
    "shoulder_y_norm",
    "bbox_w_norm",
    "bbox_h_norm",
    "bbox_aspect",
]

TEMPORAL_FEATURE_NAMES = [
    "hip_v_norm",
    "shoulder_v_norm",
    "torso_angle_v_deg",
    "bbox_h_v_norm",
]

FEATURE_NAMES = BASE_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES
BASE_FEATURE_COUNT = len(BASE_FEATURE_NAMES)


def _idx(landmark_enum):
    """Convert MediaPipe PoseLandmark enum to integer index."""
    return landmark_enum.value if hasattr(landmark_enum, "value") else int(landmark_enum)


def extract_base_features(results, image_shape, mp_pose_module):
    """
    Extract posture features from one frame.

    These features describe the current body posture:
    - torso angle
    - hip/shoulder vertical position
    - body bounding box size
    - bounding box aspect ratio
    """
    if not results.pose_landmarks:
        return None

    height, width, _ = image_shape
    landmarks = results.pose_landmarks.landmark

    def get_point(idx):
        lm = landmarks[_idx(idx)]
        return np.array([lm.x * width, lm.y * height], dtype=np.float32)

    left_shoulder = get_point(mp_pose_module.PoseLandmark.LEFT_SHOULDER)
    right_shoulder = get_point(mp_pose_module.PoseLandmark.RIGHT_SHOULDER)
    left_hip = get_point(mp_pose_module.PoseLandmark.LEFT_HIP)
    right_hip = get_point(mp_pose_module.PoseLandmark.RIGHT_HIP)

    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    mid_hip = (left_hip + right_hip) / 2.0

    # Vector hip -> shoulder
    vec = mid_shoulder - mid_hip

    # Angle vs vertical: 0 = upright, ~90 = horizontal
    angle_rad = np.arctan2(abs(vec[0]), abs(vec[1]) + 1e-6)
    torso_angle_deg = float(np.degrees(angle_rad))

    hip_y_norm = float(mid_hip[1] / height)
    shoulder_y_norm = float(mid_shoulder[1] / height)

    xs = np.array([lm.x for lm in landmarks], dtype=np.float32) * width
    ys = np.array([lm.y for lm in landmarks], dtype=np.float32) * height

    bbox_w_norm = float((xs.max() - xs.min()) / width)
    bbox_h_norm = float((ys.max() - ys.min()) / height)
    bbox_aspect = float(bbox_w_norm / (bbox_h_norm + 1e-6))

    return [
        torso_angle_deg,
        hip_y_norm,
        shoulder_y_norm,
        bbox_w_norm,
        bbox_h_norm,
        bbox_aspect,
    ]


def extract_features(
    results,
    image_shape,
    mp_pose_module,
    prev_base_features=None,
    dt_sec=1.0,
):
    """
    Extract full feature vector.

    The output contains:
    1. posture features of the current frame
    2. temporal/motion features compared with the previous detected frame

    The temporal features help distinguish:
    - fall: sudden posture/motion change
    - resting/lying down: low or controlled motion
    """
    base = extract_base_features(results, image_shape, mp_pose_module)
    if base is None:
        return None

    if prev_base_features is None:
        temporal = [0.0, 0.0, 0.0, 0.0]
    else:
        prev = list(prev_base_features[:BASE_FEATURE_COUNT])
        dt = max(float(dt_sec or 1.0), 1e-6)

        hip_v_norm = float((base[1] - prev[1]) / dt)
        shoulder_v_norm = float((base[2] - prev[2]) / dt)
        torso_angle_v_deg = float((base[0] - prev[0]) / dt)
        bbox_h_v_norm = float((base[4] - prev[4]) / dt)

        temporal = [
            hip_v_norm,
            shoulder_v_norm,
            torso_angle_v_deg,
            bbox_h_v_norm,
        ]

    return base + temporal


def motion_score(features):
    """
    Estimate how much body motion exists in one feature vector.

    This is useful in realtime post-processing:
    lying/resting usually has low motion,
    while falling usually has sudden motion.
    """
    f = dict(zip(FEATURE_NAMES, features))

    score = (
        abs(f.get("hip_v_norm", 0.0))
        + abs(f.get("shoulder_v_norm", 0.0))
        + abs(f.get("bbox_h_v_norm", 0.0))
        + abs(f.get("torso_angle_v_deg", 0.0)) / 180.0
    )

    return float(score)


def should_trigger_alarm(
    prob_window,
    feature_window,
    prob_threshold=0.5,
    fall_ratio_threshold=0.6,
    motion_score_threshold=0.20,
):
    """
    Realtime alarm confirmation helper.

    Alarm is confirmed only when:
    1. enough recent frames are predicted as fall
    2. recent body motion is large enough

    This helps suppress false alarms when a person is already lying/resting.
    """
    if not prob_window or not feature_window:
        return False

    probs = np.array(prob_window, dtype=np.float32)
    fall_ratio = float(np.mean(probs >= prob_threshold))

    max_motion = max(motion_score(feats) for feats in feature_window)

    return (
        fall_ratio >= fall_ratio_threshold
        and max_motion >= motion_score_threshold
    )