import numpy as np

BASE_FEATURE_NAMES = [
    "torso_angle_deg",
    "hip_y_norm",
    "shoulder_y_norm",
    "center_y_norm",
    "bbox_w_norm",
    "bbox_h_norm",
    "bbox_aspect",
    "mean_visibility",
    "min_visibility",
]

TEMPORAL_FEATURE_NAMES = [
    "hip_v_norm",
    "shoulder_v_norm",
    "center_v_norm",
    "torso_angle_v_deg",
    "bbox_w_v_norm",
    "bbox_h_v_norm",
    "bbox_aspect_v",
]

FEATURE_NAMES = BASE_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES
BASE_FEATURE_COUNT = len(BASE_FEATURE_NAMES)


def _idx(landmark_enum):
    """Convert MediaPipe PoseLandmark enum to integer index."""
    return landmark_enum.value if hasattr(landmark_enum, "value") else int(landmark_enum)


def _safe_point(landmarks, idx, width, height):
    lm = landmarks[_idx(idx)]
    return np.array([lm.x * width, lm.y * height], dtype=np.float32), float(lm.visibility)


def extract_base_features(results, image_shape, mp_pose_module):
    """
    Extract posture features from one frame.

    The features describe:
    - current body posture
    - body bounding box
    - pose visibility / confidence

    Return None if pose is not detected.
    """
    if not results.pose_landmarks:
        return None

    height, width, _ = image_shape
    landmarks = results.pose_landmarks.landmark

    left_shoulder, vis_ls = _safe_point(
        landmarks, mp_pose_module.PoseLandmark.LEFT_SHOULDER, width, height
    )
    right_shoulder, vis_rs = _safe_point(
        landmarks, mp_pose_module.PoseLandmark.RIGHT_SHOULDER, width, height
    )
    left_hip, vis_lh = _safe_point(
        landmarks, mp_pose_module.PoseLandmark.LEFT_HIP, width, height
    )
    right_hip, vis_rh = _safe_point(
        landmarks, mp_pose_module.PoseLandmark.RIGHT_HIP, width, height
    )

    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    mid_hip = (left_hip + right_hip) / 2.0
    body_center = (mid_shoulder + mid_hip) / 2.0

    # Vector hip -> shoulder
    vec = mid_shoulder - mid_hip

    # Angle vs vertical: 0 = upright, ~90 = horizontal
    angle_rad = np.arctan2(abs(vec[0]), abs(vec[1]) + 1e-6)
    torso_angle_deg = float(np.degrees(angle_rad))

    hip_y_norm = float(mid_hip[1] / height)
    shoulder_y_norm = float(mid_shoulder[1] / height)
    center_y_norm = float(body_center[1] / height)

    xs_all = np.array([lm.x for lm in landmarks], dtype=np.float32) * width
    ys_all = np.array([lm.y for lm in landmarks], dtype=np.float32) * height
    vis_all = np.array([lm.visibility for lm in landmarks], dtype=np.float32)

    # Use visible landmarks for a more stable bbox.
    visible_mask = vis_all >= 0.30

    if np.sum(visible_mask) >= 5:
        xs = xs_all[visible_mask]
        ys = ys_all[visible_mask]
    else:
        xs = xs_all
        ys = ys_all

    bbox_w_norm = float((xs.max() - xs.min()) / width)
    bbox_h_norm = float((ys.max() - ys.min()) / height)
    bbox_aspect = float(bbox_w_norm / (bbox_h_norm + 1e-6))

    key_visibilities = [
        vis_ls,
        vis_rs,
        vis_lh,
        vis_rh,
    ]

    mean_visibility = float(np.mean(key_visibilities))
    min_visibility = float(np.min(key_visibilities))

    return [
        torso_angle_deg,
        hip_y_norm,
        shoulder_y_norm,
        center_y_norm,
        bbox_w_norm,
        bbox_h_norm,
        bbox_aspect,
        mean_visibility,
        min_visibility,
    ]


def extract_features(
    results,
    image_shape,
    mp_pose_module,
    prev_base_features=None,
    dt_sec=1.0,
):
    """
    Extract full feature vector for one sampled frame.

    Output:
    - base posture features
    - temporal features compared with previous detected frame
    """
    base = extract_base_features(results, image_shape, mp_pose_module)

    if base is None:
        return None

    if prev_base_features is None:
        temporal = [0.0 for _ in TEMPORAL_FEATURE_NAMES]
    else:
        prev = list(prev_base_features[:BASE_FEATURE_COUNT])
        dt = max(float(dt_sec or 1.0), 1e-6)

        hip_v_norm = float((base[1] - prev[1]) / dt)
        shoulder_v_norm = float((base[2] - prev[2]) / dt)
        center_v_norm = float((base[3] - prev[3]) / dt)
        torso_angle_v_deg = float((base[0] - prev[0]) / dt)
        bbox_w_v_norm = float((base[4] - prev[4]) / dt)
        bbox_h_v_norm = float((base[5] - prev[5]) / dt)
        bbox_aspect_v = float((base[6] - prev[6]) / dt)

        temporal = [
            hip_v_norm,
            shoulder_v_norm,
            center_v_norm,
            torso_angle_v_deg,
            bbox_w_v_norm,
            bbox_h_v_norm,
            bbox_aspect_v,
        ]

    return base + temporal


def motion_score(features):
    """
    Estimate body motion intensity from one feature vector.

    Higher score usually means stronger body movement.
    """
    f = dict(zip(FEATURE_NAMES, features))

    score = (
        abs(f.get("hip_v_norm", 0.0))
        + abs(f.get("shoulder_v_norm", 0.0))
        + abs(f.get("center_v_norm", 0.0))
        + abs(f.get("bbox_h_v_norm", 0.0))
        + abs(f.get("bbox_w_v_norm", 0.0))
        + abs(f.get("bbox_aspect_v", 0.0))
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
    1. enough recent frames/windows are predicted as fall
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