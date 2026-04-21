from pathlib import Path
import csv

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


# --------------------
# Paths
# --------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_FILE = PROCESSED_DIR / "features.csv"

MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "fall_detector_rf_test1.pkl"


# --------------------
# Load dataset
# --------------------
def load_dataset():
    X = []
    y = []

    with open(FEATURES_FILE, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        # Expecting columns:
        # torso_angle_deg, hip_y_norm, shoulder_y_norm, bbox_w_norm,
        # bbox_h_norm, label, video_name, frame_idx
        for row in reader:
            if not row:
                continue

            # First 5 columns are features
            features = list(map(float, row[0:5]))
            label = int(row[5])

            X.append(features)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"[INFO] Loaded dataset from {FEATURES_FILE}")
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
    return X, y


# --------------------
# Train model
# --------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=100, # 4/3: thu 100 thay vi 200
        max_depth=15, # 4/3: thu 15 thay vi none
        random_state=42,
        n_jobs=-1,
        class_weight='balanced' # 4/3: them thu
        min_sample_leaf=5 # 17/4: them thu, giam overfitting, tang do cham cua model
    )

    print("[INFO] Training RandomForest model...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Test accuracy: {acc:.3f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["no_fall", "fall"]))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return clf


def main():
    X, y = load_dataset()
    clf = train_model(X, y)

    joblib.dump(clf, MODEL_PATH)
    print(f"\n[OK] Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
