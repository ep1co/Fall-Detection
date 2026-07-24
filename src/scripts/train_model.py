from pathlib import Path
import csv
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)
import joblib


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

FEATURES_FILE = PROCESSED_DIR / "window_features_kfold_v3.csv"

MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "fall_detector_rf_window_kfold_v3.pkl"

K = 5
DEFAULT_THRESHOLD = 0.50

META_COLUMNS = {
    "label",
    "video_label",
    "fold",
    "activity_type",
    "video_name",
    "window_start_sec",
    "window_end_sec",
}


def load_dataset():
    X = []
    y = []
    folds = []
    activity_types = []
    video_names = []
    feature_names = None

    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        all_columns = reader.fieldnames
        if all_columns is None:
            raise RuntimeError(f"Empty CSV file: {FEATURES_FILE}")

        feature_names = [
            col for col in all_columns
            if col not in META_COLUMNS
        ]

        for row in reader:
            features = [float(row[name]) for name in feature_names]
            label = int(row["label"])

            X.append(features)
            y.append(label)
            folds.append(row["fold"])
            activity_types.append(row["activity_type"])
            video_names.append(row["video_name"])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    folds = np.array(folds)
    activity_types = np.array(activity_types)
    video_names = np.array(video_names)

    print(f"[INFO] Loaded dataset from {FEATURES_FILE}")
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
    print(f"[INFO] Number of features: {len(feature_names)}")
    print(f"[INFO] Label distribution: {Counter(y)}")
    print(f"[INFO] Fold distribution: {Counter(folds)}")
    print(f"[INFO] Activity distribution: {Counter(activity_types)}")
    print(f"[INFO] Number of videos: {len(set(video_names))}")

    return X, y, folds, activity_types, video_names, feature_names


def make_model():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        min_samples_split=6,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


def fold_sort_key(name):
    try:
        return int(name.split("_")[-1])
    except Exception:
        return name


def safe_rate(num, den):
    return float(num / den) if den > 0 else 0.0


def predict_with_threshold(clf, X, threshold):
    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(np.int64)
    return pred, proba


def evaluate_window_level(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = safe_rate(tn, tn + fp)
    false_alarm_rate = safe_rate(fp, fp + tn)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_alarm_rate": false_alarm_rate,
        "confusion_matrix": cm,
    }


def evaluate_video_level(y_true, y_pred, video_names):
    """
    Video-level evaluation:
    - A video is predicted fall if any of its windows is predicted fall.
    - Useful for estimating false alarm per ADL video.
    """
    grouped_true = defaultdict(list)
    grouped_pred = defaultdict(list)

    for true_label, pred_label, video_name in zip(y_true, y_pred, video_names):
        grouped_true[video_name].append(int(true_label))
        grouped_pred[video_name].append(int(pred_label))

    video_true = []
    video_pred = []

    for video_name in sorted(grouped_true):
        # If any window is true fall, video is treated as fall.
        vt = 1 if any(v == 1 for v in grouped_true[video_name]) else 0

        # If any window is predicted fall, video is treated as predicted fall.
        vp = 1 if any(v == 1 for v in grouped_pred[video_name]) else 0

        video_true.append(vt)
        video_pred.append(vp)

    video_true = np.array(video_true, dtype=np.int64)
    video_pred = np.array(video_pred, dtype=np.int64)

    cm = confusion_matrix(video_true, video_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "num_videos": len(video_true),
        "video_fall_recall": safe_rate(tp, tp + fn),
        "video_adl_false_alarm_rate": safe_rate(fp, fp + tn),
        "video_confusion_matrix": cm,
    }


def print_feature_importance(clf, feature_names, top_k=20):
    importances = clf.feature_importances_
    pairs = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"\nTop {top_k} feature importances:")
    for name, value in pairs[:top_k]:
        print(f"  {name}: {value:.5f}")


def train_kfold(X, y, folds, activity_types, video_names, feature_names):
    unique_folds = sorted(set(folds), key=fold_sort_key)

    if len(unique_folds) != K:
        print(f"[WARN] Expected {K} folds, but found {len(unique_folds)}: {unique_folds}")

    results = []

    for val_fold in unique_folds:
        print("\n" + "=" * 80)
        print(f"[K-FOLD] Validation fold: {val_fold}")
        print("=" * 80)

        train_mask = folds != val_fold
        val_mask = folds == val_fold

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        activity_val = activity_types[val_mask]
        video_val = video_names[val_mask]

        print(f"[INFO] Train windows: {len(y_train)}")
        print(f"[INFO] Val windows:   {len(y_val)}")
        print(f"[INFO] Train label distribution: {Counter(y_train)}")
        print(f"[INFO] Val label distribution:   {Counter(y_val)}")
        print(f"[INFO] Val activity distribution: {Counter(activity_val)}")
        print(f"[INFO] Val videos: {len(set(video_val))}")

        if len(set(y_train)) < 2 or len(set(y_val)) < 2:
            print("[WARN] This fold does not contain both classes. Result may be unreliable.")

        clf = make_model()

        print("[INFO] Training RandomForest model...")
        clf.fit(X_train, y_train)

        y_pred, y_proba = predict_with_threshold(
            clf,
            X_val,
            threshold=DEFAULT_THRESHOLD,
        )

        metrics = evaluate_window_level(y_val, y_pred)
        video_metrics = evaluate_video_level(y_val, y_pred, video_val)

        print(f"\n[RESULT] Window-level metrics on {val_fold}")
        print(f"Accuracy:          {metrics['accuracy']:.3f}")
        print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
        print(f"Precision:         {metrics['precision']:.3f}")
        print(f"Fall recall:       {metrics['recall']:.3f}")
        print(f"F1-score:          {metrics['f1']:.3f}")
        print(f"Specificity:       {metrics['specificity']:.3f}")
        print(f"False alarm rate:  {metrics['false_alarm_rate']:.3f}")

        print("\nClassification report:")
        print(
            classification_report(
                y_val,
                y_pred,
                labels=[0, 1],
                target_names=["ADL/no_fall", "fall"],
                zero_division=0,
            )
        )

        print("Window-level confusion matrix [labels: 0=ADL/no_fall, 1=fall]:")
        print(metrics["confusion_matrix"])

        print(f"\n[RESULT] Video-level metrics on {val_fold}")
        print(f"Videos:                      {video_metrics['num_videos']}")
        print(f"Video fall recall:           {video_metrics['video_fall_recall']:.3f}")
        print(f"Video ADL false alarm rate:  {video_metrics['video_adl_false_alarm_rate']:.3f}")
        print("Video-level confusion matrix [labels: 0=ADL/no_fall, 1=fall]:")
        print(video_metrics["video_confusion_matrix"])

        print_feature_importance(clf, feature_names, top_k=15)

        results.append(
            {
                "fold": val_fold,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision": metrics["precision"],
                "fall_recall": metrics["recall"],
                "f1": metrics["f1"],
                "specificity": metrics["specificity"],
                "window_false_alarm_rate": metrics["false_alarm_rate"],
                "video_fall_recall": video_metrics["video_fall_recall"],
                "video_adl_false_alarm_rate": video_metrics["video_adl_false_alarm_rate"],
            }
        )

    print("\n" + "=" * 80)
    print("[SUMMARY] K-fold results")
    print("=" * 80)

    for r in results:
        print(
            f"{r['fold']}: "
            f"acc={r['accuracy']:.3f}, "
            f"bal_acc={r['balanced_accuracy']:.3f}, "
            f"precision={r['precision']:.3f}, "
            f"recall={r['fall_recall']:.3f}, "
            f"f1={r['f1']:.3f}, "
            f"specificity={r['specificity']:.3f}, "
            f"win_FA={r['window_false_alarm_rate']:.3f}, "
            f"video_recall={r['video_fall_recall']:.3f}, "
            f"video_FA={r['video_adl_false_alarm_rate']:.3f}"
        )

    def mean_metric(name):
        return float(np.mean([r[name] for r in results]))

    print("\n[AVERAGE]")
    print(f"Accuracy mean:                 {mean_metric('accuracy'):.3f}")
    print(f"Balanced accuracy mean:        {mean_metric('balanced_accuracy'):.3f}")
    print(f"Precision mean:                {mean_metric('precision'):.3f}")
    print(f"Fall recall mean:              {mean_metric('fall_recall'):.3f}")
    print(f"F1-score mean:                 {mean_metric('f1'):.3f}")
    print(f"Specificity mean:              {mean_metric('specificity'):.3f}")
    print(f"Window false alarm mean:       {mean_metric('window_false_alarm_rate'):.3f}")
    print(f"Video fall recall mean:        {mean_metric('video_fall_recall'):.3f}")
    print(f"Video ADL false alarm mean:    {mean_metric('video_adl_false_alarm_rate'):.3f}")

    return results


def train_final_model(X, y, feature_names):
    print("\n" + "=" * 80)
    print("[FINAL] Training final model on all window-level data")
    print("=" * 80)

    clf = make_model()
    clf.fit(X, y)

    artifact = {
        "model": clf,
        "feature_names": feature_names,
        "threshold": DEFAULT_THRESHOLD,
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"[OK] Saved final model artifact to {MODEL_PATH}")

    return clf


def main():
    X, y, folds, activity_types, video_names, feature_names = load_dataset()

    train_kfold(
        X=X,
        y=y,
        folds=folds,
        activity_types=activity_types,
        video_names=video_names,
        feature_names=feature_names,
    )

    train_final_model(X, y, feature_names)


if __name__ == "__main__":
    main()