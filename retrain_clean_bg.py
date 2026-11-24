import os
import joblib
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_DIR = "processed_dataset"
META_CSV = os.path.join(DATA_DIR, "dataset_metadata.csv")
MODEL_PATH = os.path.join("models", "svm.joblib")
TARGET_SR = 22050

# Loại bỏ các lớp ESC-50 có xung đột trực tiếp với tiếng chó sủa trầm qua loa nhỏ
CONFLICT_BG_CLASSES = {
    "ESC50_coughing", "ESC50_sneezing", "ESC50_snoring",
    "ESC50_door_wood_knock", "ESC50_clapping", "ESC50_footsteps",
    "ESC50_can_opening", "ESC50_engine", "ESC50_chainsaw",
    "ESC50_hand_saw", "ESC50_thunderstorm", "ESC50_fireworks",
    "ESC50_pig", "ESC50_cow", "ESC50_sheep", "ESC50_frog"
}

def extract_78_features(y, sr=TARGET_SR):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])
    return np.concatenate([np.mean(combined, axis=1), np.std(combined, axis=1)])

def main():
    df_meta = pd.read_csv(META_CSV)
    # Giữ toàn bộ dog, cat và chỉ giữ các lớp background môi trường sạch
    df_filtered = df_meta[~df_meta["source"].isin(CONFLICT_BG_CLASSES)].copy()

    print("Phân bố dữ liệu sau khi loại bỏ các lớp background gây nhầm với tiếng chó:")
    print(pd.crosstab(df_filtered["split"], df_filtered["label"], margins=True))

    X_list, y_list, groups_list, splits_list = [], [], [], []
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
        y_audio, _ = librosa.load(row["filepath"], sr=TARGET_SR, mono=True)
        X_list.append(extract_78_features(y_audio, sr=TARGET_SR))
        y_list.append(row["label"])
        groups_list.append(row["group_id"])
        splits_list.append(row["split"])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    groups = np.array(groups_list)
    splits = np.array(splits_list)

    train_mask = splits == "train"
    val_mask = splits == "val"

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)),
    ])

    param_grid = {
        "svm__C": [1, 5, 10, 20],
        "svm__gamma": ["scale", 0.005, 0.01],
    }

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True)
    grid.fit(X[train_mask], y[train_mask], groups=groups[train_mask])

    best_model = grid.best_estimator_
    joblib.dump(best_model, MODEL_PATH)

    y_pred = best_model.predict(X[val_mask])
    print(f"\nBest Params: {grid.best_params_}")
    print(f"Validation Accuracy: {accuracy_score(y[val_mask], y_pred)*100:.2f}% | Macro-F1: {f1_score(y[val_mask], y_pred, average='macro'):.4f}")
    print(classification_report(y[val_mask], y_pred, digits=4))

if __name__ == "__main__":
    main()