import os
import joblib
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ
# ==========================================
DATA_DIR = "processed_dataset"
META_CSV = os.path.join(DATA_DIR, "dataset_metadata.csv")
MODEL_DIR = "models"
REPORT_DIR = "reports"

TARGET_SR = 22050
N_MFCC = 13

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ==========================================
# 2. HÀM TRÍCH XUẤT 78 ĐẶC TRƯNG MFCC
# ==========================================
def extract_78_features(y, sr=TARGET_SR):
    """
    Trích xuất 78 đặc trưng theo đúng đề cương:
    13 MFCC + 13 Delta + 13 Delta-Delta -> lấy mean và std -> 39 * 2 = 78 chiều.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Ghép thành ma trận (39, số_khung_thời_gian)
    combined = np.vstack([mfcc, delta, delta2])

    feat_mean = np.mean(combined, axis=1)
    feat_std = np.std(combined, axis=1)

    # Vector 1D gồm 78 đặc trưng
    return np.concatenate([feat_mean, feat_std])

def load_features_from_metadata(df_meta):
    X_list, y_list, groups_list, splits_list, filenames_list = [], [], [], [], []

    print("Đang trích xuất 78 đặc trưng (MFCC + Delta + Delta-Delta) cho toàn bộ 2.364 file...")
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
        fpath = row["filepath"]
        y_audio, _ = librosa.load(fpath, sr=TARGET_SR, mono=True)
        features = extract_78_features(y_audio, sr=TARGET_SR)

        X_list.append(features)
        y_list.append(row["label"])
        groups_list.append(row["group_id"])
        splits_list.append(row["split"])
        filenames_list.append(row["filename"])

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list),
        np.array(groups_list),
        np.array(splits_list),
        np.array(filenames_list),
    )

# ==========================================
# 3. CHIA FOLD TUNING & HUẤN LUYỆN SVM
# ==========================================
def main():
    if not os.path.exists(META_CSV):
        print(f"Không tìm thấy {META_CSV}. Hãy chạy các bước chuẩn bị dữ liệu trước!")
        return

    df_meta = pd.read_csv(META_CSV)
    X, y, groups, splits, filenames = load_features_from_metadata(df_meta)

    # Lưu bảng vector 78 đặc trưng ra file .npz để tái sử dụng nhanh (Buổi 3 trong đề cương)
    np.savez_compressed(
        os.path.join(REPORT_DIR, "features_78d.npz"),
        X=X, y=y, groups=groups, splits=splits, filenames=filenames
    )
    print(f"-> Đã lưu ma trận đặc trưng kích thước {X.shape} tại {REPORT_DIR}/features_78d.npz")

    # Tách tập Train (1.891 mẫu) và Val/Test độc lập (473 mẫu)
    train_mask = splits == "train"
    val_mask = splits == "val"

    X_train, y_train, groups_train = X[train_mask], y[train_mask], groups[train_mask]
    X_val, y_val, filenames_val = X[val_mask], y[val_mask], filenames[val_mask]

    print(f"\nKích thước tập Train: {X_train.shape[0]} mẫu | Tập Val: {X_val.shape[0]} mẫu")

    # Xây dựng Pipeline: StandardScaler -> SVM (RBF)
    # Bật probability=True để lấy điểm xác suất từng lớp hiển thị lên giao diện Streamlit
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)),
    ])

    # Lưới siêu tham số C và gamma cần dò tìm (Step 2 - Model tuning)
    param_grid = {
        "svm__C": [0.1, 1, 5, 10, 20, 50],
        "svm__gamma": ["scale", 0.001, 0.005, 0.01, 0.02, 0.05],
    }

    # Chia 5-Fold trên tập Train theo StratifiedGroupKFold (chống rò rỉ src_file giữa các fold)
    cv_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nĐang chạy 5-Fold Cross-Validation Tuning (tối ưu theo macro-F1)...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_splitter,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,  # Tự động train lại mô hình tốt nhất trên toàn bộ tập Train
    )
    grid_search.fit(X_train, y_train, groups=groups_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_f1 = grid_search.best_score_

    print("\n================ KẾT QUẢ TUNING (5-FOLD CV) ================")
    print(f"Tham số tối ưu nhất : {best_params}")
    print(f"Mean CV macro-F1    : {best_cv_f1:.4f}")

    # Lưu bảng kết quả tuning toàn bộ tổ hợp (C, gamma) ra CSV cho báo cáo
    df_tuning = pd.DataFrame(grid_search.cv_results_)[
        ["param_svm__C", "param_svm__gamma", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    df_tuning.to_csv(os.path.join(REPORT_DIR, "svm_tuning_report.csv"), index=False)

    # Lưu mô hình tốt nhất ra models/svm.joblib (đúng tên file trong đề cương)
    model_path = os.path.join(MODEL_DIR, "svm.joblib")
    joblib.dump(best_model, model_path)
    print(f"-> Đã lưu mô hình tốt nhất tại: {model_path}")

    # ==========================================
    # 4. ĐÁNH GIÁ TRÊN TẬP VAL ĐỘC LẬP (473 MẪU)
    # ==========================================
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)
    classes = best_model.classes_

    val_acc = accuracy_score(y_val, y_pred)
    val_macro_f1 = f1_score(y_val, y_pred, average="macro")

    print("\n================ ĐÁNH GIÁ TRÊN TẬP VALIDATION ================")
    print(f"Accuracy : {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"Macro-F1 : {val_macro_f1:.4f}")
    print("\nBáo cáo chi tiết từng lớp:")
    report_str = classification_report(y_val, y_pred, digits=4)
    print(report_str)

    # Lưu file text báo cáo metrics
    with open(os.path.join(REPORT_DIR, "evaluation_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best Params: {best_params}\n")
        f.write(f"5-Fold CV Macro-F1: {best_cv_f1:.4f}\n")
        f.write(f"Validation Accuracy: {val_acc:.4f}\n")
        f.write(f"Validation Macro-F1: {val_macro_f1:.4f}\n\n")
        f.write(report_str)

    # Vẽ và lưu Confusion Matrix
    cm = confusion_matrix(y_val, y_pred, labels=classes)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=False,
        annot_kws={"size": 13, "weight": "bold"},
    )
    plt.title(f"Confusion Matrix - SVM (RBF)\nVal Accuracy: {val_acc*100:.2f}% | Macro-F1: {val_macro_f1:.4f}")
    plt.xlabel("Nhãn dự đoán (Predicted)")
    plt.ylabel("Nhãn thực tế (Actual)")
    plt.tight_layout()
    cm_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    print(f"-> Đã lưu biểu đồ Confusion Matrix tại: {cm_path}")

    # Xuất danh sách các file dự đoán sai để phân tích lỗi trong báo cáo (Step 2)
    misclassified = []
    for idx in range(len(y_val)):
        if y_val[idx] != y_pred[idx]:
            prob_dict = {f"prob_{c}": round(float(y_proba[idx][i]), 4) for i, c in enumerate(classes)}
            misclassified.append({
                "filename": filenames_val[idx],
                "true_label": y_val[idx],
                "pred_label": y_pred[idx],
                **prob_dict
            })
    df_errors = pd.DataFrame(misclassified)
    df_errors.to_csv(os.path.join(REPORT_DIR, "misclassified_samples.csv"), index=False, encoding="utf-8-sig")
    print(f"-> Đã lưu {len(df_errors)} mẫu dự đoán sai tại: {REPORT_DIR}/misclassified_samples.csv")

if __name__ == "__main__":
    main()