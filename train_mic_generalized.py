import os
import joblib
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
from scipy.signal import butter, lfilter
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_DIR = "processed_dataset"
META_CSV = os.path.join(DATA_DIR, "dataset_metadata.csv")
MODEL_PATH = os.path.join("models", "svm.joblib")
TARGET_SR = 22050
TARGET_LEN = int(TARGET_SR * 5.0)
TARGET_RMS = 0.05

def bandpass_phone_mic(y, sr=TARGET_SR):
    """Mô phỏng dải tần truyền qua loa điện thoại và mic laptop (250Hz - 5000Hz)"""
    b, a = butter(4, [250 / (0.5 * sr), 5000 / (0.5 * sr)], btype="band")
    return lfilter(b, a, y).astype(np.float32)

def extract_78_features(y, sr=TARGET_SR):
    # Luôn lọc thông dải trước khi trích xuất MFCC để đồng nhất file gốc và mic
    y_filt = bandpass_phone_mic(y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y_filt, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])
    return np.concatenate([np.mean(combined, axis=1), np.std(combined, axis=1)])

def main():
    print("=== HUẤN LUYỆN MÔ HÌNH TỔNG QUÁT CHO MỌI VIDEO QUA MICROPHONE ===")
    input("👉 Giữ phòng ở trạng thái bình thường (không bật tiếng chó/mèo), nhấn ENTER để thu 5s tiếng ồn nền phòng học...")
    print("🔴 Đang thu mẫu nền phòng học (5s)...")
    room_noise = sd.rec(TARGET_LEN, samplerate=TARGET_SR, channels=1, dtype="float32")
    sd.wait()
    room_noise = room_noise.flatten()
    noise_rms = np.sqrt(np.mean(room_noise ** 2)) + 1e-8
    room_noise_norm = room_noise * (TARGET_RMS / noise_rms)
    print("✅ Đã lấy mẫu đặc trưng ồn phòng học!")

    df_meta = pd.read_csv(META_CSV)
    df_train = df_meta[df_meta["split"] == "train"]

    X_list, y_list = [], []
    print(f"\nĐang trích xuất đặc trưng trên {len(df_train)} file gốc + bản mô phỏng loa/mic...")
    rng = np.random.default_rng(42)

    for _, row in tqdm(df_train.iterrows(), total=len(df_train)):
        y_audio, _ = librosa.load(row["filepath"], sr=TARGET_SR, mono=True)
        
        # 1. Mẫu gốc đã đồng nhất dải tần
        X_list.append(extract_78_features(y_audio))
        y_list.append(row["label"])

        # 2. Bản mô phỏng phát qua điện thoại + trộn tiếng ồn phòng học thực tế của bạn
        snr_mix = rng.uniform(0.15, 0.45)  # Tỷ lệ nhiễu phòng trộn vào
        y_noisy = y_audio + snr_mix * room_noise_norm
        y_noisy = y_noisy * (TARGET_RMS / (np.sqrt(np.mean(y_noisy ** 2)) + 1e-8))
        X_list.append(extract_78_features(y_noisy))
        y_list.append(row["label"])

    X_all = np.array(X_list, dtype=np.float32)
    y_all = np.array(y_list)

    print("\nĐang huấn luyện SVM (RBF) trên toàn bộ 3.782 mẫu đa điều kiện...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma=0.005, class_weight="balanced", probability=True, random_state=42)),
    ])
    pipeline.fit(X_all, y_all)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n🎉 Hoàn tất! Đã lưu mô hình tổng quát tại: {MODEL_PATH}")

if __name__ == "__main__":
    main()