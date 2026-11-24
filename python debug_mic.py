
import joblib
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf

TARGET_SR = 22050
DURATION = 5.0
MODEL_PATH = "models/svm.joblib"

def extract_78(y):
    mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=13)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    comb = np.vstack([mfcc, d1, d2])
    return np.concatenate([np.mean(comb, axis=1), np.std(comb, axis=1)]).reshape(1, -1)

print("1. Danh sách thiết bị Micro trên máy bạn:")
print(sd.query_devices())
print(f"\n-> Thiết bị thu mặc định đang dùng: {sd.default.device[0]}")

input("\n👉 Hãy bật tiếng CHÓ SỦA trên điện thoại, đặt cách mic laptop ~15cm rồi nhấn ENTER để thu 5 giây...")
print("🔴 Đang thu...")
rec = sd.rec(int(DURATION * TARGET_SR), samplerate=TARGET_SR, channels=1, dtype="float32")
sd.wait()

y = rec.flatten()
sf.write("test_mic_raw.wav", y, TARGET_SR)
print("✅ Đã lưu file: D:\\AI\\test_mic_raw.wav (Hãy mở file này nghe thử xem tiếng chó có rõ không!)")

rms = np.sqrt(np.mean(y**2))
peak = np.max(np.abs(y))
print(f"\n2. Thông số tín hiệu thu được:")
print(f"   - Raw RMS  : {rms:.5f}")
print(f"   - Max Peak : {peak:.5f}")

# Chuẩn hóa y hệt như file train
y_norm = y * (0.05 / (rms + 1e-8))
y_norm = np.clip(y_norm, -0.99, 0.99)
sf.write("test_mic_norm.wav", y_norm, TARGET_SR)

model = joblib.load(MODEL_PATH)
probs = model.predict_proba(extract_78(y_norm))[0]
print("\n3. Kết quả mô hình SVM dự đoán trực tiếp (không qua bất kỳ bộ lọc nào):")
for cls, p in zip(model.classes_, probs):
    print(f"   - {cls:10s}: {p*100:.2f}%")