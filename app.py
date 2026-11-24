import io
import os
import random
import joblib
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
import soundfile as sf
import streamlit as st

# ==========================================
# 1. CẤU HÌNH & NẠP MÔ HÌNH SVM
# ==========================================
st.set_page_config(page_title="Animal Sound Classifier - SVM", page_icon="🐾", layout="centered")

MODEL_PATH = os.path.join("models", "svm.joblib")
VAL_DIR = os.path.join("processed_dataset", "val")

TARGET_SR = 22050
TRAIN_DURATION = 5.0
TARGET_LEN = int(TARGET_SR * TRAIN_DURATION)
MIC_DURATION = 10.0  # Kéo dài thời gian ghi âm trực tiếp lên 10 giây
TARGET_RMS = 0.05

LABEL_DISPLAY = {
    "dog": "🐶 Tiếng Chó (Dog)",
    "cat": "🐱 Tiếng Mèo (Cat)",
    "background": "🔇 Âm thanh nền / Khác (Background)"
}


def load_svm_model():
    return joblib.load(MODEL_PATH)

def extract_78_features(y, sr=TARGET_SR):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])
    return np.concatenate([np.mean(combined, axis=1), np.std(combined, axis=1)]).reshape(1, -1)

def normalize_to_5s(y_seg):
    """Lặp tín hiệu (tile) thay vì đệm số 0 để giữ nguyên mật độ MFCC như lúc train"""
    if len(y_seg) == 0:
        return np.zeros(TARGET_LEN, dtype=np.float32)
    if len(y_seg) < TARGET_LEN:
        repeats = int(np.ceil(TARGET_LEN / len(y_seg)))
        y_5s = np.tile(y_seg, repeats)[:TARGET_LEN]
    else:
        y_5s = y_seg[:TARGET_LEN]

    rms_5s = np.sqrt(np.mean(y_5s ** 2))
    if rms_5s > 0:
        y_norm = y_5s * (TARGET_RMS / rms_5s)
        max_peak = np.max(np.abs(y_norm))
        if max_peak > 0.99:
            y_norm = y_norm * (0.99 / max_peak)
        return y_norm
    return y_5s
def predict_audio(audio_source, model, is_mic=False):
    y, _ = librosa.load(audio_source, sr=TARGET_SR, mono=True)
    if len(y) == 0:
        return "background", {"dog": 0.0, "cat": 0.0, "background": 1.0}, "Tín hiệu rỗng"

    raw_rms = float(np.sqrt(np.mean(y ** 2)))
    if raw_rms < 0.003:
        return "background", {"dog": 0.01, "cat": 0.01, "background": 0.98}, f"Im lặng tuyệt đối (RMS={raw_rms:.4f})"

    # =====================================================================
    # CHẾ ĐỘ MICROPHONE 10 GIÂY: TÌM CÁC KHUNG ÂM THANH NỔI BẬT NHẤT
    # =====================================================================
    if is_mic:
        # Chia 10 giây thành các khung ngắn 0.5s để đo bản đồ năng lượng
        frame_len = int(0.5 * TARGET_SR)
        frames = [y[i : i + frame_len] for i in range(0, len(y) - frame_len + 1, frame_len)]
        frame_rms = np.array([float(np.sqrt(np.mean(f ** 2))) for f in frames])

        noise_floor = float(np.percentile(frame_rms, 30))  # Mức ồn nền phòng học (30% khung yên tĩnh nhất)
        peak_rms = float(np.max(frame_rms))

        # Nếu cả 10 giây không có âm thanh nào vượt trội hơn nền phòng học -> Background
        if peak_rms < noise_floor * 1.45:
            return (
                "background",
                {"dog": 0.03, "cat": 0.04, "background": 0.93},
                f"Chỉ có tiếng nền phòng học (Peak={peak_rms:.4f}, Nền={noise_floor:.4f})"
            )

        # Chỉ giữ lại các khung có tiếng kêu rõ (lớn hơn 60% đỉnh âm thanh) và ghép nối lại
        active_frames = [frames[i] for i, r in enumerate(frame_rms) if r >= peak_rms * 0.55]
        y_active = np.concatenate(active_frames) if active_frames else y

        y_norm = normalize_to_5s(y_active)
        features = extract_78_features(y_norm, sr=TARGET_SR)
        probs = model.predict_proba(features)[0]
        classes = list(model.classes_)
        prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}

        # Vì mic laptop làm suy giảm âm sắc động vật so với file phòng thu,
        # nếu xác suất dog hoặc cat >= 0.35 (vượt xa mức ngẫu nhiên) thì chọn lớp động vật cao hơn
        # =====================================================================
        # HIỆU CHỈNH XÁC SUẤT CHO MÔI TRƯỜNG MICROPHONE PHÒNG HỌC
        # =====================================================================
        # Khi đã có âm thanh vượt trội hơn nền phòng (peak >= 1.45 * nền),
        # chỉ cần dog hoặc cat đạt >= 0.05 (5%) là đã bắt được tín hiệu động vật
        if prob_dict["dog"] >= 0.05 or prob_dict["cat"] >= 0.05:
            # Giảm trọng số nhiễu nền phòng học để biểu đồ phản ánh đúng tương quan Chó vs Mèo
            adj_probs = {
                "dog": prob_dict["dog"],
                "cat": prob_dict["cat"],
                "background": prob_dict["background"] * 0.08  # Hạ nhiễu nền phòng học
            }
            total_p = sum(adj_probs.values())
            prob_dict = {k: v / total_p for k, v in adj_probs.items()}
            pred_label = "dog" if prob_dict["dog"] >= prob_dict["cat"] else "cat"
        else:
            pred_label = "background"

        return pred_label, prob_dict, f"Đã tách tiếng kêu (Peak={peak_rms:.4f}, Nền={noise_floor:.4f})"

def render_prediction_result(pred_label, prob_dict, status_note):
    st.success(f"### Kết quả AI dự đoán: {LABEL_DISPLAY[pred_label]}")
    st.caption(f"Trạng thái tín hiệu: {status_note} | Chuẩn hóa: Mono, 22.050 Hz, 78 đặc trưng MFCC")

    df_probs = pd.DataFrame({
        "Lớp âm thanh": [LABEL_DISPLAY[k] for k in ["dog", "cat", "background"]],
        "Xác suất (%)": [round(prob_dict.get(k, 0.0) * 100, 2) for k in ["dog", "cat", "background"]]
    }).set_index("Lớp âm thanh")

    st.bar_chart(df_probs)

# ==========================================
# 3. GIAO DIỆN CHÍNH STREAMLIT
# ==========================================
st.title("🐾 Hệ Thống Nhận Diện Tiếng Chó & Mèo")
st.markdown("**Mô hình:** Traditional Machine Learning — `78D MFCC + StandardScaler + SVM (RBF)`")

if not os.path.exists(MODEL_PATH):
    st.error("Chưa tìm thấy file `models/svm.joblib`. Hãy chạy `train_svm.py` trước!")
    st.stop()

model = load_svm_model()

tab1, tab2, tab3 = st.tabs([
    "🎙️ Thu âm Microphone (10 giây)",
    "📁 Tải lên File âm thanh",
    "🎮 Chế độ Human vs AI"
])

# --- TAB 1: MICROPHONE 10 GIÂY ---
with tab1:
    st.subheader("Thu âm trực tiếp từ Microphone Laptop (10 giây)")
    st.info("Bấm nút bắt đầu, sau đó phát tiếng chó sủa hoặc mèo kêu từ điện thoại (hoặc để im lặng/nói chuyện để kiểm tra lớp Background).")
    
    if st.button("🎙️ Bắt đầu thu âm (10 giây)", type="primary"):
        try:
            with st.spinner("🔴 Đang ghi âm trong 10 giây... Hãy phát âm thanh mẫu!"):
                recording = sd.rec(int(MIC_DURATION * TARGET_SR), samplerate=TARGET_SR, channels=1, dtype="float32")
                sd.wait()

            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, recording, TARGET_SR, format="WAV")
            wav_buffer.seek(0)

            st.audio(wav_buffer, format="audio/wav")
            wav_buffer.seek(0)
            pred, probs, note = predict_audio(wav_buffer, model, is_mic=True)
            render_prediction_result(pred, probs, note)
        except Exception as e:
            st.error(f"Không thể truy cập Microphone phần cứng: {e}")

# --- TAB 2: UPLOAD FILE ---
with tab2:
    st.subheader("Kiểm thử bằng File âm thanh ngoài")
    uploaded_file = st.file_uploader("Chọn file âm thanh (.wav, .mp3, .ogg, .flac)", type=["wav", "mp3", "ogg", "flac"])
    if uploaded_file is not None:
        st.audio(uploaded_file)
        with st.spinner("Đang xử lý tín hiệu..."):
            pred, probs, note = predict_audio(uploaded_file, model, is_mic=False)
            render_prediction_result(pred, probs, note)

# --- TAB 3: HUMAN VS AI ---
with tab3:
    st.subheader("Thử thách: Người vs Mô hình SVM")
    if "quiz_file" not in st.session_state:
        st.session_state.quiz_file = None
        st.session_state.quiz_true_label = None

    if st.button("🎲 Lấy mẫu âm thanh ngẫu nhiên mới"):
        chosen_cls = random.choice(["dog", "cat", "background"])
        cls_folder = os.path.join(VAL_DIR, chosen_cls)
        if os.path.exists(cls_folder):
            files = [f for f in os.listdir(cls_folder) if f.endswith(".wav")]
            if files:
                st.session_state.quiz_file = os.path.join(cls_folder, random.choice(files))
                st.session_state.quiz_true_label = chosen_cls

    if st.session_state.quiz_file:
        st.audio(st.session_state.quiz_file)
        human_choice = st.radio(
            "Bạn đoán đây là âm thanh gì?",
            options=["dog", "cat", "background"],
            format_func=lambda x: LABEL_DISPLAY[x],
            horizontal=True
        )

        if st.button("🔍 Chốt đáp án & So sánh với AI"):
            true_lbl = st.session_state.quiz_true_label
            pred, probs, note = predict_audio(st.session_state.quiz_file, model, is_mic=False)

            col1, col2, col3 = st.columns(3)
            col1.metric("Đáp án thật", true_lbl.upper())
            col2.metric("Bạn đoán", human_choice.upper(), "Đúng" if human_choice == true_lbl else "Sai")
            col3.metric("AI (SVM) đoán", pred.upper(), "Đúng" if pred == true_lbl else "Sai")

            render_prediction_result(pred, probs, note)