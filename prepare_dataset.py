import os
import glob
import re
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

# ==========================================
# 1. CẤU HÌNH THAM SỐ CHUẨN HÓA
# ==========================================
BASE_DIR = "."  # Thư mục hiện tại chứa cats_dogs, DvC, ESC-50-master
OUTPUT_DIR = "processed_dataset"

TARGET_SR = 22050          # Chuẩn 22.050 Hz theo đề cương (có thể đổi thành 22150 nếu muốn)
DURATION = 5.0             # Đúng 5 giây như ESC-50
TARGET_LEN = int(TARGET_SR * DURATION)
TARGET_RMS = 0.05          # Chuẩn hóa năng lượng (RMS) đồng đều cho mọi file
MIN_RAW_RMS = 0.002        # Ngưỡng lọc bỏ file bị câm / quá nhỏ / lỗi
MIN_ACTIVE_SEC = 0.3       # Ngưỡng thời lượng âm thanh thực tế tối thiểu sau khi cắt khoảng lặng

# ==========================================
# 2. QUÉT VÀ GÁN NHÃN TỪ 3 NGUỒN DỮ LIỆU
# ==========================================
def collect_all_files():
    records = []

    # --- NGUỒN 1: ESC-50-master ---
    esc_meta_path = os.path.join(BASE_DIR, "ESC-50-master", "meta", "esc50.csv")
    esc_audio_dir = os.path.join(BASE_DIR, "ESC-50-master", "audio")
    if os.path.exists(esc_meta_path):
        df_esc = pd.read_csv(esc_meta_path)
        df_esc = df_esc[df_esc["category"].isin(["dog", "cat"])]
        for _, row in df_esc.iterrows():
            fpath = os.path.join(esc_audio_dir, row["filename"])
            if os.path.exists(fpath):
                records.append({
                    "filepath": fpath,
                    "label": row["category"],
                    "source": "ESC50",
                    "group_id": f"ESC50_{row['src_file']}",  # Chống data leakage theo src_file
                    "orig_name": row["filename"]
                })

    # --- NGUỒN 2: cats_dogs (Kaggle Audio Cats and Dogs) ---
    kaggle_dir = os.path.join(BASE_DIR, "cats_dogs")
    if os.path.exists(kaggle_dir):
        for fpath in glob.glob(os.path.join(kaggle_dir, "**", "*.wav"), recursive=True):
            fname = os.path.basename(fpath).lower()
            parent = os.path.basename(os.path.dirname(fpath)).lower()
            label = None
            if "dog" in fname or "dog" in parent:
                label = "dog"
            elif "cat" in fname or "cat" in parent:
                label = "cat"
            
            if label:
                # Gom nhóm theo số ID gốc của file để tránh trùng nguồn
                stem = os.path.splitext(fname)[0]
                records.append({
                    "filepath": fpath,
                    "label": label,
                    "source": "Kaggle_CatsDogs",
                    "group_id": f"Kaggle_{stem}",
                    "orig_name": os.path.basename(fpath)
                })

    # --- NGUỒN 3: DvC (Figshare Dogs vs Cats Audio Dataset) ---
    dvc_dir = os.path.join(BASE_DIR, "DvC")
    if os.path.exists(dvc_dir):
        for fpath in glob.glob(os.path.join(dvc_dir, "**", "*.*"), recursive=True):
            if not fpath.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                continue
            fname = os.path.basename(fpath).lower()
            parent = os.path.basename(os.path.dirname(fpath)).lower()
            label = None
            if "dog" in fname or "dog" in parent or fname.startswith("d_"):
                label = "dog"
            elif "cat" in fname or "cat" in parent or fname.startswith("c_"):
                label = "cat"

            if label:
                # Tách phần gốc trước dấu gạch dưới/số đoạn cắt nếu có (ví dụ dog_12_part1 -> dog_12)
                stem = os.path.splitext(fname)[0]
                base_group = re.sub(r'(_part\d+|_\d+)$', '', stem)
                records.append({
                    "filepath": fpath,
                    "label": label,
                    "source": "Figshare_DvC",
                    "group_id": f"DvC_{label}_{base_group}",
                    "orig_name": os.path.basename(fpath)
                })

    return pd.DataFrame(records)

# ==========================================
# 3. LÀM SẠCH, CHUẨN HÓA 5S & CÂN BẰNG NĂNG LƯỢNG
# ==========================================
def process_and_clean_audio(filepath):
    """
    Trả về (audio_array, status_message)
    Nếu file lỗi / im lặng -> trả về (None, lý_do_loại)
    """
    try:
        # Đọc file, ép về Mono và Resample về TARGET_SR (22050 Hz)
        y, _ = librosa.load(filepath, sr=TARGET_SR, mono=True)
    except Exception as e:
        return None, f"corrupted_file ({e})"

    # Kiểm tra NaN/Inf hoặc file rỗng
    if y is None or len(y) == 0 or not np.isfinite(y).all():
        return None, "empty_or_invalid_signal"

    # Kiểm tra năng lượng gốc (loại file câm/rè quá nhỏ)
    raw_rms = np.sqrt(np.mean(y ** 2))
    if raw_rms < MIN_RAW_RMS:
        return None, f"too_quiet_rms_{raw_rms:.5f}"

    # Cắt bỏ khoảng lặng thừa ở 2 đầu (top_db=30)
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    if len(y_trimmed) < int(MIN_ACTIVE_SEC * TARGET_SR):
        return None, "too_short_active_sound"

    # Đưa độ dài về đúng 5.0 giây (TARGET_LEN mẫu)
    cur_len = len(y_trimmed)
    if cur_len > TARGET_LEN:
        # Nếu dài hơn 5s: dùng cửa sổ trượt tìm đoạn 5s có năng lượng lớn nhất (chứa tiếng kêu rõ nhất)
        step = int(0.5 * TARGET_SR)
        best_start = 0
        max_energy = -1.0
        for start in range(0, cur_len - TARGET_LEN + 1, step):
            segment = y_trimmed[start : start + TARGET_LEN]
            energy = np.sum(segment ** 2)
            if energy > max_energy:
                max_energy = energy
                best_start = start
        y_5s = y_trimmed[best_start : best_start + TARGET_LEN]
    elif cur_len < TARGET_LEN:
        # Nếu ngắn hơn 5s: đệm khoảng lặng đối xứng hai đầu (zero-padding)
        pad_total = TARGET_LEN - cur_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        y_5s = np.pad(y_trimmed, (pad_left, pad_right), mode="constant")
    else:
        y_5s = y_trimmed

    # Chuẩn hóa năng lượng (RMS Normalization) về cùng một mức TARGET_RMS
    rms_5s = np.sqrt(np.mean(y_5s ** 2))
    if rms_5s > 0:
        y_norm = y_5s * (TARGET_RMS / rms_5s)
        # Chống vỡ tiếng (clipping) nếu đỉnh sóng vượt quá [-1.0, 1.0]
        max_peak = np.max(np.abs(y_norm))
        if max_peak > 0.99:
            y_norm = y_norm * (0.99 / max_peak)
    else:
        return None, "zero_rms_after_trim"

    return y_norm.astype(np.float32), "ok"

# ==========================================
# 4. CHẠY PIPELINE & CHIA TẬP TRAIN / VAL
# ==========================================
def main():
    print("1. Đang quét dữ liệu từ ESC-50, Kaggle (cats_dogs), và Figshare (DvC)...")
    df_raw = collect_all_files()
    print(f"   -> Tìm thấy tổng cộng {len(df_raw)} file thô.")
    print(df_raw.groupby(["source", "label"]).size())

    valid_records = []
    rejected_records = []

    print("\n2. Đang làm sạch, chuẩn hóa 5s (22.050 Hz, Mono) và cân bằng năng lượng RMS...")
    for idx, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
        audio_clean, status = process_and_clean_audio(row["filepath"])
        if audio_clean is None:
            rejected_records.append({
                "source": row["source"],
                "label": row["label"],
                "orig_name": row["orig_name"],
                "reason": status
            })
        else:
            valid_records.append({
                "audio": audio_clean,
                "label": row["label"],
                "source": row["source"],
                "group_id": row["group_id"],
                "orig_name": row["orig_name"]
            })

    df_clean = pd.DataFrame(valid_records)
    df_rejected = pd.DataFrame(rejected_records)

    # Chia tập Train (80%) và Val (20%) theo StratifiedGroupKFold (cân bằng lớp + không rò rỉ nguồn)
    print("\n3. Đang chia tập Train (80%) và Val (20%) chống rò rỉ dữ liệu...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(sgkf.split(df_clean, df_clean["label"], df_clean["group_id"]))

    df_clean["split"] = "train"
    df_clean.loc[val_idx, "split"] = "val"

    # Tạo cấu trúc thư mục đầu ra và lưu file .wav sạch
    print("\n4. Đang xuất file .wav sạch và bảng metadata...")
    metadata_rows = []
    for i, row in tqdm(df_clean.iterrows(), total=len(df_clean)):
        split = row["split"]
        label = row["label"]
        source = row["source"]
        # Đặt tên file mới chứa đầy đủ: nguồn + nhãn + số thứ tự + tên gốc
        clean_orig = re.sub(r'[^a-zA-Z0-9_.-]', '_', os.path.splitext(row["orig_name"])[0])
        new_filename = f"{source}_{label}_{i:04d}_{clean_orig}.wav"

        save_folder = os.path.join(OUTPUT_DIR, split, label)
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, new_filename)

        sf.write(save_path, row["audio"], TARGET_SR, subtype="PCM_16")

        metadata_rows.append({
            "filename": new_filename,
            "filepath": save_path,
            "split": split,
            "label": label,
            "source": source,
            "group_id": row["group_id"],
            "orig_name": row["orig_name"],
            "sample_rate": TARGET_SR,
            "duration_sec": DURATION
        })

    # Lưu CSV phục vụ train & viết báo cáo
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_meta = pd.DataFrame(metadata_rows)
    df_meta.to_csv(os.path.join(OUTPUT_DIR, "dataset_metadata.csv"), index=False, encoding="utf-8-sig")
    df_rejected.to_csv(os.path.join(OUTPUT_DIR, "rejected_files_report.csv"), index=False, encoding="utf-8-sig")

    print("\n================ TỔNG KẾT KẾT QUẢ ================")
    print(f"Tổng số file sạch giữ lại : {len(df_meta)}")
    print(f"Tổng số file lỗi bị loại  : {len(df_rejected)} (chi tiết trong rejected_files_report.csv)")
    print("\nPhân bố chi tiết theo Split / Nguồn / Nhãn:")
    print(pd.crosstab([df_meta["split"], df_meta["source"]], df_meta["label"], margins=True))

if __name__ == "__main__":
    main()