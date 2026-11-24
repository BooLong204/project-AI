import os
import re
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from prepare_dataset import process_and_clean_audio, BASE_DIR, OUTPUT_DIR, TARGET_SR, DURATION

TARGET_BG_COUNT = 780  # Cân bằng với 762 cat và 822 dog

def main():
    esc_meta_path = os.path.join(BASE_DIR, "ESC-50-master", "meta", "esc50.csv")
    esc_audio_dir = os.path.join(BASE_DIR, "ESC-50-master", "audio")
    meta_csv_path = os.path.join(OUTPUT_DIR, "dataset_metadata.csv")

    if not os.path.exists(meta_csv_path):
        print("Lỗi: Hãy chạy prepare_dataset.py trước!")
        return

    df_esc = pd.read_csv(esc_meta_path)
    # Lấy tất cả các lớp KHÔNG PHẢI dog và cat
    df_bg_pool = df_esc[~df_esc["category"].isin(["dog", "cat"])].copy()

    # Lấy mẫu đa dạng đều trên cả 48 lớp âm thanh nền còn lại
    df_bg_sampled = df_bg_pool.sample(n=min(900, len(df_bg_pool)), random_state=42)

    valid_bg = []
    print("Đang lọc và chuẩn hóa âm thanh nền (background) từ ESC-50...")
    for _, row in tqdm(df_bg_sampled.iterrows(), total=len(df_bg_sampled)):
        if len(valid_bg) >= TARGET_BG_COUNT:
            break
        fpath = os.path.join(esc_audio_dir, row["filename"])
        if not os.path.exists(fpath):
            continue

        audio_clean, status = process_and_clean_audio(fpath)
        if audio_clean is not None:
            valid_bg.append({
                "audio": audio_clean,
                "label": "background",
                "source": f"ESC50_{row['category']}",
                "group_id": f"ESC50_bg_{row['src_file']}",
                "orig_name": row["filename"]
            })

    df_bg = pd.DataFrame(valid_bg)

    # Chia 80% train - 20% val chống rò rỉ theo src_file
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    _, val_idx = next(sgkf.split(df_bg, df_bg["source"], df_bg["group_id"]))
    df_bg["split"] = "train"
    df_bg.loc[val_idx, "split"] = "val"

    # Lưu file vào processed_dataset/{train,val}/background
    df_meta_old = pd.read_csv(meta_csv_path)
    start_idx = len(df_meta_old)
    new_rows = []

    for offset, (_, row) in enumerate(df_bg.iterrows()):
        idx = start_idx + offset
        split = row["split"]
        clean_orig = re.sub(r'[^a-zA-Z0-9_.-]', '_', os.path.splitext(row["orig_name"])[0])
        new_filename = f"{row['source']}_background_{idx:04d}_{clean_orig}.wav"

        save_folder = os.path.join(OUTPUT_DIR, split, "background")
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, new_filename)

        sf.write(save_path, row["audio"], TARGET_SR, subtype="PCM_16")

        new_rows.append({
            "filename": new_filename,
            "filepath": save_path,
            "split": split,
            "label": "background",
            "source": row["source"],
            "group_id": row["group_id"],
            "orig_name": row["orig_name"],
            "sample_rate": TARGET_SR,
            "duration_sec": DURATION
        })

    df_meta_updated = pd.concat([df_meta_old, pd.DataFrame(new_rows)], ignore_index=True)
    df_meta_updated.to_csv(meta_csv_path, index=False, encoding="utf-8-sig")

    print("\n================ CẬP NHẬT DATASET 3 LỚP ================")
    print(pd.crosstab(df_meta_updated["split"], df_meta_updated["label"], margins=True))

if __name__ == "__main__":
    main()