# run_preprocessing.py
import os
from pathlib import Path
from preprocessing.pipeline import preprocess_file
from tqdm import tqdm

RAW_DIR = "./dataset/raw"
PROCESSED_DIR = "./dataset/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

qari_folders = sorted(os.listdir(RAW_DIR))
print(f"Found {len(qari_folders)} Qari folders: {qari_folders}")

for qari in tqdm(qari_folders, desc="Qaris"):
    qari_raw_path = os.path.join(RAW_DIR, qari)
    qari_out_path = os.path.join(PROCESSED_DIR, qari)
    os.makedirs(qari_out_path, exist_ok=True)

    surah_files = sorted(os.listdir(qari_raw_path))
    for surah_file in surah_files:
        in_path = os.path.join(qari_raw_path, surah_file)
        surah_name = Path(surah_file).stem
        out_path = os.path.join(qari_out_path, f"{surah_name}.wav")
        preprocess_file(in_path, out_path)

print("Preprocessing complete. Output at:", PROCESSED_DIR)