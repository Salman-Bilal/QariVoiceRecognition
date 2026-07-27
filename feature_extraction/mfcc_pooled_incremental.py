# feature_extraction/mfcc_pooled_incremental.py
import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from feature_extraction.mfcc_pooled import extract_pooled_features  # reuse your existing function

SPLITS_PATH = "dataset/processed/splits.csv"
OUTPUT_DIR = Path("feature_extraction/cache/track_b")
MANIFEST_PATH = "feature_extraction/cache/track_b_manifest.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
splits_df = pd.read_csv(SPLITS_PATH)

# Load existing manifest if present, to know what's already done
if os.path.exists(MANIFEST_PATH):
    existing_df = pd.read_csv(MANIFEST_PATH)
    done_chunk_paths = set(existing_df["chunk_path"])
else:
    existing_df = pd.DataFrame()
    done_chunk_paths = set()

new_rows = []
todo_df = splits_df[~splits_df["chunk_path"].isin(done_chunk_paths)]
print(f"Chunks already cached: {len(done_chunk_paths)}")
print(f"New chunks to process: {len(todo_df)}")

for _, row in tqdm(todo_df.iterrows(), total=len(todo_df), desc="Extracting new MFCC features"):
    feat = extract_pooled_features(row["chunk_path"])
    feat_filename = f"{row['session_id']}_{os.path.basename(row['chunk_path'])}.npy"
    feat_path = OUTPUT_DIR / feat_filename
    np.save(feat_path, feat)

    new_rows.append({
        "qari_id": row["qari_id"],
        "surah_name": row["surah_name"],
        "session_id": row["session_id"],
        "split": row["split"],
        "chunk_path": row["chunk_path"],
        "feature_path": str(feat_path)
    })

new_df = pd.DataFrame(new_rows)
combined_df = pd.concat([existing_df, new_df], ignore_index=True)
combined_df.to_csv(MANIFEST_PATH, index=False)
print(f"Done. Total in manifest now: {len(combined_df)}")