# matching/build_reference_embeddings.py
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

# --- 📂 ROBUST ABSOLUTE PATH RESOLUTION ---
BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST_PATH = BASE_DIR / "dataset" / "processed" / "track_a_features.csv"
OUTPUT_PATH = BASE_DIR / "matching" / "reference_embeddings.pkl"

# Ensure the output directory exists
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"🔍 Loading manifest from: {MANIFEST_PATH}")
if not MANIFEST_PATH.exists():
    raise FileNotFoundError(f"❌ Cannot find manifest file at {MANIFEST_PATH}!")

df = pd.read_csv(MANIFEST_PATH)

# Use only TRAIN split chunks to build reference embeddings
train_df = df[df["split"] == "train"]
print(f"📈 Extracting profiles using {len(train_df)} training chunks...")

reference_embeddings = {}

for qari_id in sorted(train_df["qari_id"].unique()):
    qari_chunks = train_df[train_df["qari_id"] == qari_id]
    
    # Dynamically resolve individual .npy embedding file paths relative to BASE_DIR
    embeddings = np.stack([
        np.load(BASE_DIR / p if not os.path.isabs(p) else p) 
        for p in qari_chunks["embedding_path"]
    ])

    # Average embedding, then re-normalize to unit length (L2 norm)
    avg_embedding = embeddings.mean(axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

    reference_embeddings[qari_id] = avg_embedding
    print(f"   ✓ {qari_id:<25} -> Averaged from {len(qari_chunks):>4} train chunks")

# Save the centralized vector library
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(reference_embeddings, f)

print(f"\n🎉 Successfully compiled and saved reference embeddings library to:")
print(f"📦 {OUTPUT_PATH}")