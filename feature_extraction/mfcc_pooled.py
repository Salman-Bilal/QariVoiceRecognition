# =====================================================================
# 🛠️ AUDIO FEATURE EXTRACTION ENGINE (TRACK B: HAND-CRAFTED ACOUSTICS)
# =====================================================================
import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# --- 📂 UNIFORM LOCAL DIRECTORY PATHS ---
# Walk up one level (.parent.parent) out of 'feature_extraction' to reach project root
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "dataset" / "processed"
SPLITS_CSV_PATH = DATA_DIR / "splits.csv"

# Target cache folder for storing raw feature arrays and structural manifests
OUTPUT_FEATURES_DIR = DATA_DIR / "track_b_features"
MANIFEST_OUTPUT_PATH = DATA_DIR / "track_b_features.csv"

# --- 🎛️ AUDIO PROCESSING PARAMETERS ---
SR = 16000
N_MFCC = 13

def extract_pooled_features(wav_path, sr=SR, n_mfcc=N_MFCC):
    """
    Extracts frame-level MFCCs, Deltas, Delta-Deltas, Pitch, and Energy, 
    then pools them across time using descriptive statistics.
    """
    # Load audio chunk cleanly using librosa
    y, _ = librosa.load(wav_path, sr=sr)

    # 1. Mel-Frequency Cepstral Coefficients (MFCC) & Derivatives
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # 2. Pitch Tracking via Probabilistic YIN (pYIN)
    pitch, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
    )
    pitch = np.nan_to_num(pitch)

    # 3. Root-Mean-Square Energy 
    energy = librosa.feature.rms(y=y)

    # Statistical Pooling Function (Collapses the time dimension)
    def pool(feature_matrix):
        return np.concatenate([
            feature_matrix.mean(axis=-1),
            feature_matrix.std(axis=-1),
            feature_matrix.min(axis=-1),
            feature_matrix.max(axis=-1),
        ])

    # Concatenate all hand-crafted feature blocks into one flat vector
    feature_vector = np.concatenate([
        pool(mfcc), pool(mfcc_delta), pool(mfcc_delta2),
        pool(pitch.reshape(1, -1)),
        pool(energy),
    ])
    return feature_vector

def extract_all(splits_path, output_dir):
    """
    Loops through the master splits CSV, extracts acoustic features for each 
    chunk, caches the numpy binaries, and saves a Track B index sheet.
    """
    print("🧠 Initializing Phase 2: Track B Acoustic Feature Extractor...")
    
    # Check if Phase 1 splits database exists
    if not Path(splits_path).exists():
        raise FileNotFoundError(f"❌ Missing data files. Please ensure Phase 1 ran successfully and created {splits_path}")
        
    # Read the master splits dataset
    splits_df = pd.read_csv(splits_path)
    print(f"📋 Loaded manifest mapping with {len(splits_df)} total chunks.")
    
    # Create target directory for numpy caching safely
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    records = []

    print("\n🚀 Starting Feature Extraction across chunks...")
    for _, row in tqdm(splits_df.iterrows(), total=len(splits_df), desc="Extracting MFCC+pitch+energy"):
        chunk_path = Path(row["chunk_path"])
        
        if not chunk_path.exists():
            print(f"⚠️ Warning: Missing chunk file at {chunk_path}, skipping.")
            continue
            
        try:
            # Run feature extraction pipeline
            feat = extract_pooled_features(str(chunk_path))
            
            # Formulate safe unique cache name for vector saving
            feat_filename = f"{row['session_id']}_{chunk_path.name.replace('.wav', '.npy')}"
            feat_save_path = Path(output_dir) / feat_filename
            np.save(feat_save_path, feat)

            # Record entries for rebuilding structural dataframes 
            records.append({
                "qari_id": row["qari_id"],
                "surah_name": row["surah_name"],
                "session_id": row["session_id"],
                "split": row["split"],
                "chunk_path": str(chunk_path.resolve()),
                "feature_path": str(feat_save_path.resolve())
            })
        except Exception as e:
            print(f"❌ Error processing chunk {chunk_path.name}: {str(e)}")
            continue

    return pd.DataFrame(records)

if __name__ == "__main__":
    # Execute extraction engine pointing directly to processed directory structure
    df = extract_all(SPLITS_CSV_PATH, OUTPUT_FEATURES_DIR)
    
    # Save master sheet index
    df.to_csv(MANIFEST_OUTPUT_PATH, index=False)
    
    print("\n" + "="*50)
    print("🎉 Phase 2 - Track B Feature Extraction Complete!")
    print(f"🎯 Total acoustic feature vectors cached: {len(df)}")
    print(f"📦 Master feature indexing sheet saved at: {MANIFEST_OUTPUT_PATH.resolve()}")
    print("="*50)