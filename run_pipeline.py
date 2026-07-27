import os
import json
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Import local pipeline components
from preprocessing.normalize import normalize_audio, TARGET_SR
from preprocessing.silence import trim_silence
from preprocessing.noise import reduce_noise
from preprocessing.segment import segment_audio

# --- 🛠️ STEP 3 SPLIT FUNCTION INTEGRATED DIRECTLY FOR SAFETY ---
def create_splits(manifest_df, held_out_val_keyword, held_out_test_keyword):
    """
    Dynamically maps splits by checking if the core surah keyword exists 
    anywhere inside the filename string, handling numeric prefixes automatically.
    """
    def assign_split(surah_name):
        name_lower = str(surah_name).lower()
        if held_out_val_keyword.lower() in name_lower:
            return "val"
        elif held_out_test_keyword.lower() in name_lower:
            return "test"
        else:
            return "train"

    manifest_df = manifest_df.copy()
    manifest_df["split"] = manifest_df["surah_name"].apply(assign_split)
    return manifest_df

# --- 📂 UNIFORM LOCAL DIRECTORY PATHS ---
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "normalized"
CHUNKS_DIR = BASE_DIR / "data" / "processed" / "chunks"

# Core keywords for soft-matching (handles "18_Surah_Kahf", "55_Surah_Rahman" flawlessly)
HELD_OUT_VAL = "Kahf"
HELD_OUT_TEST = "Rehman"

def preprocess_single_file(input_path, output_path):
    normalize_audio(input_path, output_path)
    y, sr = librosa.load(str(output_path), sr=TARGET_SR, mono=True)
    y = reduce_noise(y, sr)
    y = trim_silence(y, sr)
    sf.write(str(output_path), y, sr, subtype="PCM_16")

def main():
    global RAW_DIR, PROCESSED_DIR, CHUNKS_DIR
    
    print("🎬 Starting Local Phase 1 Preprocessing Pipeline...")
    
    # Critical directory adjustments for local workspace
    if not RAW_DIR.exists():
        # Fallback support check for 'dataset/raw' layout if already populated
        ALT_RAW = BASE_DIR / "dataset" / "raw"
        if ALT_RAW.exists():
            # (We already declared them global at the top, so we just re-assign them here safely)
            RAW_DIR = ALT_RAW
            PROCESSED_DIR = BASE_DIR / "dataset" / "processed" / "normalized"
            CHUNKS_DIR = BASE_DIR / "dataset" / "processed" / "chunks"
        else:
            raise FileNotFoundError(f"❌ Raw dataset folder not found at {RAW_DIR}. Please place your Qari folders inside data/raw/")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    qari_folders = sorted([f for f in RAW_DIR.iterdir() if f.is_dir()])
    print(f"📂 Found {len(qari_folders)} Qari folders.")

    # --- 🔀 SMART SKIP CHECK ---
    existing_wavs = list(PROCESSED_DIR.glob("**/*.wav"))
    
    if len(existing_wavs) > 0:
        print(f"⏩ Found {len(existing_wavs)} already normalized audio files. Skipping Step 1!")
    else:
        print("🎙️ Step 1: Normalizing, Denoising & Trimming raw files...")
        for qari_folder in tqdm(qari_folders, desc="Processing Qaris"):
            qari_id = qari_folder.name
            qari_out_path = PROCESSED_DIR / qari_id
            qari_out_path.mkdir(parents=True, exist_ok=True)

            for surah_file in qari_folder.glob("*"):
                if surah_file.suffix.lower() in ['.mp3', '.wav', '.m4a', '.flac']:
                    out_path = qari_out_path / f"{surah_file.stem}.wav"
                    preprocess_single_file(surah_file, out_path)
        print("\n✅ Step 1: Normalization, Denoising & Trimming Complete.")

    # 2. Run Audio Segmentation
    print("✂️ Step 2: Running Window Segmentation...")
    manifest_df = segment_audio(PROCESSED_DIR, CHUNKS_DIR)
    
    # Save precisely into the shared parent environment path
    OUTPUT_PARENT_DIR = PROCESSED_DIR.parent
    manifest_path = OUTPUT_PARENT_DIR / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Total chunks created: {len(manifest_df)}")

    # 3. Create Leakage-Safe Splits
    print("⚖️ Step 3: Enforcing Split Constraints...")
    splits_df = create_splits(manifest_df, HELD_OUT_VAL, HELD_OUT_TEST)
    
    splits_csv_path = OUTPUT_PARENT_DIR / "splits.csv"
    splits_json_path = OUTPUT_PARENT_DIR / "splits.json"
    
    splits_df.to_csv(splits_csv_path, index=False)
    
    splits_json = splits_df.to_dict(orient="records")
    with open(splits_json_path, "w") as f:
        json.dump(splits_json, f, indent=2)
    print("Data splits saved successfully.")

    # 4. Final Verification Check
    print("\n🧐 Step 4: Verification Checklist Running...")
    
    for qari_folder in PROCESSED_DIR.iterdir():
        if qari_folder.is_dir():
            n = len(list(qari_folder.glob("*.wav")))
            assert n == 8, f"⚠️ Mismatch: {qari_folder.name} has {n} files, expected 8."
            
    print("✓ Verification 1 Passed: All Qaris have 8 base files.")
    
    leakage_check = splits_df.groupby("session_id")["split"].nunique()
    assert (leakage_check == 1).all(), "❌ LEAKAGE DETECTED across splits!"
    print("✓ Verification 2 Passed: Zero session leakage.")
    
    val_surahs = splits_df[splits_df["split"] == "val"]["surah_name"].unique()
    test_surahs = splits_df[splits_df["split"] == "test"]["surah_name"].unique()
    
    print(f"📋 Validation Set Surah matches: {val_surahs}")
    print(f"📋 Testing Set Surah matches: {test_surahs}")
    
    assert len(val_surahs) >= 1 and len(test_surahs) >= 1, "❌ Failed to isolate target validation/testing sets!"
    print("✓ Verification 3 Passed: Held-out sets isolated cleanly.")
    
    print(f"\n🎉 Phase 1 is officially complete and safe! All outputs stored uniformly under: {OUTPUT_PARENT_DIR.resolve()}")

if __name__ == "__main__":
    main()