# =====================================================================
# 🛠️ PYTORCH, TORCHAUDIO & HUGGINGFACE LAYER COMPATIBILITY PATCH
# =====================================================================
import sys
import torchaudio
import huggingface_hub
from types import ModuleType

# 1. Mock the legacy audio backends listing function if missing
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

# 2. Mock torchaudio.io sub-module structures if system ffmpeg is missing
if not hasattr(torchaudio, "io"):
    mock_io = ModuleType("torchaudio.io")
    class DummyStreamer:
        def __init__(self, *args, **kwargs): pass
    mock_io.StreamReader = DummyStreamer
    mock_io.StreamWriter = DummyStreamer
    torchaudio.io = mock_io
    sys.modules["torchaudio.io"] = mock_io

# 3. Patch Hugging Face download rules to fix 'use_auth_token' and mimic legacy 404 behavior
orig_hf_hub_download = huggingface_hub.hf_hub_download

def patched_hf_hub_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
        
    try:
        return orig_hf_hub_download(*args, **kwargs)
    except Exception as e:
        err_msg = str(e)
        if "404" in err_msg or "Entry Not Found" in err_msg or "RemoteEntryNotFoundError" in e.__class__.__name__:
            raise huggingface_hub.errors.EntryNotFoundError(
                "404 Client Error: Entry Not Found for url (Safe pipeline bypass style)"
            )
        raise e

huggingface_hub.hf_hub_download = patched_hf_hub_download
# =====================================================================

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

# --- 📂 UNIFORM LOCAL DIRECTORY PATHS ---
BASE_DIR = Path(__file__).parent.parent 
DATA_DIR = BASE_DIR / "dataset" / "processed"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
SPLITS_CSV_PATH = DATA_DIR / "splits.csv"
OUTPUT_EMBEDDINGS_DIR = DATA_DIR / "embeddings"

def main():
    print("🧠 Initializing Phase 2: Track A Embedding Extractor...")
    
    if not SPLITS_CSV_PATH.exists():
        raise FileNotFoundError(f"❌ Missing data files. Please ensure Phase 1 ran successfully and created {SPLITS_CSV_PATH}")
        
    OUTPUT_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(SPLITS_CSV_PATH)
    print(f"📋 Loaded manifest mapping with {len(df)} total chunks.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using processing hardware: {device.upper()}")

    print("📥 Loading pretrained speechbrain/spkrec-ecapa-voxceleb model...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        run_opts={"device": device}
    )
    
    # 4. ✨ NEW WORKAROUND: Force bypass the torchcodec requirements flag
    if hasattr(classifier, "hparams") and hasattr(classifier.hparams, "sample_rate"):
        # Force overwrite any loader backend rules to fallback natively to standard soundfile/torchaudio
        classifier.hparams.load_with_torchcodec = False
    
    embedding_records = []

    print("\n🚀 Starting Feature Extraction across chunks...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Embeddings"):
        chunk_path = Path(row["chunk_path"])
        
        if not chunk_path.exists():
            print(f"⚠️ Warning: Missing chunk file at {chunk_path}, skipping.")
            continue
            
        try:
            # Load audio file natively via updated SpeechBrain routines
            signal, fs = torchaudio.load(str(chunk_path))
            if fs != 16000:
                signal = torchaudio.functional.resample(signal, fs, 16000)
            
            with torch.no_grad():
                embeddings = classifier.encode_batch(signal)
                embedding_vector = embeddings.squeeze().cpu().numpy()
                
                l2_norm = np.linalg.norm(embedding_vector)
                if l2_norm > 0:
                    embedding_vector = embedding_vector / l2_norm
            
            vector_filename = f"{chunk_path.stem}.npy"
            vector_save_path = OUTPUT_EMBEDDINGS_DIR / vector_filename
            np.save(vector_save_path, embedding_vector)
            
            embedding_records.append({
                "qari_id": row["qari_id"],
                "surah_name": row["surah_name"],
                "session_id": row["session_id"],
                "split": row["split"],
                "chunk_path": str(chunk_path.resolve()),
                "embedding_path": str(vector_save_path.resolve())
            })
            
        except Exception as e:
            print(f"❌ Error processing chunk {chunk_path.name}: {str(e)}")
            continue

    feature_df = pd.DataFrame(embedding_records)
    feature_csv_path = DATA_DIR / "track_a_features.csv"
    feature_df.to_csv(feature_csv_path, index=False)
    
    print("\n" + "="*50)
    print("🎉 Phase 2 - Track A Feature Extraction Complete!")
    print(f"🎯 Total embeddings cached successfully: {len(feature_df)}")
    print(f"📦 Master embedding indexing sheet saved at: {feature_csv_path.resolve()}")
    print("="*50)

if __name__ == "__main__":
    main()