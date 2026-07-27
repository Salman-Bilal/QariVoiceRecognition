import os
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path

WINDOW_SEC = 4
HOP_SEC = 2
SR = 16000

def segment_audio(processed_dir, chunks_dir):
    processed_dir = Path(processed_dir)
    chunks_dir = Path(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_rows = []
    window_samples = int(WINDOW_SEC * SR)
    hop_samples = int(HOP_SEC * SR)

    qari_folders = sorted([f for f in processed_dir.iterdir() if f.is_dir()])
    for qari_folder in qari_folders:
        qari_id = qari_folder.name
        surah_files = sorted(list(qari_folder.glob("*.wav")))

        for surah_file in surah_files:
            surah_name = surah_file.stem
            session_id = f"{qari_id}_{surah_name}"  
            
            y, sr = librosa.load(str(surah_file), sr=SR)
            total_samples = len(y)

            chunk_idx = 0
            start = 0
            while start + window_samples <= total_samples:
                chunk = y[start:start + window_samples]

                chunk_filename = f"{session_id}_chunk{chunk_idx:03d}.wav"
                qari_chunk_dir = chunks_dir / qari_id
                qari_chunk_dir.mkdir(parents=True, exist_ok=True)
                chunk_path = qari_chunk_dir / chunk_filename

                sf.write(str(chunk_path), chunk, SR, subtype="PCM_16")

                manifest_rows.append({
                    "qari_id": qari_id,
                    "surah_name": surah_name,
                    "session_id": session_id,
                    "chunk_start_time": start / SR,
                    "chunk_path": str(chunk_path.resolve())
                })

                chunk_idx += 1
                start += hop_samples

    manifest_df = pd.DataFrame(manifest_rows)
    return manifest_df