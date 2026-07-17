# preprocessing/segment.py
import os
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path

WINDOW_SEC = 4
HOP_SEC = 2
SR = 16000

def segment_audio(processed_dir, chunks_dir):
    os.makedirs(chunks_dir, exist_ok=True)
    manifest_rows = []

    window_samples = int(WINDOW_SEC * SR)
    hop_samples = int(HOP_SEC * SR)

    qari_folders = sorted(os.listdir(processed_dir))
    for qari_id in qari_folders:
        qari_path = os.path.join(processed_dir, qari_id)
        surah_files = sorted(os.listdir(qari_path))

        for surah_file in surah_files:
            surah_name = Path(surah_file).stem
            session_id = f"{qari_id}_{surah_name}"  # one session per surah recording
            wav_path = os.path.join(qari_path, surah_file)

            y, sr = librosa.load(wav_path, sr=SR)
            total_samples = len(y)

            chunk_idx = 0
            start = 0
            while start + window_samples <= total_samples:
                chunk = y[start:start + window_samples]

                chunk_filename = f"{session_id}_chunk{chunk_idx:03d}.wav"
                qari_chunk_dir = os.path.join(chunks_dir, qari_id)
                os.makedirs(qari_chunk_dir, exist_ok=True)
                chunk_path = os.path.join(qari_chunk_dir, chunk_filename)

                sf.write(chunk_path, chunk, SR, subtype="PCM_16")

                manifest_rows.append({
                    "qari_id": qari_id,
                    "surah_name": surah_name,
                    "session_id": session_id,
                    "chunk_start_time": start / SR,
                    "chunk_path": chunk_path
                })

                chunk_idx += 1
                start += hop_samples

    manifest_df = pd.DataFrame(manifest_rows)
    return manifest_df