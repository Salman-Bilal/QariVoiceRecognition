# analysis/melody.py
import librosa
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import sys
from pathlib import Path

def extract_pitch_contour(audio_path, sr=16000):
    path_obj = Path(audio_path)
    if not path_obj.is_absolute():
        base_dir = Path(__file__).resolve().parent.parent
        possible_root_path = base_dir / audio_path
        if possible_root_path.exists():
            path_obj = possible_root_path

    abs_path = str(path_obj.resolve())
    if not Path(abs_path).exists():
        raise FileNotFoundError(f"❌ Cannot find audio file at: {abs_path}")

    y, _ = librosa.load(abs_path, sr=sr)
    pitch, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
    )
    pitch = np.nan_to_num(pitch)

    voiced_pitch = pitch[pitch > 0]
    if len(voiced_pitch) > 0:
        mean_pitch = voiced_pitch.mean()
        std_pitch = voiced_pitch.std() if voiced_pitch.std() > 0 else 1.0
        pitch_normalized = (pitch - mean_pitch) / std_pitch
    else:
        pitch_normalized = pitch

    return pitch_normalized.reshape(-1, 1)

def compute_melody_score(user_audio_path, reference_audio_path):
    user_contour = extract_pitch_contour(user_audio_path)
    ref_contour = extract_pitch_contour(reference_audio_path)

    distance, path = fastdtw(user_contour, ref_contour, dist=euclidean)
    normalized_distance = distance / max(len(path), 1)

    melody_score = max(0, 100 * (1 - normalized_distance / 3.0))
    melody_score = min(100, melody_score)

    return {
        "melody_score": round(melody_score, 2),
        "normalized_dtw_distance": round(float(normalized_distance), 4)
    }

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analysis/melody.py <user_audio> <reference_audio>")
        sys.exit(1)
    result = compute_melody_score(sys.argv[1], sys.argv[2])
    print("\n🎵 Melody Analysis Result:")
    import json
    print(json.dumps(result, indent=2))