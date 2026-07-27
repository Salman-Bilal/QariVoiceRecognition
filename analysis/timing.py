# analysis/timing.py
import librosa
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import sys
from pathlib import Path

def extract_mfcc_sequence(audio_path, sr=16000, n_mfcc=13):
    path_obj = Path(audio_path)
    
    # If it's a relative path, check if it belongs to the project root instead of the current terminal folder
    if not path_obj.is_absolute():
        base_dir = Path(__file__).resolve().parent.parent
        possible_root_path = base_dir / audio_path
        if possible_root_path.exists():
            path_obj = possible_root_path

    abs_path = str(path_obj.resolve())
    if not Path(abs_path).exists():
        raise FileNotFoundError(f"❌ Cannot find audio file at: {abs_path}")
        
    y, _ = librosa.load(abs_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # shape: (time_frames, n_mfcc)

# analysis/timing.py — replace the deviation calculation
def compute_timing_score(user_audio_path, reference_audio_path):
    user_seq = extract_mfcc_sequence(user_audio_path)
    ref_seq = extract_mfcc_sequence(reference_audio_path)

    distance, path = fastdtw(user_seq, ref_seq, dist=euclidean)

    path_arr = np.array(path)
    user_idx = path_arr[:, 0] / max(path_arr[:, 0].max(), 1)
    ref_idx = path_arr[:, 1] / max(path_arr[:, 1].max(), 1)

    # Deviation = how far the warping path strays from mapping user_idx == ref_idx
    deviation = np.mean(np.abs(user_idx - ref_idx))

    # Also normalize the raw DTW distance itself by sequence length,
    # so timing score reflects actual alignment quality, not just path shape
    avg_len = (len(user_seq) + len(ref_seq)) / 2
    normalized_distance = distance / max(avg_len, 1)

    # Combine both signals: path shape deviation AND raw alignment cost
    shape_score = max(0, 100 * (1 - deviation / 0.3))
    cost_score = max(0, 100 * (1 - normalized_distance / 700.0))  # tune 50.0 based on calibration

    timing_score = min(shape_score, cost_score)  # take the more conservative of the two
    timing_score = min(100, max(0, timing_score))

    return {
        "timing_score": round(timing_score, 2),
        "dtw_distance": round(float(distance), 2),
        "normalized_distance": round(float(normalized_distance), 4),
        "path_deviation": round(float(deviation), 4)
    }
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analysis/timing.py <user_audio> <reference_audio>")
        sys.exit(1)
        
    result = compute_timing_score(sys.argv[1], sys.argv[2])
    print("\n⏱️ Timing Analysis Result:")
    import json
    print(json.dumps(result, indent=2))