# analysis/breath.py
import librosa
import numpy as np
import sys
import math
from pathlib import Path

def detect_pauses(audio_path, sr=16000, top_db=30, min_pause_sec=0.12):
    """
    Detect breathing pauses in Quranic recitation audio
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        top_db: Silence threshold in dB (higher = more sensitive, detects quieter pauses)
        min_pause_sec: Minimum pause duration to count as a breath (seconds)
    
    Returns:
        List of pause durations in seconds
    """
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
    
    # Adaptive threshold: calculate RMS energy and adjust top_db if needed
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = np.mean(rms)
    
    # If audio is very quiet, reduce threshold to detect more pauses
    # If audio is loud, increase threshold to avoid false positives
    if mean_rms < 0.01:
        top_db = min(top_db + 5, 40)  # Make more sensitive for quiet audio
    elif mean_rms > 0.1:
        top_db = max(top_db - 5, 20)  # Make less sensitive for loud audio
    
    # Split audio into non-silent segments
    non_silent_intervals = librosa.effects.split(y, top_db=top_db)
    
    # If no splits detected, try with more sensitive threshold
    if len(non_silent_intervals) <= 1:
        non_silent_intervals = librosa.effects.split(y, top_db=35)
    
    pauses = []
    for i in range(len(non_silent_intervals) - 1):
        gap_start = non_silent_intervals[i][1]
        gap_end = non_silent_intervals[i + 1][0]
        gap_duration = (gap_end - gap_start) / sr
        
        # Count pauses that are long enough but not too long (avoid counting long silences)
        if min_pause_sec <= gap_duration <= 3.0:
            pauses.append(gap_duration)

    return pauses

def compute_breath_score(user_audio_path, reference_audio_path):
    """
    Compute breath/phrasing similarity score between user and reference recitation.
    
    Returns a score 0-100 based on:
    - How similar the number of pauses is (weighted 60%)
    - How similar the average pause duration is (weighted 40%)
    
    Handles edge cases:
    - If user has NO detected pauses, score is based on partial match rather than 0
    - Soft penalty curve instead of harsh zeroing
    """
    user_pauses = detect_pauses(user_audio_path)
    ref_pauses = detect_pauses(reference_audio_path)

    user_count = len(user_pauses)
    ref_count = len(ref_pauses)

    user_avg_duration = float(np.mean(user_pauses)) if user_pauses else 0.0
    ref_avg_duration = float(np.mean(ref_pauses)) if ref_pauses else 0.0

    # ── Count-based scoring ────────────────────────────────────────────
    # If ref has no pauses at all, any count by user is a perfect match
    if ref_count == 0:
        count_score = 100.0
    else:
        # Soft penalty: score drops gradually as counts diverge
        count_diff_ratio = abs(user_count - ref_count) / max(ref_count, 1)
        count_score = 100.0 * math.exp(-0.4 * count_diff_ratio)

    # ── Duration-based scoring ─────────────────────────────────────────
    # If both have no pauses, duration match is perfect
    if ref_avg_duration == 0.0 and user_avg_duration == 0.0:
        duration_score = 100.0
    elif ref_avg_duration == 0.0:
        # User has pauses, reference doesn't — mild penalty
        duration_score = 60.0
    elif user_avg_duration == 0.0:
        # User has NO pauses but reference does
        # Check if user even had audio long enough to require pauses
        duration_score = max(30.0, 60.0 * math.exp(-0.5 * ref_count))
    else:
        duration_diff_ratio = abs(user_avg_duration - ref_avg_duration) / max(ref_avg_duration, 0.1)
        duration_score = 100.0 * math.exp(-0.5 * duration_diff_ratio)

    # ── Weighted final score ───────────────────────────────────────────
    breath_score = (count_score * 0.6) + (duration_score * 0.4)
    breath_score = min(100.0, max(0.0, breath_score))

    return {
        "breath_score": round(breath_score, 2),
        "user_pause_count": user_count,
        "reference_pause_count": ref_count,
        "user_avg_pause_duration": round(user_avg_duration, 2),
        "reference_avg_pause_duration": round(ref_avg_duration, 2)
    }

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analysis/breath.py <user_audio> <reference_audio>")
        sys.exit(1)
    result = compute_breath_score(sys.argv[1], sys.argv[2])
    print("\n🗣️ Breath Analysis Result:")
    import json
    print(json.dumps(result, indent=2))