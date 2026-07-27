"""
matching/style_profiles.py

Extracts and stores recitation STYLE profiles for each Qari.
A style profile captures HOW a Qari recites — not who they are as a person.

Three features per Qari (averaged across all their Surahs):
  1. Pitch contour statistics  — shape of the melody (mean, std, percentiles of pitch)
  2. Onset rhythm envelope     — syllable/note timing pattern (tempo, inter-onset intervals)
  3. Breath/pause pattern      — where pauses occur, how long, phrasing density

These are deliberately speaker-independent:
  - Pitch is normalized relative to each speaker's own mean (speaker-agnostic)
  - Rhythm is measured in time ratios, not absolute Hz or energy
  - Breath is counted as ratios, not raw silence lengths

This makes the comparison fair: a user with a deep voice can still match
the style of a Qari with a high voice, as long as the recitation pattern matches.
"""

import numpy as np
import librosa
import pickle
from pathlib import Path
from typing import Dict, List, Optional


BASE_DIR = Path(__file__).resolve().parent.parent
NORMALIZED_DIR = BASE_DIR / "dataset" / "processed" / "normalized"
STYLE_PROFILES_PATH = BASE_DIR / "matching" / "style_profiles.pkl"

SR = 16000  # Target sample rate


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extractors
# ─────────────────────────────────────────────────────────────────────────────

def extract_pitch_features(y: np.ndarray, sr: int = SR) -> Dict:
    """
    Extract speaker-normalized pitch statistics.

    Uses pyin (probabilistic YIN) to estimate fundamental frequency (F0).
    All pitch values are normalized relative to the speaker's own median —
    so a high-pitched voice and low-pitched voice can still compare their
    melodic contour shape fairly.

    Returns a feature dict with:
      - pitch_mean_norm    : mean of the speaker-normalized voiced pitch
      - pitch_std_norm     : std dev (how much pitch varies — vibrato vs flat)
      - pitch_range_norm   : max - min of normalized pitch (melodic range)
      - pitch_p25/p75      : quartiles (shape of pitch distribution)
      - voiced_ratio       : fraction of frames that are voiced (0.0–1.0)
    """
    pitch, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr
    )

    voiced_pitch = pitch[voiced_flag & (pitch > 0)]

    if len(voiced_pitch) < 10:
        # Not enough voiced frames — return neutral profile
        return {
            "pitch_mean_norm": 0.0,
            "pitch_std_norm": 0.0,
            "pitch_range_norm": 0.0,
            "pitch_p25": 0.0,
            "pitch_p75": 0.0,
            "voiced_ratio": 0.0,
        }

    # Normalize relative to speaker's own median → speaker-agnostic shape
    median_pitch = float(np.median(voiced_pitch))
    if median_pitch > 0:
        norm_pitch = voiced_pitch / median_pitch
    else:
        norm_pitch = voiced_pitch

    total_frames = len(pitch)
    voiced_ratio = len(voiced_pitch) / total_frames if total_frames > 0 else 0.0

    return {
        "pitch_mean_norm": float(np.mean(norm_pitch)),
        "pitch_std_norm": float(np.std(norm_pitch)),
        "pitch_range_norm": float(np.ptp(norm_pitch)),          # peak-to-peak
        "pitch_p25": float(np.percentile(norm_pitch, 25)),
        "pitch_p75": float(np.percentile(norm_pitch, 75)),
        "voiced_ratio": float(voiced_ratio),
    }


def extract_rhythm_features(y: np.ndarray, sr: int = SR) -> Dict:
    """
    Extract tempo and onset (syllable) rhythm features.

    Onset strength measures when new sounds/syllables begin.
    We capture:
      - tempo_bpm          : estimated beats per minute
      - onset_rate_hz      : onsets per second (recitation speed)
      - ioi_mean_sec       : mean inter-onset interval (avg time between syllables)
      - ioi_std_sec        : std dev of IOI (consistency of rhythm)
      - ioi_cv             : coefficient of variation = std/mean (relative rhythm regularity)

    ioi_cv is the most important: a low value means very regular/metronomic rhythm,
    a high value means rhythmically free (mujawwad-style).
    """
    # Onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    # Onset frame positions → times
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, units="time",
        backtrack=True
    )

    if len(onset_frames) < 2:
        return {
            "tempo_bpm": tempo,
            "onset_rate_hz": 0.0,
            "ioi_mean_sec": 0.0,
            "ioi_std_sec": 0.0,
            "ioi_cv": 0.0,
        }

    duration = len(y) / sr
    onset_rate = len(onset_frames) / duration if duration > 0 else 0.0

    ioi = np.diff(onset_frames)  # inter-onset intervals in seconds
    ioi_mean = float(np.mean(ioi))
    ioi_std = float(np.std(ioi))
    ioi_cv = ioi_std / ioi_mean if ioi_mean > 0 else 0.0

    return {
        "tempo_bpm": tempo,
        "onset_rate_hz": float(onset_rate),
        "ioi_mean_sec": ioi_mean,
        "ioi_std_sec": ioi_std,
        "ioi_cv": float(ioi_cv),
    }


def extract_breath_features(y: np.ndarray, sr: int = SR) -> Dict:
    """
    Extract breathing/phrasing pattern features.

    Pauses are silences where the reciter breathes or phrases their recitation.
    We capture:
      - pause_rate         : pauses per second (phrasing density)
      - pause_ratio        : fraction of total audio that is silence
      - avg_pause_sec      : mean pause duration
      - avg_phrase_sec     : mean spoken phrase duration between pauses
      - phrase_pause_ratio : avg_phrase_sec / avg_pause_sec (talking vs. pausing balance)
    """
    # Adaptive threshold based on RMS
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms))
    top_db = 28 if mean_rms < 0.01 else 32

    non_silent = librosa.effects.split(y, top_db=top_db)

    duration = len(y) / sr if sr > 0 else 1.0

    if len(non_silent) <= 1:
        # No or minimal pauses detected
        return {
            "pause_rate": 0.0,
            "pause_ratio": 0.0,
            "avg_pause_sec": 0.0,
            "avg_phrase_sec": float(duration),
            "phrase_pause_ratio": 10.0,  # Almost all speaking
        }

    pauses = []
    phrases = []

    for i, (start, end) in enumerate(non_silent):
        phrase_duration = (end - start) / sr
        phrases.append(phrase_duration)

        if i < len(non_silent) - 1:
            gap_start = non_silent[i][1]
            gap_end = non_silent[i + 1][0]
            gap_sec = (gap_end - gap_start) / sr
            if 0.10 <= gap_sec <= 4.0:  # Valid breath pause range
                pauses.append(gap_sec)

    total_pause_time = sum(pauses)
    pause_rate = len(pauses) / duration if duration > 0 else 0.0
    pause_ratio = total_pause_time / duration if duration > 0 else 0.0
    avg_pause = float(np.mean(pauses)) if pauses else 0.0
    avg_phrase = float(np.mean(phrases)) if phrases else float(duration)

    phrase_pause_ratio = avg_phrase / avg_pause if avg_pause > 0 else 10.0
    phrase_pause_ratio = min(phrase_pause_ratio, 20.0)  # cap to avoid outliers

    return {
        "pause_rate": float(pause_rate),
        "pause_ratio": float(pause_ratio),
        "avg_pause_sec": avg_pause,
        "avg_phrase_sec": avg_phrase,
        "phrase_pause_ratio": float(phrase_pause_ratio),
    }


def extract_style_features_from_file(audio_path: str) -> Optional[Dict]:
    """
    Load an audio file and extract all three style feature groups.
    Returns None if the file cannot be loaded or has too little content.
    """
    try:
        y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
    except Exception as e:
        print(f"    [WARN] Could not load {audio_path}: {e}")
        return None

    if len(y) < SR * 2:  # Need at least 2 seconds
        print(f"    [WARN] Audio too short (<2s): {audio_path}")
        return None

    # Peak-normalize so loudness doesn't affect feature magnitudes
    peak = float(np.abs(y).max())
    if peak > 0:
        y = y / peak

    pitch_feats = extract_pitch_features(y, SR)
    rhythm_feats = extract_rhythm_features(y, SR)
    breath_feats = extract_breath_features(y, SR)

    return {
        "pitch": pitch_feats,
        "rhythm": rhythm_feats,
        "breath": breath_feats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Profile Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_style_profile_for_qari(qari_name: str) -> Optional[Dict]:
    """
    Build a style profile for a single Qari by averaging features
    across all their available Surah recordings.

    Returns a dict with keys: "pitch", "rhythm", "breath", each being
    a dict of averaged feature values. Returns None if no audio found.
    """
    qari_dir = NORMALIZED_DIR / qari_name
    if not qari_dir.exists():
        print(f"  [ERROR] Directory not found: {qari_dir}")
        return None

    wav_files = sorted(qari_dir.glob("*.wav"))
    if not wav_files:
        print(f"  [ERROR] No .wav files found for: {qari_name}")
        return None

    print(f"  [DIR] {qari_name}: processing {len(wav_files)} Surah(s)...")

    all_pitch: List[Dict] = []
    all_rhythm: List[Dict] = []
    all_breath: List[Dict] = []

    for wav_path in wav_files:
        print(f"      -> {wav_path.name}")
        feats = extract_style_features_from_file(str(wav_path))
        if feats is None:
            continue
        all_pitch.append(feats["pitch"])
        all_rhythm.append(feats["rhythm"])
        all_breath.append(feats["breath"])

    if not all_pitch:
        print(f"  [ERROR] No valid features extracted for: {qari_name}")
        return None

    def avg_dicts(dict_list: List[Dict]) -> Dict:
        """Average all numeric values across a list of same-keyed dicts."""
        keys = dict_list[0].keys()
        return {
            k: float(np.mean([d[k] for d in dict_list]))
            for k in keys
        }

    profile = {
        "qari_name": qari_name,
        "num_surahs": len(all_pitch),
        "pitch": avg_dicts(all_pitch),
        "rhythm": avg_dicts(all_rhythm),
        "breath": avg_dicts(all_breath),
    }

    return profile


def build_all_style_profiles() -> Dict[str, Dict]:
    """
    Build style profiles for all Qaris found in the normalized dataset directory.
    Returns a dict: { qari_name: style_profile }
    """
    if not NORMALIZED_DIR.exists():
        raise FileNotFoundError(
            f"Normalized audio directory not found: {NORMALIZED_DIR}\n"
            "Run the preprocessing pipeline first."
        )

    # Find all Qari subdirectories (exclude loose files at root level)
    qari_dirs = sorted([d for d in NORMALIZED_DIR.iterdir() if d.is_dir()])

    if not qari_dirs:
        raise FileNotFoundError(
            f"No Qari subdirectories found in {NORMALIZED_DIR}"
        )

    print(f"\nBuilding style profiles for {len(qari_dirs)} Qaris...\n")

    profiles: Dict[str, Dict] = {}

    for qari_dir in qari_dirs:
        qari_name = qari_dir.name
        profile = build_style_profile_for_qari(qari_name)
        if profile is not None:
            profiles[qari_name] = profile
            print(f"  [OK] {qari_name} -- profile built from {profile['num_surahs']} Surah(s)\n")
        else:
            print(f"  [SKIP] {qari_name} -- no valid audio\n")

    return profiles


def compute_inter_qari_stats(profiles: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Compute the mean and standard deviation of each style feature ACROSS all
    Qari profiles.  These population-level statistics are used at inference
    time to Z-score-normalise user features so that the full similarity range
    is always exercised, regardless of how tightly individual feature values
    cluster.

    Returns a nested dict:
        {
          "pitch":  { feature_name: {"mean": float, "std": float}, ... },
          "rhythm": { ... },
          "breath": { ... },
        }
    """
    if not profiles:
        raise ValueError("Cannot compute stats: no profiles provided.")

    groups = ("pitch", "rhythm", "breath")
    stats: Dict[str, Dict] = {g: {} for g in groups}

    for group in groups:
        # Collect every feature name from the first valid profile
        sample_profile = next(iter(profiles.values()))
        feature_names = list(sample_profile[group].keys())

        for feat in feature_names:
            values = np.array(
                [p[group][feat] for p in profiles.values() if feat in p[group]],
                dtype=float,
            )
            feat_mean = float(np.mean(values))
            feat_std  = float(np.std(values))
            # Floor std to avoid division-by-zero for degenerate features
            feat_std  = max(feat_std, 1e-6)
            stats[group][feat] = {"mean": feat_mean, "std": feat_std}

    return stats


def save_style_profiles(
    profiles: Dict[str, Dict],
    stats: Dict[str, Dict],
    output_path: Path = STYLE_PROFILES_PATH,
) -> None:
    """
    Serialize style profiles AND inter-Qari feature statistics to disk.

    The file stores a dict with two keys:
        "profiles" -> { qari_name: profile_dict }
        "stats"    -> { group: { feature: {"mean": float, "std": float} } }
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"profiles": profiles, "stats": stats}
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nStyle profiles + inter-Qari stats saved -> {output_path}")


def load_style_profiles(profiles_path: Path = STYLE_PROFILES_PATH) -> Dict[str, Dict]:
    """
    Load pre-built style profiles from disk.

    Supports both the new bundle format (dict with "profiles" / "stats" keys)
    and the old flat format (plain dict of qari_name → profile) for backward
    compatibility.  Always returns just the profiles dict; use
    load_style_profiles_and_stats() to also get the statistics.
    """
    if not profiles_path.exists():
        raise FileNotFoundError(
            f"Style profiles not found at {profiles_path}\n"
            "Run: python matching/build_style_profiles.py"
        )
    with open(profiles_path, "rb") as f:
        data = pickle.load(f)

    # New bundle format
    if isinstance(data, dict) and "profiles" in data:
        return data["profiles"]

    # Legacy flat format — return as-is
    return data


def load_style_profiles_and_stats(
    profiles_path: Path = STYLE_PROFILES_PATH,
) -> tuple:
    """
    Load pre-built style profiles AND inter-Qari feature statistics.

    Returns:
        (profiles: Dict[str, Dict], stats: Dict[str, Dict])

    stats will be None if the file was saved in the old format (rebuild recommended).
    """
    if not profiles_path.exists():
        raise FileNotFoundError(
            f"Style profiles not found at {profiles_path}\n"
            "Run: python matching/build_style_profiles.py"
        )
    with open(profiles_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "profiles" in data:
        return data["profiles"], data.get("stats", None)

    # Legacy format — no stats stored
    return data, None
