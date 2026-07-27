"""
matching/style_similarity.py

Core recitation style similarity engine.

Given a user's audio file, this module:
  1. Extracts the user's recitation style features (pitch, rhythm, breath)
  2. Compares them against each Qari's prebuilt style profile
  3. Returns a ranked list of Qaris with similarity scores (0–100%)

This replaces ECAPA-based speaker identity matching.
The scores now reflect HOW similarly someone recites, not who they sound like biometrically.

Score components (weighted):
  - Pitch similarity   40% — melody contour, vocal range, voiced fraction
  - Rhythm similarity  35% — tempo, syllable rate, rhythmic regularity
  - Breath similarity  25% — phrasing density, pause pattern, phrase length

Why these weights?
  Pitch is the most distinctive aspect of a Qari's style (mujawwad vs. murattal,
  maqam color, melodic movement). Rhythm is slightly less weighted because users
  often recite at different speeds. Breath is the lightest weight as it varies
  the most with recording length.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional

from matching.style_profiles import (
    extract_style_features_from_file,
    load_style_profiles,
    STYLE_PROFILES_PATH,
)

BASE_DIR = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loaded profiles cache
# ─────────────────────────────────────────────────────────────────────────────

_style_profiles: Optional[Dict[str, Dict]] = None


def _get_profiles() -> Dict[str, Dict]:
    global _style_profiles
    if _style_profiles is None:
        _style_profiles = load_style_profiles(STYLE_PROFILES_PATH)
    return _style_profiles


# ─────────────────────────────────────────────────────────────────────────────
# Per-feature similarity functions
# ─────────────────────────────────────────────────────────────────────────────

def _feature_similarity(user_val: float, ref_val: float, scale: float) -> float:
    """
    Compute similarity for a single scalar feature using a Gaussian decay.

    similarity = exp( -(user_val - ref_val)^2 / (2 * scale^2) ) * 100

    `scale` controls tolerance: small scale = strict matching, large scale = lenient.
    Result is 0–100.
    """
    diff = user_val - ref_val
    return float(100.0 * np.exp(-(diff ** 2) / (2.0 * scale ** 2)))


def compute_pitch_similarity(user_pitch: Dict, ref_pitch: Dict) -> float:
    """
    Compare user pitch features to a Qari's reference pitch profile.

    Features and their tolerances (scale):
      - pitch_std_norm   (0.15) — how much pitch variation; key style indicator
      - pitch_range_norm (0.25) — melodic range width
      - pitch_p25        (0.20) — lower quartile of pitch distribution
      - pitch_p75        (0.20) — upper quartile of pitch distribution
      - voiced_ratio     (0.15) — fraction of time spent on voiced sound

    pitch_mean_norm is intentionally excluded — it's always ~1.0 after
    normalization and adds no discriminative value.
    """
    scores = [
        _feature_similarity(user_pitch["pitch_std_norm"],   ref_pitch["pitch_std_norm"],   0.15),
        _feature_similarity(user_pitch["pitch_range_norm"], ref_pitch["pitch_range_norm"],  0.25),
        _feature_similarity(user_pitch["pitch_p25"],        ref_pitch["pitch_p25"],         0.20),
        _feature_similarity(user_pitch["pitch_p75"],        ref_pitch["pitch_p75"],         0.20),
        _feature_similarity(user_pitch["voiced_ratio"],     ref_pitch["voiced_ratio"],      0.15),
    ]
    return float(np.mean(scores))


def compute_rhythm_similarity(user_rhythm: Dict, ref_rhythm: Dict) -> float:
    """
    Compare user rhythm features to a Qari's reference rhythm profile.

    Features and tolerances:
      - onset_rate_hz   (1.5)  — syllables/onsets per second (recitation speed)
      - ioi_mean_sec    (0.08) — average time between syllables
      - ioi_cv          (0.20) — rhythm regularity (murattal=regular, mujawwad=free)

    tempo_bpm excluded because it correlates strongly with onset_rate and
    ioi_mean — including it would double-count those effects.
    ioi_std excluded to avoid redundancy with ioi_cv.
    """
    scores = [
        _feature_similarity(user_rhythm["onset_rate_hz"], ref_rhythm["onset_rate_hz"], 1.5),
        _feature_similarity(user_rhythm["ioi_mean_sec"],  ref_rhythm["ioi_mean_sec"],  0.08),
        _feature_similarity(user_rhythm["ioi_cv"],        ref_rhythm["ioi_cv"],        0.20),
    ]
    return float(np.mean(scores))


def compute_breath_similarity(user_breath: Dict, ref_breath: Dict) -> float:
    """
    Compare user breath/phrasing features to a Qari's reference breath profile.

    Features and tolerances:
      - pause_rate          (0.10) — pauses per second (phrasing density)
      - pause_ratio         (0.08) — fraction of time silent
      - avg_pause_sec       (0.25) — how long each pause is
      - phrase_pause_ratio  (2.0)  — talking-vs-pausing balance
    """
    scores = [
        _feature_similarity(user_breath["pause_rate"],         ref_breath["pause_rate"],         0.10),
        _feature_similarity(user_breath["pause_ratio"],        ref_breath["pause_ratio"],         0.08),
        _feature_similarity(user_breath["avg_pause_sec"],      ref_breath["avg_pause_sec"],       0.25),
        _feature_similarity(user_breath["phrase_pause_ratio"], ref_breath["phrase_pause_ratio"],  2.0),
    ]
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison function
# ─────────────────────────────────────────────────────────────────────────────

# Weights for the three components (must sum to 1.0)
PITCH_WEIGHT  = 0.40
RHYTHM_WEIGHT = 0.35
BREATH_WEIGHT = 0.25


def compare_style_to_all_qaris(audio_path: str) -> List[Dict]:
    """
    Compare a user's recitation audio against all Qari style profiles.

    Args:
        audio_path: Path to the user's audio file (wav, mp3, m4a, flac supported)

    Returns:
        List of dicts, sorted by overall_score descending:
        [
          {
            "qari":              str,   # Qari name
            "overall_score":     float, # 0–100, weighted combination
            "pitch_score":       float, # 0–100
            "rhythm_score":      float, # 0–100
            "breath_score":      float, # 0–100
            "match_level":       str,   # "Excellent" / "Good" / "Moderate" / "Weak"
            "style_description": str,   # Human-readable description
          },
          ...
        ]

    Raises:
        FileNotFoundError: if audio_path does not exist
        RuntimeError: if style profiles have not been built yet
    """
    # 1. Extract user style features
    user_features = extract_style_features_from_file(audio_path)
    if user_features is None:
        raise ValueError(
            f"Could not extract style features from: {audio_path}\n"
            "Check that the file is a valid audio file with at least 2 seconds of content."
        )

    user_pitch  = user_features["pitch"]
    user_rhythm = user_features["rhythm"]
    user_breath = user_features["breath"]

    # 2. Load Qari profiles
    profiles = _get_profiles()
    if not profiles:
        raise RuntimeError(
            "No Qari style profiles found. Run: python matching/build_style_profiles.py"
        )

    # 3. Compare against each Qari
    results = []
    for qari_name, profile in profiles.items():
        pitch_score  = compute_pitch_similarity(user_pitch,  profile["pitch"])
        rhythm_score = compute_rhythm_similarity(user_rhythm, profile["rhythm"])
        breath_score = compute_breath_similarity(user_breath, profile["breath"])

        overall = (
            pitch_score  * PITCH_WEIGHT +
            rhythm_score * RHYTHM_WEIGHT +
            breath_score * BREATH_WEIGHT
        )
        overall = round(float(np.clip(overall, 0.0, 100.0)), 2)

        # Match level label
        if overall >= 72:
            match_level = "Excellent Match"
        elif overall >= 55:
            match_level = "Good Match"
        elif overall >= 35:
            match_level = "Moderate Match"
        else:
            match_level = "Weak Match"

        # Style description based on dominant matching feature
        dominant = max(
            [("pitch", pitch_score), ("rhythm", rhythm_score), ("breath", breath_score)],
            key=lambda x: x[1]
        )
        desc_map = {
            "pitch":  "Your melody and pitch contour closely resemble this Qari's style.",
            "rhythm": "Your recitation rhythm and syllable timing closely resemble this Qari.",
            "breath": "Your phrasing and breath pattern closely resemble this Qari.",
        }
        style_description = desc_map[dominant[0]]

        results.append({
            "qari":              qari_name,
            "overall_score":     overall,
            "pitch_score":       round(float(pitch_score),  2),
            "rhythm_score":      round(float(rhythm_score), 2),
            "breath_score":      round(float(breath_score), 2),
            "match_level":       match_level,
            "style_description": style_description,
        })

    # 4. Sort by overall score, best first
    results.sort(key=lambda x: x["overall_score"], reverse=True)

    return results


def get_top_style_matches(audio_path: str, top_n: int = 5) -> List[Dict]:
    """
    Convenience wrapper — returns only the top N style matches.
    Same output format as compare_style_to_all_qaris().
    """
    all_results = compare_style_to_all_qaris(audio_path)
    return all_results[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# CLI diagnostic tool
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python matching/style_similarity.py <path_to_audio>")
        sys.exit(1)

    audio = sys.argv[1]
    print(f"\nAnalyzing recitation style: {audio}")
    print("Extracting features and comparing against all Qaris...\n")

    try:
        results = compare_style_to_all_qaris(audio)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"{'Rank':<5} {'Qari':<30} {'Overall':>8} {'Pitch':>7} {'Rhythm':>8} {'Breath':>8}  Level")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        print(
            f"{i:<5} {r['qari']:<30} "
            f"{r['overall_score']:>7.1f}% "
            f"{r['pitch_score']:>6.1f}% "
            f"{r['rhythm_score']:>7.1f}% "
            f"{r['breath_score']:>7.1f}%  "
            f"{r['match_level']}"
        )

    print("\nBest match:", results[0]["qari"], f"-- {results[0]['overall_score']}%")
    print(f"   {results[0]['style_description']}")
