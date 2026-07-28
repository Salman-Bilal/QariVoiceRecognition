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

Similarity method — Z-score normalisation + cosine similarity:
  Each feature is Z-score normalised using the population mean/std computed
  across ALL Qari profiles at build time.  This guarantees that:
    * Features with low inter-Qari variance (e.g. voiced_ratio) do not dominate.
    * Features with high inter-Qari variance (e.g. pitch_std_norm) contribute
      proportionally more.
    * The full 0–100 score range is exercised for every query, eliminating
      the "compressed distribution" problem seen with Gaussian decay.
  After normalisation the cosine similarity between the user Z-vector and each
  Qari Z-vector is mapped from [-1, 1] to [0, 100].
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from matching.style_profiles import (
    extract_style_features_from_file,
    load_style_profiles_and_stats,
    load_style_profiles,
    STYLE_PROFILES_PATH,
)

BASE_DIR = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loaded profiles + stats cache
# ─────────────────────────────────────────────────────────────────────────────

_style_profiles: Optional[Dict[str, Dict]] = None
_inter_qari_stats: Optional[Dict[str, Dict]] = None


def _get_profiles_and_stats() -> Tuple[Dict[str, Dict], Optional[Dict[str, Dict]]]:
    global _style_profiles, _inter_qari_stats
    if _style_profiles is None:
        _style_profiles, _inter_qari_stats = load_style_profiles_and_stats(STYLE_PROFILES_PATH)
    return _style_profiles, _inter_qari_stats


# ─────────────────────────────────────────────────────────────────────────────
# Z-score + cosine similarity helpers
# ─────────────────────────────────────────────────────────────────────────────

def _zscore_vector(
    feat_dict: Dict,
    feature_names: List[str],
    group_stats: Dict,
) -> np.ndarray:
    """
    Build a Z-score normalised feature vector from a feature dict.

    For each feature f:
        z_f = (value_f - population_mean_f) / population_std_f

    population_mean/std come from group_stats (computed across all 12 Qaris
    at profile-build time).  If stats are unavailable (legacy file), raw values
    are used unchanged.
    """
    vec = np.array([feat_dict.get(f, 0.0) for f in feature_names], dtype=float)

    if group_stats:
        means = np.array([group_stats[f]["mean"] for f in feature_names], dtype=float)
        stds  = np.array(
            [max(group_stats[f]["std"], 1e-6) for f in feature_names], dtype=float
        )
        vec = (vec - means) / stds

    return vec


def _cosine_similarity_to_score(user_vec: np.ndarray, ref_vec: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors and map to [0, 100].

    cosine ∈ [-1, 1]  →  score = (cosine + 1) / 2 × 100  ∈ [0, 100]

    A score of 100 means identical direction (perfect style match).
    A score of  50 means orthogonal (unrelated style).
    A score of   0 means opposite direction (maximally different style).
    """
    norm_u = np.linalg.norm(user_vec)
    norm_r = np.linalg.norm(ref_vec)

    if norm_u < 1e-8 or norm_r < 1e-8:
        return 50.0  # degenerate vector → neutral score

    cosine = float(np.dot(user_vec, ref_vec) / (norm_u * norm_r))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float((cosine + 1.0) / 2.0 * 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Per-component similarity functions
# ─────────────────────────────────────────────────────────────────────────────

# Features used for each component (must match keys in the profile dicts)
PITCH_FEATURES  = ["pitch_std_norm", "pitch_range_norm", "pitch_p25", "pitch_p75", "voiced_ratio"]
RHYTHM_FEATURES = ["onset_rate_hz", "ioi_mean_sec", "ioi_cv"]
BREATH_FEATURES = ["pause_rate", "pause_ratio", "avg_pause_sec", "phrase_pause_ratio"]


def compute_pitch_similarity(
    user_pitch: Dict,
    ref_pitch: Dict,
    pitch_stats: Optional[Dict] = None,
) -> float:
    """
    Compare pitch features via Z-score normalised cosine similarity.

    Features:
      - pitch_std_norm   — how much pitch variation (vibrato vs flat delivery)
      - pitch_range_norm — melodic range width
      - pitch_p25        — lower quartile of pitch distribution
      - pitch_p75        — upper quartile of pitch distribution
      - voiced_ratio     — fraction of time spent on voiced sound

    pitch_mean_norm is excluded — it's always ~1.0 after per-speaker
    normalisation and provides no discriminative value.
    """
    user_vec = _zscore_vector(user_pitch, PITCH_FEATURES, pitch_stats or {})
    ref_vec  = _zscore_vector(ref_pitch,  PITCH_FEATURES, pitch_stats or {})
    return _cosine_similarity_to_score(user_vec, ref_vec)


def compute_rhythm_similarity(
    user_rhythm: Dict,
    ref_rhythm: Dict,
    rhythm_stats: Optional[Dict] = None,
) -> float:
    """
    Compare rhythm features via Z-score normalised cosine similarity.

    Features:
      - onset_rate_hz   — syllables/onsets per second (recitation speed)
      - ioi_mean_sec    — average time between syllables
      - ioi_cv          — rhythm regularity (murattal=regular, mujawwad=free)

    tempo_bpm and ioi_std excluded — strongly correlated with the above,
    would double-count those effects.
    """
    user_vec = _zscore_vector(user_rhythm, RHYTHM_FEATURES, rhythm_stats or {})
    ref_vec  = _zscore_vector(ref_rhythm,  RHYTHM_FEATURES, rhythm_stats or {})
    return _cosine_similarity_to_score(user_vec, ref_vec)


def compute_breath_similarity(
    user_breath: Dict,
    ref_breath: Dict,
    breath_stats: Optional[Dict] = None,
) -> float:
    """
    Compare breath/phrasing features via Z-score normalised cosine similarity.

    Features:
      - pause_rate          — pauses per second (phrasing density)
      - pause_ratio         — fraction of time silent
      - avg_pause_sec       — how long each pause is
      - phrase_pause_ratio  — talking-vs-pausing balance
    """
    user_vec = _zscore_vector(user_breath, BREATH_FEATURES, breath_stats or {})
    ref_vec  = _zscore_vector(ref_breath,  BREATH_FEATURES, breath_stats or {})
    return _cosine_similarity_to_score(user_vec, ref_vec)


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison function
# ─────────────────────────────────────────────────────────────────────────────

# Weights for the three components (must sum to 1.0)
PITCH_WEIGHT  = 0.50
RHYTHM_WEIGHT = 0.30
BREATH_WEIGHT = 0.20


def compare_style_to_all_qaris(audio_path: str) -> List[Dict]:
    """
    Compare a user's recitation audio against all Qari style profiles.

    Uses Z-score normalised cosine similarity so that the full 0–100 score
    range is always exercised and low-variance features do not suppress the
    spread of the distribution.

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

    # 2. Load Qari profiles + inter-Qari statistics
    profiles, stats = _get_profiles_and_stats()
    if not profiles:
        raise RuntimeError(
            "No Qari style profiles found. Run: python matching/build_style_profiles.py"
        )

    if stats is None:
        print(
            "[WARN] Inter-Qari statistics not found in profile file.\n"
            "       Falling back to raw cosine similarity (no Z-score normalisation).\n"
            "       Re-run: python matching/build_style_profiles.py"
        )

    pitch_stats  = stats["pitch"]  if stats else None
    rhythm_stats = stats["rhythm"] if stats else None
    breath_stats = stats["breath"] if stats else None

    # 3. Compare against each Qari
    results = []
    for qari_name, profile in profiles.items():
        pitch_score  = compute_pitch_similarity(user_pitch,  profile["pitch"],  pitch_stats)
        rhythm_score = compute_rhythm_similarity(user_rhythm, profile["rhythm"], rhythm_stats)
        breath_score = compute_breath_similarity(user_breath, profile["breath"], breath_stats)

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
            key=lambda x: x[1],
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
