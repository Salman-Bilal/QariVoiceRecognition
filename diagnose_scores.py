"""
diagnose_scores.py
------------------
Run this script to see the REAL raw cosine scores your model produces.
This tells us exactly what thresholds to use for the similarity percentage.

Usage:
    python diagnose_scores.py <path_to_audio_file>

Example:
    python diagnose_scores.py "dataset/raw/Mishary Al-Fasay/1_Surah_Fatiha.wav"
    python diagnose_scores.py "Recording (13).wav"
"""

import sys
import numpy as np
import pickle
import torch
import torchaudio
from pathlib import Path

# Patch for SpeechBrain compatibility
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from matching.similarity import extract_user_embedding, _load_reference_embeddings

def cosine_similarity(a, b):
    return float(np.dot(a, b))

def diagnose(audio_path: str):
    print(f"\n{'='*60}")
    print(f"  DIAGNOSTIC SCORE REPORT")
    print(f"  File: {audio_path}")
    print(f"{'='*60}\n")

    print("Loading model and reference embeddings...")
    refs = _load_reference_embeddings()

    print(f"Extracting embedding from your audio file...")
    user_emb = extract_user_embedding(audio_path)

    print(f"\nRaw cosine scores (L2-normalized, so max possible = 1.0):\n")
    print(f"  {'Qari':<30} {'Raw Score':>12}  {'Old % (×100)':>14}  {'New % (calibrated)':>20}")
    print(f"  {'-'*30} {'-'*12}  {'-'*14}  {'-'*20}")

    scores = {}
    for qari, ref_emb in refs.items():
        score = cosine_similarity(user_emb, ref_emb)
        scores[qari] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    min_s = min(scores.values())
    max_s = max(scores.values())

    ABS_MIN = 0.10
    ABS_MAX = 0.30

    for qari, score in ranked:
        old_pct    = score * 100
        new_pct    = max(0.0, min(100.0, ((score - ABS_MIN) / (ABS_MAX - ABS_MIN)) * 100))
        print(f"  {qari:<30} {score:>12.4f}  {old_pct:>13.2f}%  {new_pct:>19.2f}%")

    print(f"\n  Min raw score : {min_s:.4f}")
    print(f"  Max raw score : {max_s:.4f}")
    print(f"  Score spread  : {max_s - min_s:.4f}")
    print(f"\n  ⭐ Best match : {ranked[0][0]} (raw={ranked[0][1]:.4f})")
    print(f"\n{'='*60}")
    print("  CALIBRATION GUIDE:")
    print(f"  - If best match raw score is ~{ranked[0][1]:.2f} and should show ~85-95%,")
    print(f"    set ABS_MAX = {ranked[0][1] * 1.05:.2f} in similarity.py and main.py")
    print(f"  - Current ABS_MIN={ABS_MIN}, ABS_MAX={ABS_MAX}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_scores.py <path_to_audio_file>")
        print("\nExamples:")
        print('  python diagnose_scores.py "Recording (13).wav"')
        print('  python diagnose_scores.py "dataset/raw/Mishary Al-Fasay/1_Surah_Fatiha.wav"')
        sys.exit(1)

    diagnose(sys.argv[1])
