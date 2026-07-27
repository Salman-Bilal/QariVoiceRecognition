# matching/similarity.py
import numpy as np
import pickle
import torch
import torchaudio

# --- 🩹 TORCHAUDIO MONKEYPATCH FOR SPEECHBRAIN COMPATIBILITY ---
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

import json
import os
import sys
from pathlib import Path
from speechbrain.inference.speaker import EncoderClassifier

# --- 📂 ROBUST ABSOLUTE PATH RESOLUTION ---
BASE_DIR = Path(__file__).resolve().parent.parent

REFERENCE_EMBEDDINGS_PATH = BASE_DIR / "matching" / "reference_embeddings.pkl"
MODEL_CACHE_DIR = BASE_DIR / "model_cache" / "spkrec-ecapa-voxceleb"

_classifier = None
_reference_embeddings = None

def _load_classifier():
    global _classifier
    if _classifier is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🤖 Loading ECAPA-TDNN feature extractor on device: {device.upper()}...")
        _classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(MODEL_CACHE_DIR),
            run_opts={"device": device}
        )
    return _classifier

def _load_reference_embeddings():
    global _reference_embeddings
    if _reference_embeddings is None:
        if not REFERENCE_EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(f"❌ Missing reference file at {REFERENCE_EMBEDDINGS_PATH}! Run build_reference_embeddings.py first.")
        with open(REFERENCE_EMBEDDINGS_PATH, "rb") as f:
            _reference_embeddings = pickle.load(f)
    return _reference_embeddings

def extract_user_embedding(audio_path):
    classifier = _load_classifier()
    
    # Resolve relative paths securely if provided
    abs_audio_path = BASE_DIR / audio_path if not os.path.isabs(audio_path) else Path(audio_path)
    
    if not abs_audio_path.exists():
        raise FileNotFoundError(f"❌ Cannot find audio file at: {abs_audio_path}")
        
    signal, fs = torchaudio.load(str(abs_audio_path))

    # Convert stereo to mono
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    if fs != 16000:
        signal = torchaudio.functional.resample(signal, fs, 16000)
        
    # 💡 FIX 1: Peak volume normalization for consistency
    if signal.abs().max() > 0:
        signal = signal / signal.abs().max()
        
    with torch.no_grad():
        emb = classifier.encode_batch(signal).squeeze().cpu().numpy()
        
    # 💡 FIX 2: Safe L2 normalization
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
        
    return emb

def cosine_similarity(a, b):
    return float(np.dot(a, b))  # L2-normalized vectors

def get_top5_similar(audio_path, top_n=5):
    reference_embeddings = _load_reference_embeddings()
    user_embedding = extract_user_embedding(audio_path)

    # Compute raw cosine scores against ALL Qaris
    all_scores = {
        qari_id: cosine_similarity(user_embedding, ref_emb)
        for qari_id, ref_emb in reference_embeddings.items()
    }

    ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    top5 = ranked[:top_n]

    # --- Dual-Mode Similarity Scoring ---
    #
    # Two very different contexts require two different scoring strategies:
    #
    # MODE A — Qari's own recording (top raw score >= 0.65):
    #   The audio IS from a known Qari. Rank #1 should show 100%.
    #   Others are shown relative to the top score.
    #   Formula: pct = ((score/top - BIAS) / (1.0 - BIAS)) * 100
    #   BIAS = 0.60 → scores below 60% of the top → 0%
    #
    # MODE B — User voice (top raw score < 0.65):
    #   The audio is from an unknown user. The top match should NOT
    #   be forced to 100% — that would be dishonest. Instead we use
    #   absolute thresholds derived from real user data:
    #     ABS_MIN = 0.15  → 0%   (no resemblance)
    #     ABS_MAX = 0.55  → 100% (ceiling — practically unreachable for a user)
    #   This produces honest scores like 49%, 36%, 13% that reflect
    #   actual degree of resemblance.
    #
    # Detection threshold (0.65) chosen because:
    #   - Qari's own audio consistently scores 0.70–0.85 against their centroid
    #   - Regular user audio never exceeds ~0.50 against any Qari centroid

    top_score = top5[0][1] if top5 else 1.0

    QARI_MODE_THRESHOLD = 0.65  # above this → treat as Qari's own recording
    BIAS = 0.60                 # for relative mode: ratios below this → 0%
    ABS_MIN = 0.15              # for user mode: floor
    ABS_MAX = 0.55              # for user mode: ceiling

    is_qari_recording = top_score >= QARI_MODE_THRESHOLD

    results = []
    for qari_id, score in top5:
        if is_qari_recording:
            # MODE A: Relative scaling — rank #1 = 100%
            raw_ratio = score / top_score if top_score > 0 else 0.0
            pct = ((raw_ratio - BIAS) / (1.0 - BIAS)) * 100.0
            pct = round(max(0.0, min(100.0, pct)), 2)
        else:
            # MODE B: Absolute scaling — honest user similarity
            pct = ((score - ABS_MIN) / (ABS_MAX - ABS_MIN)) * 100.0
            pct = round(max(0.0, min(100.0, pct)), 2)

        results.append({
            "qari_id": qari_id,
            "similarity_score": round(score, 4),
            "similarity_percent": pct
        })

    return results

def diagnose_raw_scores(audio_path):
    """
    Diagnostic tool: prints ALL raw cosine scores against every Qari
    so you can calibrate ABS_MIN / ABS_MAX correctly for your use case.
    Run this with your own recitation audio to see the real score range.

    Usage:
        python matching/similarity.py --diagnose <path_to_audio_file>
    """
    reference_embeddings = _load_reference_embeddings()
    user_embedding = extract_user_embedding(audio_path)

    all_scores = {
        qari_id: cosine_similarity(user_embedding, ref_emb)
        for qari_id, ref_emb in reference_embeddings.items()
    }
    ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n📊 RAW COSINE SCORES for: {audio_path}")
    print(f"{'Rank':<5} {'Qari':<30} {'Raw Score':>12} {'Mapped %':>10}")
    print("-" * 60)

    QARI_MODE_THRESHOLD = 0.65
    BIAS = 0.60
    ABS_MIN = 0.15
    ABS_MAX = 0.55
    top_score = ranked[0][1] if ranked else 1.0
    is_qari_recording = top_score >= QARI_MODE_THRESHOLD

    mode_label = "MODE A: Qari Recording (Relative)" if is_qari_recording else "MODE B: User Voice (Absolute)"
    print(f"  Detected: {mode_label} (top score={top_score:.4f})\n")

    for i, (qari_id, score) in enumerate(ranked, 1):
        if is_qari_recording:
            raw_ratio = score / top_score if top_score > 0 else 0.0
            pct = round(max(0.0, min(100.0, ((raw_ratio - BIAS) / (1.0 - BIAS)) * 100.0)), 2)
        else:
            pct = round(max(0.0, min(100.0, ((score - ABS_MIN) / (ABS_MAX - ABS_MIN)) * 100.0)), 2)
        print(f"{i:<5} {qari_id:<30} {score:>12.4f} {pct:>9.2f}%")
    print("-" * 60)
    print(f"  Min raw score: {min(s for _, s in ranked):.4f}")
    print(f"  Max raw score: {max(s for _, s in ranked):.4f}")
    print(f"\n💡 Use these values to fine-tune ABS_MIN / ABS_MAX in get_top5_similar()")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python matching/similarity.py <path_to_audio_file>")
        print("  python matching/similarity.py --diagnose <path_to_audio_file>")
        sys.exit(1)

    if sys.argv[1] == "--diagnose" and len(sys.argv) >= 3:
        input_audio = sys.argv[2]
        try:
            diagnose_raw_scores(input_audio)
        except Exception as e:
            print(f"\n❌ Error: {e}")
    else:
        input_audio = sys.argv[1]
        try:
            results = get_top5_similar(input_audio)
            print(f"\n🎯 Top-5 Similar Qaris for: {input_audio}\n")
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['qari_id']:<25} — {r['similarity_percent']:.2f}% (raw score: {r['similarity_score']:.4f})")
        except Exception as e:
            print(f"\n❌ Error processing audio: {e}")