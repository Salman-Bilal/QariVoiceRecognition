"""
matching/build_style_profiles.py

One-time script to precompute and save style profiles for all 12 Qaris.

Run this ONCE before starting the API server:
    python matching/build_style_profiles.py

What it does:
  - Scans dataset/processed/normalized/{Qari_Name}/*.wav
  - Extracts pitch, rhythm, and breath style features from every Surah
  - Averages features across all Surahs for each Qari
  - Computes inter-Qari mean/std for every feature (used for Z-score normalisation)
  - Saves profiles + statistics bundle to matching/style_profiles.pkl

After running this, the API will use these profiles for all
/api/identify-qari and /api/compare-all-qaris requests.

Re-run any time you add new Qaris or Surahs to the dataset.
"""

import sys
import time
from pathlib import Path

# Ensure project root is importable
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from matching.style_profiles import (
    build_all_style_profiles,
    compute_inter_qari_stats,
    save_style_profiles,
    NORMALIZED_DIR,
    STYLE_PROFILES_PATH,
)


def main():
    print("=" * 60)
    print("  Qari Recitation Style Profile Builder")
    print("=" * 60)
    print(f"\nDataset directory : {NORMALIZED_DIR}")
    print(f"Output path       : {STYLE_PROFILES_PATH}")
    print()

    if not NORMALIZED_DIR.exists():
        print(f"ERROR: Normalized audio directory does not exist:")
        print(f"   {NORMALIZED_DIR}")
        print("\nRun the preprocessing pipeline first:")
        print("   python run_pipeline.py")
        sys.exit(1)

    qari_dirs = [d for d in NORMALIZED_DIR.iterdir() if d.is_dir()]
    if not qari_dirs:
        print(f"ERROR: No Qari subdirectories found in {NORMALIZED_DIR}")
        sys.exit(1)

    print(f"Found {len(qari_dirs)} Qari directories:")
    for d in sorted(qari_dirs):
        wav_count = len(list(d.glob("*.wav")))
        print(f"   - {d.name} ({wav_count} Surah(s))")

    print("\nStarting feature extraction...")
    print("   This may take a few minutes depending on your hardware.\n")

    t_start = time.time()

    # ── Step 1: Build per-Qari style profiles ────────────────────────────────
    try:
        profiles = build_all_style_profiles()
    except Exception as e:
        print(f"\nFailed to build profiles: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if not profiles:
        print("\nNo profiles were built. Check audio files and try again.")
        sys.exit(1)

    # ── Step 2: Compute inter-Qari feature statistics ─────────────────────────
    print("\nComputing inter-Qari feature statistics (mean/std per feature)...")
    try:
        stats = compute_inter_qari_stats(profiles)
    except Exception as e:
        print(f"\nFailed to compute inter-Qari stats: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print a quick summary of which features have the most inter-Qari spread
    print("\n  Feature spread across Qaris (std = how distinctive that feature is):")
    for group in ("pitch", "rhythm", "breath"):
        print(f"\n  [{group.upper()}]")
        for feat, s in stats[group].items():
            bar = "█" * min(int(s["std"] * 20), 30)
            print(f"    {feat:<25}  mean={s['mean']:+.4f}  std={s['std']:.4f}  {bar}")

    # ── Step 3: Save bundled profiles + stats ────────────────────────────────
    save_style_profiles(profiles, stats, STYLE_PROFILES_PATH)

    elapsed = time.time() - t_start
    mins, secs = divmod(int(elapsed), 60)

    print("\n" + "=" * 60)
    print("  Style profiles built successfully!")
    print("=" * 60)
    print(f"\n  Qaris profiled  : {len(profiles)}")
    print(f"  Time taken      : {mins}m {secs}s")
    print(f"  Profiles saved  : {STYLE_PROFILES_PATH}")
    print()

    # Print a summary of each Qari's profile
    print("  Summary:")
    print(f"  {'Qari':<30} {'Surahs':>6}  {'Tempo(bpm)':>10}  {'Voiced%':>8}  {'Pause rate':>10}")
    print("  " + "-" * 70)
    for name, p in sorted(profiles.items()):
        print(
            f"  {name:<30} {p['num_surahs']:>6}  "
            f"{p['rhythm']['tempo_bpm']:>10.1f}  "
            f"{p['pitch']['voiced_ratio']*100:>7.1f}%  "
            f"{p['breath']['pause_rate']:>10.3f}"
        )

    print()
    print("  You can now start the API server:")
    print("    python api/main.py")
    print()


if __name__ == "__main__":
    main()
