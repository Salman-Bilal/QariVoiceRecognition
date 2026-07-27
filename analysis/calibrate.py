# analysis/calibrate.py
import os
import sys
from pathlib import Path

# Fix module visibility so it can find aggregate.py cleanly
sys.path.append(str(Path(__file__).resolve().parent))
from aggregate import generate_recitation_report

SURAH = "55_Surah_Rehman"

# --- 📂 ROBUST ABSOLUTE PATH RESOLUTION ---
BASE_DIR = Path(__file__).resolve().parent.parent
NORMALIZED_DIR = BASE_DIR / "dataset" / "processed" / "normalized"

if not NORMALIZED_DIR.exists():
    raise FileNotFoundError(f"❌ Cannot find normalized directory at: {NORMALIZED_DIR}")

results = []
for qari_dir in sorted(NORMALIZED_DIR.iterdir()):
    if not qari_dir.is_dir():
        continue
    audio_path = qari_dir / f"{SURAH}.wav"
    if not audio_path.exists():
        continue
    try:
        report = generate_recitation_report(str(audio_path), SURAH)
        results.append((qari_dir.name, report["overall_score"]))
    except Exception as e:
        print(f"❌ Failed for {qari_dir.name}: {e}")

results.sort(key=lambda x: x[1], reverse=True)
print(f"\n📈 Score distribution for {SURAH} (Reference: Abdul Basit Abdul Samad):\n")
print(f"{'Qari Name':<30} {'Overall Acoustic Match Score'}")
print("-" * 65)
for qari, score in results:
    print(f"{qari:<30} {score}/100")