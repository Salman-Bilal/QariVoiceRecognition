# analysis/test_threshold.py
import sys
from pathlib import Path

# Automatically handle paths so it executes correctly from any directory
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "analysis"))

try:
    from breath import detect_pauses

    # Construct absolute path to the target audio file
    path = BASE_DIR / "dataset" / "processed" / "normalized" / "Yasser Al-Dosari" / "55_Surah_Rehman.wav"
    
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found at: {path}")

    # Run pause detection with the tighter 0.15s gap duration threshold
    pauses = detect_pauses(str(path), min_pause_sec=0.15)
    
    print("\n🔍 Tighter Threshold Diagnostic Result:")
    print("=" * 40)
    print(f"Pauses detected with 0.15s threshold: {len(pauses)}")
    if pauses:
        print(f"Avg duration:                       {sum(pauses)/len(pauses):.4f} seconds")
    print("=" * 40)

except Exception as e:
    print(f"❌ Execution failed: {e}")