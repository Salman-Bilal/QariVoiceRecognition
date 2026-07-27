# analysis/diagnose_breath.py
import sys
import librosa
from pathlib import Path

# Fix visibility paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "analysis"))

from breath import detect_pauses
from librosa.effects import split

path = 'dataset/processed/normalized/Yasser Al-Dosari/55_Surah_Rehman.wav'
abs_path = str((BASE_DIR / path).resolve())

try:
    print("⏳ Loading audio track details...")
    y, sr = librosa.load(abs_path, sr=16000)
    print(f"✅ Duration: {len(y)/sr:.2f} seconds")

    pauses = detect_pauses(abs_path)
    print(f"📊 Pauses detected (default threshold): {len(pauses)}")

    # Check non-silent intervals directly across different decibel levels
    intervals_15 = split(y, top_db=15)
    print(f"🔊 Non-silent intervals at top_db=15: {len(intervals_15)}")

    intervals_25 = split(y, top_db=25)
    print(f"🔉 Non-silent intervals at top_db=25: {len(intervals_25)}")
    
    intervals_35 = split(y, top_db=35)
    print(f"🔈 Non-silent intervals at top_db=35: {len(intervals_35)}")

except Exception as e:
    print(f"❌ Error running diagnostic: {e}")