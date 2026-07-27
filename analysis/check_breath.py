# analysis/check_breath.py
import sys
from pathlib import Path

# Automatically resolve paths to point to the correct folder
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "analysis"))

from aggregate import generate_recitation_report

user_audio = "dataset/processed/normalized/Yasser Al-Dosari/55_Surah_Rehman.wav"
surah_name = "55_Surah_Rehman"

try:
    report = generate_recitation_report(user_audio, surah_name)
    
    print("\n📊 Updated Calibration Check:")
    print("=" * 35)
    print(f"Breath Pacing Score: {report['breath']['breath_score']}/100")
    print(f"User Pause Count:    {report['breath']['user_pause_count']}")
    print(f"Ref Pause Count:     {report['breath']['reference_pause_count']}")
    print(f"Avg User Duration:   {report['breath']['user_avg_pause_duration']}s")
    print(f"Avg Ref Duration:    {report['breath']['reference_avg_pause_duration']}s")
    print("=" * 35)

except Exception as e:
    print(f"❌ Error running evaluation: {e}")