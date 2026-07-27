# analysis/test_any_surah.py
import sys
import json
from pathlib import Path

# Setup paths for execution safety
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "analysis"))

from aggregate import generate_recitation_report

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("❌ Usage: python analysis/test_any_surah.py <path_to_your_voice> <surah_name>")
        print("💡 Example: python analysis/test_any_surah.py dataset/processed/normalized/my_fatiha.wav 1_Surah_Fatiha")
        sys.exit(1)

    my_voice_path = sys.argv[1]
    surah_name = sys.argv[2]

    print(f"🚀 Initializing Custom Recitation Evaluation Engine...")
    print(f"📁 Evaluating: {my_voice_path} -> Standard: {surah_name}")
    print("-" * 60)

    try:
        # Pass the custom parameters dynamically into the master engine
        report = generate_recitation_report(my_voice_path, surah_name)
        
        print("\n" + "="*50)
        print(f"🎙️ CUSTOM VOICE EVALUATION REPORT: {surah_name}")
        print("="*50)
        print(f"Overall Accuracy Score: {report['overall_score']}/100 — {report['overall_interpretation']}")
        print("-" * 50)
        print(f"⏱️ Rhythm/Timing Score:  {report['timing']['timing_score']}/100")
        print(f"🎵 Melodic Pitch Score:  {report['melody']['melody_score']}/100")
        print(f"🗣️ Breath Phrasing Score: {report['breath']['breath_score']}/100")
        print("="*50)

    except Exception as e:
        print(f"❌ Could not complete verification pass: {e}")
        print("💡 Make sure the corresponding master reference file exists in your Qari dataset directory!")
        