# analysis/test_user_voice.py
import sys
import json
from pathlib import Path

# Resolve project directories dynamically
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "analysis"))

from aggregate import generate_recitation_report

# 🎤 Define input variables
USER_AUDIO_RELATIVE = "dataset/processed/normalized/Recording (6).wav"
TARGET_SURAH = "1_Surah_Fatiha"

def run_voice_test():
    user_path = BASE_DIR / USER_AUDIO_RELATIVE
    
    if not user_path.exists():
        print(f"❌ Cannot find your voice sample at: {user_path}")
        print("💡 Please make sure you copied your .wav file to the correct directory!")
        return

    print("🚀 Initializing Acoustic Evaluation Matrix...")
    print(f"🎙️ Aligning input track against master reference standard...")
    print("-" * 60)

    try:
        # Run the complete timing, pitch contour, and breath metrics suite
        report = generate_recitation_report(str(user_path), TARGET_SURAH)
        
        print("\n" + "="*50)
        print(f"🏆 PERSONAL VOICE EVALUATION REPORT — {TARGET_SURAH}")
        print("="*50)
        print(f"Overall Match Score: {report['overall_score']}/100 — {report['overall_interpretation']}")
        print("-" * 50)
        print(f"⏱️ Timing/Rhythm Score:  {report['timing']['timing_score']}/100")
        print(f"🎵 Melody/Pitch Score:   {report['melody']['melody_score']}/100")
        print(f"🗣️ Breath Phrasing Score: {report['breath']['breath_score']}/100")
        print("="*50)
        
        # Save a log file of your voice metrics
        log_out = BASE_DIR / "evaluation" / "reports" / "user_voice_test_report.json"
        with open(log_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📦 Full report metrics saved to:\n   {log_out}")

    except Exception as e:
        print(f"❌ Evaluation sequence failed: {e}")

if __name__ == "__main__":
    run_voice_test()