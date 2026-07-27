# analysis/predict_and_analyze.py
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import validated modules from your pipeline packages
from matching.similarity import get_top5_similar
from analysis.aggregate import generate_recitation_report

def predict_and_analyze(user_audio_path, surah_name):
    print("="*60)
    print("🌟 STEP 1 — Identifying Closest Qari Match (Phase 4 Similarity Engine)")
    print("="*60)

    # 1. Run the high-accuracy voice classifier
    top5 = get_top5_similar(user_audio_path, top_n=5)

    for i, r in enumerate(top5, 1):
        print(f"{i}. {r['qari_id']} — {r['similarity_percent']}%")

    predicted_qari = top5[0]["qari_id"]
    confidence = top5[0]["similarity_percent"]

    print(f"\n🎯 Predicted Master Qari: {predicted_qari} ({confidence}% confidence)\n")

    print("="*60)
    print(f"📊 STEP 2 — Generating Recitation Performance Report against {predicted_qari} (Phase 5)")
    print("="*60)

    # 2. Feed the winning Qari profile straight into the grader pipeline
    report = generate_recitation_report(user_audio_path, surah_name, reference_qari=predicted_qari)

    print(f"\n✨ Overall Recitation Score: {report['overall_score']}/100 — {report['overall_interpretation']}")
    print(f"🔹 Timing Score:  {report['timing']['timing_score']}/100")
    print(f"🔹 Melody Score:  {report['melody']['melody_score']}/100")
    print(f"🔹 Breath Score:  {report['breath']['breath_score']}/100")

    return {
        "predicted_qari": predicted_qari,
        "identity_confidence_percent": confidence,
        "top5_matches": top5,
        "recitation_analysis": report
    }

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("❌ Usage: python analysis/predict_and_analyze.py <audio_path> <surah_name>")
        print('💡 Example: python analysis/predict_and_analyze.py "dataset/processed/normalized/testing 3.wav" "1_Surah_Fatiha"')
        sys.exit(1)

    audio_path = sys.argv[1]
    surah_name = sys.argv[2]

    predict_and_analyze(audio_path, surah_name)