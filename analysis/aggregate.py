# analysis/aggregate.py
import sys
import json
from pathlib import Path

# Fix path visibility
sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis.timing import compute_timing_score
from analysis.melody import compute_melody_score
from analysis.breath import compute_breath_score
from analysis.config import get_reference_path, REFERENCE_QARI

def generate_recitation_report(user_audio_path, surah_name, reference_qari=None):
    """
    Generate comprehensive recitation analysis report
    
    Args:
        user_audio_path: Path to user's audio recording
        surah_name: Name of the Surah (e.g., "1_Surah_Fatiha")
        reference_qari: Optional Qari to compare against (defaults to Abdul Basit Abdul Samad)
    
    Returns:
        Dictionary containing timing, melody, breath scores and interpretation
    """
    # Pass the override parameter to get_reference_path
    reference_path = get_reference_path(surah_name, reference_qari)
    display_qari = reference_qari or REFERENCE_QARI
    
    print(f"📥 Analyzing against reference standard: {display_qari} — {surah_name}")    
    print("⏳ Computing timing alignment score...")
    timing_result = compute_timing_score(user_audio_path, reference_path)

    print("🎵 Computing pitch contour melody score...")
    melody_result = compute_melody_score(user_audio_path, reference_path)

    print("🗣️ Computing breath pacing metric score...")
    breath_result = compute_breath_score(user_audio_path, reference_path)

    overall_score = round(
        (timing_result["timing_score"] + melody_result["melody_score"] + breath_result["breath_score"]) / 3,
        2
    )

    def interpret(score):
        if score >= 85: return "Strong match"
        elif score >= 70: return "Good, minor differences"
        elif score >= 50: return "Noticeable differences"
        else: return "Significant differences — review recommended"

    report = {
        "surah": surah_name,
        "reference_qari": display_qari,  # Use the actual Qari being compared against
        "overall_score": overall_score,
        "overall_interpretation": interpret(overall_score),
        "timing": {**timing_result, "interpretation": interpret(timing_result["timing_score"])},
        "melody": {**melody_result, "interpretation": interpret(melody_result["melody_score"])},
        "breath": {**breath_result, "interpretation": interpret(breath_result["breath_score"])},
        "note": "Pronunciation/tajweed accuracy scoring is deferred by design — this report covers acoustic rhythm, melody, and breath consistency metrics only."
    }
    
    return report
   

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analysis/aggregate.py <user_audio_path> <surah_name>")
        sys.exit(1)

    user_p = sys.argv[1]
    surah_n = sys.argv[2]

    report = generate_recitation_report(user_p, surah_n)

    print("\n" + "="*50)
    print(f"🏆 RECITATION ANALYSIS REPORT — {report['surah']}")
    print("="*50)
    print(f"Overall System Score: {report['overall_score']}/100 — {report['overall_interpretation']}")
    print(f"Timing Alignment:      {report['timing']['timing_score']}/100")
    print(f"Melody Alignment:      {report['melody']['melody_score']}/100")
    print(f"Breath Phrasing:       {report['breath']['breath_score']}/100")
    
    out_path = Path(__file__).resolve().parent.parent / "evaluation" / "reports" / "sample_recitation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📦 Metrics summary report saved to: {out_path}")