# analysis/config.py
from pathlib import Path

# analysis/config.py
REFERENCE_QARI = "Abdul Basit Abdul Samad"
REFERENCE_AUDIO_DIR = "dataset/processed/normalized"

def get_reference_path(surah_name, reference_qari=None):
    """Retrieves the reference file path, allowing dynamic overrides."""
    qari = reference_qari or REFERENCE_QARI
    return f"{REFERENCE_AUDIO_DIR}/{qari}/{surah_name}.wav"