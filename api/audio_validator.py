"""
audio_validator.py — Audio file validation and quality checks
Validates uploaded audio before any processing begins
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import librosa

logger = logging.getLogger(__name__)

# ── Validation thresholds ────────────────────────────────────────────
MIN_DURATION_SEC = 1.0      # Minimum 1 second
MAX_DURATION_SEC = 600.0    # Maximum 10 minutes
MIN_SAMPLE_RATE  = 8000     # At least 8kHz
MAX_FILE_SIZE_MB = 50       # 50 MB max
MIN_RMS_ENERGY   = 0.0005   # Very quiet threshold — below this = mostly silence
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac'}


@dataclass
class ValidationResult:
    """Container for validation results"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    audio_info: dict = field(default_factory=dict)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)


def validate_audio_file(file_path: str) -> ValidationResult:
    """
    Perform comprehensive validation of an audio file.

    Checks:
    1. File exists and is readable
    2. File extension is supported
    3. File size is within limits
    4. Audio can be decoded
    5. Duration is within acceptable range
    6. Audio is not silence (has actual content)
    7. Sample rate is acceptable

    Returns:
        ValidationResult with detailed status, errors, warnings, and audio info
    """
    result = ValidationResult(is_valid=True)
    path = Path(file_path)

    # ── 1. File existence ─────────────────────────────────────────────
    if not path.exists():
        result.add_error(f"File not found: {file_path}")
        return result

    if not path.is_file():
        result.add_error(f"Path is not a file: {file_path}")
        return result

    # ── 2. File extension ─────────────────────────────────────────────
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        result.add_error(
            f"Unsupported file format '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
        return result

    # ── 3. File size ──────────────────────────────────────────────────
    file_size_mb = path.stat().st_size / (1024 * 1024)
    result.audio_info['file_size_mb'] = round(file_size_mb, 2)

    if file_size_mb > MAX_FILE_SIZE_MB:
        result.add_error(
            f"File too large ({file_size_mb:.1f} MB). "
            f"Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
        )
        return result

    # ── 4. Audio loading / decoding ───────────────────────────────────
    try:
        y, sr = librosa.load(str(path), sr=None, mono=True)
    except Exception as e:
        result.add_error(f"Could not decode audio file: {e}")
        return result

    # ── 5. Sample rate ────────────────────────────────────────────────
    result.audio_info['sample_rate'] = sr
    if sr < MIN_SAMPLE_RATE:
        result.add_error(
            f"Sample rate too low ({sr} Hz). "
            f"Minimum required: {MIN_SAMPLE_RATE} Hz."
        )

    if sr < 16000:
        result.add_warning(
            f"Sample rate is {sr} Hz. 16000 Hz or higher recommended for best accuracy."
        )

    # ── 6. Duration ───────────────────────────────────────────────────
    duration = len(y) / sr
    result.audio_info['duration_sec'] = round(duration, 2)

    if duration < MIN_DURATION_SEC:
        result.add_error(
            f"Audio too short ({duration:.1f}s). "
            f"Minimum required: {MIN_DURATION_SEC}s."
        )

    if duration > MAX_DURATION_SEC:
        result.add_error(
            f"Audio too long ({duration:.0f}s / {duration/60:.1f} min). "
            f"Maximum allowed: {MAX_DURATION_SEC/60:.0f} minutes."
        )

    if duration < 3.0:
        result.add_warning(
            f"Audio is very short ({duration:.1f}s). "
            "Longer recordings (10+ seconds) give much better identification accuracy."
        )

    # ── 7. Silence / energy check ─────────────────────────────────────
    rms = float(np.sqrt(np.mean(y ** 2)))
    result.audio_info['rms_energy'] = round(rms, 6)

    if rms < MIN_RMS_ENERGY:
        result.add_error(
            "Audio appears to be silent or nearly silent. "
            "Please ensure your microphone is working and you are close enough to it."
        )
    elif rms < MIN_RMS_ENERGY * 10:
        result.add_warning(
            "Audio volume is very low. This may reduce analysis accuracy. "
            "Try recording closer to the microphone."
        )

    # ── 8. Clipping check ─────────────────────────────────────────────
    clipped_ratio = float(np.mean(np.abs(y) > 0.99))
    result.audio_info['clipped_ratio'] = round(clipped_ratio, 4)

    if clipped_ratio > 0.05:
        result.add_warning(
            f"Audio appears to be clipping ({clipped_ratio*100:.1f}% of samples). "
            "Recording volume may be too high."
        )

    # ── 9. Log summary ────────────────────────────────────────────────
    if result.is_valid:
        logger.info(
            f"Audio validated: {path.name} | "
            f"{duration:.1f}s | {sr}Hz | {file_size_mb:.2f}MB | "
            f"RMS={rms:.4f}"
        )
        if result.warnings:
            for w in result.warnings:
                logger.warning(f"Audio warning for {path.name}: {w}")
    else:
        for e in result.errors:
            logger.error(f"Audio validation failed for {path.name}: {e}")

    return result
