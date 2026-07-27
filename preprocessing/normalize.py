<<<<<<< HEAD
import librosa
import soundfile as sf
import pyloudnorm as pyln
import numpy as np

TARGET_SR = 16000
TARGET_LUFS = -23.0  

def normalize_audio(input_path, output_path, target_sr=TARGET_SR):
    y, sr = librosa.load(str(input_path), sr=target_sr, mono=True)

    meter = pyln.Meter(target_sr)
    loudness = meter.integrated_loudness(y)

    if np.isfinite(loudness):
        y = pyln.normalize.loudness(y, loudness, TARGET_LUFS)

    y = np.clip(y, -1.0, 1.0)  
    sf.write(str(output_path), y, target_sr, subtype="PCM_16")
=======
# preprocessing/normalize.py
import os
import librosa
import soundfile as sf
import pyloudnorm as pyln
import numpy as np

TARGET_SR = 16000
TARGET_LUFS = -23.0  # broadcast-standard loudness target

def normalize_audio(input_path, output_path, target_sr=TARGET_SR):
    y, sr = librosa.load(input_path, sr=target_sr, mono=True)

    meter = pyln.Meter(target_sr)
    loudness = meter.integrated_loudness(y)

    if np.isfinite(loudness):
        y = pyln.normalize.loudness(y, loudness, TARGET_LUFS)

    y = np.clip(y, -1.0, 1.0)  # avoid clipping artifacts after normalization
    sf.write(output_path, y, target_sr, subtype="PCM_16")
>>>>>>> a090ded35b0a085b6b5c18aa578a35c9d63d14a3
    return output_path