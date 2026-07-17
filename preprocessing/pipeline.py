# preprocessing/pipeline.py
import os
import librosa
import soundfile as sf
from preprocessing.normalize import normalize_audio, TARGET_SR
from preprocessing.silence import trim_silence
from preprocessing.noise import reduce_noise

def preprocess_file(input_path, output_path):
    # Step 1: format normalize (also handles resample + mono + loudness)
    normalize_audio(input_path, output_path)

    # Step 2: reload, denoise, trim
    y, sr = librosa.load(output_path, sr=TARGET_SR, mono=True)
    y = reduce_noise(y, sr)
    y = trim_silence(y, sr)

    sf.write(output_path, y, sr, subtype="PCM_16")
    return output_path