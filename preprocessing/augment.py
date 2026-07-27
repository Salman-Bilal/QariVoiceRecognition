<<<<<<< HEAD
import librosa
import numpy as np

def pitch_shift_aug(y, sr, n_steps=1):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def time_stretch_aug(y, rate=1.1):
    return librosa.effects.time_stretch(y, rate=rate)

def add_noise_aug(y, noise_level=0.005):
    noise = np.random.randn(len(y))
=======
# preprocessing/augment.py
import librosa
import numpy as np

def pitch_shift_aug(y, sr, n_steps=1):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def time_stretch_aug(y, rate=1.1):
    return librosa.effects.time_stretch(y, rate=rate)

def add_noise_aug(y, noise_level=0.005):
    noise = np.random.randn(len(y))
>>>>>>> a090ded35b0a085b6b5c18aa578a35c9d63d14a3
    return y + noise_level * noise