# preprocessing/noise.py
import noisereduce as nr

def reduce_noise(y, sr, prop_decrease=0.4):
    """
    prop_decrease kept low (0.4 vs default 1.0) — aggressive denoising
    distorts vocal timbre, which both tracks depend on.
    """
    return nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease)