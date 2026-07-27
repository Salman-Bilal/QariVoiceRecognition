<<<<<<< HEAD
# preprocessing/noise.py
import noisereduce as nr

def reduce_noise(y, sr, prop_decrease=0.4):
    """
    prop_decrease kept low (0.4 vs default 1.0) — aggressive denoising
    distorts vocal timbre, which both tracks depend on.
    """
=======
# preprocessing/noise.py
import noisereduce as nr

def reduce_noise(y, sr, prop_decrease=0.4):
    """
    prop_decrease kept low (0.4 vs default 1.0) — aggressive denoising
    distorts vocal timbre, which both tracks depend on.
    """
>>>>>>> a090ded35b0a085b6b5c18aa578a35c9d63d14a3
    return nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease)