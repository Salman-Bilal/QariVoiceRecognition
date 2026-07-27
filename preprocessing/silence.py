<<<<<<< HEAD
# preprocessing/silence.py
import librosa
import numpy as np

def trim_silence(y, sr, top_db=30):
    """
    Conservative trim — only removes leading/trailing silence,
    preserves internal pauses/breaths for Phase 5 analysis later.
    """
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
=======
# preprocessing/silence.py
import librosa
import numpy as np

def trim_silence(y, sr, top_db=30):
    """
    Conservative trim — only removes leading/trailing silence,
    preserves internal pauses/breaths for Phase 5 analysis later.
    """
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
>>>>>>> a090ded35b0a085b6b5c18aa578a35c9d63d14a3
    return y_trimmed