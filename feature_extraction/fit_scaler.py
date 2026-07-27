# feature_extraction/fit_scaler.py
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- 📂 ROBUST ABSOLUTE PATH RESOLUTION ---
# __file__ is /home/.../QariVoiceRecognition/feature_extraction/fit_scaler.py
# .parent is /home/.../QariVoiceRecognition/feature_extraction
# .parent.parent scales back to project root: /home/.../QariVoiceRecognition
BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST_PATH = BASE_DIR / "dataset" / "processed" / "track_b_features.csv"
SCALER_SAVE_PATH = BASE_DIR / "feature_extraction" / "scaler.pkl"

print(f"🔍 Looking for features file at: {MANIFEST_PATH}")

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(f"❌ Cannot find manifest file at {MANIFEST_PATH}. Did you run mfcc_pooled.py first?")

# Load manifest cleanly using absolute paths
df = pd.read_csv(MANIFEST_PATH)
train_df = df[df["split"] == "train"]

print(f"📥 Loading numpy arrays for {len(train_df)} training vectors...")
X_train = np.stack([np.load(p) for p in train_df["feature_path"]])

# Initialize and fit the scaler
print("⚖️ Calibrating StandardScaler distribution variables...")
scaler = StandardScaler()
scaler.fit(X_train)

# Ensure save folder exists and serialize scaler artifact
SCALER_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(SCALER_SAVE_PATH, "wb") as f:
    pickle.dump(scaler, f)

print(f"🎉 Success! Scaler fit on {X_train.shape[0]} training vectors.")
print(f"📦 Saved calibration artifact to: {SCALER_SAVE_PATH}")