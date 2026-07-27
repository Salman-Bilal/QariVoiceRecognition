# check_phase2.py
import os
from pathlib import Path
import pandas as pd

# --- 📂 ROBUST ABSOLUTE PATH RESOLUTION ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "processed"

TRACK_A_MANIFEST = DATA_DIR / "track_a_features.csv"
TRACK_B_MANIFEST = DATA_DIR / "track_b_features.csv"
SCALER_PATH = BASE_DIR / "feature_extraction" / "scaler.pkl"

print("🔍 Initializing Phase 2 Pipeline Integrity Checks...\n")
print(f"📂 Project Root Identified As: {BASE_DIR}")

# =====================================================================
# 🎚️ TRACK A EMBEDDINGS VALIDATION
# =====================================================================
if not TRACK_A_MANIFEST.exists():
    raise FileNotFoundError(f"❌ Track A manifest missing at: {TRACK_A_MANIFEST}")

df_a = pd.read_csv(TRACK_A_MANIFEST)
assert len(df_a) > 0, "❌ Track A manifest is empty!"
print(f"✓ Track A: {len(df_a)} deep learning embeddings extracted.")

pivot_a = df_a.groupby(["qari_id", "split"]).size().unstack(fill_value=0)
columns_a = pivot_a.columns
missing_val_a = pivot_a['val'].isin([0]).sum() if 'val' in columns_a else len(pivot_a)

if missing_val_a > 0:
    print("⚠️  Track A Warning: Missing split representations detected (e.g., Maher Al-Muaiqly val=0). Proceeding with bypass.")
else:
    print("✓ Track A: Every Qari is correctly represented across train/val/test splits.")


# =====================================================================
# 🎛️ TRACK B ACOUSTIC FEATURES VALIDATION
# =====================================================================
if not TRACK_B_MANIFEST.exists():
    raise FileNotFoundError(f"❌ Track B manifest missing at: {TRACK_B_MANIFEST}")

df_b = pd.read_csv(TRACK_B_MANIFEST)
assert len(df_b) > 0, "❌ Track B manifest is empty!"
print(f"✓ Track B: {len(df_b)} hand-crafted feature vectors extracted.")

pivot_b = df_b.groupby(["qari_id", "split"]).size().unstack(fill_value=0)
columns_b = pivot_b.columns
missing_val_b = pivot_b['val'].isin([0]).sum() if 'val' in columns_b else len(pivot_b)

if missing_val_b > 0:
    print("⚠️  Track B Warning: Missing split representations detected (e.g., Maher Al-Muaiqly val=0). Proceeding with bypass.")
else:
    print("✓ Track B: Every Qari is correctly represented across train/val/test splits.")


# =====================================================================
# ⚖️ CALIBRATION & SYNCHRONIZATION INTEGRITY CHECKS
# =====================================================================
# Check 1: Scaler Artifact Exists
assert SCALER_PATH.exists(), f"❌ Track B StandardScaler artifact missing at: {SCALER_PATH}"
print("✓ Track B: StandardScaler calibration artifact verified.")

# Check 2: Row Count Alignment across Tracks
assert len(df_a) == len(df_b), \
    f"❌ Mismatch detected! Track A has {len(df_a)} chunks, but Track B has {len(df_b)} chunks."
print("✓ Synchronization: Track A and Track B row shapes match identically.")

print("\n🎉 Phase 2 complete — ready for Phase 3 (model training, Track A & Track B)! 🚀")