# models/train_baseline_nn.py
import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# --- 📂 ROBUST ABSOLUTE PATH RESOLUTION ---
# __file__ is /home/.../QariVoiceRecognition/models/train_baseline_nn.py
# .parent is /home/.../QariVoiceRecognition/models
# .parent.parent scales back up to your project root folder
BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST_PATH = BASE_DIR / "dataset" / "processed" / "track_b_features.csv"
SCALER_PATH = BASE_DIR / "feature_extraction" / "scaler.pkl"
CHECKPOINT_DIR = BASE_DIR / "models" / "checkpoints" / "track_b"
REPORT_DIR = BASE_DIR / "evaluation" / "reports"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print(f"🔍 Looking for Track B manifest at: {MANIFEST_PATH}")
if not MANIFEST_PATH.exists():
    raise FileNotFoundError(f"❌ Cannot find Track B manifest at {MANIFEST_PATH}!")

if not SCALER_PATH.exists():
    raise FileNotFoundError(f"❌ Cannot find scaler calibration artifact at {SCALER_PATH}! Run fit_scaler.py first.")

df = pd.read_csv(MANIFEST_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

def load_split(split_name):
    split_df = df[df["split"] == split_name]
    # Dynamically force file paths relative to BASE_DIR if they are relative strings
    X = np.stack([np.load(BASE_DIR / p if not os.path.isabs(p) else p) for p in split_df["feature_path"]])
    y = split_df["qari_id"].values
    return X, y

print("📥 Loading and validating Track B dataset splits...")
X_train, y_train_raw = load_split("train")
X_val, y_val_raw = load_split("val")
X_test, y_test_raw = load_split("test")

# Normalize features utilizing the scaler we generated in Step 1
print("⚖️ Applying StandardScaler normalization across vectors...")
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_val = label_encoder.transform(y_val_raw)
y_test = label_encoder.transform(y_test_raw)

with open(CHECKPOINT_DIR / "label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# --- 🧠 Neural Network Structural Setup ---
class BaselineNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): 
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Initializing Baseline Feedforward NN on device: {device.upper()}")
model = BaselineNN(X_train.shape[1], len(label_encoder.classes_)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=32, shuffle=True)
X_val_t, y_val_t = torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(y_val, dtype=torch.long).to(device)

best_val_acc, patience, patience_counter = 0, 10, 0

print("\n" + "="*50)
print("🏋️‍♂️ Starting Track B Neural Network Model Training Loop")
print("="*50)

for epoch in range(100):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        criterion(model(xb), yb).backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_acc = (model(X_val_t).argmax(dim=1) == y_val_t).float().mean().item()
    
    scheduler.step(val_acc)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), CHECKPOINT_DIR / "baseline_nn_best.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience: 
            print(f"🛑 Early stopping triggered at epoch {epoch+1} (No improvement on Val Accuracy).")
            break

# Final Evaluation on the test dataset
print("\n🏁 Training complete. Loading best checkpoint for Evaluation...")
model.load_state_dict(torch.load(CHECKPOINT_DIR / "baseline_nn_best.pt"))
model.eval()
with torch.no_grad():
    test_preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)).argmax(dim=1).cpu().numpy()

test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, test_preds, average="macro")
cm = confusion_matrix(y_test, test_preds)

results = {
    "track": "B", "best_model": "feedforward_nn", "val_accuracy": float(best_val_acc),
    "test_accuracy": float(accuracy_score(y_test, test_preds)),
    "test_macro_precision": float(test_precision), "test_macro_recall": float(test_recall), "test_macro_f1": float(test_f1),
    "confusion_matrix": cm.tolist(), "classes": list(label_encoder.classes_)
}

OUTPUT_REPORT_PATH = REPORT_DIR / "track_b_metrics.json"
with open(OUTPUT_REPORT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print("🎉 Track B performance metrics compiled and saved successfully!")
print(f"📦 Performance Summary Location: {OUTPUT_REPORT_PATH}")