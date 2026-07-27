# models/train_embedding_classifier.py
import pandas as pd
import numpy as np
import pickle
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# --- 📂 ROBUST ABSOLUTE PATH RESOLUTION ---
# Automatically scales back up to your project root folder
BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST_PATH = BASE_DIR / "dataset" / "processed" / "track_a_features.csv"
CHECKPOINT_DIR = BASE_DIR / "models" / "checkpoints" / "track_a"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print(f"🔍 Looking for Track A manifest at: {MANIFEST_PATH}")

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(f"❌ Cannot find manifest file at {MANIFEST_PATH}!")

df = pd.read_csv(MANIFEST_PATH)
print(f"Total Track A embeddings loaded: {len(df)}")
print(df.groupby(["qari_id", "split"]).size().unstack(fill_value=0))

# --- Update your file loading functions to use absolute paths too ---
def load_split(split_name):
    split_df = df[df["split"] == split_name]
    # We resolve the file paths dynamically relative to BASE_DIR if they are saved as relative paths
    X = np.stack([np.load(BASE_DIR / p if not os.path.isabs(p) else p) for p in split_df["embedding_path"]])
    y = split_df["qari_id"].values
    return X, y

X_train, y_train_raw = load_split("train")
X_val, y_val_raw = load_split("val")
X_test, y_test_raw = load_split("test")

# Convert Qari names (strings) into integers (0, 1, 2...) for our mathematical classifiers
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_val = label_encoder.transform(y_val_raw)
y_test = label_encoder.transform(y_test_raw)

with open(CHECKPOINT_DIR / "label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# --- 1. Train SVM Classifier ---
print("\n" + "="*50)
print("Training SVM classifier on embeddings...")
print("="*50)
svm_clf = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
svm_clf.fit(X_train, y_train)

val_preds = svm_clf.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)

with open(CHECKPOINT_DIR / "svm_classifier.pkl", "wb") as f:
    pickle.dump(svm_clf, f)

# --- 2. Train Dense NN Classifier ---
print("\n" + "="*50)
print("Training Dense NN classifier on embeddings...")
print("="*50)

class EmbeddingClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = len(label_encoder.classes_)
input_dim = X_train.shape[1]

model = EmbeddingClassifier(input_dim, num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=32, shuffle=True)
X_val_t, y_val_t = torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(y_val, dtype=torch.long).to(device)

best_val_acc, patience, patience_counter = 0, 10, 0

for epoch in range(100):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        criterion(model(xb), yb).backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_acc_nn = (model(X_val_t).argmax(dim=1) == y_val_t).float().mean().item()

    if val_acc_nn > best_val_acc:
        best_val_acc = val_acc_nn
        patience_counter = 0
        torch.save(model.state_dict(), CHECKPOINT_DIR / "dense_nn_best.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

print(f"SVM Val Acc: {val_acc:.4f} | Dense NN Val Acc: {best_val_acc:.4f}")

# --- 3. Evaluate the Winner on Test ---
os.makedirs("evaluation/reports", exist_ok=True)
if val_acc >= best_val_acc:
    print("🏆 Winner: SVM")
    test_preds = svm_clf.predict(X_test)
    best_model_name = "svm"
else:
    print("🏆 Winner: Dense NN")
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "dense_nn_best.pt"))
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)).argmax(dim=1).cpu().numpy()
    best_model_name = "dense_nn"

test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, test_preds, average="macro")
cm = confusion_matrix(y_test, test_preds)

results = {
    "track": "A", "best_model": best_model_name,
    "val_accuracy": float(val_acc if best_model_name == "svm" else best_val_acc),
    "test_accuracy": float(accuracy_score(y_test, test_preds)),
    "test_macro_precision": float(test_precision), "test_macro_recall": float(test_recall), "test_macro_f1": float(test_f1),
    "confusion_matrix": cm.tolist(), "classes": list(label_encoder.classes_)
}

with open("evaluation/reports/track_a_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("Track A evaluation metrics cached successfully!")