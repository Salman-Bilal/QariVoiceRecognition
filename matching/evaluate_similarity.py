# matching/evaluate_similarity.py
import pandas as pd
import numpy as np
import pickle
import os
import json
from pathlib import Path

# --- 📂 ROBUST ABSOLUTE PATH RESOLUTION ---
BASE_DIR = Path(__file__).resolve().parent.parent

REFERENCE_EMBEDDINGS_PATH = BASE_DIR / "matching" / "reference_embeddings.pkl"
MANIFEST_PATH = BASE_DIR / "dataset" / "processed" / "track_a_features.csv"
REPORT_OUTPUT_PATH = BASE_DIR / "evaluation" / "reports" / "similarity_engine_metrics.json"

print("📥 Loading system resources...")
if not REFERENCE_EMBEDDINGS_PATH.exists():
    raise FileNotFoundError(f"❌ Missing reference weights file at {REFERENCE_EMBEDDINGS_PATH}")

with open(REFERENCE_EMBEDDINGS_PATH, "rb") as f:
    reference_embeddings = pickle.load(f)

df = pd.read_csv(MANIFEST_PATH)
test_df = df[df["split"] == "test"]
print(f"📈 Batch evaluating {len(test_df)} unseen test data chunks...")

def cosine_similarity(a, b):
    return float(np.dot(a, b))

top1_correct = 0
top5_correct = 0
total = 0

for _, row in test_df.iterrows():
    # Absolute path check for numpy vectors
    emb_path = BASE_DIR / row["embedding_path"] if not os.path.isabs(row["embedding_path"]) else Path(row["embedding_path"])
    user_embedding = np.load(emb_path)
    true_qari = row["qari_id"]

    scores = {
        qari_id: cosine_similarity(user_embedding, ref_emb)
        for qari_id, ref_emb in reference_embeddings.items()
    }pkl
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top5_qaris = [q for q, _ in ranked[:5]]

    if ranked[0][0] == true_qari:
        top1_correct += 1
    if true_qari in top5_qaris:
        top5_correct += 1
    total += 1

# Calculate ratios
top1_acc = top1_correct / total
top5_acc = top5_correct / total

print("\n==================================================")
print("🏁 Batch Similarity Evaluation Complete")
print("==================================================")
print(f"🔹 Total Test Chunks Evaluated: {total}")
print(f"🏆 Top-1 Matching Accuracy:    {top1_acc * 100:.2f}%")
print(f"🏅 Top-5 Matching Accuracy:    {top5_acc * 100:.2f}%")

# Save reports directory cleanly
REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_OUTPUT_PATH, "w") as f:
    json.dump({
        "total_evaluated": total,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc
    }, f, indent=2)

print(f"\n📦 Metrics report successfully exported to:\n🎯 {REPORT_OUTPUT_PATH}")