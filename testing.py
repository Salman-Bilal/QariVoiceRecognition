from matching.similarity import extract_user_embedding, _load_reference_embeddings, cosine_similarity
import numpy as np
from pathlib import Path

# Load all pre-computed Qari reference vectors
refs = _load_reference_embeddings()

# ⚠️ REPLACE THIS WITH AN ACTUAL WAV FILE PATH THAT EXISTS ON YOUR PC!
test_file = "dataset/processed/chunks/Abdul Basit Abdul Samad/someChunk.wav"

user_emb = extract_user_embedding(test_file)

for qari, ref_emb in refs.items():
    score = cosine_similarity(user_emb, ref_emb)
    print(f"{qari}: {score:.4f}")