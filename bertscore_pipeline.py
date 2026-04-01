import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DATASET_NAME = "xsum"   # change to: "cmv", "eli5", "yelp", etc.
EMB_PATH = f"bert_embeddings_{DATASET_NAME}_fp16.pkl"

CHUNK_SIZE = 100   # process 100 rows at a time
BATCH_SIZE = 256   # batching inside BERTScore

cache_path = f"output/bertscore_{DATASET_NAME}_cache.pkl"

# ---------------------------------------------------------
# DEVICE SETUP
# ---------------------------------------------------------

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

# ---------------------------------------------------------
# LOAD EMBEDDINGS
# ---------------------------------------------------------

with open(EMB_PATH, "rb") as f:
    obj = pickle.load(f)

embeddings = obj["embeddings"]
text_to_id = obj["text_to_id"]

print(f"Loaded embeddings for {DATASET_NAME}:")
print("  Number of texts:", len(text_to_id))
print("  Number of embedding entries:", len(embeddings))

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def flatten_chunks_torch(chunks):
    arr = np.vstack(chunks)
    return torch.tensor(arr, dtype=torch.float32, device=device)

def bertscore_batched_torch(t1_chunks, t0_chunks, batch_size=BATCH_SIZE):
    """
    Memory-safe BERTScore using batching.
    Returns (precision, recall, f1).
    """
    A = flatten_chunks_torch(t1_chunks)
    B = flatten_chunks_torch(t0_chunks)

    A_norm = A / (A.norm(dim=1, keepdim=True) + 1e-8)
    B_norm = B / (B.norm(dim=1, keepdim=True) + 1e-8)

    precision_rows = []
    recall_max = torch.full((B_norm.size(0),), -1e9, device=device)

    for i in range(0, A_norm.size(0), batch_size):
        A_block = A_norm[i:i+batch_size]
        sim_block = A_block @ B_norm.T

        precision_rows.append(sim_block.max(dim=1).values)
        recall_max = torch.maximum(recall_max, sim_block.max(dim=0).values)

    precision = torch.cat(precision_rows).mean().item()
    recall = recall_max.mean().item()
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1

# ---------------------------------------------------------
# Load dataset subset
# ---------------------------------------------------------

subset = paired_t1[paired_t1["dataset"] == DATASET_NAME].copy()
subset = subset.reset_index(drop=True)

paraphrasers = [
    "text_chatgpt",
    "text_palm",
    "text_dipper_low",
    "text_dipper",
    "text_dipper_high",
    "text_pegasus_slight",
    "text_pegasus_full",
]

# ---------------------------------------------------------
# Load or initialize cache
# ---------------------------------------------------------

if os.path.exists(cache_path):
    print(f"Resuming from existing cache: {cache_path}")
    with open(cache_path, "rb") as f:
        bert_results = pickle.load(f)
else:
    print("Starting fresh cache...")
    bert_results = {p: [] for p in paraphrasers}

# ---------------------------------------------------------
# Determine starting chunk
# ---------------------------------------------------------

completed = len(next(iter(bert_results.values())))
start_chunk = completed // CHUNK_SIZE

print(f"Already completed {completed} rows. Resuming at chunk {start_chunk}.")

# ---------------------------------------------------------
# Chunked processing loop
# ---------------------------------------------------------

num_rows = len(subset)
num_chunks = (num_rows + CHUNK_SIZE - 1) // CHUNK_SIZE

for chunk_idx in range(start_chunk, num_chunks):
    start = chunk_idx * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, num_rows)
    chunk = subset.iloc[start:end]

    print(f"\nProcessing chunk {chunk_idx+1}/{num_chunks} "
          f"({start} → {end})")

    for _, row in tqdm(chunk.iterrows(), total=len(chunk)):
        t0 = row["text_T0"]
        if t0 not in text_to_id:
            continue

        emb_t0 = embeddings[text_to_id[t0]]

        for p in paraphrasers:
            t1 = row[p]
            if pd.isna(t1) or t1 not in text_to_id:
                bert_results[p].append((None, None, None))
                continue

            emb_t1 = embeddings[text_to_id[t1]]

            prec, rec, f1 = bertscore_batched_torch(emb_t1, emb_t0)
            bert_results[p].append((prec, rec, f1))

    os.makedirs("output", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(bert_results, f)

    print(f"Saved progress to {cache_path}")

print("\nAll chunks completed!")
