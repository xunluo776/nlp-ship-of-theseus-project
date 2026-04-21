"""
04_bertscore_computation.py
============================
BERTScore Computation Pipeline

Computes BERTScore (precision, recall, F1) between each paraphrased
text and its original (T0) for all paraphrasers and rounds (T1/T2/T3),
using pre-computed RoBERTa-base token embeddings stored as part files.

Architecture:
  - Embedding files are split into 4 parts per dataset (T0, T1, T2, T3)
  - Only 2 parts are loaded at once to stay within RAM limits
  - BERTScore is computed by batched token-level cosine similarity
  - Results for each dataset are cached individually, then merged

Expected input:
  paired_all_t123.pkl
  bert_embeddings_{dataset}_part{1..4}_fp16.pkl  (for each of 7 datasets)

Expected output:
  output/update3/bertscore_{dataset}_t123.pkl   (one per dataset)
  output/update3/bertscore_all_datasets_t123.pkl  (merged, final)

Usage:
  # Process one dataset at a time (change DATASET_NAME):
  python 04_bertscore_computation.py
"""

import gc
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration — change DATASET_NAME to process each dataset in turn
# ---------------------------------------------------------------------------

DATASET_NAME = "cmv"   # one of: cmv, eli5, sci_gen, tldr, wp, xsum, yelp

# Maps each paraphraser to its three text column names (T1/T2/T3)
PARAPHRASER_TIERS = {
    "chatgpt":        ["text_chatgpt_T1",        "text_chatgpt_T2",        "text_chatgpt_T3"],
    "palm":           ["text_palm_T1",            "text_palm_T2",           "text_palm_T3"],
    "dipper_low":     ["text_dipper_low_T1",      "text_dipper_low_T2",     "text_dipper_low_T3"],
    "dipper_mid":     ["text_dipper_T1",          "text_dipper_T2",         "text_dipper_T3"],
    "dipper_high":    ["text_dipper_high_T1",     "text_dipper_high_T2",    "text_dipper_high_T3"],
    "pegasus_slight": ["text_pegasus_slight_T1",  "text_pegasus_slight_T2", "text_pegasus_slight_T3"],
    "pegasus_full":   ["text_pegasus_full_T1",    "text_pegasus_full_T2",   "text_pegasus_full_T3"],
}

TIERS      = ["T1", "T2", "T3"]
# Maps each tier to the embedding part file index (T0=part1, T1=part2, etc.)
TIER_PARTS = {"T0": 1, "T1": 2, "T2": 3, "T3": 4}
BATCH_SIZE = 256

# Adjust paths to match your local setup
LOCAL_DIR  = Path(".")
PARTS_DIR  = Path(".")
OUTPUT_DIR = Path("output/update3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / f"bertscore_{DATASET_NAME}_t123.pkl"


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


# ---------------------------------------------------------------------------
# BERTScore: batched token-level cosine similarity
# ---------------------------------------------------------------------------

def flatten_chunks_torch(chunks) -> torch.Tensor:
    """Stack all chunk arrays into one (n_tokens, 768) float32 tensor."""
    arr = np.vstack(chunks)
    return torch.tensor(arr, dtype=torch.float32, device=device)


def bertscore_batched_torch(t1_chunks, t0_chunks, batch_size: int = BATCH_SIZE):
    """
    Compute BERTScore precision, recall, and F1 between two texts.

    Both inputs are lists of FP16 numpy arrays (one per sliding-window chunk).
    Tokens are compared via cosine similarity after L2 normalization.

    Precision: for each token in the paraphrase, find its best match in T0
    Recall:    for each token in T0, find its best match in the paraphrase
    F1:        harmonic mean of precision and recall
    """
    A = flatten_chunks_torch(t1_chunks)   # paraphrase tokens
    B = flatten_chunks_torch(t0_chunks)   # original tokens

    A_norm = A / (A.norm(dim=1, keepdim=True) + 1e-8)
    B_norm = B / (B.norm(dim=1, keepdim=True) + 1e-8)

    precision_rows = []
    # recall_max tracks the best paraphrase match for each T0 token
    recall_max = torch.full((B_norm.size(0),), -1e9, device=device)

    # Process A in batches to avoid OOM for long texts
    for i in range(0, A_norm.size(0), batch_size):
        A_block   = A_norm[i:i + batch_size]
        sim_block = A_block @ B_norm.T   # (batch, n_B)
        precision_rows.append(sim_block.max(dim=1).values)
        recall_max = torch.maximum(recall_max, sim_block.max(dim=0).values)

    precision = torch.cat(precision_rows).mean().item()
    recall    = recall_max.mean().item()
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Embedding loader — loads exactly 2 parts at a time to limit RAM usage
# ---------------------------------------------------------------------------

def load_two_parts(dataset_name: str, part_t0: int, part_tx: int) -> dict:
    """
    Load T0 embeddings and one tier's embeddings into a text-keyed dict.

    Only 2 part files are in memory at once, keeping peak RAM under ~15GB
    even for the largest datasets.

    Returns
    -------
    dict mapping text string → list of FP16 numpy arrays (one per chunk)
    """
    emb_lookup = {}
    for part in [part_t0, part_tx]:
        path = PARTS_DIR / f"bert_embeddings_{dataset_name}_part{part}_fp16.pkl"
        print(f"  Loading part {part} ({path.stat().st_size / 1e9:.2f} GB)...")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        id_to_text = {v: k for k, v in obj["text_to_id"].items()}
        for tid, emb in obj["embeddings"].items():
            emb_lookup[id_to_text[tid]] = emb
        print(f"  Loaded {len(obj['embeddings'])} embeddings from part {part}")
        del obj
        gc.collect()
    print(f"  Total texts in lookup: {len(emb_lookup)}")
    return emb_lookup


# ---------------------------------------------------------------------------
# Main computation loop — processes one dataset, one tier at a time
# ---------------------------------------------------------------------------

def compute_bertscore_for_dataset(dataset_name: str, output_path: Path):
    """
    Compute BERTScore for all rows × paraphrasers × tiers for one dataset.

    Saves a checkpoint after each tier so progress is not lost on interruption.
    Already-computed rows are skipped on resume.
    """
    paired_t123 = pd.read_pickle(LOCAL_DIR / "paired_all_t123.pkl")
    subset = paired_t123[paired_t123["dataset"] == dataset_name].reset_index(drop=True)
    print(f"\nProcessing {dataset_name}: {len(subset)} rows")

    # Resume from existing checkpoint if available
    if output_path.exists():
        print("Loading existing results...")
        existing_df = pd.read_pickle(output_path)
        rows = existing_df.to_dict("records")
        computed = set(
            (r["dataset"], r["key"], r["source"], r["paraphraser"], r["tier"])
            for r in rows
        )
        print(f"Already computed: {len(rows)} rows")
    else:
        rows = []
        computed = set()
        print("Starting fresh...")

    for tier_idx, tier in enumerate(TIERS):
        print(f"\n{'='*40}\nTier: {tier}")

        # Skip tier if already fully computed
        tier_computed = sum(
            1 for r in rows
            if r["dataset"] == dataset_name and r["tier"] == tier
        )
        expected = len(subset) * len(PARAPHRASER_TIERS)
        if tier_computed >= expected:
            print(f"  Already complete ({tier_computed} rows) — skipping")
            continue

        # Load T0 embeddings + this tier's embeddings (2 parts only)
        emb_lookup = load_two_parts(dataset_name, TIER_PARTS["T0"], TIER_PARTS[tier])

        for _, row in tqdm(subset.iterrows(), total=len(subset)):
            t0_text = row["text_T0"]
            if not isinstance(t0_text, str) or t0_text not in emb_lookup:
                continue
            emb_t0 = emb_lookup[t0_text]

            for paraphraser, tier_cols in PARAPHRASER_TIERS.items():
                tx_col  = tier_cols[tier_idx]
                tx_text = row[tx_col]

                key = (dataset_name, row["key"], row["source"], paraphraser, tier)
                if key in computed:
                    continue

                # Record None if paraphrase text is missing
                if not isinstance(tx_text, str) or tx_text not in emb_lookup:
                    rows.append({
                        "dataset": dataset_name, "key": row["key"],
                        "source": row["source"], "paraphraser": paraphraser,
                        "tier": tier,
                        "bert_precision": None, "bert_recall": None, "bert_f1": None,
                    })
                    computed.add(key)
                    continue

                emb_tx = emb_lookup[tx_text]
                prec, rec, f1 = bertscore_batched_torch(emb_tx, emb_t0)
                rows.append({
                    "dataset": dataset_name, "key": row["key"],
                    "source": row["source"], "paraphraser": paraphraser,
                    "tier": tier,
                    "bert_precision": prec, "bert_recall": rec, "bert_f1": f1,
                })
                computed.add(key)

        # Save checkpoint after each tier
        bert_df = pd.DataFrame(rows)
        bert_df.to_pickle(output_path)
        print(f"  Saved {len(rows)} rows → {output_path}")

        # Free memory before loading next tier's embeddings
        del emb_lookup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nFinished {dataset_name}!")
    print(f"Output shape: {pd.DataFrame(rows).shape}")


# ---------------------------------------------------------------------------
# Merge all 7 per-dataset files into one unified table
# ---------------------------------------------------------------------------

def merge_all_datasets():
    """
    Concatenate the 7 per-dataset BERTScore pickle files into a single
    unified dataframe and save as bertscore_all_datasets_t123.pkl.

    Expected final shape: (~406,182 rows, 8 columns)
    """
    datasets = ["cmv", "eli5", "sci_gen", "tldr", "wp", "xsum", "yelp"]
    dfs = []
    for ds in datasets:
        path = OUTPUT_DIR / f"bertscore_{ds}_t123.pkl"
        if path.exists():
            df = pd.read_pickle(path)
            print(f"  {ds}: {df.shape[0]} rows")
            dfs.append(df)
        else:
            print(f"  {ds}: NOT FOUND ✗")

    bert_all = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal shape: {bert_all.shape}")
    print("\nTier counts:")
    print(bert_all["tier"].value_counts().to_string())
    print("\nDataset counts:")
    print(bert_all["dataset"].value_counts().to_string())
    print("\nParaphraser counts:")
    print(bert_all["paraphraser"].value_counts().to_string())

    output_path = OUTPUT_DIR / "bertscore_all_datasets_t123.pkl"
    bert_all.to_pickle(output_path)
    print(f"\nSaved to {output_path}")
    return bert_all


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Step 1: compute BERTScore for the configured dataset
    compute_bertscore_for_dataset(DATASET_NAME, OUTPUT_PATH)

    # Step 2: sanity check
    bert_df = pd.read_pickle(OUTPUT_PATH)
    print("\nShape:", bert_df.shape)
    print("Columns:", bert_df.columns.tolist())
    print("\nTier counts:")
    print(bert_df["tier"].value_counts().to_string())
    print("\nParaphraser counts:")
    print(bert_df["paraphraser"].value_counts().to_string())

    # Step 3 (run after all 7 datasets are complete): merge into unified table
    # Uncomment when all 7 per-dataset files are ready:
    # merge_all_datasets()
