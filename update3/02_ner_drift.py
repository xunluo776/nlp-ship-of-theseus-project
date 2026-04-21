"""
02_ner_drift.py
===============
NER Drift Analysis — Computation Pipeline

Extracts named entity sets from all text columns using spaCy,
then computes absolute NER drift metrics (vs T0) and incremental
NER drift metrics (T1→T2, T2→T3) for all paraphrasers and rounds.

All results are cached to disk so this heavy computation only
runs once. Expected runtime: ~5 hours on a modern laptop.

Expected input:  paired_all_t123.pkl
Expected output: paired_ner_t123.pkl
                 ner_metrics_t123_absolute.pkl
                 ner_metrics_t123_incremental.pkl
"""

import numpy as np
import pandas as pd
import spacy
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NER_SETS_PICKLE        = Path("paired_ner_t123.pkl")
NER_METRICS_ABS_PICKLE = Path("ner_metrics_t123_absolute.pkl")
NER_METRICS_INC_PICKLE = Path("ner_metrics_t123_incremental.pkl")

# Maps each paraphraser name to its three entity-set column names (T1/T2/T3)
PARAPHRASER_TIERS = {
    "chatgpt":        ["ents_chatgpt_T1",        "ents_chatgpt_T2",        "ents_chatgpt_T3"],
    "palm":           ["ents_palm_T1",            "ents_palm_T2",           "ents_palm_T3"],
    "dipper_low":     ["ents_dipper_low_T1",      "ents_dipper_low_T2",     "ents_dipper_low_T3"],
    "dipper_mid":     ["ents_dipper_T1",          "ents_dipper_T2",         "ents_dipper_T3"],
    "dipper_high":    ["ents_dipper_high_T1",     "ents_dipper_high_T2",    "ents_dipper_high_T3"],
    "pegasus_slight": ["ents_pegasus_slight_T1",  "ents_pegasus_slight_T2", "ents_pegasus_slight_T3"],
    "pegasus_full":   ["ents_pegasus_full_T1",    "ents_pegasus_full_T2",   "ents_pegasus_full_T3"],
}

TIERS = ["T1", "T2", "T3"]


# ---------------------------------------------------------------------------
# Helper: extract NER set from a single text
# ---------------------------------------------------------------------------

def get_ner_set(text: str, nlp) -> set:
    """
    Extract a lowercase deduplicated set of named entities from text.

    Returns an empty set for non-string or empty inputs, so downstream
    comparisons are safe without further null checks.
    """
    if not isinstance(text, str) or text.strip() == "":
        return set()
    doc = nlp(text)
    return {ent.text.lower().strip() for ent in doc.ents if ent.text.strip() != ""}


# ---------------------------------------------------------------------------
# Step 1: Extract NER sets for all text columns
# ---------------------------------------------------------------------------

def load_or_compute_ner_sets(paired: pd.DataFrame) -> pd.DataFrame:
    """
    Add one ents_* column per text column to the paired dataframe.

    Each ents_* column contains a set of lowercase named entity strings
    extracted by spaCy's en_core_web_sm model.

    Cached as paired_ner_t123.pkl (shape: 19343 × 47).
    """
    if NER_SETS_PICKLE.exists():
        print("Loading cached paired_ner_t123.pkl ...")
        return pd.read_pickle(NER_SETS_PICKLE)

    print("Computing NER sets for T123 dataset...")
    nlp = spacy.load("en_core_web_sm")
    df = paired.copy()

    # T0 (original text)
    df["ents_T0"]                = df["text_T0"].apply(lambda x: get_ner_set(x, nlp))

    # ChatGPT T1/T2/T3
    df["ents_chatgpt_T1"]        = df["text_chatgpt_T1"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_chatgpt_T2"]        = df["text_chatgpt_T2"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_chatgpt_T3"]        = df["text_chatgpt_T3"].apply(lambda x: get_ner_set(x, nlp))

    # PaLM T1/T2/T3
    df["ents_palm_T1"]           = df["text_palm_T1"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_palm_T2"]           = df["text_palm_T2"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_palm_T3"]           = df["text_palm_T3"].apply(lambda x: get_ner_set(x, nlp))

    # Dipper (low) T1/T2/T3
    df["ents_dipper_low_T1"]     = df["text_dipper_low_T1"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_dipper_low_T2"]     = df["text_dipper_low_T2"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_dipper_low_T3"]     = df["text_dipper_low_T3"].apply(lambda x: get_ner_set(x, nlp))

    # Dipper (mid) T1/T2/T3
    df["ents_dipper_T1"]         = df["text_dipper_T1"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_dipper_T2"]         = df["text_dipper_T2"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_dipper_T3"]         = df["text_dipper_T3"].apply(lambda x: get_ner_set(x, nlp))

    # Dipper (high) T1/T2/T3
    df["ents_dipper_high_T1"]    = df["text_dipper_high_T1"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_dipper_high_T2"]    = df["text_dipper_high_T2"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_dipper_high_T3"]    = df["text_dipper_high_T3"].apply(lambda x: get_ner_set(x, nlp))

    # Pegasus (slight) T1/T2/T3
    df["ents_pegasus_slight_T1"] = df["text_pegasus_slight_T1"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_pegasus_slight_T2"] = df["text_pegasus_slight_T2"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_pegasus_slight_T3"] = df["text_pegasus_slight_T3"].apply(lambda x: get_ner_set(x, nlp))

    # Pegasus (full) T1/T2/T3
    df["ents_pegasus_full_T1"]   = df["text_pegasus_full_T1"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_pegasus_full_T2"]   = df["text_pegasus_full_T2"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_pegasus_full_T3"]   = df["text_pegasus_full_T3"].apply(lambda x: get_ner_set(x, nlp))

    df.to_pickle(NER_SETS_PICKLE)
    print("Saved paired_ner_t123.pkl")
    return df


# ---------------------------------------------------------------------------
# Helper: compute Jaccard, Recall, Precision between two entity sets
# ---------------------------------------------------------------------------

def ner_metrics(A: set, B: set):
    """
    Compute Jaccard similarity, recall, and precision between entity sets A and B.

    A is treated as the reference set (T0 entities or previous-tier entities).
    Returns (nan, nan, nan) when A is empty (undefined comparison baseline).

    Jaccard  = |A ∩ B| / |A ∪ B|   — overall set similarity
    Recall   = |A ∩ B| / |A|        — fraction of A's entities surviving into B
    Precision= |A ∩ B| / |B|        — fraction of B's entities that came from A
    """
    if len(A) == 0:
        return np.nan, np.nan, np.nan
    inter = len(A & B)
    union = len(A | B)
    j = inter / union if union > 0 else 0
    r = inter / len(A)
    p = inter / len(B) if len(B) > 0 else np.nan
    return j, r, p


# ---------------------------------------------------------------------------
# Step 2a: Absolute drift metrics — each Tx compared against T0
# ---------------------------------------------------------------------------

def load_or_compute_ner_metrics_absolute(paired_ner: pd.DataFrame) -> pd.DataFrame:
    """
    Compute NER drift metrics for every (row, paraphraser, tier) combination,
    always comparing against the original T0 entity set.

    Output shape: ~406,203 rows × 9 columns
    Columns: dataset, key, source, paraphraser, tier,
             jaccard, recall, precision, has_entities_T0

    Cached as ner_metrics_t123_absolute.pkl.
    """
    if NER_METRICS_ABS_PICKLE.exists():
        print("Loading cached ner_metrics_t123_absolute.pkl ...")
        return pd.read_pickle(NER_METRICS_ABS_PICKLE)

    print("Computing absolute NER metrics (vs T0)...")
    rows = []
    for _, row in paired_ner.iterrows():
        A = row["ents_T0"]  # reference: original text entities
        for paraphraser, tier_cols in PARAPHRASER_TIERS.items():
            for tier, col in zip(TIERS, tier_cols):
                B = row[col]
                j, r, p = ner_metrics(A, B)
                rows.append({
                    "dataset":         row["dataset"],
                    "key":             row["key"],
                    "source":          row["source"],
                    "paraphraser":     paraphraser,
                    "tier":            tier,
                    "jaccard":         j,
                    "recall":          r,
                    "precision":       p,
                    "has_entities_T0": len(A) > 0,
                })

    ner_abs = pd.DataFrame(rows)
    ner_abs.to_pickle(NER_METRICS_ABS_PICKLE)
    print("Saved ner_metrics_t123_absolute.pkl")
    return ner_abs


# ---------------------------------------------------------------------------
# Step 2b: Incremental drift metrics — T1→T2 and T2→T3 transitions
# ---------------------------------------------------------------------------

def load_or_compute_ner_metrics_incremental(paired_ner: pd.DataFrame) -> pd.DataFrame:
    """
    Compute NER drift metrics for consecutive paraphrasing round transitions:
    T1→T2 and T2→T3 (i.e., how much additional drift occurs each round).

    Output shape: ~270,802 rows × 9 columns
    Columns: dataset, key, source, paraphraser, transition,
             jaccard, recall, precision, has_entities_T1

    Cached as ner_metrics_t123_incremental.pkl.
    """
    if NER_METRICS_INC_PICKLE.exists():
        print("Loading cached ner_metrics_t123_incremental.pkl ...")
        return pd.read_pickle(NER_METRICS_INC_PICKLE)

    print("Computing incremental NER metrics (T1→T2, T2→T3)...")
    # Each transition: (label, index_of_A_in_tier_cols, index_of_B_in_tier_cols)
    transitions = [("T1_to_T2", 0, 1), ("T2_to_T3", 1, 2)]
    rows = []
    for _, row in paired_ner.iterrows():
        for paraphraser, tier_cols in PARAPHRASER_TIERS.items():
            for transition, idx_a, idx_b in transitions:
                A = row[tier_cols[idx_a]]   # entity set from the earlier round
                B = row[tier_cols[idx_b]]   # entity set from the later round
                j, r, p = ner_metrics(A, B)
                rows.append({
                    "dataset":         row["dataset"],
                    "key":             row["key"],
                    "source":          row["source"],
                    "paraphraser":     paraphraser,
                    "transition":      transition,
                    "jaccard":         j,
                    "recall":          r,
                    "precision":       p,
                    # Flag rows where T1 has no entities (undefined baseline)
                    "has_entities_T1": len(row[tier_cols[0]]) > 0,
                })

    ner_inc = pd.DataFrame(rows)
    ner_inc.to_pickle(NER_METRICS_INC_PICKLE)
    print("Saved ner_metrics_t123_incremental.pkl")
    return ner_inc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from01 = __import__("01_data_processing")
    paired_t123 = from01.load_or_build_dataset()

    # Step 1: extract NER sets (slow, ~5 hours; cached after first run)
    paired_ner = load_or_compute_ner_sets(paired_t123)

    # Step 2: compute absolute and incremental metrics (fast once NER sets exist)
    ner_abs = load_or_compute_ner_metrics_absolute(paired_ner)
    ner_inc = load_or_compute_ner_metrics_incremental(paired_ner)

    # Sanity check
    print("\npaired_ner shape:  ", paired_ner.shape)   # expected: (19343, 47)
    print("ner_abs shape:     ", ner_abs.shape)         # expected: (~406203, 9)
    print("ner_inc shape:     ", ner_inc.shape)         # expected: (~270802, 9)
