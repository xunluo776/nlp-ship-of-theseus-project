"""
01_data_processing.py
=====================
Data Processing & Version Alignment

Loads all seven paraphrased CSV datasets, normalizes version names,
pivots into a unified paired dataframe with all 22 text versions
(T0 + T1/T2/T3 for each of 7 paraphrasers), and caches the result
as paired_all_t123.pkl for reuse across the analysis pipeline.

Expected input:  data/{dataset}_paraphrased.csv  (7 files)
Expected output: paired_all_t123.pkl
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Display settings — useful for debugging / inspection
# ---------------------------------------------------------------------------
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All seven source dataset CSV filenames
DATASETS = [
    "cmv_paraphrased.csv",
    "eli5_paraphrased.csv",
    "sci_gen_paraphrased.csv",
    "tldr_paraphrased.csv",
    "wp_paraphrased.csv",
    "xsum_paraphrased.csv",
    "yelp_paraphrased.csv",
]

# Directory containing the CSV files
DATA_DIR = Path("data")

# Cached pickle file — avoids rebuilding the paired dataset on every run
PAIRED_ALL_PICKLE = Path("paired_all_t123.pkl")

# Maps all 22 raw version_name values to clean column names.
# The naming convention is text_{paraphraser}_{tier} where tier is T1/T2/T3.
# T0 = original text; each paraphraser has three successive rounds.
TARGET_VERSIONS = {
    "original":                                         "text_T0",
    "chatgpt":                                          "text_chatgpt_T1",
    "chatgpt_chatgpt":                                  "text_chatgpt_T2",
    "chatgpt_chatgpt_chatgpt":                          "text_chatgpt_T3",
    "palm":                                             "text_palm_T1",
    "palm_palm":                                        "text_palm_T2",
    "palm_palm_palm":                                   "text_palm_T3",
    "dipper(low)":                                      "text_dipper_low_T1",
    "dipper(low)_dipper(low)":                          "text_dipper_low_T2",
    "dipper(low)_dipper(low)_dipper(low)":              "text_dipper_low_T3",
    "dipper":                                           "text_dipper_T1",
    "dipper_dipper":                                    "text_dipper_T2",
    "dipper_dipper_dipper":                             "text_dipper_T3",
    "dipper(high)":                                     "text_dipper_high_T1",
    "dipper(high)_dipper(high)":                        "text_dipper_high_T2",
    "dipper(high)_dipper(high)_dipper(high)":           "text_dipper_high_T3",
    "pegasus(slight)":                                  "text_pegasus_slight_T1",
    "pegasus(slight)_pegasus(slight)":                  "text_pegasus_slight_T2",
    "pegasus(slight)_pegasus(slight)_pegasus(slight)":  "text_pegasus_slight_T3",
    "pegasus(full)":                                    "text_pegasus_full_T1",
    "pegasus(full)_pegasus(full)":                      "text_pegasus_full_T2",
    "pegasus(full)_pegasus(full)_pegasus(full)":        "text_pegasus_full_T3",
}

# Final column order for readability
COLUMN_ORDER = [
    "dataset", "key", "source",
    "text_T0",
    "text_chatgpt_T1", "text_chatgpt_T2", "text_chatgpt_T3",
    "text_palm_T1",    "text_palm_T2",    "text_palm_T3",
    "text_dipper_low_T1",  "text_dipper_low_T2",  "text_dipper_low_T3",
    "text_dipper_T1",      "text_dipper_T2",      "text_dipper_T3",
    "text_dipper_high_T1", "text_dipper_high_T2", "text_dipper_high_T3",
    "text_pegasus_slight_T1", "text_pegasus_slight_T2", "text_pegasus_slight_T3",
    "text_pegasus_full_T1",   "text_pegasus_full_T2",   "text_pegasus_full_T3",
]


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def load_or_build_dataset() -> pd.DataFrame:
    """
    Load the pivoted paired dataset from cache, or build it from raw CSVs.

    Returns
    -------
    pd.DataFrame
        Shape (19343, 25): one row per (dataset, key, source) triple,
        with 22 text columns covering T0 and all T1/T2/T3 paraphrase rounds.
    """
    if PAIRED_ALL_PICKLE.exists():
        print("Loading cached paired_all_t123.pkl ...")
        return pd.read_pickle(PAIRED_ALL_PICKLE)

    print("Building paired_all_t123 dataset from CSV files...")
    dfs = []
    for fname in DATASETS:
        df = pd.read_csv(DATA_DIR / fname)
        # Tag each row with its source dataset name (e.g. "cmv", "eli5")
        df["dataset"] = fname.replace("_paraphrased.csv", "")
        # Fix occasional typo in version_name ("orignal" → "original")
        df["version_name"] = df["version_name"].replace("orignal", "original")
        dfs.append(df)

    # Concatenate all seven datasets into one long-format dataframe
    full = pd.concat(dfs, ignore_index=True)

    # Map version_name strings to clean column names; drop unrecognized versions
    filtered = full.copy()
    filtered["version_col"] = filtered["version_name"].map(TARGET_VERSIONS)
    filtered = filtered[filtered["version_col"].notna()].copy()

    # Pivot: each row = one (dataset, key, source) with all text versions as columns
    paired = filtered.pivot(
        index=["dataset", "key", "source"],
        columns="version_col",
        values="text"
    ).reset_index()

    # Reorder columns for readability
    paired = paired[COLUMN_ORDER]

    # Cache to disk for fast reuse
    paired.to_pickle(PAIRED_ALL_PICKLE)
    print("Saved paired_all_t123.pkl")
    return paired


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    paired_t123 = load_or_build_dataset()

    print("paired_all_t123 shape:", paired_t123.shape)

    # Quick structural sanity check: show two rows per dataset
    for ds in paired_t123["dataset"].unique():
        print(f"\n--- {ds} ---")
        print(paired_t123[paired_t123["dataset"] == ds].head(2).to_string())
