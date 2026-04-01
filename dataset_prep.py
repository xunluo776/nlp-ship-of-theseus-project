import pandas as pd
from pathlib import Path
import spacy
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure pandas display for easier debugging and inspection
pd.set_option("display.max_colwidth", None)   # Show full text in columns
pd.set_option("display.max_columns", None)    # Show all columns
pd.set_option("display.width", 2000)          # Increase display width

# List of all paraphrased dataset CSV files
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

# Cached pickle file to avoid rebuilding the paired dataset every run
PAIRED_ALL_PICKLE = Path("paired_all_t1.pkl")


def load_or_build_dataset():
    """
    Load all CSVs, normalize version names, pivot into paired format, and cache result.

    Returns:
        pd.DataFrame: The paired dataset where each row contains multiple paraphrase versions.
    """

    # If cached file exists, load it instead of recomputing
    if PAIRED_ALL_PICKLE.exists():
        print("Loading cached paired_all_t1.pkl ...")
        return pd.read_pickle(PAIRED_ALL_PICKLE)

    print("Building paired_all_t1 dataset from CSV files...")

    dfs = []  # Store individual dataset DataFrames

    # Loop through each dataset file
    for fname in DATASETS:
        df = pd.read_csv(DATA_DIR / fname)  # Read CSV file

        # Add dataset name (e.g., "cmv", "eli5") based on filename
        df["dataset"] = fname.replace("_paraphrased.csv", "")

        # Fix common typo in version_name column
        df["version_name"] = df["version_name"].replace("orignal", "original")

        dfs.append(df)  # Collect dataframe

    # Combine all datasets into one large dataframe
    full = pd.concat(dfs, ignore_index=True)

    # Map version names to standardized column names for pivoting
    target_versions = {
        "original": "text_T0",
        "chatgpt": "text_chatgpt",
        "dipper(low)": "text_dipper_low",
        "dipper": "text_dipper",
        "dipper(high)": "text_dipper_high",
        "pegasus(slight)": "text_pegasus_slight",
        "pegasus(full)": "text_pegasus_full",
        "palm": "text_palm",
    }

    # Create a new column mapping version_name → standardized column name
    filtered = full.copy()
    filtered["version_col"] = filtered["version_name"].map(target_versions)

    # Keep only rows with valid/recognized versions
    filtered = filtered[filtered["version_col"].notna()].copy()

    # Pivot the dataset:
    # Each row becomes a unique (dataset, key, source),
    # columns become different paraphrase versions
    paired = filtered.pivot(
        index=["dataset", "key", "source"],
        columns="version_col",
        values="text"
    ).reset_index()

    # Reorder columns for better readability
    paired = paired[[
        "dataset", "key", "source",
        "text_T0",
        "text_chatgpt",
        "text_palm",
        "text_dipper_low",
        "text_dipper",
        "text_dipper_high",
        "text_pegasus_slight",
        "text_pegasus_full",
    ]]

    # Cache the result for faster future runs
    paired.to_pickle(PAIRED_ALL_PICKLE)
    print("Saved paired_all_t1.pkl")

    return paired


if __name__ == "__main__":
    # Load or build the paired dataset
    paired_t1 = load_or_build_dataset()

    # Print dataset shape (rows, columns)
    print("paired_all_t1 shape:", paired_t1.shape)

    # Show two sample rows from each dataset to verify structure
    for ds in paired_t1["dataset"].unique():
        print(f"\n--- {ds} ---")
        print(paired_t1[paired_t1["dataset"] == ds].head(2))