import pandas as pd
from pathlib import Path
import spacy
import numpy as np

# Cached pickle files to avoid recomputing expensive steps
NER_SETS_PICKLE = Path("paired_ner_t1.pkl")
NER_METRICS_PICKLE = Path("ner_metrics_t1.pkl")


def get_ner_set(text, nlp):
    """
    Extract named entities from text and return as a normalized set.

    - Converts to lowercase
    - Removes empty strings
    - Returns unique entities only
    """
    # Handle missing or invalid text
    if not isinstance(text, str) or text.strip() == "":
        return set()

    # Run spaCy NER pipeline
    doc = nlp(text)

    # Return cleaned set of entity strings
    return {ent.text.lower().strip() for ent in doc.ents if ent.text.strip() != ""}


def load_or_compute_ner_sets(paired):
    """
    For each text version (original + paraphrases),
    compute the set of named entities.

    Uses caching to avoid recomputation.
    """

    # Load cached results if available
    if NER_SETS_PICKLE.exists():
        print("Loading cached paired_ner_t1.pkl ...")
        return pd.read_pickle(NER_SETS_PICKLE)

    print("Computing NER sets for T1 dataset...")

    # Load spaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Copy dataset to avoid modifying original
    df = paired.copy()

    # Compute entity sets for each version of the text
    df["ents_T0"] = df["text_T0"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_chatgpt"] = df["text_chatgpt"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_palm"] = df["text_palm"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_dip_low"] = df["text_dipper_low"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_dip_mid"] = df["text_dipper"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_dip_high"] = df["text_dipper_high"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_pg_slight"] = df["text_pegasus_slight"].apply(lambda x: get_ner_set(x, nlp))
    df["ents_pg_full"] = df["text_pegasus_full"].apply(lambda x: get_ner_set(x, nlp))

    # Save computed entity sets for reuse
    df.to_pickle(NER_SETS_PICKLE)
    print("Saved paired_ner_t1.pkl")

    return df


def ner_metrics(A, B):
    """
    Compute similarity metrics between two entity sets:

    A = original entities (T0)
    B = paraphrased entities

    Returns:
        jaccard: overlap / union
        recall: overlap / original size
        precision: overlap / paraphrase size
    """

    # If original has no entities, metrics are undefined
    if len(A) == 0:
        return np.nan, np.nan, np.nan

    # Compute intersection and union
    inter = len(A & B)
    union = len(A | B)

    # Jaccard similarity (overlap / union)
    j = inter / union if union > 0 else 0

    # Recall (how many original entities are preserved)
    r = inter / len(A)

    # Precision (how many paraphrase entities are correct)
    p = inter / len(B) if len(B) > 0 else np.nan

    return j, r, p


def load_or_compute_ner_metrics(paired_ner):
    """
    Compute NER-based similarity metrics between original text (T0)
    and each paraphrased version.

    Uses caching to avoid recomputation.
    """

    # Load cached results if available
    if NER_METRICS_PICKLE.exists():
        print("Loading cached ner_metrics_t1.pkl ...")
        return pd.read_pickle(NER_METRICS_PICKLE)

    print("Computing NER metrics for T1 dataset...")

    rows = []

    # Iterate through each article
    for _, row in paired_ner.iterrows():
        A = row["ents_T0"]  # Original entity set

        # Compare against each paraphraser
        for name, B in [
            ("chatgpt", row["ents_chatgpt"]),
            ("palm", row["ents_palm"]),
            ("dipper_low", row["ents_dip_low"]),
            ("dipper_mid", row["ents_dip_mid"]),
            ("dipper_high", row["ents_dip_high"]),
            ("pegasus_slight", row["ents_pg_slight"]),
            ("pegasus_full", row["ents_pg_full"]),
        ]:
            # Compute similarity metrics
            j, r, p = ner_metrics(A, B)

            # Store result as one row
            rows.append({
                "dataset": row["dataset"],
                "key": row["key"],
                "source": row["source"],
                "paraphraser": name,
                "jaccard": j,
                "recall": r,
                "precision": p,
                "has_entities_T0": len(A) > 0,  # Used later for filtering
            })

    # Convert results into DataFrame
    ner_df = pd.DataFrame(rows)

    # Cache results
    ner_df.to_pickle(NER_METRICS_PICKLE)
    print("Saved ner_metrics_t1.pkl")

    return ner_df


if __name__ == "__main__":
    # Import dataset preparation function
    from dataset_prep import load_or_build_dataset

    # Load dataset
    paired_t1 = load_or_build_dataset()

    # Compute entity sets
    paired_ner = load_or_compute_ner_sets(paired_t1)

    # Compute similarity metrics
    ner_df = load_or_compute_ner_metrics(paired_ner)