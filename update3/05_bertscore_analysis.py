"""
05_bertscore_analysis.py
========================
BERTScore Analysis — Visualization, Statistics & Authorship Similarity

Produces all plots and summary tables for the BERTScore section and the
T0 authorship similarity section, organized into five analyses:

  1. Global overview  — F1, precision, recall across all paraphrasers/rounds
  2. Human vs. LLM   — F1 gap between source groups
  3. Domain analysis  — F1 by dataset, all paraphrasers
  4. NER vs. BERTScore comparison — side-by-side 3×2 plot
  5. T0 authorship similarity — NER Jaccard between Human and LLM source texts

Outputs saved to output/update3/
Toggle REPORT_MODE = True for ACM single-column figures.

Expected input:
  output/update3/bertscore_all_datasets_t123.pkl
  ner_metrics_t123_absolute.pkl
  paired_ner_t123.pkl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

REPORT_MODE = False
suffix = "_report" if REPORT_MODE else ""

PARAPHRASER_ORDER  = ["chatgpt", "palm", "dipper_low", "dipper_mid",
                       "dipper_high", "pegasus_slight", "pegasus_full"]
TIER_ORDER         = ["T1", "T2", "T3"]
TIER_ORDER_WITH_T0 = ["T0", "T1", "T2", "T3"]
DATASET_ORDER      = ["cmv", "eli5", "sci_gen", "tldr", "wp", "xsum", "yelp"]
TIER_COLORS        = {"T1": "#4C72B0", "T2": "#DD8452", "T3": "#55A868"}

OUT_DIR = Path("output/update3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if REPORT_MODE:
    LINEPLOT_FIGSIZE = (10, 16);  LINEPLOT_NROWS = 4;  LINEPLOT_NCOLS = 2
    ALL_FIGSIZE      = (10, 4);   ALL_NROWS      = 4;  ALL_NCOLS      = 2
    TITLE_FS = 14;  LABEL_FS = 12;  TICK_FS = 11;  LEGEND_FS = 10;  SUP_Y = 1.01
else:
    LINEPLOT_FIGSIZE = (20, 10);  LINEPLOT_NROWS = 2;  LINEPLOT_NCOLS = 4
    ALL_FIGSIZE      = (28, 6);   ALL_NROWS      = 2;  ALL_NCOLS      = 4
    TITLE_FS = 16;  LABEL_FS = 14;  TICK_FS = 12;  LEGEND_FS = 11;  SUP_Y = 1.02


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    """Load unified BERTScore table, NER absolute metrics, and NER entity sets."""
    bert_all    = pd.read_pickle("output/update3/bertscore_all_datasets_t123.pkl")
    bert_valid  = bert_all.dropna(subset=["bert_f1", "bert_precision", "bert_recall"])
    excluded    = len(bert_all) - len(bert_valid)
    print(f"BERTScore: {len(bert_valid)} valid rows "
          f"({excluded} excluded for missing scores)")

    ner_abs     = pd.read_pickle("ner_metrics_t123_absolute.pkl")
    valid_abs   = ner_abs[ner_abs["has_entities_T0"]].copy()

    paired_ner  = pd.read_pickle("paired_ner_t123.pkl")

    return bert_valid, valid_abs, paired_ner


# ---------------------------------------------------------------------------
# Section 1 — Global overview: F1/precision/recall across rounds
# ---------------------------------------------------------------------------

def plot_global_overview(bert_valid: pd.DataFrame):
    """
    Summary table and line plots for BERTScore F1, precision, and recall
    across all paraphrasers and rounds. T0 = 1.0 reference point prepended.
    """
    summary = (
        bert_valid
        .groupby(["paraphraser", "tier"])[["bert_f1", "bert_precision", "bert_recall"]]
        .agg(["mean", "std"])
        .reindex(pd.MultiIndex.from_product(
            [PARAPHRASER_ORDER, TIER_ORDER], names=["paraphraser", "tier"]
        ))
    )
    print("=== Global BERTScore Summary ===")
    print(summary.round(4).to_string())

    # Line plots for F1, precision, recall
    for metric, label in [
        ("bert_f1",        "F1"),
        ("bert_precision", "Precision"),
        ("bert_recall",    "Recall"),
    ]:
        plt.figure(figsize=(9.5, 6.5))
        for paraphraser in PARAPHRASER_ORDER:
            subset = (
                bert_valid[bert_valid["paraphraser"] == paraphraser]
                .groupby("tier")[metric].mean().reindex(TIER_ORDER)
            )
            values = [1.0] + subset.values.tolist()   # prepend T0 = 1.0
            plt.plot(TIER_ORDER_WITH_T0, values, marker="o", label=paraphraser)
        plt.title(f"BERTScore {label} vs Paraphrasing Round (All Paraphrasers)",
                  fontsize=TITLE_FS + 2)
        plt.xlabel("Paraphrasing Round", fontsize=LABEL_FS)
        plt.ylabel(f"Mean BERTScore {label}", fontsize=LABEL_FS)
        plt.xticks(fontsize=TICK_FS);  plt.yticks(fontsize=TICK_FS)
        plt.legend(title="Paraphraser", fontsize=LEGEND_FS,
                   title_fontsize=LEGEND_FS + 1)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"bertscore_{metric.replace('bert_', '')}_by_round{suffix}.png",
                    dpi=300, bbox_inches="tight")
        plt.show()


# ---------------------------------------------------------------------------
# Section 2 — Human vs. LLM
# ---------------------------------------------------------------------------

def plot_human_vs_llm(bert_valid: pd.DataFrame):
    """
    Line plots comparing BERTScore F1 for Human vs. LLM source texts.
    Also prints summary tables and the LLM-minus-Human gap table.
    """
    bert_valid = bert_valid.copy()
    bert_valid["source_group"] = bert_valid["source"].apply(
        lambda x: "Human" if x == "Human" else "LLM"
    )

    # Summary tables with T0 = 1.0 column
    for group in ["Human", "LLM"]:
        print(f"\n=== BERTScore F1 — {group} ===")
        tbl = (
            bert_valid[bert_valid["source_group"] == group]
            .groupby(["paraphraser", "tier"])["bert_f1"].mean()
            .unstack("tier").reindex(index=PARAPHRASER_ORDER, columns=TIER_ORDER)
        )
        tbl.insert(0, "T0", 1.0)
        print(tbl.round(4).to_string())

    # Gap table
    print("\n=== F1 Gap (LLM minus Human) ===")
    human_f1 = (
        bert_valid[bert_valid["source_group"] == "Human"]
        .groupby(["paraphraser", "tier"])["bert_f1"].mean()
        .unstack("tier").reindex(index=PARAPHRASER_ORDER, columns=TIER_ORDER)
    )
    llm_f1 = (
        bert_valid[bert_valid["source_group"] == "LLM"]
        .groupby(["paraphraser", "tier"])["bert_f1"].mean()
        .unstack("tier").reindex(index=PARAPHRASER_ORDER, columns=TIER_ORDER)
    )
    print((llm_f1 - human_f1).round(4).to_string())

    # Line plot: F1 vs round, Human vs LLM, one subplot per paraphraser
    fig, axes = plt.subplots(LINEPLOT_NROWS, LINEPLOT_NCOLS,
                             figsize=LINEPLOT_FIGSIZE, sharey=True)
    axes = axes.flatten()
    for i, paraphraser in enumerate(PARAPHRASER_ORDER):
        ax = axes[i]
        for group in ["Human", "LLM"]:
            subset = (
                bert_valid[
                    (bert_valid["paraphraser"] == paraphraser) &
                    (bert_valid["source_group"] == group)
                ].groupby("tier")["bert_f1"].mean().reindex(TIER_ORDER)
            )
            values = [1.0] + subset.values.tolist()
            ax.plot(TIER_ORDER_WITH_T0, values, marker="o", label=group)
        ax.set_title(paraphraser, fontsize=TITLE_FS)
        ax.set_xlabel("Paraphrasing Round", fontsize=LABEL_FS)
        ax.set_ylabel("Mean BERTScore F1", fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS)
        ax.legend(fontsize=LEGEND_FS)
    for j in range(len(PARAPHRASER_ORDER), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("BERTScore F1: Human vs LLM Source across Paraphrasing Rounds",
                 fontsize=TITLE_FS + 2, y=SUP_Y)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"bertscore_human_vs_llm_f1_by_round{suffix}.png",
                dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Section 3 — Domain analysis
# ---------------------------------------------------------------------------

def plot_domain_analysis(bert_valid: pd.DataFrame):
    """
    Grouped bar plots showing BERTScore F1 by dataset for all paraphrasers.
    Also prints summary tables for F1 and recall by domain.
    """
    # Summary tables
    print("=== BERTScore F1 by Dataset, Paraphraser, and Round ===")
    domain_summary = (
        bert_valid.groupby(["dataset", "paraphraser", "tier"])["bert_f1"]
        .mean().unstack("tier").reindex(columns=TIER_ORDER)
    )
    print(domain_summary.round(4).to_string())

    datasets_ordered = DATASET_ORDER
    x       = np.arange(len(datasets_ordered))
    bar_w   = 0.25
    offsets = np.linspace(-(len(TIER_ORDER) - 1) / 2,
                           (len(TIER_ORDER) - 1) / 2,
                           len(TIER_ORDER)) * bar_w

    # F1 and recall grouped bar plots
    for metric, ylabel, fname_suffix in [
        ("bert_f1",     "Mean BERTScore F1",     "f1"),
        ("bert_recall", "Mean BERTScore Recall", "recall"),
    ]:
        fig, axes = plt.subplots(ALL_NROWS, ALL_NCOLS,
                                 figsize=(ALL_FIGSIZE[0], ALL_FIGSIZE[1] * ALL_NROWS),
                                 sharey=True)
        axes = axes.flatten()
        for i, paraphraser in enumerate(PARAPHRASER_ORDER):
            ax = axes[i]
            subset = bert_valid[bert_valid["paraphraser"] == paraphraser]
            for j, tier in enumerate(TIER_ORDER):
                td = subset[subset["tier"] == tier]
                means = [td[td["dataset"] == ds][metric].mean()
                         for ds in datasets_ordered]
                stds  = [td[td["dataset"] == ds][metric].std()
                         for ds in datasets_ordered]
                ax.bar(x + offsets[j], means, width=bar_w,
                       label=tier, color=TIER_COLORS[tier], alpha=0.9)
                ax.errorbar(x + offsets[j], means, yerr=stds,
                            fmt="none", ecolor="black", capsize=4, linewidth=1.0)
            ax.set_title(paraphraser, fontsize=TITLE_FS)
            ax.set_xticks(x)
            ax.set_xticklabels(datasets_ordered, rotation=45, ha="right",
                               fontsize=TICK_FS)
            ax.set_xlabel("Dataset", fontsize=LABEL_FS)
            ax.set_ylabel(ylabel, fontsize=LABEL_FS)
            ax.set_ylim(0.8, 1.02)
            ax.tick_params(labelsize=TICK_FS)
            ax.legend(title="Round", fontsize=LEGEND_FS,
                      title_fontsize=LEGEND_FS + 1)
        for j in range(len(PARAPHRASER_ORDER), len(axes)):
            axes[j].set_visible(False)
        metric_label = "F1" if metric == "bert_f1" else "Recall"
        plt.suptitle(
            f"BERTScore {metric_label} by Dataset and Paraphrasing Round "
            f"(All Paraphrasers)", fontsize=TITLE_FS + 2, y=SUP_Y)
        plt.tight_layout()
        plt.savefig(
            OUT_DIR / f"bertscore_{fname_suffix}_by_dataset_all_paraphrasers{suffix}.png",
            dpi=300, bbox_inches="tight")
        plt.show()


# ---------------------------------------------------------------------------
# Section 4 — NER vs. BERTScore side-by-side comparison
# ---------------------------------------------------------------------------

def plot_ner_vs_bertscore(bert_valid: pd.DataFrame, valid_abs: pd.DataFrame):
    """
    3×2 subplot comparing NER and BERTScore metrics side by side:
      Row 0: NER Recall   | BERTScore Recall
      Row 1: NER Precision | BERTScore Precision
      Row 2: NER Jaccard  | BERTScore F1

    Also prints all six summary tables for cross-metric comparison.
    """
    if REPORT_MODE:
        figsize = (14, 18);  title_fs = 13;  label_fs = 11
        tick_fs = 10;        legend_fs = 9;  sup_y    = 1.01
    else:
        figsize = (20, 15);  title_fs = 14;  label_fs = 12
        tick_fs = 11;        legend_fs = 10; sup_y    = 1.02

    # Print all six tables
    for label, data, metric in [
        ("NER Recall",          valid_abs,  "recall"),
        ("BERTScore Recall",    bert_valid, "bert_recall"),
        ("NER Precision",       valid_abs,  "precision"),
        ("BERTScore Precision", bert_valid, "bert_precision"),
        ("NER Jaccard",         valid_abs,  "jaccard"),
        ("BERTScore F1",        bert_valid, "bert_f1"),
    ]:
        tbl = (
            data.groupby(["paraphraser", "tier"])[metric].mean()
            .unstack("tier").reindex(index=PARAPHRASER_ORDER, columns=TIER_ORDER)
        )
        tbl.insert(0, "T0", 1.0)
        print(f"\n=== {label} by Paraphraser and Round ===")
        print(tbl.round(4).to_string())

    # Side-by-side line plots
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharey=False)
    plot_configs = [
        (0, 0, valid_abs,   "recall",         "NER Recall"),
        (0, 1, bert_valid,  "bert_recall",    "BERTScore Recall"),
        (1, 0, valid_abs,   "precision",      "NER Precision"),
        (1, 1, bert_valid,  "bert_precision", "BERTScore Precision"),
        (2, 0, valid_abs,   "jaccard",        "NER Jaccard"),
        (2, 1, bert_valid,  "bert_f1",        "BERTScore F1"),
    ]
    for row, col, data, metric, title in plot_configs:
        ax = axes[row][col]
        for paraphraser in PARAPHRASER_ORDER:
            subset = (
                data[data["paraphraser"] == paraphraser]
                .groupby("tier")[metric].mean().reindex(TIER_ORDER)
            )
            values = [1.0] + subset.values.tolist()
            ax.plot(TIER_ORDER_WITH_T0, values, marker="o", label=paraphraser)
        ax.set_title(title, fontsize=title_fs)
        ax.set_xlabel("Paraphrasing Round", fontsize=label_fs)
        ax.set_ylabel("Mean Score", fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.legend(title="Paraphraser", fontsize=legend_fs,
                  title_fontsize=legend_fs + 1)
    plt.suptitle("NER Drift vs BERTScore across Paraphrasing Rounds",
                 fontsize=title_fs + 3, y=sup_y)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"ner_vs_bertscore_comparison{suffix}.png",
                dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Section 5 — T0 authorship similarity: Human vs. LLM
# ---------------------------------------------------------------------------

def analyze_t0_authorship_similarity(paired_ner: pd.DataFrame,
                                     valid_abs: pd.DataFrame):
    """
    Test whether the null Human-vs-LLM paraphrasing result is trivially
    explained by the source texts already being similar at T0.

    For each document key, computes NER Jaccard, recall, and precision
    between the Human T0 entity set and each LLM source's T0 entity set.
    Compares the resulting T0 similarity against T1 paraphrasing Jaccard
    to determine whether authorship differences exist at the source level.
    """
    rows = []
    for (dataset, key), group in paired_ner.groupby(["dataset", "key"]):
        # Get Human T0 entity set (skip if Human has no entities)
        human_rows = group[group["source"] == "Human"]
        if len(human_rows) == 0:
            continue
        human_ents = human_rows.iloc[0]["ents_T0"]
        if len(human_ents) == 0:
            continue

        # Compare against each LLM source
        for _, llm_row in group[group["source"] != "Human"].iterrows():
            llm_ents = llm_row["ents_T0"]
            if len(llm_ents) == 0:
                continue

            inter     = len(human_ents & llm_ents)
            union     = len(human_ents | llm_ents)
            jaccard   = inter / union if union > 0 else 0.0
            recall    = inter / len(human_ents)
            precision = inter / len(llm_ents) if len(llm_ents) > 0 else float("nan")

            rows.append({
                "dataset":      dataset,
                "key":          key,
                "llm_source":   llm_row["source"],
                "jaccard":      jaccard,
                "recall":       recall,
                "precision":    precision,
                "human_n_ents": len(human_ents),
                "llm_n_ents":   len(llm_ents),
            })

    t0_sim_df = pd.DataFrame(rows)
    print(f"Total Human-LLM T0 comparisons: {len(t0_sim_df)}")

    # Global summary statistics
    print("\n=== Global T0 Similarity: Human vs LLM ===")
    print(t0_sim_df[["jaccard", "recall", "precision"]]
          .agg(["mean", "std", "median"]).round(4).to_string())

    # Per LLM source
    print("\n=== T0 Similarity by LLM Source ===")
    print(t0_sim_df.groupby("llm_source")[["jaccard", "recall", "precision"]]
          .agg(["mean", "std"]).round(4).to_string())

    # Per dataset
    print("\n=== T0 Similarity by Dataset ===")
    print(t0_sim_df.groupby("dataset")[["jaccard", "recall", "precision"]]
          .agg(["mean", "std"]).round(4).to_string())

    # Distribution plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, metric, color in zip(
        axes,
        ["jaccard", "recall", "precision"],
        ["#4C72B0", "#DD8452", "#55A868"]
    ):
        ax.hist(t0_sim_df[metric].dropna(), bins=50,
                color=color, alpha=0.8, edgecolor="white")
        ax.set_title(f"T0 {metric.capitalize()}\n(Human vs LLM)", fontsize=14)
        ax.set_xlabel(metric.capitalize(), fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        mean_val = t0_sim_df[metric].mean()
        ax.axvline(mean_val, color="red", linestyle="--",
                   label=f"Mean: {mean_val:.3f}")
        ax.legend(fontsize=11)
    plt.suptitle("NER Similarity between Human and LLM T0 Texts (Same Document Key)",
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "t0_human_vs_llm_ner_similarity.png",
                dpi=300, bbox_inches="tight")
    plt.show()

    # Box plots by dataset
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, metric in zip(axes, ["jaccard", "recall", "precision"]):
        sns.boxplot(data=t0_sim_df, x="dataset", y=metric,
                    ax=ax, order=DATASET_ORDER)
        ax.set_title(f"T0 {metric.capitalize()} by Dataset", fontsize=14)
        ax.set_xlabel("Dataset", fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.tick_params(axis="x", rotation=45)
    plt.suptitle("NER Similarity between Human and LLM T0 Texts by Dataset",
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "t0_human_vs_llm_ner_similarity_by_dataset.png",
                dpi=300, bbox_inches="tight")
    plt.show()

    # Key comparison: T0 Human-LLM similarity vs T1 paraphrasing Jaccard
    print("\n=== T0 Human vs LLM Jaccard (baseline) ===")
    print(f"Mean:   {t0_sim_df['jaccard'].mean():.4f}")
    print(f"Median: {t0_sim_df['jaccard'].median():.4f}")
    print(f"Std:    {t0_sim_df['jaccard'].std():.4f}")

    print("\n=== T1 Paraphrasing Jaccard (reference) ===")
    t1_jaccard = (
        valid_abs[valid_abs["tier"] == "T1"]
        .groupby("paraphraser")["jaccard"].mean()
        .reindex(PARAPHRASER_ORDER)
    )
    print(t1_jaccard.round(4).to_string())

    t0_mean = t0_sim_df["jaccard"].mean()
    t1_mean = valid_abs[valid_abs["tier"] == "T1"]["jaccard"].mean()
    print(f"\nMean T0 Human-LLM Jaccard:    {t0_mean:.4f}")
    print(f"Mean T1 paraphrasing Jaccard: {t1_mean:.4f}")
    if t0_mean < t1_mean:
        print("→ Human and LLM T0 texts are LESS similar to each other than")
        print("  T1 paraphrases are to their T0 originals.")
        print("  Authorship differences DO exist at T0.")
    else:
        print("→ Human and LLM T0 texts are MORE similar to each other than")
        print("  T1 paraphrases are to their T0 originals.")
        print("  Authorship differences are minimal at T0.")

    return t0_sim_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bert_valid, valid_abs, paired_ner = load_data()

    print("\n--- Section 1: Global Overview ---")
    plot_global_overview(bert_valid)

    print("\n--- Section 2: Human vs. LLM ---")
    plot_human_vs_llm(bert_valid)

    print("\n--- Section 3: Domain Analysis ---")
    plot_domain_analysis(bert_valid)

    print("\n--- Section 4: NER vs. BERTScore Comparison ---")
    plot_ner_vs_bertscore(bert_valid, valid_abs)

    print("\n--- Section 5: T0 Authorship Similarity ---")
    t0_sim_df = analyze_t0_authorship_similarity(paired_ner, valid_abs)
