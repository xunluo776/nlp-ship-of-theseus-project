"""
03_ner_analysis.py
==================
NER Drift Analysis — Visualization & Statistics

Produces all plots and summary tables for the NER drift section,
organized into five analysis angles:

  1. Global overview  — all paraphrasers, all rounds
  2. Domain × round  — Pegasus and all-paraphrasers breakdown by dataset
  3. Dipper intensity — low / mid / high comparison (notebook only)
  4. Human vs. LLM   — source group comparison across rounds
  5. Incremental     — T0→T1, T1→T2, T2→T3 survival rate table

Outputs saved to output/update3/
Toggle REPORT_MODE = True to produce single-column figures for the ACM report.

Expected input:  ner_metrics_t123_absolute.pkl
                 ner_metrics_t123_incremental.pkl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

# Toggle to switch between wide (presentation) and narrow (ACM report) layouts
REPORT_MODE = False
suffix = "_report" if REPORT_MODE else ""

# Consistent ordering used across all plots and tables
PARAPHRASER_ORDER = [
    "chatgpt", "palm", "dipper_low", "dipper_mid",
    "dipper_high", "pegasus_slight", "pegasus_full",
]
TIER_ORDER           = ["T1", "T2", "T3"]
TIER_ORDER_WITH_T0   = ["T0", "T1", "T2", "T3"]
DATASET_ORDER        = ["cmv", "eli5", "sci_gen", "tldr", "wp", "xsum", "yelp"]
TIER_COLORS          = {"T1": "#4C72B0", "T2": "#DD8452", "T3": "#55A868"}

OUT_DIR = Path("output/update3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Layout parameters switch based on REPORT_MODE
if REPORT_MODE:
    LINEPLOT_FIGSIZE  = (10, 16);  LINEPLOT_NROWS = 4;  LINEPLOT_NCOLS = 2
    BARPLOT_FIGSIZE   = (8, 15);   BARPLOT_NROWS  = 3;  BARPLOT_NCOLS  = 1
    TITLE_FS = 14;  LABEL_FS = 12;  TICK_FS = 11;  LEGEND_FS = 10;  SUP_Y = 1.01
else:
    LINEPLOT_FIGSIZE  = (20, 10);  LINEPLOT_NROWS = 2;  LINEPLOT_NCOLS = 4
    BARPLOT_FIGSIZE   = (20, 6);   BARPLOT_NROWS  = 1;  BARPLOT_NCOLS  = 3
    TITLE_FS = 16;  LABEL_FS = 14;  TICK_FS = 12;  LEGEND_FS = 11;  SUP_Y = 1.02


# ---------------------------------------------------------------------------
# Load cached metrics
# ---------------------------------------------------------------------------

def load_metrics():
    """Load absolute and incremental NER metrics from cache."""
    ner_abs = pd.read_pickle("ner_metrics_t123_absolute.pkl")
    ner_inc = pd.read_pickle("ner_metrics_t123_incremental.pkl")
    # Keep only rows where T0 had at least one entity (undefined otherwise)
    valid_abs = ner_abs[ner_abs["has_entities_T0"]].copy()
    valid_inc = ner_inc[ner_inc["has_entities_T1"]].copy()
    excluded = len(ner_abs) - len(valid_abs)
    print(f"Excluded rows (no T0 entities): {excluded} ({excluded/len(ner_abs):.2%})")
    return valid_abs, valid_inc


# ---------------------------------------------------------------------------
# Section 1 — Global overview: all paraphrasers, all rounds
# ---------------------------------------------------------------------------

def plot_global_overview(valid_abs: pd.DataFrame):
    """
    Summary table and line plots for recall, precision, and Jaccard
    across all paraphrasers and rounds. T0 = 1.0 reference point added.
    """
    # Summary table
    summary = (
        valid_abs
        .groupby(["paraphraser", "tier"])[["jaccard", "recall", "precision"]]
        .agg(["mean", "std"])
        .reindex(pd.MultiIndex.from_product(
            [PARAPHRASER_ORDER, TIER_ORDER], names=["paraphraser", "tier"]
        ))
    )
    print("=== Global NER Summary (mean ± std) ===")
    print(summary.round(4).to_string())

    # Line plots for all three metrics including T0 = 1.0
    for metric, ylabel, fname in [
        ("recall",    "Mean Recall",    "ner_global_recall_by_round_t123.png"),
        ("precision", "Mean Precision", "ner_global_precision_by_round_t123.png"),
        ("jaccard",   "Mean Jaccard",   "ner_global_jaccard_by_round_t123.png"),
    ]:
        plt.figure(figsize=(9.5, 6.5))
        for paraphraser in PARAPHRASER_ORDER:
            subset = (
                valid_abs[valid_abs["paraphraser"] == paraphraser]
                .groupby("tier")[metric].mean().reindex(TIER_ORDER)
            )
            values = [1.0] + subset.values.tolist()   # prepend T0 = 1.0
            plt.plot(TIER_ORDER_WITH_T0, values, marker="o", label=paraphraser)
        plt.title(f"NER {ylabel.replace('Mean ', '')} vs Paraphrasing Round "
                  f"(All Paraphrasers)", fontsize=18)
        plt.xlabel("Paraphrasing Round", fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.xticks(fontsize=14);  plt.yticks(fontsize=14)
        plt.legend(title="Paraphraser", fontsize=11, title_fontsize=12)
        plt.tight_layout()
        plt.savefig(OUT_DIR / fname, dpi=300, bbox_inches="tight")
        plt.show()


# ---------------------------------------------------------------------------
# Section 2 — Domain × round: Pegasus and all paraphrasers
# ---------------------------------------------------------------------------

def plot_domain_analysis(valid_abs: pd.DataFrame):
    """
    Grouped bar plots showing NER recall by dataset for:
      a) Pegasus slight vs full
      b) All 7 paraphrasers
    """
    pg = valid_abs[valid_abs["paraphraser"].isin(["pegasus_slight", "pegasus_full"])]
    datasets_ordered = DATASET_ORDER

    x        = np.arange(len(datasets_ordered))
    bar_w    = 0.25
    offsets  = np.linspace(-(len(TIER_ORDER) - 1) / 2,
                            (len(TIER_ORDER) - 1) / 2,
                            len(TIER_ORDER)) * bar_w

    # ---- Pegasus summary tables ----
    pg_summary = (
        pg.groupby(["paraphraser", "tier"])[["jaccard", "recall", "precision"]]
        .agg(["mean", "std"])
        .reindex(pd.MultiIndex.from_product(
            [["pegasus_slight", "pegasus_full"], TIER_ORDER],
            names=["paraphraser", "tier"]
        ))
    )
    print("=== Pegasus Overall Summary ===")
    print(pg_summary.round(4).to_string())

    for metric in ["jaccard", "recall", "precision"]:
        print(f"\n=== Pegasus {metric.capitalize()} by Dataset and Round ===")
        tbl = (
            pg.groupby(["dataset", "paraphraser", "tier"])[metric]
            .mean().unstack("tier").reindex(columns=TIER_ORDER)
        )
        print(tbl.round(4).to_string())

    # ---- All-paraphraser recall summary ----
    print("\n=== Recall by Dataset, Paraphraser, and Round ===")
    all_summary = (
        valid_abs.groupby(["dataset", "paraphraser", "tier"])["recall"]
        .mean().unstack("tier").reindex(columns=TIER_ORDER)
    )
    print(all_summary.round(4).to_string())

    # ---- Grouped bar: all paraphrasers ----
    if REPORT_MODE:
        all_figsize = (10, 4);  all_nrows = 4;  all_ncols = 2
    else:
        all_figsize = (27, 6);  all_nrows = 2;  all_ncols = 4

    fig, axes = plt.subplots(all_nrows, all_ncols,
                             figsize=(all_figsize[0], all_figsize[1] * all_nrows),
                             sharey=True)
    axes = axes.flatten()
    for i, paraphraser in enumerate(PARAPHRASER_ORDER):
        ax = axes[i]
        subset = valid_abs[valid_abs["paraphraser"] == paraphraser]
        for j, tier in enumerate(TIER_ORDER):
            td = subset[subset["tier"] == tier]
            means = [td[td["dataset"] == ds]["recall"].mean() for ds in datasets_ordered]
            stds  = [td[td["dataset"] == ds]["recall"].std()  for ds in datasets_ordered]
            ax.bar(x + offsets[j], means, width=bar_w,
                   label=tier, color=TIER_COLORS[tier], alpha=0.9)
            ax.errorbar(x + offsets[j], means, yerr=stds,
                        fmt="none", ecolor="black", capsize=4, linewidth=1.0)
        ax.set_title(paraphraser, fontsize=TITLE_FS)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets_ordered, rotation=45, ha="right", fontsize=TICK_FS)
        ax.set_xlabel("Dataset", fontsize=LABEL_FS)
        ax.set_ylabel("Mean Recall", fontsize=LABEL_FS)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=TICK_FS)
        ax.legend(title="Round", fontsize=LEGEND_FS, title_fontsize=LEGEND_FS + 1)
    for j in range(len(PARAPHRASER_ORDER), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("NER Recall by Dataset and Paraphrasing Round (All Paraphrasers)",
                 fontsize=TITLE_FS + 2, y=SUP_Y)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"ner_all_paraphrasers_recall_by_dataset{suffix}.png",
                dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Section 3 — Dipper intensity (notebook-only; included for completeness)
# ---------------------------------------------------------------------------

def plot_dipper_intensity(valid_abs: pd.DataFrame):
    """
    Line and bar plots for Dipper low / mid / high across rounds.
    Note: included for completeness; not used in the final report/slides.
    """
    dipper_only  = valid_abs[valid_abs["paraphraser"].isin(
        ["dipper_low", "dipper_mid", "dipper_high"])]
    dipper_order = ["dipper_low", "dipper_mid", "dipper_high"]

    dipper_summary = (
        dipper_only.groupby(["paraphraser", "tier"])[["recall", "precision"]]
        .agg(["mean", "std"])
        .reindex(pd.MultiIndex.from_product(
            [dipper_order, TIER_ORDER], names=["paraphraser", "tier"]
        ))
    )
    print("=== Dipper Summary ===")
    print(dipper_summary.round(4).to_string())

    # Line plot: recall
    plt.figure(figsize=(8, 5))
    for p in dipper_order:
        s = (dipper_only[dipper_only["paraphraser"] == p]
             .groupby("tier")["recall"].mean().reindex(TIER_ORDER))
        plt.plot(TIER_ORDER, s.values, marker="o", label=p)
    plt.title("Dipper: NER Recall vs Paraphrasing Round", fontsize=TITLE_FS)
    plt.xlabel("Paraphrasing Round", fontsize=LABEL_FS)
    plt.ylabel("Mean Recall", fontsize=LABEL_FS)
    plt.legend(title="Dipper Intensity", fontsize=LEGEND_FS, title_fontsize=LEGEND_FS + 1)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"ner_dipper_recall_by_round{suffix}.png",
                dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Section 4 — Human vs. LLM across rounds
# ---------------------------------------------------------------------------

def plot_human_vs_llm(valid_abs: pd.DataFrame):
    """
    Line plots comparing NER recall and precision for Human vs. LLM
    source texts across all paraphrasers and paraphrasing rounds.
    Also prints summary tables and the LLM-minus-Human gap table.
    """
    valid_abs = valid_abs.copy()
    valid_abs["source_group"] = valid_abs["source"].apply(
        lambda x: "Human" if x == "Human" else "LLM"
    )

    # Summary tables
    rp_stats = (
        valid_abs
        .groupby(["source_group", "paraphraser", "tier"])[["recall", "precision"]]
        .agg(["mean", "std"])
        .reindex(pd.MultiIndex.from_product(
            [["Human", "LLM"], PARAPHRASER_ORDER, TIER_ORDER],
            names=["source_group", "paraphraser", "tier"]
        ))
    )
    print("=== Recall and Precision by Source Group, Paraphraser, and Tier ===")
    print(rp_stats.round(4).to_string())

    for group in ["Human", "LLM"]:
        print(f"\n=== Recall — {group} ===")
        tbl = (
            valid_abs[valid_abs["source_group"] == group]
            .groupby(["paraphraser", "tier"])["recall"].mean()
            .unstack("tier").reindex(index=PARAPHRASER_ORDER, columns=TIER_ORDER)
        )
        print(tbl.round(4).to_string())

    # Gap table: LLM minus Human
    print("\n=== Recall Gap (LLM minus Human) ===")
    human_recall = (
        valid_abs[valid_abs["source_group"] == "Human"]
        .groupby(["paraphraser", "tier"])["recall"].mean()
        .unstack("tier").reindex(index=PARAPHRASER_ORDER, columns=TIER_ORDER)
    )
    llm_recall = (
        valid_abs[valid_abs["source_group"] == "LLM"]
        .groupby(["paraphraser", "tier"])["recall"].mean()
        .unstack("tier").reindex(index=PARAPHRASER_ORDER, columns=TIER_ORDER)
    )
    gap_table = (llm_recall - human_recall).round(4)
    print(gap_table.to_string())

    # Line plots for recall and precision
    for metric, ylabel, fname_suffix in [
        ("recall",    "Mean Recall",    "recall"),
        ("precision", "Mean Precision", "precision"),
    ]:
        fig, axes = plt.subplots(LINEPLOT_NROWS, LINEPLOT_NCOLS,
                                 figsize=LINEPLOT_FIGSIZE, sharey=True)
        axes = axes.flatten()
        for i, paraphraser in enumerate(PARAPHRASER_ORDER):
            ax = axes[i]
            for group in ["Human", "LLM"]:
                subset = (
                    valid_abs[
                        (valid_abs["paraphraser"] == paraphraser) &
                        (valid_abs["source_group"] == group)
                    ].groupby("tier")[metric].mean().reindex(TIER_ORDER)
                )
                ax.plot(TIER_ORDER, subset.values, marker="o", label=group)
            ax.set_title(paraphraser, fontsize=TITLE_FS)
            ax.set_xlabel("Paraphrasing Round", fontsize=LABEL_FS)
            ax.set_ylabel(ylabel, fontsize=LABEL_FS)
            ax.tick_params(labelsize=TICK_FS)
            ax.legend(fontsize=LEGEND_FS)
        for j in range(len(PARAPHRASER_ORDER), len(axes)):
            axes[j].set_visible(False)
        plt.suptitle(
            f"NER {ylabel.replace('Mean ', '')}: Human vs LLM Source across "
            f"Paraphrasing Rounds", fontsize=TITLE_FS + 2, y=SUP_Y)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"ner_human_vs_llm_{fname_suffix}_by_round_t123{suffix}.png",
                    dpi=300, bbox_inches="tight")
        plt.show()


# ---------------------------------------------------------------------------
# Section 5 — Incremental drift table
# ---------------------------------------------------------------------------

def print_incremental_table(valid_abs: pd.DataFrame, valid_inc: pd.DataFrame):
    """
    Print the incremental survival rate table:
    T0→T1, T1→T2, T2→T3, plus two difference columns showing deceleration.
    """
    # T1→T2 and T2→T3 from incremental metrics
    inc_t1t2_t2t3 = (
        valid_inc.groupby(["paraphraser", "transition"])["recall"].mean()
        .unstack("transition")
        .reindex(index=PARAPHRASER_ORDER, columns=["T1_to_T2", "T2_to_T3"])
    )

    # T0→T1 from absolute metrics (just the T1 recall)
    t0_to_t1 = (
        valid_abs[valid_abs["tier"] == "T1"]
        .groupby("paraphraser")["recall"].mean()
        .reindex(PARAPHRASER_ORDER).rename("T0_to_T1")
    )

    inc_full = pd.concat([t0_to_t1, inc_t1t2_t2t3], axis=1)[
        ["T0_to_T1", "T1_to_T2", "T2_to_T3"]
    ]
    # Difference columns reveal whether drift decelerates each round
    inc_full["T1→T2 minus T0→T1"] = (inc_full["T1_to_T2"] - inc_full["T0_to_T1"]).round(4)
    inc_full["T2→T3 minus T1→T2"] = (inc_full["T2_to_T3"] - inc_full["T1_to_T2"]).round(4)

    print("=== Incremental Recall: Mean survival rate per transition ===")
    print("(Higher = more entities survived that transition)")
    print("(Positive difference = deceleration, negative = acceleration)")
    print(inc_full.round(4).to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    valid_abs, valid_inc = load_metrics()

    print("\n--- Section 1: Global Overview ---")
    plot_global_overview(valid_abs)

    print("\n--- Section 2: Domain Analysis ---")
    plot_domain_analysis(valid_abs)

    print("\n--- Section 3: Dipper Intensity ---")
    plot_dipper_intensity(valid_abs)

    print("\n--- Section 4: Human vs. LLM ---")
    plot_human_vs_llm(valid_abs)

    print("\n--- Section 5: Incremental Drift ---")
    print_incremental_table(valid_abs, valid_inc)
