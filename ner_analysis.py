import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_ner_analysis(ner_df):
    # Filter out rows where the original text (T0) has no entities
    valid = ner_df[ner_df["has_entities_T0"]]

    # Count and report how many rows were excluded
    excluded = len(ner_df) - len(valid)
    print(f"Excluded articles (no T0 entities): {excluded} ({excluded/len(ner_df):.2%})")

    # Define consistent ordering for paraphrasers in outputs and plots
    paraphraser_order = [
        "chatgpt",
        "palm",
        "dipper_low",
        "dipper_mid",
        "dipper_high",
        "pegasus_slight",
        "pegasus_full",
    ]

    # Compute summary statistics (mean and std) for all three metrics
    summary = (
        valid
        .groupby("paraphraser")[["jaccard", "recall", "precision"]]
        .agg(["mean", "std"])
        .reindex(paraphraser_order)
    )
    print(summary)

    # Compute global recall and precision statistics (used for main comparison plot)
    global_metrics = (
        valid
        .groupby("paraphraser")[["recall", "precision"]]
        .agg(["mean", "std"])
        .reindex(paraphraser_order)
    )
    print(global_metrics)

    # Prepare data for plotting (flatten grouped stats into rows)
    rows = []
    for paraphraser in paraphraser_order:
        for metric in ["recall", "precision"]:
            mean_val = global_metrics.loc[paraphraser, (metric, "mean")]
            std_val = global_metrics.loc[paraphraser, (metric, "std")]
            rows.append([paraphraser, metric, mean_val, std_val])

    plot_df = pd.DataFrame(rows, columns=["paraphraser", "metric", "value_mean", "value_std"])

    # Plot global recall and precision as grouped bar chart
    plt.figure(figsize=(9.5, 6.5))
    sns.barplot(
        data=plot_df,
        x="paraphraser",
        y="value_mean",
        hue="metric",
        order=paraphraser_order,
        errorbar=None,   # Disable seaborn error bars (we add manually)
        capsize=0.15
    )

    ax = plt.gca()
    num_paraphrasers = len(paraphraser_order)
    bar_width = 0.8 / 2  # Width per bar (2 metrics)

    # Manually add error bars (std deviation)
    for _, row in plot_df.iterrows():
        paraphraser_index = paraphraser_order.index(row["paraphraser"])
        metric_offset = -bar_width/2 if row["metric"] == "recall" else bar_width/2
        x_loc = paraphraser_index + metric_offset
        plt.errorbar(
            x=x_loc,
            y=row["value_mean"],
            yerr=row["value_std"],
            fmt="none",
            c="black",
            capsize=5
        )

    # Styling and saving the plot
    plt.title("Global Recall and Precision by Paraphraser", fontsize=18)
    plt.xlabel("Paraphraser", fontsize=16)
    plt.ylabel("Mean", fontsize=16)
    plt.xticks(range(num_paraphrasers), paraphraser_order, rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title="Metric", fontsize=11, title_fontsize=12)
    plt.tight_layout()
    plt.savefig("output/ner_global_recall_precision_t1.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Focus analysis on Pegasus variants only
    pg = valid[valid["paraphraser"].isin(["pegasus_slight", "pegasus_full"])]

    # Compute stats for Pegasus variants
    pg_summary = pg.groupby("paraphraser")[["jaccard", "recall", "precision"]].agg(["mean", "std"])
    print(pg_summary)

    # Prepare grouped bar plot comparing Pegasus slight vs full
    metrics = ["jaccard", "recall", "precision"]
    x = np.arange(len(metrics))
    bar_width = 0.25

    means_s = [pg_summary.loc["pegasus_slight"][(m, "mean")] for m in metrics]
    means_f = [pg_summary.loc["pegasus_full"][(m, "mean")] for m in metrics]
    std_s = [pg_summary.loc["pegasus_slight"][(m, "std")] for m in metrics]
    std_f = [pg_summary.loc["pegasus_full"][(m, "std")] for m in metrics]

    # Plot Pegasus comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - bar_width/2, means_s, width=bar_width, label="Pegasus (slight)", alpha=0.9)
    ax.errorbar(x - bar_width/2, means_s, yerr=std_s, fmt="none", ecolor="black", capsize=5, linewidth=1.2)
    ax.bar(x + bar_width/2, means_f, width=bar_width, label="Pegasus (full)", alpha=0.9)
    ax.errorbar(x + bar_width/2, means_f, yerr=std_f, fmt="none", ecolor="black", capsize=5, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(["Jaccard", "Recall", "Precision"], fontsize=12)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("Entity Stability: Pegasus(slight) vs Pegasus(full)", fontsize=16)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("output/ner_pegasus_comparison_t1.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Compare Pegasus variants across datasets
    metrics = ["jaccard", "recall", "precision"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)

    for ax, metric in zip(axes, metrics):
        sns.barplot(
            data=pg,
            x="dataset",
            y=metric,
            hue="paraphraser",
            ax=ax,
            errorbar="sd",
            capsize=0.15
        )
        ax.set_title(f"{metric.capitalize()}", fontsize=16)
        ax.set_ylabel(f"Mean {metric.capitalize()}", fontsize=14)
        ax.set_xlabel("Dataset", fontsize=14)
        ax.tick_params(axis="x", rotation=45, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.legend(title="Paraphraser", fontsize=11, title_fontsize=12, loc="lower right")

    plt.tight_layout()

    # Save individual plots per metric
    for metric in metrics:
        fig_single, ax_single = plt.subplots(figsize=(7, 5))
        sns.barplot(
            data=pg,
            x="dataset",
            y=metric,
            hue="paraphraser",
            ax=ax_single,
            errorbar="sd",
            capsize=0.15
        )
        ax_single.set_title(f"{metric.capitalize()}", fontsize=16)
        ax_single.set_ylabel(f"Mean {metric.capitalize()}", fontsize=14)
        ax_single.set_xlabel("Dataset", fontsize=14)
        ax_single.tick_params(axis="x", rotation=45, labelsize=12)
        ax_single.tick_params(axis="y", labelsize=12)
        ax_single.legend(title="Paraphraser", fontsize=11, title_fontsize=12, loc="lower right")
        plt.tight_layout()
        plt.savefig(f"output/ner_by_dataset_peg_{metric}_t1.png", dpi=300, bbox_inches="tight")
        plt.close(fig_single)

    plt.show()

    # Analyze Dipper variants (low → high intensity)
    dipper_only = ner_df[ner_df["paraphraser"].isin(["dipper_low", "dipper_mid", "dipper_high"])]

    stats = (
        dipper_only
        .groupby("paraphraser")[["recall", "precision"]]
        .agg(["mean", "std"])
        .reindex(["dipper_low", "dipper_mid", "dipper_high"])
    )

    paraphrasers = ["dipper_low", "dipper_mid", "dipper_high"]
    metrics = ["recall", "precision"]
    x = np.arange(len(paraphrasers))
    bar_width = 0.25

    plt.figure(figsize=(8, 5))

    # Plot recall and precision vs dipper intensity
    for i, metric in enumerate(metrics):
        means = stats[(metric, "mean")].values
        stds = stats[(metric, "std")].values
        offset = (i - 0.5) * bar_width

        plt.bar(x + offset, means, width=bar_width, label=metric.capitalize(), alpha=0.9)
        plt.errorbar(x + offset, means, yerr=stds, fmt="none", ecolor="black", capsize=5, linewidth=1.2)

    plt.xticks(x, ["dipper_low", "dipper_mid", "dipper_high"], fontsize=12)
    plt.ylabel("Score (Higher = Better Entity Preservation)", fontsize=14)
    plt.xlabel("Dipper Intensity", fontsize=14)
    plt.title("NER Recall and Precision vs Dipper Intensity", fontsize=16)
    plt.ylim(0, 1)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("output/ner_dipper_recall_precision_t1.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Compare Human vs LLM source groups
    rp_stats = (
        valid
        .groupby(["paraphraser", "source_group"])[["recall", "precision"]]
        .agg(["mean", "std"])
        .reindex(paraphraser_order, level=0)
    )

    # Plot recall comparison (Human vs LLM)
    fig_r, ax_r = plt.subplots(figsize=(9, 6))
    bar_width = 0.35
    x = np.arange(len(paraphraser_order))

    means_h = rp_stats[rp_stats.index.get_level_values("source_group") == "Human"][("recall", "mean")].values
    means_l = rp_stats[rp_stats.index.get_level_values("source_group") == "LLM"][("recall", "mean")].values
    std_h = rp_stats[rp_stats.index.get_level_values("source_group") == "Human"][("recall", "std")].values
    std_l = rp_stats[rp_stats.index.get_level_values("source_group") == "LLM"][("recall", "std")].values

    ax_r.bar(x - bar_width/2, means_h, width=bar_width, label="Human", alpha=0.85)
    ax_r.errorbar(x - bar_width/2, means_h, yerr=std_h, fmt="none", ecolor="black", capsize=4)
    ax_r.bar(x + bar_width/2, means_l, width=bar_width, label="LLM", alpha=0.85)
    ax_r.errorbar(x + bar_width/2, means_l, yerr=std_l, fmt="none", ecolor="black", capsize=4)

    ax_r.set_title("Recall", fontsize=16)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(paraphraser_order, rotation=45, ha="right", fontsize=12)
    ax_r.set_ylim(0, 1.1)
    ax_r.set_yticks(np.arange(0, 1.1, 0.2))
    ax_r.set_ylabel("Score", fontsize=14)
    ax_r.legend(title="Source Group", fontsize=11, title_fontsize=12)

    plt.tight_layout()
    plt.savefig("output/ner_human_vs_llm_recall_t1.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot precision comparison (Human vs LLM)
    fig_p, ax_p = plt.subplots(figsize=(9, 6))

    means_h = rp_stats[rp_stats.index.get_level_values("source_group") == "Human"][("precision", "mean")].values
    means_l = rp_stats[rp_stats.index.get_level_values("source_group") == "LLM"][("precision", "mean")].values
    std_h = rp_stats[rp_stats.index.get_level_values("source_group") == "Human"][("precision", "std")].values
    std_l = rp_stats[rp_stats.index.get_level_values("source_group") == "LLM"][("precision", "std")].values

    ax_p.bar(x - bar_width/2, means_h, width=bar_width, label="Human", alpha=0.85)
    ax_p.errorbar(x - bar_width/2, means_h, yerr=std_h, fmt="none", ecolor="black", capsize=4)
    ax_p.bar(x + bar_width/2, means_l, width=bar_width, label="LLM", alpha=0.85)
    ax_p.errorbar(x + bar_width/2, means_l, yerr=std_l, fmt="none", ecolor="black", capsize=4)

    ax_p.set_title("Precision", fontsize=16)
    ax_p.set_xticks(x)
    ax_p.set_xticklabels(paraphraser_order, rotation=45, ha="right", fontsize=12)
    ax_p.set_ylim(0, 1.1)
    ax_p.set_yticks(np.arange(0, 1.1, 0.2))
    ax_p.set_ylabel("Score", fontsize=14)
    ax_p.legend(title="Source Group", fontsize=11, title_fontsize=12)

    plt.tight_layout()
    plt.savefig("output/ner_human_vs_llm_precision_t1.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Import preprocessing and dataset preparation functions
    from ner_processing import load_or_compute_ner_metrics, load_or_compute_ner_sets
    from dataset_prep import load_or_build_dataset

    # Build dataset and compute NER metrics
    paired = load_or_build_dataset()
    paired_ner = load_or_compute_ner_sets(paired)
    ner_df = load_or_compute_ner_metrics(paired_ner)

    # Run full analysis pipeline
    run_ner_analysis(ner_df)