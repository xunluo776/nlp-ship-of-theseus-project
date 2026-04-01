import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

datasets = ["cmv", "eli5", "sci_gen", "tldr", "wp", "xsum", "yelp"]
paraphrasers = [
    "text_chatgpt",
    "text_palm",
    "text_dipper_low",
    "text_dipper",
    "text_dipper_high",
    "text_pegasus_slight",
    "text_pegasus_full",
]

cache_dir = "output"

# ---------------------------------------------------------
# LOAD AND AGGREGATE GLOBAL MEAN + STD F1
# ---------------------------------------------------------

global_mean_f1 = {}
global_std_f1 = {}

for p in paraphrasers:
    all_f1 = []

    for ds in datasets:
        path = os.path.join(cache_dir, f"bertscore_{ds}_cache.pkl")
        with open(path, "rb") as f:
            bert_results = pickle.load(f)

        f1_vals = [triple[2] for triple in bert_results[p] if triple[2] is not None]
        all_f1.extend(f1_vals)

    global_mean_f1[p] = np.mean(all_f1)
    global_std_f1[p] = np.std(all_f1)

# ---------------------------------------------------------
# BAR PLOT WITH VALUE LABELS
# ---------------------------------------------------------

labels = [p.replace("text_", "") for p in paraphrasers]
means = [global_mean_f1[p] for p in paraphrasers]
stds  = [global_std_f1[p] for p in paraphrasers]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    labels,
    means,
    yerr=stds,
    capsize=4,
    width=0.6,
    color="#1e81b0"
)

plt.ylabel("Global Mean BERTScore F1")
plt.title("Global Mean BERTScore F1 per Paraphraser (Across 7 Datasets)")
plt.xticks(rotation=30)
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.ylim(min(means) - max(stds)*1.2, max(means) + max(stds)*1.2)

for bar, value in zip(bars, means):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() * 1,
        height + max(stds) * 0.1,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.tight_layout()
plt.savefig("output/global_bertscore_f1.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------
# SECOND BLOCK — BOXPLOTS PER DATASET
# ---------------------------------------------------------

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

datasets = ["cmv", "eli5", "sci_gen", "tldr", "wp", "xsum", "yelp"]
paraphrasers = [
    "text_chatgpt",
    "text_palm",
    "text_dipper_low",
    "text_dipper",
    "text_dipper_high",
    "text_pegasus_slight",
    "text_pegasus_full",
]

cache_dir = "output"

all_f1 = {}

for ds in datasets:
    path = os.path.join(cache_dir, f"bertscore_{ds}_cache.pkl")
    print(f"Loading {path} ...")

    with open(path, "rb") as f:
        bert_results = pickle.load(f)

    f1_dict = {}
    for p in paraphrasers:
        f1_dict[p] = [triple[2] for triple in bert_results[p] if triple[2] is not None]

    all_f1[ds] = f1_dict

fig, axes = plt.subplots(2, 4, figsize=(18, 10), sharey=True)
axes = axes.flatten()

for i, ds in enumerate(datasets):
    ax = axes[i]
    f1_lists = [all_f1[ds][p] for p in paraphrasers]

    ax.boxplot(
        f1_lists,
        labels=[p.replace("text_", "") for p in paraphrasers],
        showfliers=False
    )
    ax.set_title(ds.upper())
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

axes[-1].axis("off")

axes[0].set_ylabel("BERTScore F1")
plt.suptitle("BERTScore F1 Across Paraphrasers for 7 Datasets", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig("output/bertscore_f1_all_datasets_t1.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------
# THIRD BLOCK — GROUPED BAR PLOT BY DATASET
# ---------------------------------------------------------

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

datasets = ["cmv", "eli5", "sci_gen", "tldr", "wp", "xsum", "yelp"]
paraphrasers = [
    "text_chatgpt",
    "text_palm",
    "text_dipper_low",
    "text_dipper",
    "text_dipper_high",
    "text_pegasus_slight",
    "text_pegasus_full",
]

cache_dir = "output"

mean_f1 = {ds: [] for ds in datasets}
std_f1  = {ds: [] for ds in datasets}

for ds in datasets:
    path = os.path.join(cache_dir, f"bertscore_{ds}_cache.pkl")
    with open(path, "rb") as f:
        bert_results = pickle.load(f)

    for p in paraphrasers:
        f1_vals = [triple[2] for triple in bert_results[p] if triple[2] is not None]
        mean_f1[ds].append(np.mean(f1_vals))
        std_f1[ds].append(np.std(f1_vals))

plt.figure(figsize=(14, 7))

x = np.arange(len(paraphrasers))
labels = [p.replace("text_", "") for p in paraphrasers]

num_datasets = len(datasets)
bar_width = 0.11

soft_colors = [
    "#7FC8C9",
    "#9ED9A0",
    "#F5C16C",
    "#F2E88B",
    "#A7D3E8",
    "#FF9AA2",
    "#9ED7D5",
]

for i, ds in enumerate(datasets):
    plt.bar(
        x + i * bar_width,
        mean_f1[ds],
        yerr=std_f1[ds],
        width=bar_width,
        capsize=3,
        color=soft_colors[i],
        label=ds.upper()
    )

plt.xticks(x + bar_width * (num_datasets - 1) / 2, labels, rotation=30)
plt.ylabel("Mean BERTScore F1")
plt.title("Mean BERTScore F1 Across Paraphrasers")
plt.ylim(0.8, 1.025)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

os.makedirs("output", exist_ok=True)
plt.savefig("output/mean_f1_by_paraphraser_grouped.png", dpi=300, bbox_inches="tight")

plt.show()


# ---------------------------------------------------------
# UNIFIED BERTSCORE CACHE + HUMAN vs LLM-SOURCED DRIFT (F1 ONLY)
# ---------------------------------------------------------

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

ALL_DATASETS = ["cmv", "eli5", "sci_gen", "tldr", "wp", "xsum", "yelp"]

paraphraser_cols = [
    ("chatgpt",        "text_chatgpt"),
    ("palm",           "text_palm"),
    ("dipper_low",     "text_dipper_low"),
    ("dipper_mid",     "text_dipper"),
    ("dipper_high",    "text_dipper_high"),
    ("pegasus_slight", "text_pegasus_slight"),
    ("pegasus_full",   "text_pegasus_full"),
]

paraphraser_order = [
    "chatgpt",
    "palm",
    "dipper_low",
    "dipper_mid",
    "dipper_high",
    "pegasus_slight",
    "pegasus_full",
]

# ---------------------------------------------------------
# LOAD ALL BERTSCORE CACHES + MERGE INTO ONE TABLE
# ---------------------------------------------------------

rows = []

for ds in ALL_DATASETS:
    cache_path = Path(f"output/bertscore_{ds}_cache.pkl")
    print(f"Loading {cache_path} ...")

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    subset = paired_t1[paired_t1["dataset"] == ds].reset_index(drop=True)
    n_rows = len(subset)

    for p_name, p_col in paraphraser_cols:
        score_list = cache[p_col]

        usable_len = min(len(score_list), n_rows)

        for i in range(usable_len):
            P, R, F1 = score_list[i]
            if F1 is None:
                continue

            row = subset.iloc[i]

            rows.append({
                "dataset": ds,
                "key": row["key"],
                "source": row["source"],
                "paraphraser": p_name,
                "bert_f1": F1,
            })

bert_df = pd.DataFrame(rows)
print("Unified BERTScore table shape:", bert_df.shape)

# ---------------------------------------------------------
# SAVE UNIFIED CACHE
# ---------------------------------------------------------

bert_df.to_pickle("output/bertscore_all_datasets.pkl")
print("Saved unified BERTScore cache → output/bertscore_all_datasets.pkl")

# ---------------------------------------------------------
# HUMAN vs LLM SOURCE GROUPING
# ---------------------------------------------------------

bert_df["source_group"] = bert_df["source"].apply(
    lambda s: "Human" if s == "Human" else "LLM"
)

bert_df["paraphraser"] = pd.Categorical(
    bert_df["paraphraser"],
    categories=paraphraser_order,
    ordered=True
)

# ---------------------------------------------------------
# COMPUTE MEAN + STD FOR HUMAN vs LLM (F1 ONLY)
# ---------------------------------------------------------

stats = (
    bert_df
    .groupby(["source_group", "paraphraser"])["bert_f1"]
    .agg(["mean", "std"])
    .reset_index()
    .sort_values(["paraphraser", "source_group"])
)

# ---------------------------------------------------------
# PLOT — BERTScore F1 (Human vs LLM)
# ---------------------------------------------------------

x = np.arange(len(paraphraser_order))
bar_width = 0.35

means_h = stats[stats["source_group"]=="Human"]["mean"].values
means_l = stats[stats["source_group"]=="LLM"]["mean"].values
std_h   = stats[stats["source_group"]=="Human"]["std"].values
std_l   = stats[stats["source_group"]=="LLM"]["std"].values

fig, ax = plt.subplots(figsize=(9,6))

ax.bar(x - bar_width/2, means_h, width=bar_width, label="Human", alpha=0.85)
ax.errorbar(x - bar_width/2, means_h, yerr=std_h, fmt="none", ecolor="black", capsize=4)

ax.bar(x + bar_width/2, means_l, width=bar_width, label="LLM", alpha=0.85)
ax.errorbar(x + bar_width/2, means_l, yerr=std_l, fmt="none", ecolor="black", capsize=4)

ax.set_title("BERTScore F1 — Human vs LLM Sources", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(paraphraser_order, rotation=45, ha="right", fontsize=12)

ax.set_ylim(0.8, 1.02)
ax.set_yticks(np.arange(0.775, 1.101, 0.05))

ax.set_ylabel("BERTScore F1", fontsize=14)
ax.legend(title="Source Group", fontsize=11, title_fontsize=12)

plt.tight_layout()
plt.savefig("output/bertscore_f1_human_vs_llm.png", dpi=300, bbox_inches="tight")
plt.show()
