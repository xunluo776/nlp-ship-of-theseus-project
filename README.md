# Ship of Theseus: How Much of a Text Survives Repeated Paraphrasing?

If a text is paraphrased again and again, at what point is it no longer the same text? This project measures how much **meaning** and how many **named entities** survive when texts are paraphrased by 7 different paraphrasers over up to three successive rounds (T0 → T1 → T2 → T3).

## Data

- **Sources:** paraphrased versions of texts from 7 datasets: CMV, ELI5, SciGen, TLDR, WritingPrompts, XSum and Yelp. They are loaded from `data/{dataset}_paraphrased.csv`, which is not included in this repo.
- **Paraphrasers:** ChatGPT, PaLM, DIPPER (low, mid and high diversity) and Pegasus (slight and full).
- **Versions:** each original text (T0) is aligned with all of its paraphrases into one paired table: 22 text versions per row.

## Metrics

| Metric | How it is computed |
|---|---|
| **Entity drift** | Named-entity sets from spaCy `en_core_web_sm`. Jaccard, recall and precision are measured against the original (absolute drift) and round to round, T1 → T2 → T3 (incremental drift). |
| **Semantic similarity** | BERTScore F1 from `roberta-base` embeddings. Long texts are split into 512-token chunks with 256-token overlap, and scoring is batched on the GPU (Apple MPS when available). |

## Pipeline

The latest version is in [`update3/`](update3) and covers all three rounds:

| Step | File | What it does |
|---|---|---|
| 1 | `01_data_processing.py` | Load the 7 CSVs, normalize version names, pivot into the paired table and cache it |
| 2 | `02_ner_drift.py` | Extract entity sets and compute absolute and incremental drift metrics |
| 3 | `03_ner_analysis.py` | Entity-drift plots and tables: global overview, by dataset and round, DIPPER intensity, human vs. LLM source texts, and round-to-round survival rates |
| 4 | `04_bertscore_computation.py` | BERTScore precision, recall and F1 of every paraphrase against its original (T0), for all paraphrasers and rounds. Embeddings are stored in parts and only two are loaded at once to stay within RAM. |
| 5 | `05_bertscore_analysis.py` | BERTScore plots and tables: global overview, human vs. LLM gap, by dataset, NER vs. BERTScore side by side, and authorship similarity of the original texts |

`update3/update3.ipynb` runs the full analysis with its outputs. The scripts in the repo root (`dataset_prep.py`, `ner_processing.py`, `ner_analysis.py`, `embedding_pipeline.py`, `bertscore_pipeline.py` and `bertscore_analysis.py`) are the earlier, first-round (T1-only) version.

Expensive steps (NER, embeddings, BERTScore) cache their results as pickle files, so later runs reuse them.

## Tech stack

Python · pandas · NumPy · spaCy · PyTorch · Hugging Face Transformers · scikit-learn · Matplotlib · seaborn
