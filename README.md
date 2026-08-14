# OpenML Dataset Recommender — End-to-End Data Science Project

A content-based **recommender system that suggests similar machine-learning
datasets** from a catalogue of **6,408 OpenML datasets**. Given a dataset (or a
free-text description), it returns the most similar datasets ranked by cosine
similarity over learned vector embeddings.

The project is a complete, reproducible data-science workflow covering the
mandatory work packages plus several optional ones.

---

## Work packages covered

| Work package | Where | Notes |
|---|---|---|
| **Data Quality*** | [src/data_quality.py](src/data_quality.py) | 6 DQ dimensions (completeness, uniqueness, validity, consistency, conformity, timeliness), composite score, remediation. |
| **Vector Embeddings*** | [src/embeddings.py](src/embeddings.py) | TF-IDF, TF-IDF+SVD (LSA), and Sentence-Transformer (`all-MiniLM-L6-v2`) embeddings. |
| **Recommender System*** | [src/recommender.py](src/recommender.py) | Content-based item-item cosine k-NN, with by-index / by-vector / free-text query modes. |
| **Performance Evaluation*** | [src/evaluation.py](src/evaluation.py) | Precision@k, Recall@k, MAP@k, NDCG@k against **two** relevance ground-truths. |
| **Experiments Logging*** | [run_pipeline.py](run_pipeline.py), [src/tuning.py](src/tuning.py) | Weights & Biases logging of configs, metrics, hyperparameters. |
| **Hyperparameter Tuning** | [src/tuning.py](src/tuning.py) | Optuna (TPE) search over the TF-IDF embedding. |
| **Perturbation Analysis** | [src/perturbation.py](src/perturbation.py) | Robustness to typos, word-dropout, truncation, shuffle. |
| **Frontend Application** | [app/streamlit_app.py](app/streamlit_app.py) | Interactive Streamlit app. |

`*` = mandatory work package.

> **On Data Scraping:** the provided `openml_raw_metadata.json` is the raw
> metadata harvested from the OpenML REST API (`/data/{id}` + qualities/features
> endpoints) — i.e. it was *scraped*, not downloaded as a tidy CSV. This project
> focuses its effort on data quality, embeddings, modelling, evaluation and the
> application rather than re-scraping.

---

## The recommendation problem & why it is sound

The metadata has **no explicit user–item interactions**, so we build a
**content-based** recommender: each dataset is embedded from its
`name + description + creator + target attribute`, and recommendations are the
nearest neighbours in embedding space.

**Evaluation without leakage.** Relevance ground truth is derived from fields
that are *never shown to the model*:

1. **Topical-tag relevance** — two datasets are relevant if they share a
   *specific* topical tag (umbrella tags appearing in >5% of the catalogue, e.g.
   "Data Science", are filtered out).
2. **Benchmark-study relevance** — two datasets are relevant if they belong to
   the same curated OpenML *study* (`study_*` tag).

These two signals differ in nature (subject-matter vs curatorial), giving us
**two independent evaluation methods**. Combined with two metric families
(set-overlap P@k/R@k and rank-aware MAP/NDCG), the evaluation is thorough.

---

## Results (representative)

| Model | Topical NDCG@10 | Study NDCG@10 |
|---|---|---|
| Random baseline (P@10) | ~0.04 | ~0.17 |
| TF-IDF | 0.50 | 0.84 |
| TF-IDF + SVD | 0.51 | 0.85 |
| TF-IDF (Optuna-tuned) | 0.52 | 0.85 |
| Semantic (MiniLM) | see `outputs/metrics/pipeline_results.json` | |

The recommender is **~12× better than random** on the specific topical signal and
sanity-checks correctly (querying *iris* returns iris variants). Exact numbers
for the latest run are written to `outputs/metrics/pipeline_results.json`.

---

## Project layout

```
.
├── data/
│   ├── raw/openml_raw_metadata.json     # scraped OpenML metadata (input)
│   └── processed/datasets_clean.parquet # cleaned, modelling-ready table
├── src/
│   ├── config.py          # paths & hyper-parameters
│   ├── data_loading.py    # JSON -> tidy DataFrame, text cleaning
│   ├── data_quality.py    # DQ metrics + remediation
│   ├── relevance.py       # held-out ground-truth signals
│   ├── embeddings.py      # TF-IDF / SVD / semantic embeddings
│   ├── recommender.py     # content-based cosine k-NN engine
│   ├── evaluation.py      # P@k / R@k / MAP / NDCG
│   ├── tuning.py          # Optuna hyperparameter search
│   ├── perturbation.py    # robustness analysis
│   └── plots.py           # figure generation
├── app/streamlit_app.py   # interactive frontend
├── run_pipeline.py        # end-to-end orchestrator (+ W&B logging)
├── outputs/               # metrics JSON, figures, cached embeddings
└── requirements.txt
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Full pipeline: DQ -> embeddings -> tuning -> eval -> perturbation -> figures
python run_pipeline.py                 # logs to W&B in offline mode by default

# Faster iteration (skip transformer + perturbation):
python run_pipeline.py --fast

# Sync experiments to the W&B cloud instead of offline:
WANDB_MODE=online python run_pipeline.py

# Launch the interactive app:
streamlit run app/streamlit_app.py
```

### Useful flags
`--no-semantic` skip the transformer embedding · `--no-tuning` skip Optuna ·
`--no-perturbation` skip robustness · `--trials N` Optuna trials ·
`--no-wandb` disable logging.

---

## Weights & Biases

Every run logs the data-quality score, per-model evaluation metrics, every
Optuna trial's hyperparameters/score, and perturbation retention curves to the
`openml-dataset-recommender` W&B project. It defaults to **offline** mode (no
login needed); set `WANDB_MODE=online` and `wandb login` to sync to the cloud.

---

## Reproducibility

All randomness is seeded (`config.RANDOM_STATE = 42`). The cleaned dataset,
cached embeddings, metrics and figures are written under `data/processed/` and
`outputs/`, so the Streamlit app and report can be regenerated without re-running
the heavy steps.
