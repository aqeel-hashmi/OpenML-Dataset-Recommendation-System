from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "openml_raw_metadata.json"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_DATA_PATH = PROCESSED_DIR / "datasets_clean.parquet"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "models"

for _d in (PROCESSED_DIR, FIGURE_DIR, METRIC_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TEXT_FIELDS = ["name", "description", "creator_str", "default_target_attribute"]

SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"

# TF-IDF defaults (overridable via Optuna)
TFIDF_DEFAULTS = dict(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True,
    svd_components=300,
)

TOP_K = 10
EVAL_K_VALUES = [5, 10, 20]
RANDOM_STATE = 42


NON_TOPICAL_TAG_PREFIXES = ("study_", "mythbusting_", "uci")
NOISE_TAGS = {
    "auth_verified", "exploit_test_2", "finaltest", "real_key_", "sample",
    "test", "training", "BNG", "OpenML100", "OpenML-CC18", "OpenML_Friendly",
}

WANDB_PROJECT = "openml-dataset-recommender"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")

# WANDB in offline mode, try online last
# NOT YET FIXED
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")
