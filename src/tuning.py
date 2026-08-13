
from __future__ import annotations

import json

import numpy as np
import optuna

from . import config, embeddings, evaluation, relevance
from .recommender import ContentRecommender


def _objective_value(vectors, df, val_queries, signals, k=10):
    rec = ContentRecommender(vectors, ids=df["dataset_id"].values)
    scores = []
    for name, rel in signals.items():
        sub = {q: rel[q] for q in val_queries if q in rel}
        if not sub:
            continue
        res = evaluation.evaluate(rec, sub, k_values=(k,), max_queries=None)
        scores.append(res[f"ndcg@{k}"])
    return float(np.mean(scores)) if scores else 0.0


def make_objective(df, signals, val_queries, wandb_run=None):
    docs = df["document"].tolist()

    def objective(trial: optuna.Trial) -> float:
        ngram_choice = trial.suggest_categorical("ngram_range", ["1-1", "1-2"])
        params = dict(
            max_features=trial.suggest_categorical("max_features", [5000, 10000, 20000, 40000]),
            ngram_range={"1-1": (1, 1), "1-2": (1, 2)}[ngram_choice],
            min_df=trial.suggest_int("min_df", 1, 5),
            max_df=trial.suggest_float("max_df", 0.7, 1.0),
            sublinear_tf=trial.suggest_categorical("sublinear_tf", [True, False]),
            svd_components=trial.suggest_categorical("svd_components", [0, 100, 200, 300, 500]),
        )
        emb, _ = embeddings.build_tfidf(docs, params)
        value = _objective_value(emb.vectors, df, val_queries, signals)
        if wandb_run is not None:
            wandb_run.log({"trial": trial.number, "val_ndcg@10": value,
                           **{f"param/{k}": str(v) for k, v in params.items()}})
        return value

    return objective


def split_queries(signals, val_frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    all_q = sorted(set().union(*[set(s.keys()) for s in signals.values()]))
    rng.shuffle(all_q)
    n_val = int(len(all_q) * val_frac)
    val = set(all_q[:n_val])
    train = set(all_q[n_val:])
    return train, val


def run_tuning(df, n_trials=30, use_wandb=False, val_frac=0.3, wandb_run=None):
    signals = {m: relevance.build_relevance(df, m) for m in ("topical", "study")}
    _, val = split_queries(signals, val_frac=val_frac)
    val_queries = list(val)

    own_run = False
    if wandb_run is None and use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY,
            name="optuna-tfidf-tuning", job_type="tuning",
            config={"n_trials": n_trials, "val_frac": val_frac}, reinit=True,
        )
        own_run = True

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_objective(df, signals, val_queries, wandb_run),
                   n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    if isinstance(best_params.get("ngram_range"), str):
        best_params["ngram_range"] = {"1-1": (1, 1), "1-2": (1, 2)}[best_params["ngram_range"]]
    best = {"best_value": study.best_value, "best_params": best_params,
            "n_trials": n_trials}
    if wandb_run is not None:
        wandb_run.summary.update({"tuning/best_val_ndcg@10": study.best_value})
        wandb_run.summary.update({f"tuning/best/{k}": str(v)
                                  for k, v in study.best_params.items()})
        if own_run:
            wandb_run.finish()

    with open(config.METRIC_DIR / "tuning_best.json", "w") as fh:
        json.dump(best, fh, indent=2, default=str)
    return study, best
