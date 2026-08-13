"""End-to-end pipeline orchestrator for the OpenML dataset recommender.

Runs the full data-science workflow and logs everything to Weights & Biases:

  1. Data Quality   - assess + clean the raw metadata
  2. Embeddings     - TF-IDF, TF-IDF+SVD, semantic (Sentence-Transformers)
  3. Recommender    - content-based item-item cosine kNN
  4. Evaluation     - Precision@k / Recall@k / MAP / NDCG vs two relevance signals
  5. Tuning         - Optuna search over TF-IDF hyperparameters
  6. Perturbation   - robustness of the best model to noisy input
  7. Reporting      - metrics JSON + figures in outputs/

Usage:
    python run_pipeline.py                 # full run (semantic + tuning + perturbation)
    python run_pipeline.py --fast          # skip semantic + perturbation, few trials
    python run_pipeline.py --no-semantic   # skip the transformer embedding
    python run_pipeline.py --trials 50     # Optuna trials
    WANDB_MODE=online python run_pipeline.py   # sync to W&B cloud
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from src import (
    config, data_quality, embeddings, evaluation, perturbation, plots,
    relevance, tuning,
)
from src.recommender import ContentRecommender

EVAL_MAX_QUERIES = 1500  # cap eval query set for speed; set None for all


def _init_wandb(args):
    import wandb
    return wandb.init(
        project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY,
        name=f"pipeline-{int(time.time())}", job_type="pipeline",
        config=vars(args), reinit=True,
    )


def build_embedding(kind: str, docs, tuned_params=None):
    if kind == "tfidf":
        return embeddings.build_tfidf(docs, {"svd_components": 0})
    if kind == "tfidf_svd":
        return embeddings.build_tfidf(docs, None)  # defaults (SVD 300)
    if kind == "tfidf_tuned":
        return embeddings.build_tfidf(docs, tuned_params)
    if kind == "semantic":
        return embeddings.build_semantic(docs)
    raise ValueError(kind)


def main(args):
    t0 = time.time()
    run = _init_wandb(args) if args.wandb else None

    # ---- 1. Data Quality --------------------------------------------------
    print("[1/6] Data quality + cleaning ...")
    clean_df, dq_report = data_quality.run(save=True)
    plots.data_quality_plot(dq_report)
    print(f"      composite DQ score = {dq_report['composite_score']}  "
          f"({len(clean_df)} clean rows)")
    if run:
        run.log({"dq/composite_score": dq_report["composite_score"],
                 **{f"dq/completeness/{k}": v
                    for k, v in dq_report["completeness"].items()}})

    docs = clean_df["document"].tolist()
    ids = clean_df["dataset_id"].values
    names = clean_df["name"].values

    # ---- 2. Relevance signals --------------------------------------------
    signals = {m: relevance.build_relevance(clean_df, m) for m in ("topical", "study")}
    for name, rel in signals.items():
        print(f"      relevance[{name}]: {relevance.relevance_stats(rel)}")

    # ---- 3. Optuna tuning (before final eval, to add a tuned model) -------
    tuned_params = None
    if not args.no_tuning:
        print(f"[2/6] Optuna tuning ({args.trials} trials) ...")
        study, best = tuning.run_tuning(clean_df, n_trials=args.trials,
                                        wandb_run=run)
        tuned_params = best["best_params"]
        print(f"      best val NDCG@10 = {best['best_value']:.4f}")
        print(f"      best params = {tuned_params}")

    # ---- 4. Build + evaluate all embedding models ------------------------
    model_kinds = ["tfidf", "tfidf_svd"]
    if tuned_params is not None:
        model_kinds.append("tfidf_tuned")
    if not args.no_semantic:
        model_kinds.append("semantic")

    comparison = {}
    fitted_store = {}
    print(f"[3/6] Building + evaluating models: {model_kinds}")
    for kind in model_kinds:
        emb, fitted = build_embedding(kind, docs, tuned_params)
        embeddings.save_embedding(emb)
        fitted_store[kind] = (emb, fitted)
        rec = ContentRecommender(emb.vectors, ids=ids, names=names)
        res = evaluation.evaluate_all_signals(
            rec, signals, k_values=tuple(config.EVAL_K_VALUES),
            max_queries=EVAL_MAX_QUERIES)
        comparison[kind] = res
        print(f"      {kind:14s} dim={emb.dim:5d}  "
              f"topical NDCG@10={res['topical']['ndcg@10']:.3f}  "
              f"study NDCG@10={res['study']['ndcg@10']:.3f}")
        if run:
            for sig, m in res.items():
                run.log({f"eval/{kind}/{sig}/{k}": v for k, v in m.items()
                         if isinstance(v, (int, float))})

    # random baseline lower bound
    comparison["_random_baseline"] = {
        sig: evaluation.random_baseline(rel, len(clean_df),
                                        k_values=tuple(config.EVAL_K_VALUES),
                                        max_queries=EVAL_MAX_QUERIES)
        for sig, rel in signals.items()
    }

    # ---- 5. Perturbation analysis on the best model ----------------------
    # pick best model by mean NDCG@10 across signals
    rankable = {k: v for k, v in comparison.items() if not k.startswith("_")}
    best_model = max(rankable,
                     key=lambda k: (comparison[k]["topical"]["ndcg@10"]
                                    + comparison[k]["study"]["ndcg@10"]) / 2)
    print(f"[4/6] Best model = {best_model}")

    pert = None
    if not args.no_perturbation:
        print("[5/6] Perturbation / robustness analysis ...")
        emb, fitted = fitted_store[best_model]
        rec = ContentRecommender(emb.vectors, ids=ids, names=names)
        embed_fn = (embeddings.transform_semantic if best_model == "semantic"
                    else embeddings.transform_tfidf)
        pert = perturbation.run_perturbation_study(
            clean_df, rec, fitted, embed_fn, signals["topical"],
            n_queries=args.pert_queries)
        plots.perturbation_plot(pert)
        print(f"      clean P@10={pert['clean_precision@10']}  "
              f"results saved")
        if run:
            for kind, levels in pert["perturbations"].items():
                for lvl, m in levels.items():
                    run.log({f"perturb/{kind}/{lvl}/retention":
                             m["precision_retention"]})

    # ---- 6. Reporting -----------------------------------------------------
    print("[6/6] Generating figures + saving metrics ...")
    for sig in signals:
        plots.precision_recall_curve(comparison, sig,
                                     k_values=tuple(config.EVAL_K_VALUES))
        plots.model_comparison_bar(comparison, sig, "ndcg@10")

    summary = {
        "best_model": best_model,
        "data_quality": {"composite_score": dq_report["composite_score"],
                         "dimension_scores": dq_report["dimension_scores"],
                         "remediation": dq_report["remediation"]},
        "relevance_stats": {m: relevance.relevance_stats(r)
                            for m, r in signals.items()},
        "tuned_params": tuned_params,
        "comparison": comparison,
        "perturbation": pert,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(config.METRIC_DIR / "pipeline_results.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    if run:
        run.summary.update({"best_model": best_model,
                            "runtime_seconds": summary["runtime_seconds"]})
        run.finish()

    print(f"\nDone in {summary['runtime_seconds']}s. "
          f"Results -> {config.METRIC_DIR}/pipeline_results.json")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="quick run: no semantic, no perturbation, few trials")
    ap.add_argument("--no-semantic", action="store_true")
    ap.add_argument("--no-tuning", action="store_true")
    ap.add_argument("--no-perturbation", action="store_true")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--pert-queries", type=int, default=300)
    ap.add_argument("--wandb", action="store_true", default=True)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    args = ap.parse_args()
    if args.fast:
        args.no_semantic = True
        args.no_perturbation = True
        args.trials = min(args.trials, 8)
    main(args)
