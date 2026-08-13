
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})


def _models(comparison: dict):
    return [m for m in comparison if not m.startswith("_")]


def model_comparison_bar(comparison: dict, signal: str, metric: str = "ndcg@10"):
    models = _models(comparison)
    vals = [comparison[m][signal].get(metric, 0) for m in models]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(models, vals, color="#4C72B0")
    ax.set_title(f"{metric} on {signal} relevance")
    ax.set_ylabel(metric)
    ax.set_ylim(0, max(vals) * 1.2 if vals else 1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    path = config.FIGURE_DIR / f"compare_{signal}_{metric.replace('@', '')}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def precision_recall_curve(comparison: dict, signal: str, k_values=(5, 10, 20)):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m in _models(comparison):
        p = [comparison[m][signal][f"precision@{k}"] for k in k_values]
        r = [comparison[m][signal][f"recall@{k}"] for k in k_values]
        axes[0].plot(k_values, p, marker="o", label=m)
        axes[1].plot(k_values, r, marker="o", label=m)
    axes[0].set(title=f"Precision@k ({signal})", xlabel="k", ylabel="precision")
    axes[1].set(title=f"Recall@k ({signal})", xlabel="k", ylabel="recall")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    path = config.FIGURE_DIR / f"pr_curve_{signal}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def perturbation_plot(pert: dict, k: int = 10):
    kinds = list(pert["perturbations"].keys())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind in kinds:
        levels = sorted(pert["perturbations"][kind].keys(), key=float)
        xs = [float(l) for l in levels]
        ys = [pert["perturbations"][kind][l]["precision_retention"] for l in levels]
        ax.plot(xs, ys, marker="o", label=kind)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.6)
    ax.set(title="Recommender robustness under input perturbation",
           xlabel="perturbation level", ylabel=f"precision@{k} retention")
    ax.legend()
    ax.grid(alpha=0.3)
    path = config.FIGURE_DIR / "perturbation_robustness.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def data_quality_plot(report: dict):
    comp = report["completeness"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fields = list(comp.keys())
    vals = [comp[f] for f in fields]
    colors = ["#55A868" if v >= 0.9 else "#DD8452" if v >= 0.5 else "#C44E52"
              for v in vals]
    ax.barh(fields, vals, color=colors)
    ax.set(title="Field completeness", xlabel="non-missing fraction", xlim=(0, 1))
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    path = config.FIGURE_DIR / "data_quality_completeness.png"
    fig.savefig(path)
    plt.close(fig)
    return path
