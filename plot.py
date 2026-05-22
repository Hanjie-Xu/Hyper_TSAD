"""Visualisation utilities for Hyper_TSAD.

Three plot types
----------------
1. plot_loss          – training loss curve across epochs
2. plot_pr_roc        – Precision-Recall and ROC curves on the test set
3. plot_anomaly_score – anomaly score time-series overlaid with ground-truth
                        anomaly intervals and the decision threshold
4. plot_all           – convenience wrapper that calls all three

Standalone usage (requires a saved checkpoint and processed data)
-----------------------------------------------------------------
    python plot.py \\
        --dataset SWaT \\
        --checkpoint_path checkpoints/hyper_tsad_run.pt \\
        --loss_history_path checkpoints/hyper_tsad_run_loss.json \\
        --save_dir plots
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")           # non-interactive backend; safe on headless servers
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_C_LOSS    = "#2196F3"   # blue
_C_SCORE   = "#333333"   # near-black
_C_THRESH  = "#F44336"   # red
_C_ANOMALY = "#FFCDD2"   # light-red fill for anomaly regions
_C_PR      = "#9C27B0"   # purple
_C_ROC     = "#FF9800"   # orange


# ---------------------------------------------------------------------------
# 1. Training loss curve
# ---------------------------------------------------------------------------

def plot_loss(
    loss_history: List[float],
    *,
    title: str = "Training Loss",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot epoch-level training loss.

    Args:
        loss_history: List of per-epoch average losses.
        title:        Figure title.
        save_path:    If given, save the figure to this path.
    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = list(range(1, len(loss_history) + 1))
    ax.plot(epochs, loss_history, color=_C_LOSS, linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.set_xlim(1, max(len(loss_history), 1))
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. PR and ROC curves
# ---------------------------------------------------------------------------

def plot_pr_roc(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: Optional[float] = None,
    *,
    title_prefix: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Precision-Recall and ROC curves side by side.

    Args:
        scores:       Anomaly scores per test window, shape [n_windows].
        labels:       Binary ground-truth labels, shape [n_windows].
        threshold:    If provided, marks the operating point on both curves.
        title_prefix: Optional prefix added to subplot titles.
        save_path:    If given, save the figure to this path.
    Returns:
        The matplotlib Figure object.
    """
    labels = labels.astype(int)

    if np.unique(labels).shape[0] < 2:
        print("[plot_pr_roc] Only one class present in labels – skipping curve plots.")
        return plt.figure()

    pr_auc  = average_precision_score(labels, scores)
    roc_auc = roc_auc_score(labels, scores)

    prec, rec, pr_thresholds  = precision_recall_curve(labels, scores)
    fpr,  tpr, roc_thresholds = roc_curve(labels, scores)

    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(12, 5))

    # --- PR curve ---
    ax_pr.plot(rec, prec, color=_C_PR, linewidth=2, label=f"AP = {pr_auc:.3f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    pfx = f"{title_prefix} " if title_prefix else ""
    ax_pr.set_title(f"{pfx}Precision-Recall Curve")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)
    ax_pr.grid(True, linestyle="--", alpha=0.5)

    if threshold is not None:
        # Find the (recall, precision) point closest to this threshold
        idx = np.searchsorted(pr_thresholds, threshold, side="left")
        idx = min(idx, len(rec) - 2)   # pr arrays are 1 longer than thresholds
        ax_pr.scatter(rec[idx], prec[idx], s=80, zorder=5,
                      color=_C_THRESH, label=f"threshold={threshold:.4f}")

    ax_pr.legend(loc="lower left")

    # --- ROC curve ---
    ax_roc.plot(fpr, tpr, color=_C_ROC, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"{pfx}ROC Curve")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.grid(True, linestyle="--", alpha=0.5)

    if threshold is not None:
        idx = np.searchsorted(roc_thresholds[::-1], threshold, side="left")
        idx = max(0, len(fpr) - 1 - idx)
        ax_roc.scatter(fpr[idx], tpr[idx], s=80, zorder=5,
                       color=_C_THRESH, label=f"threshold={threshold:.4f}")
        ax_roc.legend(loc="lower right")
    else:
        ax_roc.legend(loc="lower right")

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Anomaly score vs. ground-truth
# ---------------------------------------------------------------------------

def plot_anomaly_score(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    *,
    var_series: Optional[np.ndarray] = None,
    var_index: int = 0,
    var_name: Optional[str] = None,
    max_points: int = 4000,
    title: str = "Anomaly Score vs. Ground Truth",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot anomaly comparison with one subplot per variable.

    Each variable subplot contains:
      - normalized variable trend
      - predicted anomaly regions (score >= threshold)
      - ground-truth anomaly regions
      - anomaly score (on a secondary axis) with threshold line

    If var_series is None, falls back to a single score-only subplot.
    """
    n = len(scores)
    step = max(1, n // max_points)
    idx = np.arange(0, n, step)

    scores_ds = scores[idx]
    labels_ds = labels.astype(int)[idx]
    pred_ds = (scores_ds >= threshold).astype(int)

    if var_series is None:
        fig, ax = plt.subplots(figsize=(14, 4))
        _shade_binary_regions(ax, labels_ds, step, color=_C_ANOMALY, alpha=0.45)
        _shade_binary_regions(ax, pred_ds, step, color="#C8E6C9", alpha=0.40)
        ax.plot(idx, scores_ds, color=_C_SCORE, linewidth=0.9, alpha=0.9, label="Anomaly score")
        ax.axhline(threshold, color=_C_THRESH, linewidth=1.4, linestyle="--", label=f"Threshold = {threshold:.4f}")
        ax.set_xlabel("Test window index")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.set_xlim(0, n - 1)
        gt_patch = mpatches.Patch(color=_C_ANOMALY, label="Ground-truth anomaly")
        pred_patch = mpatches.Patch(color="#C8E6C9", label="Predicted anomaly")
        handles, legend_labels = ax.get_legend_handles_labels()
        ax.legend(handles + [gt_patch, pred_patch], legend_labels + ["Ground-truth anomaly", "Predicted anomaly"], loc="upper left", fontsize=8)
        fig.tight_layout()
        _maybe_save(fig, save_path)
        return fig

    if var_series.ndim == 1:
        var_series = var_series[:, None]

    n_vars = int(var_series.shape[1])
    fig_height = max(3.0 * n_vars, 5.0)
    fig, axes = plt.subplots(n_vars, 1, figsize=(15, fig_height), sharex=True)
    if n_vars == 1:
        axes = [axes]

    ts_idx = np.linspace(0, len(var_series) - 1, num=n, dtype=int)

    for v in range(n_vars):
        ax = axes[v]
        ts = var_series[:, v]
        ts_ds = ts[ts_idx][idx]
        ts_min, ts_max = ts_ds.min(), ts_ds.max()
        if ts_max > ts_min:
            ts_norm = (ts_ds - ts_min) / (ts_max - ts_min)
        else:
            ts_norm = np.zeros_like(ts_ds)

        _shade_binary_regions(ax, labels_ds, step, color=_C_ANOMALY, alpha=0.42)
        _shade_binary_regions(ax, pred_ds, step, color="#C8E6C9", alpha=0.35)

        this_var_name = var_name if (var_name is not None and n_vars == 1) else f"var[{v}]"
        ax.plot(idx, ts_norm, color="#2E7D32", linewidth=0.9, alpha=0.85, label=f"{this_var_name} (normalized)")
        ax.set_ylabel(this_var_name)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, linestyle="--", alpha=0.25)

        ax_score = ax.twinx()
        ax_score.plot(idx, scores_ds, color=_C_SCORE, linewidth=0.7, alpha=0.45, label="Score")
        ax_score.axhline(threshold, color=_C_THRESH, linewidth=1.0, linestyle="--", alpha=0.8)
        ax_score.set_ylim(float(scores_ds.min()) - 1e-6, float(scores_ds.max()) + 1e-6)
        if v == 0:
            ax_score.set_ylabel("Score")

        if v == 0:
            gt_patch = mpatches.Patch(color=_C_ANOMALY, label="Ground-truth anomaly")
            pred_patch = mpatches.Patch(color="#C8E6C9", label="Predicted anomaly")
            sig_line = plt.Line2D([0], [0], color="#2E7D32", lw=1.2, label="Variable trend")
            thr_line = plt.Line2D([0], [0], color=_C_THRESH, lw=1.2, linestyle="--", label="Threshold")
            ax.legend(handles=[sig_line, thr_line, gt_patch, pred_patch], loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Test window index")
    axes[0].set_title(title)
    axes[-1].set_xlim(0, n - 1)

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Convenience wrapper
# ---------------------------------------------------------------------------

def plot_all(
    loss_history: List[float],
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
    *,
    test_data: Optional[np.ndarray] = None,
    var_index: int = 0,
    title_prefix: str = "",
    save_dir: str = "plots",
    run_name: str = "run",
) -> None:
    """Save all three plots to save_dir.

    Files written:
        <save_dir>/<run_name>_loss.png
        <save_dir>/<run_name>_pr_roc.png
        <save_dir>/<run_name>_anomaly_score.png
    """
    os.makedirs(save_dir, exist_ok=True)
    pfx = f"{title_prefix} – " if title_prefix else ""

    plot_loss(
        loss_history,
        title=f"{pfx}Training Loss",
        save_path=os.path.join(save_dir, f"{run_name}_loss.png"),
    )
    plot_pr_roc(
        test_scores,
        test_labels,
        threshold=threshold,
        title_prefix=title_prefix,
        save_path=os.path.join(save_dir, f"{run_name}_pr_roc.png"),
    )
    plot_anomaly_score(
        test_scores,
        test_labels,
        threshold,
        var_series=test_data,
        var_index=var_index,
        title=f"{pfx}Anomaly Score vs. Ground Truth",
        save_path=os.path.join(save_dir, f"{run_name}_anomaly_score.png"),
    )
    print(f"Plots saved to: {save_dir}/")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _shade_binary_regions(
    ax: plt.Axes,
    labels: np.ndarray,
    step: int,
    *,
    color: str,
    alpha: float,
) -> None:
    """Fill contiguous label==1 blocks with background color."""
    in_anomaly = False
    start = 0
    indices = np.where(labels)[0]
    if len(indices) == 0:
        return

    for i, flag in enumerate(labels):
        if flag and not in_anomaly:
            start = i * step
            in_anomaly = True
        elif not flag and in_anomaly:
            ax.axvspan(start, i * step, color=color, alpha=alpha, linewidth=0)
            in_anomaly = False
    if in_anomaly:
        ax.axvspan(start, len(labels) * step, color=color, alpha=alpha, linewidth=0)


def _maybe_save(fig: plt.Figure, save_path: Optional[str]) -> None:
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _standalone_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate Hyper_TSAD plots from a saved checkpoint.")
    p.add_argument("--dataset",            required=True)
    p.add_argument("--entity",             default=None)
    p.add_argument("--processed_dir",      default="processed")
    p.add_argument("--checkpoint_path",    required=True)
    p.add_argument("--loss_history_path",  default=None,
                   help="Path to a JSON file with a list of epoch losses.")
    p.add_argument("--save_dir",           default="plots")
    p.add_argument("--run_name",           default="run")
    p.add_argument("--var_index",          type=int, default=0,
                   help="Sensor variable index to overlay on the anomaly score plot.")
    p.add_argument("--window_size",        type=int, default=64)
    p.add_argument("--test_stride",        type=int, default=1)
    p.add_argument("--threshold_method",   default="pot", choices=["pot", "quantile"])
    p.add_argument("--threshold_quantile", type=float, default=0.99)
    p.add_argument("--pot_init_level",     type=float, default=0.98)
    p.add_argument("--pot_risk",           type=float, default=1e-3)
    p.add_argument("--score_normalize",    action="store_true", default=False)
    p.add_argument("--score_horizons",     type=int, default=3)
    p.add_argument("--batch_size",         type=int, default=64)
    p.add_argument("--num_workers",        type=int, default=0)
    return p


if __name__ == "__main__":
    import torch
    from evaluation import evaluate_model
    from training import (
        build_trainer,
        choose_device,
        load_processed_arrays,
        make_dataloaders,
        make_window_labels,
        make_windows,
        set_seed,
    )
    from models.model_prototype_v1 import ModelPrototype

    pargs = _standalone_parser().parse_args()
    set_seed(42)
    device = choose_device()

    train_data, test_data, labels = load_processed_arrays(
        pargs.processed_dir, pargs.dataset, pargs.entity
    )

    ckpt = torch.load(pargs.checkpoint_path, map_location=device)
    num_vars   = int(ckpt["num_vars"])
    model_args = ckpt.get("model_args", {})

    model = ModelPrototype(
        num_vars=num_vars,
        hidden_dim=int(model_args.get("hidden_dim", 64)),
        topk=int(model_args.get("topk", 5)),
        graph_ablation=model_args.get("graph_ablation", "dynamic"),
        graph_update_freq=int(model_args.get("graph_update_freq", 1)),
        graph_similarity_metric=model_args.get("graph_similarity_metric", "dot_product"),
        hypergraph_encoder_type=model_args.get("hypergraph_encoder_type", "conv"),
        hypergraph_attn_heads=int(model_args.get("hypergraph_attn_heads", 4)),
        hypergraph_attn_dropout=float(model_args.get("hypergraph_attn_dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Build a minimal args-like namespace for build_trainer
    import types
    fake_args = types.SimpleNamespace(
        lr=1e-3, weight_decay=1e-5,
        w_mse=1.0, w_graph_diff=0.01, w_graph_sparse=0.01,
        score_aggregation="topk", score_topk_ratio=0.2,
        score_normalize=ckpt.get("score_normalize", pargs.score_normalize),
        score_horizons=ckpt.get("score_horizons", pargs.score_horizons),
    )
    trainer, _ = build_trainer(fake_args, model, device)
    if "calib_mean" in ckpt:
        trainer._calib_mean = ckpt["calib_mean"].to("cpu")
        trainer._calib_std  = ckpt["calib_std"].to("cpu")

    _, train_eval_loader, test_loader, test_window_labels = make_dataloaders(
        pargs, train_data, test_data, labels
    )

    if trainer._calib_mean is None:
        print("Calibrating…")
        trainer.calibrate(train_eval_loader)
    train_scores = trainer.inference(train_eval_loader).numpy()

    metrics, test_scores = evaluate_model(
        trainer=trainer,
        test_loader=test_loader,
        test_window_labels=test_window_labels,
        train_scores=train_scores,
        threshold_method=pargs.threshold_method,
        threshold_quantile=pargs.threshold_quantile,
        pot_init_level=pargs.pot_init_level,
        pot_risk=pargs.pot_risk,
    )
    threshold = float(metrics["threshold"])

    loss_history: List[float] = []
    if pargs.loss_history_path and os.path.exists(pargs.loss_history_path):
        with open(pargs.loss_history_path, encoding="utf-8") as f:
            loss_history = json.load(f)

    plot_all(
        loss_history=loss_history,
        test_scores=test_scores,
        test_labels=test_window_labels,
        threshold=threshold,
        test_data=test_data,
        var_index=pargs.var_index,
        title_prefix=f"{pargs.dataset}",
        save_dir=pargs.save_dir,
        run_name=pargs.run_name,
    )
