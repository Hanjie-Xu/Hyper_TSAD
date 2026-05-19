from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def pick_threshold(train_scores: np.ndarray, quantile: float) -> float:
    if not (0.0 < quantile < 1.0):
        raise ValueError("threshold_quantile must be in (0, 1)")
    return float(np.quantile(train_scores, quantile))


def compute_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(f"scores and labels length mismatch: {scores.shape[0]} vs {labels.shape[0]}")

    y_true = labels.astype(int)
    y_pred = (scores >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    unique = np.unique(y_true)
    if unique.shape[0] < 2:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, scores))
        pr_auc = float(average_precision_score(y_true, scores))

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "num_samples": int(scores.shape[0]),
        "num_anomalies": int(y_true.sum()),
    }


def evaluate_model(trainer, test_loader, test_window_labels: np.ndarray, train_scores: np.ndarray, threshold_quantile: float):
    test_scores = trainer.inference(test_loader).numpy()
    threshold = pick_threshold(train_scores, threshold_quantile)
    metrics = compute_metrics(test_scores, test_window_labels, threshold)
    return metrics, test_scores
