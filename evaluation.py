from typing import Dict

import numpy as np
from scipy.stats import genpareto
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def pick_threshold(train_scores: np.ndarray, quantile: float) -> float:
    if not (0.0 < quantile < 1.0):
        raise ValueError("threshold_quantile must be in (0, 1)")
    return float(np.quantile(train_scores, quantile))


def pick_threshold_pot(
    train_scores: np.ndarray,
    init_level: float,
    risk: float,
    min_excess: int = 20,
) -> float:
    """Estimate anomaly threshold via POT (GPD tail fitting).

    Args:
        train_scores: Calibration scores from normal training windows.
        init_level: Initial high quantile u used to extract exceedances.
        risk: Target tail probability P(X > t).
        min_excess: Minimum number of exceedances needed for stable GPD fitting.
    """
    if not (0.5 < init_level < 1.0):
        raise ValueError("pot_init_level must be in (0.5, 1)")
    if not (0.0 < risk < 1.0):
        raise ValueError("pot_risk must be in (0, 1)")

    scores = np.asarray(train_scores, dtype=np.float64)
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    if scores.shape[0] < 50:
        # POT is unstable on tiny samples; fallback to quantile.
        return float(np.quantile(scores, max(init_level, 0.99)))

    u = float(np.quantile(scores, init_level))
    excess = scores[scores > u] - u
    n = scores.shape[0]
    nu = excess.shape[0]

    if nu < min_excess:
        # Not enough tail samples to fit GPD robustly.
        return float(np.quantile(scores, max(init_level, 0.99)))

    p_u = nu / float(n)
    # If requested risk is above empirical exceedance rate, keep threshold at u.
    if risk >= p_u:
        return u

    try:
        shape, _, scale = genpareto.fit(excess, floc=0.0)
        shape = float(shape)
        scale = float(scale)
        if scale <= 0:
            return u

        if abs(shape) < 1e-8:
            t = u + scale * np.log(p_u / risk)
        else:
            t = u + (scale / shape) * ((p_u / risk) ** shape - 1.0)

        # Keep a valid finite threshold.
        if not np.isfinite(t):
            return u

        # Hard upper cap: threshold may not exceed the empirical 99.9th percentile
        # of the training scores.  Without this guard, heavy-tailed GPD fits can
        # extrapolate to unrealistically high values that classify almost every
        # test window as normal, collapsing recall toward zero.
        cap = float(np.quantile(scores, min(0.999, 1.0 - risk * 5)))
        t = min(float(max(t, u)), cap)
        return t
    except Exception:
        return u


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


def evaluate_model(
    trainer,
    test_loader,
    test_window_labels: np.ndarray,
    train_scores: np.ndarray,
    threshold_method: str,
    threshold_quantile: float,
    pot_init_level: float,
    pot_risk: float,
):
    test_scores = trainer.inference(test_loader).numpy()

    if threshold_method == "pot":
        threshold = pick_threshold_pot(
            train_scores=train_scores,
            init_level=pot_init_level,
            risk=pot_risk,
        )
    else:
        threshold = pick_threshold(train_scores, threshold_quantile)

    metrics = compute_metrics(test_scores, test_window_labels, threshold)
    metrics["threshold_method"] = threshold_method
    if threshold_method == "pot":
        metrics["pot_init_level"] = float(pot_init_level)
        metrics["pot_risk"] = float(pot_risk)
    else:
        metrics["threshold_quantile"] = float(threshold_quantile)
    return metrics, test_scores
