"""
Model Evaluation Metrics
Implements RMSE, MAE, MAPE, Directional Accuracy, and Diebold-Mariano test.
Based on thesis section 3-8.
"""

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.
    Based on thesis section 3-8-1.

    RMSE = sqrt(1/n * sum((y_i - y_hat_i)^2))
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.
    Based on thesis section 3-8-1.

    MAE = 1/n * sum(|y_i - y_hat_i|)
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    Based on thesis section 3-8-1.

    MAPE = 100%/n * sum(|y_i - y_hat_i| / |y_i|)
    """
    # Avoid division by zero
    mask = np.abs(y_true) > 1e-10
    if mask.sum() == 0:
        return float("inf")
    return float(100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional Accuracy - percentage of correct direction predictions.
    Based on thesis section 3-8-2.

    Measures if the model correctly predicts the direction of change
    (increase or decrease in peg deviation).
    """
    if len(y_true) < 2:
        return 0.0

    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))

    correct = np.sum(true_direction == pred_direction)
    total = len(true_direction)

    return float(correct / total * 100)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R^2)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def diebold_mariano_test(
    y_true: np.ndarray,
    pred_model1: np.ndarray,
    pred_model2: np.ndarray,
    loss_fn: str = "mse",
    h: int = 1,
) -> dict:
    """
    Diebold-Mariano test for comparing predictive accuracy of two models.
    Based on thesis section 3-8-3.

    H0: The two models have equal predictive accuracy.
    H1: Model 1 is more accurate than Model 2.

    Args:
        y_true: Actual values
        pred_model1: Predictions from model 1 (proposed)
        pred_model2: Predictions from model 2 (baseline)
        loss_fn: Loss function ('mse', 'mae', 'mape')
        h: Forecast horizon

    Returns:
        Dictionary with test statistic, p-value, and conclusion
    """
    n = len(y_true)

    # Compute loss differentials
    if loss_fn == "mse":
        loss1 = (y_true - pred_model1) ** 2
        loss2 = (y_true - pred_model2) ** 2
    elif loss_fn == "mae":
        loss1 = np.abs(y_true - pred_model1)
        loss2 = np.abs(y_true - pred_model2)
    else:
        loss1 = (y_true - pred_model1) ** 2
        loss2 = (y_true - pred_model2) ** 2

    d = loss1 - loss2  # Loss differential series
    d_mean = np.mean(d)

    # Compute autocovariance
    gamma = np.zeros(h)
    for k in range(h):
        gamma[k] = np.mean((d[k:] - d_mean) * (d[: n - k] - d_mean))

    # Long-run variance estimate
    V = gamma[0] + 2 * np.sum(gamma[1:])
    V = max(V, 1e-10)  # Prevent division by zero

    # DM statistic
    dm_stat = d_mean / np.sqrt(V / n)

    # Two-sided p-value
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    # Conclusion
    if p_value < 0.01:
        significance = "***"
        conclusion = "Highly significant difference (p < 0.01)"
    elif p_value < 0.05:
        significance = "**"
        conclusion = "Significant difference (p < 0.05)"
    elif p_value < 0.10:
        significance = "*"
        conclusion = "Marginally significant (p < 0.10)"
    else:
        significance = ""
        conclusion = "No significant difference"

    # Which model is better
    if d_mean < 0:
        better = "model1"
    else:
        better = "model2"

    result = {
        "dm_statistic": float(dm_stat),
        "p_value": float(p_value),
        "significance": significance,
        "conclusion": conclusion,
        "mean_loss_differential": float(d_mean),
        "better_model": better,
    }

    logger.info(
        f"DM Test: stat={dm_stat:.4f}, p={p_value:.4f} {significance} "
        f"(better: {better})"
    )
    return result


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
) -> dict:
    """
    Compute all evaluation metrics for a model.

    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "r_squared": r_squared(y_true, y_pred),
    }

    logger.info(f"=== {model_name} Evaluation ===")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.6f}")

    return metrics


def compare_models(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    proposed_model_name: str = "Deep-XAI-Stable",
) -> dict:
    """
    Compare proposed model against all baselines.

    Args:
        y_true: Actual values
        predictions: Dict mapping model name -> predictions array
        proposed_model_name: Name of the proposed model

    Returns:
        Comparison results including DM tests
    """
    results = {}

    # Evaluate each model
    for name, preds in predictions.items():
        results[name] = evaluate_model(y_true, preds, name)

    # Run DM tests comparing proposed model vs each baseline
    if proposed_model_name in predictions:
        proposed_preds = predictions[proposed_model_name]
        dm_results = {}

        for name, preds in predictions.items():
            if name != proposed_model_name:
                dm = diebold_mariano_test(y_true, proposed_preds, preds)
                dm_results[f"{proposed_model_name}_vs_{name}"] = dm

        results["diebold_mariano"] = dm_results

    return results
