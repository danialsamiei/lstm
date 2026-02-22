"""
Visualization Module
Generates plots for data analysis, model evaluation, and results reporting.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Use non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12, "figure.figsize": (12, 6)})


class ResultVisualizer:
    """Generates all visualization plots for the thesis."""

    def __init__(self, output_dir: str = "outputs/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_price_history(
        self,
        data: dict[str, pd.DataFrame],
        price_col: str = "close",
        filename: str = "price_history.png",
    ):
        """Plot price history for all stablecoins."""
        fig, axes = plt.subplots(len(data), 1, figsize=(14, 4 * len(data)), sharex=True)
        if len(data) == 1:
            axes = [axes]

        for ax, (name, df) in zip(axes, data.items()):
            ax.plot(df.index, df[price_col], linewidth=0.8, label=name)
            ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Peg ($1.00)")
            ax.set_ylabel("Price (USD)")
            ax.set_title(f"{name} Price History")
            ax.legend()

        plt.xlabel("Date")
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Price history plot saved")

    def plot_peg_deviation(
        self,
        data: dict[str, pd.DataFrame],
        filename: str = "peg_deviation.png",
    ):
        """Plot peg deviation over time."""
        fig, ax = plt.subplots(figsize=(14, 6))

        for name, df in data.items():
            if "peg_deviation" in df.columns:
                ax.plot(df.index, df["peg_deviation"], linewidth=0.6, label=name, alpha=0.8)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_ylabel("Peg Deviation ($)")
        ax.set_xlabel("Date")
        ax.set_title("Stablecoin Peg Deviation Over Time")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_training_history(
        self,
        history,
        filename: str = "training_history.png",
    ):
        """Plot training and validation loss curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        axes[0].plot(history.history["loss"], label="Training Loss")
        axes[0].plot(history.history["val_loss"], label="Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss (MSE)")
        axes[0].set_title("Model Loss")
        axes[0].legend()

        # MAE
        if "mae" in history.history:
            axes[1].plot(history.history["mae"], label="Training MAE")
            axes[1].plot(history.history["val_mae"], label="Validation MAE")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("MAE")
            axes[1].set_title("Model MAE")
            axes[1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Training history plot saved")

    def plot_predictions_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: pd.DatetimeIndex = None,
        model_name: str = "Deep-XAI-Stable",
        filename: str = "predictions_vs_actual.png",
    ):
        """Plot predicted vs actual values."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        x = dates if dates is not None else range(len(y_true))

        # Time series comparison
        axes[0].plot(x, y_true, label="Actual", linewidth=0.8, alpha=0.8)
        axes[0].plot(x, y_pred, label=f"{model_name} Predicted", linewidth=0.8, alpha=0.8)
        axes[0].set_ylabel("Peg Deviation")
        axes[0].set_title(f"{model_name}: Predicted vs Actual")
        axes[0].legend()

        # Scatter plot
        axes[1].scatter(y_true, y_pred, alpha=0.3, s=10)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")
        axes[1].set_xlabel("Actual")
        axes[1].set_ylabel("Predicted")
        axes[1].set_title("Prediction Scatter Plot")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Predictions vs actual plot saved")

    def plot_model_comparison(
        self,
        results: dict[str, dict],
        filename: str = "model_comparison.png",
    ):
        """Plot comparison of all models' metrics."""
        metrics_to_plot = ["rmse", "mae", "mape", "directional_accuracy"]
        model_names = [k for k in results.keys() if k != "diebold_mariano"]
        n_metrics = len(metrics_to_plot)

        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

        for i, metric in enumerate(metrics_to_plot):
            values = []
            names = []
            for model in model_names:
                if metric in results[model]:
                    values.append(results[model][metric])
                    names.append(model)

            colors = ["steelblue" if "XAI" in n or "Deep" in n else "lightcoral" for n in names]
            axes[i].bar(names, values, color=colors)
            axes[i].set_title(metric.upper())
            axes[i].set_xticklabels(names, rotation=45, ha="right")

        plt.suptitle("Model Comparison", fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Model comparison plot saved")

    def plot_attention_weights(
        self,
        attention_weights: np.ndarray,
        sample_indices: list[int] = None,
        filename: str = "attention_weights.png",
    ):
        """Plot attention weights heatmap."""
        if sample_indices is None:
            sample_indices = list(range(min(10, len(attention_weights))))

        weights = attention_weights[sample_indices]

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(
            weights,
            cmap="YlOrRd",
            ax=ax,
            xticklabels=5,
            yticklabels=[f"Sample {i}" for i in sample_indices],
        )
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Sample")
        ax.set_title("Attention Weights Heatmap")

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Attention weights plot saved")

    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        filename: str = "correlation_matrix.png",
    ):
        """Plot feature correlation matrix."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            cmap="RdBu_r",
            center=0,
            annot=False,
            fmt=".2f",
            ax=ax,
        )
        ax.set_title("Feature Correlation Matrix")

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Correlation matrix plot saved")

    def plot_data_distribution(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        columns: list[str] = None,
        filename: str = "normalization_comparison.png",
    ):
        """Plot data distributions before and after normalization."""
        if columns is None:
            columns = df_before.select_dtypes(include=[np.number]).columns[:6].tolist()

        n_cols = min(len(columns), 6)
        fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

        for i, col in enumerate(columns[:n_cols]):
            if col in df_before.columns:
                axes[0, i].hist(df_before[col].dropna(), bins=50, alpha=0.7, color="steelblue")
                axes[0, i].set_title(f"{col}\n(Before)")

            if col in df_after.columns:
                axes[1, i].hist(df_after[col].dropna(), bins=50, alpha=0.7, color="coral")
                axes[1, i].set_title(f"{col}\n(After)")

        plt.suptitle("Data Distribution: Before vs After Normalization", fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Distribution comparison plot saved")

    def plot_error_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Deep-XAI-Stable",
        filename: str = "error_distribution.png",
    ):
        """Plot prediction error distribution."""
        errors = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(errors, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
        axes[0].axvline(x=0, color="red", linestyle="--")
        axes[0].set_xlabel("Prediction Error")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title(f"{model_name}: Error Distribution")

        # QQ plot
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[1])
        axes[1].set_title("Q-Q Plot")

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Error distribution plot saved")
