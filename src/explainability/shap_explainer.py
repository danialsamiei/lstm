"""
XAI Module - SHAP-based Model Explainability
Implements SHAP (SHapley Additive exPlanations) for model interpretation.
Based on thesis section 3-7-3.

Provides:
1. Feature Importance plots (global explanations)
2. Dependence plots (feature interaction effects)
3. Local explanations (individual prediction explanations)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based explainability for the Deep-XAI-Stable model.

    Uses DeepExplainer for neural network models based on cooperative
    game theory (Shapley values).

    phi_i = sum over S subset of N\\{i}:
        |S|! * (|N| - |S| - 1)! / |N|! * [f(S union {i}) - f(S)]
    """

    def __init__(
        self,
        model,
        feature_names: list[str] = None,
        output_dir: str = "outputs/plots",
    ):
        self.model = model
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.explainer = None
        self.shap_values = None

    def initialize_explainer(
        self,
        background_data: np.ndarray,
        explainer_type: str = "deep",
    ):
        """
        Initialize the SHAP explainer.

        Args:
            background_data: Background dataset for computing expectations
            explainer_type: 'deep' for DeepExplainer, 'kernel' for KernelExplainer
        """
        import shap

        if explainer_type == "deep":
            try:
                self.explainer = shap.DeepExplainer(self.model, background_data)
                logger.info("Initialized SHAP DeepExplainer")
            except Exception as e:
                logger.warning(
                    f"DeepExplainer failed ({e}), falling back to GradientExplainer"
                )
                try:
                    self.explainer = shap.GradientExplainer(self.model, background_data)
                    logger.info("Initialized SHAP GradientExplainer")
                except Exception as e2:
                    logger.warning(
                        f"GradientExplainer failed ({e2}), using KernelExplainer"
                    )
                    explainer_type = "kernel"

        if explainer_type == "kernel":
            # For KernelExplainer, we need a prediction function
            predict_fn = lambda x: self.model.predict(x, verbose=0).flatten()
            # Reshape background for kernel explainer
            bg_2d = background_data.reshape(len(background_data), -1)
            self.explainer = shap.KernelExplainer(
                lambda x: predict_fn(x.reshape(-1, *background_data.shape[1:])),
                bg_2d,
            )
            logger.info("Initialized SHAP KernelExplainer")

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for input data.

        Args:
            X: Input data (n_samples, sequence_length, n_features)

        Returns:
            SHAP values array with same shape as X
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")

        import shap

        logger.info(f"Computing SHAP values for {len(X)} samples...")
        self.shap_values = self.explainer.shap_values(X)

        # Handle different output formats
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[0]

        logger.info(f"SHAP values shape: {np.array(self.shap_values).shape}")
        return self.shap_values

    def get_feature_importance(
        self,
        shap_values: np.ndarray = None,
    ) -> pd.DataFrame:
        """
        Compute global feature importance from SHAP values.
        Based on thesis section 3-7-3, Feature Importance plot.

        Returns:
            DataFrame with feature names and their mean |SHAP| values
        """
        if shap_values is None:
            shap_values = self.shap_values

        if shap_values is None:
            raise ValueError("No SHAP values available. Call compute_shap_values first.")

        # Average absolute SHAP values across time steps and samples
        # Shape: (n_samples, seq_length, n_features) -> (n_features,)
        if len(shap_values.shape) == 3:
            importance = np.mean(np.abs(shap_values), axis=(0, 1))
        else:
            importance = np.mean(np.abs(shap_values), axis=0)

        if self.feature_names and len(self.feature_names) == len(importance):
            names = self.feature_names
        else:
            names = [f"feature_{i}" for i in range(len(importance))]

        df = pd.DataFrame({
            "feature": names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        return df

    def plot_feature_importance(
        self,
        shap_values: np.ndarray = None,
        X: np.ndarray = None,
        max_features: int = 20,
        filename: str = "feature_importance.png",
    ):
        """
        Plot global feature importance bar chart.
        Based on thesis section 3-7-3.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        importance_df = self.get_feature_importance(shap_values)
        top_features = importance_df.head(max_features)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            range(len(top_features)),
            top_features["importance"].values,
            color="steelblue",
        )
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features["feature"].values)
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Feature Importance (SHAP Values)")
        ax.invert_yaxis()
        plt.tight_layout()

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Feature importance plot saved to {filepath}")

    def plot_summary(
        self,
        shap_values: np.ndarray = None,
        X: np.ndarray = None,
        max_features: int = 20,
        filename: str = "shap_summary.png",
    ):
        """
        Plot SHAP summary (beeswarm) plot.
        Shows both feature importance and direction of effects.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        if shap_values is None:
            shap_values = self.shap_values

        # For 3D data, aggregate over time dimension
        if len(shap_values.shape) == 3:
            sv_2d = np.mean(shap_values, axis=1)  # Average over time steps
            if X is not None and len(X.shape) == 3:
                X_2d = np.mean(X, axis=1)
            else:
                X_2d = X
        else:
            sv_2d = shap_values
            X_2d = X

        feature_names = self.feature_names or [
            f"feature_{i}" for i in range(sv_2d.shape[-1])
        ]

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            sv_2d,
            X_2d,
            feature_names=feature_names[:sv_2d.shape[-1]],
            max_display=max_features,
            show=False,
        )

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary plot saved to {filepath}")

    def plot_dependence(
        self,
        feature_idx: int,
        shap_values: np.ndarray = None,
        X: np.ndarray = None,
        filename: str = None,
    ):
        """
        Plot dependence plot for a specific feature.
        Based on thesis section 3-7-3, Dependence Plot.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        if shap_values is None:
            shap_values = self.shap_values

        if len(shap_values.shape) == 3:
            sv_2d = np.mean(shap_values, axis=1)
            if X is not None and len(X.shape) == 3:
                X_2d = np.mean(X, axis=1)
            else:
                X_2d = X
        else:
            sv_2d = shap_values
            X_2d = X

        feature_names = self.feature_names or [
            f"feature_{i}" for i in range(sv_2d.shape[-1])
        ]

        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feature_idx,
            sv_2d,
            X_2d,
            feature_names=feature_names[:sv_2d.shape[-1]],
            show=False,
        )

        if filename is None:
            fname = feature_names[feature_idx] if feature_idx < len(feature_names) else f"feature_{feature_idx}"
            filename = f"dependence_{fname}.png"

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Dependence plot saved to {filepath}")

    def explain_single_prediction(
        self,
        X_single: np.ndarray,
        prediction: float,
        shap_values_single: np.ndarray = None,
        top_n: int = 10,
    ) -> dict:
        """
        Generate local explanation for a single prediction.
        Based on thesis section 3-7-3, Local Explanations.

        Args:
            X_single: Single input sample
            prediction: Model's prediction for this sample
            shap_values_single: SHAP values for this sample
            top_n: Number of top contributing features

        Returns:
            Dictionary with explanation details
        """
        if shap_values_single is None:
            if self.explainer is not None:
                shap_values_single = self.explainer.shap_values(
                    X_single.reshape(1, *X_single.shape)
                )
                if isinstance(shap_values_single, list):
                    shap_values_single = shap_values_single[0]
                shap_values_single = shap_values_single[0]

        if shap_values_single is None:
            return {"error": "Could not compute SHAP values"}

        # Aggregate over time steps
        if len(shap_values_single.shape) == 2:
            feature_contributions = np.mean(np.abs(shap_values_single), axis=0)
            feature_directions = np.mean(shap_values_single, axis=0)
        else:
            feature_contributions = np.abs(shap_values_single)
            feature_directions = shap_values_single

        # Get top contributing features
        top_indices = np.argsort(feature_contributions)[::-1][:top_n]

        feature_names = self.feature_names or [
            f"feature_{i}" for i in range(len(feature_contributions))
        ]

        explanations = []
        for idx in top_indices:
            name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            contribution = float(feature_contributions[idx])
            direction = "increases" if feature_directions[idx] > 0 else "decreases"
            explanations.append({
                "feature": name,
                "contribution": contribution,
                "direction": direction,
            })

        return {
            "prediction": float(prediction),
            "top_features": explanations,
            "explanation_text": self._generate_explanation_text(
                prediction, explanations
            ),
        }

    def _generate_explanation_text(
        self, prediction: float, explanations: list[dict]
    ) -> str:
        """Generate human-readable explanation text."""
        if not explanations:
            return "No explanation available."

        direction = "increase" if prediction > 0 else "decrease"
        text = f"The model predicts a peg deviation of {prediction:.6f} ({direction}). "
        text += "The main contributing factors are: "

        factors = []
        for exp in explanations[:3]:
            factors.append(
                f"{exp['feature']} ({exp['direction']} the deviation by {exp['contribution']:.4f})"
            )

        text += ", ".join(factors) + "."
        return text

    def evaluate_fidelity(
        self,
        X: np.ndarray,
        top_k: int = 5,
    ) -> dict:
        """
        Evaluate fidelity of SHAP explanations.
        Based on thesis section 3-8-4.

        Measures correlation between SHAP-identified important features
        and actual model output changes when those features are removed.
        """
        if self.shap_values is None or self.model is None:
            return {"error": "SHAP values or model not available"}

        # Get feature importance ranking
        importance = self.get_feature_importance()
        top_features = importance.head(top_k).index.tolist()

        # Get baseline predictions
        baseline_pred = self.model.predict(X, verbose=0).flatten()

        # Remove top features and measure prediction change
        X_modified = X.copy()
        for feat_idx in top_features:
            if isinstance(feat_idx, int) and feat_idx < X.shape[-1]:
                X_modified[:, :, feat_idx] = 0  # Zero out feature

        modified_pred = self.model.predict(X_modified, verbose=0).flatten()
        pred_change = np.mean(np.abs(baseline_pred - modified_pred))

        return {
            "prediction_change_top_k": float(pred_change),
            "top_k_features_removed": top_k,
            "fidelity_score": float(pred_change),  # Higher = more faithful
        }

    def evaluate_stability(
        self,
        X: np.ndarray,
        n_perturbations: int = 10,
        noise_level: float = 0.01,
    ) -> dict:
        """
        Evaluate stability of SHAP explanations.
        Based on thesis section 3-8-4.

        Checks if similar inputs produce similar explanations.
        """
        if self.explainer is None:
            return {"error": "Explainer not initialized"}

        import shap

        # Get SHAP values for original data
        original_sv = self.explainer.shap_values(X[:5])
        if isinstance(original_sv, list):
            original_sv = original_sv[0]

        # Get SHAP values for perturbed data
        correlations = []
        for _ in range(n_perturbations):
            noise = np.random.normal(0, noise_level, X[:5].shape)
            X_perturbed = X[:5] + noise

            perturbed_sv = self.explainer.shap_values(X_perturbed)
            if isinstance(perturbed_sv, list):
                perturbed_sv = perturbed_sv[0]

            # Compute correlation between original and perturbed explanations
            corr = np.corrcoef(
                original_sv.flatten(), perturbed_sv.flatten()
            )[0, 1]
            correlations.append(corr)

        return {
            "mean_stability": float(np.mean(correlations)),
            "std_stability": float(np.std(correlations)),
            "min_stability": float(np.min(correlations)),
        }
