"""
Model Training Module
Handles data preparation, training loop, and model checkpointing.
Based on thesis sections 3-6-5, 3-7-4.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


class SequenceDataset:
    """
    Creates sliding window sequences from time series data for LSTM input.

    Converts a flat DataFrame into (X, y) pairs where:
    - X: (num_samples, sequence_length, n_features)
    - y: (num_samples, 1)
    """

    def __init__(
        self,
        sequence_length: int = 48,
        prediction_horizon: int = 1,
        target_column: str = "peg_deviation",
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.target_column = target_column

    def create_sequences(
        self, df: pd.DataFrame, feature_columns: list[str] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create input/output sequences from DataFrame.

        Args:
            df: DataFrame with features and target
            feature_columns: List of feature column names.
                If None, uses all columns except target.

        Returns:
            X: (n_samples, sequence_length, n_features)
            y: (n_samples,)
        """
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != self.target_column]

        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in DataFrame"
            )

        features = df[feature_columns].values
        target = df[self.target_column].values

        X, y = [], []
        for i in range(len(features) - self.sequence_length - self.prediction_horizon + 1):
            X.append(features[i : i + self.sequence_length])
            y.append(target[i + self.sequence_length + self.prediction_horizon - 1])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        logger.info(
            f"Created {len(X)} sequences: X{X.shape}, y{y.shape}"
        )
        return X, y


class DataSplitter:
    """
    Splits data chronologically into train/validation/test sets.
    Based on thesis section 3-6-5.

    IMPORTANT: Chronological split, NOT random, to preserve time structure.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame chronologically.

        Returns:
            (train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        logger.info(
            f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
        return train_df, val_df, test_df

    def split_arrays(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple:
        """Split numpy arrays chronologically."""
        n = len(X)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        logger.info(
            f"Array split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )
        return X_train, y_train, X_val, y_val, X_test, y_test


class ModelTrainer:
    """
    Handles model training with callbacks and checkpointing.
    Based on thesis section 3-7-4.
    """

    def __init__(
        self,
        model: keras.Model,
        output_dir: str = "outputs/models",
        model_name: str = "deep_xai_stable",
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.history = None

    def get_callbacks(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        lr_patience: int = 5,
        lr_factor: float = 0.5,
    ) -> list[keras.callbacks.Callback]:
        """
        Create training callbacks.
        - Early stopping to prevent overfitting (Section 3-7-4)
        - Learning rate reduction on plateau
        - Model checkpointing
        """
        callbacks = [
            # Early Stopping
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True,
                verbose=1,
            ),
            # Reduce LR on Plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=lr_factor,
                patience=lr_patience,
                min_lr=1e-7,
                verbose=1,
            ),
            # Model Checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.output_dir / f"{self.model_name}_best.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            # TensorBoard Logging
            keras.callbacks.TensorBoard(
                log_dir=str(self.output_dir.parent / "logs" / self.model_name),
                histogram_freq=1,
            ),
        ]
        return callbacks

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 200,
        batch_size: int = 64,
        callbacks: list = None,
    ) -> keras.callbacks.History:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum number of epochs
            batch_size: Batch size
            callbacks: Custom callbacks (uses defaults if None)

        Returns:
            Training history
        """
        if callbacks is None:
            callbacks = self.get_callbacks()

        logger.info(
            f"Starting training: {epochs} epochs, batch_size={batch_size}, "
            f"train_size={len(X_train)}, val_size={len(X_val)}"
        )

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        logger.info(
            f"Training complete. Best val_loss: "
            f"{min(self.history.history['val_loss']):.6f}"
        )
        return self.history

    def save_model(self, filename: str = None):
        """Save the trained model."""
        if filename is None:
            filename = f"{self.model_name}_final.keras"
        filepath = self.output_dir / filename
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a saved model."""
        from .attention import AttentionLayer

        self.model = keras.models.load_model(
            filepath,
            custom_objects={"AttentionLayer": AttentionLayer},
        )
        logger.info(f"Model loaded from {filepath}")
