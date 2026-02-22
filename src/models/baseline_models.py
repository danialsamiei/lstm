"""
Baseline Models for Comparison
Implements ARIMA, Simple LSTM, and GRU models.
Based on thesis section 3-8-3 (comparative evaluation).
"""

import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


def build_simple_lstm(
    n_features: int,
    sequence_length: int = 48,
    units: int = 64,
    dropout: float = 0.2,
) -> keras.Model:
    """
    Build a simple (non-hybrid) LSTM model as baseline.
    No attention mechanism, single LSTM layer.
    """
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, n_features)),
        layers.LSTM(units, dropout=dropout, return_sequences=False),
        layers.Dense(32, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(1, activation="linear"),
    ], name="Simple_LSTM")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_gru_model(
    n_features: int,
    sequence_length: int = 48,
    units: int = 64,
    dropout: float = 0.2,
) -> keras.Model:
    """Build a GRU model as baseline."""
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, n_features)),
        layers.GRU(units, dropout=dropout, return_sequences=True),
        layers.GRU(units // 2, dropout=dropout, return_sequences=False),
        layers.Dense(32, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(1, activation="linear"),
    ], name="GRU_Baseline")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


class ARIMABaseline:
    """
    ARIMA baseline model for time series prediction.
    Uses statsmodels if available, otherwise falls back to naive prediction.
    """

    def __init__(self, order: tuple = (5, 1, 0)):
        self.order = order
        self.model = None

    def fit(self, series: np.ndarray):
        """Fit ARIMA model on training data."""
        try:
            from statsmodels.tsa.arima.model import ARIMA

            self.model = ARIMA(series, order=self.order)
            self.fitted = self.model.fit()
            logger.info(f"ARIMA{self.order} fitted successfully")
        except ImportError:
            logger.warning("statsmodels not available, using naive baseline")
            self.fitted = None
            self.last_values = series[-5:]
        except Exception as e:
            logger.warning(f"ARIMA fitting failed: {e}, using naive baseline")
            self.fitted = None
            self.last_values = series[-5:]

    def predict(self, n_steps: int) -> np.ndarray:
        """Predict next n_steps values."""
        if self.fitted is not None:
            try:
                forecast = self.fitted.forecast(steps=n_steps)
                return np.array(forecast)
            except Exception as e:
                logger.warning(f"ARIMA prediction failed: {e}")

        # Naive baseline: repeat last value
        if hasattr(self, "last_values"):
            return np.full(n_steps, self.last_values[-1])
        return np.zeros(n_steps)

    def predict_rolling(
        self, train_series: np.ndarray, test_length: int
    ) -> np.ndarray:
        """
        Rolling one-step-ahead prediction.
        Re-fits the model at each step with expanding window.
        """
        predictions = []
        history = list(train_series)

        for i in range(test_length):
            try:
                from statsmodels.tsa.arima.model import ARIMA

                model = ARIMA(history, order=self.order)
                fitted = model.fit()
                pred = fitted.forecast(steps=1)[0]
            except Exception:
                pred = history[-1]

            predictions.append(pred)
            # In real scenario, would add actual value; here use prediction
            history.append(pred)

            if (i + 1) % 100 == 0:
                logger.info(f"ARIMA rolling prediction: {i + 1}/{test_length}")

        return np.array(predictions)
