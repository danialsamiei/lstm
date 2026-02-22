"""
Feature Engineering Module
Creates derived features from raw data for model input.
Based on thesis section 3-6-4.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates engineered features from raw market and on-chain data."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def compute_log_returns(self, df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        """
        Compute logarithmic returns to make the series stationary.
        R_t = ln(P_t / P_{t-1})

        Based on thesis equation in section 3-6-4.
        """
        df = df.copy()
        df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
        df["log_return"] = df["log_return"].fillna(0)
        return df

    def compute_peg_deviation(
        self, df: pd.DataFrame, price_col: str = "close", peg_value: float = 1.0
    ) -> pd.DataFrame:
        """
        Compute deviation from the peg value.
        This is the primary target variable for stablecoin prediction.
        """
        df = df.copy()
        df["peg_deviation"] = df[price_col] - peg_value
        df["peg_deviation_pct"] = (df[price_col] - peg_value) / peg_value * 100
        df["abs_peg_deviation"] = np.abs(df["peg_deviation"])
        return df

    def compute_rsi(self, df: pd.DataFrame, price_col: str = "close", period: int = 14) -> pd.DataFrame:
        """
        Compute Relative Strength Index (RSI).
        """
        df = df.copy()
        delta = df[price_col].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    def compute_macd(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """Compute MACD (Moving Average Convergence Divergence)."""
        df = df.copy()
        ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
        ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()

        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        return df

    def compute_bollinger_bands(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        period: int = 20,
        std_dev: int = 2,
    ) -> pd.DataFrame:
        """Compute Bollinger Bands."""
        df = df.copy()
        sma = df[price_col].rolling(window=period).mean()
        std = df[price_col].rolling(window=period).std()

        df["bb_upper"] = sma + std_dev * std
        df["bb_lower"] = sma - std_dev * std
        df["bb_middle"] = sma
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_middle"] + 1e-10)
        df["bb_position"] = (df[price_col] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"] + 1e-10
        )
        return df

    def compute_volatility(
        self, df: pd.DataFrame, price_col: str = "close", windows: list[int] = None
    ) -> pd.DataFrame:
        """Compute rolling volatility at different windows."""
        df = df.copy()
        if windows is None:
            windows = [6, 12, 24]

        returns = df[price_col].pct_change()
        for w in windows:
            df[f"volatility_{w}h"] = returns.rolling(window=w).std()
        return df

    def add_lag_features(
        self, df: pd.DataFrame, columns: list[str], lags: list[int] = None
    ) -> pd.DataFrame:
        """
        Create time-lagged features.
        Based on thesis section 3-6-4 (Lags).
        """
        df = df.copy()
        if lags is None:
            lags = [1, 2, 3, 6, 12, 24]

        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        return df

    def add_rolling_features(
        self, df: pd.DataFrame, columns: list[str], windows: list[int] = None
    ) -> pd.DataFrame:
        """Add rolling statistics (mean, std) for given columns."""
        df = df.copy()
        if windows is None:
            windows = [6, 12, 24]

        for col in columns:
            if col in df.columns:
                for w in windows:
                    df[f"{col}_ma_{w}"] = df[col].rolling(window=w).mean()
                    df[f"{col}_std_{w}"] = df[col].rolling(window=w).std()

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical time features (hour of day, day of week)."""
        df = df.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
            df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        return df

    def engineer_all_features(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        lag_columns: list[str] = None,
    ) -> pd.DataFrame:
        """
        Run the full feature engineering pipeline.

        Args:
            df: Raw data with at least OHLCV columns
            price_col: Name of the price column
            lag_columns: Columns to create lag features for

        Returns:
            DataFrame with all engineered features
        """
        logger.info(f"Starting feature engineering. Initial shape: {df.shape}")

        # Log returns
        df = self.compute_log_returns(df, price_col)

        # Peg deviation (target variable)
        df = self.compute_peg_deviation(df, price_col)

        # Technical indicators
        df = self.compute_rsi(df, price_col)
        df = self.compute_macd(df, price_col)
        df = self.compute_bollinger_bands(df, price_col)
        df = self.compute_volatility(df, price_col)

        # Time features
        df = self.add_time_features(df)

        # Lag features
        if lag_columns is None:
            lag_columns = ["close", "volume", "log_return", "peg_deviation"]

        existing_lag_cols = [c for c in lag_columns if c in df.columns]
        df = self.add_lag_features(df, existing_lag_cols)

        # Rolling features
        rolling_cols = [c for c in ["close", "volume"] if c in df.columns]
        df = self.add_rolling_features(df, rolling_cols)

        # Drop rows with NaN from feature computation
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)

        logger.info(
            f"Feature engineering complete. Final shape: {df.shape} "
            f"(dropped {dropped} rows with NaN)"
        )
        return df
