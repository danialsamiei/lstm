"""
Data Cleaning and Preprocessing Pipeline
Handles missing values, outliers, time alignment, and normalization.
Based on thesis section 3-6.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans and preprocesses raw data for model input."""

    def __init__(
        self,
        missing_method_short: str = "linear_interpolation",
        missing_method_long: str = "forward_fill",
        max_gap_hours: int = 6,
        outlier_method: str = "iqr",
        iqr_multiplier: float = 1.5,
    ):
        self.missing_method_short = missing_method_short
        self.missing_method_long = missing_method_long
        self.max_gap_hours = max_gap_hours
        self.outlier_method = outlier_method
        self.iqr_multiplier = iqr_multiplier

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in time series data.
        - Short gaps: linear interpolation
        - Long gaps: forward fill

        Based on thesis section 3-6-1.
        """
        df = df.copy()
        initial_missing = df.isnull().sum().sum()

        if initial_missing == 0:
            logger.info("No missing values found")
            return df

        for col in df.columns:
            if df[col].isnull().any():
                # Identify gap lengths
                mask = df[col].isnull()
                groups = mask.ne(mask.shift()).cumsum()
                gap_sizes = mask.groupby(groups).transform("sum")

                # Short gaps: linear interpolation
                short_mask = mask & (gap_sizes <= self.max_gap_hours)
                if short_mask.any():
                    df[col] = df[col].interpolate(method="linear")

                # Long gaps: forward fill
                if df[col].isnull().any():
                    df[col] = df[col].ffill()

                # Any remaining NaN at the start: backward fill
                if df[col].isnull().any():
                    df[col] = df[col].bfill()

        final_missing = df.isnull().sum().sum()
        logger.info(
            f"Missing values: {initial_missing} -> {final_missing} "
            f"(filled {initial_missing - final_missing})"
        )
        return df

    def remove_outliers(self, df: pd.DataFrame, columns: list[str] = None) -> pd.DataFrame:
        """
        Detect and handle outliers using IQR method.
        Outliers are clipped to the boundary values instead of removed.

        Based on thesis section 3-6-1.
        """
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        total_outliers = 0

        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR

            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            n_outliers = outlier_mask.sum()

            if n_outliers > 0:
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                total_outliers += n_outliers
                logger.debug(f"Clipped {n_outliers} outliers in '{col}'")

        logger.info(f"Total outliers clipped: {total_outliers}")
        return df

    def clean(self, df: pd.DataFrame, columns: list[str] = None) -> pd.DataFrame:
        """Run full cleaning pipeline."""
        df = self.handle_missing_values(df)
        df = self.remove_outliers(df, columns)
        return df


class TimeAligner:
    """
    Aligns multiple time series to a common frequency.
    Based on thesis section 3-6-2.
    """

    def __init__(self, target_freq: str = "1h"):
        self.target_freq = target_freq

    def align(self, *dataframes: pd.DataFrame) -> pd.DataFrame:
        """
        Align multiple DataFrames to the same time frequency.
        Uses LOCF (Last Observation Carried Forward) for lower-frequency data.
        """
        if not dataframes:
            return pd.DataFrame()

        # Get the common date range
        all_starts = [df.index.min() for df in dataframes if not df.empty]
        all_ends = [df.index.max() for df in dataframes if not df.empty]

        if not all_starts:
            return pd.DataFrame()

        common_start = max(all_starts)
        common_end = min(all_ends)

        # Create target index
        target_index = pd.date_range(
            start=common_start, end=common_end, freq=self.target_freq
        )

        aligned_dfs = []
        for df in dataframes:
            if df.empty:
                continue

            # Resample to target frequency
            resampled = df.reindex(target_index, method="ffill")
            aligned_dfs.append(resampled)

        if not aligned_dfs:
            return pd.DataFrame()

        combined = pd.concat(aligned_dfs, axis=1)
        combined = combined.ffill().bfill()

        logger.info(
            f"Aligned {len(dataframes)} DataFrames to {self.target_freq} frequency. "
            f"Shape: {combined.shape}"
        )
        return combined


class DataNormalizer:
    """
    Normalizes features using Min-Max or Standard scaling.
    Based on thesis section 3-6-3.

    IMPORTANT: Parameters are fitted ONLY on training data to prevent data leakage.
    """

    def __init__(self, method: str = "minmax"):
        self.method = method
        self.scalers: dict[str, MinMaxScaler | StandardScaler] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "DataNormalizer":
        """Fit scalers on training data only."""
        for col in df.select_dtypes(include=[np.number]).columns:
            if self.method == "minmax":
                scaler = MinMaxScaler(feature_range=(0, 1))
            else:
                scaler = StandardScaler()

            scaler.fit(df[[col]])
            self.scalers[col] = scaler

        self.is_fitted = True
        logger.info(f"Fitted {self.method} scalers on {len(self.scalers)} features")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scalers."""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform. Call fit() first.")

        df_normalized = df.copy()
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in self.scalers:
                df_normalized[col] = self.scalers[col].transform(df[[col]])
            else:
                logger.warning(f"No scaler found for column '{col}', skipping")

        return df_normalized

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform normalized data back to original scale."""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted first.")

        df_original = df.copy()
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in self.scalers:
                df_original[col] = self.scalers[col].inverse_transform(df[[col]])

        return df_original

    def inverse_transform_column(
        self, values: np.ndarray, column_name: str
    ) -> np.ndarray:
        """Inverse transform a single column's values."""
        if column_name not in self.scalers:
            raise ValueError(f"No scaler found for column '{column_name}'")

        values_2d = values.reshape(-1, 1)
        return self.scalers[column_name].inverse_transform(values_2d).flatten()
