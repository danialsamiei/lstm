"""
Market Data Collector
Collects OHLCV price data from cryptocurrency exchanges using CCXT.
Based on thesis section 3-5-1.
"""

import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import ccxt
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MarketDataCollector:
    """Collects market data (price, volume) from exchanges via CCXT."""

    def __init__(self, exchange_id: str = "binance"):
        self.exchange = getattr(ccxt, exchange_id)({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        logger.info(f"Initialized exchange: {exchange_id}")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: str = "2022-01-01",
        end_date: str = "2024-01-01",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a given symbol.

        Args:
            symbol: Trading pair (e.g., 'USDT/USDT' or 'USDC/USDT')
            timeframe: Candle timeframe ('1h', '1d', etc.)
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        since = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        all_candles = []
        current = since

        logger.info(f"Fetching {symbol} from {start_date} to {end_date} ({timeframe})")

        while current < end_ts:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=current, limit=1000
                )
                if not candles:
                    break

                all_candles.extend(candles)
                current = candles[-1][0] + 1
                time.sleep(self.exchange.rateLimit / 1000)

            except ccxt.RateLimitExceeded:
                logger.warning("Rate limit hit, waiting 60s...")
                time.sleep(60)
            except ccxt.NetworkError as e:
                logger.warning(f"Network error: {e}, retrying in 10s...")
                time.sleep(10)
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                break

        if not all_candles:
            logger.warning(f"No data fetched for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(
            all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        # Filter to date range
        df = df[start_date:end_date]

        logger.info(f"Fetched {len(df)} candles for {symbol}")
        return df

    def fetch_multiple(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        start_date: str = "2022-01-01",
        end_date: str = "2024-01-01",
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple symbols."""
        data = {}
        for symbol in symbols:
            df = self.fetch_ohlcv(symbol, timeframe, start_date, end_date)
            if not df.empty:
                coin_name = symbol.split("/")[0]
                data[coin_name] = df
        return data

    def save_data(self, data: dict[str, pd.DataFrame], output_dir: str = "data/raw"):
        """Save collected data to CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, df in data.items():
            filepath = output_path / f"{name}_market.csv"
            df.to_csv(filepath)
            logger.info(f"Saved {name} market data to {filepath}")


def generate_synthetic_market_data(
    start_date: str = "2022-01-01",
    end_date: str = "2024-01-01",
    freq: str = "1h",
    base_price: float = 1.0,
    volatility: float = 0.002,
) -> pd.DataFrame:
    """
    Generate synthetic stablecoin market data for testing.
    Simulates price fluctuations around a peg value.

    Args:
        start_date: Start date
        end_date: End date
        freq: Frequency
        base_price: Peg price (1.0 for USD stablecoins)
        volatility: Price volatility around peg

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(dates)

    # Simulate depeg events
    returns = np.random.normal(0, volatility, n)

    # Add occasional depeg events (e.g., Terra crash period)
    depeg_events = [
        (500, 600, -0.01),   # Mild depeg
        (3000, 3200, -0.05), # Severe depeg (like UST)
        (5000, 5100, -0.008),  # Moderate depeg
    ]
    for start_idx, end_idx, magnitude in depeg_events:
        if end_idx < n:
            returns[start_idx:end_idx] += np.linspace(
                magnitude, magnitude / 10, end_idx - start_idx
            )

    # Generate close prices with mean reversion
    close = np.zeros(n)
    close[0] = base_price
    mean_reversion_speed = 0.01

    for i in range(1, n):
        deviation = close[i - 1] - base_price
        close[i] = close[i - 1] + returns[i] - mean_reversion_speed * deviation

    # Generate OHLV from close
    high = close + np.abs(np.random.normal(0, volatility * 0.5, n))
    low = close - np.abs(np.random.normal(0, volatility * 0.5, n))
    open_price = close + np.random.normal(0, volatility * 0.3, n)
    volume = np.abs(np.random.lognormal(mean=15, sigma=1.5, size=n))

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df
