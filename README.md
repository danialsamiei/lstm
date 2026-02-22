# Deep-XAI-Stable

Stablecoin price prediction using hybrid deep learning (LSTM + Attention) with SHAP-based explainability. Integrates on-chain data and market sentiment analysis.

## Project Structure

```
lstm/
├── configs/
│   └── config.yaml              # Pipeline configuration
├── src/
│   ├── data/
│   │   ├── market_data.py        # Market data collection (CCXT)
│   │   ├── onchain_data.py       # On-chain metrics collection
│   │   └── sentiment_data.py     # Sentiment analysis (FinBERT)
│   ├── preprocessing/
│   │   ├── data_cleaner.py       # Cleaning, alignment, normalization
│   │   └── feature_engineering.py # Technical indicators, lag features
│   ├── models/
│   │   ├── attention.py          # Self-Attention & Multi-Head Attention
│   │   ├── deep_xai_stable.py    # Main model architecture
│   │   ├── baseline_models.py    # ARIMA, Simple LSTM, GRU baselines
│   │   └── trainer.py            # Training loop & data splitting
│   ├── evaluation/
│   │   ├── metrics.py            # RMSE, MAE, MAPE, DA, Diebold-Mariano
│   │   └── visualizer.py         # Result visualization plots
│   ├── explainability/
│   │   └── shap_explainer.py     # SHAP-based model interpretation
│   └── utils/
│       └── helpers.py            # Config, logging, seed utilities
├── main.py                       # Pipeline orchestrator
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py                    # Full pipeline (synthetic data)
python main.py --use-real-data    # With real API data
```

## Stablecoins Analyzed

- **USDT** (Tether) - Centralized, fiat-backed
- **USDC** (USD Coin) - Centralized, transparent
- **DAI** - Decentralized, crypto-collateralized
