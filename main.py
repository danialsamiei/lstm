"""
Deep-XAI-Stable: Main Pipeline Orchestrator
============================================

Stablecoin Price Prediction with LSTM + Attention + SHAP Explainability.

This is the main entry point that orchestrates the entire pipeline:
1. Data Collection (Market, On-Chain, Sentiment)
2. Data Preprocessing (Cleaning, Alignment, Normalization)
3. Feature Engineering
4. Model Training (Deep-XAI-Stable + Baselines)
5. Evaluation & Comparison
6. XAI Analysis (SHAP)

Usage:
    python main.py                          # Full pipeline with synthetic data
    python main.py --mode train             # Train only
    python main.py --mode evaluate          # Evaluate only
    python main.py --config configs/config.yaml  # Custom config
    python main.py --use-real-data          # Use real API data
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project modules
from src.utils.helpers import setup_logging, load_config, set_random_seed, create_output_dirs
from src.data.market_data import MarketDataCollector, generate_synthetic_market_data
from src.data.onchain_data import OnChainDataCollector
from src.data.sentiment_data import SentimentDataCollector, SentimentAnalyzer
from src.preprocessing.data_cleaner import DataCleaner, TimeAligner, DataNormalizer
from src.preprocessing.feature_engineering import FeatureEngineer
from src.models.trainer import SequenceDataset, DataSplitter, ModelTrainer
from src.evaluation.metrics import evaluate_model, compare_models
from src.evaluation.visualizer import ResultVisualizer

logger = logging.getLogger(__name__)


def collect_data(config: dict, use_real_data: bool = False) -> dict[str, pd.DataFrame]:
    """
    Stage 1: Data Collection
    Collects market, on-chain, and sentiment data.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: DATA COLLECTION")
    logger.info("=" * 60)

    stablecoins = config["data"]["stablecoins"]
    start_date = config["data"]["time_range"]["start"]
    end_date = config["data"]["time_range"]["end"]

    all_data = {}

    for coin in stablecoins:
        coin_name = coin["symbol"].split("/")[0]
        logger.info(f"\n--- Collecting data for {coin_name} ---")

        # Market data
        if use_real_data:
            collector = MarketDataCollector(config["data"]["market"]["exchange"])
            market_df = collector.fetch_ohlcv(
                coin["symbol"], "1h", start_date, end_date
            )
        else:
            logger.info(f"Generating synthetic market data for {coin_name}")
            volatility = 0.002 if coin_name in ["USDT", "USDC"] else 0.005
            market_df = generate_synthetic_market_data(
                start_date, end_date, "1h",
                base_price=1.0, volatility=volatility,
            )

        # On-chain data
        onchain_collector = OnChainDataCollector()
        onchain_df = onchain_collector.fetch_all_metrics(
            contract_address="",
            start_date=start_date,
            end_date=end_date,
        )

        # Sentiment data
        sentiment_collector = SentimentDataCollector()
        sentiment_df = sentiment_collector.collect_twitter_data(
            keywords=config["data"]["sentiment"]["keywords"],
            start_date=start_date,
            end_date=end_date,
        )

        all_data[coin_name] = {
            "market": market_df,
            "onchain": onchain_df,
            "sentiment": sentiment_df,
        }

        logger.info(
            f"  Market: {market_df.shape}, "
            f"On-chain: {onchain_df.shape}, "
            f"Sentiment: {sentiment_df.shape}"
        )

    # Save raw data
    raw_dir = config["paths"]["raw_data"]
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    for coin_name, data in all_data.items():
        for data_type, df in data.items():
            df.to_csv(f"{raw_dir}/{coin_name}_{data_type}.csv")

    logger.info(f"\nRaw data saved to {raw_dir}/")
    return all_data


def preprocess_data(
    all_data: dict, config: dict
) -> dict[str, pd.DataFrame]:
    """
    Stage 2: Data Preprocessing
    Cleans, aligns, and normalizes data.
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: DATA PREPROCESSING")
    logger.info("=" * 60)

    cleaner = DataCleaner(
        outlier_method=config["preprocessing"]["outlier_method"],
        iqr_multiplier=config["preprocessing"]["iqr_multiplier"],
    )
    aligner = TimeAligner(target_freq="1h")

    processed_data = {}

    for coin_name, data in all_data.items():
        logger.info(f"\n--- Preprocessing {coin_name} ---")

        market_df = data["market"]
        onchain_df = data["onchain"]
        sentiment_df = data["sentiment"]

        # Clean each dataset
        market_clean = cleaner.clean(market_df)
        onchain_clean = cleaner.handle_missing_values(onchain_df)

        # Keep only numeric sentiment columns
        sentiment_numeric = sentiment_df.select_dtypes(include=[np.number])
        sentiment_clean = cleaner.handle_missing_values(sentiment_numeric)

        # Align all datasets to hourly frequency
        combined = aligner.align(market_clean, onchain_clean, sentiment_clean)

        logger.info(f"  Combined shape after alignment: {combined.shape}")
        processed_data[coin_name] = combined

    return processed_data


def engineer_features(
    processed_data: dict[str, pd.DataFrame], config: dict
) -> dict[str, pd.DataFrame]:
    """
    Stage 3: Feature Engineering
    Creates derived features for model input.
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: FEATURE ENGINEERING")
    logger.info("=" * 60)

    engineer = FeatureEngineer(config.get("features", {}))
    featured_data = {}

    for coin_name, df in processed_data.items():
        logger.info(f"\n--- Engineering features for {coin_name} ---")
        featured_df = engineer.engineer_all_features(df, price_col="close")
        featured_data[coin_name] = featured_df
        logger.info(f"  Features shape: {featured_df.shape}")

    # Save processed data
    processed_dir = config["paths"]["processed_data"]
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    for coin_name, df in featured_data.items():
        df.to_csv(f"{processed_dir}/{coin_name}_features.csv")

    return featured_data


def train_and_evaluate(
    featured_data: dict[str, pd.DataFrame], config: dict
) -> dict:
    """
    Stage 4 & 5: Model Training and Evaluation
    Trains Deep-XAI-Stable and baseline models, then evaluates.

    NOTE: SHAP is imported ONLY after all training is complete to avoid
    TensorFlow gradient registry conflicts.
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 4: MODEL TRAINING & EVALUATION")
    logger.info("=" * 60)

    import tensorflow as tf

    model_config = config.get("model", {})
    train_config = config.get("training", {})
    sequence_length = model_config.get("sequence_length", 48)
    target_column = "peg_deviation"

    all_results = {}
    # Store data needed for SHAP analysis after all training
    shap_queue = []
    visualizer = ResultVisualizer(config["paths"]["plots"])

    for coin_name, df in featured_data.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"Training models for {coin_name}")
        logger.info(f"{'='*40}")

        # Determine feature columns (exclude target-related columns)
        target_cols = ["peg_deviation", "peg_deviation_pct", "abs_peg_deviation"]
        feature_columns = [c for c in df.columns if c not in target_cols]

        # Create sequences
        seq_creator = SequenceDataset(
            sequence_length=sequence_length,
            prediction_horizon=model_config.get("prediction_horizon", 1),
            target_column=target_column,
        )
        X, y = seq_creator.create_sequences(df, feature_columns)

        n_features = X.shape[2]
        logger.info(f"  Input shape: {X.shape}, Target shape: {y.shape}")

        # Split data chronologically
        splitter = DataSplitter(
            train_ratio=config["split"]["train"],
            val_ratio=config["split"]["validation"],
            test_ratio=config["split"]["test"],
        )
        X_train, y_train, X_val, y_val, X_test, y_test = splitter.split_arrays(X, y)

        # Normalize using training data only (prevent data leakage)
        normalizer = DataNormalizer(method=config["preprocessing"]["normalization"])
        train_df_for_norm = df.iloc[: int(len(df) * config["split"]["train"])]
        normalizer.fit(train_df_for_norm[feature_columns])

        # --- Train Deep-XAI-Stable (Proposed Model) ---
        logger.info("\n--- Training Deep-XAI-Stable ---")
        from src.models.deep_xai_stable import build_functional_model

        model = build_functional_model(
            n_features=n_features,
            sequence_length=sequence_length,
            config={
                **model_config,
                "learning_rate": train_config.get("optimizer", {}).get("learning_rate", 0.001),
            },
        )

        trainer = ModelTrainer(
            model=model,
            output_dir=config["paths"]["models"],
            model_name=f"deep_xai_stable_{coin_name}",
        )

        history = trainer.train(
            X_train, y_train, X_val, y_val,
            epochs=train_config.get("epochs", 200),
            batch_size=train_config.get("batch_size", 64),
        )

        # Predict
        y_pred_proposed = model.predict(X_test, verbose=0).flatten()

        # Plot training history
        visualizer.plot_training_history(
            history, filename=f"{coin_name}_training_history.png"
        )

        # --- Train Baseline Models ---
        predictions = {"Deep-XAI-Stable": y_pred_proposed}

        # Simple LSTM
        logger.info("\n--- Training Simple LSTM ---")
        from src.models.baseline_models import build_simple_lstm

        lstm_simple = build_simple_lstm(n_features, sequence_length)
        lstm_simple.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=train_config.get("epochs", 200),
            batch_size=train_config.get("batch_size", 64),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=10, restore_best_weights=True
                ),
            ],
            verbose=0,
        )
        predictions["Simple_LSTM"] = lstm_simple.predict(X_test, verbose=0).flatten()

        # GRU
        logger.info("--- Training GRU ---")
        from src.models.baseline_models import build_gru_model

        gru_model = build_gru_model(n_features, sequence_length)
        gru_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=train_config.get("epochs", 200),
            batch_size=train_config.get("batch_size", 64),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=10, restore_best_weights=True
                ),
            ],
            verbose=0,
        )
        predictions["GRU"] = gru_model.predict(X_test, verbose=0).flatten()

        # ARIMA
        logger.info("--- Training ARIMA ---")
        from src.models.baseline_models import ARIMABaseline

        arima = ARIMABaseline(order=(5, 1, 0))
        train_target = y_train.flatten()
        arima.fit(train_target)
        predictions["ARIMA"] = arima.predict(len(y_test))

        # --- Evaluate All Models ---
        logger.info("\n--- Evaluation ---")
        results = compare_models(y_test, predictions, "Deep-XAI-Stable")
        all_results[coin_name] = results

        # Visualize
        visualizer.plot_predictions_vs_actual(
            y_test, y_pred_proposed,
            model_name=f"Deep-XAI-Stable ({coin_name})",
            filename=f"{coin_name}_predictions.png",
        )
        visualizer.plot_model_comparison(
            results, filename=f"{coin_name}_model_comparison.png"
        )
        visualizer.plot_error_distribution(
            y_test, y_pred_proposed,
            model_name=f"Deep-XAI-Stable ({coin_name})",
            filename=f"{coin_name}_error_distribution.png",
        )

        # Save model
        trainer.save_model(f"deep_xai_stable_{coin_name}_final.keras")

        # Queue SHAP analysis for after all training is done
        shap_queue.append({
            "coin_name": coin_name,
            "model": model,
            "X_train": X_train,
            "X_test": X_test,
            "y_pred": y_pred_proposed,
            "feature_columns": feature_columns,
        })

    # --- SHAP Explainability (after ALL training is complete) ---
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 5: XAI ANALYSIS (SHAP)")
    logger.info("=" * 60)

    for item in shap_queue:
        coin_name = item["coin_name"]
        logger.info(f"\n--- SHAP analysis for {coin_name} ---")
        try:
            import shap
            from src.explainability.shap_explainer import SHAPExplainer

            explainer = SHAPExplainer(
                model=item["model"],
                feature_names=item["feature_columns"],
                output_dir=config["paths"]["plots"],
            )

            # Use KernelExplainer to avoid TF gradient conflicts
            n_bg = min(config["xai"].get("num_background_samples", 100), len(item["X_train"]))
            bg_indices = np.random.choice(len(item["X_train"]), n_bg, replace=False)
            background = item["X_train"][bg_indices]

            # Use KernelExplainer with a wrapper function (most compatible)
            predict_fn = lambda x: item["model"].predict(x, verbose=0).flatten()
            bg_mean = background.mean(axis=0, keepdims=True)
            kernel_explainer = shap.KernelExplainer(
                predict_fn,
                bg_mean,
            )

            # Compute SHAP values for a small subset of test samples
            n_explain = min(50, len(item["X_test"]))
            shap_values = kernel_explainer.shap_values(item["X_test"][:n_explain])
            explainer.shap_values = shap_values

            # Feature importance
            explainer.plot_feature_importance(
                filename=f"{coin_name}_shap_importance.png"
            )

            # Local explanation
            local_exp = explainer.explain_single_prediction(
                item["X_test"][0],
                float(item["y_pred"][0]),
            )
            logger.info(f"  Local explanation: {local_exp.get('explanation_text', 'N/A')}")

            # Save feature importance
            importance_df = explainer.get_feature_importance()
            importance_df.to_csv(
                f"{config['paths']['results']}/{coin_name}_feature_importance.csv",
                index=False,
            )

        except Exception as e:
            logger.warning(f"SHAP analysis failed for {coin_name}: {e}")
            logger.warning("Continuing without XAI analysis...")

    return all_results


def save_results(results: dict, config: dict):
    """Save all results to files."""
    results_dir = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Save as JSON
    serializable = json.loads(json.dumps(results, default=convert))
    with open(results_dir / "all_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # Print summary
    from src.utils.helpers import print_summary
    for coin_name, coin_results in results.items():
        print(f"\n{'='*60}")
        print(f"  Results for {coin_name}")
        print(f"{'='*60}")
        for model_name, metrics in coin_results.items():
            if model_name != "diebold_mariano" and isinstance(metrics, dict):
                print_summary(metrics, model_name)

        # Print DM test results
        if "diebold_mariano" in coin_results:
            print("\n  Diebold-Mariano Tests:")
            for comparison, dm in coin_results["diebold_mariano"].items():
                print(
                    f"    {comparison}: "
                    f"stat={dm['dm_statistic']:.4f}, "
                    f"p={dm['p_value']:.4f} {dm['significance']}"
                )


def main():
    """Main entry point for the Deep-XAI-Stable pipeline."""
    parser = argparse.ArgumentParser(
        description="Deep-XAI-Stable: Stablecoin Price Prediction Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "collect", "preprocess", "train", "evaluate"],
        default="full",
        help="Pipeline mode",
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Use real API data instead of synthetic",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Setup
    config = load_config(args.config)
    setup_logging(config["paths"]["logs"])
    set_random_seed(args.seed)
    create_output_dirs(config)

    logger.info("=" * 60)
    logger.info("  Deep-XAI-Stable Pipeline")
    logger.info("  Stablecoin Price Prediction with Explainability")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Real data: {args.use_real_data}")

    # Run pipeline
    if args.mode in ["full", "collect"]:
        all_data = collect_data(config, args.use_real_data)

    if args.mode in ["full", "preprocess"]:
        if args.mode == "preprocess":
            # Load saved raw data
            all_data = load_saved_data(config["paths"]["raw_data"])
        processed_data = preprocess_data(all_data, config)

    if args.mode in ["full", "train", "evaluate"]:
        if args.mode in ["train", "evaluate"]:
            # Load saved processed data
            processed_data = load_saved_features(config["paths"]["processed_data"])
        else:
            processed_data = engineer_features(processed_data, config)

        results = train_and_evaluate(processed_data, config)
        save_results(results, config)

    logger.info("\n" + "=" * 60)
    logger.info("  Pipeline Complete!")
    logger.info("=" * 60)


def load_saved_data(raw_dir: str) -> dict:
    """Load previously saved raw data."""
    data = {}
    raw_path = Path(raw_dir)
    for market_file in raw_path.glob("*_market.csv"):
        coin_name = market_file.stem.replace("_market", "")
        data[coin_name] = {
            "market": pd.read_csv(market_file, index_col=0, parse_dates=True),
            "onchain": pd.read_csv(
                raw_path / f"{coin_name}_onchain.csv", index_col=0, parse_dates=True
            ),
            "sentiment": pd.read_csv(
                raw_path / f"{coin_name}_sentiment.csv", index_col=0, parse_dates=True
            ),
        }
    return data


def load_saved_features(processed_dir: str) -> dict:
    """Load previously saved feature-engineered data."""
    data = {}
    processed_path = Path(processed_dir)
    for feat_file in processed_path.glob("*_features.csv"):
        coin_name = feat_file.stem.replace("_features", "")
        data[coin_name] = pd.read_csv(feat_file, index_col=0, parse_dates=True)
    return data


if __name__ == "__main__":
    main()
