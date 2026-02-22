"""
Utility functions and helpers.
"""

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml


def setup_logging(log_dir: str = "outputs/logs", level: str = "INFO"):
    """Configure logging for the project."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / "experiment.log"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_random_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    os.environ["PYTHONHASHSEED"] = str(seed)


def create_output_dirs(config: dict):
    """Create all output directories specified in config."""
    paths = config.get("paths", {})
    for key, path in paths.items():
        Path(path).mkdir(parents=True, exist_ok=True)


def print_summary(results: dict, model_name: str = "Deep-XAI-Stable"):
    """Print a formatted summary of model results."""
    print("\n" + "=" * 60)
    print(f"  Model Evaluation Summary: {model_name}")
    print("=" * 60)

    for metric, value in results.items():
        if isinstance(value, float):
            print(f"  {metric:25s}: {value:.6f}")
        elif isinstance(value, dict):
            print(f"\n  {metric}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"    {k:23s}: {v:.6f}")
                else:
                    print(f"    {k:23s}: {v}")

    print("=" * 60 + "\n")
