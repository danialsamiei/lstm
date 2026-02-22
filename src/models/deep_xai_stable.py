"""
Deep-XAI-Stable Model Architecture
Hybrid deep learning model combining LSTM, Attention, and XAI.
Based on thesis section 3-7.

Architecture:
    1. Feature Extraction Module (Dense layers with ReLU)
    2. Temporal Deep Learning Core (Stacked Bidirectional LSTM + Attention)
    3. Output layers (Dense -> Linear for regression)
"""

import logging
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from .attention import AttentionLayer, MultiHeadAttentionBlock

logger = logging.getLogger(__name__)


class DeepXAIStable(Model):
    """
    Deep-XAI-Stable: Hybrid model for stablecoin price prediction.

    Combines:
    - Dense feature extraction layers
    - Stacked Bidirectional LSTM for temporal patterns
    - Self-Attention mechanism for focusing on key time steps
    - Linear output for regression
    """

    def __init__(
        self,
        n_features: int,
        sequence_length: int = 48,
        lstm_units: list[int] = None,
        attention_units: int = 64,
        dense_units: list[int] = None,
        output_dense_units: list[int] = None,
        dropout: float = 0.2,
        recurrent_dropout: float = 0.1,
        bidirectional: bool = True,
        use_multi_head_attention: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_features = n_features
        self.sequence_length = sequence_length
        self.lstm_unit_list = lstm_units or [128, 64]
        self.attention_units = attention_units
        self.dense_unit_list = dense_units or [128, 64]
        self.output_dense_list = output_dense_units or [32, 16]
        self.dropout_rate = dropout
        self.recurrent_dropout_rate = recurrent_dropout
        self.bidirectional = bidirectional
        self.use_multi_head_attention = use_multi_head_attention

        # --- Feature Extraction Module (Section 3-7-1) ---
        self.feature_dense_layers = []
        self.feature_dropout_layers = []
        for units in self.dense_unit_list:
            self.feature_dense_layers.append(
                layers.Dense(units, activation="relu", name=f"feat_dense_{units}")
            )
            self.feature_dropout_layers.append(
                layers.Dropout(dropout, name=f"feat_dropout_{units}")
            )

        # Batch normalization after feature extraction
        self.feature_bn = layers.BatchNormalization(name="feat_bn")

        # --- Temporal Deep Learning Core (Section 3-7-2) ---
        self.lstm_layers = []
        for i, units in enumerate(self.lstm_unit_list):
            return_sequences = True  # All LSTM layers return sequences for attention
            lstm = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                name=f"lstm_{i}",
            )
            if bidirectional:
                lstm = layers.Bidirectional(lstm, name=f"bilstm_{i}")
            self.lstm_layers.append(lstm)

        # --- Attention Mechanism ---
        if use_multi_head_attention:
            self.attention = MultiHeadAttentionBlock(
                num_heads=4, key_dim=32, name="multi_head_attention"
            )
            # Need a pooling layer after multi-head attention
            self.attention_pool = AttentionLayer(
                units=attention_units, name="attention_pool"
            )
        else:
            self.attention = AttentionLayer(
                units=attention_units, name="self_attention"
            )

        # --- Output Layers (Section 3-7-2c) ---
        self.output_dense_layers = []
        self.output_dropout_layers = []
        for units in self.output_dense_list:
            self.output_dense_layers.append(
                layers.Dense(units, activation="relu", name=f"out_dense_{units}")
            )
            self.output_dropout_layers.append(
                layers.Dropout(dropout, name=f"out_dropout_{units}")
            )

        # Final output: single neuron with linear activation (regression)
        self.output_layer = layers.Dense(1, activation="linear", name="prediction")

    def call(self, inputs, training=False, return_attention=False):
        """
        Forward pass.

        Args:
            inputs: Tensor of shape (batch_size, sequence_length, n_features)
            training: Whether in training mode
            return_attention: Whether to return attention weights

        Returns:
            predictions: (batch_size, 1)
            attention_weights: (optional) (batch_size, sequence_length)
        """
        x = inputs

        # Feature extraction on each time step
        for dense, dropout in zip(self.feature_dense_layers, self.feature_dropout_layers):
            x = dense(x)
            x = dropout(x, training=training)
        x = self.feature_bn(x, training=training)

        # LSTM layers
        for lstm in self.lstm_layers:
            x = lstm(x, training=training)

        # Attention
        attention_weights = None
        if self.use_multi_head_attention:
            x = self.attention(x)
            if return_attention:
                x, attention_weights = self.attention_pool(x, return_attention=True)
            else:
                x = self.attention_pool(x)
        else:
            if return_attention:
                x, attention_weights = self.attention(x, return_attention=True)
            else:
                x = self.attention(x)

        # Output dense layers
        for dense, dropout in zip(self.output_dense_layers, self.output_dropout_layers):
            x = dense(x)
            x = dropout(x, training=training)

        # Final prediction
        output = self.output_layer(x)

        if return_attention:
            return output, attention_weights
        return output

    def get_config(self):
        return {
            "n_features": self.n_features,
            "sequence_length": self.sequence_length,
            "lstm_units": self.lstm_unit_list,
            "attention_units": self.attention_units,
            "dense_units": self.dense_unit_list,
            "output_dense_units": self.output_dense_list,
            "dropout": self.dropout_rate,
            "recurrent_dropout": self.recurrent_dropout_rate,
            "bidirectional": self.bidirectional,
            "use_multi_head_attention": self.use_multi_head_attention,
        }


def build_model(
    n_features: int,
    sequence_length: int = 48,
    config: dict = None,
) -> DeepXAIStable:
    """
    Build and compile the Deep-XAI-Stable model.

    Args:
        n_features: Number of input features
        sequence_length: Length of input sequences (lookback window)
        config: Model configuration dictionary

    Returns:
        Compiled Keras model
    """
    config = config or {}

    model = DeepXAIStable(
        n_features=n_features,
        sequence_length=sequence_length,
        lstm_units=config.get("lstm", {}).get("units", [128, 64]),
        attention_units=config.get("attention", {}).get("units", 64),
        dense_units=config.get("feature_extractor", {}).get("dense_units", [128, 64]),
        output_dense_units=config.get("output", {}).get("dense_units", [32, 16]),
        dropout=config.get("dropout", 0.2),
        recurrent_dropout=config.get("recurrent_dropout", 0.1),
        bidirectional=config.get("lstm", {}).get("bidirectional", True),
    )

    # Build model by passing a dummy input
    dummy_input = tf.zeros((1, sequence_length, n_features))
    _ = model(dummy_input)

    # Compile with Adam optimizer and MSE loss (Section 3-7-4)
    lr = config.get("learning_rate", 0.001)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )

    logger.info(f"Built Deep-XAI-Stable model: {model.count_params()} parameters")
    model.summary(print_fn=logger.info)

    return model


def build_functional_model(
    n_features: int,
    sequence_length: int = 48,
    config: dict = None,
) -> keras.Model:
    """
    Build model using Keras Functional API (alternative, for SHAP compatibility).
    """
    config = config or {}
    lstm_units = config.get("lstm", {}).get("units", [128, 64])
    dense_units = config.get("feature_extractor", {}).get("dense_units", [128, 64])
    output_units = config.get("output", {}).get("dense_units", [32, 16])
    dropout = config.get("dropout", 0.2)
    bidirectional = config.get("lstm", {}).get("bidirectional", True)
    attention_units = config.get("attention", {}).get("units", 64)

    # Input
    inputs = keras.Input(shape=(sequence_length, n_features), name="input")

    # Feature extraction
    x = inputs
    for units in dense_units:
        x = layers.TimeDistributed(layers.Dense(units, activation="relu"))(x)
        x = layers.Dropout(dropout)(x)
    x = layers.BatchNormalization()(x)

    # LSTM layers
    for i, units in enumerate(lstm_units):
        lstm = layers.LSTM(
            units,
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=0.1,
            name=f"lstm_{i}",
        )
        if bidirectional:
            lstm = layers.Bidirectional(lstm, name=f"bilstm_{i}")
        x = lstm(x)

    # Attention
    x = AttentionLayer(units=attention_units, name="attention")(x)

    # Output layers
    for units in output_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(dropout)(x)

    # Prediction
    outputs = layers.Dense(1, activation="linear", name="prediction")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Deep-XAI-Stable")

    lr = config.get("learning_rate", 0.001)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )

    return model
