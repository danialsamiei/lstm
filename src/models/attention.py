"""
Attention Mechanism Module
Implements self-attention for weighting temporal features.
Based on thesis section 3-7-2 (Attention Mechanism).

Equations:
    u_t = tanh(W_w * h_t + b_w)
    alpha_t = exp(u_t) / sum(exp(u_t'))
    c = sum(alpha_t * h_t)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AttentionLayer(layers.Layer):
    """
    Self-Attention layer for sequence models.
    Computes attention weights over the temporal dimension,
    allowing the model to focus on important time steps.
    """

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # W_w and b_w for computing u_t = tanh(W_w * h_t + b_w)
        self.W = self.add_weight(
            name="attention_weight",
            shape=(int(input_shape[-1]), self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        # Context vector for computing attention scores
        self.u = self.add_weight(
            name="context_vector",
            shape=(self.units,),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, return_attention=False):
        """
        Args:
            inputs: Tensor of shape (batch_size, time_steps, features)
            return_attention: Whether to return attention weights

        Returns:
            context_vector: Weighted sum of inputs (batch_size, features)
            attention_weights: (optional) (batch_size, time_steps)
        """
        # u_t = tanh(W * h_t + b)
        u = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)

        # alpha_t = softmax(u_t . u)
        score = tf.tensordot(u, self.u, axes=1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context = sum(alpha_t * h_t)
        context_vector = tf.reduce_sum(
            inputs * tf.expand_dims(attention_weights, -1), axis=1
        )

        if return_attention:
            return context_vector, attention_weights
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class MultiHeadAttentionBlock(layers.Layer):
    """
    Multi-Head Attention block for more expressive attention patterns.
    Uses TensorFlow's built-in MultiHeadAttention with residual connection.
    """

    def __init__(self, num_heads: int = 4, key_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim

    def build(self, input_shape):
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
        )
        self.layernorm = layers.LayerNormalization()
        self.dense = layers.Dense(int(input_shape[-1]))
        super().build(input_shape)

    def call(self, inputs):
        # Self-attention
        attn_output = self.mha(inputs, inputs)
        # Residual connection + layer norm
        output = self.layernorm(inputs + attn_output)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
        })
        return config
