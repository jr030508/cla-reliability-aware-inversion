from __future__ import annotations

import numpy as np
import tensorflow as tf


class FixedGraphPropagation(tf.keras.layers.Layer):
    """Propagate station measurements with a fixed normalized adjacency matrix."""

    def __init__(self, adjacency, **kwargs):
        super().__init__(**kwargs)
        self.adjacency_array = np.asarray(adjacency, dtype=np.float32)

    def build(self, input_shape):
        self.adjacency = self.add_weight(
            name="adjacency",
            shape=self.adjacency_array.shape,
            initializer=tf.keras.initializers.Constant(self.adjacency_array),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs):
        propagated = tf.einsum("ij,btj->bti", self.adjacency, inputs)
        return tf.stack([inputs, propagated], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"adjacency": self.adjacency_array.tolist()})
        return config


def temporal_attention_pool(x, name: str):
    score = tf.keras.layers.Dense(1, activation="tanh", name=f"{name}_score")(x)
    weight = tf.keras.layers.Softmax(axis=1, name=f"{name}_weights")(score)
    weighted = tf.keras.layers.Multiply(name=f"{name}_weighted")([x, weight])
    return tf.keras.layers.GlobalAveragePooling1D(name=f"{name}_pool")(weighted)


def output_head(x, output_dim: int, params: dict):
    x = tf.keras.layers.Dense(params["dense_1_units"], activation="relu", name="head_dense_1")(x)
    x = tf.keras.layers.Dropout(params["dropout"], name="head_dropout")(x)
    x = tf.keras.layers.Dense(params["dense_2_units"], activation="relu", name="head_dense_2")(x)
    return tf.keras.layers.Dense(output_dim, activation="linear", name="normalized_targets")(x)


def conv_encoder(inputs, input_shape, params, station_kernel_width: int = 3):
    x = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1), name="station_time_image")(inputs)
    x = tf.keras.layers.Conv2D(
        params["filters_1"], (3, station_kernel_width), padding="same", activation="relu", name="conv_1"
    )(x)
    x = tf.keras.layers.Conv2D(
        params["filters_2"], (3, station_kernel_width), padding="same", activation="relu", name="conv_2"
    )(x)
    return x


def build_model(
    model_name: str,
    input_shape: tuple[int, int],
    output_dim: int,
    params: dict,
    adjacency: np.ndarray | None = None,
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=input_shape, name="concentration")

    if model_name == "2D_CNN":
        x = conv_encoder(inputs, input_shape, params)
        x = tf.keras.layers.GlobalAveragePooling2D(name="cnn_pool")(x)

    elif model_name == "2D_CNN_Attention":
        x = conv_encoder(inputs, input_shape, params)
        x = tf.keras.layers.Reshape((input_shape[0], input_shape[1] * params["filters_2"]))(x)
        x = temporal_attention_pool(x, "temporal_attention")

    elif model_name == "LSTM":
        x = tf.keras.layers.LSTM(params["lstm_1_units"], return_sequences=True, name="lstm_1")(inputs)
        x = tf.keras.layers.Dropout(params["dropout"], name="lstm_dropout_1")(x)
        # The second layer consumes x, not the original input. This fixes the old disconnected branch.
        x = tf.keras.layers.LSTM(params["lstm_2_units"], return_sequences=False, name="lstm_2")(x)

    elif model_name == "LSTM_Attention":
        x = tf.keras.layers.LSTM(params["lstm_1_units"], return_sequences=True, name="lstm_1")(inputs)
        x = tf.keras.layers.Dropout(params["dropout"], name="lstm_dropout_1")(x)
        x = tf.keras.layers.LSTM(params["lstm_2_units"], return_sequences=True, name="lstm_2")(x)
        x = temporal_attention_pool(x, "temporal_attention")

    elif model_name == "CNN_LSTM":
        x = conv_encoder(inputs, input_shape, params)
        x = tf.keras.layers.Reshape((input_shape[0], input_shape[1] * params["filters_2"]))(x)
        x = tf.keras.layers.LSTM(params["lstm_1_units"], return_sequences=False, name="lstm_1")(x)

    elif model_name in {"CLA", "CLA_NoAttention", "CLA_NoStationMix"}:
        station_kernel_width = 1 if model_name == "CLA_NoStationMix" else 3
        x = conv_encoder(inputs, input_shape, params, station_kernel_width=station_kernel_width)
        x = tf.keras.layers.Reshape((input_shape[0], input_shape[1] * params["filters_2"]))(x)
        x = tf.keras.layers.LSTM(params["lstm_1_units"], return_sequences=True, name="lstm_1")(x)
        x = tf.keras.layers.Dropout(params["dropout"], name="lstm_dropout_1")(x)
        x = tf.keras.layers.LSTM(params["lstm_2_units"], return_sequences=True, name="lstm_2")(x)
        if model_name in {"CLA", "CLA_NoStationMix"}:
            x = temporal_attention_pool(x, "temporal_attention")
        else:
            # Strict attention ablation: every upstream layer is identical to CLA.
            # Only learned temporal attention is replaced by parameter-free mean pooling.
            x = tf.keras.layers.GlobalAveragePooling1D(name="temporal_mean_pool")(x)

    elif model_name == "Graph_CLA":
        if adjacency is None:
            raise ValueError("Graph_CLA requires a station adjacency matrix.")
        x = FixedGraphPropagation(adjacency, name="fixed_graph")(inputs)
        x = tf.keras.layers.Dense(params["graph_units"], activation="relu", name="graph_projection")(x)
        x = tf.keras.layers.Reshape((input_shape[0], input_shape[1] * params["graph_units"]))(x)
        x = tf.keras.layers.LSTM(params["lstm_1_units"], return_sequences=True, name="lstm_1")(x)
        x = tf.keras.layers.Dropout(params["dropout"], name="lstm_dropout_1")(x)
        x = tf.keras.layers.LSTM(params["lstm_2_units"], return_sequences=True, name="lstm_2")(x)
        x = temporal_attention_pool(x, "temporal_attention")

    else:
        raise ValueError(f"Unknown model: {model_name}")

    outputs = output_head(x, output_dim, params)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
