"""
CNN Model Module for AI-Based Multimodal Deepfake Video Detection.

Defines the deep learning architecture: CNN backbone + Bi-LSTM + Attention
for temporal deepfake video detection. Supports ResNet50, EfficientNetB0,
and MobileNetV3Small as CNN feature extractors.

Classes:
    AttentionLayer: Custom attention mechanism for temporal feature aggregation.

Functions:
    build_cnn_backbone: Constructs and returns a frozen CNN feature extractor.
    build_deepfake_model: Assembles the full CNN + Bi-LSTM + Attention model.
    get_model_summary: Returns a string representation of the model summary.
    count_parameters: Returns parameter count statistics.
"""

import logging

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    EfficientNetB0,
    MobileNetV3Small,
    ResNet50,
)

from config import (
    CNN_BACKBONES,
    DEFAULT_BACKBONE,
    DENSE_UNITS,
    DROPOUT_RATE,
    LEARNING_RATE,
    LSTM_UNITS,
)

logger = logging.getLogger(__name__)

# Registry mapping backbone names to their tf.keras.applications class and
# the default input size expected by the pretrained weights.
_BACKBONE_REGISTRY = {
    "resnet50": {
        "class": ResNet50,
        "preprocess": tf.keras.applications.resnet50.preprocess_input,
    },
    "efficientnetb0": {
        "class": EfficientNetB0,
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
    },
    "mobilenetv3": {
        "class": MobileNetV3Small,
        "preprocess": tf.keras.applications.mobilenet_v3.preprocess_input,
    },
}


# ---------------------------------------------------------------------------
# Attention Layer
# ---------------------------------------------------------------------------


class AttentionLayer(tf.keras.layers.Layer):
    """Custom additive attention layer for temporal feature aggregation.

    Computes a learned attention weight for each time-step in the input
    sequence using a single dense projection followed by softmax
    normalisation over the time dimension. The output *context vector* is
    the weighted sum of the input sequence.

    Parameters
    ----------
    units : int
        Dimensionality of the attention hidden projection. Must be a
        positive integer.

    Examples
    --------
    >>> attn = AttentionLayer(units=64)
    >>> context, weights = attn(sequence)  # sequence shape: (batch, timesteps, features)
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        if units <= 0:
            raise ValueError(f"AttentionLayer units must be > 0, got {units}")
        self.units = units

    def build(self, input_shape):
        """Create the attention weight matrices.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the input tensor, expected to be 3-D
            ``(batch_size, timesteps, features)``.
        """
        if len(input_shape) != 3:
            raise ValueError(
                f"AttentionLayer expects a 3-D input (batch, timesteps, features), "
                f"got shape {input_shape}"
            )

        feature_dim = input_shape[-1]

        # W maps input features → attention hidden space
        self.W = self.add_weight(
            name="attention_W",
            shape=(feature_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )

        # bias for the hidden projection
        self.b = self.add_weight(
            name="attention_b",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

        # v computes the scalar score from the hidden projection
        self.v = self.add_weight(
            name="attention_v",
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs):
        """Compute attention-weighted context vector and attention weights.

        The computation follows:
            1. Project:  hidden = tanh(inputs @ W + b)   shape: (batch, T, units)
            2. Score:   score = hidden @ v                 shape: (batch, T, 1)
            3. Weights: alpha = softmax(score, axis=1)       shape: (batch, T, 1)
            4. Context: context = sum(alpha * inputs, axis=1) shape: (batch, features)

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor of shape ``(batch_size, timesteps, features)``.

        Returns
        -------
        context_vector : tf.Tensor
            Weighted sum of input features, shape ``(batch_size, features)``.
        attention_weights : tf.Tensor
            Softmax attention weights, shape ``(batch_size, timesteps, 1)``.
        """
        # Step 1 & 2: hidden projection → scalar scores
        # hidden: (batch, T, units)
        hidden = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)

        # score: (batch, T, 1)
        score = tf.tensordot(hidden, self.v, axes=1)

        # Step 3: softmax over time dimension (axis=1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # Step 4: weighted sum → context vector
        # attention_weights broadcasts over inputs
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)

        return context_vector, attention_weights

    def compute_output_shape(self, input_shape):
        """Return the output shapes for ``call``.

        Parameters
        ----------
        input_shape : tuple or tf.TensorShape
            Input shape ``(batch_size, timesteps, features)``.

        Returns
        -------
        list[tuple]
            ``[(batch_size, features), (batch_size, timesteps, 1)]``
        """
        batch_size = input_shape[0] if input_shape[0] is not None else None
        timesteps = input_shape[1] if input_shape[1] is not None else None
        features = input_shape[2]
        return [
            tf.TensorShape([batch_size, features]),
            tf.TensorShape([batch_size, timesteps, 1]),
        ]

    def get_config(self):
        """Return a serialisable configuration dictionary."""
        config = super().get_config()
        config.update({"units": self.units})
        return config


# ---------------------------------------------------------------------------
# CNN Backbone Builder
# ---------------------------------------------------------------------------


def build_cnn_backbone(backbone_name: str, input_shape: tuple) -> tf.keras.Model:
    """Build a frozen CNN feature extractor from pretrained ImageNet weights.

    The returned model maps a single frame ``(H, W, 3)`` to a compact
    feature vector via global average pooling, with all base layers
    frozen (``trainable = False``) so that only the downstream temporal
    layers are trained initially.

    Supported backbones (case-insensitive):
        * ``"resnet50"``       – ResNet50
        * ``"efficientnetb0"`` – EfficientNetB0
        * ``"mobilenetv3"``    – MobileNetV3Small

    Parameters
    ----------
    backbone_name : str
        Key identifying the CNN architecture. Must be one of the keys in
        ``config.CNN_BACKBONES``.
    input_shape : tuple of int
        Spatial input shape ``(height, width, channels)`` for a single
        frame, e.g. ``(224, 224, 3)``.

    Returns
    -------
    tf.keras.Model
        A ``tf.keras.Model`` that accepts ``(H, W, 3)`` input and
        produces a feature vector of size equal to the backbone's final
        convolutional output channels.

    Raises
    ------
    ValueError
        If *backbone_name* is not recognised.
    """
    backbone_key = backbone_name.lower().strip()

    if backbone_key not in _BACKBONE_REGISTRY:
        supported = ", ".join(sorted(_BACKBONE_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. "
            f"Supported backbones: {supported}"
        )

    registry_entry = _BACKBONE_REGISTRY[backbone_key]
    backbone_class = registry_entry["class"]

    logger.info(
        "Building CNN backbone '%s' with input shape %s",
        backbone_key,
        input_shape,
    )

    # Load pretrained weights (ImageNet) without the classification head
    base_model = backbone_class(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
        pooling=None,  # we add GlobalAveragePooling2D manually for clarity
    )

    # Freeze all base layers – they serve as a fixed feature extractor
    base_model.trainable = False

    # Build the full backbone model: input → base → global avg pool → output
    inputs = layers.Input(shape=input_shape, name="frame_input")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    backbone_model = tf.keras.Model(
        inputs=inputs,
        outputs=x,
        name=f"{backbone_key}_backbone",
    )

    logger.info(
        "CNN backbone '%s' built successfully. Output feature dim: %d",
        backbone_key,
        backbone_model.output_shape[-1],
    )

    return backbone_model


# ---------------------------------------------------------------------------
# Full Deepfake Detection Model
# ---------------------------------------------------------------------------


def build_deepfake_model(
    backbone_name: str = DEFAULT_BACKBONE,
    input_shape: tuple = (224, 224, 3),
    sequence_length: int = 30,
    lstm_units: int = LSTM_UNITS,
    dropout_rate: float = DROPOUT_RATE,
) -> tf.keras.Model:
    """Build and compile the full CNN + Bi-LSTM + Attention deepfake model.

    Architecture overview::

        Input (batch, T, H, W, 3)
          │
          ▼
        TimeDistributed(CNN Backbone)   ← frozen ImageNet feature extractor
          │  → (batch, T, D)             D = feature dim of backbone
          ▼
        Bidirectional(LSTM)             ← captures forward & backward temporal context
          │  → (batch, T, 2*units)
          ▼
        Attention Layer                  ← learns which frames matter most
          │  → (batch, 2*units)
          ▼
        Dense(64, relu)
          │
        Dropout(0.5)
          │
        Dense(1, sigmoid)                ← deepfake probability

    The model is compiled with:
        * Optimiser: Adam (lr=1e-4)
        * Loss: binary cross-entropy
        * Metrics: accuracy, AUC, Precision, Recall

    Parameters
    ----------
    backbone_name : str
        CNN backbone identifier (``"resnet50"``, ``"efficientnetb0"``,
        or ``"mobilenetv3"``). Defaults to ``config.DEFAULT_BACKBONE``.
    input_shape : tuple of int
        Single-frame spatial shape ``(H, W, C)``. Default ``(224, 224, 3)``.
    sequence_length : int
        Number of frames per video clip. Default ``30``.
    lstm_units : int
        Number of units in each direction of the Bi-LSTM. Default from
        ``config.LSTM_UNITS`` (128).
    dropout_rate : float
        Dropout probability applied after the dense layer. Default from
        ``config.DROPOUT_RATE`` (0.5).

    Returns
    -------
    tf.keras.Model
        Compiled Keras model ready for training.
    """
    logger.info(
        "Building deepfake model: backbone=%s, input_shape=%s, "
        "sequence_length=%d, lstm_units=%d, dropout_rate=%.2f",
        backbone_name,
        input_shape,
        sequence_length,
        lstm_units,
        dropout_rate,
    )

    # ---- 1. CNN Backbone ----
    cnn_backbone = build_cnn_backbone(backbone_name, input_shape)
    feature_dim = cnn_backbone.output_shape[-1]  # e.g. 1280 for EfficientNetB0
    logger.debug("CNN backbone output feature dimension: %d", feature_dim)

    # ---- 2. TimeDistributed wrapper ----
    # Wraps the per-frame CNN so it processes every frame in the sequence.
    frame_input = layers.Input(
        shape=(sequence_length, *input_shape),
        name="video_sequence_input",
    )
    x = layers.TimeDistributed(cnn_backbone, name="td_cnn")(frame_input)
    # x shape: (batch, sequence_length, feature_dim)

    # ---- 3. Bidirectional LSTM ----
    x = layers.Bidirectional(
        layers.LSTM(
            units=lstm_units,
            return_sequences=True,
            name="lstm",
        ),
        name="bidirectional_lstm",
    )(x)
    # x shape: (batch, sequence_length, 2 * lstm_units)

    # ---- 4. Attention Layer ----
    attention_layer = AttentionLayer(
        units=lstm_units, name="attention"
    )
    # Attention receives (batch, T, 2*units) and returns
    #   context_vector: (batch, 2*units)
    #   attention_weights: (batch, T, 1)
    context_vector, attention_weights = attention_layer(x)
    # We only use context_vector for downstream classification;
    # attention_weights are accessible as a secondary output if needed.

    # ---- 5. Classification head ----
    x = layers.Dense(
        DENSE_UNITS, activation="relu", name="dense_fc"
    )(context_vector)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    output = layers.Dense(1, activation="sigmoid", name="deepfake_output")(x)

    # ---- 6. Assemble model ----
    model = tf.keras.Model(
        inputs=frame_input,
        outputs=output,
        name=f"deepfake_{backbone_name.lower()}_model",
    )

    # ---- 7. Compile ----
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    logger.info(
        "Deepfake model compiled successfully. Name: '%s'", model.name
    )

    return model


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def get_model_summary(model: tf.keras.Model) -> str:
    """Return the Keras model summary as a string.

    Keras ``model.summary()`` prints directly to stdout by default. This
    utility captures that output and returns it so callers can log or
    display it programmatically.

    Parameters
    ----------
    model : tf.keras.Model
        A compiled or uncompiled Keras model.

    Returns
    -------
    str
        Multi-line string representation of the model architecture.
    """
    string_buffer = []

    # Redirect stdout to capture the summary
    class _StringIO:
        """Minimal string buffer with a ``write`` method for stdout redirect."""

        def __init__(self):
            self.contents = []

        def write(self, s):
            self.contents.append(s)

        def getvalue(self):
            return "".join(self.contents)

    buf = _StringIO()

    # Temporarily redirect sys.stdout
    import sys

    original_stdout = sys.stdout
    sys.stdout = buf
    try:
        model.summary()
    finally:
        sys.stdout = original_stdout

    summary_str = buf.getvalue()
    logger.debug("Model summary generated (%d chars).", len(summary_str))
    return summary_str


def count_parameters(model: tf.keras.Model) -> dict:
    """Count total, trainable, and non-trainable parameters in a model.

    Parameters
    ----------
    model : tf.keras.Model
        A Keras model.

    Returns
    -------
    dict
        Dictionary with three integer keys:
        ``"total"``, ``"trainable"``, ``"non_trainable"``.
    """
    total = int(model.count_params())
    trainable = sum(int(w.numpy().size) for w in model.trainable_weights)
    non_trainable = sum(int(w.numpy().size) for w in model.non_trainable_weights)

    # Safety check: trainable + non_trainable should equal total
    if trainable + non_trainable != total:
        logger.warning(
            "Parameter count mismatch: trainable (%d) + non_trainable (%d) != total (%d). "
            "Falling back to model.count_params() arithmetic.",
            trainable,
            non_trainable,
            total,
        )
        non_trainable = total - trainable

    result = {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }

    logger.info(
        "Model parameter counts – total: %s | trainable: %s | non-trainable: %s",
        f"{total:,}",
        f"{trainable:,}",
        f"{non_trainable:,}",
    )

    return result
