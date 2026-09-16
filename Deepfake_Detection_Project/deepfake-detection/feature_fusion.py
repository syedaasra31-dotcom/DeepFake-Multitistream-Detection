"""
Feature Fusion Module for AI-Based Multimodal Deepfake Video Detection.

This module fuses CNN spatial features, lip synchronisation features, remote
photoplethysmography (rPPG) features, and temporal statistics into a unified
feature vector that can be consumed by downstream classifiers.

Functions:
    extract_cnn_features        - Extract deep feature embeddings from face
                                  images using a pretrained CNN backbone.
    normalize_features          - Z-score normalise feature arrays.
    compute_temporal_features   - Derive temporal statistics from a
                                  per-frame feature sequence.
    fuse_lip_rppg_features      - Merge lip-sync and rPPG scalar / summary
                                  statistics into a single vector.
    create_fusion_vector        - Assemble and optionally reduce the final
                                  multimodal fusion vector.
    build_feature_dataset       - Batch-process a list of per-video feature
                                  dicts into an (X, y) dataset.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow.keras.applications import (
        EfficientNetB0,
        MobileNetV3Small,
        ResNet50,
        efficientnet,
        mobilenet_v3,
        resnet50,
    )
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backbone registry: maps config name -> (model class, preprocess fn, output dim)
# ---------------------------------------------------------------------------
_BACKBONE_REGISTRY = {
    "resnet50": {
        "class": ResNet50 if TF_AVAILABLE else None,
        "preprocess": resnet50.preprocess_input if TF_AVAILABLE else None,
        "output_dim": 2048,
    },
    "efficientnetb0": {
        "class": EfficientNetB0 if TF_AVAILABLE else None,
        "preprocess": efficientnet.preprocess_input if TF_AVAILABLE else None,
        "output_dim": 1280,
    },
    "mobilenetv3": {
        "class": MobileNetV3Small if TF_AVAILABLE else None,
        "preprocess": mobilenet_v3.preprocess_input if TF_AVAILABLE else None,
        "output_dim": 576,
    },
}

# Cache for loaded backbone models to avoid redundant loading
_model_cache: Dict[str, "tf.keras.Model"] = {}


# ==========================================================================
# 1. CNN Feature Extraction
# ==========================================================================


def extract_cnn_features(
    face_images: np.ndarray,
    backbone: str = "efficientnetb0",
) -> np.ndarray:
    """Extract deep feature embeddings from a batch of aligned face images.

    Loads the specified pretrained CNN backbone **without** its classification
    head (``include_top=False``) and extracts the global average-pooled
    feature vector from the penultimate layer for every image in the batch.

    Supported backbones (via ``config.CNN_BACKBONES``):

    * ``'resnet50'``         – ResNet-50  (2 048-d)
    * ``'efficientnetb0'``   – EfficientNet-B0 (1 280-d)
    * ``'mobilenetv3'``      – MobileNetV3-Small (576-d)

    Parameters
    ----------
    face_images : numpy.ndarray
        Batch of face images with shape ``(N, H, W, 3)`` in **RGB** order
        and ``uint8`` or ``float32`` dtype.  Typical input size is
        ``(N, 224, 224, 3)``.
    backbone : str, optional
        Name of the CNN backbone.  Must be one of the keys in
        ``config.CNN_BACKBONES``.  Defaults to ``config.DEFAULT_BACKBONE``.

    Returns
    -------
    numpy.ndarray
        Feature matrix of shape ``(N, D)`` where *D* is the backbone-
        specific feature dimension.  Dtype is ``float32``.

    Raises
    ------
    ValueError
        If *backbone* is not recognised, or if *face_images* does not have
        the expected number of dimensions.
    RuntimeError
        If TensorFlow is not available.
    """
    # --- Validate backbone name ---
    backbone = backbone.lower()
    if backbone not in config.CNN_BACKBONES:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Must be one of {config.CNN_BACKBONES}."
        )

    if not TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow is required for CNN feature extraction but is not "
            "available in this environment."
        )

    # --- Validate input shape ---
    face_images = np.asarray(face_images, dtype=np.float32)
    if face_images.ndim != 4 or face_images.shape[-1] != 3:
        raise ValueError(
            f"face_images must have shape (N, H, W, 3), got {face_images.shape}."
        )

    n_images = face_images.shape[0]
    logger.info(
        "Extracting CNN features for %d images using backbone '%s'.",
        n_images,
        backbone,
    )

    # --- Load / retrieve cached model ---
    if backbone not in _model_cache:
        logger.debug("Loading %s backbone (include_top=False, pooling='avg')...", backbone)
        registry_entry = _BACKBONE_REGISTRY[backbone]
        model = registry_entry["class"](
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(224, 224, 3),
        )
        _model_cache[backbone] = model
        logger.debug("Backbone '%s' loaded successfully.", backbone)
    else:
        logger.debug("Reusing cached '%s' backbone model.", backbone)

    model = _model_cache[backbone]
    preprocess_fn = _BACKBONE_REGISTRY[backbone]["preprocess"]
    expected_dim = _BACKBONE_REGISTRY[backbone]["output_dim"]

    # --- Preprocess: resize to 224x224 if needed, then apply model-specific preprocessing ---
    h, w = face_images.shape[1], face_images.shape[2]
    if h != 224 or w != 224:
        logger.debug("Resizing face images from (%d, %d) to (224, 224).", h, w)
        resized = tf.image.resize(face_images, [224, 224]).numpy()
    else:
        resized = face_images

    # Apply the backbone-specific preprocessing function
    preprocessed = preprocess_fn(resized)

    # --- Forward pass in batches to limit memory ---
    batch_size = config.BATCH_SIZE
    features_list: List[np.ndarray] = []

    for start_idx in range(0, n_images, batch_size):
        end_idx = min(start_idx + batch_size, n_images)
        batch = preprocessed[start_idx:end_idx]
        batch_features = model.predict(batch, verbose=0)
        features_list.append(batch_features)

    features = np.concatenate(features_list, axis=0).astype(np.float32)

    # --- Sanity check ---
    if features.shape[1] != expected_dim:
        logger.warning(
            "Backbone '%s' output dim is %d but expected %d. "
            "Using actual output dim.",
            backbone,
            features.shape[1],
            expected_dim,
        )

    logger.info(
        "CNN feature extraction complete: %d images -> %s.",
        n_images,
        features.shape,
    )
    return features


# ==========================================================================
# 2. Feature Normalisation
# ==========================================================================


def normalize_features(
    features: np.ndarray,
) -> Tuple[np.ndarray, StandardScaler]:
    """Apply Z-score normalisation to a feature array.

    Uses :class:`sklearn.preprocessing.StandardScaler` to centre the data to
    zero mean and unit variance along each feature dimension.

    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix of shape ``(N, D)`` or a 1-D vector of shape ``(D,)``.

    Returns
    -------
    tuple[numpy.ndarray, sklearn.preprocessing.StandardScaler]
        * **normalised** – Z-score normalised features with the same shape
          as the input.
        * **scaler** – The fitted :class:`StandardScaler` instance (useful
          for consistent transform of future data).
    """
    features = np.asarray(features, dtype=np.float64)

    # Handle 1-D input by adding a batch dimension
    was_1d = features.ndim == 1
    if was_1d:
        features = features.reshape(1, -1)

    scaler = StandardScaler()
    normalised = scaler.fit_transform(features)

    # Restore original dimensionality
    if was_1d:
        normalised = normalised.flatten()

    logger.debug(
        "Features normalised: input shape %s -> output shape %s.",
        features.shape,
        normalised.shape,
    )
    return normalised, scaler


# ==========================================================================
# 3. Temporal Feature Computation
# ==========================================================================


def compute_temporal_features(feature_sequence: np.ndarray) -> Dict[str, float]:
    """Compute descriptive temporal statistics from a per-frame feature sequence.

    Given a sequence of feature vectors extracted frame-by-frame (e.g. CNN
    features across all frames of a video), this function collapses the
    temporal dimension into a fixed-length set of summary statistics.

    The following statistics are computed **per feature dimension** and then
    averaged across dimensions, producing a single scalar per statistic:

    * mean, standard deviation, minimum, maximum, median
    * skewness (Fisher's, via :func:`scipy.stats.skew`)
    * kurtosis (Fisher's, excess, via :func:`scipy.stats.kurtosis`)
    * linear trend slope (ordinary least-squares)
    * first-order differences – mean and standard deviation
    * zero-crossing rate (fraction of sign changes)

    Parameters
    ----------
    feature_sequence : numpy.ndarray
        Either a 1-D array of shape ``(T,)`` or a 2-D array of shape
        ``(T, D)`` where *T* is the number of frames and *D* the per-frame
        feature dimensionality.

    Returns
    -------
    dict[str, float]
        Dictionary keyed by statistic name with scalar float values.
    """
    feature_sequence = np.asarray(feature_sequence, dtype=np.float64)

    if feature_sequence.ndim == 1:
        feature_sequence = feature_sequence.reshape(-1, 1)

    n_frames, n_dims = feature_sequence.shape

    if n_frames < 2:
        logger.warning(
            "Temporal features requested but sequence has only %d frame(s). "
            "Returning zero-valued statistics.",
            n_frames,
        )
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "trend_slope": 0.0,
            "diff_mean": 0.0,
            "diff_std": 0.0,
            "zero_crossing_rate": 0.0,
        }

    stats_per_dim: List[np.ndarray] = []

    for d in range(n_dims):
        col = feature_sequence[:, d]

        # Basic descriptive statistics
        col_mean = float(np.mean(col))
        col_std = float(np.std(col, ddof=1)) if n_frames > 1 else 0.0
        col_min = float(np.min(col))
        col_max = float(np.max(col))
        col_median = float(np.median(col))

        # Higher-order moments via scipy
        col_skew = float(scipy_stats.skew(col, bias=True))
        col_kurt = float(scipy_stats.kurtosis(col, fisher=True, bias=True))

        # Linear trend slope via polyfit (degree 1)
        x = np.arange(n_frames, dtype=np.float64)
        slope, _ = np.polyfit(x, col, 1)
        col_slope = float(slope)

        # First-order differences
        diffs = np.diff(col)
        col_diff_mean = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
        col_diff_std = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

        # Zero-crossing rate: count sign changes / (N - 1)
        sign_changes = np.sum(np.diff(np.sign(col)) != 0)
        col_zcr = float(sign_changes) / float(n_frames - 1)

        stats_per_dim.append(
            np.array([
                col_mean, col_std, col_min, col_max, col_median,
                col_skew, col_kurt, col_slope,
                col_diff_mean, col_diff_std, col_zcr,
            ])
        )

    # Average across dimensions
    all_stats = np.vstack(stats_per_dim)  # (D, 11)
    avg_stats = np.mean(all_stats, axis=0)  # (11,)

    result = {
        "mean": float(avg_stats[0]),
        "std": float(avg_stats[1]),
        "min": float(avg_stats[2]),
        "max": float(avg_stats[3]),
        "median": float(avg_stats[4]),
        "skewness": float(avg_stats[5]),
        "kurtosis": float(avg_stats[6]),
        "trend_slope": float(avg_stats[7]),
        "diff_mean": float(avg_stats[8]),
        "diff_std": float(avg_stats[9]),
        "zero_crossing_rate": float(avg_stats[10]),
    }

    logger.debug(
        "Temporal statistics computed over %d frames x %d dims: %s",
        n_frames, n_dims, {k: round(v, 4) for k, v in result.items()},
    )
    return result


# ==========================================================================
# 4. Lip + rPPG Feature Fusion
# ==========================================================================


def fuse_lip_rppg_features(
    lip_features: Dict[str, Union[np.ndarray, float, str, Dict]],
    rppg_features: Dict[str, Union[float, np.ndarray, Dict]],
) -> np.ndarray:
    """Concatenate lip-synchronisation and rPPG summary statistics.

    Builds a compact scalar feature vector from the high-level outputs of
    :func:`lip_sync.analyze_lip_sync` and :func:`rppg.extract_rppg_features`.

    **Lip-derived components** (in order):

    1. Lip sync score (1 value)
    2. Lip movement statistics: mean, std, max, min (4 values)
    3. Lip aspect ratio statistics: mean, std (2 values)
    4. Mouth opening statistics: mean, std (2 values)
    5. Temporal lip motion: dominant frequency, spectral energy (2 values)
    6. Audio feature statistics – MFCC means (13), spectral centroid mean/std (2),
       spectral bandwidth mean/std (2), ZCR mean/std (2), chroma means (12),
       mel-spectrogram means (128), audio energy mean/std (2) → 161 values

    **rPPG-derived components** (in order):

    7. Heart rate (1 value)
    8. Pulse stability (1 value)
    9. rPPG confidence (1 value)

    Parameters
    ----------
    lip_features : dict
        Output dictionary from :func:`lip_sync.analyze_lip_sync`.
    rppg_features : dict
        Output dictionary from :func:`rppg.extract_rppg_features`.

    Returns
    -------
    numpy.ndarray
        1-D feature vector of dtype ``float64``.
    """
    components: List[float] = []

    # ------------------------------------------------------------------
    # Lip sync score
    # ------------------------------------------------------------------
    lip_sync_score = float(lip_features.get("lip_sync_score", 0.0))
    components.append(lip_sync_score)
    logger.debug("fuse_lip_rppg: lip_sync_score = %.4f", lip_sync_score)

    # ------------------------------------------------------------------
    # Lip movement statistics
    # ------------------------------------------------------------------
    lip_movements = np.asarray(
        lip_features.get("lip_movements", np.array([])), dtype=np.float64
    )
    if lip_movements.size > 0:
        components.extend([
            float(np.mean(lip_movements)),
            float(np.std(lip_movements, ddof=1)) if lip_movements.size > 1 else 0.0,
            float(np.max(lip_movements)),
            float(np.min(lip_movements)),
        ])
    else:
        components.extend([0.0, 0.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    # Lip aspect ratio statistics
    # ------------------------------------------------------------------
    lip_aspect_ratios = np.asarray(
        lip_features.get("lip_aspect_ratios", np.array([])), dtype=np.float64
    )
    if lip_aspect_ratios.size > 0:
        components.extend([
            float(np.mean(lip_aspect_ratios)),
            float(np.std(lip_aspect_ratios, ddof=1)) if lip_aspect_ratios.size > 1 else 0.0,
        ])
    else:
        components.extend([0.0, 0.0])

    # ------------------------------------------------------------------
    # Mouth opening statistics
    # ------------------------------------------------------------------
    mouth_openings = np.asarray(
        lip_features.get("mouth_openings", np.array([])), dtype=np.float64
    )
    if mouth_openings.size > 0:
        components.extend([
            float(np.mean(mouth_openings)),
            float(np.std(mouth_openings, ddof=1)) if mouth_openings.size > 1 else 0.0,
        ])
    else:
        components.extend([0.0, 0.0])

    # ------------------------------------------------------------------
    # Temporal lip motion features
    # ------------------------------------------------------------------
    temporal_lip = lip_features.get("temporal_features", {})
    if isinstance(temporal_lip, dict) and temporal_lip:
        components.append(float(temporal_lip.get("dominant_frequency", 0.0)))
        components.append(float(temporal_lip.get("spectral_energy", 0.0)))
    else:
        components.extend([0.0, 0.0])

    # ------------------------------------------------------------------
    # Audio feature statistics
    # ------------------------------------------------------------------
    audio_features = lip_features.get("audio_features", {})
    if isinstance(audio_features, dict) and audio_features:
        # MFCC means (13 coefficients)
        mfccs = np.asarray(audio_features.get("mfccs", np.array([])), dtype=np.float64)
        if mfccs.ndim == 2 and mfccs.shape[0] > 0:
            components.extend(float(v) for v in np.mean(mfccs, axis=1))
        else:
            components.extend([0.0] * 13)

        # Spectral centroid: mean, std
        sc = np.asarray(audio_features.get("spectral_centroid", np.array([])), dtype=np.float64)
        sc_flat = sc.flatten()
        if sc_flat.size > 0:
            components.append(float(np.mean(sc_flat)))
            components.append(
                float(np.std(sc_flat, ddof=1)) if sc_flat.size > 1 else 0.0
            )
        else:
            components.extend([0.0, 0.0])

        # Spectral bandwidth: mean, std
        sb = np.asarray(audio_features.get("spectral_bandwidth", np.array([])), dtype=np.float64)
        sb_flat = sb.flatten()
        if sb_flat.size > 0:
            components.append(float(np.mean(sb_flat)))
            components.append(
                float(np.std(sb_flat, ddof=1)) if sb_flat.size > 1 else 0.0
            )
        else:
            components.extend([0.0, 0.0])

        # Zero-crossing rate: mean, std
        zcr = np.asarray(audio_features.get("zero_crossing_rate", np.array([])), dtype=np.float64)
        zcr_flat = zcr.flatten()
        if zcr_flat.size > 0:
            components.append(float(np.mean(zcr_flat)))
            components.append(
                float(np.std(zcr_flat, ddof=1)) if zcr_flat.size > 1 else 0.0
            )
        else:
            components.extend([0.0, 0.0])

        # Chroma means (12 bins)
        chroma = np.asarray(audio_features.get("chroma", np.array([])), dtype=np.float64)
        if chroma.ndim == 2 and chroma.shape[0] > 0:
            components.extend(float(v) for v in np.mean(chroma, axis=1))
        else:
            components.extend([0.0] * 12)

        # Mel-spectrogram means (128 bins)
        mel = np.asarray(audio_features.get("mel_spectrogram", np.array([])), dtype=np.float64)
        if mel.ndim == 2 and mel.shape[0] > 0:
            components.extend(float(v) for v in np.mean(mel, axis=1))
        else:
            components.extend([0.0] * 128)

        # Audio energy: mean, std
        energy = np.asarray(audio_features.get("audio_energy", np.array([])), dtype=np.float64)
        if energy.size > 0:
            components.append(float(np.mean(energy)))
            components.append(
                float(np.std(energy, ddof=1)) if energy.size > 1 else 0.0
            )
        else:
            components.extend([0.0, 0.0])
    else:
        # No audio features available – pad with zeros
        components.extend([0.0] * 13)   # MFCC means
        components.extend([0.0] * 2)    # spectral centroid
        components.extend([0.0] * 2)    # spectral bandwidth
        components.extend([0.0] * 2)    # ZCR
        components.extend([0.0] * 12)   # chroma
        components.extend([0.0] * 128)  # mel spectrogram
        components.extend([0.0] * 2)    # audio energy

    # ------------------------------------------------------------------
    # rPPG features
    # ------------------------------------------------------------------
    heart_rate = float(rppg_features.get("heart_rate", 0.0))
    pulse_stability = float(rppg_features.get("pulse_stability", 0.0))
    rppg_confidence = float(rppg_features.get("rppg_confidence", 0.0))

    components.extend([heart_rate, pulse_stability, rppg_confidence])

    vector = np.array(components, dtype=np.float64)
    logger.debug(
        "fuse_lip_rppg: assembled vector of length %d.", len(vector),
    )
    return vector


# ==========================================================================
# 5. Main Fusion Vector Creation
# ==========================================================================

_FUSION_TARGET_DIM = 256


def create_fusion_vector(
    cnn_features: np.ndarray,
    lip_features_dict: Dict[str, Union[np.ndarray, float, str, Dict]],
    rppg_features_dict: Dict[str, Union[float, np.ndarray, Dict]],
    temporal_features_dict: Dict[str, float],
    target_dim: int = _FUSION_TARGET_DIM,
    pca: Optional[PCA] = None,
) -> np.ndarray:
    """Assemble the final multimodal fusion vector for a single video.

    The function:

    1. Aggregates per-frame CNN features (mean pooling across frames).
    2. Computes lip + rPPG summary statistics via :func:`fuse_lip_rppg_features`.
    3. Normalises each sub-vector independently (Z-score).
    4. Concatenates all normalised sub-vectors.
    5. Applies PCA dimensionality reduction if the concatenated dimension
       exceeds *target_dim* (default 256).

    Parameters
    ----------
    cnn_features : numpy.ndarray
        Per-frame CNN feature matrix of shape ``(T, D)`` or an already-
        aggregated 1-D vector of shape ``(D,)``.
    lip_features_dict : dict
        Lip-synchronisation analysis output (see :func:`fuse_lip_rppg_features`).
    rppg_features_dict : dict
        rPPG analysis output (see :func:`fuse_lip_rppg_features`).
    temporal_features_dict : dict
        Temporal statistics from :func:`compute_temporal_features`.
    target_dim : int, optional
        Target dimension after PCA.  Applied only if the concatenated
        vector exceeds this size.  Defaults to 256.
    pca : sklearn.decomposition.PCA, optional
        A pre-fitted PCA instance.  If *None* and PCA is needed, a new
        instance will be fitted on this single sample (note: PCA on a
        single sample is degenerate – provide a pre-fitted PCA for
        meaningful reduction).  If the vector is already ≤ *target_dim*,
        PCA is not applied regardless.

    Returns
    -------
    numpy.ndarray
        1-D fusion vector of dtype ``float64``.
    """
    cnn_features = np.asarray(cnn_features, dtype=np.float64)

    # --- 1. Aggregate CNN features to a single vector ---
    if cnn_features.ndim == 2:
        cnn_agg = np.mean(cnn_features, axis=0)  # (D,)
        logger.debug("CNN features aggregated via mean pooling: %s -> %s.",
                      cnn_features.shape, cnn_agg.shape)
    elif cnn_features.ndim == 1:
        cnn_agg = cnn_features
    else:
        raise ValueError(
            f"cnn_features must be 1-D or 2-D, got {cnn_features.ndim}-D."
        )

    # --- 2. Fuse lip + rPPG into a single vector ---
    lip_rppg_vec = fuse_lip_rppg_features(lip_features_dict, rppg_features_dict)

    # --- 3. Pack temporal statistics into a vector ---
    temporal_keys = [
        "mean", "std", "min", "max", "median",
        "skewness", "kurtosis", "trend_slope",
        "diff_mean", "diff_std", "zero_crossing_rate",
    ]
    temporal_vec = np.array(
        [float(temporal_features_dict.get(k, 0.0)) for k in temporal_keys],
        dtype=np.float64,
    )

    # --- 4. Normalise each sub-vector independently ---
    cnn_norm, _ = normalize_features(cnn_agg.reshape(1, -1))
    cnn_norm = cnn_norm.flatten()

    lip_rppg_norm, _ = normalize_features(lip_rppg_vec.reshape(1, -1))
    lip_rppg_norm = lip_rppg_norm.flatten()

    temporal_norm, _ = normalize_features(temporal_vec.reshape(1, -1))
    temporal_norm = temporal_norm.flatten()

    # --- 5. Concatenate ---
    fused = np.concatenate([cnn_norm, lip_rppg_norm, temporal_norm])
    logger.debug(
        "Concatenated fusion vector: CNN(%d) + LipRPPG(%d) + Temporal(%d) = %d total.",
        len(cnn_norm), len(lip_rppg_norm), len(temporal_norm), len(fused),
    )

    # --- 6. Optional PCA dimensionality reduction ---
    if fused.shape[0] > target_dim:
        if pca is not None:
            # Use the pre-fitted PCA
            reduced = pca.transform(fused.reshape(1, -1)).flatten()
            logger.debug(
                "PCA reduced fusion vector from %d to %d dimensions.",
                fused.shape[0], reduced.shape[0],
            )
            fused = reduced
        else:
            # Cannot fit PCA on a single sample.  The dimensionality
            # reduction is intentionally deferred to build_feature_dataset,
            # which operates across all samples and can fit a proper PCA.
            logger.debug(
                "No pre-fitted PCA provided; returning full %d-dim vector. "
                "Use build_feature_dataset for dataset-level PCA reduction.",
                fused.shape[0],
            )

    return fused


# ==========================================================================
# 6. Build Feature Dataset
# ==========================================================================


def build_feature_dataset(
    video_data_list: List[Dict[str, Union[np.ndarray, Dict, int, float]]],
    backbone: str = "",
    target_dim: int = _FUSION_TARGET_DIM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch-process per-video feature dicts into a unified (X, y) dataset.

    For each video in *video_data_list* the function:

    1. Computes temporal statistics from the per-frame CNN features.
    2. Fuses lip-sync and rPPG features.
    3. Assembles the full fusion vector via :func:`create_fusion_vector`.
    4. Collects all vectors and labels into ``X`` and ``y`` arrays.

    After concatenation, if the total feature dimension exceeds
    *target_dim*, PCA is fitted on the entire dataset (properly, across
    all samples) and applied to reduce to *target_dim* dimensions.

    Parameters
    ----------
    video_data_list : list[dict]
        Each element **must** contain the keys:

        * ``'cnn_features'`` – numpy.ndarray of shape ``(T, D)`` (per-frame
          CNN embeddings).
        * ``'lip_features'`` – dict from :func:`lip_sync.analyze_lip_sync`.
        * ``'rppg_features'`` – dict from :func:`rppg.extract_rppg_features`.
        * ``'label'`` – int class label (e.g. 0 = real, 1 = fake).

    backbone : str, optional
        CNN backbone name (used for logging only).  Falls back to
        ``config.DEFAULT_BACKBONE``.
    target_dim : int, optional
        Target dimensionality after PCA.  Defaults to 256.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        * **X** – feature matrix of shape ``(N, target_dim)`` (or ``(N, D)``
          if *D ≤ target_dim*).
        * **y** – label array of shape ``(N,)``.
    """
    if not video_data_list:
        logger.warning("build_feature_dataset called with empty video_data_list.")
        return np.array([]).reshape(0, 0), np.array([], dtype=np.int64)

    if not backbone:
        backbone = config.DEFAULT_BACKBONE

    n_videos = len(video_data_list)
    logger.info(
        "Building feature dataset from %d videos (backbone=%s, target_dim=%d).",
        n_videos,
        backbone,
        target_dim,
    )

    # --- Phase 1: assemble raw fusion vectors per video ---
    fusion_vectors: List[np.ndarray] = []
    labels: List[int] = []

    for idx, video_data in enumerate(video_data_list):
        try:
            cnn_feats = np.asarray(
                video_data["cnn_features"], dtype=np.float64
            )
            lip_feats = video_data["lip_features"]
            rppg_feats = video_data["rppg_features"]
            label = int(video_data["label"])

            # Compute temporal statistics from per-frame CNN features
            temporal = compute_temporal_features(cnn_feats)

            # Assemble the fusion vector (without PCA reduction yet)
            fused = create_fusion_vector(
                cnn_features=cnn_feats,
                lip_features_dict=lip_feats,
                rppg_features_dict=rppg_feats,
                temporal_features_dict=temporal,
                target_dim=999999,  # disable per-sample PCA
            )

            fusion_vectors.append(fused)
            labels.append(label)

        except Exception as exc:
            logger.error(
                "Failed to process video at index %d: %s. Skipping.", idx, exc,
            )
            continue

    if not fusion_vectors:
        logger.error("No valid fusion vectors were produced.")
        return np.array([]).reshape(0, 0), np.array([], dtype=np.int64)

    X_raw = np.vstack(fusion_vectors).astype(np.float64)
    y = np.array(labels, dtype=np.int64)

    logger.info(
        "Raw fusion matrix: X shape %s, y shape %s.", X_raw.shape, y.shape,
    )

    # --- Phase 2: PCA dimensionality reduction (if needed) ---
    if X_raw.shape[1] > target_dim:
        # PCA requires n_components <= min(n_samples, n_features).
        max_components = min(X_raw.shape[0], X_raw.shape[1])
        effective_dim = min(target_dim, max_components)

        if effective_dim < target_dim:
            logger.warning(
                "Requested target_dim=%d but only %d components are possible "
                "(n_samples=%d, n_features=%d). Using %d components.",
                target_dim, effective_dim,
                X_raw.shape[0], X_raw.shape[1], effective_dim,
            )

        logger.info(
            "Applying PCA: %d -> %d dimensions.",
            X_raw.shape[1], effective_dim,
        )
        pca = PCA(n_components=effective_dim, random_state=42)
        X = pca.fit_transform(X_raw)
        explained_var = float(np.sum(pca.explained_variance_ratio_))
        logger.info(
            "PCA complete: %d components explain %.2f%% of variance.",
            effective_dim, explained_var * 100.0,
        )
    else:
        X = X_raw
        logger.info(
            "No PCA needed: feature dim %d <= target_dim %d.",
            X_raw.shape[1], target_dim,
        )

    logger.info(
        "Final dataset: X %s, y %s, label distribution: %s.",
        X.shape, y.shape,
        {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
    )
    return X, y
