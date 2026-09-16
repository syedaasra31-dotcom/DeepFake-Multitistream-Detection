"""
Prediction / Inference Script for AI-Based Multimodal Deepfake Video Detection.

Provides a complete end-to-end inference pipeline for classifying a single
video as *Real* or *Fake*:

    1. Load a trained Keras model (.h5) with custom layers.
    2. Preprocess the video: validate, extract frames, detect & align faces.
    3. Extract multimodal features: CNN embeddings, lip-sync analysis, rPPG.
    4. Fuse features into a vector, reshape into pseudo-image sequence, and
       run model prediction.
    5. Generate Grad-CAM visualisation, a text report, and a DB record.

Usage:
    python predict.py --video path/to/video.mp4 --model models/best_model.h5 \\
                      --output-dir outputs/
"""

import argparse
import datetime
import logging
import os
import sqlite3
import time

import cv2
import numpy as np

import config
import feature_fusion
import lip_sync
import preprocess
import rppg
from cnn_model import AttentionLayer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# Pseudo-image reshape constants (must match train.py)
_PSEUDO_H = 4
_PSEUDO_W = 4
_CHANNELS = 3
_TARGET_IMAGE_SIZE = 224
_FEATURES_PER_STEP = _PSEUDO_H * _PSEUDO_W * _CHANNELS  # 48


# ======================================================================
# 1. Model Loading
# ======================================================================


def load_trained_model(model_path: str):
    """Load a saved Keras model from a ``.h5`` file.

    Registers :class:`cnn_model.AttentionLayer` as a custom object so that
    Keras can deserialise models that use the attention mechanism.

    Parameters
    ----------
    model_path : str
        Path to a ``.h5`` checkpoint produced by ``model.save()`` during
        training.

    Returns
    -------
    tensorflow.keras.Model
        The loaded Keras model, ready for inference.

    Raises
    ------
    FileNotFoundError
        If *model_path* does not exist.
    RuntimeError
        If the file cannot be loaded (e.g. corrupt or missing layers).
    """
    import tensorflow as tf

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Please train a model first using train.py."
        )

    logger.info("Loading trained model from: %s", model_path)

    custom_objects = {"AttentionLayer": AttentionLayer}

    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from '{model_path}'. "
            f"Ensure the file is a valid Keras .h5 checkpoint. Error: {exc}"
        ) from exc

    logger.info("Model loaded successfully: %s", model.name)
    return model


# ======================================================================
# 2. Feature Extraction & Reshaping Helpers
# ======================================================================


def _reshape_single_fusion_vector(fusion_vector: np.ndarray):
    """Reshape a 1-D fusion vector into a pseudo-image sequence for the model.

    Replicates the logic in ``train.py:_reshape_features_to_pseudo_images``
    for a single sample (batch-size 1).

    Parameters
    ----------
    fusion_vector : numpy.ndarray
        1-D feature vector of shape ``(D,)`` produced by
        :func:`feature_fusion.create_fusion_vector`.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(1, sequence_length, 224, 224, 3)`` ready for model
        input.
    """
    import tensorflow as tf

    feature_dim = int(fusion_vector.shape[0])
    features_per_step = _PSEUDO_H * _PSEUDO_W * _CHANNELS

    sequence_length = max(1, feature_dim // features_per_step)
    total_features_needed = sequence_length * features_per_step

    # Pad or truncate to match the expected dimensionality
    if feature_dim < total_features_needed:
        padding = total_features_needed - feature_dim
        padded = np.zeros(total_features_needed, dtype=np.float32)
        padded[:feature_dim] = fusion_vector.astype(np.float32)
        logger.debug(
            "Padded fusion vector from %d to %d dims for pseudo-image reshape.",
            feature_dim, total_features_needed,
        )
    else:
        padded = fusion_vector[:total_features_needed].astype(np.float32)
        unused = feature_dim - total_features_needed
        if unused > 0:
            logger.debug("Truncated %d unused features.", unused)

    # Reshape to (sequence_length, pseudo_h, pseudo_w, channels)
    x_3d = padded.reshape(sequence_length, _PSEUDO_H, _PSEUDO_W, _CHANNELS)

    # Upsample each pseudo-image to 224x224x3
    x_4d = x_3d[np.newaxis, ...]  # add batch dim → (1, T, 4, 4, 3)
    n_steps = x_4d.shape[1]
    x_flat = x_4d.reshape(-1, _PSEUDO_H, _PSEUDO_W, _CHANNELS)  # (T, 4, 4, 3)
    x_upsampled = tf.image.resize(
        x_flat, [_TARGET_IMAGE_SIZE, _TARGET_IMAGE_SIZE], method="bilinear"
    )
    x_upsampled = np.array(x_upsampled, dtype=np.float32)
    x_reshaped = x_upsampled.reshape(
        1, n_steps, _TARGET_IMAGE_SIZE, _TARGET_IMAGE_SIZE, _CHANNELS
    )

    logger.info(
        "Fusion vector reshaped: (%d,) → %s (seq_len=%d)",
        feature_dim, x_reshaped.shape, sequence_length,
    )
    return x_reshaped


# ======================================================================
# 3. Single Video Prediction
# ======================================================================


def predict_single_video(video_path: str, model, output_dir: str) -> dict:
    """Run the full inference pipeline on a single video.

    Pipeline steps:

    1. Validate the video file (format, duration, size).
    2. Extract frames at the configured interval.
    3. Detect, crop, and align faces for each frame.
    4. Extract CNN deep features from the detected faces.
    5. Analyse lip synchronisation (visual lip landmarks vs audio).
    6. Extract remote photoplethysmography (rPPG) features.
    7. Assemble the multimodal fusion vector.
    8. Reshape the fusion vector into a pseudo-image sequence matching the
       model's expected input shape.
    9. Run model prediction and collect all results.

    Parameters
    ----------
    video_path : str
        Absolute or relative path to the video file.
    model : tensorflow.keras.Model
        A loaded Keras model (via :func:`load_trained_model`).
    output_dir : str
        Directory where intermediate artefacts (Grad-CAM, report) will be
        saved.

    Returns
    -------
    dict
        Comprehensive prediction result containing:

        - ``prediction`` (int): ``0`` = Real, ``1`` = Fake.
        - ``confidence`` (float): Probability in ``[0, 100]``.
        - ``label`` (str): ``"Real"`` or ``"Fake"``.
        - ``inference_time`` (float): Wall-clock seconds for the full pipeline.
        - ``fps`` (float): Video frame rate.
        - ``lip_sync_score`` (float): Lip-sync score in ``[0, 1]``.
        - ``heart_rate`` (float): Estimated BPM from rPPG.
        - ``pulse_stability`` (float): Pulse signal stability ``[0, 1]``.
        - ``rppg_confidence`` (float): Overall rPPG confidence ``[0, 1]``.
        - ``face_count`` (int): Number of frames with detected faces.
        - ``frame_count`` (int): Total frames extracted.
        - ``video_metadata`` (dict): Raw video metadata from validation.
    """
    import tensorflow as tf

    os.makedirs(output_dir, exist_ok=True)

    pipeline_start = time.time()
    video_name = os.path.basename(video_path)

    logger.info("=" * 70)
    logger.info("Starting deepfake detection for: %s", video_name)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Validate video
    # ------------------------------------------------------------------
    is_valid, video_metadata = preprocess.validate_video(video_path)
    if not is_valid:
        logger.error("Video validation failed: %s", video_metadata.get("error"))
        return {
            "prediction": -1,
            "confidence": 0.0,
            "label": "Error",
            "inference_time": 0.0,
            "fps": 0.0,
            "lip_sync_score": 0.0,
            "heart_rate": 0.0,
            "pulse_stability": 0.0,
            "rppg_confidence": 0.0,
            "face_count": 0,
            "frame_count": 0,
            "video_metadata": video_metadata,
            "error": video_metadata.get("error"),
        }

    fps = video_metadata.get("fps", 0.0)
    logger.info("Video validated: fps=%.2f, frames=%d, duration=%.1fs",
                fps, video_metadata.get("frame_count", 0),
                video_metadata.get("duration", 0.0))

    # ------------------------------------------------------------------
    # Step 2: Extract frames
    # ------------------------------------------------------------------
    t0 = time.time()
    frames, frame_metadata = preprocess.extract_frames(video_path)
    frame_count = len(frames)
    logger.info("Extracted %d frames in %.2fs", frame_count, time.time() - t0)

    if frame_count == 0:
        logger.error("No frames extracted from video.")
        return {
            "prediction": -1,
            "confidence": 0.0,
            "label": "Error",
            "inference_time": time.time() - pipeline_start,
            "fps": fps,
            "lip_sync_score": 0.0,
            "heart_rate": 0.0,
            "pulse_stability": 0.0,
            "rppg_confidence": 0.0,
            "face_count": 0,
            "frame_count": 0,
            "video_metadata": video_metadata,
            "error": "No frames extracted",
        }

    # ------------------------------------------------------------------
    # Step 3: Detect & preprocess faces
    # ------------------------------------------------------------------
    t0 = time.time()
    face_images_list = []
    face_count = 0

    for idx, frame in enumerate(frames):
        face = preprocess.preprocess_face(frame)
        if face is not None:
            face_images_list.append(face)
            face_count += 1
        else:
            logger.debug("No face detected in frame %d", idx)

    logger.info("Detected faces in %d/%d frames (%.1f%%) in %.2fs",
                face_count, frame_count,
                100.0 * face_count / max(1, frame_count),
                time.time() - t0)

    if face_count == 0:
        logger.error("No faces detected in any frame. Cannot proceed.")
        return {
            "prediction": -1,
            "confidence": 0.0,
            "label": "Error",
            "inference_time": time.time() - pipeline_start,
            "fps": fps,
            "lip_sync_score": 0.0,
            "heart_rate": 0.0,
            "pulse_stability": 0.0,
            "rppg_confidence": 0.0,
            "face_count": 0,
            "frame_count": frame_count,
            "video_metadata": video_metadata,
            "error": "No faces detected in any frame",
        }

    # ------------------------------------------------------------------
    # Step 4: Extract CNN features
    # ------------------------------------------------------------------
    t0 = time.time()
    # Stack faces into a 4-D array (N, 224, 224, 3) in float32
    # preprocess_face returns normalised float32 images, but extract_cnn_features
    # expects them.  Ensure the array is contiguous.
    face_array = np.stack(face_images_list, axis=0).astype(np.float32)
    logger.info("Face array shape for CNN: %s", face_array.shape)

    cnn_features = feature_fusion.extract_cnn_features(face_array)
    logger.info("CNN features extracted: shape=%s in %.2fs",
                cnn_features.shape, time.time() - t0)

    # ------------------------------------------------------------------
    # Step 5: Analyse lip synchronisation
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        lip_sync_result = lip_sync.analyze_lip_sync(video_path, frames)
        lip_sync_score = float(lip_sync_result.get("lip_sync_score", 0.0))
    except Exception as exc:
        logger.warning("Lip-sync analysis failed: %s", exc)
        lip_sync_result = {"lip_sync_score": 0.0, "lip_sync_label": "unknown"}
        lip_sync_score = 0.0
    logger.info("Lip-sync score: %.4f (label='%s') in %.2fs",
                lip_sync_score,
                lip_sync_result.get("lip_sync_label", "unknown"),
                time.time() - t0)

    # ------------------------------------------------------------------
    # Step 6: Extract rPPG features
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        # rPPG expects BGR face images of shape (N, 224, 224, 3)
        # We need the original BGR faces; preprocess_face returns normalised
        # float32 in [0,1].  Convert back to uint8 BGR for rPPG.
        face_bgr_uint8 = (np.clip(face_array, 0.0, 1.0) * 255).astype(np.uint8)
        rppg_result = rppg.extract_rppg_features(
            face_images=face_bgr_uint8,
            fps=fps,
        )
        heart_rate = float(rppg_result.get("heart_rate", 0.0))
        pulse_stability = float(rppg_result.get("pulse_stability", 0.0))
        rppg_confidence = float(rppg_result.get("rppg_confidence", 0.0))
    except Exception as exc:
        logger.warning("rPPG analysis failed: %s", exc)
        rppg_result = {
            "heart_rate": 0.0, "pulse_stability": 0.0, "rppg_confidence": 0.0,
        }
        heart_rate = 0.0
        pulse_stability = 0.0
        rppg_confidence = 0.0
    logger.info("rPPG: HR=%.1f BPM, stability=%.3f, confidence=%.3f in %.2fs",
                heart_rate, pulse_stability, rppg_confidence, time.time() - t0)

    # ------------------------------------------------------------------
    # Step 7: Compute temporal features from CNN sequence
    # ------------------------------------------------------------------
    t0 = time.time()
    temporal_features = feature_fusion.compute_temporal_features(cnn_features)
    logger.info("Temporal features computed in %.2fs", time.time() - t0)

    # ------------------------------------------------------------------
    # Step 8: Create fusion vector
    # ------------------------------------------------------------------
    t0 = time.time()
    fusion_vector = feature_fusion.create_fusion_vector(
        cnn_features=cnn_features,
        lip_features_dict=lip_sync_result,
        rppg_features_dict=rppg_result,
        temporal_features_dict=temporal_features,
    )
    logger.info("Fusion vector created: dim=%d in %.2fs",
                fusion_vector.shape[0], time.time() - t0)

    # ------------------------------------------------------------------
    # Step 9: Reshape for model input and predict
    # ------------------------------------------------------------------
    t0 = time.time()
    model_input = _reshape_single_fusion_vector(fusion_vector)
    logger.info("Model input shape: %s", model_input.shape)

    predictions = model.predict(model_input, verbose=0)
    raw_probability = float(predictions[0][0])

    prediction_class = int(raw_probability >= 0.5)
    confidence_pct = float(raw_probability * 100.0) if prediction_class == 1 \
        else float((1.0 - raw_probability) * 100.0)
    label = config.CLASS_LABELS.get(prediction_class, "Unknown")

    logger.info("Model prediction: class=%d, prob=%.4f, label='%s' in %.2fs",
                prediction_class, raw_probability, label, time.time() - t0)

    # ------------------------------------------------------------------
    # Total pipeline timing
    # ------------------------------------------------------------------
    total_inference_time = time.time() - pipeline_start
    fps_processed = frame_count / max(total_inference_time, 0.001)

    logger.info("Total inference time: %.2fs (%.1f frames/sec)", total_inference_time, fps_processed)

    result = {
        "prediction": prediction_class,
        "confidence": round(confidence_pct, 2),
        "label": label,
        "inference_time": round(total_inference_time, 3),
        "fps": fps_processed,
        "lip_sync_score": round(lip_sync_score, 4),
        "heart_rate": round(heart_rate, 1),
        "pulse_stability": round(pulse_stability, 4),
        "rppg_confidence": round(rppg_confidence, 4),
        "face_count": face_count,
        "frame_count": frame_count,
        "video_metadata": video_metadata,
        "video_name": video_name,
        "raw_probability": round(raw_probability, 6),
    }

    logger.info("=" * 70)
    logger.info("RESULT: %s (confidence=%.2f%%)", label, confidence_pct)
    logger.info("=" * 70)

    return result


# ======================================================================
# 4. Grad-CAM Visualisation
# ======================================================================


def generate_gradcam(model, face_image, layer_name: str = "block5c_project_conv",
                     output_dir: str = None) -> str:
    """Generate a Grad-CAM (Gradient-weighted Class Activation Mapping) heatmap.

    Grad-CAM produces a visual explanation of which regions of the input
    image most influenced the model's prediction.  The heatmap is computed
    as a weighted sum of the feature maps from a target convolutional layer,
    where the weights are the global-average-pooled gradients of the predicted
    class score with respect to each feature map.

    The function searches for *layer_name* in the model's layer list
    (supports partial name matching).  If not found, it falls back to the
    last convolutional layer in the model.

    Parameters
    ----------
    model : tensorflow.keras.Model
        A loaded Keras model.
    face_image : numpy.ndarray
        Input face image of shape ``(H, W, 3)`` in RGB order, ``uint8`` or
        ``float32``.  If ``float32`` in ``[0, 1]``, it is converted to
        ``uint8`` ``[0, 255]`` for overlay purposes.
    layer_name : str
        Name (or substring) of the target convolutional layer from which to
        extract feature maps.  Defaults to
        ``"block5c_project_conv"`` (EfficientNet top block).
    output_dir : str or None
        Directory to save the output image.  If ``None``, uses
        ``config.OUTPUT_DIR``.

    Returns
    -------
    str
        Absolute path to the saved heatmap image file.

    Raises
    ------
    ValueError
        If no suitable convolutional layer is found in the model.
    """
    import tensorflow as tf

    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Prepare image for model consumption
    img_array = np.asarray(face_image, dtype=np.float32)
    if img_array.max() <= 1.0:
        img_display = (img_array * 255.0).astype(np.uint8)
    else:
        img_display = img_array.astype(np.uint8)

    img_tensor = tf.convert_to_tensor(
        img_array[np.newaxis, ...], dtype=tf.float32
    )

    # ---------------------------------------------------------------
    # Find target conv layer
    # ---------------------------------------------------------------
    target_layer = None
    for layer in model.layers:
        if layer_name in layer.name:
            target_layer = layer
            break

    # Fallback: search for any conv layer
    if target_layer is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                target_layer = layer
                layer_name = layer.name
                logger.info(
                    "Target layer '%s' not found; fell back to last Conv2D: '%s'",
                    layer_name, target_layer.name,
                )
                break

    if target_layer is None:
        raise ValueError(
            "No convolutional layer found in the model. "
            "Grad-CAM requires at least one Conv2D layer."
        )

    logger.info("Grad-CAM target layer: '%s'", target_layer.name)

    # ---------------------------------------------------------------
    # Build gradient model: input → target_layer → model output
    # ---------------------------------------------------------------
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[target_layer.output, model.output],
    )

    # ---------------------------------------------------------------
    # Compute gradients of predicted class w.r.t. feature maps
    # ---------------------------------------------------------------
    with tf.GradientTape() as tape:
        # We need the image to have the expected model input shape.
        # The model expects (batch, T, H, W, 3).  If the input is a single
        # face image (H, W, 3), we need to handle the mismatch.  We build a
        # minimal sub-model that takes a single image input.
        conv_outputs, predictions = grad_model(img_tensor, training=False)

        # For binary classification the output shape is (batch, 1)
        if predictions.shape[-1] == 1:
            loss = predictions[0]
        else:
            predicted_class = tf.argmax(predictions[0])
            loss = predictions[:, predicted_class]

    # Gradients: shape = (batch, H_conv, W_conv, n_filters)
    gradients = tape.gradient(loss, conv_outputs)
    if gradients is None:
        logger.warning("Gradients are None – returning blank heatmap.")
        heatmap_path = os.path.join(output_dir, "gradcam_blank.png")
        cv2.imwrite(heatmap_path, img_display)
        return heatmap_path

    # ---------------------------------------------------------------
    # Global Average Pool the gradients → weights
    # ---------------------------------------------------------------
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    # conv_outputs shape: (batch, H_conv, W_conv, n_filters) → weight each
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ---------------------------------------------------------------
    # ReLU + normalise
    # ---------------------------------------------------------------
    heatmap = np.maximum(heatmap, 0)
    max_val = heatmap.max()
    if max_val > 0:
        heatmap = heatmap / max_val

    heatmap_uint8 = np.uint8(255 * heatmap)

    # ---------------------------------------------------------------
    # Resize heatmap to original face size and apply colour-map
    # ---------------------------------------------------------------
    face_h, face_w = img_display.shape[:2]
    heatmap_resized = cv2.resize(heatmap_uint8, (face_w, face_h))

    # Apply JET colour-map
    heatmap_colour = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # ---------------------------------------------------------------
    # Overlay on original image
    # ---------------------------------------------------------------
    # Convert BGR display image to RGB for consistent colour appearance
    img_bgr = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(img_bgr, 0.6, heatmap_colour, 0.4, 0)

    # Save result
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    heatmap_filename = f"gradcam_{timestamp}.png"
    heatmap_path = os.path.join(output_dir, heatmap_filename)
    cv2.imwrite(heatmap_path, superimposed)

    logger.info("Grad-CAM heatmap saved to: %s", heatmap_path)
    return os.path.abspath(heatmap_path)


# ======================================================================
# 5. Prediction Report
# ======================================================================


def generate_prediction_report(prediction_result: dict, output_dir: str = None) -> str:
    """Generate a human-readable text report for a prediction result.

    Parameters
    ----------
    prediction_result : dict
        Dictionary returned by :func:`predict_single_video`.
    output_dir : str or None
        Directory to save the report.  If ``None``, uses
        ``config.OUTPUT_DIR``.

    Returns
    -------
    str
        Absolute path to the saved report text file.
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    video_name = prediction_result.get("video_name", "unknown_video")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    separator = "=" * 60

    lines = [
        separator,
        "  DEEPFAKE DETECTION – PREDICTION REPORT",
        separator,
        "",
        f"  Video File          : {video_name}",
        f"  Analysis Timestamp  : {timestamp}",
        "",
        "-" * 60,
        "  PREDICTION RESULT",
        "-" * 60,
        f"  Prediction          : {prediction_result.get('prediction', 'N/A')}  "
        f"(0 = Real, 1 = Fake)",
        f"  Label               : {prediction_result.get('label', 'N/A')}",
        f"  Confidence          : {prediction_result.get('confidence', 'N/A')}%",
        f"  Raw Probability     : {prediction_result.get('raw_probability', 'N/A')}",
        "",
        "-" * 60,
        "  PERFORMANCE",
        "-" * 60,
        f"  Inference Time      : {prediction_result.get('inference_time', 'N/A')} seconds",
        f"  Processing FPS      : {prediction_result.get('fps', 'N/A')} frames/sec",
        "",
        "-" * 60,
        "  LIP SYNC ANALYSIS",
        "-" * 60,
        f"  Lip Sync Score      : {prediction_result.get('lip_sync_score', 'N/A')}",
        "    (>0.55 = synced, <=0.55 = desynced)",
        "",
        "-" * 60,
        "  rPPG ANALYSIS",
        "-" * 60,
        f"  Heart Rate          : {prediction_result.get('heart_rate', 'N/A')} BPM",
        f"  Pulse Stability     : {prediction_result.get('pulse_stability', 'N/A')}",
        f"  rPPG Confidence     : {prediction_result.get('rppg_confidence', 'N/A')}",
        "",
        "-" * 60,
        "  VIDEO & FRAME STATISTICS",
        "-" * 60,
        f"  Faces Detected      : {prediction_result.get('face_count', 'N/A')} / "
        f"{prediction_result.get('frame_count', 'N/A')} frames",
        "",
    ]

    # Video metadata
    video_meta = prediction_result.get("video_metadata", {})
    if video_meta:
        lines.append("-" * 60)
        lines.append("  VIDEO METADATA")
        lines.append("-" * 60)
        lines.append(f"  Resolution          : {video_meta.get('width', 'N/A')}x"
                     f"{video_meta.get('height', 'N/A')}")
        lines.append(f"  Frame Rate          : {video_meta.get('fps', 'N/A')} FPS")
        lines.append(f"  Duration            : {video_meta.get('duration', 'N/A'):.2f} sec")
        lines.append(f"  Total Frames        : {video_meta.get('frame_count', 'N/A')}")
        lines.append(f"  File Size           : {video_meta.get('file_size_mb', 'N/A')} MB")
        lines.append(f"  Format              : {video_meta.get('format', 'N/A')}")
        lines.append("")

    # Assessment summary
    lines.append(separator)
    lines.append("  ASSESSMENT SUMMARY")
    lines.append(separator)

    pred_label = prediction_result.get("label", "Unknown")
    confidence = prediction_result.get("confidence", 0.0)
    lip_score = prediction_result.get("lip_sync_score", 0.0)
    rppg_conf = prediction_result.get("rppg_confidence", 0.0)

    if pred_label == "Fake":
        lines.append("")
        lines.append("  ⚠  This video is classified as FAKE (deepfake).")
        lines.append("")
        lines.append("  Contributing factors:")
        if confidence > 80:
            lines.append(f"    - High model confidence ({confidence}%) suggests strong")
            lines.append("      deepfake artifacts were detected.")
        if lip_score < config.LIP_SYNC_THRESHOLD:
            lines.append(f"    - Lip desynchronisation detected (score={lip_score:.4f}).")
        if rppg_conf < 0.3:
            lines.append(f"    - Low rPPG confidence ({rppg_conf:.4f}) indicates possible")
            lines.append("      lack of authentic physiological signals.")
    elif pred_label == "Real":
        lines.append("")
        lines.append("  ✓  This video is classified as REAL (authentic).")
        lines.append("")
        lines.append("  Indicators of authenticity:")
        if confidence > 80:
            lines.append(f"    - High model confidence ({confidence}%).")
        if lip_score >= config.LIP_SYNC_THRESHOLD:
            lines.append(f"    - Lip movements are synchronised with audio "
                         f"(score={lip_score:.4f}).")
        if rppg_conf > 0.3:
            lines.append(f"    - rPPG confidence is healthy ({rppg_conf:.4f}), suggesting")
            lines.append("      authentic physiological signals.")
    else:
        lines.append("")
        lines.append("  Prediction could not be completed.")
        if prediction_result.get("error"):
            lines.append(f"  Error: {prediction_result['error']}")

    lines.append("")
    lines.append(separator)
    lines.append(f"  Generated by Deepfake Detection System – {timestamp}")
    lines.append(separator)

    report_text = "\n".join(lines)

    # Save to file
    report_filename = f"prediction_report_{video_name}_{timestamp.replace(':', '').replace(' ', '_')}.txt"
    report_path = os.path.join(output_dir, report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info("Prediction report saved to: %s", report_path)
    return os.path.abspath(report_path)


# ======================================================================
# 6. Database Persistence
# ======================================================================


def save_prediction_to_db(prediction_result: dict, db_path: str = None) -> int:
    """Save a prediction record to the SQLite database.

    Creates the ``predictions`` table if it does not already exist.

    Parameters
    ----------
    prediction_result : dict
        Dictionary returned by :func:`predict_single_video`.
    db_path : str or None
        Path to the SQLite database file.  If ``None``, uses
        ``config.DATABASE_PATH``.

    Returns
    -------
    int
        The row id of the inserted record.

    Raises
    ------
    sqlite3.Error
        If the insert operation fails.
    """
    if db_path is None:
        db_path = config.DATABASE_PATH

    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS predictions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        video_name      TEXT    NOT NULL,
        prediction      INTEGER NOT NULL,
        confidence      REAL    NOT NULL,
        lip_sync_score  REAL    DEFAULT 0.0,
        heart_rate      REAL    DEFAULT 0.0,
        inference_time  REAL    DEFAULT 0.0,
        created_at      TEXT    NOT NULL
    );
    """

    insert_sql = """
    INSERT INTO predictions
        (video_name, prediction, confidence, lip_sync_score, heart_rate,
         inference_time, created_at)
    VALUES
        (?, ?, ?, ?, ?, ?, ?);
    """

    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(create_table_sql)
        conn.commit()

        cursor.execute(insert_sql, (
            prediction_result.get("video_name", "unknown"),
            prediction_result.get("prediction", -1),
            prediction_result.get("confidence", 0.0),
            prediction_result.get("lip_sync_score", 0.0),
            prediction_result.get("heart_rate", 0.0),
            prediction_result.get("inference_time", 0.0),
            created_at,
        ))
        row_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(
            "Prediction saved to database (id=%d): %s → %s (%.1f%%)",
            row_id,
            prediction_result.get("video_name"),
            prediction_result.get("label"),
            prediction_result.get("confidence"),
        )
        return row_id

    except sqlite3.Error as exc:
        logger.error("Failed to save prediction to database: %s", exc)
        raise


# ======================================================================
# 7. CLI Entry Point
# ======================================================================


def main():
    """Command-line entry point for deepfake video prediction.

    Parses arguments, runs the full inference pipeline, prints results to
    the console, and saves a prediction report, Grad-CAM heatmap, and
    database record.

    Usage::

        python predict.py --video path/to/video.mp4 \\
                          --model models/best_deepfake_model.h5 \\
                          --output-dir outputs/
    """
    parser = argparse.ArgumentParser(
        description="AI-Based Multimodal Deepfake Video Detection – Prediction Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --video test.mp4 --model best_deepfake_model.h5
  python predict.py --video test.mp4 --model best_deepfake_model.h5 --output-dir results/
        """,
    )

    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to the input video file to analyse.",
    )
    parser.add_argument(
        "--model", type=str, default=config.BEST_MODEL_PATH,
        help=f"Path to the trained .h5 model file (default: {config.BEST_MODEL_PATH}).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=config.OUTPUT_DIR,
        help=f"Directory for output artefacts (default: {config.OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--db-path", type=str, default=config.DATABASE_PATH,
        help=f"Path to the SQLite database (default: {config.DATABASE_PATH}).",
    )
    parser.add_argument(
        "--no-gradcam", action="store_true",
        help="Skip Grad-CAM heatmap generation.",
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Skip text report generation.",
    )
    parser.add_argument(
        "--no-db", action="store_true",
        help="Skip saving prediction to database.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not os.path.isfile(args.video):
        print(f"ERROR: Video file not found: {args.video}")
        return 1

    if not os.path.isfile(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        print("  Train a model first using: python train.py")
        return 1

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  DEEPFAKE VIDEO DETECTION – INFERENCE")
    print(f"{'=' * 60}")
    print(f"  Video     : {os.path.basename(args.video)}")
    print(f"  Model     : {os.path.basename(args.model)}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"{'=' * 60}\n")

    model = load_trained_model(args.model)

    # ------------------------------------------------------------------
    # Run prediction
    # ------------------------------------------------------------------
    result = predict_single_video(args.video, model, args.output_dir)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  PREDICTION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Video              : {result.get('video_name', 'N/A')}")
    print(f"  Prediction         : {result.get('label', 'N/A')} "
          f"(class={result.get('prediction', 'N/A')})")
    print(f"  Confidence         : {result.get('confidence', 0):.2f}%")
    print(f"  Inference Time     : {result.get('inference_time', 0):.3f} seconds")
    print(f"  Processing FPS     : {result.get('fps', 0):.1f}")
    print(f"  Lip Sync Score      : {result.get('lip_sync_score', 0):.4f}")
    print(f"  Heart Rate          : {result.get('heart_rate', 0):.1f} BPM")
    print(f"  Pulse Stability     : {result.get('pulse_stability', 0):.4f}")
    print(f"  rPPG Confidence     : {result.get('rppg_confidence', 0):.4f}")
    print(f"  Faces / Frames      : {result.get('face_count', 0)} / "
          f"{result.get('frame_count', 0)}")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # Generate Grad-CAM
    # ------------------------------------------------------------------
    gradcam_path = None
    if not args.no_gradcam:
        print("\n[INFO] Generating Grad-CAM heatmap...")
        try:
            # Extract the first detected face for Grad-CAM
            frames, _ = preprocess.extract_frames(args.video)
            if frames:
                for frame in frames:
                    face = preprocess.preprocess_face(frame)
                    if face is not None:
                        # Convert normalised face to RGB uint8 for display
                        face_rgb = (np.clip(face, 0, 1) * 255).astype(np.uint8)
                        gradcam_path = generate_gradcam(
                            model, face_rgb,
                            layer_name="block5c_project_conv",
                            output_dir=args.output_dir,
                        )
                        print(f"  Grad-CAM saved to: {gradcam_path}")
                        break
            else:
                print("  [WARN] No frames available for Grad-CAM.")
        except Exception as exc:
            print(f"  [WARN] Grad-CAM generation failed: {exc}")
            logger.warning("Grad-CAM generation failed: %s", exc)

    # ------------------------------------------------------------------
    # Generate report
    # ------------------------------------------------------------------
    report_path = None
    if not args.no_report:
        print("\n[INFO] Generating prediction report...")
        try:
            report_path = generate_prediction_report(result, args.output_dir)
            print(f"  Report saved to: {report_path}")
        except Exception as exc:
            print(f"  [WARN] Report generation failed: {exc}")
            logger.warning("Report generation failed: %s", exc)

    # ------------------------------------------------------------------
    # Save to database
    # ------------------------------------------------------------------
    if not args.no_db:
        print("\n[INFO] Saving prediction to database...")
        try:
            row_id = save_prediction_to_db(result, args.db_path)
            print(f"  Saved to DB with row id: {row_id}")
        except Exception as exc:
            print(f"  [WARN] Database save failed: {exc}")
            logger.warning("Database save failed: %s", exc)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    if result.get("label") in ("Real", "Fake"):
        print(f"  FINAL VERDICT: {result['label']} "
              f"({result['confidence']:.2f}% confidence)")
    else:
        print(f"  PREDICTION FAILED: {result.get('error', 'Unknown error')}")
    print(f"{'=' * 60}\n")

    return 0 if result.get("prediction", -1) in (0, 1) else 1


if __name__ == "__main__":
    exit(main())
