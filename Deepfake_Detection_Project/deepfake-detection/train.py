"""
Training Script for AI-Based Multimodal Deepfake Video Detection.

Provides end-to-end training pipeline:
  1. Dataset loading from real/fake directory structure.
  2. Per-video feature extraction (frames, faces, lip sync, rPPG, CNN).
  3. Feature fusion into unified dataset.
  4. Model training with CNN+BiLSTM+Attention architecture.
  5. Evaluation on held-out test set.

Usage:
    python train.py --dataset-dir ./dataset --output-dir ./outputs \
                   --backbone efficientnetb0 --epochs 50 --batch-size 16
"""

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time

import numpy as np

import config
import preprocess
import lip_sync
import rppg
import feature_fusion
import cnn_model
import evaluation

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Video extensions supported for dataset scanning
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}


# ======================================================================
# 1. load_dataset
# ======================================================================


def load_dataset(dataset_dir: str) -> list:
    """Scan dataset directory for video files and assign labels.

    Expects the directory layout::

        dataset_dir/
        ├── real/
        │   ├── video1.mp4
        │   └── ...
        └── fake/
            ├── video2.avi
            └── ...

    Parameters
    ----------
    dataset_dir : str
        Root directory containing ``real/`` and ``fake/`` subdirectories.

    Returns
    -------
    list[dict]
        Each dict has keys ``'path'``, ``'label'``, ``'filename'``.
        Label ``0`` = real, ``1`` = fake.  Missing directories are handled
        gracefully (treated as empty).
    """
    real_dir = os.path.join(dataset_dir, "real")
    fake_dir = os.path.join(dataset_dir, "fake")
    video_list = []

    # --- Scan real ---
    if os.path.isdir(real_dir):
        for fname in sorted(os.listdir(real_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in _VIDEO_EXTENSIONS:
                video_list.append({
                    "path": os.path.join(real_dir, fname),
                    "label": 0,
                    "filename": fname,
                })
        logger.info("Found %d real videos in %s", len([v for v in video_list if v["label"] == 0]), real_dir)
    else:
        logger.warning("Real video directory not found: %s", real_dir)

    # --- Scan fake ---
    if os.path.isdir(fake_dir):
        for fname in sorted(os.listdir(fake_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in _VIDEO_EXTENSIONS:
                video_list.append({
                    "path": os.path.join(fake_dir, fname),
                    "label": 1,
                    "filename": fname,
                })
        logger.info("Found %d fake videos in %s", len([v for v in video_list if v["label"] == 1]), fake_dir)
    else:
        logger.warning("Fake video directory not found: %s", fake_dir)

    logger.info(
        "Dataset loaded: %d total videos (%d real, %d fake)",
        len(video_list),
        len([v for v in video_list if v["label"] == 0]),
        len([v for v in video_list if v["label"] == 1]),
    )
    return video_list


# ======================================================================
# 2. process_single_video
# ======================================================================


def _cache_path(output_dir: str, filename: str, suffix: str) -> str:
    """Build a cache file path under *output_dir*/cache/.

    Parameters
    ----------
    output_dir : str
        Root output directory.
    filename : str
        Original video filename (used to derive a unique cache key).
    suffix : str
        File suffix (e.g. ``'cnn_features.npy'``).

    Returns
    -------
    str
        Full path to the cache file.
    """
    base = os.path.splitext(filename)[0]
    cache_dir = os.path.join(output_dir, "cache", base)
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, suffix)


def process_single_video(video_info: dict, output_dir: str, backbone: str = "efficientnetb0") -> dict:
    """Extract all features from a single video and cache intermediate results.

    Pipeline per video:

    1. **Frame extraction** via :func:`preprocess.extract_frames`.
    2. **Face detection & preprocessing** via :func:`preprocess.preprocess_face`
       on every extracted frame.
    3. **Lip sync analysis** via :func:`lip_sync.analyze_lip_sync` on the raw
       frames.
    4. **rPPG feature extraction** via :func:`rppg.extract_rppg_features` on
       the preprocessed face crops.
    5. **CNN feature extraction** via :func:`feature_fusion.extract_cnn_features`
       on the BGR→RGB converted face crops.

    Intermediate results (CNN features, lip features, rPPG features) are saved
    as ``.npy`` / ``.npz`` files under ``<output_dir>/cache/<video_stem>/``
    so that subsequent runs can skip already-processed videos.

    Parameters
    ----------
    video_info : dict
        Must contain ``'path'``, ``'label'``, ``'filename'``.
    output_dir : str
        Directory where cached features are stored.
    backbone : str
        CNN backbone name for feature extraction.

    Returns
    -------
    dict
        Keys: ``'cnn_features'``, ``'lip_features'``, ``'rppg_features'``,
        ``'label'``, ``'filename'``, ``'path'``.
        Returns an empty dict if processing fails entirely.
    """
    video_path = video_info["path"]
    label = video_info["label"]
    filename = video_info["filename"]

    # ---- Check cache: if CNN features file exists, skip processing ----
    cnn_cache_file = _cache_path(output_dir, filename, "cnn_features.npy")
    lip_cache_file = _cache_path(output_dir, filename, "lip_features.npz")
    rppg_cache_file = _cache_path(output_dir, filename, "rppg_features.npz")

    if (os.path.isfile(cnn_cache_file)
            and os.path.isfile(lip_cache_file)
            and os.path.isfile(rppg_cache_file)):
        logger.info("Loading cached features for %s", filename)
        try:
            cnn_features = np.load(cnn_cache_file)
            lip_data = dict(np.load(lip_cache_file, allow_pickle=True))
            rppg_data = dict(np.load(rppg_cache_file, allow_pickle=True))
            # Convert numpy arrays back where appropriate
            for key in lip_data:
                if isinstance(lip_data[key], np.ndarray) and lip_data[key].ndim == 0:
                    lip_data[key] = lip_data[key].item()
            for key in rppg_data:
                if isinstance(rppg_data[key], np.ndarray) and rppg_data[key].ndim == 0:
                    rppg_data[key] = rppg_data[key].item()
            return {
                "cnn_features": cnn_features,
                "lip_features": lip_data,
                "rppg_features": rppg_data,
                "label": label,
                "filename": filename,
                "path": video_path,
            }
        except Exception as exc:
            logger.warning("Failed to load cache for %s: %s. Re-processing.", filename, exc)

    logger.info("Processing video: %s (label=%d)", filename, label)

    try:
        # ------------------------------------------------------------------
        # Step 1: Extract frames
        # ------------------------------------------------------------------
        frames, frame_metadata = preprocess.extract_frames(video_path)
        if not frames:
            logger.warning("No frames extracted from %s", filename)
            return {}

        fps = frame_metadata.get("fps", config.RPPG_FPS)
        logger.info("Extracted %d frames from %s (fps=%.1f)", len(frames), filename, fps)

        # ------------------------------------------------------------------
        # Step 2: Detect & preprocess faces
        # ------------------------------------------------------------------
        face_images_bgr = []  # Normalised BGR faces for rPPG
        face_images_rgb = []  # RGB faces for CNN feature extraction

        for frame in frames:
            face_norm = preprocess.preprocess_face(frame)  # (224, 224, 3) float32 BGR [0,1]
            if face_norm is not None:
                face_images_bgr.append(face_norm)
                # Convert BGR -> RGB for CNN backbone (expects RGB)
                face_rgb = face_norm[:, :, ::-1].copy()
                face_images_rgb.append(face_rgb)

        if not face_images_bgr:
            logger.warning("No faces detected in any frame of %s", filename)
            return {}

        faces_bgr = np.array(face_images_bgr, dtype=np.float32)  # (N, 224, 224, 3)
        faces_rgb = np.array(face_images_rgb, dtype=np.float32)  # (N, 224, 224, 3)
        logger.info("Preprocessed %d face images from %s", len(faces_bgr), filename)

        # ------------------------------------------------------------------
        # Step 3: Lip sync analysis (on raw BGR frames)
        # ------------------------------------------------------------------
        lip_features = lip_sync.analyze_lip_sync(video_path, frames)
        logger.info(
            "Lip sync analysis for %s: score=%.4f, label='%s'",
            filename,
            lip_features.get("lip_sync_score", 0.0),
            lip_features.get("lip_sync_label", "unknown"),
        )

        # ------------------------------------------------------------------
        # Step 4: rPPG feature extraction (on normalised BGR faces)
        # ------------------------------------------------------------------
        # rPPG works on relative pixel values, so normalised [0,1] is fine
        rppg_features = rppg.extract_rppg_features(
            face_images=faces_bgr,
            fps=fps,
        )
        logger.info(
            "rPPG for %s: HR=%.1f, stability=%.3f, confidence=%.3f",
            filename,
            rppg_features.get("heart_rate", 0.0),
            rppg_features.get("pulse_stability", 0.0),
            rppg_features.get("rppg_confidence", 0.0),
        )

        # ------------------------------------------------------------------
        # Step 5: CNN feature extraction (on RGB faces)
        # ------------------------------------------------------------------
        cnn_features = feature_fusion.extract_cnn_features(faces_rgb, backbone=backbone)
        logger.info(
            "CNN features for %s: shape=%s",
            filename,
            cnn_features.shape,
        )

        # ------------------------------------------------------------------
        # Step 6: Cache intermediate results
        # ------------------------------------------------------------------
        np.save(cnn_cache_file, cnn_features)

        # Save lip features (mix of arrays and scalars)
        lip_save_dict = {}
        for k, v in lip_features.items():
            if isinstance(v, np.ndarray):
                lip_save_dict[k] = v
            elif isinstance(v, dict):
                # Nested dict (e.g. audio_features, temporal_features)
                # Serialise as a JSON string stored in a 0-d array
                lip_save_dict[k] = np.array(json.dumps({kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                                                         for kk, vv in v.items()}))
            else:
                lip_save_dict[k] = np.array(v)
        np.savez(lip_cache_file, **lip_save_dict)

        # Save rPPG features
        rppg_save_dict = {}
        for k, v in rppg_features.items():
            if isinstance(v, np.ndarray):
                rppg_save_dict[k] = v
            elif isinstance(v, dict):
                rppg_save_dict[k] = np.array(json.dumps({kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                                                           for kk, vv in v.items()}))
            else:
                rppg_save_dict[k] = np.array(v)
        np.savez(rppg_cache_file, **rppg_save_dict)

        logger.info("Cached features for %s", filename)

        return {
            "cnn_features": cnn_features,
            "lip_features": lip_features,
            "rppg_features": rppg_features,
            "label": label,
            "filename": filename,
            "path": video_path,
        }

    except Exception as exc:
        logger.error("Failed to process video %s: %s", filename, exc, exc_info=True)
        return {}


# ======================================================================
# 3. prepare_training_data
# ======================================================================


def prepare_training_data(
    dataset_dir: str,
    output_dir: str,
    max_videos_per_class: int = None,
    backbone: str = "efficientnetb0",
) -> tuple:
    """Process all videos and build a fused feature dataset.

    Steps:

    1. Load video list via :func:`load_dataset`.
    2. Optionally limit per-class samples for balanced subsets.
    3. Process each video in parallel using :mod:`concurrent.futures`.
    4. Build the fused feature dataset via :func:`feature_fusion.build_feature_dataset`.

    Parameters
    ----------
    dataset_dir : str
        Root dataset directory (``real/`` and ``fake/`` subdirs).
    output_dir : str
        Directory for intermediate cache files.
    max_videos_per_class : int, optional
        Maximum number of videos to process per class.  ``None`` means
        use all available videos.
    backbone : str
        CNN backbone for feature extraction.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(X, y)`` feature matrix and label array ready for training.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Load dataset ---
    video_list = load_dataset(dataset_dir)
    if not video_list:
        logger.error("No videos found in %s. Exiting.", dataset_dir)
        return np.array([]).reshape(0, 0), np.array([], dtype=np.int64)

    # --- Step 2: Optional per-class limit ---
    if max_videos_per_class is not None and max_videos_per_class > 0:
        real_videos = [v for v in video_list if v["label"] == 0]
        fake_videos = [v for v in video_list if v["label"] == 1]
        real_videos = real_videos[:max_videos_per_class]
        fake_videos = fake_videos[:max_videos_per_class]
        video_list = real_videos + fake_videos
        logger.info(
            "Limited to %d videos per class: %d real + %d fake = %d total",
            max_videos_per_class, len(real_videos), len(fake_videos), len(video_list),
        )

    # --- Step 3: Parallel processing ---
    video_data_list = []
    failed_count = 0
    n_videos = len(video_list)

    # Try to use tqdm for progress bar; fall back to simple logging
    try:
        from tqdm import tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False
        logger.info("tqdm not available; using simple progress logging.")

    logger.info(
        "Processing %d videos with parallel workers (backbone=%s)...",
        n_videos, backbone,
    )

    start_time = time.time()

    # Use ThreadPoolExecutor: video I/O and CPU-bound feature extraction
    # release the GIL via NumPy/OpenCV C extensions.
    max_workers = min(os.cpu_count() or 4, 8)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_video = {
            executor.submit(process_single_video, vinfo, output_dir, backbone): vinfo
            for vinfo in video_list
        }

        # Collect results with progress bar
        if _has_tqdm:
            futures = concurrent.futures.as_completed(future_to_video)
            futures = tqdm(futures, total=n_videos, desc="Processing videos", unit="video")
        else:
            futures = concurrent.futures.as_completed(future_to_video)

        for future in futures:
            vinfo = future_to_video[future]
            try:
                result = future.result()
                if result:
                    video_data_list.append(result)
                else:
                    failed_count += 1
                    logger.warning("Skipped video (no features): %s", vinfo["filename"])
            except Exception as exc:
                failed_count += 1
                logger.error(
                    "Exception processing %s: %s", vinfo["filename"], exc
                )

    elapsed = time.time() - start_time
    logger.info(
        "Feature extraction complete: %d/%d videos succeeded in %.1fs (%.1f videos/s)",
        len(video_data_list), n_videos, elapsed,
        len(video_data_list) / max(elapsed, 1e-6),
    )

    if failed_count > 0:
        logger.warning("%d videos failed feature extraction and were skipped.", failed_count)

    if not video_data_list:
        logger.error("No videos produced valid features. Cannot build dataset.")
        return np.array([]).reshape(0, 0), np.array([], dtype=np.int64)

    # --- Step 4: Build fused feature dataset ---
    logger.info("Building fused feature dataset from %d videos...", len(video_data_list))
    X, y = feature_fusion.build_feature_dataset(
        video_data_list,
        backbone=backbone,
    )

    logger.info("Final training dataset: X shape=%s, y shape=%s", X.shape, y.shape)
    return X, y


# ======================================================================
# 4. get_callbacks
# ======================================================================


def get_callbacks(model_dir: str) -> list:
    """Create Keras training callbacks.

    Parameters
    ----------
    model_dir : str
        Directory where the best model checkpoint and TensorBoard logs
        are saved.

    Returns
    -------
    list[tf.keras.callbacks.Callback]
        A list containing:

        * :class:`~tf.keras.callbacks.EarlyStopping` — patience 10,
          monitors ``val_loss``, restores best weights.
        * :class:`~tf.keras.callbacks.ReduceLROnPlateau` — patience 5,
          factor 0.5, minimum LR ``1e-7``.
        * :class:`~tf.keras.callbacks.ModelCheckpoint` — saves the best
          model (lowest ``val_loss``) to ``<model_dir>/best_model.h5``.
        * :class:`~tf.keras.callbacks.TensorBoard` — logs to
          ``<model_dir>/tensorboard_logs/``.
    """
    os.makedirs(model_dir, exist_ok=True)

    import tensorflow as tf

    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1,
        ),
        # Model checkpoint — save best
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "best_model.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, "tensorboard_logs"),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
        ),
    ]

    logger.info("Created %d training callbacks.", len(callbacks))
    return callbacks


# ======================================================================
# 5. train_model
# ======================================================================


def _reshape_features_to_pseudo_images(
    X: np.ndarray,
    target_image_size: int = 224,
    channels: int = 3,
) -> tuple:
    """Reshape fused 1-D feature vectors into pseudo-image sequences.

    Each sample's feature vector (dim D) is reshaped into a sequence of
    small ``pseudo_h × pseudo_w × channels`` images, which are then
    upsampled to ``target_image_size × target_image_size × channels`` via
    bilinear interpolation so they can be fed into the CNN+BiLSTM model.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape ``(N, D)``.
    target_image_size : int
        Spatial size each pseudo-image is resized to.
    channels : int
        Number of channels per pseudo-image.

    Returns
    -------
    tuple[numpy.ndarray, int, int, int]
        ``(X_reshaped, sequence_length, pseudo_h, pseudo_w)`` where
        ``X_reshaped`` has shape ``(N, sequence_length, target_image_size,
        target_image_size, channels)``.
    """
    import tensorflow as tf

    n_samples, feature_dim = X.shape
    logger.info(
        "Reshaping %d samples with %d features into pseudo-image sequences "
        "(target=%dx%d, channels=%d)",
        n_samples, feature_dim, target_image_size, target_image_size, channels,
    )

    # Determine pseudo-image size: small grid so we get a reasonable seq length
    # Use 4x4 as the base pseudo-image size
    pseudo_h = 4
    pseudo_w = 4
    features_per_step = pseudo_h * pseudo_w * channels  # 48

    # Determine sequence length
    sequence_length = max(1, feature_dim // features_per_step)
    total_features_needed = sequence_length * features_per_step

    # Pad features if necessary
    if feature_dim < total_features_needed:
        padding = total_features_needed - feature_dim
        X_padded = np.zeros((n_samples, total_features_needed), dtype=np.float32)
        X_padded[:, :feature_dim] = X.astype(np.float32)
        logger.info(
            "Padded features from %d to %d dims (added %d zeros).",
            feature_dim, total_features_needed, padding,
        )
    else:
        X_padded = X[:, :total_features_needed].astype(np.float32)
        unused = feature_dim - total_features_needed
        if unused > 0:
            logger.info("Truncated %d unused features (used %d of %d).", unused, total_features_needed, feature_dim)

    # Reshape to (N * sequence_length, pseudo_h, pseudo_w, channels)
    X_3d = X_padded.reshape(n_samples * sequence_length, pseudo_h, pseudo_w, channels)

    # Upsample each pseudo-image to target_image_size × target_image_size
    X_upsampled = tf.image.resize(X_3d, [target_image_size, target_image_size], method="bilinear")
    X_upsampled = np.array(X_upsampled, dtype=np.float32)

    # Reshape back to (N, sequence_length, target_image_size, target_image_size, channels)
    X_reshaped = X_upsampled.reshape(
        n_samples, sequence_length, target_image_size, target_image_size, channels
    )

    logger.info(
        "Pseudo-image reshaping complete: %s → %s (seq_len=%d, pseudo=%dx%d)",
        X.shape,
        X_reshaped.shape,
        sequence_length,
        pseudo_h,
        pseudo_w,
    )

    return X_reshaped, sequence_length, pseudo_h, pseudo_w


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_save_dir: str,
    backbone: str = "efficientnetb0",
    epochs: int = 50,
    batch_size: int = 16,
) -> dict:
    """Train the deepfake detection model.

    Pipeline:

    1. Split data 70/15/15 (train / val / test) using stratified split.
    2. Reshape fused feature vectors into pseudo-image sequences.
    3. Build model via :func:`cnn_model.build_deepfake_model`.
    4. Train with callbacks (early stopping, LR reduction, checkpoint,
       TensorBoard).
    5. Evaluate on the held-out test set.
    6. Save model and training history.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape ``(N, D)``.
    y : numpy.ndarray
        Label array of shape ``(N,)``.
    model_save_dir : str
        Directory to save the trained model and history.
    backbone : str
        CNN backbone name.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Training batch size.

    Returns
    -------
    dict
        Keys: ``'model'``, ``'history'``, ``'test_metrics'``.
    """
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    os.makedirs(model_save_dir, exist_ok=True)

    n_samples = X.shape[0]
    logger.info(
        "Starting model training: %d samples, backbone=%s, epochs=%d, batch_size=%d",
        n_samples, backbone, epochs, batch_size,
    )

    if n_samples < 4:
        logger.error(
            "Insufficient samples (%d) for train/val/test split. Need at least 4.",
            n_samples,
        )
        raise ValueError(f"Need at least 4 samples for train/val/test split, got {n_samples}.")

    # ------------------------------------------------------------------
    # Step 1: Train / val / test split  (70 / 15 / 15)
    # ------------------------------------------------------------------
    test_size = config.TEST_SPLIT  # 0.15
    val_size = config.VAL_SPLIT    # 0.15
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42,
    )
    # Second split: separate val from train
    relative_val_size = val_size / (1.0 - test_size)  # 0.15 / 0.85 ≈ 0.1765
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=relative_val_size,
        stratify=y_temp,
        random_state=42,
    )

    logger.info(
        "Data split: train=%d, val=%d, test=%d",
        len(X_train), len(X_val), len(X_test),
    )
    logger.info(
        "Label distribution — train: %s, val: %s, test: %s",
        {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        {int(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
        {int(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
    )

    # ------------------------------------------------------------------
    # Step 2: Reshape fused features into pseudo-image sequences
    # ------------------------------------------------------------------
    X_train_reshaped, seq_len, pseudo_h, pseudo_w = _reshape_features_to_pseudo_images(X_train)
    X_val_reshaped, _, _, _ = _reshape_features_to_pseudo_images(X_val)
    X_test_reshaped, _, _, _ = _reshape_features_to_pseudo_images(X_test)

    input_shape = (X_train_reshaped.shape[2], X_train_reshaped.shape[3], X_train_reshaped.shape[4])
    logger.info(
        "Pseudo-image input_shape=%s, sequence_length=%d for model.",
        input_shape, seq_len,
    )

    # ------------------------------------------------------------------
    # Step 3: Build model
    # ------------------------------------------------------------------
    model = cnn_model.build_deepfake_model(
        backbone_name=backbone,
        input_shape=input_shape,
        sequence_length=seq_len,
    )

    # Log model summary
    summary = cnn_model.get_model_summary(model)
    logger.info("Model summary:\n%s", summary)
    param_counts = cnn_model.count_parameters(model)
    logger.info("Parameters: %s", json.dumps(param_counts, indent=2))

    # ------------------------------------------------------------------
    # Step 4: Train
    # ------------------------------------------------------------------
    callbacks = get_callbacks(model_save_dir)

    logger.info("Starting training for up to %d epochs...", epochs)
    train_start = time.time()

    history = model.fit(
        X_train_reshaped,
        y_train,
        validation_data=(X_val_reshaped, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    train_elapsed = time.time() - train_start
    logger.info("Training completed in %.1f seconds (%.1f min)", train_elapsed, train_elapsed / 60.0)

    # ------------------------------------------------------------------
    # Step 5: Evaluate on test set
    # ------------------------------------------------------------------
    logger.info("Evaluating model on test set (%d samples)...", len(X_test))

    # Measure inference time per sample
    inference_times = []
    y_prob_list = []
    for i in range(len(X_test_reshaped)):
        sample = np.expand_dims(X_test_reshaped[i], axis=0)
        t0 = time.time()
        prob = model.predict(sample, verbose=0)
        t1 = time.time()
        inference_times.append(t1 - t0)
        y_prob_list.append(prob[0, 0])

    y_prob = np.array(y_prob_list, dtype=np.float64)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    inference_times = np.array(inference_times, dtype=np.float64)

    test_metrics = evaluation.compute_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        inference_times=inference_times,
    )

    logger.info("Test set metrics:")
    for metric_name, metric_value in test_metrics.items():
        logger.info("  %s: %.4f", metric_name, metric_value)

    # ------------------------------------------------------------------
    # Step 6: Save model and history
    # ------------------------------------------------------------------
    # Save final model
    final_model_path = os.path.join(model_save_dir, "final_model.h5")
    model.save(final_model_path)
    logger.info("Final model saved to %s", final_model_path)

    # Save training history as JSON
    history_dict = {
        "epoch": list(range(1, len(history.history["loss"]) + 1)),
        "loss": [float(v) for v in history.history["loss"]],
        "accuracy": [float(v) for v in history.history.get("accuracy", [])],
        "val_loss": [float(v) for v in history.history.get("val_loss", [])],
        "val_accuracy": [float(v) for v in history.history.get("val_accuracy", [])],
        "auc": [float(v) for v in history.history.get("auc", [])],
        "val_auc": [float(v) for v in history.history.get("val_auc", [])],
        "precision": [float(v) for v in history.history.get("precision", [])],
        "recall": [float(v) for v in history.history.get("recall", [])],
        "lr": [float(v) for v in history.history.get("lr", [])],
    }
    history_path = os.path.join(model_save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    logger.info("Training history saved to %s", history_path)

    # Save test metrics
    metrics_path = os.path.join(model_save_dir, "test_metrics.json")
    # Convert numpy types to Python native for JSON serialisation
    serialisable_metrics = {}
    for k, v in test_metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            serialisable_metrics[k] = float(v)
        elif isinstance(v, np.ndarray):
            serialisable_metrics[k] = v.tolist()
        else:
            serialisable_metrics[k] = v
    with open(metrics_path, "w") as f:
        json.dump(serialisable_metrics, f, indent=2)
    logger.info("Test metrics saved to %s", metrics_path)

    # Save training configuration
    config_path = os.path.join(model_save_dir, "training_config.json")
    training_config = {
        "backbone": backbone,
        "epochs": epochs,
        "batch_size": batch_size,
        "input_shape": list(input_shape),
        "sequence_length": seq_len,
        "pseudo_image_size": [pseudo_h, pseudo_w],
        "feature_dim": int(X.shape[1]),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "train_split": float(config.TRAIN_SPLIT),
        "val_split": float(config.VAL_SPLIT),
        "test_split": float(config.TEST_SPLIT),
        "train_duration_sec": round(train_elapsed, 2),
    }
    with open(config_path, "w") as f:
        json.dump(training_config, f, indent=2)
    logger.info("Training config saved to %s", config_path)

    return {
        "model": model,
        "history": history.history,
        "test_metrics": test_metrics,
    }


# ======================================================================
# 6. main
# ======================================================================


def main():
    """CLI entry point for the training pipeline.

    Parses command-line arguments, configures logging, runs the full
    training pipeline, and prints final metrics.
    """
    parser = argparse.ArgumentParser(
        description="Train the deepfake detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=config.DATASET_DIR,
        help="Path to the dataset root directory (containing real/ and fake/ subdirs).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.OUTPUT_DIR,
        help="Directory for intermediate outputs and cached features.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=config.DEFAULT_BACKBONE,
        choices=config.CNN_BACKBONES,
        help="CNN backbone for feature extraction and model architecture.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Training batch size.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of videos per class (None = use all).",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Logging configuration
    # ------------------------------------------------------------------
    log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format=config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOG_FILE, mode="a"),
        ],
    )

    # Suppress overly verbose third-party loggers
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("mediapipe").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logger.info("=" * 70)
    logger.info("Deepfake Detection Model Training Pipeline")
    logger.info("=" * 70)
    logger.info("Configuration:")
    logger.info("  Dataset dir   : %s", args.dataset_dir)
    logger.info("  Output dir    : %s", args.output_dir)
    logger.info("  Backbone      : %s", args.backbone)
    logger.info("  Epochs        : %d", args.epochs)
    logger.info("  Batch size    : %d", args.batch_size)
    logger.info("  Max videos    : %s", args.max_videos)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Phase 1: Feature extraction & dataset preparation
    # ------------------------------------------------------------------
    pipeline_start = time.time()

    logger.info("[Phase 1] Preparing training data...")
    X, y = prepare_training_data(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_videos_per_class=args.max_videos,
        backbone=args.backbone,
    )

    if X.size == 0 or len(X) == 0:
        logger.error("No training data produced. Aborting.")
        sys.exit(1)

    logger.info(
        "[Phase 1] Complete. Dataset: X=%s, y=%s",
        X.shape, y.shape,
    )

    # ------------------------------------------------------------------
    # Phase 2: Model training
    # ------------------------------------------------------------------
    model_save_dir = os.path.join(args.output_dir, "trained_model")

    logger.info("[Phase 2] Training model...")
    result = train_model(
        X=X,
        y=y,
        model_save_dir=model_save_dir,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    pipeline_elapsed = time.time() - pipeline_start

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total pipeline time : {pipeline_elapsed:.1f}s ({pipeline_elapsed / 60:.1f} min)")
    print(f"  Backbone             : {args.backbone}")
    print(f"  Dataset size         : {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Model saved to       : {model_save_dir}")
    print("\n  Test Set Metrics:")
    print("  " + "-" * 40)
    for metric_name, metric_value in result["test_metrics"].items():
        if isinstance(metric_value, float):
            print(f"    {metric_name:<25s}: {metric_value:.4f}")
        else:
            print(f"    {metric_name:<25s}: {metric_value}")
    print("=" * 70)

    logger.info("Training pipeline finished successfully in %.1f seconds.", pipeline_elapsed)


if __name__ == "__main__":
    main()
