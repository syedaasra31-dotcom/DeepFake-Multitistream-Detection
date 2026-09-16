"""
Lip Synchronization Analysis Module for AI-Based Multimodal Deepfake Video Detection.

This module provides comprehensive lip synchronization analysis using MediaPipe Face Mesh
for visual lip landmark extraction and Librosa for audio feature extraction. It detects
desynchronization between lip movements and audio signals, a common artifact in
deepfake videos.

Functions:
    extract_lip_landmarks    - Extract mouth landmarks from a video frame via MediaPipe.
    compute_lip_aspect_ratio - Compute LAR = vertical / horizontal mouth distance.
    compute_mouth_opening    - Compute vertical distance between inner lip landmarks.
    compute_lip_movement     - Temporal derivative of lip features across frames.
    compute_temporal_lip_motion - FFT-based frequency domain features of lip motion.
    extract_audio_features   - MFCCs, spectral, chroma, and mel spectrogram features.
    compute_lip_sync_score   - Pearson + DTW based 0-1 sync score.
    analyze_lip_sync         - Full pipeline orchestrating all analysis steps.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import librosa
import mediapipe as mp
import moviepy.editor as mpy
import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import pearsonr

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark index constants
# ---------------------------------------------------------------------------
# Outer lip landmarks (as specified for this project)
OUTER_LIP_INDICES: List[int] = [
    61, 291, 13, 14, 78, 308, 82, 312, 87, 317, 324, 84, 181, 91, 146,
]

# Inner lip contour landmarks (MediaPipe Face Mesh 468-point topology)
INNER_LIP_INDICES: List[int] = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
]

# Deduplicated union of outer + inner lip indices (preserves ordering)
_ALL_LIP_INDICES: List[int] = list(dict.fromkeys(OUTER_LIP_INDICES + INNER_LIP_INDICES))

# Key landmark indices used for geometric measurements
UPPER_LIP_CENTER_IDX: int = 13   # Upper inner-lip centre
LOWER_LIP_CENTER_IDX: int = 14   # Lower inner-lip centre
LEFT_MOUTH_CORNER_IDX: int = 61  # Left mouth corner
RIGHT_MOUTH_CORNER_IDX: int = 291  # Right mouth corner

# ---------------------------------------------------------------------------
# Module-level FaceMesh instance (lazy, thread-hostile - single-thread use)
# ---------------------------------------------------------------------------
_face_mesh: Optional[mp.solutions.face_mesh.FaceMesh] = None


def _get_face_mesh() -> mp.solutions.face_mesh.FaceMesh:
    """Return a cached :class:`mediapipe.solutions.face_mesh.FaceMesh` instance.

    The instance is created on first call with sensible defaults for real-time
    video processing and reused for every subsequent call.
    """
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        logger.info("MediaPipe FaceMesh initialised successfully")
    return _face_mesh


# ======================================================================
# 1. extract_lip_landmarks
# ======================================================================


def extract_lip_landmarks(frame: np.ndarray) -> Optional[np.ndarray]:
    """Extract lip landmarks from a single video frame using MediaPipe Face Mesh.

    Extracts the outer lip landmarks (indices 61, 291, 13, 14, 78, 308, 82,
    312, 87, 317, 324, 84, 181, 91, 146) together with the full inner lip
    contour and returns their ``(x, y)`` pixel coordinates.

    Parameters
    ----------
    frame : numpy.ndarray
        Input BGR image of shape ``(H, W, 3)``.

    Returns
    -------
    numpy.ndarray or None
        Array of shape ``(N, 2)`` - ``N`` unique lip landmarks - with columns
        ``[x, y]`` in pixel coordinates, or ``None`` when no face is detected.
    """
    if frame is None or frame.size == 0:
        logger.warning("Empty frame passed to extract_lip_landmarks")
        return None

    face_mesh = _get_face_mesh()

    # MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks is None or len(results.multi_face_landmarks) == 0:
        logger.debug("No face detected in frame")
        return None

    face_landmarks = results.multi_face_landmarks[0]
    h, w = frame.shape[:2]

    landmarks: List[List[float]] = []
    for idx in _ALL_LIP_INDICES:
        lm = face_landmarks.landmark[idx]
        landmarks.append([lm.x * w, lm.y * h])

    landmarks_array = np.array(landmarks, dtype=np.float64)
    logger.debug("Extracted %d lip landmarks from frame", len(landmarks_array))
    return landmarks_array


# ======================================================================
# 2. compute_lip_aspect_ratio
# ======================================================================


def compute_lip_aspect_ratio(landmarks: np.ndarray) -> float:
    """Compute the Lip Aspect Ratio (LAR) from extracted lip landmarks.

    LAR = d(upper lip centre, lower lip centre) / d(left corner, right corner)

    Parameters
    ----------
    landmarks : numpy.ndarray
        Array of shape ``(N, 2)`` as returned by :func:`extract_lip_landmarks`.

    Returns
    -------
    float
        Lip aspect ratio.  Returns ``0.0`` when landmarks are insufficient.
    """
    if landmarks is None or landmarks.shape[0] < 20:
        logger.warning("Insufficient landmarks (%d) for LAR",
                        0 if landmarks is None else landmarks.shape[0])
        return 0.0

    # Build a quick lookup: landmark-index -> (x, y)
    idx_to_pos = {idx: landmarks[i] for i, idx in enumerate(_ALL_LIP_INDICES)}

    try:
        upper_lip = idx_to_pos[UPPER_LIP_CENTER_IDX]
        lower_lip = idx_to_pos[LOWER_LIP_CENTER_IDX]
        left_corner = idx_to_pos[LEFT_MOUTH_CORNER_IDX]
        right_corner = idx_to_pos[RIGHT_MOUTH_CORNER_IDX]
    except KeyError as exc:
        logger.warning("Key landmark %s missing for LAR computation", exc)
        return 0.0

    vertical_distance = np.linalg.norm(upper_lip - lower_lip)
    horizontal_distance = np.linalg.norm(left_corner - right_corner)

    if horizontal_distance < 1e-6:
        logger.warning("Near-zero horizontal mouth distance")
        return 0.0

    lar = float(vertical_distance / horizontal_distance)
    logger.debug("Computed LAR: %.4f", lar)
    return lar


# ======================================================================
# 3. compute_mouth_opening
# ======================================================================


def compute_mouth_opening(landmarks: np.ndarray) -> float:
    """Compute the mouth opening distance from lip landmarks.

    Returns the vertical Euclidean distance between the upper and lower
    inner-lip centres (landmarks 13 and 14), averaged over additional
    neighbouring inner-lip points for robustness.

    Parameters
    ----------
    landmarks : numpy.ndarray
        Array of shape ``(N, 2)`` as returned by :func:`extract_lip_landmarks`.

    Returns
    -------
    float
        Mouth opening in pixels.  Returns ``0.0`` on failure.
    """
    if landmarks is None or landmarks.shape[0] < 20:
        logger.warning("Insufficient landmarks for mouth opening computation")
        return 0.0

    idx_to_pos = {idx: landmarks[i] for i, idx in enumerate(_ALL_LIP_INDICES)}

    # Primary upper/lower inner-lip centre points
    upper_primary = idx_to_pos.get(UPPER_LIP_CENTER_IDX)
    lower_primary = idx_to_pos.get(LOWER_LIP_CENTER_IDX)
    if upper_primary is None or lower_primary is None:
        logger.warning("Primary inner-lip landmarks missing")
        return 0.0

    # Additional inner-lip indices for a robust average
    upper_inner_ids = [82, UPPER_LIP_CENTER_IDX, 312]
    lower_inner_ids = [87, LOWER_LIP_CENTER_IDX, 317]

    upper_pts = np.array([idx_to_pos[i] for i in upper_inner_ids if i in idx_to_pos])
    lower_pts = np.array([idx_to_pos[i] for i in lower_inner_ids if i in idx_to_pos])

    if len(upper_pts) == 0 or len(lower_pts) == 0:
        return float(np.linalg.norm(upper_primary - lower_primary))

    upper_centre = np.mean(upper_pts, axis=0)
    lower_centre = np.mean(lower_pts, axis=0)

    opening = float(np.linalg.norm(upper_centre - lower_centre))
    logger.debug("Computed mouth opening: %.2f px", opening)
    return opening


# ======================================================================
# 4. compute_lip_movement
# ======================================================================


def compute_lip_movement(frame_sequence: List[np.ndarray]) -> np.ndarray:
    """Compute temporal derivative of lip features across a sequence of frames.

    For every frame the Lip Aspect Ratio and mouth-opening distance are
    extracted.  The first-order temporal derivative of each feature is then
    computed via ``np.diff``, and the per-frame movement magnitude is the
    Euclidean norm of the two derivative signals.

    Parameters
    ----------
    frame_sequence : list of numpy.ndarray
        Consecutive video frames (BGR).

    Returns
    -------
    numpy.ndarray
        1-D array of shape ``(len(frame_sequence) - 1,)`` containing the
        movement magnitude between each pair of consecutive frames.  Empty
        when fewer than 2 frames are provided.
    """
    if not frame_sequence or len(frame_sequence) < 2:
        logger.warning("Fewer than 2 frames - cannot compute lip movement")
        return np.array([], dtype=np.float64)

    num_frames = len(frame_sequence)
    lar_seq = np.zeros(num_frames, dtype=np.float64)
    opening_seq = np.zeros(num_frames, dtype=np.float64)

    for i, frame in enumerate(frame_sequence):
        lm = extract_lip_landmarks(frame)
        if lm is not None:
            lar_seq[i] = compute_lip_aspect_ratio(lm)
            opening_seq[i] = compute_mouth_opening(lm)
        else:
            # Forward-fill from the last valid frame
            if i > 0:
                lar_seq[i] = lar_seq[i - 1]
                opening_seq[i] = opening_seq[i - 1]
            logger.debug("Frame %d: no face - forward-filled lip features", i)

    # Temporal first-order derivatives
    lar_deriv = np.diff(lar_seq)
    opening_deriv = np.diff(opening_seq)

    # Combined movement magnitude per frame transition
    movement = np.sqrt(lar_deriv ** 2 + opening_deriv ** 2)

    logger.debug("Lip movement computed: %d transitions over %d frames",
                 len(movement), num_frames)
    return movement


# ======================================================================
# 5. compute_temporal_lip_motion
# ======================================================================


def compute_temporal_lip_motion(lip_features_window: np.ndarray) -> Dict[str, float]:
    """Compute frequency-domain features of lip motion via FFT.

    A Hanning window is applied before the FFT to reduce spectral leakage.
    The DC component is excluded when identifying the dominant frequency
    and computing spectral energy.

    Parameters
    ----------
    lip_features_window : numpy.ndarray
        1-D array of a scalar lip feature (e.g. LAR or mouth-opening)
        sampled uniformly over time.

    Returns
    -------
    dict
        ``'dominant_frequency'`` - FFT bin index of the peak magnitude
        (excluding DC), in units of *cycles per window*.\n
        ``'spectral_energy'`` - total power (sum of squared magnitudes)
        excluding DC.
    """
    if lip_features_window is None or len(lip_features_window) < 4:
        logger.warning("Window too short (%d samples) for FFT analysis",
                        0 if lip_features_window is None else len(lip_features_window))
        return {"dominant_frequency": 0.0, "spectral_energy": 0.0}

    # Remove DC (mean) and apply a Hanning window
    centered = lip_features_window.astype(np.float64) - np.mean(lip_features_window)
    n = len(centered)
    window_func = np.hanning(n)
    windowed = centered * window_func

    # Real FFT (input is real-valued)
    fft_coeffs = np.fft.rfft(windowed)
    magnitudes = np.abs(fft_coeffs)
    power = magnitudes ** 2

    # Exclude DC (index 0)
    if magnitudes.shape[0] <= 1:
        return {"dominant_frequency": 0.0, "spectral_energy": 0.0}

    mag_no_dc = magnitudes[1:]
    pwr_no_dc = power[1:]

    dominant_bin = int(np.argmax(mag_no_dc))
    dominant_frequency = float(dominant_bin + 1)  # +1 because DC was removed
    spectral_energy = float(np.sum(pwr_no_dc))

    logger.debug("Temporal lip motion - dominant_freq: %.2f, spectral_energy: %.4f",
                 dominant_frequency, spectral_energy)
    return {
        "dominant_frequency": dominant_frequency,
        "spectral_energy": spectral_energy,
    }


# ======================================================================
# 6. extract_audio_features
# ======================================================================


def extract_audio_features(video_path: str) -> Dict[str, np.ndarray]:
    """Extract a rich set of audio features from a video file.

    Pipeline:

    1. **MoviePy** reads the video and exports the audio track as a NumPy
       array at its native sample rate.
    2. The signal is down-mixed to mono (if required by config) and
       normalised to ``[-1, 1]``.
    3. **Librosa** resamples to ``config.AUDIO_SAMPLE_RATE`` (16 kHz) and
       computes:

       - 13 MFCCs
       - Spectral centroid
       - Spectral bandwidth
       - Zero-crossing rate
       - 12-dimensional chroma features
       - 128-band Mel spectrogram
       - Frame-wise RMS energy

    Parameters
    ----------
    video_path : str
        Absolute or relative path to the source video.

    Returns
    -------
    dict
        Keys: ``'mfccs'``, ``'spectral_centroid'``, ``'spectral_bandwidth'``,
        ``'zero_crossing_rate'``, ``'chroma'``, ``'mel_spectrogram'``,
        ``'audio_energy'``.  Empty dict on failure.
    """
    if not os.path.isfile(video_path):
        logger.error("Video file not found: %s", video_path)
        return {}

    try:
        video_clip = mpy.VideoFileClip(video_path)

        if video_clip.audio is None:
            logger.warning("No audio track in video: %s", video_path)
            video_clip.close()
            return {}

        # Extract raw audio as float64 array at the clip's native sample rate
        audio_raw = video_clip.audio.to_soundarray()
        native_fps = video_clip.audio.fps
        video_clip.close()

        if audio_raw.size == 0:
            logger.warning("Empty audio array extracted from %s", video_path)
            return {}

        # ---- Mono conversion ----
        if audio_raw.ndim > 1:
            if config.AUDIO_MONO:
                audio_mono = np.mean(audio_raw, axis=1)
            else:
                audio_mono = audio_raw[:, 0]
        else:
            audio_mono = audio_raw

        # ---- Normalise to [-1, 1] ----
        audio_mono = audio_mono.astype(np.float32)
        peak = np.max(np.abs(audio_mono))
        if peak > 0:
            audio_mono = audio_mono / peak

        # ---- Resample to target sample rate ----
        if native_fps != config.AUDIO_SAMPLE_RATE:
            audio_mono = librosa.resample(
                audio_mono,
                orig_sr=native_fps,
                target_sr=config.AUDIO_SAMPLE_RATE,
            )
        sr = config.AUDIO_SAMPLE_RATE

        # ---- Feature extraction ----
        mfccs = librosa.feature.mfcc(y=audio_mono, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_mono, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_mono, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_mono)
        chroma = librosa.feature.chroma_stft(y=audio_mono, sr=sr)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_mono, sr=sr, n_mels=128)
        rms = librosa.feature.rms(y=audio_mono)
        audio_energy = rms[0]  # shape (T,)

        features: Dict[str, np.ndarray] = {
            "mfccs": mfccs,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "zero_crossing_rate": zero_crossing_rate,
            "chroma": chroma,
            "mel_spectrogram": mel_spectrogram,
            "audio_energy": audio_energy,
        }

        logger.info(
            "Audio features extracted from %s - MFCCs %s, energy frames %d",
            video_path, mfccs.shape, len(audio_energy),
        )
        return features

    except Exception as exc:
        logger.error("Failed to extract audio features from %s: %s", video_path, exc)
        return {}


# ======================================================================
# Internal: DTW distance
# ======================================================================


def _compute_dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Compute the Dynamic Time Warping distance between two 1-D sequences.

    Standard O(n·m) dynamic-programming formulation using squared-Euclidean
    local cost and returning the square root of the optimal accumulated cost.

    Parameters
    ----------
    seq1, seq2 : numpy.ndarray
        1-D sequences.

    Returns
    -------
    float
        DTW distance (≥ 0).  Returns ``inf`` when either sequence is empty.
    """
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return float("inf")

    # Cost matrix initialised to infinity; (0, 0) = 0
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (seq1[i - 1] - seq2[j - 1]) ** 2
            dtw[i, j] = cost + min(
                dtw[i - 1, j],      # insertion
                dtw[i, j - 1],       # deletion
                dtw[i - 1, j - 1],   # match
            )

    return float(np.sqrt(dtw[n, m]))


# ======================================================================
# 7. compute_lip_sync_score
# ======================================================================


def compute_lip_sync_score(
    lip_features: Dict[str, np.ndarray],
    audio_features: Dict[str, np.ndarray],
) -> float:
    """Compute a normalised lip-sync score in [0, 1].

    The score combines two complementary metrics:

    1. **Pearson correlation** between the (smoothed) lip-movement energy
       and the audio energy - measures linear co-variation.
    2. **DTW distance** between the same two signals - measures temporal
       alignment irrespective of non-linear timing differences.

    Both metrics are normalised to ``[0, 1]`` and blended with a 60/40
    weighting (correlation / DTW).  A score of **1** means perfectly
    synchronised; **0** means completely desynchronised.

    Parameters
    ----------
    lip_features : dict
        Must contain ``'lip_movements'`` - a 1-D numpy array of per-frame
        movement magnitudes.
    audio_features : dict
        Must contain ``'audio_energy'`` - a 1-D numpy array of per-audio-frame
        RMS energy values.

    Returns
    -------
    float
        Synchronisation score in ``[0, 1]``.  Returns ``0.0`` on error.
    """
    lip_movements = lip_features.get("lip_movements")
    audio_energy = audio_features.get("audio_energy")

    if lip_movements is None or audio_energy is None:
        logger.error("Missing 'lip_movements' or 'audio_energy' for sync score")
        return 0.0

    if len(lip_movements) < 3 or len(audio_energy) < 3:
        logger.error("Insufficient data for sync scoring (lip=%d, audio=%d)",
                      len(lip_movements), len(audio_energy))
        return 0.0

    # ---- Resample lip movements to audio-energy length via linear interp ----
    lip_resampled = np.interp(
        np.linspace(0.0, 1.0, len(audio_energy)),
        np.linspace(0.0, 1.0, len(lip_movements)),
        lip_movements,
    )

    # ---- Smooth both signals (moving-average, window ≤ 5) ----
    win = min(5, max(1, len(lip_resampled) // 4))
    if win >= 3 and win % 2 == 0:
        win += 1  # keep odd for symmetry
    if win >= 3:
        kernel = np.ones(win) / win
        lip_smooth = np.convolve(lip_resampled, kernel, mode="same")
        aud_smooth = np.convolve(audio_energy, kernel, mode="same")
    else:
        lip_smooth = lip_resampled
        aud_smooth = audio_energy

    # ---- 1. Pearson correlation ----
    try:
        correlation, _ = pearsonr(lip_smooth, aud_smooth)
        correlation = float(np.clip(correlation, -1.0, 1.0))
        logger.debug("Pearson r = %.4f", correlation)
    except Exception as exc:
        logger.warning("Pearson correlation failed: %s", exc)
        correlation = 0.0

    # ---- 2. DTW distance ----
    try:
        dtw_dist = _compute_dtw_distance(lip_smooth, aud_smooth)
        logger.debug("DTW distance = %.4f", dtw_dist)
    except Exception as exc:
        logger.warning("DTW computation failed: %s", exc)
        dtw_dist = float("inf")

    # ---- 3. Normalise correlation to [0, 1] ----
    # Only positive correlation is meaningful for sync (negative → 0)
    corr_score = max(0.0, correlation)

    # ---- 4. Normalise DTW to [0, 1] ----
    max_len = max(len(lip_smooth), len(aud_smooth))
    if max_len > 0 and np.isfinite(dtw_dist):
        audio_std = float(np.std(aud_smooth)) + 1e-8
        normalised_dtw = dtw_dist / (max_len * audio_std)
        dtw_score = 1.0 / (1.0 + normalised_dtw)
    else:
        dtw_score = 0.0

    # ---- 5. Weighted blend ----
    alpha = 0.6  # weight for Pearson correlation
    sync_score = float(np.clip(alpha * corr_score + (1.0 - alpha) * dtw_score, 0.0, 1.0))

    logger.info("Lip-sync score = %.4f  (corr=%.4f, dtw_score=%.4f)",
                sync_score, corr_score, dtw_score)
    return sync_score


# ======================================================================
# 8. analyze_lip_sync
# ======================================================================


def analyze_lip_sync(
    video_path: str,
    frames: List[np.ndarray],
) -> Dict[str, Union[np.ndarray, float, str, Dict]]:
    """Run the full lip-synchronisation analysis pipeline.

    Steps
    -----
    1. Extract per-frame lip landmarks, LAR, and mouth-opening distance.
    2. Compute temporal lip-movement magnitudes (first-order derivative).
    3. Compute FFT-based temporal frequency features on a sliding window.
    4. Extract audio features from the video file.
    5. Compute the lip-sync score (Pearson + DTW).
    6. Classify as ``'synced'`` or ``'desynced'`` using
       :pydata:`config.LIP_SYNC_THRESHOLD`.

    Parameters
    ----------
    video_path : str
        Path to the source video (used for audio extraction).
    frames : list of numpy.ndarray
        Video frames (BGR) for visual lip analysis.

    Returns
    -------
    dict
        ``lip_aspect_ratios`` : numpy.ndarray - per-frame LAR.
        ``mouth_openings``     : numpy.ndarray - per-frame mouth opening (px).
        ``lip_movements``      : numpy.ndarray - movement magnitude array.
        ``temporal_features``  : dict - ``dominant_frequency``, ``spectral_energy``.
        ``audio_features``     : dict - full audio feature set (or empty).
        ``lip_sync_score``     : float - ``[0, 1]``.
        ``lip_sync_label``     : str - ``'synced'`` or ``'desynced'``.
    """
    num_frames = len(frames)
    logger.info("Starting lip-sync analysis: %d frames from %s", num_frames, video_path)

    # ------------------------------------------------------------------
    # Step 1 - per-frame lip geometry
    # ------------------------------------------------------------------
    lip_aspect_ratios = np.zeros(num_frames, dtype=np.float64)
    mouth_openings = np.zeros(num_frames, dtype=np.float64)
    valid_count = 0

    for i, frame in enumerate(frames):
        lm = extract_lip_landmarks(frame)
        if lm is not None:
            lip_aspect_ratios[i] = compute_lip_aspect_ratio(lm)
            mouth_openings[i] = compute_mouth_opening(lm)
            valid_count += 1
        else:
            # Forward-fill from previous valid frame
            if i > 0:
                lip_aspect_ratios[i] = lip_aspect_ratios[i - 1]
                mouth_openings[i] = mouth_openings[i - 1]
            logger.debug("Frame %d: no face detected, values forward-filled", i)

    logger.info("Faces detected in %d / %d frames", valid_count, num_frames)

    # ------------------------------------------------------------------
    # Step 2 - temporal lip movement
    # ------------------------------------------------------------------
    lip_movements = compute_lip_movement(frames)

    # ------------------------------------------------------------------
    # Step 3 - FFT temporal features (use last WINDOW_SIZE frames)
    # ------------------------------------------------------------------
    window_size = config.LIP_SYNC_WINDOW_SIZE
    if len(mouth_openings) >= window_size:
        windowed_openings = mouth_openings[-window_size:]
    else:
        windowed_openings = mouth_openings

    temporal_features = compute_temporal_lip_motion(windowed_openings)

    # ------------------------------------------------------------------
    # Step 4 - audio feature extraction
    # ------------------------------------------------------------------
    audio_features = extract_audio_features(video_path)

    # ------------------------------------------------------------------
    # Step 5 - lip-sync score
    # ------------------------------------------------------------------
    lip_feature_dict: Dict[str, np.ndarray] = {
        "lip_movements": lip_movements,
        "lip_aspect_ratios": lip_aspect_ratios,
        "mouth_openings": mouth_openings,
    }

    if audio_features and len(lip_movements) > 0:
        lip_sync_score = compute_lip_sync_score(lip_feature_dict, audio_features)
    else:
        logger.warning("Cannot compute lip-sync score - missing audio or lip data")
        lip_sync_score = 0.0

    # ------------------------------------------------------------------
    # Step 6 - classification
    # ------------------------------------------------------------------
    lip_sync_label = "synced" if lip_sync_score >= config.LIP_SYNC_THRESHOLD else "desynced"

    result: Dict[str, Union[np.ndarray, float, str, Dict]] = {
        "lip_aspect_ratios": lip_aspect_ratios,
        "mouth_openings": mouth_openings,
        "lip_movements": lip_movements,
        "temporal_features": temporal_features,
        "audio_features": audio_features,
        "lip_sync_score": lip_sync_score,
        "lip_sync_label": lip_sync_label,
    }

    logger.info(
        "Lip-sync analysis complete: score=%.4f, label='%s'",
        lip_sync_score, lip_sync_label,
    )
    return result
