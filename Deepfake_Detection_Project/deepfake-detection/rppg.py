"""
Remote Photoplethysmography (rPPG) Module for AI-Based Multimodal Deepfake Video Detection.

This module implements non-contact heart pulse estimation from video frames using
remote photoplethysmography. It extracts subtle skin colour variations caused by
blood volume changes and derives physiological signals (heart rate, pulse waveform,
stability) that serve as liveness indicators. Deepfake videos typically lack
authentic physiological signals, making rPPG a powerful discriminative feature.

Functions:
    extract_face_roi          - Extract forehead, left cheek, right cheek ROIs.
    compute_rgb_signals       - Compute mean RGB channel values across ROIs and frames.
    bandpass_filter           - Butterworth bandpass filter for physiological frequencies.
    compute_heart_rate_fft    - FFT-based dominant heart rate estimation in BPM.
    compute_pulse_signal      - POS (Plane-Orthogonal-to-Skin) chrominance pulse extraction.
    compute_pulse_stability   - Rolling window amplitude stability metric.
    compute_rppg_confidence   - Multi-criteria confidence score (0-1).
    extract_rppg_features     - End-to-end rPPG feature extraction pipeline.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy import ndimage as scipy_ndimage
from scipy.signal import butter, filtfilt

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ROI coordinate ratios (relative to a 224x224 aligned face image)
# ---------------------------------------------------------------------------
# Each ROI is defined as (y_start_ratio, y_end_ratio, x_start_ratio, x_end_ratio)
# Values chosen based on standard facial landmark atlases for rPPG.
_ROI_DEFINITIONS: Dict[str, Tuple[float, float, float, float]] = {
    "forehead":     (0.05, 0.35, 0.20, 0.80),
    "left_cheek":   (0.35, 0.70, 0.02, 0.30),
    "right_cheek":  (0.35, 0.70, 0.70, 0.98),
}

# ---------------------------------------------------------------------------
# POS method constants
# ---------------------------------------------------------------------------
# The Plane-Orthogonal-to-Skin projection matrix (Wang, den Brinker, de Haan 2017).
# Derived from the empirical skin-tone chrominance model.
_POS_S = np.array([[0.0, 1.0, -1.0],
                   [-2.0, 1.0, 1.0]], dtype=np.float64)

# Pre-computed POS bandpass filter taps (frame-index domain, not frequency).
# Uses a 1st-order IIR approximation equivalent to the paper's temporal filter
# applied in the frame-index domain.  h[n] = (1/n) for n = 1..N  (moving average).
_POS_H_FRAME_SCALE = 0.8  # fraction of total frames used for the averaging window

# Heart rate plausibility bounds (BPM)
_HR_MIN_BPM = 40.0
_HR_MAX_BPM = 180.0

# Heart rate frequency bounds (Hz)
_HR_MIN_HZ = _HR_MIN_BPM / 60.0
_HR_MAX_HZ = _HR_MAX_BPM / 60.0

# Confidence weighting factors
_HR_WEIGHT = 0.35
_SNR_WEIGHT = 0.40
_STABILITY_WEIGHT = 0.25


# ==========================================================================
# 1. Extract face ROI
# ==========================================================================

def extract_face_roi(face_image: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract regions of interest (forehead, left cheek, right cheek) from an
    aligned 224x224 face image.

    Parameters
    ----------
    face_image : np.ndarray
        Aligned face crop of shape (224, 224, 3) in BGR or RGB order (returned
        as-is from the source; the caller is responsible for colour space).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of ROI name to cropped image patch.
        Keys: ``"forehead"``, ``"left_cheek"``, ``"right_cheek"``.
        Each value has shape ``(H_roi, W_roi, 3)``.
    """
    h, w = face_image.shape[:2]
    rois: Dict[str, np.ndarray] = {}

    for name, (y0r, y1r, x0r, x1r) in _ROI_DEFINITIONS.items():
        y0 = int(round(y0r * h))
        y1 = int(round(y1r * h))
        x0 = int(round(x0r * w))
        x1 = int(round(x1r * w))

        # Clamp to image bounds (safety for non-standard sizes)
        y0 = max(0, min(y0, h))
        y1 = max(0, min(y1, h))
        x0 = max(0, min(x0, w))
        x1 = max(0, min(x1, w))

        if y1 <= y0 or x1 <= x0:
            logger.warning(
                "ROI '%s' produced an empty crop after clamping "
                "(y=%d:%d, x=%d:%d) for image shape %s.",
                name, y0, y1, x0, x1, face_image.shape,
            )
            rois[name] = np.zeros((1, 1, 3), dtype=face_image.dtype)
            continue

        rois[name] = face_image[y0:y1, x0:x1].copy()
        logger.debug(
            "Extracted ROI '%s': shape=%s, bounds=(y=%d:%d, x=%d:%d)",
            name, rois[name].shape, y0, y1, x0, x1,
        )

    return rois


# ==========================================================================
# 2. Compute RGB signals
# ==========================================================================

def compute_rgb_signals(
    roi_images: List[Dict[str, np.ndarray]],
    frames_sequence: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute mean R, G, B channel values for each frame across ROIs.

    For every frame, the mean of each colour channel is computed inside each
    of the three ROIs.  The per-ROI signals are then averaged across ROIs to
    produce a single robust R, G, B temporal signal.

    Parameters
    ----------
    roi_images : list[dict[str, np.ndarray]]
        Length-*N* list where each element is the dict returned by
        :func:`extract_face_roi` for one frame.
    frames_sequence : np.ndarray, optional
        Raw frames array (unused directly; kept for API compatibility and
        potential future extensions that may need raw pixel data).

    Returns
    -------
    dict[str, np.ndarray]
        Keys ``"R"``, ``"G"``, ``"B"`` each mapping to a 1-D float64 array
        of length *N* (the number of frames) representing the mean channel
        intensity averaged across the three ROIs.
    """
    n_frames = len(roi_images)
    roi_names = ["forehead", "left_cheek", "right_cheek"]

    # Accumulators: shape (n_frames, n_channels)
    # channels are ordered as [B, G, R] because OpenCV uses BGR by default.
    # However, we always report R, G, B regardless of input colour space.
    accumulator = np.zeros((n_frames, 3), dtype=np.float64)
    valid_counts = np.zeros(n_frames, dtype=np.int32)

    for frame_idx, roi_dict in enumerate(roi_images):
        for roi_name in roi_names:
            roi = roi_dict.get(roi_name)
            if roi is None or roi.size == 0:
                continue
            # Detect colour space: assume BGR (OpenCV default).
            # We treat channel index 0→B, 1→G, 2→R.
            bgr_means = roi.astype(np.float64).mean(axis=(0, 1))
            # Convert to R, G, B order for output.
            accumulator[frame_idx, 0] += bgr_means[2]  # R
            accumulator[frame_idx, 1] += bgr_means[1]  # G
            accumulator[frame_idx, 2] += bgr_means[0]  # B
            valid_counts[frame_idx] += 1

    # Average across valid ROIs
    mask = valid_counts > 0
    if not np.any(mask):
        logger.warning("No valid ROI data in compute_rgb_signals.")
        return {"R": np.array([]), "G": np.array([]), "B": np.array([])}

    accumulator[mask] /= valid_counts[mask, np.newaxis]

    # For frames with no valid ROIs, forward-fill or zero-fill
    if not np.all(mask):
        logger.debug(
            "Forward-filling %d frames with no valid ROI data.",
            int(np.sum(~mask)),
        )
        # Forward-fill from nearest valid frame
        last_valid_idx = None
        for i in range(n_frames):
            if mask[i]:
                last_valid_idx = i
            elif last_valid_idx is not None:
                accumulator[i] = accumulator[last_valid_idx]
            else:
                accumulator[i] = 0.0

    rgb_signals = {
        "R": accumulator[:, 0].copy(),
        "G": accumulator[:, 1].copy(),
        "B": accumulator[:, 2].copy(),
    }

    logger.debug(
        "RGB signals computed: %d frames, R range=[%.3f, %.3f], "
        "G range=[%.3f, %.3f], B range=[%.3f, %.3f]",
        n_frames,
        rgb_signals["R"].min(), rgb_signals["R"].max(),
        rgb_signals["G"].min(), rgb_signals["G"].max(),
        rgb_signals["B"].min(), rgb_signals["B"].max(),
    )

    return rgb_signals


# ==========================================================================
# 3. Bandpass filter
# ==========================================================================

def bandpass_filter(
    signal: np.ndarray,
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None,
    fps: Optional[float] = None,
) -> np.ndarray:
    """Apply a 4th-order Butterworth bandpass filter using zero-phase filtering.

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal.
    lowcut : float, optional
        Lower cutoff frequency in Hz.  Defaults to ``config.RPPG_BANDPASS_LOW``
        (0.7 Hz).
    highcut : float, optional
        Upper cutoff frequency in Hz.  Defaults to ``config.RPPG_BANDPASS_HIGH``
        (4.0 Hz).
    fps : float, optional
        Sampling frequency in Hz.  Defaults to ``config.RPPG_FPS`` (30).

    Returns
    -------
    np.ndarray
        Filtered signal of the same length as *signal*.
    """
    if lowcut is None:
        lowcut = config.RPPG_BANDPASS_LOW
    if highcut is None:
        highcut = config.RPPG_BANDPASS_HIGH
    if fps is None:
        fps = float(config.RPPG_FPS)

    if len(signal) < 4:
        logger.warning(
            "Signal too short (%d samples) for bandpass filter; returning zeros.",
            len(signal),
        )
        return np.zeros_like(signal, dtype=np.float64)

    nyquist = fps / 2.0

    # Guard against invalid cutoff frequencies
    low = np.clip(lowcut / nyquist, 0.001, 0.999)
    high = np.clip(highcut / nyquist, 0.001, 0.999)

    if low >= high:
        logger.warning(
            "Bandpass lowcut (%.2f Hz) >= highcut (%.2f Hz); "
            "swapping and clamping.",
            lowcut, highcut,
        )
        low, high = 0.001, 0.999

    try:
        # 4th-order Butterworth (order=2 for each pass → 4th total with filtfilt)
        b, a = butter(N=2, Wn=[low, high], btype="band")
        filtered = filtfilt(b, a, signal.astype(np.float64))
    except Exception as exc:
        logger.error("Bandpass filter failed: %s. Returning original signal.", exc)
        filtered = signal.astype(np.float64).copy()

    return filtered


# ==========================================================================
# 4. Heart rate estimation via FFT
# ==========================================================================

def compute_heart_rate_fft(
    filtered_signal: np.ndarray,
    fps: Optional[float] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Estimate heart rate from a filtered pulse signal using FFT.

    The dominant frequency within the valid heart-rate range (42–180 BPM,
    i.e. 0.7–3.0 Hz) is identified and converted to beats per minute.

    Parameters
    ----------
    filtered_signal : np.ndarray
        1-D bandpass-filtered pulse signal.
    fps : float, optional
        Sampling rate in Hz.  Defaults to ``config.RPPG_FPS`` (30).

    Returns
    -------
    heart_rate : float
        Estimated heart rate in BPM.  Returns ``0.0`` if estimation fails.
    fft_frequencies : np.ndarray
        Frequency bins (Hz) from the FFT.
    fft_magnitudes : np.ndarray
        Magnitude spectrum corresponding to *fft_frequencies*.
    """
    if fps is None:
        fps = float(config.RPPG_FPS)

    n = len(filtered_signal)
    default_hr = 0.0
    empty_freqs = np.array([], dtype=np.float64)
    empty_mags = np.array([], dtype=np.float64)

    if n < 4:
        logger.warning(
            "Signal too short (%d samples) for FFT-based HR estimation.", n,
        )
        return default_hr, empty_freqs, empty_mags

    # Remove DC offset and apply Hanning window to reduce spectral leakage
    signal_centered = filtered_signal - np.mean(filtered_signal)
    window = np.hanning(n)
    windowed_signal = signal_centered * window

    # Compute one-sided FFT
    fft_result = np.fft.rfft(windowed_signal)
    fft_magnitudes = np.abs(fft_result)
    fft_frequencies = np.fft.rfftfreq(n, d=1.0 / fps)

    # Restrict search to plausible HR range: 42–180 BPM → 0.7–3.0 Hz
    hr_min_hz = 42.0 / 60.0   # 0.7 Hz
    hr_max_hz = 180.0 / 60.0  # 3.0 Hz

    valid_mask = (fft_frequencies >= hr_min_hz) & (fft_frequencies <= hr_max_hz)

    if not np.any(valid_mask):
        logger.warning("No FFT bins fall within the valid HR frequency range.")
        return default_hr, fft_frequencies, fft_magnitudes

    valid_frequencies = fft_frequencies[valid_mask]
    valid_magnitudes = fft_magnitudes[valid_mask]

    # Find the dominant peak
    peak_idx = np.argmax(valid_magnitudes)
    dominant_freq = valid_frequencies[peak_idx]
    heart_rate = dominant_freq * 60.0  # Convert Hz → BPM

    logger.debug(
        "HR estimate: %.1f BPM (dominant freq = %.3f Hz, "
        "peak magnitude = %.4f)",
        heart_rate, dominant_freq, valid_magnitudes[peak_idx],
    )

    return float(heart_rate), fft_frequencies, fft_magnitudes


# ==========================================================================
# 5. POS pulse signal
# ==========================================================================

def compute_pulse_signal(
    rgb_signals: Dict[str, np.ndarray],
    fps: Optional[float] = None,
) -> np.ndarray:
    """Extract the pulse signal using the Plane-Orthogonal-to-Skin (POS) method.

    The POS algorithm projects the raw RGB temporal traces onto a plane
    orthogonal to the skin-tone direction, effectively suppressing motion
    artefacts and isolating the pulsatile component.

    Reference: Wang, den Brinker, de Haan, "Single-Element Robust 
    rPPG Using a Color Camera", IEEE TBIOM 2021.

    Parameters
    ----------
    rgb_signals : dict[str, np.ndarray]
        Dictionary with keys ``"R"``, ``"G"``, ``"B"``, each mapping to a
        1-D array of length *N* (number of frames).
    fps : float, optional
        Sampling rate in Hz.  Defaults to ``config.RPPG_FPS`` (30).

    Returns
    -------
    np.ndarray
        1-D pulse signal of length *N*.
    """
    if fps is None:
        fps = float(config.RPPG_FPS)

    r = np.asarray(rgb_signals.get("R", []), dtype=np.float64)
    g = np.asarray(rgb_signals.get("G", []), dtype=np.float64)
    b = np.asarray(rgb_signals.get("B", []), dtype=np.float64)

    n = len(r)
    if n == 0:
        logger.warning("Empty RGB signals provided to compute_pulse_signal.")
        return np.array([], dtype=np.float64)

    # Stack into (3, N) matrix
    rgb_stack = np.vstack([r, g, b])  # shape (3, N)

    # Apply the POS projection: S = W @ C  where C is (3, N)
    # S has shape (2, N)
    S = _POS_S @ rgb_stack  # (2, N)

    # --- Temporal filtering (frame-index domain) ---
    # The POS paper uses a linear FIR filter h[n] = 1/n for n = 1..N_frames
    # applied as a moving average in the frame-index domain, followed by
    # subtraction and bandpass filtering.
    #
    # Step 1: Compute the moving-average filtered version of each POS channel.
    #   window length L = min(N, round(0.8 * N))
    L = max(1, min(n, int(round(_POS_H_FRAME_SCALE * n))))

    # Cumulative sum for efficient moving average
    cumsum_ch0 = np.cumsum(S[0])
    cumsum_ch1 = np.cumsum(S[1])

    # Moving average of the full window
    def _moving_avg(cumsum: np.ndarray, win_len: int) -> np.ndarray:
        padded = np.concatenate([[0.0], cumsum])
        return (padded[win_len:] - padded[:-win_len]) / win_len

    ma_ch0 = _moving_avg(cumsum_ch0, L)
    ma_ch1 = _moving_avg(cumsum_ch1, L)

    # Align lengths (moving average shortens the signal by L-1)
    min_len = min(len(ma_ch0), len(ma_ch1))
    ma_ch0 = ma_ch0[:min_len]
    ma_ch1 = ma_ch1[:min_len]

    # Step 2: Combine the two POS channels using the standard deviation weighting
    #   h_i = std(ch_i) / (std(ch_0) + std(ch_1))
    std_ch0 = np.std(ma_ch0)
    std_ch1 = np.std(ma_ch1)
    denom = std_ch0 + std_ch1

    if denom < 1e-10:
        logger.warning("POS channels have near-zero combined std; returning zero signal.")
        pulse = np.zeros(n, dtype=np.float64)
    else:
        h0 = std_ch0 / denom
        h1 = std_ch1 / denom
        pulse_raw = h0 * ma_ch0 - h1 * ma_ch1

        # Pad/truncate back to original length n
        if len(pulse_raw) < n:
            pulse = np.zeros(n, dtype=np.float64)
            pulse[-len(pulse_raw):] = pulse_raw
        else:
            pulse = pulse_raw[:n]

    # Step 3: Bandpass filter the combined pulse signal
    pulse_filtered = bandpass_filter(pulse, fps=fps)

    logger.debug(
        "POS pulse signal: %d samples, range=[%.6f, %.6f], "
        "std=%.6f",
        n, pulse_filtered.min(), pulse_filtered.max(),
        np.std(pulse_filtered),
    )

    return pulse_filtered


# ==========================================================================
# 6. Pulse stability
# ==========================================================================

def compute_pulse_stability(
    pulse_signal: np.ndarray,
    window_size: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """Compute rolling standard deviation of pulse signal amplitude.

    A stable (authentic) physiological pulse has consistent amplitude across
    time.  Deepfake videos often exhibit irregular or absent pulse rhythms.

    Parameters
    ----------
    pulse_signal : np.ndarray
        1-D filtered pulse signal.
    window_size : int, optional
        Number of samples per rolling window.  Defaults to
        ``config.RPPG_WINDOW_SIZE`` (300).

    Returns
    -------
    stability_score : float
        Value in [0, 1] where 1 indicates a perfectly stable pulse signal.
        Computed as ``1 - normalised_mean_std``, clamped to [0, 1].
    stability_values : np.ndarray
        1-D array of rolling standard deviation values for each window centre.
    """
    if window_size is None:
        window_size = config.RPPG_WINDOW_SIZE

    n = len(pulse_signal)
    default_return = (0.0, np.array([], dtype=np.float64))

    if n < 4:
        logger.warning(
            "Pulse signal too short (%d samples) for stability computation.", n,
        )
        return default_return

    # Adapt window size if signal is shorter than requested window
    win = min(window_size, n)
    if win < 2:
        return default_return

    # Compute rolling standard deviation using a uniform filter.
    # 1. Rolling mean
    mean_signal = scipy_ndimage.uniform_filter1d(pulse_signal, size=win, mode="reflect")
    # 2. Rolling mean of squares
    sq_signal = pulse_signal ** 2
    mean_sq = scipy_ndimage.uniform_filter1d(sq_signal, size=win, mode="reflect")
    # 3. Rolling variance = E[X^2] - (E[X])^2
    rolling_var = mean_sq - (mean_signal ** 2)
    # Guard against tiny negative values from floating-point noise
    rolling_var = np.maximum(rolling_var, 0.0)
    # 4. Rolling standard deviation
    stability_values = np.sqrt(rolling_var)

    # Compute stability score: low relative variability → high stability
    # Normalise the mean std by the overall signal range
    signal_range = np.ptp(pulse_signal)  # max - min
    if signal_range < 1e-10:
        # Flat signal — indeterminate stability
        stability_score = 0.5
    else:
        mean_std = np.mean(stability_values)
        # The ratio of mean std to signal range indicates variability.
        # Typical physiological signals have ratios < 0.3.
        normalised_mean_std = np.clip(mean_std / signal_range, 0.0, 1.0)
        stability_score = float(np.clip(1.0 - normalised_mean_std, 0.0, 1.0))

    logger.debug(
        "Pulse stability: score=%.4f, mean_std=%.6f, signal_range=%.6f, "
        "n_windows=%d",
        stability_score,
        float(np.mean(stability_values)) if len(stability_values) > 0 else 0.0,
        signal_range,
        len(stability_values),
    )

    return stability_score, stability_values


# ==========================================================================
# 7. rPPG confidence
# ==========================================================================

def compute_rppg_confidence(
    heart_rate: float,
    pulse_signal: np.ndarray,
    stability: float,
) -> float:
    """Compute an overall confidence score for the rPPG estimation.

    The score combines three criteria:

    1. **Heart-rate plausibility** — HR in [40, 180] BPM is fully plausible;
       scores decrease smoothly outside this range.
    2. **Signal-to-noise ratio (SNR)** — ratio of pulse signal power in the
       cardiac band to total power.
    3. **Pulse stability** — the stability metric from
       :func:`compute_pulse_stability`.

    Parameters
    ----------
    heart_rate : float
        Estimated heart rate in BPM (0 if unavailable).
    pulse_signal : np.ndarray
        1-D filtered pulse signal.
    stability : float
        Stability score in [0, 1] from :func:`compute_pulse_stability`.

    Returns
    -------
    float
        Confidence score in [0, 1].
    """
    # --- 1. HR plausibility score ---
    if heart_rate <= 0.0:
        hr_score = 0.0
    elif _HR_MIN_BPM <= heart_rate <= _HR_MAX_BPM:
        hr_score = 1.0
    elif heart_rate < _HR_MIN_BPM:
        # Linear decay from 1.0 at HR_MIN to 0.0 at 0 BPM
        hr_score = max(0.0, heart_rate / _HR_MIN_BPM)
    else:
        # Linear decay from 1.0 at HR_MAX to 0.0 at 240 BPM
        hr_score = max(0.0, 1.0 - (heart_rate - _HR_MAX_BPM) / (_HR_MAX_BPM * 0.33))

    # --- 2. SNR of the pulse signal ---
    n = len(pulse_signal)
    if n < 8:
        snr_score = 0.0
    else:
        # Total signal power
        total_power = np.sum(pulse_signal ** 2)
        if total_power < 1e-20:
            snr_score = 0.0
        else:
            # Power in the cardiac frequency band using FFT
            fft_vals = np.fft.rfft(pulse_signal)
            fft_freqs = np.fft.rfftfreq(n)
            fft_power = np.abs(fft_vals) ** 2

            # Cardiac band: 0.7–4.0 Hz
            cardiac_mask = (fft_freqs >= config.RPPG_BANDPASS_LOW) & (
                fft_freqs <= config.RPPG_BANDPASS_HIGH
            )
            cardiac_power = np.sum(fft_power[cardiac_mask]) if np.any(cardiac_mask) else 0.0

            # Noise power = total - cardiac
            noise_power = total_power - cardiac_power
            if noise_power < 1e-20:
                snr_score = 1.0
            else:
                snr_linear = cardiac_power / noise_power
                # Map SNR to [0, 1] using a sigmoid-like transform
                # SNR > 10 → ~1.0; SNR < 0.1 → ~0.0
                snr_score = float(np.clip(snr_linear / (snr_linear + 1.0), 0.0, 1.0))

    # --- 3. Stability score (already in [0, 1]) ---
    stab_score = float(np.clip(stability, 0.0, 1.0))

    # --- Weighted combination ---
    confidence = (
        _HR_WEIGHT * hr_score
        + _SNR_WEIGHT * snr_score
        + _STABILITY_WEIGHT * stab_score
    )
    confidence = float(np.clip(confidence, 0.0, 1.0))

    logger.debug(
        "rPPG confidence: %.4f (HR=%.1f, hr_score=%.3f, "
        "snr_score=%.3f, stab_score=%.3f)",
        confidence, heart_rate, hr_score, snr_score, stab_score,
    )

    return confidence


# ==========================================================================
# 8. Main pipeline
# ==========================================================================

def extract_rppg_features(
    frames: Optional[np.ndarray] = None,
    face_images: Optional[np.ndarray] = None,
    fps: Optional[float] = None,
) -> Dict[str, Union[float, np.ndarray, Dict[str, np.ndarray]]]:
    """End-to-end rPPG feature extraction pipeline.

    Given a sequence of aligned face images, this function:

    1. Extracts ROIs (forehead, left/right cheek) from each face.
    2. Computes mean RGB temporal signals across ROIs.
    3. Applies bandpass filtering to isolate the cardiac frequency band.
    4. Estimates heart rate via FFT peak detection.
    5. Extracts the pulse waveform using the POS method.
    6. Computes pulse stability and overall rPPG confidence.

    Parameters
    ----------
    frames : np.ndarray, optional
        Raw video frames of shape ``(N, H, W, 3)``.  May be ``None`` if
        *face_images* is provided directly.
    face_images : np.ndarray
        Aligned face crops of shape ``(N, 224, 224, 3)`` in BGR order.
    fps : float, optional
        Video frame rate in Hz.  Defaults to ``config.RPPG_FPS`` (30).

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``"heart_rate"`` (float): Estimated BPM.
        - ``"pulse_signal"`` (np.ndarray): Filtered POS pulse waveform.
        - ``"pulse_stability"`` (float): Stability score in [0, 1].
        - ``"rppg_confidence"`` (float): Overall confidence in [0, 1].
        - ``"fft_frequencies"`` (np.ndarray): FFT frequency bins (Hz).
        - ``"fft_magnitudes"`` (np.ndarray): FFT magnitude spectrum.
        - ``"rgb_signals"`` (dict): Raw RGB temporal traces.
    """
    if fps is None:
        fps = float(config.RPPG_FPS)

    # Default empty results
    empty_result = {
        "heart_rate": 0.0,
        "pulse_signal": np.array([], dtype=np.float64),
        "pulse_stability": 0.0,
        "rppg_confidence": 0.0,
        "fft_frequencies": np.array([], dtype=np.float64),
        "fft_magnitudes": np.array([], dtype=np.float64),
        "rgb_signals": {"R": np.array([], dtype=np.float64),
                        "G": np.array([], dtype=np.float64),
                        "B": np.array([], dtype=np.float64)},
    }

    # --- Input validation ---
    if face_images is None and frames is None:
        logger.error("extract_rppg_features: both face_images and frames are None.")
        return empty_result

    if face_images is None:
        logger.error(
            "extract_rppg_features: face_images must be provided "
            "(pre-aligned 224x224 face crops)."
        )
        return empty_result

    face_images = np.asarray(face_images)
    if face_images.ndim != 4 or face_images.shape[1] != 224 or face_images.shape[2] != 224:
        logger.error(
            "extract_rppg_features: expected face_images of shape (N, 224, 224, 3), "
            "got %s.",
            face_images.shape,
        )
        return empty_result

    n_frames = face_images.shape[0]
    logger.info("Starting rPPG feature extraction for %d frames at %.1f FPS.", n_frames, fps)

    # Minimum number of frames for meaningful rPPG analysis
    min_frames_for_fft = int(np.ceil(fps * 2.0))  # At least ~2 seconds
    if n_frames < min_frames_for_fft:
        logger.warning(
            "Too few frames (%d < %d) for reliable rPPG analysis.",
            n_frames, min_frames_for_fft,
        )
        return empty_result

    # --- Step 1: Extract ROIs from each face image ---
    logger.debug("Step 1: Extracting face ROIs from %d frames.", n_frames)
    roi_sequence: List[Dict[str, np.ndarray]] = []
    for i in range(n_frames):
        rois = extract_face_roi(face_images[i])
        roi_sequence.append(rois)

    # --- Step 2: Compute RGB signals ---
    logger.debug("Step 2: Computing RGB temporal signals.")
    rgb_signals = compute_rgb_signals(roi_sequence, frames)

    if len(rgb_signals["R"]) == 0:
        logger.error("RGB signal computation returned empty results.")
        return empty_result

    # --- Step 3: Compute pulse signal using POS method ---
    logger.debug("Step 3: Computing POS pulse signal.")
    pulse_signal = compute_pulse_signal(rgb_signals, fps=fps)

    if len(pulse_signal) == 0:
        logger.error("POS pulse signal computation returned empty result.")
        return empty_result

    # --- Step 4: Estimate heart rate via FFT on the pulse signal ---
    logger.debug("Step 4: Estimating heart rate via FFT.")
    heart_rate, fft_frequencies, fft_magnitudes = compute_heart_rate_fft(pulse_signal, fps=fps)

    # Also try estimating HR from the raw G channel (green is strongest for rPPG)
    # and pick the more plausible estimate.
    g_filtered = bandpass_filter(rgb_signals["G"], fps=fps)
    g_hr, _, _ = compute_heart_rate_fft(g_filtered, fps=fps)

    if g_hr > 0.0 and (
        heart_rate <= 0.0
        or abs(g_hr - 72.0) < abs(heart_rate - 72.0)  # closer to resting HR
    ):
        logger.debug(
            "Using G-channel HR estimate (%.1f BPM) over POS HR (%.1f BPM).",
            g_hr, heart_rate,
        )
        heart_rate = g_hr

    # --- Step 5: Compute pulse stability ---
    logger.debug("Step 5: Computing pulse stability.")
    window_size = config.RPPG_WINDOW_SIZE
    pulse_stability, stability_values = compute_pulse_stability(pulse_signal, window_size=window_size)

    # --- Step 6: Compute overall rPPG confidence ---
    logger.debug("Step 6: Computing rPPG confidence.")
    rppg_confidence = compute_rppg_confidence(heart_rate, pulse_signal, pulse_stability)

    result = {
        "heart_rate": heart_rate,
        "pulse_signal": pulse_signal,
        "pulse_stability": pulse_stability,
        "rppg_confidence": rppg_confidence,
        "fft_frequencies": fft_frequencies,
        "fft_magnitudes": fft_magnitudes,
        "rgb_signals": rgb_signals,
    }

    logger.info(
        "rPPG extraction complete: HR=%.1f BPM, stability=%.3f, "
        "confidence=%.3f, pulse_samples=%d",
        heart_rate, pulse_stability, rppg_confidence, len(pulse_signal),
    )

    return result
