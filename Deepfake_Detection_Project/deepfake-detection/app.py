"""
Flask Web Application for AI-Based Multimodal Deepfake Video Detection.

Provides a full-featured web interface and REST API for uploading videos,
running the deepfake detection pipeline, viewing results with interactive
charts (lip-motion, pulse signal), and browsing training / evaluation metrics.
"""

import json
import logging
import os
import sqlite3
import time
from datetime import datetime

import cv2
import numpy as np
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

import config
from predict import (
    generate_gradcam,
    load_trained_model,
    predict_single_video,
    save_prediction_to_db,
)

# ======================================================================
# Logging
# ======================================================================
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ======================================================================
# Flask Application
# ======================================================================
app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH


# ======================================================================
# Database Initialization
# ======================================================================


def get_db_connection():
    """Create and return a new SQLite database connection."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the predictions and training_logs tables if they do not exist."""
    os.makedirs(os.path.dirname(config.DATABASE_PATH) or ".", exist_ok=True)

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name        TEXT    NOT NULL,
            prediction        TEXT    NOT NULL,
            confidence        REAL    NOT NULL,
            lip_sync_score    REAL    DEFAULT 0.0,
            heart_rate        REAL    DEFAULT 0.0,
            pulse_stability   REAL    DEFAULT 0.0,
            rppg_confidence   REAL    DEFAULT 0.0,
            inference_time    REAL    DEFAULT 0.0,
            created_at        TIMESTAMP NOT NULL
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_logs (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name        TEXT    NOT NULL,
            accuracy          REAL    DEFAULT 0.0,
            val_accuracy      REAL    DEFAULT 0.0,
            loss              REAL    DEFAULT 0.0,
            val_loss          REAL    DEFAULT 0.0,
            epochs            INTEGER DEFAULT 0,
            training_time     REAL    DEFAULT 0.0,
            created_at        TIMESTAMP NOT NULL
        );
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialised at %s", config.DATABASE_PATH)


# Run DB init when the module is imported
init_db()


# ======================================================================
# Helpers
# ======================================================================


def allowed_file(filename):
    """Return True if *filename* has an allowed video extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


def _save_prediction_record(result: dict):
    """Insert a prediction row into the application predictions table.

    This uses the **app-level** schema (with pulse_stability and
    rppg_confidence columns) which is richer than the one in
    ``predict.save_prediction_to_db``.
    """
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO predictions
                (video_name, prediction, confidence, lip_sync_score,
                 heart_rate, pulse_stability, rppg_confidence,
                 inference_time, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.get("video_name", "unknown"),
                result.get("label", "Error"),
                result.get("confidence", 0.0),
                result.get("lip_sync_score", 0.0),
                result.get("heart_rate", 0.0),
                result.get("pulse_stability", 0.0),
                result.get("rppg_confidence", 0.0),
                result.get("inference_time", 0.0),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        logger.info(
            "Prediction record saved: %s -> %s (%.1f%%)",
            result.get("video_name"),
            result.get("label"),
            result.get("confidence"),
        )
    except sqlite3.Error as exc:
        logger.error("Failed to save prediction record: %s", exc)
    finally:
        conn.close()


def _load_metrics_json():
    """Load evaluation metrics from the first metrics.json found in plots.

    Searches recursively under ``config.PLOTS_DIR`` for ``metrics.json``.
    Returns an empty dict if none is found.
    """
    for root, _dirs, files in os.walk(config.PLOTS_DIR):
        if "metrics.json" in files:
            path = os.path.join(root, "metrics.json")
            try:
                with open(path, "r") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load metrics from %s: %s", path, exc)
                return {}
    return {}


def _build_lip_chart_data(lip_sync_result: dict) -> str:
    """Convert lip-sync arrays into a JSON string suitable for Chart.js.

    Returns a JSON object with keys:
        - ``frames``: list of frame indices
        - ``mouth_openings``: list of per-frame mouth-opening distances
        - ``lip_aspect_ratios``: list of per-frame LAR values
        - ``lip_movements``: list of per-frame lip-movement magnitudes
    """
    mouth_openings = lip_sync_result.get("mouth_openings", np.array([]))
    lip_aspect_ratios = lip_sync_result.get("lip_aspect_ratios", np.array([]))
    lip_movements = lip_sync_result.get("lip_movements", np.array([]))

    # Convert numpy arrays to plain Python lists, truncating to at most
    # 500 data-points to keep the payload manageable for the browser.
    max_points = 500
    n = len(mouth_openings)
    step = max(1, n // max_points)
    indices = list(range(0, n, step))

    def _tolist(arr):
        a = np.asarray(arr)
        if a.size == 0:
            return []
        return [round(float(v), 4) for v in a[::step]]

    data = {
        "frames": indices,
        "mouth_openings": _tolist(mouth_openings),
        "lip_aspect_ratios": _tolist(lip_aspect_ratios),
        "lip_movements": _tolist(lip_movements),
    }
    return json.dumps(data)


def _build_pulse_chart_data(rppg_result: dict) -> str:
    """Convert rPPG arrays into a JSON string suitable for Chart.js.

    Returns a JSON object with keys:
        - ``time_seconds``: list of time values in seconds
        - ``pulse_signal``: list of POS pulse waveform samples
        - ``fft_frequencies``: list of FFT frequency bins (Hz)
        - ``fft_magnitudes``: list of FFT magnitude values
    """
    pulse_signal = rppg_result.get("pulse_signal", np.array([]))
    fft_freqs = rppg_result.get("fft_frequencies", np.array([]))
    fft_mags = rppg_result.get("fft_magnitudes", np.array([]))

    max_points = 500

    # Pulse signal -> time-domain chart
    n_pulse = len(pulse_signal)
    step_pulse = max(1, n_pulse // max_points)
    pulse_indices = list(range(0, n_pulse, step_pulse))
    pulse_values = (
        [round(float(v), 6) for v in pulse_signal[::step_pulse]]
        if n_pulse > 0
        else []
    )

    # FFT -> frequency-domain chart
    n_fft = len(fft_freqs)
    step_fft = max(1, n_fft // max_points)
    fft_freq_values = (
        [round(float(v), 2) for v in fft_freqs[::step_fft]]
        if n_fft > 0
        else []
    )
    fft_mag_values = (
        [round(float(v), 6) for v in fft_mags[::step_fft]]
        if n_fft > 0
        else []
    )

    data = {
        "time_samples": pulse_indices,
        "pulse_signal": pulse_values,
        "fft_frequencies": fft_freq_values,
        "fft_magnitudes": fft_mag_values,
    }
    return json.dumps(data)


def _detect_and_save_face(frames, output_dir, video_name):
    """Detect a face from the first valid frame and save it.

    Returns the relative URL path for the saved face image, or None.
    """
    import preprocess as prep

    face_image = None
    for frame in frames:
        face = prep.preprocess_face(frame)
        if face is not None:
            face_image = face
            break

    if face_image is None:
        logger.warning("No face detected in any frame for %s", video_name)
        return None

    # Convert normalised float32 [0,1] image to uint8 BGR for saving
    face_bgr = (np.clip(face_image, 0.0, 1.0) * 255).astype(np.uint8)
    face_filename = f"detected_face_{secure_filename(video_name)}.jpg"
    face_path = os.path.join(output_dir, face_filename)
    cv2.imwrite(face_path, face_bgr)

    relative_path = os.path.join("outputs", face_filename)
    logger.info("Detected face saved to %s", face_path)
    return relative_path


# ======================================================================
# Context Processor
# ======================================================================


@app.context_processor
def inject_globals():
    """Inject common template variables into every render call."""
    return {
        "app_name": "DeepFake Detection",
        "current_year": datetime.now().year,
    }


# ======================================================================
# Template Routes
# ======================================================================


@app.route("/")
def index():
    """Home page."""
    logger.info("Home page accessed")
    return render_template("index.html")


@app.route("/about")
def about():
    """About page."""
    logger.info("About page accessed")
    return render_template("about.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """Upload page – show form on GET, handle upload on POST."""
    if request.method == "GET":
        logger.info("Upload form page accessed")
        return render_template("upload.html")

    # --- POST: handle file upload ---
    logger.info("File upload request received")

    if "video" not in request.files:
        logger.warning("Upload attempt with no file part")
        return render_template("upload.html", error="No file selected. Please choose a video file.")

    file = request.files["video"]

    if file.filename == "" or file.filename is None:
        logger.warning("Upload attempt with empty filename")
        return render_template("upload.html", error="No file selected. Please choose a video file.")

    if not allowed_file(file.filename):
        logger.warning("Upload of disallowed file type: %s", file.filename)
        return render_template(
            "upload.html",
            error=f"Invalid file type. Allowed extensions: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}",
        )

    filename = secure_filename(file.filename)
    if filename == "":
        # secure_filename may return empty string for purely non-ASCII names
        filename = f"upload_{int(time.time())}.mp4"

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    # Avoid overwriting existing files by appending a timestamp suffix
    if os.path.exists(save_path):
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{int(time.time())}{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(save_path)
        logger.info("File saved: %s (%d bytes)", save_path, os.path.getsize(save_path))
    except Exception as exc:
        logger.error("Failed to save uploaded file: %s", exc)
        return render_template("upload.html", error="Failed to save the uploaded file. Please try again.")

    return redirect(url_for("predict", filename=filename))


@app.route("/predict/<filename>")
def predict(filename):
    """Run the full prediction pipeline on an uploaded video and display results.

    Template variables supplied to ``prediction.html``:
        - video_info (dict)          – raw video metadata
        - extracted_frames_count (int)
        - detected_face_path (str|None) – relative URL to face crop JPEG
        - lip_motion_data (str)      – JSON for lip-motion Chart.js chart
        - pulse_data (str)           – JSON for pulse-signal Chart.js chart
        - prediction_result (dict)   – full result from predict_single_video
        - confidence (float)
        - inference_time (float)
        - gradcam_path (str|None)    – relative URL to Grad-CAM image
    """
    logger.info("Prediction requested for: %s", filename)

    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    if not os.path.isfile(video_path):
        logger.error("Video file not found: %s", video_path)
        return render_template("upload.html", error=f"Video file '{filename}' not found. Please upload again.")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model_path = (
        config.BEST_MODEL_PATH
        if os.path.isfile(config.BEST_MODEL_PATH)
        else config.FINAL_MODEL_PATH
    )

    if not os.path.isfile(model_path):
        logger.error("No trained model found. Checked: %s, %s", config.BEST_MODEL_PATH, config.FINAL_MODEL_PATH)
        return render_template(
            "upload.html",
            error="No trained model available. Please train a model first using train.py.",
        )

    try:
        model = load_trained_model(model_path)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Model loading failed: %s", exc)
        return render_template("upload.html", error=f"Failed to load model: {exc}")

    # ------------------------------------------------------------------
    # Run main prediction pipeline
    # ------------------------------------------------------------------
    output_dir = os.path.join(config.OUTPUT_DIR, os.path.splitext(filename)[0])
    os.makedirs(output_dir, exist_ok=True)

    try:
        result = predict_single_video(video_path, model, output_dir)
    except Exception as exc:
        logger.exception("Prediction pipeline failed for %s", filename)
        return render_template("upload.html", error=f"Prediction failed: {exc}")

    # Check for pipeline errors
    if result.get("error"):
        logger.warning("Prediction returned error: %s", result["error"])
        return render_template(
            "upload.html",
            error=f"Prediction error: {result['error']}. The video may be corrupt or too short.",
        )

    # ------------------------------------------------------------------
    # Extract chart data (lip motion & pulse)
    # ------------------------------------------------------------------
    lip_motion_data_json = "{}"
    pulse_data_json = "{}"

    try:
        import lip_sync as ls_mod
        import preprocess as prep

        frames, _frame_meta = prep.extract_frames(video_path)
        if len(frames) > 0:
            # Lip-sync chart data
            try:
                lip_result = ls_mod.analyze_lip_sync(video_path, frames)
                lip_motion_data_json = _build_lip_chart_data(lip_result)
            except Exception as exc:
                logger.warning("Lip-sync chart data extraction failed: %s", exc)

            # rPPG / pulse chart data
            try:
                import rppg as rppg_mod

                face_images_list = []
                for frm in frames:
                    face = prep.preprocess_face(frm)
                    if face is not None:
                        face_images_list.append(face)

                if len(face_images_list) > 0:
                    face_bgr_uint8 = (
                        np.clip(np.stack(face_images_list, axis=0), 0.0, 1.0) * 255
                    ).astype(np.uint8)
                    fps = result.get("video_metadata", {}).get("fps", config.RPPG_FPS)
                    rppg_result = rppg_mod.extract_rppg_features(
                        face_images=face_bgr_uint8, fps=fps
                    )
                    pulse_data_json = _build_pulse_chart_data(rppg_result)
            except Exception as exc:
                logger.warning("rPPG chart data extraction failed: %s", exc)
    except Exception as exc:
        logger.warning("Chart data extraction failed: %s", exc)

    # ------------------------------------------------------------------
    # Detect & save a face crop image
    # ------------------------------------------------------------------
    detected_face_url = None
    try:
        import preprocess as prep

        frames_for_face, _ = prep.extract_frames(video_path)
        detected_face_url = _detect_and_save_face(frames_for_face, output_dir, filename)
    except Exception as exc:
        logger.warning("Face detection for display failed: %s", exc)

    # ------------------------------------------------------------------
    # Generate Grad-CAM
    # ------------------------------------------------------------------
    gradcam_url = None
    try:
        import preprocess as prep

        face_for_gradcam = None
        frames_gc, _ = prep.extract_frames(video_path)
        for frm in frames_gc:
            face = prep.preprocess_face(frm)
            if face is not None:
                face_for_gradcam = face
                break

        if face_for_gradcam is not None:
            gradcam_rel_path = generate_gradcam(model, face_for_gradcam, output_dir)
            if gradcam_rel_path:
                gradcam_url = os.path.join(
                    "outputs", os.path.splitext(filename)[0], os.path.basename(gradcam_rel_path)
                )
    except Exception as exc:
        logger.warning("Grad-CAM generation failed: %s", exc)

    # ------------------------------------------------------------------
    # Save prediction to the application database
    # ------------------------------------------------------------------
    _save_prediction_record(result)

    # ------------------------------------------------------------------
    # Build video info summary
    # ------------------------------------------------------------------
    video_info = result.get("video_metadata", {})
    video_info["filename"] = filename
    video_info["file_size_mb"] = round(
        os.path.getsize(video_path) / (1024 * 1024), 2
    )

    logger.info(
        "Prediction complete: %s -> %s (%.1f%%) in %.2fs",
        filename,
        result.get("label"),
        result.get("confidence"),
        result.get("inference_time"),
    )

    return render_template(
        "prediction.html",
        video_info=video_info,
        extracted_frames_count=result.get("frame_count", 0),
        detected_face_path=detected_face_url,
        lip_motion_data=lip_motion_data_json,
        pulse_data=pulse_data_json,
        prediction_result=result,
        confidence=result.get("confidence", 0.0),
        inference_time=result.get("inference_time", 0.0),
        gradcam_path=gradcam_url,
    )


@app.route("/dashboard")
def dashboard():
    """Admin training dashboard showing training logs, recent predictions, and dataset stats."""
    logger.info("Dashboard accessed")

    conn = get_db_connection()
    try:
        # --- Training logs ---
        training_logs = conn.execute(
            "SELECT * FROM training_logs ORDER BY created_at DESC LIMIT 20"
        ).fetchall()

        training_logs_list = [dict(row) for row in training_logs]

        # --- Recent predictions ---
        recent_predictions = conn.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 50"
        ).fetchall()

        recent_predictions_list = [dict(row) for row in recent_predictions]

        # --- Aggregated prediction stats ---
        total_predictions = conn.execute(
            "SELECT COUNT(*) as cnt FROM predictions"
        ).fetchone()["cnt"]

        real_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM predictions WHERE prediction = 'Real'"
        ).fetchone()["cnt"]

        fake_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM predictions WHERE prediction = 'Fake'"
        ).fetchone()["cnt"]

        avg_confidence = conn.execute(
            "SELECT AVG(confidence) as avg_conf FROM predictions"
        ).fetchone()["avg_conf"]

        avg_inference_time = conn.execute(
            "SELECT AVG(inference_time) as avg_time FROM predictions"
        ).fetchone()["avg_time"]

        # --- Dataset statistics ---
        real_videos = 0
        fake_videos = 0
        if os.path.isdir(config.REAL_DIR):
            real_videos = len([
                f for f in os.listdir(config.REAL_DIR)
                if os.path.isfile(os.path.join(config.REAL_DIR, f))
            ])
        if os.path.isdir(config.FAKE_DIR):
            fake_videos = len([
                f for f in os.listdir(config.FAKE_DIR)
                if os.path.isfile(os.path.join(config.FAKE_DIR, f))
            ])

        total_dataset = real_videos + fake_videos

        dataset_stats = {
            "real_videos": real_videos,
            "fake_videos": fake_videos,
            "total_videos": total_dataset,
            "real_pct": round(100 * real_videos / max(1, total_dataset), 1),
            "fake_pct": round(100 * fake_videos / max(1, total_dataset), 1),
        }

        # --- Chart data for prediction distribution ---
        prediction_distribution = [
            {"label": "Real", "count": real_count, "color": config.CLASS_COLORS.get(0, "#00C853")},
            {"label": "Fake", "count": fake_count, "color": config.CLASS_COLORS.get(1, "#FF1744")},
        ]

    finally:
        conn.close()

    return render_template(
        "dashboard.html",
        training_logs=training_logs_list,
        recent_predictions=recent_predictions_list,
        total_predictions=total_predictions,
        real_count=real_count,
        fake_count=fake_count,
        avg_confidence=round(avg_confidence, 2) if avg_confidence else 0.0,
        avg_inference_time=round(avg_inference_time, 3) if avg_inference_time else 0.0,
        dataset_stats=dataset_stats,
        prediction_distribution=json.dumps(prediction_distribution),
    )


@app.route("/metrics")
def metrics():
    """Model evaluation metrics page with chart-ready data."""
    logger.info("Metrics page accessed")

    metrics_data = _load_metrics_json()

    # Prepare chart data structures
    metrics_labels = []
    metrics_values = []
    chart_colors = config.CHART_COLORS

    display_keys = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall / Sensitivity"),
        ("f1_score", "F1 Score"),
        ("auc", "AUC-ROC"),
        ("specificity", "Specificity"),
        ("mcc", "MCC"),
        ("cohen_kappa", "Cohen's Kappa"),
    ]

    for key, label in display_keys:
        val = metrics_data.get(key)
        if val is not None:
            metrics_labels.append(label)
            # Scale MCC and Kappa to percentage for visual consistency
            if key in ("mcc", "cohen_kappa"):
                metrics_values.append(round(float(val) * 100, 2))
            else:
                metrics_values.append(round(float(val) * 100, 2))

    performance_metrics = [
        {"label": label, "value": round(metrics_data.get(key, 0) * 100, 2)}
        for key, label in display_keys
        if metrics_data.get(key) is not None
    ]

    # Additional metrics that are not percentages
    additional_metrics = {}
    if "avg_inference_time_ms" in metrics_data:
        additional_metrics["avg_inference_time_ms"] = round(
            metrics_data["avg_inference_time_ms"], 2
        )
    if "fps" in metrics_data:
        additional_metrics["fps"] = round(metrics_data["fps"], 2)

    # Check for available plot images
    available_plots = []
    if os.path.isdir(config.PLOTS_DIR):
        for root, _dirs, files in os.walk(config.PLOTS_DIR):
            for f in sorted(files):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, config.BASE_DIR)
                    available_plots.append({"filename": f, "path": rel_path.replace(os.sep, "/")})

    return render_template(
        "metrics.html",
        metrics=metrics_data,
        metrics_labels=json.dumps(metrics_labels),
        metrics_values=json.dumps(metrics_values),
        chart_colors=json.dumps(chart_colors[: len(metrics_labels)]),
        performance_metrics=performance_metrics,
        additional_metrics=additional_metrics,
        available_plots=available_plots,
        has_metrics=bool(metrics_data),
    )


@app.route("/dataset")
def dataset():
    """Dataset information page."""
    logger.info("Dataset page accessed")

    # Gather dataset file counts
    real_files = []
    fake_files = []

    if os.path.isdir(config.REAL_DIR):
        real_files = sorted(os.listdir(config.REAL_DIR))
    if os.path.isdir(config.FAKE_DIR):
        fake_files = sorted(os.listdir(config.FAKE_DIR))

    real_count = len(real_files)
    fake_count = len(fake_files)
    total_count = real_count + fake_count

    # Compute approximate total size
    def _dir_size(path):
        total = 0
        if not os.path.isdir(path):
            return 0
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
        return total

    real_size = _dir_size(config.REAL_DIR)
    fake_size = _dir_size(config.FAKE_DIR)

    dataset_info = {
        "real_count": real_count,
        "fake_count": fake_count,
        "total_count": total_count,
        "real_size_gb": round(real_size / (1024 ** 3), 2),
        "fake_size_gb": round(fake_size / (1024 ** 3), 2),
        "total_size_gb": round((real_size + fake_size) / (1024 ** 3), 2),
        "real_dir": config.REAL_DIR,
        "fake_dir": config.FAKE_DIR,
        "allowed_extensions": sorted(config.ALLOWED_EXTENSIONS),
        "train_split": config.TRAIN_SPLIT,
        "val_split": config.VAL_SPLIT,
        "test_split": config.TEST_SPLIT,
    }

    # Sample files to display (first 20 of each class)
    sample_real = real_files[:20]
    sample_fake = fake_files[:20]

    return render_template(
        "dataset.html",
        dataset_info=dataset_info,
        sample_real=sample_real,
        sample_fake=sample_fake,
    )


@app.route("/contact")
def contact():
    """Contact page."""
    logger.info("Contact page accessed")
    return render_template("contact.html")


# ======================================================================
# API Routes
# ======================================================================


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for video deepfake prediction.

    Accepts a multipart form upload with a ``video`` field.
    Returns a JSON object with the full prediction result.
    """
    logger.info("API predict request received")

    if "video" not in request.files:
        return jsonify({"error": "No file part in request. Send a 'video' field."}), 400

    file = request.files["video"]

    if file.filename == "" or file.filename is None:
        return jsonify({"error": "Empty filename provided."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Invalid file type. Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}",
        }), 400

    filename = secure_filename(file.filename)
    if filename == "":
        filename = f"api_upload_{int(time.time())}.mp4"

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(save_path):
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{int(time.time())}{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(save_path)
    except Exception as exc:
        logger.error("API: failed to save file: %s", exc)
        return jsonify({"error": "Failed to save uploaded file."}), 500

    # Load model
    model_path = (
        config.BEST_MODEL_PATH
        if os.path.isfile(config.BEST_MODEL_PATH)
        else config.FINAL_MODEL_PATH
    )

    if not os.path.isfile(model_path):
        return jsonify({"error": "No trained model available on the server."}), 503

    try:
        model = load_trained_model(model_path)
    except (FileNotFoundError, RuntimeError) as exc:
        return jsonify({"error": f"Model loading failed: {exc}"}), 500

    # Run prediction
    output_dir = os.path.join(config.OUTPUT_DIR, f"api_{os.path.splitext(filename)[0]}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        result = predict_single_video(video_path=save_path, model=model, output_dir=output_dir)
    except Exception as exc:
        logger.exception("API prediction failed for %s", filename)
        return jsonify({"error": f"Prediction pipeline failed: {exc}"}), 500

    # Save to DB
    _save_prediction_record(result)

    # Build clean JSON response (ensure all values are JSON-serialisable)
    response_data = {
        "video_name": result.get("video_name", filename),
        "prediction": result.get("prediction", -1),
        "label": result.get("label", "Error"),
        "confidence": result.get("confidence", 0.0),
        "inference_time": result.get("inference_time", 0.0),
        "fps": result.get("fps", 0.0),
        "lip_sync_score": result.get("lip_sync_score", 0.0),
        "heart_rate": result.get("heart_rate", 0.0),
        "pulse_stability": result.get("pulse_stability", 0.0),
        "rppg_confidence": result.get("rppg_confidence", 0.0),
        "face_count": result.get("face_count", 0),
        "frame_count": result.get("frame_count", 0),
        "raw_probability": result.get("raw_probability", 0.0),
        "error": result.get("error"),
    }

    status_code = 200 if not result.get("error") else 422
    return jsonify(response_data), status_code


@app.route("/api/history")
def api_history():
    """REST API endpoint returning prediction history from the database.

    Query parameters:
        - ``limit`` (int, default 50): maximum number of records
        - ``offset`` (int, default 0): skip first N records
    """
    logger.info("API history request")

    try:
        limit = min(int(request.args.get("limit", 50)), 500)
    except (ValueError, TypeError):
        limit = 50

    try:
        offset = max(int(request.args.get("offset", 0)), 0)
    except (ValueError, TypeError):
        offset = 0

    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

        total = conn.execute("SELECT COUNT(*) as cnt FROM predictions").fetchone()["cnt"]

        history = [dict(row) for row in rows]
    finally:
        conn.close()

    return jsonify({
        "total": total,
        "limit": limit,
        "offset": offset,
        "predictions": history,
    })


@app.route("/api/dataset-stats")
def api_dataset_stats():
    """REST API endpoint returning dataset statistics."""
    logger.info("API dataset-stats request")

    real_files = []
    fake_files = []

    if os.path.isdir(config.REAL_DIR):
        real_files = [f for f in os.listdir(config.REAL_DIR) if os.path.isfile(os.path.join(config.REAL_DIR, f))]
    if os.path.isdir(config.FAKE_DIR):
        fake_files = [f for f in os.listdir(config.FAKE_DIR) if os.path.isfile(os.path.join(config.FAKE_DIR, f))]

    def _dir_size(path):
        total = 0
        if not os.path.isdir(path):
            return 0
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
        return total

    real_size = _dir_size(config.REAL_DIR)
    fake_size = _dir_size(config.FAKE_DIR)

    # Collect file extensions present in the dataset
    all_extensions = set()
    for f in real_files + fake_files:
        ext = os.path.splitext(f)[1].lower().lstrip(".")
        if ext:
            all_extensions.add(ext)

    return jsonify({
        "dataset_dir": config.DATASET_DIR,
        "real": {
            "count": len(real_files),
            "size_bytes": real_size,
            "size_gb": round(real_size / (1024 ** 3), 4),
            "directory": config.REAL_DIR,
        },
        "fake": {
            "count": len(fake_files),
            "size_bytes": fake_size,
            "size_gb": round(fake_size / (1024 ** 3), 4),
            "directory": config.FAKE_DIR,
        },
        "total": {
            "count": len(real_files) + len(fake_files),
            "size_bytes": real_size + fake_size,
            "size_gb": round((real_size + fake_size) / (1024 ** 3), 4),
        },
        "extensions": sorted(all_extensions),
        "splits": {
            "train": config.TRAIN_SPLIT,
            "validation": config.VAL_SPLIT,
            "test": config.TEST_SPLIT,
        },
        "config": {
            "target_face_size": list(config.TARGET_FACE_SIZE),
            "frame_extraction_interval_ms": config.FRAME_EXTRACTION_INTERVAL_MS,
            "max_video_duration_sec": config.MAX_VIDEO_DURATION_SEC,
            "max_video_size_mb": config.MAX_VIDEO_SIZE_MB,
        },
    })


# ======================================================================
# Serve uploaded / output files
# ======================================================================


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """Serve files from the uploads directory."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/outputs/<path:filename>")
def output_file(filename):
    """Serve files from the outputs directory (faces, Grad-CAM, etc.)."""
    return send_from_directory(config.OUTPUT_DIR, filename)


# ======================================================================
# Error Handlers
# ======================================================================


@app.errorhandler(404)
def page_not_found(error):
    """Handle 404 Not Found errors."""
    logger.warning("404 Not Found: %s", request.path)
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_server_error(error):
    """Handle 500 Internal Server Error."""
    logger.error("500 Internal Server Error: %s", error)
    return render_template("500.html"), 500


# ======================================================================
# Entry Point
# ======================================================================


if __name__ == "__main__":
    logger.info("Starting DeepFake Detection Flask application")
    logger.info("Upload folder: %s", app.config["UPLOAD_FOLDER"])
    logger.info("Database: %s", config.DATABASE_PATH)
    app.run(host="0.0.0.0", port=5000, debug=True)
