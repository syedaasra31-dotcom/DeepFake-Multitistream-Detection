"""
Configuration Module for AI-Based Multimodal Deepfake Video Detection.
Contains all project-wide constants, paths, and hyperparameters.
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REAL_DIR = os.path.join(DATASET_DIR, "real")
FAKE_DIR = os.path.join(DATASET_DIR, "fake")

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
BEST_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "best_deepfake_model.h5")
FINAL_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "final_deepfake_model.h5")

# Upload and output paths
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Allowed video extensions
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

# Video processing
FRAME_EXTRACTION_INTERVAL_MS = 100  # Extract frame every 100ms
TARGET_FACE_SIZE = (224, 224)
MAX_VIDEO_DURATION_SEC = 300  # 5 minutes max
MAX_VIDEO_SIZE_MB = 500

# Audio processing
AUDIO_SAMPLE_RATE = 16000
AUDIO_MONO = True

# Lip sync analysis
LIP_SYNC_THRESHOLD = 0.55
LIP_SYNC_WINDOW_SIZE = 5

# rPPG parameters
RPPG_BANDPASS_LOW = 0.7  # Hz
RPPG_BANDPASS_HIGH = 4.0  # Hz
RPPG_FPS = 30
RPPG_WINDOW_SIZE = 300  # frames (~10 seconds at 30fps)

# Model hyperparameters
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.5
LSTM_UNITS = 128
DENSE_UNITS = 64

# Training splits
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Callbacks
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# Supported CNN backbones
CNN_BACKBONES = ["resnet50", "efficientnetb0", "mobilenetv3"]
DEFAULT_BACKBONE = "efficientnetb0"

# Database
DATABASE_PATH = os.path.join(BASE_DIR, "deepfake_detection.db")

# Flask settings
SECRET_KEY = os.environ.get("SECRET_KEY", "deepfake-detection-secret-key-2025")
UPLOAD_FOLDER = UPLOAD_DIR
MAX_CONTENT_LENGTH = MAX_VIDEO_SIZE_MB * 1024 * 1024

# Class labels
CLASS_LABELS = {0: "Real", 1: "Fake"}
CLASS_COLORS = {0: "#00C853", 1: "#FF1744"}

# Ensure directories exist
for directory in [DATASET_DIR, REAL_DIR, FAKE_DIR, MODEL_DIR, SAVED_MODEL_DIR,
                   UPLOAD_DIR, OUTPUT_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(BASE_DIR, "app.log")

# Visualization settings
PLOT_DPI = 300
PLOT_FIGSIZE = (12, 8)
CHART_COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
