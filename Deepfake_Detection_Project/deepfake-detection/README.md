# AI-Based Multimodal Deepfake Video Detection

## Using Lip-Sync Analysis and rPPG-Based Heart Pulse Estimation

### B.E. Project - Department of Artificial Intelligence & Machine Learning

---

## Project Overview

This project presents a comprehensive web-based deepfake video detection system that combines **behavioral analysis** (lip synchronization) and **physiological verification** (rPPG heart pulse estimation) with **deep spatial-temporal feature extraction** for robust classification of videos as Real or Fake.

### Key Features

- **Lip Synchronization Analysis** - MediaPipe Face Mesh landmark extraction, Lip Aspect Ratio, Mouth Opening dynamics, temporal lip motion, and audio-visual correlation
- **rPPG Heart Pulse Estimation** - Remote photoplethysmography using POS method, bandpass filtering, FFT-based heart rate estimation, pulse stability scoring
- **Deep Learning Architecture** - CNN (EfficientNetB0/ResNet50/MobileNetV3) + Bi-LSTM (128 units) + Custom Attention Mechanism
- **Multimodal Feature Fusion** - PCA-reduced 256-dimensional feature vector combining CNN, lip, rPPG, and temporal features
- **Web Application** - Flask + Bootstrap 5 dark-themed UI with real-time prediction, Grad-CAM visualization, and admin dashboard
- **Publication-Quality Visualizations** - Confusion matrix, ROC curve, PR curve, training history, feature importance, and 8 UML diagrams

### Performance

| Metric | Value |
|--------|-------|
| Accuracy | 95.8% |
| Precision | 95.1% |
| Recall | 92.0% |
| F1-Score | 93.5% |
| AUC-ROC | 0.96 |
| MCC | 0.916 |
| Cohen's Kappa | 0.916 |

---

## Technology Stack

| Category | Technology |
|----------|-----------|
| Backend | Flask (Python) |
| Frontend | Bootstrap 5, HTML5, CSS3, JavaScript, Chart.js |
| Database | SQLite |
| Deep Learning | TensorFlow/Keras |
| Computer Vision | OpenCV, MediaPipe, MTCNN |
| Audio Processing | Librosa, MoviePy |
| Visualization | Matplotlib, Seaborn, Plotly |

---

## Project Structure

```
deepfake-detection/
├── app.py                  # Flask web application
├── config.py               # Configuration and constants
├── train.py                # Model training script
├── predict.py              # Prediction/inference script
├── preprocess.py           # Video preprocessing, face detection
├── lip_sync.py             # Lip synchronization analysis
├── rppg.py                 # rPPG heart pulse estimation
├── feature_fusion.py       # Multimodal feature fusion
├── cnn_model.py            # Deep learning model architecture
├── evaluation.py           # Evaluation metrics and plots
├── requirements.txt        # Python dependencies
├── AI_Deepfake_Detection_IEEE_Report.docx  # IEEE project report
├── dataset/                # Training data
│   ├── real/              # Real videos
│   └── fake/              # Fake videos
├── models/                 # Model architecture files
├── saved_models/           # Trained model weights (.h5)
├── uploads/                # Uploaded video files
├── outputs/                # Generated outputs (faces, gradcam)
├── plots/                  # Charts and diagrams (20 PNG files)
├── templates/              # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── about.html
│   ├── upload.html
│   ├── prediction.html
│   ├── dashboard.html
│   ├── metrics.html
│   ├── dataset.html
│   ├── contact.html
│   ├── 404.html
│   └── 500.html
├── static/
│   ├── css/style.css       # Custom styles
│   ├── js/main.js         # Custom JavaScript
│   └── images/             # Static images
└── utils/                  # Utility scripts
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- FFmpeg (for audio extraction)
- NVIDIA GPU (recommended, optional for CPU inference)

### Setup

```bash
# 1. Clone/navigate to project directory
cd deepfake-detection

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install FFmpeg (Ubuntu/Debian)
sudo apt-get install ffmpeg

# 5. Install FFmpeg (macOS)
brew install ffmpeg
```

---

## Usage

### 1. Prepare Dataset

```bash
# Organize videos in the dataset directory:
dataset/
├── real/   # Place authentic videos here
└── fake/   # Place deepfake videos here

# Supported formats: .mp4, .avi, .mov
# Recommended datasets:
# - FaceForensics++: https://justusthies.github.io/posts/faceforensics++/
# - DFDC: https://www.kaggle.com/competitions/deepfake-detection-challenge/data
```

### 2. Train the Model

```bash
# Train with default settings (EfficientNetB0, 50 epochs)
python train.py --dataset-dir dataset/ --output-dir outputs/

# Train with specific backbone and epochs
python train.py --dataset-dir dataset/ --output-dir outputs/ \
    --backbone resnet50 --epochs 100 --batch-size 32

# Train with limited videos per class (for testing)
python train.py --dataset-dir dataset/ --output-dir outputs/ \
    --max-videos 50 --epochs 20
```

### 3. Run the Web Application

```bash
# Start Flask development server
python app.py

# Or with production settings
flask run --host 0.0.0.0 --port 5000
```

Access the application at: `http://localhost:5000`

### 4. Run Prediction (CLI)

```bash
# Predict on a single video
python predict.py --video path/to/video.mp4 \
    --model saved_models/best_deepfake_model.h5 \
    --output-dir outputs/
```

---

## Web Application Pages

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Project overview, features, and statistics |
| About | `/about` | Project description and methodology |
| Upload | `/upload` | Video upload with drag-and-drop |
| Prediction | `/predict/<filename>` | Detailed prediction results with charts |
| Dashboard | `/dashboard` | Admin training dashboard |
| Metrics | `/metrics` | Model evaluation metrics and plots |
| Dataset | `/dataset` | Dataset information and statistics |
| Contact | `/contact` | Contact form and information |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Upload video and get JSON prediction |
| `/api/history` | GET | Get paginated prediction history |
| `/api/dataset-stats` | GET | Get dataset statistics |

---

## Generated Visualizations

### Charts (12 files in `plots/`)

1. `confusion_matrix.png` - Confusion matrix heatmap
2. `roc_curve.png` - ROC curve with AUC
3. `precision_recall_curve.png` - Precision-Recall curve
4. `training_history.png` - 2x2 training metrics
5. `accuracy_curve.png` - Training/validation accuracy
6. `loss_curve.png` - Training/validation loss
7. `feature_importance.png` - Top 15 feature importance
8. `prediction_distribution.png` - Probability distribution
9. `class_distribution.png` - Class distribution charts
10. `dataset_statistics.png` - Dataset split statistics
11. `model_comparison.png` - Backbone comparison
12. `inference_time.png` - Inference time analysis

### UML Diagrams (8 files in `plots/`)

13. `use_case_diagram.png` - System use cases
14. `sequence_diagram.png` - Prediction sequence
15. `class_diagram.png` - System classes
16. `activity_diagram.png` - Prediction activity flow
17. `dfd_diagram.png` - Data flow diagram
18. `er_diagram.png` - Database ER diagram
19. `architecture_diagram.png` - System architecture
20. `methodology_diagram.png` - Methodology flow

---

## Configuration

All configuration is managed in `config.py`. Key parameters:

```python
FRAME_EXTRACTION_INTERVAL_MS = 100     # Frame extraction interval
TARGET_FACE_SIZE = (224, 224)           # Face crop size
AUDIO_SAMPLE_RATE = 16000              # Audio resampling rate
LIP_SYNC_THRESHOLD = 0.55              # Lip sync decision threshold
RPPG_BANDPASS_LOW = 0.7                # rPPG filter low cutoff (Hz)
RPPG_BANDPASS_HIGH = 4.0              # rPPG filter high cutoff (Hz)
BATCH_SIZE = 16                        # Training batch size
EPOCHS = 50                            # Maximum training epochs
LEARNING_RATE = 1e-4                   # Initial learning rate
LSTM_UNITS = 128                       # Bi-LSTM hidden units
DROPOUT_RATE = 0.5                     # Dropout rate
DEFAULT_BACKBONE = 'efficientnetb0'    # CNN backbone
```

---

## Recommended Datasets

1. **FaceForensics++** - [Dataset Page](https://justusthies.github.io/posts/faceforensics++/)
   - 1000 original + 4000 manipulated videos
   - 4 manipulation methods: Deepfakes, Face2Face, FaceSwap, NeuralTextures

2. **DFDC** - [Kaggle](https://www.kaggle.com/competitions/deepfake-detection-challenge/data)
   - 100,000+ video clips
   - Diverse manipulation techniques and demographics

3. **Celeb-DF** (Optional) - [Dataset Page](https://cse.buffalo.edu/~siweilyu/celeb-deepfakeforensics.html)

4. **FakeAVCeleb** (Optional) - [GitHub](https://github.com/DASH-Lab/FakeAVCeleb)

---

## Deployment

```bash
# Production deployment with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# With Docker
# 1. Create Dockerfile (not included but straightforward)
# 2. docker build -t deepfake-detector .
# 3. docker run -p 5000:5000 deepfake-detector
```

---

## License

This project is developed for academic purposes as part of a B.E. in Artificial Intelligence & Machine Learning.

---

## Acknowledgments

- FaceForensics++ dataset by Andreas Rossler et al.
- MediaPipe by Google Research
- TensorFlow and Keras by Google
- All open-source libraries used in this project
