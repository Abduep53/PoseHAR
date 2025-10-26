# PRISM — Privacy-Preserving, Robust, Open-Set Smart Monitoring

**PRISM** is an edge AI system for real-time human action recognition using pose-only data. It processes MediaPipe pose estimates through a Temporal Convolutional Network (TCN) to classify 30 everyday human actions while maintaining privacy by discarding faces and visual features.


![Python](https://img.shields.io/badge/Python-3.10-blue)
![Torch](https://img.shields.io/badge/PyTorch-2.2-red)
![ONNX](https://img.shields.io/badge/ONNX-Supported-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-success)


## 🎯 Features

- **Privacy-Preserving**: Processes pose keypoints only, no RGB frames stored
- **30 Action Classes**: Comprehensive set of everyday human actions
- **Real-Time Inference**: Optimized for edge deployment with ONNX export
- **Open-Set Detection**: Threshold-based unknown action detection
- **Scalable Pipeline**: Automated data collection, training, and evaluation
- **Multiple Demos**: Gradio web UI, USB webcam, RTSP streaming support

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Pipeline](#data-pipeline)
- [Training](#training)
- [Inference](#inference)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Results](#results)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- Windows PowerShell (or compatible shell)
- CUDA-capable GPU (optional, for faster training)

### Setup

```powershell
# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
. .\.venv\Scripts\Activate.ps1

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- **numpy** ≥1.26: Numerical computations
- **scipy** ≥1.11: Scientific computing
- **pandas** ≥2.0: Data manipulation
- **matplotlib** ≥3.8: Visualizations
- **scikit-learn** ≥1.4: Machine learning utilities
- **torch** ≥2.2: Deep learning framework
- **torchvision** ≥0.17: Computer vision utilities
- **opencv-python** ≥4.9: Video processing
- **mediapipe** ≥0.10: Pose estimation
- **gradio** ≥4.0: Web interface
- **onnx** ≥1.15: ONNX model format
- **onnxruntime** ≥1.18: ONNX inference runtime

## 🚀 Quick Start

### Minimal Example

```powershell
# Extract poses from a demo video
python -m src.data.make_dataset --video app/demo_clip.mp4 --out data/mini --window_sec 2.0

# Train model (5 epochs for demo)
python -m src.train --data_root data/mini --epochs 5 --model tcn_tiny --save runs/baseline

# Test inference
python -m src.infer --weights runs/baseline/best.ckpt --video app/demo_clip.mp4

# Launch web interface
python app/gradio_app.py

# Export to ONNX
python deploy/export_onnx.py --weights runs/baseline/best.ckpt --out deploy/prism_tcn.onnx
```

### Live Demo

```powershell
# USB webcam
python app/usb_live.py --weights runs/baseline/best.ckpt

# RTSP stream
python app/rtsp_live.py --weights runs/baseline/best.ckpt --rtsp rtsp://your-stream-url
```

### Full 30-Class Pipeline

```powershell
# Run complete end-to-end pipeline
.\scripts\run_full_pipeline.ps1
```

This executes:
1. YouTube video download (30 classes)
2. Pose extraction (parallel processing)
3. Model training (25 epochs)
4. Evaluation with metrics
5. ONNX export
6. Latency benchmarks

## 📊 Data Pipeline

### 1. Video Collection

The system downloads videos from YouTube using search queries defined in `classes.yaml`:

```yaml
classes:
  - name: walk
    queries: ["walking outside full body", "man walking sidewalk"]
  - name: run
    queries: ["jogging outdoors", "runner sprinting"]
  # ... 28 more classes
```

**Configuration**:
- Duration: 6-15 seconds per clip
- Resolution: Minimum 480p
- Format: MP4
- Deduplication: SHA-1 based

### 2. Pose Extraction

MediaPipe extracts 33 body landmark keypoints from each frame:
- **Input**: Raw video frames
- **Output**: 2D pose keypoints (x, y) normalized coordinates
- **Window**: 3 seconds (90 frames @ 30fps)
- **Stride**: 0.5 seconds overlap

**Processing**:
```python
# Normalization steps
1. Center pose (relative to nose)
2. Scale normalization (based on shoulder distance)
3. Window extraction with temporal overlap
```

### 3. Dataset Structure

```
data/
├── mini/
│   ├── index.json          # Metadata: [path, label, video_id]
│   └── clips/
│       ├── clip_00000000.npy
│       ├── clip_00000001.npy
│       └── ...
├── raw_overlap/            # Per-video pose storage
│   └── {class}/
│       └── {video_id}/
│           ├── vid_clip_00000.npy
│           └── ...
└── app/                    # Source videos (per class)
    ├── walk/
    ├── run/
    └── ...
```

**Data Split**: 70% train / 15% validation / 15% test (grouped by video_id to prevent leakage)

## 🎓 Training

### Training Configuration

```python
# Default training parameters
data_root = "data/mini"
epochs = 25
batch_size = 64
learning_rate = 1e-3
model = "tcn_tiny"
save_dir = "runs/n30"
```

### Augmentation

- **Joint Dropout** (0.07): Randomly masks out joints
- **Gaussian Jitter** (2.0): Adds noise to keypoints
- **Class Weighting**: Balances imbalanced datasets (1/√frequency)

### Class Weights

Automatically computed from training data frequencies to handle class imbalance:

```python
weight = 1.0 / sqrt(class_frequency + 1e-6)
```

### Early Stopping

Stops training if validation F1 doesn't improve for 10 epochs.

### Training Output

```
runs/n30/
├── best.ckpt              # Best model checkpoint
├── metrics_val.json       # Validation metrics per epoch
├── metrics_test.json      # Final test metrics
├── labels.json            # Class names for inference
├── confusion_val.png      # Validation confusion matrix
└── confusion_test.png     # Test confusion matrix
```

## 🔮 Inference

### Batch Processing

```python
python -m src.infer \
    --weights runs/n30/best.ckpt \
    --video input.mp4 \
    --threshold 0.6 \
    --alpha 0.6
```

### Key Features

- **EMA Smoothing**: Temporal smoothing of predictions
- **Majority Voting**: Aggregate predictions over N windows
- **Unknown Detection**: Low-confidence predictions marked as "Unknown"
- **Per-Window Probabilities**: Detailed output for each temporal window

### Parameters

- `--threshold`: Minimum probability for known actions (default: 0.6)
- `--alpha`: EMA smoothing coefficient (default: 0.6)
- `--window_size`: Majority vote window size (default: 7)

## 🏗️ Architecture

### Model: TCN_Tiny

A lightweight Temporal Convolutional Network optimized for pose sequences:

```python
TCN_Tiny(
    joints=33,      # MediaPipe body landmarks
    classes=30,     # Number of action classes
)
```

**Architecture**:
- **Input**: `[B, 2, 33, 90]` (batch, x/y, joints, frames)
- **Stem**: 1D Conv (66 → 64 channels)
- **Temporal Blocks**: 
  - Block 1: 64 → 64 (dilation=1)
  - Block 2: 64 → 96 (dilation=2)
  - Block 3: 96 → 128 (dilation=4)
- **Global Pooling**: Adaptive average pooling
- **Classifier**: Linear layers (128 → 64 → 30)

### Temporal Block

Residual temporal convolution with identity connection:

```python
class TemporalBlock(nn.Module):
    def forward(self, x):
        return self.net(x) + self.down(x)  # Residual connection
```

### Normalization Strategy

1. **Centering**: Translate to nose-relative coordinates
2. **Scaling**: Normalize by average shoulder distance
3. **Pose Representation**: 33 joints × 2 coordinates = 66 features per frame

## 📁 Project Structure

```
prism/
├── app/                    # Source videos per class
│   ├── normal/            # Videos for each action class
│   ├── walk/
│   ├── run/
│   ├── dance/
│   └── ...
├── data/                   # Processed datasets
│   ├── mini/              # Unified dataset
│   ├── raw_overlap/       # Per-video poses
│   └── app/              # Source videos organized by class
├── deploy/                # Deployment files
│   ├── export_onnx.py    # ONNX export script
│   └── prism_tcn.onnx   # Exported model
├── runs/                  # Training outputs
│   ├── baseline/        # Baseline experiments
│   ├── n30/            # 30-class training
│   └── n30_bal/        # Balanced 30-class training
├── scripts/              # Automation scripts
│   ├── yt_bulk_dl.py    # YouTube downloader
│   ├── build_dataset_from_app.py  # Pose extraction
│   ├── eval_metrics.py  # Evaluation metrics
│   ├── benchmark_*.py    # Latency benchmarks
│   └── run_full_pipeline.ps1  # One-shot pipeline
├── src/                   # Source code
│   ├── config.py        # Training configuration
│   ├── train.py         # Training script
│   ├── infer.py         # Inference script
│   ├── data/
│   │   ├── datasets.py   # Dataset utilities
│   │   ├── make_dataset.py  # Pose extraction
│   │   └── splits.py    # Data splitting logic
│   ├── models/
│   │   └── temporal_tcn.py  # TCN model architecture
│   ├── utils.py         # General utilities
│   └── utils_classes.py # Class name loading
├── classes.yaml          # 30-class configuration
├── requirements.txt      # Python dependencies
├── MODEL_CARD.md        # Model documentation
├── DATA_CARD.md        # Dataset documentation
└── README.md           # This file
```

## 💡 Usage Examples

### Example 1: Custom Training

```powershell
python -m src.train \
    --data_root data/mini \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-3 \
    --model tcn_tiny \
    --save runs/custom_experiment \
    --aug_joint_dropout 0.1 \
    --aug_jitter 3.0 \
    --use_class_weights 1 \
    --early_stop 1
```

### Example 2: Evaluation

```powershell
python scripts/eval_metrics.py \
    --weights runs/n30/best.ckpt \
    --index data/mini/index.json \
    --out_dir runs/evaluation \
    --classes_yaml classes.yaml
```

### Example 3: ONNX Export

```powershell
python deploy/export_onnx.py \
    --weights runs/n30/best.ckpt \
    --out deploy/prism_tcn_custom.onnx \
    --joints 33 \
    --frames 90
```

### Example 4: Single Video Analysis

```python
import cv2
from src.infer import load_model, predict_window

model = load_model('runs/n30/best.ckpt')
video = 'my_video.mp4'

results = predict_window(model, video)
for result in results:
    print(f"Window {result['window']}: {result['action']} ({result['confidence']:.2f})")
```

## 📈 Results

### Performance Metrics

Example metrics for 30-class model (see `runs/n30/` for your results):

**Validation Set**:
- Accuracy: ~0.85
- Macro F1: ~0.82
- Precision: ~0.85
- Recall: ~0.82

**Test Set**:
- Accuracy: ~0.83
- Macro F1: ~0.80
- Per-class metrics available in `metrics_test.json`

### Latency Benchmarks

```json
{
  "torch_ms": 15.2,
  "onnx_cpu_ms": 8.7,
  "onnx_gpu_ms": 4.3
}
```

See `runs/n30/latency_torch.json` and `runs/n30/latency_onnx.json` for detailed benchmarks.

### Confusion Matrices

Visual confusion matrices saved as:
- `runs/n30/confusion_val.png`
- `runs/n30/confusion_test.png`

## 🎯 30 Action Classes

The model recognizes these everyday human actions:

**Basic Movements**:
- `normal`, `walk`, `run`, `sit`, `stand_up`, `lie_down`, `fall`

**Stair Actions**:
- `climb_stairs`, `descend_stairs`

**Body Movements**:
- `bend`, `reach`, `jump`, `stretch`, `squat`, `lunge`

**Object Interactions**:
- `pick_up`, `put_down`, `push`, `pull`, `carry`

**Door Actions**:
- `open_door`, `close_door`

**Daily Activities**:
- `drink`, `eat`, `phone`, `type`, `read`

**Gestures**:
- `wave`, `wave_two_hands`

**Entertainment**:
- `dance`

## 🔍 Advanced Features

### Open-Set Recognition

PRISM detects unknown actions using confidence thresholding:
- Predictions below threshold marked as "Unknown"
- Configurable via `--threshold` parameter
- Default: 0.6 probability

### Temporal Smoothing

Exponential Moving Average (EMA) smooths predictions across windows:
```python
smoothed_probs = α × old_probs + (1 - α) × new_probs
```

### Group-Based Splits

Prevents data leakage by splitting at video_id level, not clip level.

### Export Formats

- **PyTorch**: `.ckpt` files with metadata
- **ONNX**: Optimized for production inference
- **TensorFlow Lite**: (planned)

## 🤝 Contributing

See individual experiment directories in `runs/` for training results and metrics.

## 📄 License

See LICENSE file for details.

## 📚 Citations

If you use PRISM in your research, please cite:

```
PRISM: Privacy-Preserving Robust Open-Set Smart Monitoring
Pose-Only Edge AI for Human Action Recognition
```

## 🎓 Model Card

See `MODEL_CARD.md` for detailed model documentation.

## 📊 Data Card

See `DATA_CARD.md` for dataset documentation.
