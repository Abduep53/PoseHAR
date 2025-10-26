
# Model Card — PRISM TCN Tiny

## Model Details

### Basic Information

- **Name**: PRISM TCN Tiny
- **Version**: 1.0
- **Type**: Temporal Convolutional Network (TCN)
- **Task**: Multi-class human action recognition from pose sequences
- **Input Format**: 2D pose keypoints [T×J×2]
  - T = 90 frames (3 seconds @ 30fps)
  - J = 33 joints (MediaPipe body landmarks)
  - 2 = (x, y) normalized coordinates
- **Output Format**: Class probabilities + open-set score
- **Privacy**: No faces or visual features stored/processed

### Architecture

```
TCN_Tiny(
    joints=33,      # MediaPipe body landmarks
    classes=30,     # Number of action classes
)

Input: [B, 2, 33, 90]  # batch, x/y, joints, frames

Architecture:
├── Stem: Conv1d(66 → 64)
├── Block 1: TemporalBlock(64 → 64, dilation=1)
├── Block 2: TemporalBlock(64 → 96, dilation=2)
├── Block 3: TemporalBlock(96 → 128, dilation=4)
├── Pool: AdaptiveAvgPool1d(1)
└── Head: Sequential(
    Flatten(),
    Linear(128 → 64),
    ReLU(),
    Linear(64 → 30)
)
```

### Temporal Block Structure

Each temporal block uses dilated causal convolutions with residual connections:

```python
class TemporalBlock:
    - Conv1d(kernel=3, dilation=d) → ReLU → Conv1d(kernel=3, dilation=d)
    - Downsample connection: Conv1d(1) or Identity
    - Output: net(x) + downsample(x)
```

## Intended Use

### Primary Use Cases

1. **Real-Time Action Recognition**: Classify human actions from live video streams
2. **Privacy-Preserving Monitoring**: Surveillance without storing identifiable visual data
3. **Fall Detection**: Identify falls among other normal and abnormal activities
4. **Activity Analysis**: Track and analyze daily human activities in constrained environments

### Out-of-Scope Use Cases

1. **Multi-Person Scenarios**: Model processes single-person poses only
2. **Fine-Grained Gestures**: Limited to full-body actions, not detailed hand/finger gestures
3. **Complex Interactions**: Focus on individual actions, not group dynamics
4. **High Occlusion Environments**: Performance degrades with poor pose detection

### Known Limitations

- **Class Imbalance**: Some classes (e.g., "normal") may be over-represented
- **Open-Set Detection**: Threshold-based, not learned
- **Temporal Window**: 3-second windows may miss longer-duration actions
- **Single View**: Not robust to extreme camera angles or viewpoints
- **No Multi-Person**: Cannot handle crowded scenes

## Training Details

### Training Data

- **Source**: YouTube videos (public domain)
- **Classes**: 30 everyday actions (defined in `classes.yaml`)
- **Video Duration**: 6-15 seconds per clip
- **Resolution**: Minimum 480p
- **Total Clips**: ~10,000+ pose sequences
- **Pose Extraction**: MediaPipe Pose (33 landmarks)

### Preprocessing

1. **Pose Extraction**: MediaPipe processes each video frame
2. **Normalization**:
   - Center pose relative to nose (joint 0)
   - Scale by average shoulder distance (joints 11-12)
3. **Windowing**: Extract 90-frame windows with 0.5s stride
4. **Padding**: Truncate or pad to exactly 90 frames

### Training Procedure

- **Optimizer**: Adam (lr=1e-3)
- **Loss Function**: CrossEntropyLoss with class weights
- **Batch Size**: 64
- **Epochs**: 25 (with early stopping)
- **Device**: CUDA GPU (if available)
- **Augmentations**:
  - Joint dropout: 7% probability
  - Gaussian jitter: σ=2.0
  - Class weighting: 1/√frequency

### Class Weights

Automatically computed to handle imbalanced data:

```python
weight[class_i] = 1.0 / sqrt(frequency[class_i] + 1e-6)
```

### Evaluation

- **Split Strategy**: Group-based (by video_id) 70/15/15
- **Metrics**: Accuracy, Macro F1, Per-class Precision/Recall
- **Early Stopping**: Based on validation F1 (patience=10 epochs)

## Evaluation Results

### Expected Performance (30-Class Model)

**Validation Set**:
- Accuracy: ~0.85
- Macro F1: ~0.82
- Precision: ~0.85
- Recall: ~0.82

**Test Set**:
- Accuracy: ~0.83
- Macro F1: ~0.80
- Per-class metrics: See `runs/n30/metrics_test.json`

### Latency Benchmarks

- **PyTorch (GPU)**: ~15ms per window
- **ONNX (CPU)**: ~9ms per window
- **ONNX (GPU)**: ~4ms per window

See `runs/n30/latency_*.json` for detailed benchmarks.

### Confusion Patterns

Common misclassifications:
- Similar body movements (stretch vs. reach)
- Temporal order (sit vs. stand_up)
- Intensity variations (walk vs. run)
- Occluded poses (fall vs. lie_down)

## Ethical Considerations

### Privacy

- **No RGB Storage**: Only pose keypoints stored
- **No Face Detection**: No facial recognition capability
- **De-identifying**: Pose data cannot identify individuals
- **Compliance**: Suitable for privacy-sensitive environments

### Bias

- **Gender**: Training data may not be gender-balanced
- **Age**: Primarily adult actions
- **Cultural**: Western-centric action definitions
- **Mobility**: Limited to standing/upright actions

### Surveillance Concerns

- **Oversight Required**: System should not be fully autonomous
- **Fair Use**: Follow local regulations on video monitoring
- **Data Retention**: Implement appropriate data deletion policies

## Usage Instructions

### Installation

```powershell
pip install torch mediapipe opencv-python
```

### Load Model

```python
import torch
from src.models.temporal_tcn import TCN_Tiny

# Load checkpoint
ckpt = torch.load('runs/n30/best.ckpt', map_location='cpu')
model = TCN_Tiny(joints=33, classes=30)
model.load_state_dict(ckpt['state_dict'])
model.eval()
```

### Prepare Input

```python
import numpy as np

# pose_data shape: [90, 33, 2]
pose_data = extract_poses_from_video(video_path, window_sec=3.0)

# Normalize
ref = pose_data[:, :1, :]  # nose reference
pose_data = pose_data - ref
scale = np.maximum(1e-3, np.linalg.norm(
    pose_data[:,11,:] - pose_data[:,12,:], 
    axis=-1, keepdims=True
)).mean()
pose_data = pose_data / scale

# Convert to tensor
x = torch.from_numpy(pose_data).permute(2, 1, 0).unsqueeze(0)
# x shape: [1, 2, 33, 90]
```

### Inference

```python
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1)
    
    # Open-set detection
    confidence = probs.max().item()
    if confidence < 0.6:
        action = "Unknown"
    else:
        action = class_names[pred_idx.item()]
```

### Export to ONNX

```powershell
python deploy/export_onnx.py \
    --weights runs/n30/best.ckpt \
    --out deploy/prism_tcn.onnx \
    --joints 33 \
    --frames 90
```

## Model Artifacts

### Files

- `runs/n30/best.ckpt`: PyTorch checkpoint with metadata
- `runs/n30/labels.json`: Class names for inference
- `deploy/prism_tcn.onnx`: ONNX export for production
- `runs/n30/metrics_val.json`: Training/validation metrics
- `runs/n30/metrics_test.json`: Test set metrics

### Checkpoint Format

```python
{
    'state_dict': {...},      # Model weights
    'meta': {
        'class_names': [...], # List of class names
        'num_classes': 30,
        'model': 'tcn_tiny'
    }
}
```

## Maintenance

### Update Frequency

Model will be updated as new data is collected and evaluation metrics improve.

### Monitoring

- Track test set accuracy monthly
- Monitor per-class precision/recall
- Check for distribution shift in incoming data

### Retraining

Trigger retraining when:
- New action classes added
- Test accuracy drops below threshold (e.g., 0.75)
- New data collected (>10% dataset increase)
