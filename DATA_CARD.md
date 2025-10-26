
# Data Card — PRISM Pose-Only Mini

## Dataset Details

### Basic Information

- **Name**: PRISM Pose-Only Mini
- **Version**: 1.0
- **Description**: Pose keypoint sequences for human action recognition
- **Task**: Multi-class temporal action classification
- **Privacy Level**: High (no RGB frames, no faces)

### Data Type

- **Format**: NumPy arrays (`.npy`)
- **Structure**: `[T, J, 2]`
  - T = 90 frames (temporal dimension)
  - J = 33 joints (spatial dimension)
  - 2 = (x, y) normalized coordinates
- **Encoding**: Float32 (normalized to [0, 1])
- **Storage**: Binary format for efficiency

## Collection

### Source

- **Origin**: YouTube public videos
- **Collection Method**: Automated via `yt-dlp`
- **Selection Criteria**:
  - Duration: 6-15 seconds
  - Resolution: Minimum 480p
  - Format: MP4
  - Content: Full-body single-person actions

### Data Providers

- **Platform**: YouTube (public domain)
- **Attribution**: Original uploaders (publicly available content)
- **License**: Fair use for research purposes

### Collection Process

1. **Query Generation**: Define search terms in `classes.yaml`
   ```yaml
   classes:
     - name: walk
       queries: ["walking outside full body", "man walking sidewalk"]
   ```

2. **Video Download**: Bulk download with yt-dlp
   ```python
   yt-dlp "ytsearch120:{query}" \
       -f "mp4[height>=480]" \
       --match-filter "duration > 6 & duration < 15"
   ```

3. **Pose Extraction**: MediaPipe Pose per frame
4. **Windowing**: Extract overlapping 3-second sequences
5. **Normalization**: Center and scale transformations
6. **Storage**: Save as `.npy` files

### Dataset Statistics

- **Total Classes**: 30
- **Expected Videos**: ~3,600 (120 per query × 30 classes)
- **Actual Clips**: ~10,000+ pose sequences (after windowing)
- **Average Clips per Class**: ~333
- **Sequence Length**: 90 frames (3 seconds @ 30fps)
- **Temporal Overlap**: 0.5 seconds stride

### Class Distribution

```
Basic Movements:     7 classes (normal, walk, run, sit, stand_up, lie_down, fall)
Stair Actions:       2 classes (climb_stairs, descend_stairs)
Body Movements:      6 classes (bend, reach, jump, stretch, squat, lunge)
Object Interactions: 5 classes (pick_up, put_down, push, pull, carry)
Door Actions:        2 classes (open_door, close_door)
Daily Activities:    5 classes (drink, eat, phone, type, read)
Gestures:            2 classes (wave, wave_two_hands)
Entertainment:        1 class  (dance)
```

**Known Imbalance**:
- Some classes ("normal", "walk") may be over-represented
- Action classes with few samples need data augmentation or rebalancing

## Preprocessing

### MediaPipe Pose Extraction

**Configuration**:
```python
MediaPipe Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

**Output**: 33 landmark points
- Head: nose, eyes, ears
- Upper Body: shoulders, elbows, wrists
- Lower Body: hips, knees, ankles
- Coordination: (x, y) normalized screen coordinates [0, 1]

### Normalization Pipeline

**Step 1: Centering**
```python
ref = pose[:, :1, :]  # nose reference (joint 0)
pose_centered = pose - ref
```

**Step 2: Scaling**
```python
shoulder_dist = np.linalg.norm(
    pose[:, 11, :] - pose[:, 12, :],
    axis=-1, keepdims=True
)
scale = np.maximum(1e-3, shoulder_dist.mean())
pose_scaled = pose_centered / scale
```

**Rationale**:
- **Centering**: Removes global translation (camera/subject position)
- **Scaling**: Normalizes subject size (distance from camera)

### Windowing

**Parameters**:
- Window Length: 3 seconds (90 frames @ 30fps)
- Stride: 0.5 seconds (15 frames overlap)
- Padding: Repeat last frame if needed

**Example**:
```python
# Video: 10 seconds → 15 windows
for start in [0, 0.5, 1.0, 1.5, ...]:
    window = extract_window(start, start + 3.0)
```

## Dataset Structure

### Directory Tree

```
data/
├── mini/
│   ├── index.json              # Metadata index
│   └── clips/
│       ├── clip_00000000.npy   # Pose sequence 0
│       ├── clip_00000001.npy   # Pose sequence 1
│       └── ...
├── raw_overlap/
│   ├── walk/                   # Class organization
│   │   ├── video001/
│   │   │   ├── vid_clip_00000.npy
│   │   │   └── vid_clip_00001.npy
│   │   └── video002/
│   └── run/
└── app/                        # Source videos
    ├── walk/
    │   ├── video1.mp4
    │   └── video2.mp4
    └── run/
```

### Index Format

**`data/mini/index.json`**:
```json
[
  {
    "path": "clips/clip_00000000.npy",
    "label": "walk",
    "video_id": "video001"
  },
  {
    "path": "clips/clip_00000001.npy",
    "label": "run",
    "video_id": "video002"
  },
  ...
]
```

**Fields**:
- `path`: Relative path to `.npy` file
- `label`: Action class name
- `video_id`: Source video identifier (for group-based splits)

### Data Splits

**Strategy**: Group-based (by video_id)

```python
# Train/Val/Test: 70/15/15
train_videos = set of video_ids[:70%]
val_videos = set of video_ids[70%:85%]
test_videos = set of video_ids[85%:100%]
```

**Rationale**: Prevents data leakage
- Same video cannot appear in multiple splits
- Ensures model generalization to unseen videos
- More realistic evaluation scenario

## Composition

### Classes

**30 Action Classes** (defined in `classes.yaml`):

1. normal
2. walk
3. run
4. sit
5. stand_up
6. lie_down
7. fall
8. jump
9. climb_stairs
10. descend_stairs
11. bend
12. reach
13. pick_up
14. put_down
15. push
16. pull
17. open_door
18. close_door
19. carry
20. drink
21. eat
22. phone
23. type
24. read
25. wave
26. wave_two_hands
27. stretch
28. squat
29. lunge
30. dance

### Data Instances

- **Train**: ~7,000 windows
- **Validation**: ~1,500 windows
- **Test**: ~1,500 windows
- **Total**: ~10,000 windows

*Note: Actual numbers depend on collected data*

## Uses

### Intended Use

1. **Research**: Action recognition model development
2. **Education**: Pose-based learning demonstrations
3. **Healthcare**: Fall detection and monitoring systems
4. **Surveillance**: Privacy-preserving activity monitoring

### Out-of-Scope Use

- **Person Identification**: Cannot identify individuals from poses
- **Fine Motor Skills**: No hand/finger-level details
- **Multi-Person Interactions**: Single-person focus
- **Extreme Occlusion**: Limited when pose detection fails

## Distribution

### Storage

- **Format**: NumPy binary (`.npy`)
- **Size**: ~100 KB per sequence (90×33×2 floats)
- **Total Size**: ~1 GB for full dataset
- **Compression**: None (uncompressed for speed)

### Access

**Local**: `data/mini/` directory

**Index**: JSON file with metadata and paths

**Loading**:
```python
import numpy as np
import json

index = json.load(open('data/mini/index.json'))
for item in index:
    pose = np.load(f'data/mini/{item["path"]}')
```

### Privacy Protection

- **No RGB Data**: Only normalized coordinates
- **No Video Files**: Deleted after pose extraction
- **No Faces**: No facial data stored
- **De-identified**: Cannot identify individuals

## Maintenance

### Updates

- Dataset expands as new videos collected
- Class distribution rebalanced periodically
- Outlier sequences reviewed and removed

### Quality Control

**Checks**:
- MediaPipe confidence > 0.5
- Pose completeness (all 33 joints detected)
- Temporal consistency (no sudden jumps)
- Class label correctness

**Removal Criteria**:
- Failed pose extraction (confidence < 0.5)
- Poor video quality (resolution < 480p)
- Duplicate sequences (SHA-1 deduplication)
- Mislabeled data

### Known Issues

1. **Class Imbalance**: Some classes have fewer samples
2. **MediaPipe Failures**: Noisy detections in complex scenes
3. **Temporal Window**: 3s may miss longer actions
4. **Annotation Quality**: Automatic, not human-verified

### Future Enhancements

- More diverse action classes
- Multi-viewpoint data
- Synthetic data generation
- Data augmentation improvements
- Human-verified annotations

## Licensing

- **YouTube Videos**: Public domain / Fair use
- **Pose Data**: Derived work, permissive use
- **MediaPipe**: Apache 2.0 License
- **Research Use**: OK
- **Commercial Use**: Evaluate per use case

## Attribution

If using this dataset in research:

```
PRISM Pose-Only Dataset
Privacy-Preserving Human Action Recognition
Source: YouTube public videos
Processing: MediaPipe Pose extraction
```

## Contact

For dataset questions, see main repository documentation.

**Known Gaps**: Class imbalance, limited diversity in some classes
