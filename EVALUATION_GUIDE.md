# CCTV Anomaly Detection System - Comprehensive Evaluation Guide

This guide provides detailed instructions for evaluating the performance and accuracy of the CCTV anomaly detection system, including multi-camera ReID capabilities.

## 📋 Table of Contents

1. [Quick Start Evaluation](#quick-start-evaluation)
2. [Performance Metrics Explained](#performance-metrics-explained)
3. [Creating Ground Truth Data](#creating-ground-truth-data)
4. [Running Comprehensive Evaluation](#running-comprehensive-evaluation)
5. [Multi-Camera ReID Evaluation](#multi-camera-reid-evaluation)
6. [Understanding Results](#understanding-results)
7. [Troubleshooting](#troubleshooting)

## 🚀 Quick Start Evaluation

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure you have trained model
python train_vae_model.py  # If not already trained
```

### 1. Create Sample Ground Truth (First Time)
```bash
python run_comprehensive_evaluation.py --create-gt
```

### 2. Run Basic Evaluation
```bash
# Evaluate all test videos
python run_comprehensive_evaluation.py --video-dir working/test_anomaly

# Evaluate single video
python run_comprehensive_evaluation.py --single-video working/test_anomaly/Shoplifting055_x264.mp4
```

### 3. View Results
Results are saved in `evaluation_results/` directory:
- `evaluation_report.txt` - Human-readable summary
- `evaluation_results.json` - Detailed metrics
- `evaluation_plots.png` - Performance visualizations

## 📊 Performance Metrics Explained

### Anomaly Detection Metrics

| Metric | Description | Formula | Good Range |
|--------|-------------|---------|------------|
| **Accuracy** | Overall correctness | (TP + TN) / (TP + TN + FP + FN) | > 0.85 |
| **Precision** | Anomaly prediction accuracy | TP / (TP + FP) | > 0.80 |
| **Recall** | Anomaly detection rate | TP / (TP + FN) | > 0.75 |
| **F1-Score** | Balanced precision/recall | 2 × (Precision × Recall) / (Precision + Recall) | > 0.75 |
| **AUC-ROC** | Classification performance | Area under ROC curve | > 0.85 |

### Multi-Object Tracking Metrics

| Metric | Description | Good Range |
|--------|-------------|------------|
| **MOTA** | Multiple Object Tracking Accuracy | > 0.70 |
| **MOTP** | Multiple Object Tracking Precision | < 50 pixels |
| **IDF1** | ID F1 Score (identity consistency) | > 0.75 |
| **ID Switches** | Identity consistency failures | < 5% of tracks |

### Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Processing FPS** | Frames processed per second | > 15 FPS |
| **Detection Time** | Time per YOLO detection | < 50ms |
| **Tracking Time** | Time per tracking update | < 20ms |
## 🎯 Creating Ground Truth Data

### Method 1: Interactive Annotation Tool

For precise evaluation, create custom ground truth:

```bash
# Launch interactive annotator
python create_ground_truth.py --video working/test_anomaly/Shoplifting055_x264.mp4
```

**Controls:**
- `Space`: Play/Pause video
- `A`: Mark current frame as ANOMALY
- `N`: Mark current frame as NORMAL  
- `D`: Delete annotation for current frame
- `S`: Save annotations
- `Q`: Quit
- `Arrow Keys`: Navigate frames

**Best Practices:**
- Label at least 30% of frames for reliable evaluation
- Focus on transition periods (normal → anomaly → normal)
- Mark entire anomalous sequences, not just peak moments
- Include context frames before/after anomalies

### Method 2: Use Sample Ground Truth

For quick testing, use pre-created sample data:

```bash
python run_comprehensive_evaluation.py --create-gt
```

This creates `sample_ground_truth.json` with approximate anomaly periods for all shoplifting videos.

### Ground Truth Format

```json
{
  "video_name.mp4": {
    "anomaly_frames": [500, 501, 502, ...],
    "normal_frames": [0, 1, 2, ..., 499, 600, ...],
    "anomaly_tracks": {
      "track_id": [start_frame, end_frame]
    }
  }
}
```

## 🔬 Running Comprehensive Evaluation

### Basic Evaluation Commands

```bash
# Evaluate all videos with default settings
python run_comprehensive_evaluation.py

# Specify custom directories
python run_comprehensive_evaluation.py \
    --video-dir path/to/videos \
    --ground-truth path/to/ground_truth.json \
    --output-dir custom_results

# Enable ReID evaluation
python run_comprehensive_evaluation.py --enable-reid

# Single video with detailed output
python run_comprehensive_evaluation.py \
    --single-video video.mp4 \
    --ground-truth gt.json \
    --output-dir results
```

### Advanced Evaluation Options

```bash
# Enhanced tracker with evaluation
python enhanced_anomaly_tracker.py \
    --input video.mp4 \
    --output processed_video.mp4 \
    --ground-truth ground_truth.json \
    --enable-reid \
    --camera-id cam1 \
    --results-file detailed_results.json
```

### Batch Processing with Evaluation

```bash
# Process multiple videos with metrics
python batch_anomaly_detection.py \
    --input-dir working/test_anomaly \
    --output-dir batch_results

# Then evaluate the batch results
python run_comprehensive_evaluation.py \
    --video-dir working/test_anomaly \
    --output-dir batch_evaluation
```

## 🎥 Multi-Camera ReID Evaluation

### Setup Multi-Camera System

```python
from multi_camera_reid import MultiCameraAnomalySystem

# Initialize system
system = MultiCameraAnomalySystem()

# Add cameras
system.add_camera("cam1", "path/to/video1.mp4")
system.add_camera("cam2", "path/to/video2.mp4")

# Process frames from multiple cameras
camera_frames = {
    "cam1": frame1,
    "cam2": frame2
}
results = system.process_multi_camera_frame(camera_frames)
```

### ReID Evaluation Metrics

```bash
# Test ReID system
python multi_camera_reid.py

# Evaluate with ReID enabled
python enhanced_anomaly_tracker.py \
    --input video.mp4 \
    --enable-reid \
    --camera-id cam1
```

**ReID Performance Indicators:**
- **Cross-camera matches**: Tracks appearing in multiple cameras
- **Feature similarity**: Cosine similarity between person features
- **Global track consistency**: Stable global IDs across cameras
- **Re-identification accuracy**: Correct matching rate

## 📈 Understanding Results

### Evaluation Report Structure

```
evaluation_results/
├── evaluation_report.txt          # Human-readable summary
├── evaluation_results.json        # Detailed JSON results
├── evaluation_plots.png           # Performance visualizations
└── processed_videos/              # Output videos with annotations
```

### Interpreting Metrics

**Excellent Performance (Research Grade):**
- Accuracy > 0.90, Precision > 0.85, Recall > 0.80, F1 > 0.82
- Processing FPS > 20, Low false positive rate

**Good Performance (Production Ready):**
- Accuracy > 0.85, Precision > 0.75, Recall > 0.70, F1 > 0.72
- Processing FPS > 15, Acceptable false positive rate

**Needs Improvement:**
- Accuracy < 0.80, Precision < 0.70, Recall < 0.65, F1 < 0.67
- Processing FPS < 10, High false positive rate

### Sample Results Interpretation

```json
{
  "basic_statistics": {
    "total_frames_processed": 1250,
    "total_detections": 450,
    "total_anomalies": 89,
    "anomaly_rate": 0.198,
    "processing_fps": 18.5
  },
  "anomaly_detection_metrics": {
    "accuracy": 0.876,
    "precision": 0.823,
    "recall": 0.745,
    "f1_score": 0.782,
    "auc_roc": 0.891
  }
}
```

**Analysis:**
- **Good overall accuracy** (87.6%) indicates reliable detection
- **High precision** (82.3%) means low false positive rate
- **Moderate recall** (74.5%) suggests some anomalies are missed
- **Balanced F1-score** (78.2%) shows good overall performance
- **Excellent AUC-ROC** (89.1%) indicates strong classification ability

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Low Accuracy Scores
**Symptoms:** Accuracy < 0.70, High false positives
**Solutions:**
```bash
# Retrain with more diverse data
python train_vae_model.py

# Adjust anomaly threshold
# Edit vae_anomaly_detector.py, line 255:
self.threshold = np.percentile(reconstruction_errors, 99.5)  # More strict
```

#### 2. Poor Recall (Missing Anomalies)
**Symptoms:** Recall < 0.60, Anomalies not detected
**Solutions:**
```bash
# Lower threshold for more sensitive detection
self.threshold = np.percentile(reconstruction_errors, 98.0)  # Less strict

# Increase sequence length for better features
# Edit vae_anomaly_detector.py, line 25:
self.sequence_length = 50  # Longer sequences
```

#### 3. Slow Processing Speed
**Symptoms:** FPS < 10, High processing times
**Solutions:**
```bash
# Use GPU acceleration
export CUDA_VISIBLE_DEVICES=0

# Reduce video resolution
# Resize videos before processing

# Optimize batch size
# Edit enhanced_anomaly_tracker.py detection confidence:
conf=0.5  # Higher confidence = fewer detections
```

#### 4. ReID System Issues
**Symptoms:** Poor cross-camera matching
**Solutions:**
```python
# Adjust similarity threshold
reid_system.similarity_threshold = 0.6  # Lower = more matches

# Increase feature buffer
reid_system.feature_buffer_size = 15  # More features per track
```

#### 5. Ground Truth Mismatch
**Symptoms:** Evaluation fails, KeyError in results
**Solutions:**
```bash
# Check ground truth format
python -c "import json; print(json.load(open('ground_truth.json')))"

# Recreate ground truth
python create_ground_truth.py --video your_video.mp4

# Use sample ground truth for testing
python run_comprehensive_evaluation.py --create-gt
```

### Performance Optimization Tips

1. **Hardware Optimization:**
   ```bash
   # Use GPU if available
   nvidia-smi  # Check GPU status
   
   # Set optimal batch size
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

2. **Model Optimization:**
   ```python
   # Use smaller YOLO model for speed
   model = YOLO("yolov8s.pt")  # Instead of yolov8n.pt
   
   # Reduce detection frequency
   if frame_idx % 2 == 0:  # Process every 2nd frame
       # Run detection
   ```

3. **Memory Management:**
   ```python
   # Clear GPU cache periodically
   if frame_idx % 100 == 0:
       torch.cuda.empty_cache()
   ```

## 📚 Additional Resources

### Evaluation Best Practices
- Always use held-out test data (never seen during training)
- Evaluate on diverse scenarios (different lighting, angles, crowds)
- Compare against baseline methods when possible
- Report confidence intervals for metrics

### Research Applications
- Use this framework for academic papers
- Benchmark against other anomaly detection methods
- Analyze failure cases for system improvement
- Generate publication-quality evaluation plots

### Production Deployment
- Monitor metrics continuously in production
- Set up automated evaluation pipelines
- Use A/B testing for model updates
- Implement real-time performance monitoring

---

For more detailed information, refer to the individual script documentation and code comments.