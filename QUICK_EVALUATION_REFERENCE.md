# 🚀 Quick Evaluation Reference Card

## Essential Commands

### 1. First-Time Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Train model (if needed)
python train_vae_model.py

# Create sample ground truth
python run_comprehensive_evaluation.py --create-gt
```

### 2. Basic Evaluation
```bash
# Evaluate all test videos
python run_comprehensive_evaluation.py

# Single video evaluation
python run_comprehensive_evaluation.py -v working/test_anomaly/Shoplifting055_x264.mp4

# With ReID enabled
python run_comprehensive_evaluation.py --enable-reid
```

### 3. Custom Evaluation
```bash
# Custom directories
python run_comprehensive_evaluation.py \
    --video-dir /path/to/videos \
    --ground-truth /path/to/gt.json \
    --output-dir results

# Enhanced tracker with detailed metrics
python enhanced_anomaly_tracker.py \
    -i video.mp4 \
    -o output.mp4 \
    --ground-truth gt.json \
    --results-file metrics.json
```

## Key Metrics Quick Reference

| Metric | Good | Excellent | Command to Check |
|--------|------|-----------|------------------|
| Accuracy | >0.85 | >0.90 | Check `evaluation_report.txt` |
| Precision | >0.75 | >0.85 | Look for false positives |
| Recall | >0.70 | >0.80 | Look for missed anomalies |
| F1-Score | >0.72 | >0.82 | Balanced performance |
| FPS | >15 | >25 | Processing speed |

## Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| Low accuracy | `python train_vae_model.py` (retrain) |
| Slow processing | Add `conf=0.5` in tracker config |
| Missing anomalies | Lower threshold in `vae_anomaly_detector.py` |
| Too many false positives | Increase threshold percentile to 99.9 |
| ReID not working | Check `--enable-reid` flag |

## File Locations

- **Results**: `evaluation_results/`
- **Ground Truth**: `sample_ground_truth.json`
- **Models**: `models/vae_anomaly_detector.pth`
- **Test Videos**: `working/test_anomaly/`
- **Reports**: `evaluation_results/evaluation_report.txt`