# 🤖 AI Models - Download Instructions

## 📦 Included Models (In Repository)

The following trained models are included in the `models/` directory:

| Model | Size | Status | Description |
|-------|------|--------|-------------|
| `vae_anomaly_detector.pth` | 356 KB | ✅ Included | VAE behavioral anomaly detection |
| `quick_anomaly_detector.pth` | 58 KB | ✅ Included | Neural anomaly classifier (100% accuracy) |
| `advanced_anomaly_detector.pth` | 518 KB | ✅ Included | Advanced anomaly model (99.6% accuracy) |
| `learned_interaction_zones.pkl` | 1 KB | ✅ Included | Learned interaction zones (1,041 points) |
| `learned_interaction_zones.json` | 1 KB | ✅ Included | Zones in JSON format |
| `quick_training_summary.json` | 1 KB | ✅ Included | Training metrics and results |

## 📥 Large Model (Not Included - GitHub Size Limit)

### Person ReID Model
- **File**: `person_reid_model.pth`
- **Size**: 111 MB
- **Status**: ❌ Not included (exceeds GitHub 100MB limit)
- **Purpose**: Person re-identification for global tracking

### How to Get the ReID Model:

#### Option 1: Train It Yourself (Recommended)
```bash
# Train the ReID model from scratch
python train_simple_reid.py
```
This will create `models/person_reid_model.pth` automatically.

#### Option 2: Download from Release (If Available)
Check the [Releases](https://github.com/aayishasameer/cctv-anomaly-detection/releases) page for downloadable model files.

#### Option 3: Use Git LFS (Advanced)
If you have Git LFS installed:
```bash
git lfs install
git lfs pull
```

## 🚀 Quick Start Without ReID Model

The system can run without the ReID model with reduced functionality:

```bash
# Run basic anomaly detection (no ReID)
python enhanced_cctv_system.py

# Run with all available models
python integrated_cctv_system.py
```

The system will automatically detect which models are available and adapt accordingly.

## 🔧 Training All Models

To train all models from scratch:

```bash
# 1. Train VAE anomaly detector
python train_vae_model.py

# 2. Train neural anomaly classifier
python quick_advanced_trainer.py

# 3. Train advanced anomaly model
python advanced_anomaly_trainer.py

# 4. Train ReID model
python train_simple_reid.py

# 5. Learn interaction zones
python adaptive_zone_learning.py -n working/normal_shop/*.mp4
```

## 📊 Model Performance

| Model | Accuracy | Training Time | Use Case |
|-------|----------|---------------|----------|
| VAE Anomaly | Trained | ~30 min | Behavioral anomaly detection |
| Neural Classifier | 100% | ~5 min | Quick anomaly classification |
| Advanced Model | 99.6% | ~2 hours | Deep behavioral analysis |
| ReID Model | Trained | ~1 hour | Person tracking across frames |
| Zone Learning | 1,041 zones | ~3 hours | Interaction zone detection |

## 🎯 Model Requirements

### Minimum (Basic Functionality):
- ✅ YOLO v8 (auto-downloaded)
- ✅ VAE Anomaly Detector

### Recommended (Full Functionality):
- ✅ YOLO v8
- ✅ VAE Anomaly Detector
- ✅ Neural Anomaly Classifier
- ✅ Person ReID Model
- ✅ Learned Interaction Zones

### Advanced (Maximum Performance):
- ✅ All above models
- ✅ Advanced Anomaly Model
- ✅ Custom trained zones for your environment

## 💾 Storage Requirements

- **Minimum**: ~400 MB (without ReID)
- **Full System**: ~520 MB (with all models)
- **With Training Data**: ~2-5 GB

## 🔄 Model Updates

Models are versioned and can be retrained as needed:

```bash
# Check model versions
python -c "import torch; print(torch.load('models/quick_anomaly_detector.pth')['test_accuracy'])"

# Retrain if needed
python quick_advanced_trainer.py
```

## 📝 Notes

1. **ReID Model**: While not included in the repository, it can be easily trained in ~1 hour
2. **YOLO Model**: Auto-downloaded on first run (yolov8n.pt)
3. **All Other Models**: Included and ready to use
4. **Training Data**: Not included (add your own videos to `working/` directory)

## 🆘 Troubleshooting

### Model Not Found Error
```bash
# If you see "Model not found" errors:
python train_simple_reid.py  # For ReID model
python quick_advanced_trainer.py  # For neural classifier
```

### Model Loading Error
```bash
# Check PyTorch version compatibility
pip install torch torchvision --upgrade
```

### Performance Issues
```bash
# Use smaller models for faster processing
# Edit config in integrated_cctv_system.py
# Set performance_mode = 'fast'
```

## 📧 Support

For model-related issues:
1. Check training logs in console output
2. Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`
3. Ensure sufficient disk space for model files
4. Review training scripts for configuration options

---

**Note**: The system is designed to work with whatever models are available. Missing models will be automatically detected and the system will adapt its functionality accordingly.
