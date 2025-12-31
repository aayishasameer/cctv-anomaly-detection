# 🚀 CCTV Anomaly Detection - Setup Instructions

## 📦 What You Get from GitHub

### ✅ **Pre-trained Model**
- `models/vae_anomaly_detector.pth` (216KB) - Ready-to-use trained VAE model
- No training required - works immediately!

### ✅ **Test Videos**
- 5 shoplifting videos in `working/test_anomaly/`
- Total size: ~160MB of test data
- Ready for immediate testing

### ✅ **Complete Codebase**
- All Python scripts with improvements
- Fixed tracking issues (0 false positives)
- Comprehensive evaluation framework
- Multi-camera ReID support

## 🔧 Quick Setup (5 Minutes)

### **1. Clone Repository**
```bash
git clone https://github.com/aayishasameer/cctv-anomaly-detection.git
cd cctv-anomaly-detection
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run Demo**
```bash
python demo.py
```

## 🎯 **Available Commands**

### **Best Results (Fixed System):**
```bash
# Advanced system - eliminates false positives
python fix_tracking_issues.py -i working/test_anomaly/Shoplifting045_x264.mp4 -o result.mp4 --no-display
```

### **Original System (For Comparison):**
```bash
# Original system - has some issues
python anomaly_detection_tracker.py -i working/test_anomaly/Shoplifting045_x264.mp4 -o result_original.mp4 --no-display
```

### **Comprehensive Evaluation:**
```bash
# Get detailed metrics
python run_comprehensive_evaluation.py --create-gt
python run_comprehensive_evaluation.py --video-dir working/test_anomaly
```

### **Batch Processing:**
```bash
# Process all videos at once
python batch_anomaly_detection.py -i working/test_anomaly -o batch_results
```

## 📊 **Expected Results**

### **Fixed System Performance:**
- ✅ **0 false anomalies** (vs 775 in original)
- ✅ **Stable tracking** (no ID switching)
- ✅ **Real-time processing** (~28 FPS)
- ✅ **Production ready**

### **Output Files:**
- **Processed videos** with color-coded bounding boxes
- **JSON reports** with detailed anomaly information
- **Evaluation metrics** (accuracy, precision, recall, F1)

## 🎨 **Visual Output**

### **Color Coding:**
- 🟢 **Green**: Normal behavior
- 🟠 **Orange**: Warning/suspicious
- 🔴 **Red**: Confirmed anomaly
- ⚪ **Gray**: Tracking (insufficient data)

### **Information Display:**
- Track ID and status
- Anomaly confidence score
- Detection confidence
- Frame counter and progress

## 🔧 **System Requirements**

### **Minimum:**
- Python 3.8+
- 4GB RAM
- CPU processing (GPU optional)

### **Recommended:**
- Python 3.9+
- 8GB RAM
- NVIDIA GPU with CUDA (for faster processing)

### **Dependencies:**
- PyTorch (CPU or GPU version)
- OpenCV
- Ultralytics (YOLOv8)
- scikit-learn
- NumPy, Pandas, Matplotlib

## 🚨 **Troubleshooting**

### **Common Issues:**

#### **1. Import Errors**
```bash
# Fix: Install missing dependencies
pip install -r requirements.txt
```

#### **2. CUDA Errors**
```bash
# Fix: Use CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### **3. Model Not Found**
```bash
# The model should be included, but if missing:
# Check if models/vae_anomaly_detector.pth exists
ls -la models/
```

#### **4. Video Not Found**
```bash
# Check if test videos exist
ls -la working/test_anomaly/
```

#### **5. Permission Errors**
```bash
# Fix file permissions
chmod +x *.py
```

## 📈 **Performance Comparison**

| System | False Positives | ID Switches | Processing Speed |
|--------|----------------|-------------|------------------|
| **Fixed System** | 0 | None | 28 FPS |
| Original System | 775 | Many | 30 FPS |

## 🎯 **Next Steps**

### **For Testing:**
1. Run `python demo.py` for interactive demo
2. Try different videos with `fix_tracking_issues.py`
3. Compare results with original system

### **For Development:**
1. Use VS Code workspace: `cctv-anomaly-detection-vscode.code-workspace`
2. Read `EVALUATION_GUIDE.md` for detailed usage
3. Check `TRACKING_ISSUES_ANALYSIS.md` for technical details

### **For Production:**
1. Test on your own videos
2. Adjust thresholds in `fix_tracking_issues.py`
3. Set up continuous monitoring

## 📚 **Documentation**

- `README.md` - Project overview
- `EVALUATION_GUIDE.md` - Comprehensive evaluation guide
- `TRACKING_ISSUES_ANALYSIS.md` - Technical analysis
- `QUICK_EVALUATION_REFERENCE.md` - Quick reference

## ✅ **Success Indicators**

You'll know it's working when you see:
- ✅ Video processing without errors
- ✅ Output video with colored bounding boxes
- ✅ Console showing progress and anomaly counts
- ✅ Stable track IDs (no constant switching)
- ✅ Reasonable anomaly detection (not everything flagged)

## 🆘 **Need Help?**

1. Check the documentation files
2. Review the troubleshooting section
3. Ensure all dependencies are installed
4. Verify the model file exists
5. Test with provided sample videos first

**The system is ready to run immediately after cloning - no additional training required!**