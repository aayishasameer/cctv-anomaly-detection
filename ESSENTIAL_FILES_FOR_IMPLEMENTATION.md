# Essential Files for Complete CCTV Anomaly Detection & Stealing Detection System

## 🎯 **CORE SYSTEM FILES (Must Have)**

### **1. Main Detection Engines**
- `vae_anomaly_detector.py` - ⭐ **CRITICAL** - Core VAE anomaly detection engine
- `improved_anomaly_tracker.py` - ⭐ **CRITICAL** - Main behavioral anomaly tracking system
- `stealing_detection_system.py` - ⭐ **CRITICAL** - Enhanced stealing detection with adaptive zones
- `adaptive_zone_learning.py` - ⭐ **CRITICAL** - Learns interaction zones from normal videos

### **2. Configuration Files**
- `requirements.txt` - ⭐ **CRITICAL** - Python dependencies
- `botsort.yaml` - ⭐ **CRITICAL** - Person tracking configuration
- `botsort_improved.yaml` - 🔧 **IMPORTANT** - Enhanced tracking config

### **3. Model Files**
- `yolov8n.pt` - ⭐ **CRITICAL** - YOLO person detection model (auto-downloaded)
- `models/vae_anomaly_detector.pth` - ⭐ **CRITICAL** - Trained VAE model (created by training)
- `models/learned_interaction_zones.pkl` - ⭐ **CRITICAL** - Learned zones (created by zone learning)

## 🧠 **TRAINING & SETUP FILES (Essential for Setup)**

### **4. Training Scripts**
- `train_vae_model.py` - ⭐ **CRITICAL** - Train VAE on normal behavior videos
- `learn_and_test_adaptive_system.py` - ⭐ **CRITICAL** - Complete pipeline setup

### **5. Setup & Testing**
- `setup_stealing_detection.py` - 🔧 **IMPORTANT** - System setup and verification
- `test_fixed_system.py` - 🔧 **IMPORTANT** - Comprehensive system testing
- `quick_adaptive_test.py` - 🔧 **IMPORTANT** - Quick verification test

## 🎬 **DEMO & USAGE FILES (For Running System)**

### **6. Demo Scripts**
- `demo_stealing_detection.py` - 🎯 **RECOMMENDED** - Full stealing detection demo
- `demo_system.py` - 🎯 **RECOMMENDED** - Basic anomaly detection demo

### **7. Testing Scripts**
- `test_stealing_detection.py` - 🎯 **RECOMMENDED** - Stealing detection testing
- `evaluation_metrics.py` - 📊 **USEFUL** - Performance evaluation

## 📚 **DOCUMENTATION FILES (Important for Understanding)**

### **8. Documentation**
- `STEALING_DETECTION_GUIDE.md` - 📖 **IMPORTANT** - Complete system documentation
- `CCTV_Anomaly_Detection_Implementation_Presentation.md` - 📖 **IMPORTANT** - System overview
- `README.md` - 📖 **IMPORTANT** - Basic project information
- `SETUP_INSTRUCTIONS.md` - 📖 **USEFUL** - Setup guide

## 🗂️ **DATA DIRECTORIES (Must Exist)**

### **9. Required Directories**
- `working/normal_shop/` - ⭐ **CRITICAL** - Normal behavior videos for training
- `models/` - ⭐ **CRITICAL** - Trained models storage
- `working/test_anomaly/` - 🎯 **RECOMMENDED** - Test videos
- `results/` - 📊 **USEFUL** - Output results

## ❌ **FILES YOU CAN DELETE (Redundant/Old)**

### **10. Redundant Files**
- `enhanced_anomaly_tracker.py` - ❌ **DELETE** - Superseded by improved version
- `demo_enhanced_system.py` - ❌ **DELETE** - Old demo version
- `demo.py` - ❌ **DELETE** - Basic demo, use newer versions
- `batch_anomaly_detection.py` - ❌ **DELETE** - Old batch processing
- `fix_tracking_issues.py` - ❌ **DELETE** - Temporary fix script
- `check_threshold.py` - ❌ **DELETE** - Debugging script
- `check_training_data.py` - ❌ **DELETE** - Debugging script
- `minimal_retrain_vae.py` - ❌ **DELETE** - Use main training script
- `quick_retrain_enhanced_vae.py` - ❌ **DELETE** - Use main training script
- `retrain_enhanced_vae.py` - ❌ **DELETE** - Use main training script
- `fast_full_processing.py` - ❌ **DELETE** - Old processing script
- `quick_output_generator.py` - ❌ **DELETE** - Old output script
- `run_mot_tracking.py` - ❌ **DELETE** - Standalone tracking (integrated now)
- `multi_camera_reid.py` - ❌ **DELETE** - Advanced feature not implemented
- `run_comprehensive_evaluation.py` - ❌ **DELETE** - Use evaluation_metrics.py
- `test_improvements.py` - ❌ **DELETE** - Old testing script
- `setup_vscode_project.py` - ❌ **DELETE** - IDE setup only

### **11. Old Documentation**
- `PHASE1_IMPROVEMENTS_SUMMARY.md` - ❌ **DELETE** - Outdated
- `TRACKING_ISSUES_ANALYSIS.md` - ❌ **DELETE** - Issues resolved
- `EVALUATION_GUIDE.md` - ❌ **DELETE** - Use STEALING_DETECTION_GUIDE.md
- `QUICK_EVALUATION_REFERENCE.md` - ❌ **DELETE** - Use main guide

### **12. Output Files (Can Delete)**
- `*.mp4` files - ❌ **DELETE** - Old output videos (regenerate as needed)
- `training_log.txt` - ❌ **DELETE** - Old training logs
- `test_020_metrics.json` - ❌ **DELETE** - Old test results
- `sample_ground_truth.json` - ❌ **DELETE** - Sample only

## 🚀 **MINIMAL WORKING SYSTEM (Absolute Essentials)**

If you want the **absolute minimum** files for a working system:

### **Core Files (8 files)**
1. `vae_anomaly_detector.py`
2. `improved_anomaly_tracker.py` 
3. `stealing_detection_system.py`
4. `adaptive_zone_learning.py`
5. `train_vae_model.py`
6. `requirements.txt`
7. `botsort.yaml`
8. `learn_and_test_adaptive_system.py`

### **Required Directories**
- `working/normal_shop/` (with normal behavior videos)
- `models/` (will be created)

### **Usage**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train VAE model
python train_vae_model.py

# 3. Learn zones and test system
python learn_and_test_adaptive_system.py

# 4. Run stealing detection
python stealing_detection_system.py --input test_video.mp4
```

## 📋 **RECOMMENDED FULL SYSTEM (15 files)**

For a **complete, production-ready** system:

### **Essential Files**
1. `vae_anomaly_detector.py`
2. `improved_anomaly_tracker.py`
3. `stealing_detection_system.py`
4. `adaptive_zone_learning.py`
5. `train_vae_model.py`
6. `learn_and_test_adaptive_system.py`
7. `demo_stealing_detection.py`
8. `test_fixed_system.py`
9. `setup_stealing_detection.py`
10. `evaluation_metrics.py`
11. `requirements.txt`
12. `botsort.yaml`
13. `botsort_improved.yaml`
14. `STEALING_DETECTION_GUIDE.md`
15. `README.md`

## 🎯 **FILE PRIORITY LEVELS**

### ⭐ **CRITICAL** (System won't work without these)
- Core detection engines
- Configuration files  
- Training scripts
- Model files

### 🔧 **IMPORTANT** (Needed for proper setup/testing)
- Setup scripts
- Testing scripts
- Enhanced configurations

### 🎯 **RECOMMENDED** (For full functionality)
- Demo scripts
- Evaluation tools

### 📊 **USEFUL** (Nice to have)
- Additional documentation
- Analysis tools

### ❌ **DELETE** (Redundant/outdated)
- Old versions
- Debugging scripts
- Temporary files

## 🏆 **FINAL RECOMMENDATION**

**Keep these 20 files for complete system:**

1. `vae_anomaly_detector.py` ⭐
2. `improved_anomaly_tracker.py` ⭐
3. `stealing_detection_system.py` ⭐
4. `adaptive_zone_learning.py` ⭐
5. `train_vae_model.py` ⭐
6. `learn_and_test_adaptive_system.py` ⭐
7. `demo_stealing_detection.py` 🎯
8. `test_fixed_system.py` 🔧
9. `test_stealing_detection.py` 🎯
10. `setup_stealing_detection.py` 🔧
11. `evaluation_metrics.py` 📊
12. `requirements.txt` ⭐
13. `botsort.yaml` ⭐
14. `botsort_improved.yaml` 🔧
15. `STEALING_DETECTION_GUIDE.md` 📖
16. `CCTV_Anomaly_Detection_Implementation_Presentation.md` 📖
17. `README.md` 📖
18. `yolov8n.pt` ⭐
19. `working/normal_shop/` (directory with videos) ⭐
20. `models/` (directory for trained models) ⭐

**Delete everything else** to clean up your project!