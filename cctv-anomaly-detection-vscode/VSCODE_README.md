# CCTV Anomaly Detection - VS Code Setup

## Quick Start

1. **Open in VS Code**: Double-click `cctv-anomaly-detection-vscode.code-workspace`
2. **Install Extensions**: VS Code will prompt to install recommended extensions
3. **Install Dependencies**: 
   ```bash
   pip install -r requirements.txt
   ```
4. **Run Demo**:
   - Press `Ctrl+Shift+P` → "Tasks: Run Task" → "Run Demo"
   - Or press `F5` to debug

## Available Commands

### Run Configurations (F5 to debug):
- **Run Anomaly Detection**: Process single video
- **Run Enhanced Tracker**: With evaluation metrics  
- **Run Evaluation**: Create ground truth and evaluate
- **Train Model**: Train the VAE model

### Tasks (Ctrl+Shift+P → "Tasks: Run Task"):
- **Install Dependencies**: Install all required packages
- **Run Demo**: Interactive demo system
- **Create Ground Truth**: Generate sample annotations
- **Run Evaluation**: Full system evaluation

## File Structure
```
cctv-anomaly-detection-vscode/
├── .vscode/                    # VS Code configuration
│   ├── settings.json          # Editor settings
│   ├── launch.json            # Debug configurations  
│   └── tasks.json             # Build tasks
├── models/                     # Trained models
├── working/                    # Input videos
├── data/                       # Processed data
├── *.py                       # Python source files
└── requirements.txt           # Dependencies
```

## Usage Examples

### 1. Process Single Video
```python
python anomaly_detection_tracker.py \
    --input working/test_anomaly/Shoplifting045_x264.mp4 \
    --output result.mp4
```

### 2. Run Evaluation
```python
python run_comprehensive_evaluation.py \
    --video-dir working/test_anomaly
```

### 3. Train Model
```python
python train_vae_model.py
```

## Debugging
- Set breakpoints by clicking left of line numbers
- Press F5 to start debugging
- Use Debug Console for interactive Python
