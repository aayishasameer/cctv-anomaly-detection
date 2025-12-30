# CCTV Anomaly Detection System

A real-time anomaly detection system for CCTV surveillance using Variational Autoencoders (VAE) and multi-object tracking. The system detects suspicious behavior like theft or unusual activities and changes bounding box colors from green (normal) to red (anomaly) in real-time.

## Features

- **Multi-Object Tracking**: Uses YOLOv8 + BotSORT for robust person tracking
- **Anomaly Detection**: VAE-based behavioral analysis to detect suspicious activities
- **Real-time Visualization**: 
  - Green boxes: Normal behavior
  - Orange boxes: Warning/suspicious
  - Red boxes: Confirmed anomaly
- **Batch Processing**: Process multiple videos automatically
- **Feature Extraction**: Advanced behavioral features including motion patterns, trajectory analysis, and size variations

## System Architecture

```
Input Video → YOLO Detection → BotSORT Tracking → Feature Extraction → VAE Analysis → Anomaly Classification → Visualization
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cctv-anomaly-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLOv8 model (automatic on first run):
```bash
# The system will automatically download yolov8n.pt on first use
```

## Quick Start

### 1. Train the Anomaly Detection Model

First, train the VAE model on normal behavior videos:

```bash
python train_vae_model.py
```

This will:
- Extract behavioral features from videos in `working/normal_shop/`
- Train a Variational Autoencoder on normal patterns
- Save the trained model to `models/vae_anomaly_detector.pth`

### 2. Run Anomaly Detection on Single Video

```bash
python anomaly_detection_tracker.py --input path/to/video.mp4 --output output_video.mp4
```

### 3. Batch Process Multiple Videos

```bash
python batch_anomaly_detection.py --input-dir working/test_anomaly --output-dir results/
```

## Usage Examples

### Single Video Processing
```bash
# With display window
python anomaly_detection_tracker.py -i working/test_anomaly/Shoplifting005_x264.mp4 -o results/anomaly_output.mp4

# Without display (headless)
python anomaly_detection_tracker.py -i video.mp4 -o output.mp4 --no-display
```

### Batch Processing
```bash
# Process all videos in test_anomaly folder
python batch_anomaly_detection.py -i working/test_anomaly -o results/batch_output
```

## File Structure

```
cctv-anomaly-detection/
├── vae_anomaly_detector.py      # Core VAE anomaly detection classes
├── train_vae_model.py           # Training script for VAE model
├── anomaly_detection_tracker.py # Real-time anomaly detection
├── batch_anomaly_detection.py   # Batch processing script
├── run_mot_tracking.py          # Original tracking script
├── botsort.yaml                 # BotSORT tracker configuration
├── requirements.txt             # Python dependencies
├── models/                      # Trained models directory
│   └── vae_anomaly_detector.pth # Trained VAE model
├── data/                        # Data and results
│   ├── features/               # Extracted features
│   └── processed/              # Processing results
└── working/                     # Input videos
    ├── normal_shop/            # Normal behavior videos (training)
    └── test_anomaly/           # Test videos with anomalies
```

## How It Works

### 1. Feature Extraction
The system extracts behavioral features from tracked persons:
- **Motion features**: Speed, acceleration, direction changes
- **Trajectory features**: Path efficiency, displacement, tortuosity
- **Size features**: Bounding box variations over time
- **Position features**: Spatial movement patterns

### 2. VAE Training
- Trains on normal behavior patterns from `working/normal_shop/`
- Learns to reconstruct normal behavioral features
- Sets anomaly threshold at 95th percentile of reconstruction errors

### 3. Anomaly Detection
- Calculates reconstruction error for new behaviors
- High reconstruction error indicates anomalous behavior
- Uses temporal smoothing to reduce false positives

### 4. Visualization
- **Green**: Normal behavior (low reconstruction error)
- **Orange**: Warning (moderate reconstruction error)
- **Red**: Confirmed anomaly (high reconstruction error over multiple frames)

## Configuration

### Anomaly Detection Parameters
Edit `vae_anomaly_detector.py` to adjust:
- `sequence_length`: Number of frames for feature extraction (default: 30)
- `anomaly_threshold_frames`: Frames needed to confirm anomaly (default: 5)
- VAE architecture parameters (hidden_dim, latent_dim)

### Tracking Parameters
Edit `botsort.yaml` to adjust:
- Detection confidence threshold
- Tracking parameters
- ReID settings

## Training Data Requirements

For best results:
- **Normal videos**: 20+ videos of normal behavior (shopping, walking, etc.)
- **Video quality**: Clear visibility of people
- **Duration**: At least 30 seconds per video
- **Variety**: Different lighting, angles, and scenarios

## Performance Tips

1. **GPU Acceleration**: Install CUDA-enabled PyTorch for faster processing
2. **Batch Size**: Adjust batch size based on available memory
3. **Video Resolution**: Lower resolution videos process faster
4. **Feature Sequence**: Longer sequences provide better anomaly detection but slower processing

## Troubleshooting

### Model Not Found Error
```bash
# Train the model first
python train_vae_model.py
```

### CUDA Out of Memory
- Reduce batch size in training
- Use CPU instead: Set `device = 'cpu'` in code

### Poor Detection Performance
- Add more diverse training videos
- Adjust anomaly threshold
- Increase sequence length for feature extraction

## Results Interpretation

The system outputs:
- **Processed videos**: With colored bounding boxes
- **JSON results**: Detailed anomaly information including timestamps and scores
- **Console logs**: Real-time processing statistics

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.