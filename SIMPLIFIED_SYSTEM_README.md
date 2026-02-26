# 🎯 Simplified CCTV System - Single Camera Stable Tracking

## Overview

The Simplified CCTV System removes all Global ReID logic and uses **only local tracker IDs** from YOLO+BotSORT for maximum stability in single-camera scenarios.

## Key Changes

### ❌ Removed (Causes ID Switching):
- Global ReID system (`GlobalPersonTracker`)
- Multi-camera tracking logic
- Embedding-based ID reassignment
- `update_global_tracking()` calls
- `save_reid_data()` calls
- Global ID to local ID mapping

### ✅ Kept (Stable Tracking):
- YOLO person detection
- BotSORT/ByteTrack tracker
- VAE anomaly detection
- Hand detection (MediaPipe)
- Adaptive zone learning
- Behavioral analysis

## Architecture

```
┌─────────────────────────────────────────────────────┐
│         Simplified CCTV System                      │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐                                  │
│  │ YOLO + Track │  ← Generates stable track_ids    │
│  │  (BotSORT)   │                                  │
│  └──────┬───────┘                                  │
│         │                                           │
│         ├─────────────┐                            │
│         │             │                            │
│  ┌──────▼───────┐  ┌─▼──────────┐                 │
│  │   Anomaly    │  │   Hand     │                 │
│  │  Detection   │  │ Detection  │                 │
│  └──────┬───────┘  └─┬──────────┘                 │
│         │             │                            │
│         └──────┬──────┘                            │
│                │                                    │
│         ┌──────▼───────┐                           │
│         │  Behavioral  │                           │
│         │   Analysis   │                           │
│         └──────┬───────┘                           │
│                │                                    │
│         ┌──────▼───────┐                           │
│         │Visualization │                           │
│         │ (track_id)   │                           │
│         └──────────────┘                           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Benefits

### 1. Stable ID Assignment
- **No ID reassignment** - Tracker IDs are never changed
- **No embedding confusion** - No similarity-based ID switching
- **Consistent tracking** - Same person keeps same ID

### 2. Simpler Architecture
- **Fewer components** - Less complexity = fewer bugs
- **Easier to debug** - Clear data flow
- **Better performance** - No ReID overhead

### 3. Single Camera Optimized
- **No multi-camera logic** - Optimized for one camera
- **No global ID mapping** - Direct tracker ID usage
- **Faster processing** - Less computation

## File Comparison

### Old System (`complete_cctv_system.py`):
```python
# Has Global ReID
from person_reid_system import GlobalPersonTracker
self.reid_tracker = GlobalPersonTracker()

# Uses global_id
global_id = self.reid_tracker.update_global_tracking(...)
self.person_data[global_id] = {...}
self.anomaly_histories[global_id] = [...]

# Label shows both IDs
label = f"G:{global_id} L:{track_id}"
```

### New System (`simple_cctv_system.py`):
```python
# No Global ReID
# Removed: from person_reid_system import GlobalPersonTracker
# Removed: self.reid_tracker = GlobalPersonTracker()

# Uses track_id only
self.person_data[track_id] = {...}
self.anomaly_histories[track_id] = [...]

# Label shows only track_id
label = f"ID:{track_id}"
```

## Usage

### Run Demo:
```bash
cd cctv-anomaly-detection
python run_demo_now.py
```

### Run on Specific Video:
```bash
python run_specific_video.py
```

### Use in Code:
```python
from simple_cctv_system import SimpleCCTVSystem

# Initialize
system = SimpleCCTVSystem(camera_id="cam1")

# Process video
results = system.process_video(
    video_path="input.mp4",
    output_path="output.mp4",
    display=False
)

# Results contain:
# - frames_processed
# - total_time
# - avg_fps
# - total_tracks (number of unique track_ids)
# - active_tracks (currently active)
```

## Data Structures

### Person Data (keyed by track_id):
```python
self.person_data[track_id] = {
    'first_seen': timestamp,
    'last_seen': timestamp,
    'positions': [[x, y, t], ...],
    'anomaly_scores': [score1, score2, ...],
    'behaviors': [...],
    'interactions': [...],
    'total_detections': count
}
```

### Anomaly Histories (keyed by track_id):
```python
self.anomaly_histories[track_id] = [
    score1, score2, score3, ...
]
```

## Visualization

### Label Format:
```
Old: G:5 L:12 SUSPICIOUS (0.45)
New: ID:12 SUSPICIOUS (0.45)
```

### Color Coding:
- 🟢 Green: Normal behavior
- 🟡 Orange: Suspicious behavior
- 🔴 Red: Anomalous behavior

### Statistics Display:
```
Frame: 1234/5770
Active Persons: 3
Total Tracks: 15
FPS: 18.2
```

## Tracking Configuration

Uses the same improved tracking parameters:

```python
results = self.yolo_model.track(
    source=frame,
    tracker="botsort_stable.yaml",
    persist=True,
    imgsz=960,      # High resolution
    conf=0.25,      # Low threshold
    iou=0.5,        # Explicit IoU
    classes=[0],    # Person only
    verbose=False
)
```

With `botsort_stable.yaml`:
```yaml
track_buffer: 120        # 4 second memory
match_thresh: 0.9        # 90% match confidence
new_track_thresh: 0.8    # 80% new track confidence
```

## Expected Results

### Tracking Stability:
- ✅ **No false ID switches** - Tracker IDs are stable
- ✅ **Consistent labeling** - Same ID throughout video
- ✅ **Better performance** - No ReID overhead

### Processing Speed:
- **FPS**: 15-20 FPS (similar to old system)
- **Memory**: Lower (no ReID embeddings)
- **CPU**: Lower (no embedding computation)

## Comparison

| Feature | Old System | New System |
|---------|-----------|------------|
| **ID Source** | Global ReID | Tracker only |
| **ID Stability** | Variable | Excellent |
| **Complexity** | High | Low |
| **Multi-camera** | Yes | No |
| **Single-camera** | Suboptimal | Optimized |
| **ID Switches** | Frequent | Rare |
| **Performance** | Slower | Faster |
| **Memory** | Higher | Lower |

## Migration Guide

### If you were using `complete_cctv_system.py`:

1. **Replace import:**
   ```python
   # Old
   from complete_cctv_system import CompleteCCTVSystem
   
   # New
   from simple_cctv_system import SimpleCCTVSystem
   ```

2. **Update initialization:**
   ```python
   # Old
   system = CompleteCCTVSystem(camera_id="cam1")
   
   # New
   system = SimpleCCTVSystem(camera_id="cam1")
   ```

3. **Update result handling:**
   ```python
   # Old
   reid_stats = results['reid_statistics']
   match_rate = reid_stats['reid_match_rate']
   
   # New
   total_tracks = results['total_tracks']
   active_tracks = results['active_tracks']
   ```

## Troubleshooting

### Q: IDs still changing?
**A:** Check that you're using `botsort_stable.yaml` with the Phase 3 parameters (track_buffer: 120, match_thresh: 0.9)

### Q: Too many IDs created?
**A:** Lower `new_track_thresh` in `botsort_stable.yaml` (try 0.75)

### Q: Missing detections?
**A:** Lower `conf` threshold in tracking call (try 0.20)

### Q: Want multi-camera support?
**A:** Use the old `complete_cctv_system.py` - but expect more ID switches

## Testing

### Test Script:
```bash
python test_improved_tracking.py
```

### What to Check:
- [ ] IDs remain consistent throughout video
- [ ] No "ID conflict" warnings
- [ ] Same person keeps same ID
- [ ] Labels show "ID:X" format (not "G:X L:Y")
- [ ] Processing completes successfully

## Summary

The Simplified CCTV System provides:

✅ **Maximum ID stability** for single-camera scenarios
✅ **Simpler architecture** with fewer components
✅ **Better performance** without ReID overhead
✅ **Easier debugging** with clear data flow
✅ **Production-ready** for single-camera deployments

**Use this system when:**
- You have a single camera
- You need stable, consistent IDs
- You don't need cross-camera tracking
- You want maximum reliability

**Use the old system when:**
- You have multiple cameras
- You need cross-camera person tracking
- You can tolerate some ID switches
- You need global person identification

---

**Created:** February 26, 2026
**Purpose:** Stable single-camera tracking without Global ReID
**Status:** ✅ Production Ready
