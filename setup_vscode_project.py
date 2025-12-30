#!/usr/bin/env python3
"""
Setup script to create VS Code project structure
"""

import os
import shutil
import json

def create_vscode_project(target_dir="cctv-anomaly-detection-vscode"):
    """Create VS Code project with all necessary files"""
    
    print(f"Creating VS Code project in: {target_dir}")
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Files to copy
    files_to_copy = [
        "vae_anomaly_detector.py",
        "anomaly_detection_tracker.py", 
        "enhanced_anomaly_tracker.py",
        "evaluation_metrics.py",
        "multi_camera_reid.py",
        "run_comprehensive_evaluation.py",
        "create_ground_truth.py",
        "batch_anomaly_detection.py",
        "train_vae_model.py",
        "demo.py",
        "demo_system.py",
        "requirements.txt",
        "README.md",
        "EVALUATION_GUIDE.md",
        "QUICK_EVALUATION_REFERENCE.md",
        "botsort.yaml",
        ".gitignore"
    ]
    
    # Copy files
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy2(file, os.path.join(target_dir, file))
            print(f"✓ Copied: {file}")
        else:
            print(f"❌ Missing: {file}")
    
    # Copy directories
    dirs_to_copy = ["models", "working", "data"]
    for dir_name in dirs_to_copy:
        if os.path.exists(dir_name):
            target_path = os.path.join(target_dir, dir_name)
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            shutil.copytree(dir_name, target_path)
            print(f"✓ Copied directory: {dir_name}")
    
    # Create VS Code settings
    vscode_dir = os.path.join(target_dir, ".vscode")
    os.makedirs(vscode_dir, exist_ok=True)
    
    # VS Code settings.json
    settings = {
        "python.defaultInterpreterPath": "python",
        "python.terminal.activateEnvironment": True,
        "files.associations": {
            "*.yaml": "yaml",
            "*.yml": "yaml"
        },
        "python.linting.enabled": True,
        "python.linting.pylintEnabled": True,
        "python.formatting.provider": "black",
        "editor.formatOnSave": True,
        "python.testing.pytestEnabled": True,
        "terminal.integrated.env.windows": {
            "PYTHONPATH": "${workspaceFolder}"
        },
        "terminal.integrated.env.linux": {
            "PYTHONPATH": "${workspaceFolder}"
        },
        "terminal.integrated.env.osx": {
            "PYTHONPATH": "${workspaceFolder}"
        }
    }
    
    with open(os.path.join(vscode_dir, "settings.json"), "w") as f:
        json.dump(settings, f, indent=2)
    
    # VS Code launch.json for debugging
    launch_config = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Run Anomaly Detection",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/anomaly_detection_tracker.py",
                "args": [
                    "--input", "working/test_anomaly/Shoplifting045_x264.mp4",
                    "--output", "output_debug.mp4",
                    "--no-display"
                ],
                "console": "integratedTerminal",
                "cwd": "${workspaceFolder}"
            },
            {
                "name": "Run Enhanced Tracker",
                "type": "python", 
                "request": "launch",
                "program": "${workspaceFolder}/enhanced_anomaly_tracker.py",
                "args": [
                    "--input", "working/test_anomaly/Shoplifting045_x264.mp4",
                    "--output", "enhanced_output.mp4",
                    "--no-display"
                ],
                "console": "integratedTerminal",
                "cwd": "${workspaceFolder}"
            },
            {
                "name": "Run Evaluation",
                "type": "python",
                "request": "launch", 
                "program": "${workspaceFolder}/run_comprehensive_evaluation.py",
                "args": ["--create-gt"],
                "console": "integratedTerminal",
                "cwd": "${workspaceFolder}"
            },
            {
                "name": "Train Model",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/train_vae_model.py",
                "console": "integratedTerminal",
                "cwd": "${workspaceFolder}"
            }
        ]
    }
    
    with open(os.path.join(vscode_dir, "launch.json"), "w") as f:
        json.dump(launch_config, f, indent=2)
    
    # Create tasks.json for common tasks
    tasks_config = {
        "version": "2.0.0",
        "tasks": [
            {
                "label": "Install Dependencies",
                "type": "shell",
                "command": "pip install -r requirements.txt",
                "group": "build",
                "presentation": {
                    "echo": True,
                    "reveal": "always",
                    "focus": False,
                    "panel": "shared"
                }
            },
            {
                "label": "Run Demo",
                "type": "shell",
                "command": "python demo.py",
                "group": "build",
                "presentation": {
                    "echo": True,
                    "reveal": "always",
                    "focus": False,
                    "panel": "shared"
                }
            },
            {
                "label": "Create Ground Truth",
                "type": "shell", 
                "command": "python run_comprehensive_evaluation.py --create-gt",
                "group": "build"
            },
            {
                "label": "Run Evaluation",
                "type": "shell",
                "command": "python run_comprehensive_evaluation.py --video-dir working/test_anomaly",
                "group": "test"
            }
        ]
    }
    
    with open(os.path.join(vscode_dir, "tasks.json"), "w") as f:
        json.dump(tasks_config, f, indent=2)
    
    # Create workspace file
    workspace_config = {
        "folders": [
            {
                "path": "."
            }
        ],
        "settings": settings,
        "extensions": {
            "recommendations": [
                "ms-python.python",
                "ms-python.pylint", 
                "ms-toolsai.jupyter",
                "redhat.vscode-yaml",
                "ms-vscode.cmake-tools",
                "ms-python.black-formatter"
            ]
        }
    }
    
    workspace_file = f"{target_dir}.code-workspace"
    with open(workspace_file, "w") as f:
        json.dump(workspace_config, f, indent=2)
    
    # Create README for VS Code setup
    vscode_readme = f"""# CCTV Anomaly Detection - VS Code Setup

## Quick Start

1. **Open in VS Code**: Double-click `{target_dir}.code-workspace`
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
{target_dir}/
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
python anomaly_detection_tracker.py \\
    --input working/test_anomaly/Shoplifting045_x264.mp4 \\
    --output result.mp4
```

### 2. Run Evaluation
```python
python run_comprehensive_evaluation.py \\
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
"""
    
    with open(os.path.join(target_dir, "VSCODE_README.md"), "w") as f:
        f.write(vscode_readme)
    
    print(f"\n✅ VS Code project created successfully!")
    print(f"📁 Location: {os.path.abspath(target_dir)}")
    print(f"🚀 To open: Double-click '{workspace_file}'")
    print(f"📖 Setup guide: {target_dir}/VSCODE_README.md")

if __name__ == "__main__":
    create_vscode_project()