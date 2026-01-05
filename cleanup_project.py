#!/usr/bin/env python3
"""
Project Cleanup Script
Removes redundant and outdated files to streamline the CCTV system
"""

import os
import shutil
from pathlib import Path

# Files to delete (redundant/outdated)
FILES_TO_DELETE = [
    # Redundant detection files
    'enhanced_anomaly_tracker.py',
    'demo_enhanced_system.py', 
    'demo.py',
    'batch_anomaly_detection.py',
    'fix_tracking_issues.py',
    
    # Debugging/temporary scripts
    'check_threshold.py',
    'check_training_data.py',
    'minimal_retrain_vae.py',
    'quick_retrain_enhanced_vae.py',
    'retrain_enhanced_vae.py',
    'fast_full_processing.py',
    'quick_output_generator.py',
    'run_mot_tracking.py',
    'multi_camera_reid.py',
    'run_comprehensive_evaluation.py',
    'test_improvements.py',
    'setup_vscode_project.py',
    
    # Outdated documentation
    'PHASE1_IMPROVEMENTS_SUMMARY.md',
    'TRACKING_ISSUES_ANALYSIS.md',
    'EVALUATION_GUIDE.md',
    'QUICK_EVALUATION_REFERENCE.md',
    
    # Old output/log files
    'training_log.txt',
    'test_020_metrics.json',
    'sample_ground_truth.json',
    
    # IDE/workspace files
    'cctv-anomaly-detection-vscode.code-workspace',
    
    # PDF files (keep if needed for reference)
    # '2024VAD.pdf',  # Uncomment to delete
]

# Output video files to delete (can be regenerated)
VIDEO_FILES_TO_DELETE = [
    'ENHANCED_ANOMALY_DETECTION_OUTPUT.mp4',
    'enhanced_shoplifting_020_full_output.mp4',
    'enhanced_shoplifting_020_output.mp4',
    'improved_output_020.mp4',
]

# Directories to delete (if empty or redundant)
DIRECTORIES_TO_DELETE = [
    'cctv-anomaly-detection-vscode',
    'evaluation_020_results',
    'evaluation_results_045', 
    'final_results',
    'test_results_batch',
    '__pycache__',
]

# Essential files that should NEVER be deleted
ESSENTIAL_FILES = [
    'vae_anomaly_detector.py',
    'improved_anomaly_tracker.py',
    'stealing_detection_system.py',
    'adaptive_zone_learning.py',
    'train_vae_model.py',
    'learn_and_test_adaptive_system.py',
    'demo_stealing_detection.py',
    'test_fixed_system.py',
    'test_stealing_detection.py',
    'setup_stealing_detection.py',
    'evaluation_metrics.py',
    'requirements.txt',
    'botsort.yaml',
    'botsort_improved.yaml',
    'STEALING_DETECTION_GUIDE.md',
    'CCTV_Anomaly_Detection_Implementation_Presentation.md',
    'README.md',
    'yolov8n.pt',
    'quick_adaptive_test.py',
    'ESSENTIAL_FILES_FOR_IMPLEMENTATION.md',
    'create_ground_truth.py',  # Useful for evaluation
]

def safe_delete_file(file_path: str) -> bool:
    """Safely delete a file with error handling"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✅ Deleted: {file_path}")
            return True
        else:
            print(f"⚠️  Not found: {file_path}")
            return False
    except Exception as e:
        print(f"❌ Error deleting {file_path}: {e}")
        return False

def safe_delete_directory(dir_path: str) -> bool:
    """Safely delete a directory with error handling"""
    try:
        if os.path.exists(dir_path):
            if os.path.isdir(dir_path):
                # Check if directory is empty or contains only cache files
                contents = os.listdir(dir_path)
                if not contents or all(f.startswith('.') or f == '__pycache__' for f in contents):
                    shutil.rmtree(dir_path)
                    print(f"✅ Deleted directory: {dir_path}")
                    return True
                else:
                    print(f"⚠️  Directory not empty, skipping: {dir_path}")
                    return False
            else:
                print(f"⚠️  Not a directory: {dir_path}")
                return False
        else:
            print(f"⚠️  Directory not found: {dir_path}")
            return False
    except Exception as e:
        print(f"❌ Error deleting directory {dir_path}: {e}")
        return False

def list_current_files():
    """List all current files in the project"""
    print("📁 Current project files:")
    all_files = []
    for root, dirs, files in os.walk("."):
        # Skip hidden directories and git
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for file in files:
            if not file.startswith('.'):
                rel_path = os.path.relpath(os.path.join(root, file))
                all_files.append(rel_path)
    
    all_files.sort()
    for file in all_files:
        if file in ESSENTIAL_FILES:
            print(f"  ✅ {file} (ESSENTIAL)")
        elif file in FILES_TO_DELETE:
            print(f"  ❌ {file} (TO DELETE)")
        elif file in VIDEO_FILES_TO_DELETE:
            print(f"  🎬 {file} (VIDEO - CAN DELETE)")
        else:
            print(f"  📄 {file}")
    
    return all_files

def cleanup_project(dry_run: bool = True):
    """Clean up the project by removing redundant files"""
    
    print("🧹 CCTV Project Cleanup")
    print("=" * 50)
    
    if dry_run:
        print("🔍 DRY RUN MODE - No files will be deleted")
        print("Run with --execute to actually delete files")
    else:
        print("⚠️  EXECUTION MODE - Files will be permanently deleted!")
    
    print()
    
    # List current files
    current_files = list_current_files()
    
    print(f"\n📊 CLEANUP SUMMARY:")
    print(f"  📁 Total files found: {len(current_files)}")
    print(f"  ✅ Essential files: {len([f for f in current_files if f in ESSENTIAL_FILES])}")
    print(f"  ❌ Files to delete: {len([f for f in current_files if f in FILES_TO_DELETE])}")
    print(f"  🎬 Video files to delete: {len([f for f in current_files if f in VIDEO_FILES_TO_DELETE])}")
    
    if dry_run:
        print(f"\n🔍 FILES THAT WOULD BE DELETED:")
        for file in FILES_TO_DELETE:
            if os.path.exists(file):
                print(f"  ❌ {file}")
        
        print(f"\n🎬 VIDEO FILES THAT WOULD BE DELETED:")
        for file in VIDEO_FILES_TO_DELETE:
            if os.path.exists(file):
                print(f"  🎬 {file}")
        
        print(f"\n📁 DIRECTORIES THAT WOULD BE DELETED:")
        for directory in DIRECTORIES_TO_DELETE:
            if os.path.exists(directory):
                print(f"  📁 {directory}")
        
        return
    
    # Actually delete files
    print(f"\n🗑️  DELETING FILES:")
    deleted_files = 0
    
    # Delete redundant files
    for file in FILES_TO_DELETE:
        if safe_delete_file(file):
            deleted_files += 1
    
    # Delete video files (ask for confirmation)
    if VIDEO_FILES_TO_DELETE:
        print(f"\n🎬 Delete output video files? (y/n): ", end="")
        response = input().lower().strip()
        if response in ['y', 'yes']:
            for file in VIDEO_FILES_TO_DELETE:
                if safe_delete_file(file):
                    deleted_files += 1
        else:
            print("⚠️  Skipping video file deletion")
    
    # Delete directories
    print(f"\n📁 DELETING EMPTY DIRECTORIES:")
    deleted_dirs = 0
    for directory in DIRECTORIES_TO_DELETE:
        if safe_delete_directory(directory):
            deleted_dirs += 1
    
    print(f"\n🎯 CLEANUP COMPLETE!")
    print(f"  🗑️  Files deleted: {deleted_files}")
    print(f"  📁 Directories deleted: {deleted_dirs}")
    
    # Show remaining essential files
    print(f"\n✅ ESSENTIAL FILES REMAINING:")
    for file in ESSENTIAL_FILES:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ⚠️  {file} (missing - may need to be created)")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up CCTV project files')
    parser.add_argument('--execute', action='store_true', 
                       help='Actually delete files (default is dry run)')
    parser.add_argument('--list-only', action='store_true',
                       help='Only list current files')
    
    args = parser.parse_args()
    
    if args.list_only:
        list_current_files()
        return
    
    # Confirm execution mode
    if args.execute:
        print("⚠️  WARNING: This will permanently delete files!")
        print("Are you sure you want to continue? (type 'yes' to confirm): ", end="")
        confirmation = input().strip()
        if confirmation.lower() != 'yes':
            print("❌ Cleanup cancelled")
            return
    
    cleanup_project(dry_run=not args.execute)
    
    if not args.execute:
        print(f"\n🔄 To actually delete files, run:")
        print(f"python cleanup_project.py --execute")

if __name__ == "__main__":
    main()