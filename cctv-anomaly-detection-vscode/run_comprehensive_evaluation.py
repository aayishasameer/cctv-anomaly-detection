#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for CCTV Anomaly Detection System
Runs complete evaluation with metrics, performance analysis, and ReID testing
"""

import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from enhanced_anomaly_tracker import EnhancedAnomalyTracker
from create_ground_truth import create_sample_ground_truth
from evaluation_metrics import AnomalyEvaluator

def run_single_video_evaluation(video_path: str, ground_truth_file: str, 
                               enable_reid: bool = False) -> dict:
    """Run evaluation on a single video"""
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    
    # Initialize tracker
    tracker = EnhancedAnomalyTracker(enable_reid=enable_reid)
    
    # Process video
    output_path = f"evaluation_output_{os.path.basename(video_path)}"
    results = tracker.process_video_with_evaluation(
        video_path=video_path,
        output_path=output_path,
        display=False,  # No display for batch evaluation
        ground_truth_file=ground_truth_file,
        camera_id="cam1"
    )
    
    return results

def run_batch_evaluation(video_dir: str, ground_truth_file: str, 
                        enable_reid: bool = False) -> dict:
    """Run evaluation on multiple videos"""
    
    video_files = list(Path(video_dir).glob("*.mp4"))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return {}
    
    print(f"Found {len(video_files)} videos for evaluation")
    
    all_results = {}
    summary_stats = {
        'total_videos': len(video_files),
        'total_frames': 0,
        'total_detections': 0,
        'total_anomalies': 0,
        'average_fps': 0,
        'accuracy_scores': [],
        'precision_scores': [],
        'recall_scores': [],
        'f1_scores': []
    }
    
    for video_file in video_files:
        try:
            results = run_single_video_evaluation(
                str(video_file), ground_truth_file, enable_reid
            )
            
            video_name = video_file.name
            all_results[video_name] = results
            
            # Accumulate summary statistics
            basic_stats = results.get('basic_statistics', {})
            summary_stats['total_frames'] += basic_stats.get('total_frames_processed', 0)
            summary_stats['total_detections'] += basic_stats.get('total_detections', 0)
            summary_stats['total_anomalies'] += basic_stats.get('total_anomalies', 0)
            summary_stats['average_fps'] += basic_stats.get('processing_fps', 0)
            
            # Collect evaluation metrics if available
            anomaly_metrics = results.get('anomaly_detection_metrics', {})
            if anomaly_metrics:
                summary_stats['accuracy_scores'].append(anomaly_metrics.get('accuracy', 0))
                summary_stats['precision_scores'].append(anomaly_metrics.get('precision', 0))
                summary_stats['recall_scores'].append(anomaly_metrics.get('recall', 0))
                summary_stats['f1_scores'].append(anomaly_metrics.get('f1_score', 0))
            
            print(f"✓ Completed: {video_name}")
            
        except Exception as e:
            print(f"❌ Error processing {video_file.name}: {e}")
            continue
    
    # Calculate averages
    if len(video_files) > 0:
        summary_stats['average_fps'] /= len(video_files)
        summary_stats['overall_anomaly_rate'] = (
            summary_stats['total_anomalies'] / summary_stats['total_detections'] 
            if summary_stats['total_detections'] > 0 else 0
        )
    
    # Calculate metric averages
    for metric in ['accuracy_scores', 'precision_scores', 'recall_scores', 'f1_scores']:
        scores = summary_stats[metric]
        if scores:
            summary_stats[f'average_{metric[:-7]}'] = sum(scores) / len(scores)
            summary_stats[f'std_{metric[:-7]}'] = pd.Series(scores).std()
        else:
            summary_stats[f'average_{metric[:-7]}'] = 0
            summary_stats[f'std_{metric[:-7]}'] = 0
    
    all_results['summary_statistics'] = summary_stats
    
    return all_results

def generate_evaluation_report(results: dict, output_dir: str = "evaluation_results"):
    """Generate comprehensive evaluation report"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Raw results saved to: {results_file}")
    
    # Generate summary report
    summary_stats = results.get('summary_statistics', {})
    
    report_lines = [
        "CCTV Anomaly Detection System - Evaluation Report",
        "=" * 60,
        "",
        "SUMMARY STATISTICS:",
        f"  Total Videos Processed: {summary_stats.get('total_videos', 0)}",
        f"  Total Frames Processed: {summary_stats.get('total_frames', 0):,}",
        f"  Total Detections: {summary_stats.get('total_detections', 0):,}",
        f"  Total Anomalies: {summary_stats.get('total_anomalies', 0):,}",
        f"  Overall Anomaly Rate: {summary_stats.get('overall_anomaly_rate', 0):.3f}",
        f"  Average Processing FPS: {summary_stats.get('average_fps', 0):.2f}",
        "",
        "DETECTION PERFORMANCE:",
        f"  Average Accuracy: {summary_stats.get('average_accuracy', 0):.3f} ± {summary_stats.get('std_accuracy', 0):.3f}",
        f"  Average Precision: {summary_stats.get('average_precision', 0):.3f} ± {summary_stats.get('std_precision', 0):.3f}",
        f"  Average Recall: {summary_stats.get('average_recall', 0):.3f} ± {summary_stats.get('std_recall', 0):.3f}",
        f"  Average F1-Score: {summary_stats.get('average_f1', 0):.3f} ± {summary_stats.get('std_f1', 0):.3f}",
        "",
        "PER-VIDEO RESULTS:",
    ]
    
    # Add per-video results
    for video_name, video_results in results.items():
        if video_name == 'summary_statistics':
            continue
        
        basic_stats = video_results.get('basic_statistics', {})
        anomaly_metrics = video_results.get('anomaly_detection_metrics', {})
        
        report_lines.extend([
            f"  {video_name}:",
            f"    Frames: {basic_stats.get('total_frames_processed', 0):,}",
            f"    Detections: {basic_stats.get('total_detections', 0):,}",
            f"    Anomalies: {basic_stats.get('total_anomalies', 0):,}",
            f"    FPS: {basic_stats.get('processing_fps', 0):.2f}",
        ])
        
        if anomaly_metrics:
            report_lines.extend([
                f"    Accuracy: {anomaly_metrics.get('accuracy', 0):.3f}",
                f"    Precision: {anomaly_metrics.get('precision', 0):.3f}",
                f"    Recall: {anomaly_metrics.get('recall', 0):.3f}",
                f"    F1-Score: {anomaly_metrics.get('f1_score', 0):.3f}",
            ])
        
        report_lines.append("")
    
    # Save report
    report_file = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Evaluation report saved to: {report_file}")
    
    # Generate plots if matplotlib available
    try:
        generate_evaluation_plots(results, output_dir)
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for line in report_lines[4:16]:  # Print key statistics
        print(line)

def generate_evaluation_plots(results: dict, output_dir: str):
    """Generate evaluation plots"""
    
    # Collect data for plotting
    video_names = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    fps_scores = []
    
    for video_name, video_results in results.items():
        if video_name == 'summary_statistics':
            continue
        
        video_names.append(video_name.replace('_x264.mp4', ''))
        
        anomaly_metrics = video_results.get('anomaly_detection_metrics', {})
        basic_stats = video_results.get('basic_statistics', {})
        
        accuracy_scores.append(anomaly_metrics.get('accuracy', 0))
        precision_scores.append(anomaly_metrics.get('precision', 0))
        recall_scores.append(anomaly_metrics.get('recall', 0))
        f1_scores.append(anomaly_metrics.get('f1_score', 0))
        fps_scores.append(basic_stats.get('processing_fps', 0))
    
    if not video_names:
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CCTV Anomaly Detection - Evaluation Results', fontsize=16)
    
    # Detection metrics
    x_pos = range(len(video_names))
    width = 0.2
    
    axes[0, 0].bar([x - 1.5*width for x in x_pos], accuracy_scores, width, label='Accuracy', alpha=0.8)
    axes[0, 0].bar([x - 0.5*width for x in x_pos], precision_scores, width, label='Precision', alpha=0.8)
    axes[0, 0].bar([x + 0.5*width for x in x_pos], recall_scores, width, label='Recall', alpha=0.8)
    axes[0, 0].bar([x + 1.5*width for x in x_pos], f1_scores, width, label='F1-Score', alpha=0.8)
    
    axes[0, 0].set_xlabel('Videos')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Detection Performance Metrics')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(video_names, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Processing speed
    axes[0, 1].bar(video_names, fps_scores, color='skyblue', alpha=0.8)
    axes[0, 1].set_xlabel('Videos')
    axes[0, 1].set_ylabel('FPS')
    axes[0, 1].set_title('Processing Speed')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Metric distribution
    metrics_data = [accuracy_scores, precision_scores, recall_scores, f1_scores]
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    axes[1, 0].boxplot(metrics_data, labels=metric_labels)
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Metric Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    summary_stats = results.get('summary_statistics', {})
    summary_metrics = [
        summary_stats.get('average_accuracy', 0),
        summary_stats.get('average_precision', 0),
        summary_stats.get('average_recall', 0),
        summary_stats.get('average_f1', 0)
    ]
    
    axes[1, 1].bar(metric_labels, summary_metrics, color='lightcoral', alpha=0.8)
    axes[1, 1].set_ylabel('Average Score')
    axes[1, 1].set_title('Overall Performance')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, "evaluation_plots.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation plots saved to: {plot_file}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of anomaly detection system')
    parser.add_argument('--video-dir', '-d', default='working/test_anomaly',
                       help='Directory containing test videos')
    parser.add_argument('--single-video', '-v', help='Evaluate single video')
    parser.add_argument('--ground-truth', '-gt', default='sample_ground_truth.json',
                       help='Ground truth JSON file')
    parser.add_argument('--create-gt', action='store_true',
                       help='Create sample ground truth file')
    parser.add_argument('--enable-reid', action='store_true',
                       help='Enable ReID system evaluation')
    parser.add_argument('--output-dir', '-o', default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create sample ground truth if requested
    if args.create_gt:
        create_sample_ground_truth()
        print("Sample ground truth created. You can now run evaluation.")
        return
    
    # Check if ground truth exists
    if not os.path.exists(args.ground_truth):
        print(f"Ground truth file not found: {args.ground_truth}")
        print("Create it using: python run_comprehensive_evaluation.py --create-gt")
        return
    
    # Run evaluation
    if args.single_video:
        if not os.path.exists(args.single_video):
            print(f"Video file not found: {args.single_video}")
            return
        
        results = {
            os.path.basename(args.single_video): run_single_video_evaluation(
                args.single_video, args.ground_truth, args.enable_reid
            )
        }
    else:
        if not os.path.exists(args.video_dir):
            print(f"Video directory not found: {args.video_dir}")
            return
        
        results = run_batch_evaluation(args.video_dir, args.ground_truth, args.enable_reid)
    
    # Generate report
    generate_evaluation_report(results, args.output_dir)
    
    print(f"\n✅ Evaluation completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()