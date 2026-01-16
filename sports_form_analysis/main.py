"""
Main Entry Point for Sports Form Analysis System
Can be used for command-line analysis or to launch the Streamlit app
"""

import argparse
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from models.pose_extractor import PoseExtractor
from models.feature_engineering import FeatureEngineer
from models.rule_based_evaluator import RuleBasedEvaluator
from models.ml_model import FormClassifier
from utils.video_utils import get_video_info, save_video
from utils.visualization import draw_keypoints_on_frame, create_analysis_report, plot_feature_timeline


def analyze_video_cli(video_path: str, output_dir: str = "outputs"):
    """
    Analyze video from command line
    
    Args:
        video_path: Path to input video
        output_dir: Output directory for results
    """
    print("="*60)
    print("SPORTS FORM ANALYSIS SYSTEM")
    print("="*60)
    print(f"\nAnalyzing video: {video_path}")
    
    # Initialize components
    print("\n[1/5] Initializing components...")
    pose_extractor = PoseExtractor()
    feature_engineer = FeatureEngineer()
    rule_evaluator = RuleBasedEvaluator()
    
    # Load ML model
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'form_classifier.pkl')
    if os.path.exists(model_path):
        ml_classifier = FormClassifier()
        ml_classifier.load(model_path)
        print("  ✓ ML model loaded")
    else:
        print("  ⚠ ML model not found. Run train_model.py first.")
        ml_classifier = None
    
    # Extract keypoints
    print("\n[2/5] Extracting pose keypoints...")
    keypoints_sequence, frames_sequence = pose_extractor.extract_from_video(video_path)
    print(f"  ✓ Extracted {len(keypoints_sequence)} frames")
    
    if not keypoints_sequence:
        print("  ✗ No pose detected in video")
        return
    
    # Extract features
    print("\n[3/5] Extracting biomechanical features...")
    features_sequence = []
    for keypoints in keypoints_sequence:
        features = feature_engineer.extract_frame_features(keypoints)
        features_sequence.append(features)
    print("  ✓ Features extracted")
    
    # Extract aggregated features for ML
    aggregated_features = feature_engineer.extract_all_features(keypoints_sequence)
    
    # Rule-based evaluation
    print("\n[4/5] Performing rule-based evaluation...")
    rule_verdict, rule_issues, rule_confidence, problematic_frames = rule_evaluator.evaluate_sequence(
        features_sequence
    )
    rule_status = "CORRECT" if rule_verdict else "INCORRECT"
    print(f"  ✓ Rule-based verdict: {rule_status} (confidence: {rule_confidence:.2%})")
    
    # ML-based evaluation
    if ml_classifier:
        print("\n[5/5] Performing ML-based evaluation...")
        ml_verdict, ml_confidence = ml_classifier.predict_single(aggregated_features)
        ml_status = "CORRECT" if ml_verdict else "INCORRECT"
        print(f"  ✓ ML-based verdict: {ml_status} (confidence: {ml_confidence:.2%})")
    else:
        ml_verdict = None
        ml_confidence = 0.0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    annotated_dir = os.path.join(output_dir, 'annotated_videos')
    report_dir = os.path.join(output_dir, 'reports')
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    # Create annotated video
    print("\nCreating annotated video...")
    annotated_frames = []
    problematic_joint_indices = []
    
    if problematic_frames:
        for category, frame_indices in problematic_frames.items():
            if category == 'elbow':
                problematic_joint_indices.extend([13, 14])
            elif category == 'knee':
                problematic_joint_indices.extend([25, 26])
            elif category == 'hip':
                problematic_joint_indices.extend([23, 24])
            elif category == 'spine':
                problematic_joint_indices.extend([11, 12, 23, 24])
    
    for idx, (frame, keypoints) in enumerate(zip(frames_sequence, keypoints_sequence)):
        is_problematic = any(idx in frames for frames in problematic_frames.values())
        annotated_frame = draw_keypoints_on_frame(
            frame, keypoints,
            problematic_joints=problematic_joint_indices if is_problematic else None
        )
        annotated_frames.append(annotated_frame)
    
    # Get video info
    width, height, fps, _ = get_video_info(video_path)
    
    # Save annotated video
    output_video_path = os.path.join(annotated_dir, 'annotated_output.mp4')
    save_video(annotated_frames, output_video_path, fps)
    print(f"  ✓ Saved to: {output_video_path}")
    
    # Create report
    report_path = os.path.join(report_dir, 'analysis_report.txt')
    create_analysis_report(
        features_sequence, rule_verdict, rule_confidence, rule_issues,
        ml_verdict if ml_verdict is not None else False, ml_confidence,
        problematic_frames, report_path
    )
    
    # Create feature plot
    plot_path = os.path.join(report_dir, 'feature_timeline.png')
    plot_feature_timeline(features_sequence, plot_path)
    print(f"  ✓ Saved to: {plot_path}")
    
    # Cleanup
    pose_extractor.close()
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Rule-Based: {rule_status} ({rule_confidence:.2%})")
    if ml_classifier:
        print(f"ML-Based:   {ml_status} ({ml_confidence:.2%})")
    if rule_issues:
        print(f"\nIssues Found: {len(rule_issues)}")
        for issue in rule_issues[:5]:  # Show first 5
            print(f"  • {issue}")
    print(f"\nResults saved to: {output_dir}")
    print("="*60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Sports Form Analysis System - CLI Interface"
    )
    parser.add_argument(
        '--video',
        type=str,
        help='Path to input video file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory (default: outputs)'
    )
    parser.add_argument(
        '--app',
        action='store_true',
        help='Launch Streamlit app instead'
    )
    
    args = parser.parse_args()
    
    if args.app:
        # Launch Streamlit app
        import subprocess
        app_path = os.path.join(os.path.dirname(__file__), 'app', 'app.py')
        subprocess.run(['streamlit', 'run', app_path])
    elif args.video:
        # Analyze video
        analyze_video_cli(args.video, args.output)
    else:
        parser.print_help()
        print("\nTo launch the Streamlit app, use: python main.py --app")
        print("Or: streamlit run app/app.py")


if __name__ == "__main__":
    main()

