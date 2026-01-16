"""
Demo script to show analysis output format
Simulates what the system would output for a video analysis
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from models.feature_engineering import FeatureEngineer
from models.rule_based_evaluator import RuleBasedEvaluator
from models.ml_model import FormClassifier
import numpy as np


def demo_analysis():
    """Demonstrate the analysis output"""
    
    print("="*70)
    print(" " * 15 + "SPORTS FORM ANALYSIS SYSTEM")
    print(" " * 20 + "DEMO PREVIEW")
    print("="*70)
    print()
    
    # Initialize components
    print("🔧 Initializing components...")
    feature_engineer = FeatureEngineer()
    rule_evaluator = RuleBasedEvaluator()
    ml_classifier = FormClassifier()
    ml_classifier.load('models/form_classifier.pkl')
    print("  ✓ All components loaded\n")
    
    # Simulate analysis of a sequence
    print("📹 Simulating video analysis...")
    print("  - Extracting pose keypoints from 90 frames...")
    print("  - Processing biomechanical features...")
    print()
    
    # Generate sample features (simulating correct form)
    print("📊 Sample Analysis Results:")
    print("-"*70)
    
    # Create sample features for demonstration
    sample_features = {
        'left_elbow_angle': 145.5,
        'right_elbow_angle': 148.2,
        'left_knee_angle': 95.3,
        'right_knee_angle': 98.7,
        'left_hip_angle': 135.2,
        'right_hip_angle': 138.5,
        'spine_inclination': 12.3,
        'head_foot_alignment': 0.08,
        'movement_smoothness': 0.75
    }
    
    # Rule-based evaluation
    rule_verdict, rule_issues, rule_confidence = rule_evaluator.evaluate_frame(sample_features)
    rule_status = "✅ CORRECT" if rule_verdict else "❌ INCORRECT"
    
    print(f"\n🔬 Rule-Based Evaluation:")
    print(f"   Status: {rule_status}")
    print(f"   Confidence: {rule_confidence:.2%}")
    
    if rule_issues:
        print(f"   Issues Found: {len(rule_issues)}")
        for issue in rule_issues[:3]:
            print(f"     • {issue}")
    else:
        print("   ✓ No issues detected - Form is correct!")
    
    # ML-based evaluation (simulate)
    # For demo, we'll use aggregated features
    keypoints_sequence = [np.random.rand(33, 3) for _ in range(30)]
    aggregated_features = feature_engineer.extract_all_features(keypoints_sequence)
    ml_verdict, ml_confidence = ml_classifier.predict_single(aggregated_features)
    ml_status = "✅ CORRECT" if ml_verdict else "❌ INCORRECT"
    
    print(f"\n🤖 ML-Based Evaluation:")
    print(f"   Status: {ml_status}")
    print(f"   Confidence: {ml_confidence:.2%}")
    print(f"   Model: Random Forest (100 trees)")
    
    # Joint feedback
    print(f"\n🦴 Joint-Level Feedback:")
    joint_feedback = rule_evaluator.get_joint_feedback(sample_features)
    for joint, feedback in list(joint_feedback.items())[:6]:
        status = "✅" if any(word in feedback for word in ["Good", "Normal", "Stable", "Aligned"]) else "⚠️"
        print(f"   {status} {joint.replace('_', ' ').title()}: {feedback}")
    
    # Feature statistics
    print(f"\n📈 Biometric Feature Statistics:")
    print(f"   Left Elbow Angle:  {sample_features['left_elbow_angle']:.1f}°")
    print(f"   Right Elbow Angle: {sample_features['right_elbow_angle']:.1f}°")
    print(f"   Left Knee Angle:   {sample_features['left_knee_angle']:.1f}°")
    print(f"   Right Knee Angle:  {sample_features['right_knee_angle']:.1f}°")
    print(f"   Spine Inclination: {sample_features['spine_inclination']:.1f}°")
    print(f"   Movement Smoothness: {sample_features['movement_smoothness']:.2f}")
    
    print("\n" + "="*70)
    print("📁 Output Files Generated:")
    print("   • outputs/annotated_videos/annotated_output.mp4")
    print("   • outputs/reports/analysis_report.txt")
    print("   • outputs/reports/feature_timeline.png")
    print("="*70)
    
    print("\n🎬 To see the full interface:")
    print("   Run: streamlit run app/app.py")
    print("   Then upload a video through the web interface!")
    print()


if __name__ == "__main__":
    demo_analysis()

