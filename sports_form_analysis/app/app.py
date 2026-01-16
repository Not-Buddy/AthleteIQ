"""
Streamlit Application for Sports Form Analysis
Main interface for video upload and analysis
"""

import streamlit as st
import cv2
import numpy as np
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.pose_extractor import PoseExtractor
from models.feature_engineering import FeatureEngineer
from models.rule_based_evaluator import RuleBasedEvaluator
from models.ml_model import FormClassifier
from utils.video_utils import get_video_info, save_video
from utils.visualization import draw_keypoints_on_frame, create_analysis_report, plot_feature_timeline


# Page configuration
st.set_page_config(
    page_title="Sports Form Analysis",
    page_icon="🏃",
    layout="wide"
)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'video_path' not in st.session_state:
    st.session_state.video_path = None


def load_model():
    """Load trained ML model"""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'form_classifier.pkl')
    
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please run train_model.py first.")
        return None
    
    classifier = FormClassifier()
    try:
        classifier.load(model_path)
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def analyze_video(video_path: str):
    """Perform complete analysis on video"""
    
    # Initialize components
    pose_extractor = PoseExtractor()
    feature_engineer = FeatureEngineer()
    rule_evaluator = RuleBasedEvaluator()
    ml_classifier = load_model()
    
    if ml_classifier is None:
        return None
    
    # Extract keypoints
    st.info("Extracting pose keypoints from video...")
    keypoints_sequence, frames_sequence = pose_extractor.extract_from_video(video_path)
    
    if not keypoints_sequence:
        st.error("No pose detected in video. Please ensure the video contains a person.")
        return None
    
    # Extract features
    st.info("Extracting biomechanical features...")
    features_sequence = []
    for keypoints in keypoints_sequence:
        features = feature_engineer.extract_frame_features(keypoints)
        features_sequence.append(features)
    
    # Extract aggregated features for ML
    aggregated_features = feature_engineer.extract_all_features(keypoints_sequence)
    
    # Rule-based evaluation
    st.info("Performing rule-based evaluation...")
    rule_verdict, rule_issues, rule_confidence, problematic_frames = rule_evaluator.evaluate_sequence(
        features_sequence
    )
    
    # ML-based evaluation
    st.info("Performing ML-based evaluation...")
    ml_verdict, ml_confidence = ml_classifier.predict_single(aggregated_features)
    
    # Create annotated video
    st.info("Creating annotated video...")
    annotated_frames = []
    problematic_joint_indices = []
    
    # Collect problematic joint indices
    if problematic_frames:
        for category, frame_indices in problematic_frames.items():
            if category == 'elbow':
                problematic_joint_indices.extend([13, 14])  # Elbow indices
            elif category == 'knee':
                problematic_joint_indices.extend([25, 26])  # Knee indices
            elif category == 'hip':
                problematic_joint_indices.extend([23, 24])  # Hip indices
            elif category == 'spine':
                problematic_joint_indices.extend([11, 12, 23, 24])  # Shoulder/hip indices
    
    for idx, (frame, keypoints) in enumerate(zip(frames_sequence, keypoints_sequence)):
        annotated_frame = draw_keypoints_on_frame(
            frame, keypoints, 
            problematic_joints=problematic_joint_indices if idx in problematic_frames.get('elbow', []) + 
            problematic_frames.get('knee', []) + problematic_frames.get('hip', []) else None
        )
        annotated_frames.append(annotated_frame)
    
    # Get video info for saving
    width, height, fps, _ = get_video_info(video_path)
    
    # Save annotated video
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'annotated_videos')
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, 'annotated_output.mp4')
    save_video(annotated_frames, output_video_path, fps)
    
    # Create report
    report_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'reports')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'analysis_report.txt')
    create_analysis_report(
        features_sequence, rule_verdict, rule_confidence, rule_issues,
        ml_verdict, ml_confidence, problematic_frames, report_path
    )
    
    # Create feature plot
    plot_path = os.path.join(report_dir, 'feature_timeline.png')
    plot_feature_timeline(features_sequence, plot_path)
    
    # Cleanup
    pose_extractor.close()
    
    return {
        'rule_verdict': rule_verdict,
        'rule_confidence': rule_confidence,
        'rule_issues': rule_issues,
        'ml_verdict': ml_verdict,
        'ml_confidence': ml_confidence,
        'problematic_frames': problematic_frames,
        'features_sequence': features_sequence,
        'annotated_video_path': output_video_path,
        'report_path': report_path,
        'plot_path': plot_path,
        'keypoints_sequence': keypoints_sequence,
        'frames_sequence': frames_sequence
    }


def main():
    """Main application"""
    
    st.title("🏃 AI-Based Sports Biomechanics & Form Analysis System")
    st.markdown("---")
    
    st.markdown("""
    This system analyzes sports videos to evaluate body mechanics and form using:
    - **Pose Estimation**: MediaPipe Pose (33 keypoints)
    - **Rule-Based Evaluation**: Biomechanical threshold analysis
    - **ML-Based Classification**: Random Forest classifier
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. Upload a video of a sports movement
        2. Wait for analysis to complete
        3. Review the results and feedback
        """)
        
        st.header("📊 Supported Movements")
        st.markdown("""
        - Cricket shots
        - Gym exercises
        - General athletic movements
        """)
        
        # Check if model exists
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'form_classifier.pkl')
        if os.path.exists(model_path):
            st.success("✅ ML Model loaded")
        else:
            st.warning("⚠️ ML Model not found. Run train_model.py first.")
    
    # File upload
    st.header("📹 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file containing a person performing a movement"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        st.session_state.video_path = video_path
        
        # Display video info
        try:
            width, height, fps, frame_count = get_video_info(video_path)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Width", f"{width}px")
            col2.metric("Height", f"{height}px")
            col3.metric("FPS", f"{fps:.2f}")
            col4.metric("Frames", frame_count)
        except Exception as e:
            st.error(f"Error reading video: {e}")
            return
        
        # Display original video
        st.subheader("📺 Original Video")
        st.video(video_path)
        
        # Analyze button
        if st.button("🔍 Analyze Form", type="primary", use_container_width=True):
            with st.spinner("Analyzing video... This may take a few moments."):
                results = analyze_video(video_path)
            
            if results:
                st.session_state.analysis_complete = True
                st.session_state.results = results
                st.success("Analysis complete!")
                st.rerun()
    
    # Display results
    if st.session_state.analysis_complete and 'results' in st.session_state:
        results = st.session_state.results
        
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        # Verdict cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔬 Rule-Based Evaluation")
            rule_status = "✅ CORRECT" if results['rule_verdict'] else "❌ INCORRECT"
            st.markdown(f"### {rule_status}")
            st.progress(results['rule_confidence'])
            st.caption(f"Confidence: {results['rule_confidence']:.2%}")
            
            if results['rule_issues']:
                st.markdown("**Issues Found:**")
                for issue in results['rule_issues']:
                    st.warning(f"• {issue}")
            else:
                st.success("No issues detected!")
        
        with col2:
            st.subheader("🤖 ML-Based Evaluation")
            ml_status = "✅ CORRECT" if results['ml_verdict'] else "❌ INCORRECT"
            st.markdown(f"### {ml_status}")
            st.progress(results['ml_confidence'])
            st.caption(f"Confidence: {results['ml_confidence']:.2%}")
            
            # ML explanation
            st.info("""
            The ML model uses aggregated biomechanical features 
            to classify form as correct or incorrect.
            """)
        
        # Annotated video
        st.subheader("🎬 Annotated Video with Pose Overlay")
        if os.path.exists(results['annotated_video_path']):
            st.video(results['annotated_video_path'])
        else:
            st.error("Annotated video not found")
        
        # Feature timeline
        st.subheader("📈 Feature Timeline")
        if os.path.exists(results['plot_path']):
            st.image(results['plot_path'], use_container_width=True)
        else:
            st.warning("Feature plot not available")
        
        # Joint-level feedback
        st.subheader("🦴 Joint-Level Feedback")
        
        if results['features_sequence']:
            # Use middle frame for feedback
            mid_idx = len(results['features_sequence']) // 2
            mid_features = results['features_sequence'][mid_idx]
            
            rule_evaluator = RuleBasedEvaluator()
            joint_feedback = rule_evaluator.get_joint_feedback(mid_features)
            
            feedback_cols = st.columns(3)
            feedback_items = list(joint_feedback.items())
            
            for idx, (joint, feedback) in enumerate(feedback_items):
                with feedback_cols[idx % 3]:
                    status = "✅" if "Good" in feedback or "Normal" in feedback or "Stable" in feedback or "Aligned" in feedback else "⚠️"
                    st.markdown(f"**{joint.replace('_', ' ').title()}**")
                    st.markdown(f"{status} {feedback}")
        
        # Download results
        st.markdown("---")
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists(results['report_path']):
                with open(results['report_path'], 'r') as f:
                    st.download_button(
                        "📄 Download Report",
                        f.read(),
                        file_name="analysis_report.txt",
                        mime="text/plain"
                    )
        
        with col2:
            if os.path.exists(results['annotated_video_path']):
                with open(results['annotated_video_path'], 'rb') as f:
                    st.download_button(
                        "🎬 Download Annotated Video",
                        f.read(),
                        file_name="annotated_output.mp4",
                        mime="video/mp4"
                    )


if __name__ == "__main__":
    main()

