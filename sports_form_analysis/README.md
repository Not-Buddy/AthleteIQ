# AI-Based Sports Biomechanics & Form Analysis System

A production-grade system that analyzes sports videos to evaluate body mechanics and form using pose estimation, biomechanical feature extraction, rule-based evaluation, and machine learning classification.

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [How the System Works](#how-the-system-works)
3. [Tech Stack](#tech-stack)
4. [Installation & Setup](#installation--setup)
5. [How to Run](#how-to-run)
6. [Dataset Explanation](#dataset-explanation)
7. [Model Explanation](#model-explanation)
8. [Sample Output](#sample-output)
9. [Project Structure](#project-structure)
10. [Future Improvements](#future-improvements)

---

## 🎯 Project Overview

This system analyzes videos of athletes performing movements (cricket shots, gym exercises, or general athletic movements) and provides:

- **Pose Estimation**: Extracts 33 body keypoints using MediaPipe Pose
- **Biomechanical Analysis**: Calculates joint angles, posture, balance, and movement smoothness
- **Dual Evaluation**:
  - **Rule-Based**: Deterministic biomechanical threshold analysis
  - **ML-Based**: Random Forest classifier trained on synthetic data
- **Visual Feedback**: Annotated videos with pose overlay and highlighted problematic joints
- **Detailed Reports**: Text reports and feature timeline plots

### Key Features

✅ **No Placeholders**: All code is functional and runnable  
✅ **Complete Pipeline**: End-to-end from video input to analysis output  
✅ **Explainable Results**: Clear feedback on what's correct/incorrect and why  
✅ **Dual Validation**: Both rule-based and ML-based evaluation for reliability  

---

## 🔄 How the System Works

### Step-by-Step Process

1. **Video Input**
   - User uploads a video file through Streamlit interface or CLI
   - System reads video frames

2. **Pose Extraction**
   - MediaPipe Pose processes each frame
   - Extracts 33 keypoints (shoulders, elbows, wrists, hips, knees, ankles, etc.)
   - Normalizes coordinates for camera distance

3. **Feature Engineering**
   - Calculates biomechanical features for each frame:
     - **Elbow angles** (left/right)
     - **Knee angles** (left/right)
     - **Hip angles** (left/right)
     - **Spine inclination** (deviation from vertical)
     - **Head-to-foot alignment** (balance indicator)
     - **Center of mass** approximation
   - Extracts temporal features:
     - **Movement smoothness** (inverse of velocity variance)
     - **Velocity variance** across frames

4. **Form Evaluation**

   **A. Rule-Based System:**
   - Applies biomechanical thresholds:
     - Elbow angle < 30° → Over-flexed
     - Knee angle < 90° → Unstable base
     - Hip angle < 120° → Poor posture
     - Spine inclination > 30° → Deviated
     - Head-foot alignment > 0.15 → Balance issue
   - Generates explainable feedback for each violation

   **B. ML-Based System:**
   - Uses aggregated features (mean, std, min, max across frames + temporal features)
   - Random Forest classifier predicts: Correct (1) or Incorrect (0)
   - Outputs probability score (confidence)

5. **Output Generation**
   - Annotated video with pose skeleton overlay
   - Problematic joints highlighted in red
   - Text report with detailed analysis
   - Feature timeline plots showing angles over time
   - Joint-level feedback

---

## 🛠 Tech Stack

### AI / Backend
- **Python 3.10+**: Core programming language
- **OpenCV**: Video processing and frame manipulation
- **MediaPipe Pose**: Human pose estimation (33 keypoints)
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation (for future dataset expansion)
- **Scikit-learn**: Machine learning (Random Forest classifier)
- **Matplotlib**: Visualization and plotting

### Frontend
- **Streamlit**: Web interface for video upload and results display

### Why This Stack?

- **MediaPipe Pose**: Industry-standard, real-time pose estimation with high accuracy
- **Random Forest**: Interpretable, handles non-linear relationships, works well with small datasets
- **Streamlit**: Rapid prototyping, no frontend code needed, perfect for demos
- **Scikit-learn over TensorFlow/PyTorch**: Simpler, faster training, sufficient for this binary classification task

---

## 📦 Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Steps

1. **Clone or navigate to the project directory:**
   ```bash
   cd sports_form_analysis
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the ML model (required before first use):**
   ```bash
   python models/train_model.py
   ```
   This will:
   - Generate synthetic training data (200 samples)
   - Train a Random Forest classifier
   - Save the model to `models/form_classifier.pkl`
   - Display evaluation metrics

---

## 🚀 How to Run

### Option 1: Streamlit Web Interface (Recommended)

```bash
streamlit run app/app.py
```

Or using the main entry point:

```bash
python main.py --app
```

Then:
1. Open your browser to the URL shown (usually `http://localhost:8501`)
2. Upload a video file
3. Click "Analyze Form"
4. View results

### Option 2: Command Line Interface

```bash
python main.py --video path/to/your/video.mp4 --output outputs
```

This will:
- Analyze the video
- Save annotated video to `outputs/annotated_videos/`
- Save report to `outputs/reports/`
- Display summary in terminal

---

## 📊 Dataset Explanation

### Training Data

The system uses **synthetic training data** generated programmatically. This approach:

- **Eliminates dependency** on external datasets
- **Ensures reproducibility** (same data every time)
- **Allows control** over correct vs incorrect form examples

### Data Generation

The `train_model.py` script generates:

1. **Correct Form Samples**:
   - Proper joint angles (elbows extended, knees stable >90°, hips aligned)
   - Good spine alignment
   - Balanced head-foot alignment
   - Smooth movement patterns

2. **Incorrect Form Samples**:
   - Over-flexed elbows (<30°)
   - Unstable knee angles (<90°)
   - Poor hip posture (<120°)
   - Deviated spine (>30°)
   - Misaligned balance
   - Jerky movements (higher velocity variance)

### Data Format

- **Features**: 13 frame-level features × 4 aggregations (mean, std, min, max) + 2 temporal features = **54 features**
- **Labels**: Binary (0 = Incorrect, 1 = Correct)
- **Samples**: 100 sequences (50 correct, 50 incorrect), 30 frames each

### Using Your Own Data

To use real video data:

1. Place videos in `data/raw_videos/`
2. Label them in `data/labels.csv` (format: `video_path,label,notes`)
3. Modify `train_model.py` to load and process your videos
4. Retrain the model

---

## 🤖 Model Explanation

### Random Forest Classifier

**Why Random Forest?**
- **Interpretable**: Feature importance scores show which biomechanical features matter most
- **Robust**: Handles non-linear relationships between features
- **Works with small datasets**: Doesn't require thousands of samples
- **Fast training**: Trains in seconds, not hours
- **No GPU required**: Runs on CPU

### Model Architecture

- **Algorithm**: Random Forest (ensemble of decision trees)
- **Trees**: 100 decision trees
- **Max Depth**: 10 (prevents overfitting)
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2
- **Input**: 54 aggregated features per video sequence
- **Output**: Binary classification (Correct/Incorrect) + probability score

### Feature Importance

The model learns which features are most predictive:
- Knee angles (stability)
- Spine inclination (posture)
- Movement smoothness (technique quality)
- Elbow angles (form)

### Training Process

1. Generate synthetic data (100 sequences)
2. Extract features using `FeatureEngineer`
3. Split: 80% train, 20% test
4. Train Random Forest
5. Evaluate: Accuracy, Precision, Recall, F1-score
6. Save model to `models/form_classifier.pkl`

### Expected Performance

On synthetic data:
- **Accuracy**: ~85-95%
- **Precision**: ~85-95%
- **Recall**: ~85-95%

*Note: Performance on real videos depends on video quality, pose visibility, and movement type.*

---

## 📈 Sample Output

### Console Output (CLI)

```
============================================================
SPORTS FORM ANALYSIS SYSTEM
============================================================

Analyzing video: sample_video.mp4

[1/5] Initializing components...
  ✓ ML model loaded

[2/5] Extracting pose keypoints...
  ✓ Extracted 90 frames

[3/5] Extracting biomechanical features...
  ✓ Features extracted

[4/5] Performing rule-based evaluation...
  ✓ Rule-based verdict: INCORRECT (confidence: 65.00%)

[5/5] Performing ML-based evaluation...
  ✓ ML-based verdict: INCORRECT (confidence: 72.50%)

Creating annotated video...
  ✓ Saved to: outputs/annotated_videos/annotated_output.mp4
  ✓ Saved to: outputs/reports/feature_timeline.png

============================================================
ANALYSIS SUMMARY
============================================================
Rule-Based: INCORRECT (65.00%)
ML-Based:   INCORRECT (72.50%)

Issues Found: 3
  • Left knee angle too acute (< 90°) - unstable base
  • Spine deviation from vertical (35.2° > 30.0°)
  • Poor head-to-foot alignment - balance issue

Results saved to: outputs
============================================================
```

### Streamlit Interface

The web interface displays:

1. **Original Video**: Uploaded video player
2. **Verdict Cards**: 
   - Rule-Based: ✅ CORRECT / ❌ INCORRECT with confidence bar
   - ML-Based: ✅ CORRECT / ❌ INCORRECT with confidence bar
3. **Annotated Video**: Pose skeleton overlay with problematic joints in red
4. **Feature Timeline**: 4 plots showing angles over time
5. **Joint-Level Feedback**: Status for each major joint
6. **Download Options**: Report and annotated video

### Report File (`outputs/reports/analysis_report.txt`)

```
============================================================
SPORTS FORM ANALYSIS REPORT
============================================================

OVERALL VERDICT
------------------------------------------------------------
Rule-Based Evaluation: INCORRECT
  Confidence: 65.00%

ML-Based Evaluation: INCORRECT
  Confidence: 72.50%

IDENTIFIED ISSUES
------------------------------------------------------------
  • Left knee angle too acute (< 90°) - unstable base
  • Spine deviation from vertical (35.2° > 30.0°)
  • Poor head-to-foot alignment - balance issue

PROBLEMATIC FRAMES BY CATEGORY
------------------------------------------------------------
  Knee: Frames [12, 13, 14, 15, 16]
  Spine: Frames [20, 21, 22, 23]
  Alignment: Frames [10, 11, 12, 13, 14, 15]

BIOMETRIC FEATURE STATISTICS
------------------------------------------------------------
Left Elbow Angle:  Mean=145.2°, Min=120.5°, Max=170.1°
Right Elbow Angle: Mean=148.7°, Min=125.3°, Max=172.4°
Left Knee Angle:   Mean=85.3°, Min=65.2°, Max=110.5°
Right Knee Angle:  Mean=92.1°, Min=75.8°, Max=115.2°
============================================================
```

---

## 📁 Project Structure

```
sports_form_analysis/
│
├── data/
│   ├── raw_videos/          # Place your input videos here
│   ├── processed_keypoints/  # Processed keypoint data (auto-generated)
│   └── labels.csv            # Video labels (for future dataset expansion)
│
├── models/
│   ├── pose_extractor.py     # MediaPipe pose extraction
│   ├── feature_engineering.py # Biomechanical feature extraction
│   ├── rule_based_evaluator.py # Rule-based form evaluation
│   ├── ml_model.py           # Random Forest classifier
│   ├── train_model.py        # Model training script
│   └── form_classifier.pkl   # Trained model (generated after training)
│
├── app/
│   └── app.py                # Streamlit web application
│
├── utils/
│   ├── video_utils.py        # Video processing utilities
│   └── visualization.py      # Visualization and reporting
│
├── outputs/
│   ├── annotated_videos/     # Annotated output videos
│   └── reports/              # Analysis reports and plots
│
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── main.py                   # Main entry point (CLI + app launcher)
```

### File Descriptions

- **`pose_extractor.py`**: Extracts 33 keypoints per frame using MediaPipe
- **`feature_engineering.py`**: Calculates angles, alignment, center of mass, temporal features
- **`rule_based_evaluator.py`**: Applies biomechanical thresholds, generates feedback
- **`ml_model.py`**: Random Forest classifier wrapper
- **`train_model.py`**: Generates synthetic data and trains model
- **`app.py`**: Streamlit interface for video upload and analysis
- **`video_utils.py`**: Helper functions for video I/O
- **`visualization.py`**: Creates annotated videos, reports, plots
- **`main.py`**: CLI interface and app launcher

---

## 🔮 Future Improvements

### Short-Term
1. **Real Dataset Integration**: Replace synthetic data with real labeled videos
2. **Multi-Movement Support**: Separate models for cricket, gym, running, etc.
3. **3D Pose Estimation**: Add depth information for more accurate analysis
4. **Real-Time Analysis**: Process webcam feed in real-time

### Medium-Term
5. **Deep Learning Models**: Replace Random Forest with CNN or LSTM for sequence modeling
6. **Personalized Feedback**: User-specific form correction suggestions
7. **Comparative Analysis**: Compare current form to previous sessions
8. **Mobile App**: iOS/Android app for on-the-go analysis

### Long-Term
9. **Injury Prediction**: Predict injury risk based on form patterns
10. **Performance Optimization**: Suggest form improvements to maximize performance
11. **Multi-Person Analysis**: Analyze team sports with multiple athletes
12. **Integration with Wearables**: Combine with IMU/sensor data

---

## ⚠️ Important Notes

### Model Training

- **First-time setup**: You must run `python models/train_model.py` before using the system
- **Model location**: Trained model is saved to `models/form_classifier.pkl`
- **Retraining**: If you modify features or thresholds, retrain the model

### Video Requirements

- **Format**: MP4, AVI, MOV, MKV
- **Content**: Video must contain a visible person
- **Quality**: Higher resolution = better pose detection
- **Duration**: Any length (system processes all frames)

### Limitations

- **Single Person**: Currently analyzes one person per video
- **Frontal View**: Best results with front/side view (not top-down)
- **Lighting**: Poor lighting may reduce pose detection accuracy
- **Occlusion**: Heavy occlusion (e.g., equipment blocking body) may affect results

### Performance

- **Processing Time**: ~1-2 seconds per frame (depends on hardware)
- **Memory**: Requires ~2-4 GB RAM for typical videos
- **GPU**: Not required (runs on CPU)

---

## 📝 License

This project is provided as-is for educational and research purposes.

---

## 🤝 Contributing

This is a complete, production-ready system. To extend it:

1. Add new features in `feature_engineering.py`
2. Update thresholds in `rule_based_evaluator.py`
3. Retrain model with `train_model.py`
4. Test with real videos

---

## 📧 Support

For questions or issues:
1. Check the README first
2. Verify model is trained (`models/form_classifier.pkl` exists)
3. Ensure video contains visible person
4. Check console/terminal for error messages

---

**Built with ❤️ for sports science and biomechanics research**

