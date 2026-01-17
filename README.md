# AthleteIQ - Sports Form Analysis

AI-powered sports biomechanics analysis system that evaluates athletic form using computer vision and machine learning.

## Features

- **Pose Estimation**: Uses MediaPipe Pose to detect 33 body keypoints
- **Form Analysis**: Combines rule-based evaluation with ML classification
- **Video Processing**: Upload and analyze sports movement videos
- **Visual Feedback**: Annotated videos showing problematic joints
- **Detailed Reports**: Comprehensive analysis with biomechanical insights

## Local Development

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AthleteIQ
```

2. Install dependencies:
```bash
pip install -r requirements_unified.txt
```

3. Run the application:
```bash
streamlit run sports_form_analysis/app/app.py
```

The app will be available at `http://localhost:8501`

## Railway Deployment

This application is optimized for deployment on Railway.app using Docker.

### Deployment Steps

1. **Connect to Railway**:
   - Go to [railway.app](https://railway.app)
   - Sign in with your GitHub account
   - Click "New Project" → "Deploy from GitHub repo"

2. **Configure Deployment**:
   - Select this repository
   - Railway will automatically detect the `Dockerfile` and `railway.json`
   - The app will build and deploy automatically

3. **Environment Variables** (if needed):
   - Railway automatically assigns a `$PORT` environment variable
   - The app is configured to use this port automatically

### Deployment Files

- `Dockerfile`: Defines the container environment with all dependencies
- `railway.json`: Railway deployment configuration
- `Procfile`: Alternative deployment method (for non-Docker deployments)
- `requirements_unified.txt`: Python dependencies

### System Requirements

The application requires system libraries for OpenCV:
- `libgl1-mesa-glx`
- `libglib2.0-0`

These are included in the Dockerfile for Railway deployment.

## Usage

1. Upload a video of a sports movement (MP4, AVI, MOV, MKV formats)
2. Click "Analyze Form" to process the video
3. View the analysis results including:
   - Rule-based evaluation verdict
   - ML-based classification confidence
   - Annotated video with pose overlay
   - Joint-level feedback
   - Downloadable reports

## Supported Movements

- Cricket shots
- Gym exercises
- General athletic movements

## Technical Architecture

- **Frontend**: Streamlit web interface
- **Computer Vision**: MediaPipe Pose estimation
- **ML Model**: Random Forest classifier (pre-trained)
- **Analysis**: Biomechanical rule engine + ML classification

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure `sports_form_analysis/models/form_classifier.pkl` exists
2. **Import errors**: Verify all dependencies are installed from `requirements_unified.txt`
3. **Video processing issues**: Check that uploaded videos contain visible human subjects

### Railway-Specific Issues

1. **Build failures**: Check the build logs in Railway dashboard
2. **Memory limits**: Large video files may exceed Railway's memory limits
3. **Cold starts**: First analysis may take longer due to model loading

## Development Notes

The application uses a modular architecture:
- `models/`: Machine learning and analysis components
- `utils/`: Helper functions for video processing and visualization
- `app/`: Streamlit web interface

All paths are configured to work in both local and cloud environments.