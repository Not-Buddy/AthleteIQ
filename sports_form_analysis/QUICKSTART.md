# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python models/train_model.py
```

This will:
- Generate synthetic training data
- Train the Random Forest classifier
- Save model to `models/form_classifier.pkl`
- Display training metrics

**Expected output:**
```
Generating synthetic training data...
Generated 100 samples with 54 features each
Training Random Forest classifier...
Accuracy:  0.95
Precision: 0.96
Recall:    0.94
F1 Score:  0.95
Model saved to: models/form_classifier.pkl
```

### Step 3: Run the Application

**Option A: Streamlit Web Interface (Recommended)**
```bash
streamlit run app/app.py
```

Then open your browser to the URL shown (usually `http://localhost:8501`)

**Option B: Command Line Interface**
```bash
python main.py --video path/to/your/video.mp4
```

## 📹 Testing with Your Own Video

1. Place your video in `data/raw_videos/` or use any path
2. Ensure the video shows a person performing a movement
3. Upload through Streamlit or use CLI
4. View results:
   - Annotated video with pose overlay
   - Rule-based and ML-based verdicts
   - Joint-level feedback
   - Feature timeline plots
   - Detailed report

## ✅ Verification Checklist

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Model trained (`models/form_classifier.pkl` exists)
- [ ] Test video ready (contains visible person)
- [ ] Application runs without errors

## 🐛 Troubleshooting

**"Model not found" error:**
- Run `python models/train_model.py` first

**"No pose detected" error:**
- Ensure video contains a visible person
- Check video quality and lighting
- Try a different video

**Import errors:**
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.10+)

**Streamlit not launching:**
- Install Streamlit: `pip install streamlit`
- Try: `python -m streamlit run app/app.py`

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the code in `models/` to understand the system
- Modify thresholds in `rule_based_evaluator.py` for your use case
- Add your own training data to improve the model

---

**Ready to analyze! 🎉**

