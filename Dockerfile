FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements_unified.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_unified.txt

# Verify and reinstall mediapipe if needed to ensure proper installation
RUN python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__)" || pip install --force-reinstall mediapipe>=0.10.30

# Copy application code
COPY . .

# Expose port
EXPOSE 8501