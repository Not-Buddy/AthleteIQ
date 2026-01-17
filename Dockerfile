FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements_unified.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_unified.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "sports_form_analysis/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]