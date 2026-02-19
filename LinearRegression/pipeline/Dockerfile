FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for outputs
RUN mkdir -p data models metrics logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command - run local pipeline
CMD ["python", "app.py"]

# Usage:
# docker build -t mlops-lr-pipeline .
# docker run --env-file .env mlops-lr-pipeline
# docker run --env-file .env mlops-lr-pipeline python app.py --azure
# docker run --env-file .env mlops-lr-pipeline python demo.py
