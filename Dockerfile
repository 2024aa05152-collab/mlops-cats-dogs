# Use lightweight Python 3.10 slim base image for smaller footprint
FROM python:3.10-slim

# Set working directory for the application
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies without caching to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy main application file and source code
COPY app.py .
COPY src/ src/

# Create necessary directories for logs and generated images
RUN mkdir -p logs images

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Healthcheck to verify application is running and responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# IMPORTANT: CNN model will be dynamically loaded at runtime from MLflow Model Registry
# No model packaging required in Docker image - model artifacts fetched via MLflow client

# Start FastAPI application with uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
