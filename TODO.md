# TODO: M5 Monitoring & Logging Implementation

## Information Gathered:
- FastAPI application with `/predict` and `/health` endpoints
- TensorFlow model for cats vs dogs classification
- No existing monitoring or logging infrastructure
- Python dependencies: tensorflow, fastapi, uvicorn, numpy, pillow

## Plan:

### Step 1: Update requirements.txt
- Add `prometheus-client` for metrics collection
- Add `python-json-logger` for structured logging
- Add `loguru` for enhanced logging

### Step 2: Create monitoring module (src/monitoring.py)
- Implement metrics tracking class with:
  - Request count counter
  - Latency histogram
  - Prediction cache for batch collection
- Implement structured logging with:
  - Request/response logging (excluding image data)
  - Timestamp, endpoint, latency, prediction info

### Step 3: Update app.py with monitoring
- Integrate metrics and logging into FastAPI app
- Add middleware for automatic request/response logging
- Expose `/metrics` endpoint for Prometheus scraping

### Step 4: Create simulated requests script (scripts/collect_requests.py)
- Generate simulated requests with true labels
- Store prediction results with ground truth for performance tracking

## Dependent Files to be Edited:
- requirements.txt
- app.py
- src/monitoring.py (new file)
- scripts/collect_requests.py (new file)

## Followup Steps:
- Test the monitoring endpoints locally
- Verify logs are being generated correctly
- Ensure metrics are accessible via /metrics endpoint

## Implementation Complete:
- ✅ requirements.txt updated with prometheus-client and loguru
- ✅ src/monitoring.py created with metrics and logging
- ✅ app.py updated with monitoring integration
- ✅ scripts/collect_requests.py created for performance tracking
- ✅ logs directory created

