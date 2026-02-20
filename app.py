from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uuid
import time

from src.monitoring import (
    log_request,
    log_response,
    track_prediction,
    get_metrics,
    get_prediction_stats,
    RequestTimer,
    REQUEST_COUNT
)

app = FastAPI()
model = tf.keras.models.load_model("models/model.h5")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return Response(
        content=get_metrics(),
        media_type="text/plain"
    )


@app.get("/stats")
def stats():
    """Get prediction statistics."""
    return get_prediction_stats()


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """Predict endpoint with monitoring and logging."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Log incoming request (excluding sensitive data)
    log_request(
        endpoint="/predict",
        request_id=request_id,
        metadata={"filename": file.filename}
    )
    
    # Process image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction with timing
    with RequestTimer():
        prediction = model.predict(img_array)[0][0]
    
    label = "dog" if prediction > 0.5 else "cat"
    latency = time.time() - start_time
    
    # Track prediction
    track_prediction(
        request_id=request_id,
        prediction=prediction,
        label=label,
        latency=latency
    )
    
    # Update request count
    REQUEST_COUNT.labels(endpoint="/predict", label=label).inc()
    
    # Log response (excluding sensitive data)
    log_response(
        endpoint="/predict",
        request_id=request_id,
        status_code=200,
        label=label,
        confidence=float(prediction),
        latency=latency
    )
    
    return {
        "label": label,
        "confidence": float(prediction)
    }


@app.post("/predict_with_label")
async def predict_with_label(request: Request, file: UploadFile = File(...), true_label: str = None):
    """
    Predict endpoint that accepts true label for performance tracking.
    Use this for collecting ground truth data.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Log incoming request
    log_request(
        endpoint="/predict_with_label",
        request_id=request_id,
        metadata={"filename": file.filename, "true_label": true_label}
    )
    
    # Process image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    with RequestTimer():
        prediction = model.predict(img_array)[0][0]
    
    label = "dog" if prediction > 0.5 else "cat"
    latency = time.time() - start_time
    
    # Track prediction with true label
    track_prediction(
        request_id=request_id,
        prediction=prediction,
        label=label,
        true_label=true_label,
        latency=latency
    )
    
    # Update request count
    REQUEST_COUNT.labels(endpoint="/predict_with_label", label=label).inc()
    
    # Log response
    log_response(
        endpoint="/predict_with_label",
        request_id=request_id,
        status_code=200,
        label=label,
        confidence=float(prediction),
        latency=latency,
        metadata={"true_label": true_label}
    )
    
    return {
        "label": label,
        "confidence": float(prediction),
        "true_label": true_label,
        "correct": label == true_label if true_label else None
    }

