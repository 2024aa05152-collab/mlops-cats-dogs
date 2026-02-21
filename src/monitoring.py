"""
Monitoring module for cats/dogs classification service.
Provides metrics tracking and structured logging.
"""

import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from loguru import logger

# Configure loguru
logger.remove()
logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
)

# Console output for development
logger.add(
    lambda msg: print(msg),
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Prometheus Metrics
REQUEST_COUNT = Counter(
    'inference_requests_total',
    'Total number of inference requests',
    ['endpoint', 'label']
)

LATENCY_HISTOGRAM = Histogram(
    'inference_latency_seconds',
    'Inference request latency in seconds',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

ACTIVE_REQUESTS = Gauge(
    'inference_active_requests',
    'Number of currently active inference requests'
)

MODEL_PREDICTIONS = Counter(
    'model_predictions_total',
    'Total predictions by predicted label',
    ['predicted_label']
)

# In-memory storage for prediction tracking (for batch analysis)
# In production, this would be sent to a database or external system
class PredictionTracker:
    """Tracks predictions for post-deployment performance analysis."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.predictions: List[Dict[str, Any]] = []
    
    def add_prediction(
        self,
        request_id: str,
        prediction: float,
        label: str,
        true_label: Optional[str] = None,
        latency: Optional[float] = None
    ):
        """Add a prediction record."""
        record = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "prediction": float(prediction),
            "predicted_label": label,
            "true_label": true_label,
            "latency_seconds": latency
        }
        
        self.predictions.append(record)
        
        # Maintain max size
        if len(self.predictions) > self.max_size:
            self.predictions = self.predictions[-self.max_size:]
    
    def get_predictions(self) -> List[Dict[str, Any]]:
        """Get all tracked predictions."""
        return self.predictions
    
    def get_labeled_predictions(self) -> List[Dict[str, Any]]:
        """Get predictions that have true labels (for performance tracking)."""
        return [p for p in self.predictions if p.get("true_label") is not None]
    
    def calculate_accuracy(self) -> Optional[float]:
        """Calculate accuracy on labeled predictions."""
        labeled = self.get_labeled_predictions()
        if not labeled:
            return None
        
        correct = sum(
            1 for p in labeled 
            if p["predicted_label"] == p["true_label"]
        )
        return correct / len(labeled)
    
    def reset(self):
        """Clear all predictions."""
        self.predictions = []


# Global prediction tracker
prediction_tracker = PredictionTracker(max_size=1000)


def log_request(
    endpoint: str,
    request_id: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Log incoming request.
    Note: Excludes sensitive data like image content.
    """
    log_data = {
        "event": "request_received",
        "endpoint": endpoint,
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if metadata:
        # Only include non-sensitive metadata
        safe_metadata = {
            k: v for k, v in metadata.items() 
            if k not in ["image_data", "file_content", "password", "token"]
        }
        log_data["metadata"] = safe_metadata
    
    logger.info(json.dumps(log_data))


def log_response(
    endpoint: str,
    request_id: str,
    status_code: int,
    label: Optional[str] = None,
    confidence: Optional[float] = None,
    latency: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Log outgoing response.
    Note: Excludes sensitive data.
    """
    log_data = {
        "event": "response_sent",
        "endpoint": endpoint,
        "request_id": request_id,
        "status_code": status_code,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    # Include non-sensitive response data
    if label is not None:
        log_data["prediction_label"] = label
    if confidence is not None:
        log_data["confidence"] = round(confidence, 4)
    if latency is not None:
        log_data["latency_seconds"] = round(latency, 4)
    
    if metadata:
        safe_metadata = {
            k: v for k, v in metadata.items()
            if k not in ["image_data", "file_content", "password", "token"]
        }
        log_data["metadata"] = safe_metadata
    
    logger.info(json.dumps(log_data))


def track_prediction(
    request_id: str,
    prediction: float,
    label: str,
    true_label: Optional[str] = None,
    latency: Optional[float] = None
):
    """Track a prediction for batch analysis."""
    prediction_tracker.add_prediction(
        request_id=request_id,
        prediction=prediction,
        label=label,
        true_label=true_label,
        latency=latency
    )
    
    # Update Prometheus counter
    MODEL_PREDICTIONS.labels(predicted_label=label).inc()


def get_metrics() -> bytes:
    """Generate Prometheus metrics in text format."""
    return generate_latest()


def get_prediction_stats() -> Dict[str, Any]:
    """Get prediction statistics for monitoring."""
    predictions = prediction_tracker.get_predictions()
    labeled = prediction_tracker.get_labeled_predictions()
    
    stats = {
        "total_predictions": len(predictions),
        "labeled_predictions": len(labeled),
    }
    
    accuracy = prediction_tracker.calculate_accuracy()
    if accuracy is not None:
        stats["accuracy"] = round(accuracy, 4)
    
    # Label distribution
    label_counts = defaultdict(int)
    for p in predictions:
        label_counts[p["predicted_label"]] += 1
    stats["label_distribution"] = dict(label_counts)
    
    return stats


class RequestTimer:
    """Context manager for timing requests."""
    
    def __init__(self):
        self.start_time = None
        self.latency = None
    
    def __enter__(self):
        self.start_time = time.time()
        ACTIVE_REQUESTS.inc()
        return self
    
    def __exit__(self, *args):
        self.latency = time.time() - self.start_time
        ACTIVE_REQUESTS.dec()
        LATENCY_HISTOGRAM.observe(self.latency)

