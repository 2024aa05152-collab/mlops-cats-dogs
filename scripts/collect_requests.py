"""
Script to collect simulated requests for model performance tracking.
This generates test requests with known true labels to evaluate 
post-deployment model performance.
"""

import os
import sys
import random
import time
import json
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from PIL import Image
import io


# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "20"))
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "prediction_results.json")


def create_test_image(label: str) -> bytes:
    """
    Create a simple test image.
    In a real scenario, you would use actual cat/dog images.
    """
    # Create a simple colored image based on label
    if label == "cat":
        # Orange-ish color for cat
        color = (255, 165, 0)
    else:
        # Blue-ish color for dog
        color = (0, 165, 255)
    
    # Create a simple 224x224 image
    img = Image.new("RGB", (224, 224), color)
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes.getvalue()


def send_prediction_request(image_bytes: bytes, true_label: str = None) -> dict:
    """
    Send a prediction request to the API.
    """
    files = {"file": ("test_image.jpg", image_bytes, "image/jpeg")}
    
    # Use the endpoint that accepts true labels
    url = f"{API_BASE_URL}/predict_with_label"
    
    if true_label:
        data = {"true_label": true_label}
    else:
        data = {}
    
    try:
        response = requests.post(url, files=files, data=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return {"error": str(e)}


def collect_requests(num_samples: int = 20) -> list:
    """
    Collect simulated requests with true labels.
    """
    results = []
    labels = ["cat", "dog"]
    
    print(f"Collecting {num_samples} samples...")
    print(f"API URL: {API_BASE_URL}")
    print("-" * 50)
    
    for i in range(num_samples):
        # Alternate between cat and dog
        true_label = labels[i % 2]
        
        # Create test image
        image_bytes = create_test_image(true_label)
        
        print(f"[{i+1}/{num_samples}] Sending request with true_label={true_label}...")
        
        # Send request
        result = send_prediction_request(image_bytes, true_label)
        
        # Record result
        record = {
            "sample_id": i + 1,
            "true_label": true_label,
            "predicted_label": result.get("label"),
            "confidence": result.get("confidence"),
            "correct": result.get("correct"),
            "timestamp": time.time()
        }
        
        if "error" in result:
            record["error"] = result["error"]
            print(f"  -> Error: {result['error']}")
        else:
            print(f"  -> Predicted: {result.get('label')} (confidence: {result.get('confidence'):.4f})")
            if result.get("correct") is not None:
                print(f"  -> Correct: {result.get('correct')}")
        
        results.append(record)
        
        # Small delay between requests
        time.sleep(0.5)
    
    return results


def calculate_metrics(results: list) -> dict:
    """
    Calculate performance metrics from collected results.
    """
    valid_results = [r for r in results if "error" not in r and r.get("correct") is not None]
    
    if not valid_results:
        return {"error": "No valid results to calculate metrics"}
    
    total = len(valid_results)
    correct = sum(1 for r in valid_results if r.get("correct"))
    
    accuracy = correct / total if total > 0 else 0
    
    # Per-label accuracy
    label_stats = {}
    for label in ["cat", "dog"]:
        label_results = [r for r in valid_results if r.get("true_label") == label]
        if label_results:
            label_correct = sum(1 for r in label_results if r.get("correct"))
            label_stats[label] = {
                "total": len(label_results),
                "correct": label_correct,
                "accuracy": label_correct / len(label_results)
            }
    
    # Confidence statistics
    confidences = [r["confidence"] for r in valid_results]
    
    return {
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy": round(accuracy, 4),
        "per_label_accuracy": label_stats,
        "confidence_stats": {
            "mean": round(np.mean(confidences), 4),
            "std": round(np.std(confidences), 4),
            "min": round(min(confidences), 4),
            "max": round(max(confidences), 4)
        }
    }


def main():
    print("=" * 50)
    print("Model Performance Collection Script")
    print("=" * 50)
    
    # Check if API is available
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        health_response.raise_for_status()
        print(f"API Health Check: {health_response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error: Cannot connect to API at {API_BASE_URL}")
        print(f"Please ensure the API is running: uvicorn app:app")
        sys.exit(1)
    
    print()
    
    # Collect requests
    results = collect_requests(NUM_SAMPLES)
    
    print()
    print("-" * 50)
    print("Collection complete!")
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Prepare output
    output = {
        "collection_time": time.time(),
        "num_samples": NUM_SAMPLES,
        "api_url": API_BASE_URL,
        "results": results,
        "metrics": metrics
    }
    
    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {OUTPUT_FILE}")
    print()
    print("Metrics Summary:")
    print(f"  Total Samples: {metrics.get('total_samples', 'N/A')}")
    print(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")
    print(f"  Per-label accuracy: {metrics.get('per_label_accuracy', {})}")
    
    return output


if __name__ == "__main__":
    main()

