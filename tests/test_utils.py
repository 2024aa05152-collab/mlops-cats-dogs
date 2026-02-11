import os
import sys

import pytest
import numpy as np

# Ensure the src folder is on the Python path so we can import utilities
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# Tests for data preprocessing and model inference utilities
from model_utils import preprocess_image, get_model_prediction 

def test_data_preprocessing():
    """
    Task: Unit test for data pre-processing function.
    Checks if a raw numpy image is correctly resized and normalized.
    """
    # Create a dummy RGB image (height=300, width=300, channels=3)
    dummy_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    processed_img = preprocess_image(dummy_img)
    
    # Assertions based on standard MobileNet/ResNet input needs (224x224)
    assert processed_img.shape == (224, 224, 3), "Image was not resized correctly"
    assert np.max(processed_img) <= 1.0, "Image pixels were not normalized to [0, 1]"
    assert isinstance(processed_img, np.ndarray)

def test_model_inference_utility():
    """
    Task: Unit test for model utility/inference function.
    Checks if the model accepts a batch and returns a valid prediction shape.
    """
    # Create a dummy batch of 1 image (1, 224, 224, 3)
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    # Simulate inference
    prediction = get_model_prediction(dummy_input)
    
    # For Cats vs Dogs, check if output is a single probability per sample
    assert prediction.shape == (1, 1) or prediction.shape == (1, 2)
    assert not np.isnan(prediction).any(), "Model returned NaN values"