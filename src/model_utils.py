import numpy as np
from PIL import Image

IMG_SIZE = 224


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Resize a raw RGB numpy image to IMG_SIZE and normalize to [0, 1]."""
    if image is None:
        raise ValueError("image must not be None")

    if image.ndim != 3:
        raise ValueError("image must have shape (H, W, C)")

    pil_img = Image.fromarray(image.astype(np.uint8), mode="RGB")
    pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    return arr


def get_model_prediction(batch: np.ndarray) -> np.ndarray:
    """Simple dummy "inference" using pure NumPy.

    Expects a batch with shape (N, H, W, C) and returns probabilities
    of shape (N, 1) in (0, 1), suitable for cats vs dogs.
    """
    if batch is None:
        raise ValueError("batch must not be None")

    arr = np.asarray(batch, dtype=np.float32)

    if arr.ndim != 4:
        raise ValueError("batch must be 4D: (batch, height, width, channels)")

    n = arr.shape[0]
    # Very simple scoring: mean pixel per image
    logits = arr.reshape(n, -1).mean(axis=1, keepdims=True)
    # Sigmoid to obtain probabilities
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs