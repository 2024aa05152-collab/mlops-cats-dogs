"""
Training Script for Cats vs. Dogs Binary Classification.
This script fulfills Module 1 requirements for Model Building and Experiment Tracking.
"""

import os
import zipfile
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.keras

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parses hyperparameter arguments from the command line (useful for CI/CD automation)."""
    parser = argparse.ArgumentParser(description="Train Cats vs Dogs Baseline CNN")
    parser.add_argument("--img-size", type=int, default=224, help="Target image size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    return parser.parse_args()

def prepare_data(zip_path="data/processed.zip", extract_to="data"):
    """Unzips the DVC-tracked dataset if the processed folder doesn't exist."""
    processed_dir = os.path.join(extract_to, "processed")
    if not os.path.exists(processed_dir) and os.path.exists(zip_path):
        logger.info(f"📦 Extracting dataset from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info("✅ Dataset extraction complete.")
    elif not os.path.exists(processed_dir):
        logger.error(f"❌ Cannot find {zip_path} or {processed_dir}. Ensure data is pulled.")
        raise FileNotFoundError("Missing processed dataset.")
    else:
        logger.info("✅ Processed dataset directory already exists. Skipping extraction.")

def build_data_generators(data_dir, img_size, batch_size):
    """Creates train and validation data generators."""
    logger.info("⚙️ Initializing data generators...")
    
    # Training generator with simple offline augmentation logic already applied, 
    # so we just rescale here.
    train_gen = ImageDataGenerator(rescale=1.0 / 255)
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary"
    )

    val_data = val_gen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False # Must be False for accurate confusion matrix mapping
    )
    
    return train_data, val_data

def build_baseline_cnn(input_shape, learning_rate):
    """Builds and compiles the baseline Convolutional Neural Network."""
    logger.info("🧠 Building baseline CNN architecture...")
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5), # Added dropout for better generalization
        layers.Dense(1, activation="sigmoid")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def generate_and_log_plots(history, artifact_dir):
    """Generates loss and accuracy curves and logs them to MLflow."""
    logger.info("📊 Generating training curves...")
    
    # Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="Train Loss", color='blue')
    plt.plot(history.history["val_loss"], label="Val Loss", color='orange')
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    loss_path = os.path.join(artifact_dir, "loss_curve.png")
    plt.savefig(loss_path)
    plt.close()
    mlflow.log_artifact(loss_path)

    # Accuracy Curve
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["accuracy"], label="Train Accuracy", color='blue')
    plt.plot(history.history["val_accuracy"], label="Val Accuracy", color='orange')
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    acc_path = os.path.join(artifact_dir, "accuracy_curve.png")
    plt.savefig(acc_path)
    plt.close()
    mlflow.log_artifact(acc_path)

def generate_and_log_confusion_matrix(model, val_data, artifact_dir):
    """Generates a confusion matrix on validation data and logs it to MLflow."""
    logger.info("🧮 Generating confusion matrix...")
    
    val_preds_probs = model.predict(val_data)
    val_preds = (val_preds_probs > 0.5).astype(int).ravel()

    cm = confusion_matrix(val_data.classes, val_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Cat", "Dog"])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap="Blues", ax=ax)
    plt.title("Validation Confusion Matrix")
    
    cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

def main():
    args = parse_args()
    
    # Setup directories
    artifact_dir = "artifacts"
    model_dir = "models"
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Prepare Data
    prepare_data()
    train_data, val_data = build_data_generators("data/processed", args.img_size, args.batch_size)

    # Configure MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Cats_vs_Dogs_Prod")

    # Build Model
    model = build_baseline_cnn((args.img_size, args.img_size, 3), args.lr)

    with mlflow.start_run() as run:
        logger.info(f"🚀 Started MLflow run: {run.info.run_id}")
        
        # Log Hyperparameters
        mlflow.log_params({
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "optimizer": "adam",
            "loss_function": "binary_crossentropy"
        })

        # Train Model
        logger.info("⏳ Training started...")
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=args.epochs
        )

        # Log Metrics
        for epoch in range(args.epochs):
            mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history["accuracy"][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history["val_accuracy"][epoch], step=epoch)

        # Generate and Log Artifacts
        generate_and_log_plots(history, artifact_dir)
        generate_and_log_confusion_matrix(model, val_data, artifact_dir)

        # Save and Log Model as standard serialized .h5 format
        model_path = os.path.join(model_dir, "model.h5")
        model.save(model_path)
        logger.info(f"💾 Model saved locally to {model_path}")
        
        mlflow.keras.log_model(
            model, 
            artifact_path="model",
            registered_model_name="CatsDogs_CNN" # This pushes it directly to the DagsHub Registry
        )
        logger.info("✅ Training complete. All artifacts and models successfully logged to MLflow.")

if __name__ == "__main__":
    main()