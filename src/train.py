import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# Config
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
OPTIMIZER = "adam"
LOSS = "binary_crossentropy"

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

mlflow.set_experiment("Cats_vs_Dogs")

# -----------------------------
# Data Generators
# -----------------------------
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_gen.flow_from_directory(
    "data/processed/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data = val_gen.flow_from_directory(
    "data/processed/val",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# -----------------------------
# Model
# -----------------------------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=OPTIMIZER,
    loss=LOSS,
    metrics=["accuracy"]
)

# -----------------------------
# MLflow Tracking
# -----------------------------
with mlflow.start_run():

    # Log parameters
    mlflow.log_param("img_size", IMG_SIZE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("optimizer", OPTIMIZER)
    mlflow.log_param("loss_function", LOSS)

    # Train
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )

    # Log metrics per epoch
    for epoch in range(EPOCHS):
        mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
        mlflow.log_metric("train_accuracy", history.history["accuracy"][epoch], step=epoch)
        mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)
        mlflow.log_metric("val_accuracy", history.history["val_accuracy"][epoch], step=epoch)

    # -----------------------------
    # Loss & Accuracy Curves
    # -----------------------------
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    loss_path = f"{ARTIFACT_DIR}/loss_curve.png"
    plt.savefig(loss_path)
    plt.close()
    mlflow.log_artifact(loss_path)

    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    acc_path = f"{ARTIFACT_DIR}/accuracy_curve.png"
    plt.savefig(acc_path)
    plt.close()
    mlflow.log_artifact(acc_path)

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    val_preds = model.predict(val_data)
    val_preds = (val_preds > 0.5).astype(int).ravel()

    cm = confusion_matrix(val_data.classes, val_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Cat", "Dog"])

    disp.plot(cmap="Blues")
    cm_path = f"{ARTIFACT_DIR}/confusion_matrix.png"
    plt.title("Confusion Matrix")
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # -----------------------------
    # Save & Log Model
    # -----------------------------
    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")
    mlflow.keras.log_model(model, artifact_path="model")
