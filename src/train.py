import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow
import mlflow.keras

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

mlflow.set_experiment("Cats_vs_Dogs")

train_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1./255)

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
    class_mode="binary"
)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

with mlflow.start_run():
    history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

    model.save("models/model.h5")
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])
    mlflow.keras.log_model(model, "model")
