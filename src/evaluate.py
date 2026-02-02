import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model("models/model.h5")

test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    "data/processed/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

loss, acc = model.evaluate(test_data)
print(f"Test Accuracy: {acc}")
