import tensorflow as tf
import numpy as np

print("Downloading MobileNetV2 model...")
model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=True
)
model.save('plant_model.h5')
print("Model saved successfully!")