import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load the trained model
try:
    model = tf.keras.models.load_model("model.h5")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Load class labels
labels_path = "class_labels.txt"
if os.path.exists(labels_path):
    with open(labels_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    # Fallback class names
    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


def predict_flower(img_path):
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array, verbose=0)[0]
        class_index = np.argmax(prediction)
        label = class_names[class_index]
        confidence = float(prediction[class_index])

        return label, confidence

    except Exception as e:
        raise RuntimeError(f"Prediction error: {e}")
