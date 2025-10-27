import cv2
import tensorflow as tf
import os
import numpy as np

CATEGORIES = ["Cat", "Dog"]  #label maps

IMG_SIZE = 64

def prepare(filepath):
    img_array = cv2.imread(filepath)
    if img_array is None:
        raise FileNotFoundError(f"Image not found: {filepath}")
    resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    return normalized.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

model_path = "64x3-CNN.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

model = tf.keras.models.load_model(model_path)

image_path = "cat.jpg"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

img_input = prepare(image_path)
prediction = model.predict(img_input)

prob = prediction[0][0]
threshold = 0.5

if prob > threshold:
    predicted_class = CATEGORIES[1]
else:
    predicted_class = CATEGORIES[0]

print(f"Prediction probability: {prob:.2f}")
print(f"Predicted class: {predicted_class}")








