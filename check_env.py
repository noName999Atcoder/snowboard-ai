import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

print("TF:", tf.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)

model = hub.load(
    "https://tfhub.dev/google/movenet/singlepose/lightning/4"
)
print("MoveNet loaded OK")
