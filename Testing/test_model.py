from ultralytics import YOLO
import cv2
import numpy as np

WEIGHTS_PATH = r"C:\PROJECTS\Sign Language Detection Engine\.model\Indian_SIgn_language_detection.pt"

print("Loading model...")
model = YOLO(WEIGHTS_PATH)
print("Model loaded successfully!")

print("Running dummy inference...")
dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
results = model(dummy_image, verbose=False)
print("Inference successful! Results:", results)
