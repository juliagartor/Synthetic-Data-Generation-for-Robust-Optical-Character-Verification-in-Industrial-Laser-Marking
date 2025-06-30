import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import argparse

def load_image(image_path):
    """Loads an image from the given path and converts it to RGB."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Error: Unable to load image at {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def load_model(model_path):
    """Loads the YOLO model from the given path."""
    return YOLO(model_path)

def predict(model, image, conf_threshold=0.5, img_size=640):
    """Runs YOLO model prediction on the given image and extracts bounding boxes."""

    results = model.predict(source=image, conf=conf_threshold, imgsz=img_size)
    class_names = model.names 

    bboxes = []

    if not results: 
        return bboxes

    for result in results:
        for box in result.boxes:
            cls = int(box.cls.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            class_name = class_names[cls] if cls in class_names else f"Unknown_{cls}"
            bboxes.append([class_name, (x1, y1), (x2, y2)])
            
    return bboxes


def main(model_path, image_path):
    """Main function to load the model, process the image, and run predictions."""
    print("Loading model...")
    model = load_model(model_path)
    
    print("Processing image...")
    image_rgb = load_image(image_path)
    
    print("Running detection...")
    results = predict(model, image_rgb)
    
    print("Prediction complete.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO detection on an image.")
    parser.add_argument("model_path", type=str, help="Path to the trained YOLO model.")
    parser.add_argument("image_path", type=str, help="Path to the image to be processed.")
    
    args = parser.parse_args()
    
    results = main(args.model_path, args.image_path)
    print(results)