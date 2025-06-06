import cv2
import easyocr
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Global set to track written license plates
written_plates = set()

def get_license_plate_text(image, bbox, reader):
    """Extract text from a license plate region using EasyOCR with image preprocessing."""
    x1, y1, x2, y2 = map(int, bbox)
    plate_img = image[y1:y2, x1:x2]
    if plate_img.size == 0:
        return ""

    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Step 1: Denoise the image using Gaussian blur to reduce noise
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 2: Sharpen the image using a kernel
    sharpening_kernel = np.array([[-1, -1, -1],
                                 [-1,  9, -1],
                                 [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, sharpening_kernel)

    # Step 3: Increase contrast using histogram equalization
    contrast = cv2.equalizeHist(sharpened)

    # Step 4: Apply thresholding to make text stand out (optional, depending on image quality)
    _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use EasyOCR to extract text from the processed image
    results = reader.readtext(thresh, paragraph=False)
    text = " ".join([result[1] for result in results])
    return text.strip()

def save_to_csv(plate_text, timestamp, csv_path="outputs/detect.csv"):
    """Save license plate text and timestamp to CSV only if not already written."""
    global written_plates
    
    # Skip if the plate text has already been written
    if plate_text in written_plates:
        return
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    data = {"Timestamp": [timestamp], "License Plate": [plate_text]}
    df = pd.DataFrame(data)
    try:
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode='a', header=False, index=False)
        # Add the plate text to the set of written plates
        written_plates.add(plate_text)
    except PermissionError as e:
        print(f"PermissionError: Unable to write to {csv_path}. {e}")
        print("Please ensure the file is not open in another application and you have write permissions.")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def draw_boxes(frame, boxes, labels, confidences, track_ids=None):
    """Draw bounding boxes and labels on the frame."""
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = labels[i]
        conf = confidences[i]
        text = f"{label} {conf:.2f}"
        if track_ids is not None:
            text += f" ID:{track_ids[i]}"
        color = (0, 255, 0) if label == "car" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame