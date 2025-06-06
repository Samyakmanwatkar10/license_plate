import cv2
import numpy as np
from ultralytics import YOLO
from utils import get_license_plate_text, save_to_csv, draw_boxes
import easyocr
from datetime import datetime
import time

def main(input_source=0):
    # Load models and OCR reader
    vehicle_model = YOLO("yolov8n.pt")  # Pretrained model for vehicle detection (nano model, lightest)
    plate_model = YOLO("../outputs/runs/license_plate/weights/best.pt")  # Trained license plate model
    reader = easyocr.Reader(['en'], gpu=True)  # Initialize EasyOCR

    # Open video or webcam
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Frame skipping for OCR (process OCR less frequently)
    ocr_frame_skip = 10  # Increased to reduce OCR calls
    frame_count = 0

    # Cache for license plate texts by vehicle ID
    plate_text_cache = {}

    # Target frame rate (e.g., 30 FPS)
    target_fps = 30
    frame_time = 1.0 / target_fps

    while cap.isOpened():
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to reduce YOLOv8 inference time
        scale_factor = 0.3  # Reduced for better performance
        resized_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

        # Track vehicles (class 'car' = 2 in COCO dataset) using built-in track method
        vehicle_results = vehicle_model.track(resized_frame, classes=[2], conf=0.5, persist=True, tracker="botsort.yaml")
        vehicle_boxes = []
        vehicle_ids = []
        vehicle_confs = []

        # Extract tracked vehicles and scale bounding boxes back to original frame size
        if vehicle_results[0].boxes is not None:
            for box in vehicle_results[0].boxes:
                bbox = box.xyxy[0].cpu().numpy() / scale_factor
                vehicle_boxes.append(bbox)
                vehicle_ids.append(int(box.id.item()) if box.id is not None else -1)
                vehicle_confs.append(box.conf.item() if box.conf is not None else 0.5)

        # Detect license plates
        plate_results = plate_model(resized_frame, conf=0.5)
        plate_boxes = []
        plate_confs = []
        plate_texts = []

        if plate_results[0].boxes is not None:
            for box in plate_results[0].boxes.xyxy.cpu().numpy():
                bbox = box[:4] / scale_factor
                plate_boxes.append(bbox)
                plate_confs.append(box[4] if len(box) > 4 else 0.5)

                # Associate license plate with the nearest vehicle
                associated_vehicle_id = -1
                for i, v_box in enumerate(vehicle_boxes):
                    v_x1, v_y1, v_x2, v_y2 = v_box
                    p_x1, p_y1, p_x2, p_y2 = bbox
                    if (p_x1 >= v_x1 and p_x2 <= v_x2 and p_y1 >= v_y1 and p_y2 <= v_y2):
                        associated_vehicle_id = vehicle_ids[i]
                        break

                # Perform OCR only every nth frame or if not in cache
                text = ""
                if associated_vehicle_id != -1:
                    if frame_count % ocr_frame_skip == 0 or associated_vehicle_id not in plate_text_cache:
                        text = get_license_plate_text(frame, bbox, reader)
                        if text:
                            plate_text_cache[associated_vehicle_id] = text
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            save_to_csv(text, timestamp)
                    else:
                        # Use cached text if available
                        text = plate_text_cache.get(associated_vehicle_id, "")
                plate_texts.append(text)

        # Draw bounding boxes and labels on the original frame
        frame = draw_boxes(
            frame,
            vehicle_boxes + plate_boxes,
            ["car"] * len(vehicle_boxes) + ["license_plate"] * len(plate_boxes),
            vehicle_confs + plate_confs,
            vehicle_ids + [None] * len(plate_boxes)
        )

        # Display license plate text
        for i, text in enumerate(plate_texts):
            if text:
                x1, y1 = map(int, plate_boxes[i][:2])
                cv2.putText(frame, text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display frame
        cv2.imshow("License Plate Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

        # Control frame rate
        elapsed_time = time.time() - start_time
        if elapsed_time < frame_time:
            time.sleep(frame_time - elapsed_time)
        else:
            print(f"Frame {frame_count} took {elapsed_time:.3f}s, longer than target {frame_time:.3f}s")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use 0 for webcam, or provide a video file path
    main(input_source='demo.mp4')  # Path to video file