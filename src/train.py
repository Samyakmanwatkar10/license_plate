from ultralytics import YOLO

def train_model():
    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")  # Start with pretrained nano model

    # Train the model
    model.train(
        data="", # enter path to the data.yaml 
        epochs=50,
        imgsz=640,
        batch=16,
        # device=0,  # Use GPU if available
        project="outputs/runs",
        name="license_plate",
        exist_ok=True
    )

if __name__ == "__main__":
    train_model()