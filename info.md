### Project Information

**Project Name**: License Plate Detection and Recognition System  

#### Overview
This project is an automated system for detecting and recognizing license plates in video feeds. It leverages deep learning and computer vision techniques to:
- Detect vehicles and license plates using the YOLOv8 object detection model.
- Extract text from detected license plates using optical character recognition (OCR) with EasyOCR.
- Track vehicles across frames for consistent license plate association.
- Save the extracted license plate text along with timestamps to a CSV file.
- Display the results in real-time on the video feed with bounding boxes and text annotations.

#### Goals
- Train a YOLOv8 model to accurately detect license plates on a custom dataset.
- Perform real-time detection on a video file (`demo.mp4`) or webcam feed.
- Achieve reliable OCR for English alphanumeric license plates.
- Ensure the system is portable across different systems using dynamic path resolution.

#### File Structure
The project directory structure is as follows:

```
license_plate/
│
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
│
├── src/
│   ├── detect.py
│   ├── utils.py
│   └── train.py
│
├── demo.mp4
│
├── outputs/
│   ├── runs/
│   │   └── license_plate/
│   │       └── weights/
│   │           ├── best.pt
│   │           └── ...
│   └── results.csv
│
├── requirements.txt
└── README.md
```

- **`data/`**: Contains the dataset (extracted from `data.zip`) with training, validation, and test sets, along with `data.yaml` for YOLOv8 configuration.
- **`src/`**: Contains the Python scripts for training and detection.
- **`demo.mp4`**: Input video file for detection.
- **`outputs/`**: Stores training outputs (`runs/`) and detection results (`results.csv`).
- **`requirements.txt`**: Lists the required Python libraries.
- **`README.md`**: Project documentation with setup instructions.

---

### Code Structure

#### `train.py`
This script trains a YOLOv8 model on the custom dataset to detect license plates.

- **Purpose**: Trains a YOLOv8 model using the dataset specified in `data.yaml`.
- **Input**: `data.yaml` (dataset configuration), pretrained YOLOv8 nano model (`yolov8n.pt`).
- **Output**: Trained model weights saved to `outputs/runs/license_plate/weights/best.pt`.
- **Key Parameters**:
  - `data="../data/data.yaml"`: Path to the dataset configuration file.
  - `epochs=50`: Number of training epochs.
  - `imgsz=640`: Image size for training.
  - `batch=16`: Batch size.
  - `project="outputs/runs"`: Directory to save training outputs.
  - `name="license_plate"`: Name of the training run.
  - `exist_ok=True`: Allows overwriting existing training outputs.

#### `detect.py`
This script performs real-time detection on a video feed, extracts license plate text, and saves results.

- **Purpose**: Detects vehicles and license plates in a video, extracts text, and saves results.
- **Input**: Video file (`demo.mp4`) or webcam feed (`input_source=0`).
- **Output**: Displays video with bounding boxes and extracted text; saves results to `outputs/results.csv`.
- **Key Features**:
  - Uses two YOLOv8 models: one for vehicle detection (`yolov8n.pt`) and one for license plate detection (`best.pt`).
  - Uses EasyOCR for text extraction.
  - Implements frame skipping (`ocr_frame_skip=10`) to reduce OCR calls.
  - Resizes frames (`scale_factor=0.3`) to improve inference speed.
  - Maintains a target frame rate of 30 FPS.

#### `utils.py`
This script contains helper functions for OCR, saving results, and drawing bounding boxes.

- **Purpose**: Provides utility functions for OCR, saving results, and visualization.
- **Functions**:
  - `get_license_plate_text()`: Preprocesses the license plate image (grayscale conversion, denoising, sharpening, contrast enhancement, thresholding) and extracts text using EasyOCR.
  - `save_to_csv()`: Saves extracted license plate text and timestamps to a CSV file, avoiding duplicates using a global set (`written_plates`).
  - `draw_boxes()`: Draws bounding boxes and labels on video frames with different colors for cars (green) and license plates (red).

#### `data.yaml`
This file configures the dataset for YOLOv8 training.

- **Purpose**: Defines the dataset structure for YOLOv8.
- **Keys**:
  - `path: .`: Points to `license_plate/data/`, the directory containing `train/`, `valid/`, and `test/`.
  - `train: train/images`: Path to training images (`license_plate/data/train/images/`).
  - `val: valid/images`: Path to validation images (`license_plate/data/valid/images/`).
  - `test: test/images`: Path to test images (`license_plate/data/test/images/`).
  - `nc: 1`: Number of classes (1 for license plates).
  - `names: ['license_plate']`: Class names.

---

### Libraries Used

The project relies on the following Python libraries (based on the minimal `requirements.txt` created earlier):

#### Direct Dependencies
- **`easyocr==1.7.1`**: Used for optical character recognition to extract text from license plates.
- **`numpy==1.26.4`**: Used for numerical operations (e.g., array manipulation in OpenCV and EasyOCR).
- **`opencv-python==4.10.0.84`**: Used for image processing, video handling, and drawing bounding boxes.
- **`pandas==2.2.2`**: Used for saving detection results to a CSV file.
- **`ultralytics==8.2.58`**: Used for YOLOv8 model training and inference.

#### Indirect Dependencies
These are required by the direct dependencies and are automatically installed:
- **`torch==2.3.1`**: Backend for YOLOv8 (used by `ultralytics`).
- **`torchvision==0.18.1`**: Image handling for YOLOv8.
- **`PyYAML==6.0.2`**: Parses `data.yaml` (used by `ultralytics`).
- **`tqdm==4.67.1`**: Progress bars during training (used by `ultralytics`).
- **`pillow==11.2.1`**: Image handling (used by `easyocr` and `ultralytics`).
- **`python-bidi==0.6.6`**: Bidirectional text support (used by `easyocr`).
- **`pyclipper==1.3.0.post6`**: Polygon clipping (used by `easyocr`).
- **`shapely==2.1.1`**: Geometric operations (used by `easyocr`).
- **`scipy==1.15.3`**: Scientific computations (used by `easyocr`).
- **`ninja==1.11.1.4`**: Builds C++ extensions (used by `easyocr`).
- **`psutil==7.0.0`**: System resource monitoring (used by `ultralytics`).
- **`requests==2.32.3`**: Downloads pretrained models (used by `ultralytics`).
  - `certifi`, `charset-normalizer`, `idna`, `urllib3`: Dependencies of `requests`.
- **`python-dateutil==2.9.0.post0`**: Date parsing (used by `pandas`).
- **`pytz==2025.2`**: Timezone handling (used by `pandas`).
- **`tzdata==2025.2`**: Timezone data (used by `pandas`).
- **`typing_extensions==4.14.0`**: Type hints (used by `torch`).
- **`filelock==3.18.0`**: File locking (used by `torch`).
- **`fsspec==2025.5.1`**: File system abstractions (used by `torch`).
- **`Jinja2==3.1.6`**: Templating (used by `torch`).
  - `MarkupSafe`: Dependency of `Jinja2`.
- **`ultralytics-thop==2.0.14`**: Calculates model FLOPs (used by `ultralytics`).
- **`six==1.17.0`**: Python 2/3 compatibility (used by `python-dateutil`).

### Technologies Used

The project leverages the following technologies and tools:

1. **YOLOv8 (via `ultralytics`)**:
   - A state-of-the-art object detection model used for detecting vehicles and license plates.
   - Features built-in tracking (`botsort.yaml`) for associating license plates with vehicles across frames.
   - Pretrained model (`yolov8n.pt`) used for vehicle detection; custom-trained model (`best.pt`) for license plate detection.

2. **EasyOCR**:
   - An OCR library used to extract text from license plates.
   - Supports English (`en`) in the current configuration but can be extended for other languages.
   - Includes image preprocessing steps (denoising, sharpening, contrast enhancement, thresholding) to improve OCR accuracy.

3. **OpenCV (`opencv-python`)**:
   - Handles video capture, frame processing, and visualization.
   - Used for resizing frames, drawing bounding boxes, and displaying text on the video feed.

4. **PyTorch (`torch` and `torchvision`)**:
   - Backend for YOLOv8, providing GPU acceleration for model training and inference.
   - Enables efficient tensor operations and model loading.

5. **Pandas**:
   - Used to save detection results (license plate text and timestamps) to a CSV file in a structured format.

6. **Python Standard Libraries**:
   - `os`: For dynamic path resolution and directory creation.
   - `datetime`: For timestamp generation.
   - `time`: For frame rate control and performance monitoring.

7. **Dataset Format**:
   - The dataset follows the YOLO format, with images and corresponding label files (`.txt`) containing bounding box coordinates and class labels.
   - Configured via `data.yaml` for use with YOLOv8.

8. **Hardware Acceleration**:
   - The project supports GPU acceleration for YOLOv8 and EasyOCR (via `gpu=True` in EasyOCR). If a GPU is unavailable, it falls back to CPU, though performance may be slower.

---

### Additional Project Details

#### Assumptions
- The input video (`demo.mp4`) contains vehicles with visible license plates.
- License plates contain English alphanumeric characters (EasyOCR configured for `en` language).
- The dataset (`data.zip`) is properly annotated with license plate bounding boxes in YOLO format.
- A GPU is available for optimal performance (though the system works on CPU as well).

#### Limitations
- OCR accuracy depends on image quality (e.g., lighting, angle, resolution).
- Real-time performance may degrade on systems without a GPU or with high-resolution videos.
- The system currently supports only English license plates; other languages require additional OCR configuration.
- Duplicate license plates are avoided using a global set, but this may lead to missed detections if the same plate appears on different vehicles.

#### Future Improvements
- Add confidence-based filtering for OCR results to reduce false positives.
- Support multiple languages by configuring EasyOCR for additional languages.
- Explore alternative OCR libraries like `pytesseract` or PaddleOCR for better accuracy or performance.
- Save the output video with annotations for later review.
- Integrate a web interface for uploading videos and viewing results.
- Optimize performance further by offloading OCR to a separate thread or using a lighter model.

#### Setup and Execution
The project setup and execution steps are detailed in the `README.md` (as provided in the previous response). To summarize:
1. Extract `data.zip` to `license_plate/data/`.
2. Install dependencies using `requirements.txt`.
3. Train the model with `python src/train.py`.
4. Run detection with `python src/detect.py`.

---

### Summary

**Project**: A License Plate Detection and Recognition System using YOLOv8 and EasyOCR.  
**Code**: Consists of `train.py` (model training), `detect.py` (real-time detection), `utils.py` (helper functions), and `data.yaml` (dataset configuration).  
**Libraries**: `easyocr`, `numpy`, `opencv-python`, `pandas`, `ultralytics`, and their dependencies (`torch`, `pillow`, etc.).  
**Technologies**: YOLOv8 (object detection), EasyOCR (OCR), OpenCV (image processing), PyTorch (deep learning), Pandas (data handling).  
