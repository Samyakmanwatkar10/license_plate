# License Plate Detection and Recognition Project

## Overview

This project implements an automated license plate detection and recognition system using YOLOv8 for object detection and EasyOCR for optical character recognition (OCR). The system detects vehicles and license plates in a video feed, extracts the license plate text, and saves the results to a CSV file with timestamps. The project includes training a YOLOv8 model on a custom dataset and performing real-time detection on a video file (`demo.mp4`).

### Key Features

- Detects vehicles (cars) and license plates using YOLOv8.
- Extracts license plate text using EasyOCR with image preprocessing (denoising, sharpening, contrast enhancement, thresholding).
- Tracks vehicles across frames using YOLOv8's built-in tracking (`botsort.yaml`).
- Saves detected license plate text and timestamps to a CSV file (`outputs/results.csv`).
- Displays bounding boxes and extracted text on the video feed in real-time.

## Project Structure

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

- `data/`: Contains the dataset (extracted from `data.zip`) with training, validation, and test sets, along with `data.yaml` for YOLOv8 configuration.
- `src/`: Contains the Python scripts for training and detection.
  - `train.py`: Trains the YOLOv8 model on the custom dataset.
  - `detect.py`: Performs vehicle and license plate detection on `demo.mp4`, extracts text, and saves results.
  - `utils.py`: Helper functions for OCR, saving to CSV, and drawing bounding boxes.
- `demo.mp4`: Input video file for detection.
- `outputs/`: Stores training outputs (`runs/`) and detection results (`results.csv`).
- `requirements.txt`: Lists the required Python libraries.
- `README.md`: Project documentation (this file).

## Libraries Used

The project relies on the following Python libraries:

- `easyocr==1.7.1`: For optical character recognition (OCR) to extract text from license plates.
- `numpy==1.26.4`: For numerical operations (e.g., array manipulation in OpenCV and EasyOCR).
- `opencv-python==4.10.0.84`: For image processing, video handling, and drawing bounding boxes.
- `pandas==2.2.2`: For saving detection results to a CSV file.
- `ultralytics==8.2.58`: For YOLOv8 model training and inference.

These libraries have additional dependencies (e.g., `torch`, `pillow`, `requests`) that will be automatically installed when you install the above libraries using `requirements.txt`. The full list of dependencies can be found in `requirements.txt`.

## Code Overview

### `train.py`

Trains a YOLOv8 model on the custom dataset to detect license plates.

- **Input**: `data.yaml` (dataset configuration), pretrained YOLOv8 nano model (`yolov8n.pt`).
- **Output**: Trained model weights saved to `outputs/runs/license_plate/weights/best.pt`.
- **Key Parameters**:
  - `epochs=50`: Number of training epochs.
  - `imgsz=640`: Image size for training.
  - `batch=16`: Batch size.

### `detect.py`

Performs real-time detection on a video feed, extracts license plate text, and saves results.

- **Input**: Video file (`demo.mp4`) or webcam feed.
- **Output**: Displays video with bounding boxes and extracted text; saves results to `outputs/results.csv`.
- **Key Features**:
  - Uses YOLOv8 for vehicle detection (`yolov8n.pt`) and license plate detection (`best.pt`).
  - Uses EasyOCR for text extraction.
  - Implements frame skipping for OCR to improve performance.
  - Maintains a target frame rate of 30 FPS.

### `utils.py`

Contains helper functions for OCR, saving results, and drawing bounding boxes.

- **Functions**:
  - `get_license_plate_text()`: Preprocesses the license plate image and extracts text using EasyOCR.
  - `save_to_csv()`: Saves extracted license plate text and timestamps to a CSV file, avoiding duplicates.
  - `draw_boxes()`: Draws bounding boxes and labels on video frames.

## Setup and Installation

### Prerequisites

- Python 3.8 or higher.
- A system with a GPU (optional but recommended for faster training and inference with YOLOv8 and EasyOCR).
- `demo.mp4` video file in the project root directory (`license_plate/`).

### Steps to Run the Project

1. **Clone or Set Up the Project Directory**: Ensure your project directory matches the structure shown above. If you’re using a repository, clone it:

   ```bash
   git clone [<repository-url>](https://github.com/Samyakmanwatkar10/license_plate.git)
   cd license_plate
   ```

2. **Create a Virtual Environment** (optional but recommended):

   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install Dependencies**: Install the required libraries using the provided `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

4. **Extract the Dataset**: The dataset is provided in `data.zip`. Extract it to the `license_plate/data/` directory:

   ```bash
   unzip data.zip -d license_plate/data/
   ```

   After extraction, the `data/` directory should contain:

   ```
   data/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── valid/
   │   ├── images/
   │   └── labels/
   ├── test/
   │   ├── images/
   │   └── labels/
   └── data.yaml
   ```

5. **Update** `train.py` **with the Path to** `data.yaml`: Open `src/train.py` and set the `data` parameter to the path of `data.yaml`. Update the line:

   ```python
   data="",
   ```

   to:

   ```python
   data="../data/data.yaml",
   ```

   - The path `../data/data.yaml` is relative to `train.py` (located in `src/`), pointing to `license_plate/data/data.yaml`.

6. **Update** `utils.py` **with Dynamic CSV Path**: Open `src/utils.py` and update the `csv_path` in `save_to_csv()` to use a dynamic path. Replace:

   ```python
   csv_path="outputs/detect.csv"
   ```

   with:

   ```python
   script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of utils.py (src/)
   project_root = os.path.dirname(script_dir)  # Parent directory (license_plate/)
   csv_path=os.path.join(project_root, "outputs", "results.csv")
   ```

   This ensures the CSV path is portable across systems and matches the expected output location (`license_plate/outputs/results.csv`).

7. **Train the Model**: Run the training script to train the YOLOv8 model on your dataset:

   ```bash
   cd src
   python train.py
   ```

   - The trained model weights will be saved to `outputs/runs/license_plate/weights/best.pt`.

8. **Run Detection**: Run the detection script to process the `demo.mp4` video:

   ```bash
   cd src
   python detect.py
   ```

   - The script will display the video with detected license plates and save the results to `outputs/results.csv`.
   - Press `q` to quit the video display.

## Dataset Extraction

The dataset is provided in `data.zip`, which contains the following structure:

```
data/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

To extract the dataset:

1. Place `data.zip` in the `license_plate/` directory.

2. Extract it to the `data/` directory:

   ```bash
   unzip data.zip -d license_plate/data/
   ```

3. Verify that `data.yaml` and the `train/`, `valid/`, and `test/` directories are correctly placed in `license_plate/data/`.

**Note**: Ensure `data.yaml` correctly points to the dataset directories. It should look like this:

```yaml
path: .
train: train/images
val: valid/images
test: test/images
nc: 1
names: ['license_plate']
```

- `path: .` points to `license_plate/data/`, the directory containing `train/`, `valid/`, and `test/`.

## Project Execution Video

Below is a video demonstrating the project in action, showing license plate detection and text extraction on `demo.mp4`:

[Watch Project Demo](https://drive.google.com/file/d/1kvpXZtPyHyYHnqT_MhFvgVxci_wuFJNZ/view?usp=drive_link)

## Additional Information

### Assumptions

- The input video (`demo.mp4`) contains vehicles with visible license plates.
- The system has a GPU available for faster inference (if not, EasyOCR will fall back to CPU, which is slower).
- The dataset in `data.zip` is properly annotated with license plate bounding boxes in YOLO format.

### Limitations

- OCR accuracy depends on the quality of the license plate image (e.g., lighting, angle, resolution).
- Real-time performance may degrade on systems without a GPU or with high-resolution videos.
- The system assumes license plates contain English alphanumeric characters (EasyOCR is configured for `en` language).

### Future Improvements

- Add support for multiple languages in OCR by configuring EasyOCR for additional languages.
- Implement confidence-based filtering for OCR results to reduce false positives.
- Explore alternative OCR libraries like `pytesseract` or PaddleOCR for better accuracy or performance.
- Add options to save the output video with annotations.
- Integrate a web interface for uploading videos and viewing results.

### Troubleshooting

- **Error: "Could not open video source"**: Ensure `demo.mp4` is in the `license_plate/` directory. If using a different video, update the `input_source` in `detect.py`.
- **Error: "Dataset images not found"**: Verify that `data.yaml` paths are correct and that the `train/`, `valid/`, and `test/` directories contain images and labels.
- **Slow Performance**: If detection is slow, reduce the `scale_factor` in `detect.py` (e.g., from `0.3` to `0.2`) or increase `ocr_frame_skip` (e.g., from `10` to `20`).

## License

This project is for educational and research purposes. Ensure you have the necessary permissions to use the dataset and video files.

---

For any questions or contributions, feel free to reach out!
