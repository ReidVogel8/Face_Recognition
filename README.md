# Face Recognition

A real-time face recognition system built with OpenCV and Python. This project uses machine learning to detect and recognize faces from images, webcam streams, or video files.

## Features

- **Face Detection**: Automatically detect faces in images or video streams
- **Face Training**: Train the model with custom face datasets
- **Real-time Recognition**: Recognize faces in real-time using webcam
- **Persistent Model**: Saves trained model and labels for reuse

## Requirements

- Python 3.x
- OpenCV with contrib modules

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ReidVogel8/Face_Recognition.git
cd Face_Recognition
```

2. Install required dependencies:
```bash
pip install opencv-contrib-python
```

## Project Structure

```
Face_Recognition/
├── Images/
│   └── Reid/              # Training images organized by person name
├── faceRecognition.py     # Main script for face recognition
├── trainer.py             # Script to train the face recognizer
├── trainer.yml            # Trained model file
├── labels.pickle          # Saved label mappings
└── README.md
```

## Usage

### 1. Prepare Training Data

Organize your training images in the `Images/` directory. Create a separate folder for each person:

```
Images/
├── Person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Person2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
```

**Tip**: Use multiple images (10-20) per person with different angles, expressions, and lighting conditions for better accuracy.

### 2. Train the Model

Run the trainer script to process your images and create the recognition model:

```bash
python trainer.py
```

This will:
- Detect faces in all training images
- Extract facial features
- Train the LBPH (Local Binary Patterns Histograms) face recognizer
- Save the trained model to `trainer.yml`
- Save label mappings to `labels.pickle`

### 3. Run Face Recognition

Execute the main script to start recognizing faces:

```bash
python faceRecognition.py
```

This will:
- Open your webcam
- Detect faces in real-time
- Recognize and label known faces
- Display the video feed with recognition results

Press `q` or `ESC` to quit the application.

## How It Works

1. **Face Detection**: Uses Haar Cascade classifiers to detect faces in images
2. **Feature Extraction**: Extracts facial features using OpenCV's algorithms
3. **Training**: The LBPH face recognizer learns patterns from the training images
4. **Recognition**: Compares detected faces against trained patterns to identify individuals
5. **Confidence Score**: Each recognition comes with a confidence level (lower is better)

## Customization

### Adjust Recognition Threshold

Modify the confidence threshold in `faceRecognition.py` to make recognition more or less strict:

```python
# Lower threshold = stricter matching
# Higher threshold = more lenient matching
confidence_threshold = 100  # Default value
```

### Change Camera Source

If you have multiple cameras, change the camera index in `faceRecognition.py`:

```python
cap = cv2.VideoCapture(0)  # 0 for default camera, 1 for external camera, etc.
```

## Troubleshooting

### Camera Not Opening
- Ensure your webcam is properly connected
- Check if another application is using the camera
- Try changing the camera index in the code

### Poor Recognition Accuracy
- Add more training images per person (15-20 recommended)
- Ensure good lighting conditions in training images
- Include images with different expressions and angles
- Retrain the model after adding new images

### Import Errors
- Make sure you install `opencv-contrib-python`, not just `opencv-python`
- Verify Python version compatibility (Python 3.6+ recommended)

## Technical Details

- **Face Detection**: Haar Cascade Frontal Face Classifier
- **Face Recognition Algorithm**: LBPH (Local Binary Patterns Histograms)
- **Model Persistence**: OpenCV FileStorage format (.yml)
- **Label Storage**: Python pickle format


## Acknowledgments

- OpenCV community for the excellent computer vision library
- Contributors to face detection and recognition algorithms

## Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [Face Recognition with OpenCV](https://docs.opencv.org/master/da/d60/tutorial_face_main.html)

---

**Note**: This project is for educational purposes.
