
# Sign Language Recognition using Transformer Model

## Overview

This project implements a **sign language recognition system** using a **Transformer-based deep learning model**. It processes sign language gesture videos, extracts keypoints using **MediaPipe**, and classifies them using a custom Transformer architecture.

---

## Key Features

-  MediaPipe-based keypoint extraction (body, hands, face)
- Custom decoder-only Transformer model
- Training and evaluation pipeline with detailed metrics
- Single-video and full-dataset testing modes
- Confusion matrix and classification report visualizations

---

## Project Structure

```
sign-language-recognition/
├── data.py               # Keypoint extraction and preprocessing
├── model.py              # Transformer model architecture
├── train.py              # Training, evaluation, and visualization
├── run.py                # Entry point script
├── sign_transformer.pth  # Saved trained model (auto-generated)
├── confusion_matrix.png  # Confusion matrix (auto-generated)
└── README.md             # Project documentation
```

---

## Dependencies

This project requires Python 3.7+ and the following libraries:

```bash
pip install torch mediapipe opencv-python scikit-learn numpy tqdm matplotlib seaborn
```

---

## Dataset Preparation

1. Create a folder called `Words/` in the root directory.
2. Add your sign language videos to the `Words/` folder.
3. Name each video file after the gesture it represents. For example:

```
Words/
├── hello.mp4
├── thanks.mp4
├── body wash.mp4
```

Videos should be in `.mp4` format and limited to **30 frames** for best compatibility.

---

## Configuration

You can modify these parameters in `train.py`:

```python
VIDEO_DIR = "Words"                  # Path to video folder
SINGLE_TEST_VIDEO = "body wash.mp4" # File for single video testing
MAX_FRAMES = 30                     # Max frames per video
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 200                        # Number of training epochs
BATCH_SIZE = 2                      # Training batch size
MODEL_DIM = 512                     # Transformer dimension
NUM_HEADS = 4                       # Number of attention heads
NUM_LAYERS = 2                      # Number of Transformer layers
MODEL_PATH = "sign_transformer.pth"# Model save/load path
```

---

## How to Run

### 1. Train the Model

```bash
python run.py
```

This will:

- Load and preprocess videos from the `Words/` directory
- Extract keypoints using MediaPipe Holistic
- Train the Transformer model
- Save the trained model to `sign_transformer.pth`

If a trained model already exists, training will be skipped.

### 2. Evaluate the Model

After training, the script will automatically:

- Evaluate the model on the full dataset
- Generate a **confusion matrix** (`confusion_matrix.png`)
- Print a **classification report**
- Show per-video predictions

### 3. Single Video Prediction

It will also test the model on the `SINGLE_TEST_VIDEO` and print the predicted class.

---

##  Key Components

### 1. `data.py`: Keypoint Extraction

- Uses **MediaPipe Holistic** to extract:
  - 33 body pose landmarks (with visibility)
  - 21 left hand landmarks
  - 21 right hand landmarks
  - 468 face landmarks  
- Output per frame: **1662 features**

---

### 2. `model.py`: Transformer Model

- Decoder-only Transformer architecture
- Includes:
  - Positional Encoding
  - Multi-head Self-Attention
  - Layer Normalization
  - Feedforward Layers
  - Final Classification Head

---

### 3. `train.py`: Training & Evaluation

- Training loop with loss tracking
- Evaluation with precision, recall, and F1-score
- Confusion matrix and visualizations
- Single video testing support

---

### 4. `run.py`: Main Script

- Manages full pipeline execution
- Calls all modules in the correct order
- Trains the model if not already trained
- Performs evaluations and predictions

---

##  Results

After running the full pipeline, you’ll get:

- Training loss curve (printed per epoch)
- Confusion matrix image: `confusion_matrix.png`
- Classification report (precision, recall, F1-score)
- True vs Predicted labels for each video
- Prediction for a specific test video

---

##  Customization Options

| Feature              | How to Customize                      |
|----------------------|----------------------------------------|
| Model Architecture   | Modify `MODEL_DIM`, `NUM_HEADS`, etc. in `model.py` |
| Training Parameters  | Adjust `EPOCHS`, `BATCH_SIZE`, etc. in `train.py` |
| Input Length         | Change `MAX_FRAMES` in `train.py` and `data.py` |
| Visualizations       | Edit plotting functions in `train.py` |

---

##  Troubleshooting

| Problem                         | Solution |
|----------------------------------|----------|
| MediaPipe not working           | Ensure OpenCV and MediaPipe are correctly installed |
| CUDA Out of Memory              | Lower batch size or reduce `MAX_FRAMES` |
| Keypoint extraction errors      | Ensure videos have clear, visible signers; check MediaPipe compatibility |
| Video not predicting correctly  | Check if the video follows the expected format and frame count |

---

##  Future Enhancements

-  Real-time recognition with webcam
-  Sequence-to-sequence model for sentence-level recognition
-  Data augmentation for more robust learning
-  Live web app integration for demos

---

##  License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute!
# signlang-vision-transformer # Creates a README file
