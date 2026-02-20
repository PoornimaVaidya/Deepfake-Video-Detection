**AI-Based Deepfake Video Detection System**

A deep learning pipeline that classifies videos as Real or Fake (Deepfake) using a hybrid Xception + Bidirectional LSTM architecture — achieving 95% test accuracy.

**📌 What is this Project?**
Deepfakes are AI-generated videos where a person's face or voice is manipulated to appear as someone else. This project builds an end-to-end deepfake detection system that:

Extracts key frames from videos
Applies data augmentation to improve generalization
Uses Xception (a powerful CNN) to extract spatial features from each frame
Feeds the sequence of frame features into a Bidirectional LSTM to capture temporal patterns across frames
Classifies the video as REAL or FAKE with a confidence score

This was built using Python, TensorFlow/Keras, and OpenCV.


**🏗️ Project Architecture**
deepfake-detection/
│
├── Data_Preparation.py       # Extract frames from videos, split into train/val/test
├── Data_Augmentation.py      # Augment training frames (flip, rotate, zoom, brightness)
├── Model_Architecture.py     # Define Xception + BiLSTM model
├── Model_Training.py         # Train the model with callbacks and class weights
├── Real_Time_Detection.py    # Load trained model and predict on new videos
│
├── X_train.npy / y_train.npy          # Training data (generated)
├── X_val.npy   / y_val.npy            # Validation data (generated)
├── X_test.npy  / y_test.npy           # Test data (generated)
├── X_train_augmented.npy              # Augmented training data (generated)
├── deepfake_detection_model.keras     # Best checkpoint (generated)
└── deepfake_detection_model_final.keras  # Final saved model (generated)


**🔁 Workflow**

Raw Videos (Real & Fake)
        │
        ▼
[Data_Preparation.py]
  - Extract 10 frames per video (299×299 px)
  - Normalize pixel values (÷ 255)
  - One-hot encode labels
  - Stratified train (70%) / val (15%) / test (15%) split
        │
        ▼
[Data_Augmentation.py]
  - Horizontal flip, rotation, zoom, brightness shift
  - Doubles the training dataset
        │
        ▼
[Model_Architecture.py]
  - TimeDistributed(Xception)  → spatial features per frame
  - TimeDistributed(Flatten)
  - Dropout(0.5)
  - Bidirectional LSTM(128)    → temporal patterns across frames
  - Dropout(0.5)
  - Dense(64, relu)
  - Dense(2, softmax)          → REAL or FAKE
        │
        ▼
[Model_Training.py]
  - Adam optimizer (lr=1e-4), categorical crossentropy
  - Class weights for imbalanced data
  - ModelCheckpoint + ReduceLROnPlateau callbacks
  - 10 epochs, batch size 2
        │
        ▼
[Real_Time_Detection.py]
  - Load saved model
  - Extract & preprocess frames from new video
  - Predict: REAL / FAKE + confidence score


**⚙️ Installation**
Prerequisites

Python 3.8+
pip

1. Clone the Repository
git clone https://github.com/PoornimaVaidya/Deepfake-Video-Detection.git
cd deepfake-detection

2. Create a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

3. Install Dependencies
pip install tensorflow opencv-python numpy scikit-learn tqdm

Note for Apple Silicon (M1/M2) Mac users: Use tensorflow-macos and tensorflow-metal instead of tensorflow.

pip install tensorflow-macos tensorflow-metal



**🗂️ Dataset Setup**
This project was trained on the FaceForensics++ / DFD dataset.

Download your dataset (Real and Fake video folders)
Organize it as:

Data_set/
├── Real/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── Fake/
    ├── fake_video1.mp4
    ├── fake_video2.mp4
    └── ...

Update the paths in Data_Preparation.py:

pythonREAL_PATH = "/path/to/your/Real"
FAKE_PATH = "/path/to/your/Fake"


**▶️ How to Run**
Run the scripts in order:
Step 1 — Prepare Data

python Data_Preparation.py
Outputs: X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy, failed_videos.txt

Step 2 — Augment Training Data
bashpython Data_Augmentation.py
Outputs: X_train_augmented.npy, y_train_augmented.npy

Step 3 — Train the Model
bashpython Model_Training.py
Outputs: deepfake_detection_model.keras (best checkpoint), deepfake_detection_model_final.keras

Step 4 — Run Detection on a Video
Update the video paths in Real_Time_Detection.py:
python real_sample_path = "/path/to/your/real_video.mp4"
fake_sample_path = "/path/to/your/fake_video.mp4"

Then run:
python Real_Time_Detection.py

**📊 Sample Output**
Real Video Prediction:
[INFO] Processing: /path/to/real_video.mp4
[INFO] Input shape to model: (1, 10, 299, 299, 3)
Prediction: REAL (Confidence: 0.94)

Fake Video Prediction:
[INFO] Processing: /path/to/fake_video.mp4
[INFO] Input shape to model: (1, 10, 299, 299, 3)
Prediction: FAKE (Confidence: 0.97)


**🧠 Model Architecture Summary**
LayerDetailsInput(10, 299, 299, 3) — 10 frames per videoTimeDistributed(Xception)Pretrained on ImageNet, no top layerTimeDistributed(Flatten)Flatten spatial features per frameDropout(0.5)RegularizationBidirectional LSTM(128)Temporal feature extractionDropout(0.5)RegularizationDense(64, ReLU)Fully connected layerDense(2, Softmax)Output: REAL or FAKE

Optimizer: Adam (lr = 1e-4)
Loss: Categorical Crossentropy
Test Accuracy: ~95%


**🐛 Common Errors & Fixes**
❌ Failed to open video / Frames not extracted
Cause: Corrupted video file or unsupported codec.
Fix: Check failed_videos.txt for the list of failed videos. Re-download or skip them. Ensure video extensions are .mp4, .avi, .mov, or .mkv.

❌ OOM (Out of Memory) during training
Cause: Batch size too large for your GPU/RAM.
Fix: In Model_Training.py, reduce batch size:
pythonbatch_size=1  # Try 1 if 2 causes OOM

❌ ModuleNotFoundError: No module named 'tensorflow'
Fix:
bashpip install tensorflow
For Mac M1/M2:
bashpip install tensorflow-macos tensorflow-metal

❌ ValueError: Input 0 of layer is incompatible with the layer
Cause: Input shape mismatch — model expects (10, 299, 299, 3) but frames have a different size/count.
Fix: Ensure FRAME_COUNT=10 and OUTPUT_FRAME_SIZE=(299, 299) are consistent across all scripts.

❌ Model file not found in Real_Time_Detection.py
Cause: The model hasn't been trained yet, or the save path is incorrect.
Fix: Run Model_Training.py first, then update the path in Real_Time_Detection.py:
pythonloaded_model = load_model('/correct/path/to/deepfake_detection_model_final.keras')

❌ Slow training / pydot error when saving model diagram
Cause: pydot and graphviz are optional dependencies.
Fix: Install them if needed, or safely ignore — the model will still train:
bashpip install pydot
brew install graphviz   # macOS
sudo apt install graphviz  # Ubuntu

❌ WARNING: Only N frames available
Cause: The video is shorter than 10 frames.
Fix: The code pads with black frames automatically. For best accuracy, use videos with at least 10 frames.


**🛠️ Tech Stack**

Python 3.8+
TensorFlow / Keras — model building & training
OpenCV — video frame extraction
NumPy — array processing
scikit-learn — train/val/test split, class weights
tqdm — progress bars

**📜 License**
This project is for academic and research purposes.

**🙋 Author**
Poornima Vaidya
MS Computer Science — Advanced AI
Spring 2025



