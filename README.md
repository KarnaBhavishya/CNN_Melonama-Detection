# CNN-Based Melanoma Detection from Skin Lesion Images
**Final Year Project — Batch 14**

An end-to-end Deep Learning system for automated melanoma diagnosis from dermoscopic and skin lesion photographs. The system classifies skin lesions into **Benign** or **Malignant** categories with real-time confidence scores and provides both an interactive **Flask Web Interface** and a **Desktop GUI Application**.

---

## 📌 Features

- **Automated Deep Learning Classification**: Custom multi-layer Convolutional Neural Network (CNN) trained with Batch Normalization and Dropout for robust feature extraction.
- **Dual User Interfaces**:
  - **Web Application (Flask)**: Modern Bootstrap & Chart.js dashboard for uploading skin lesion images, viewing predictions, confidence distributions, and training history metrics.
  - **Desktop GUI (Tkinter)**: Lightweight standalone desktop app for instant image selection and diagnosis.
- **Fast Inference**: Real-time evaluation of uploaded lesions with confidence percentages.
- **One-Click Launchers**: Windows batch scripts (`run.bat` and `run_gui.bat`) for easy execution without complex terminal commands.

---

## 📁 Project Structure

```text
Final Year Project Batch-14(CNN-Based Melanoma Detection from Skin Lesion Images)/
│
├── env/                                    # Python 3.11 Virtual Environment
├── run.bat                                 # 1-Click launcher for Web Application
├── run_gui.bat                             # 1-Click launcher for Desktop GUI
├── README.md                               # Project documentation
│
└── Final Year Project Batch-14(...)/
    ├── app.py                              # Flask Web Application backend
    ├── predict.py                          # Tkinter Desktop GUI script
    ├── train.py                            # CNN Model training script
    ├── cnn_skin_lesion_high_acc.h5         # Pre-trained CNN model weights
    ├── 14_mel_requirements.txt             # Python package dependencies
    │
    ├── templates/
    │   ├── index.html                      # Upload & Home page
    │   └── result.html                     # Prediction result & Chart dashboard
    │
    ├── static/
    │   ├── uploads/                        # Temporary uploaded images
    │   └── charts/                         # Model performance visualizations
    │
    ├── dataset/
    │   ├── train/                          # Training dataset (benign / malignant)
    │   └── test/                           # Testing dataset (benign / malignant)
    │
    └── data/
        └── history.csv                     # Training history logs
```

---

## 🛠️ Tech Stack & Requirements

- **Programming Language**: Python 3.11 *(Recommended for TensorFlow binary wheel compatibility on Windows)*
- **Deep Learning Framework**: TensorFlow / Keras 2.20.0
- **Web Framework**: Flask 3.1.2, Werkzeug, Jinja2
- **Image Processing & Math**: NumPy 2.0.2, Pillow 11.3.0
- **Visualization**: Matplotlib 3.9.4, Chart.js, Bootstrap 5

---

## 🚀 How to Run

### Method 1: One-Click Launch (Recommended for Windows)

- **To run the Web Application**:  
  Double-click **`run.bat`**.  
  This starts the Flask server and automatically opens **http://127.0.0.1:5000/** in your default web browser.

- **To run the Desktop GUI**:  
  Double-click **`run_gui.bat`**.  
  This opens the Tkinter desktop window for selecting images and running predictions.

---

### Method 2: Running via Terminal / PowerShell

1. Open PowerShell or Command Prompt.
2. Navigate to the project directory:
   ```powershell
   cd "c:\Users\bhavi\Downloads\Final Year Project Batch-14(CNN-Based Melanoma Detection from Skin Lesion Images)\Final Year Project Batch-14(CNN-Based Melanoma Detection from Skin Lesion Images)"
   ```
3. Start the **Flask Web App**:
   ```powershell
   & "..\env\Scripts\python.exe" app.py
   ```
   Then open your browser and navigate to: **http://127.0.0.1:5000/**

4. *Or* start the **Desktop GUI**:
   ```powershell
   & "..\env\Scripts\python.exe" predict.py
   ```

---

## 🧪 Testing the Model

You can test the system using sample images provided in the dataset:
- **Benign sample**: `dataset/test/benign/melanoma_10000.jpg`
- **Malignant sample**: `dataset/test/malignant/`

1. Click **Choose File** / **Select Image**.
2. Select any lesion image from the test dataset or your computer.
3. Click **Predict**.
4. View the diagnosis (**BENIGN** or **MALIGNANT**), along with confidence metrics and comparative charts.

---

## 🧠 Model Architecture & Training

The CNN architecture implemented in `train.py` features:
- **Input Layer**: 128x128x3 RGB images normalized to `[0, 1]`
- **Feature Extraction**:
  - 4x Convolutional blocks (`32`, `64`, `128`, and `256` filters with 3x3 kernels and ReLU activation)
  - Batch Normalization after each Conv layer for stable convergence
  - 2x2 Max Pooling layers to downsample spatial dimensions
- **Classification Head**:
  - Flatten layer converting 3D feature maps to 1D
  - Dense layer (256 units, ReLU activation)
  - Dropout (0.5) to prevent overfitting
  - Dense output layer (1 unit, Sigmoid activation)
- **Loss & Optimizer**: Binary Cross-Entropy loss optimized with Adam
- **Callbacks**: `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau`

To retrain the model with your own dataset:
```powershell
& "..\env\Scripts\python.exe" train.py
```

---

## 📋 License & Disclaimer

This software is developed as an academic project (**Batch-14**). It is intended for educational, research, and preliminary screening support purposes and should not be used as a replacement for professional clinical dermatological diagnosis.
