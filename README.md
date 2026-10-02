# Human Action Recognition with LRCN

This project implements a Long-term Recurrent Convolutional Network (LRCN) to classify human actions from the [KTH Human Action Dataset](https://www.csc.kth.se/cvap/actions/). It also includes a live webcam inference script for real-time action recognition.

## 🧠 Architecture
The model uses a custom lightweight **CNN** (with BatchNorm) to extract spatial features from each frame and an **LSTM** to learn the temporal sequence of the action.

The model is trained on 4 classes:
- Walking
- Running
- Handwaving
- Handclapping

## 📂 Project Structure

```text
├── Dataset/                     # KTH Video Dataset (Needs to be downloaded)
│   ├── walking/                 # .avi files for walking
│   ├── running/                 # .avi files for running
│   ├── handwaving/              # .avi files for handwaving
│   └── handclapping/            # .avi files for handclapping
├── LRCN_Training.ipynb          # Jupyter Notebook for training and evaluation
├── LRCN_Training.py             # Standalone Python script for training
├── webcam_inference.py          # Real-time webcam inference script
├── best_lrcn_model.pth          # Saved trained model weights
├── training_curves.png          # Plot of training/validation loss and accuracy
└── .gitignore                   
```

## ⚙️ Setup and Installation

1. **Clone the repository** (if applicable) or copy the files to your local system.
2. **Install dependencies**:
   Ensure you have Python 3.8+ installed. Install the required libraries using pip:
   
   ```bash
   pip install torch torchvision opencv-python numpy matplotlib
   ```

   > *Note: If you have a CUDA-capable GPU, install the CUDA version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) for faster training and inference.*

3. **Download the Dataset**:
   Download the KTH dataset and extract the `.avi` files into a `Dataset` folder in the root directory. Organize them into four subfolders: `walking`, `running`, `handwaving`, and `handclapping` as shown in the project structure above.

## 🚀 Usage

### 1. Training the Model
You can train the model either using the provided Jupyter Notebook or the Python script. The model automatically splits the data subject-wise (Train: subjects 11-18, Validation: subjects 19-25, 1, 4, Test: subjects 22, 2-10).

**Using Python Script:**
```bash
python LRCN_Training.py
```
This will train the model for 40 epochs, save the best weights to `best_lrcn_model.pth`, generate training plots, and print out the test set evaluation and confusion matrix.

### 2. Live Webcam Inference
Once the model is trained (or if you already have `best_lrcn_model.pth`), you can test it on yourself in real-time!

```bash
python webcam_inference.py
```

- A window will pop up showing your webcam feed.
- Ensure your upper or full body is visible in the frame.
- **Controls**:
  - Press `Q` to quit.
  - Press `R` to reset the temporal frame buffer.
