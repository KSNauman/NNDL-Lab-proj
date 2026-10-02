"""
LRCN Live Webcam Inference
==========================

Real-time human action recognition using the trained LRCN model.
Captures frames from the webcam, maintains a sliding window buffer,
and displays predictions overlaid on the video feed.

Controls:
  Q - Quit
  R - Reset frame buffer

Model: Custom CNN (BatchNorm) + LSTM trained on KTH dataset
Classes: walking, running, handwaving, handclapping
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import time


# ============================================================
# Model Definition (must match training)
# ============================================================
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return x


class LRCN(nn.Module):
    def __init__(self, num_classes=4, dropout=0.3):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = x.view(B, T, 128)
        output, (hidden, cell) = self.lstm(x)
        x = hidden[-1]
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(r"C:\NNDL Lab Proj\KTH")
MODEL_PATH = BASE_DIR / "best_lrcn_model.pth"

CLASSES = ["walking", "running", "handwaving", "handclapping"]
NUM_FRAMES = 16          # Number of frames the model expects
SAMPLE_INTERVAL = 0.24   # Sample a frame every 0.24 seconds
IMAGE_SIZE = 112         # Model input resolution
INFERENCE_INTERVAL = 0.3 # Run inference every 0.3 seconds (to keep UI smooth)

# Colors for each class (BGR for OpenCV)
CLASS_COLORS = {
    "walking":      (0, 200, 0),     # Green
    "running":      (0, 140, 255),   # Orange
    "handwaving":   (255, 100, 0),   # Blue
    "handclapping": (180, 0, 255),   # Purple
}


# ============================================================
# Load model
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = LRCN(num_classes=4, dropout=0.0)  # No dropout at inference
model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device, weights_only=True))
model.to(device)
model.eval()
print(f"Model loaded from: {MODEL_PATH}")


# ============================================================
# Preprocessing (same as training)
# ============================================================
def preprocess_frame(frame):
    """Convert a BGR webcam frame to model input format."""
    # BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize to 112x112
    frame_resized = cv2.resize(frame_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    # Normalize 0-255 -> 0-1
    frame_norm = frame_resized.astype(np.float32) / 255.0
    # HWC -> CHW
    frame_chw = np.transpose(frame_norm, (2, 0, 1))
    return frame_chw


# ============================================================
# Drawing helpers
# ============================================================
def draw_overlay(frame, pred_class, pred_conf, probs, buffer_count, fps):
    """Draw prediction overlay on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent dark panel at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    if pred_class is not None:
        color = CLASS_COLORS.get(pred_class, (255, 255, 255))

        # Main prediction text
        cv2.putText(frame, f"Action: {pred_class.upper()}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Confidence: {pred_conf:.1%}",
                    (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Probability bars on the right side
        bar_x = w - 250
        bar_y_start = 10
        for i, (cls_name, prob) in enumerate(zip(CLASSES, probs)):
            y = bar_y_start + i * 24
            cls_color = CLASS_COLORS.get(cls_name, (255, 255, 255))

            # Class label
            cv2.putText(frame, f"{cls_name}", (bar_x, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            # Bar background
            cv2.rectangle(frame, (bar_x + 100, y + 3), (bar_x + 230, y + 18),
                          (60, 60, 60), -1)

            # Bar fill
            bar_width = int(130 * prob)
            if bar_width > 0:
                cv2.rectangle(frame, (bar_x + 100, y + 3),
                              (bar_x + 100 + bar_width, y + 18), cls_color, -1)

            # Percentage
            cv2.putText(frame, f"{prob:.0%}", (bar_x + 233, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    else:
        cv2.putText(frame, f"Collecting frames... ({buffer_count}/{NUM_FRAMES})",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

    # FPS and buffer info at bottom
    cv2.putText(frame, f"FPS: {fps:.0f} | Buffer: {buffer_count}/{NUM_FRAMES} | Q=Quit R=Reset",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    return frame


# ============================================================
# Main webcam loop
# ============================================================
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        print("Make sure your webcam is connected and not in use by another app.")
        return

    # Try to set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam resolution: {actual_w}x{actual_h}")
    print(f"\nSampling Configuration:")
    print(f"  Target sampling interval: {SAMPLE_INTERVAL:.3f} seconds")
    print(f"  LRCN input sequence length: {NUM_FRAMES} frames")
    print(f"  Approximate temporal window: {SAMPLE_INTERVAL * NUM_FRAMES:.2f} seconds")
    print(f"\nControls:")
    print(f"  Q - Quit")
    print(f"  R - Reset frame buffer")
    print(f"\nStarting live inference...")

    frame_buffer = []           # Stores preprocessed frames
    last_sample_time = 0        # Last time a frame was added to the buffer
    pred_class = None
    pred_conf = 0.0
    probs = None
    last_inference_time = 0
    fps_counter = 0
    fps_time = time.time()
    display_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam")
            break

        # FPS calculation
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            display_fps = fps_counter
            fps_counter = 0
            fps_time = time.time()

        # Sample frames temporally
        current_time = time.time()
        if current_time - last_sample_time >= SAMPLE_INTERVAL:
            last_sample_time = current_time
            preprocessed = preprocess_frame(frame)
            frame_buffer.append(preprocessed)

            # Keep only the latest NUM_FRAMES
            if len(frame_buffer) > NUM_FRAMES:
                frame_buffer = frame_buffer[-NUM_FRAMES:]

        # Run inference when we have enough frames and enough time has passed
        current_time = time.time()
        if (len(frame_buffer) >= NUM_FRAMES and
                current_time - last_inference_time >= INFERENCE_INTERVAL):

            last_inference_time = current_time

            # Build input tensor: [1, 16, 3, 112, 112]
            input_frames = np.array(frame_buffer[-NUM_FRAMES:])
            input_tensor = torch.from_numpy(input_frames).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                probs_tensor = torch.softmax(logits, dim=1).cpu().numpy()[0]

            pred_idx = np.argmax(probs_tensor)
            pred_class = CLASSES[pred_idx]
            pred_conf = probs_tensor[pred_idx]
            probs = probs_tensor

        # Draw overlay
        display_frame = draw_overlay(
            frame, pred_class, pred_conf, probs,
            len(frame_buffer), display_fps
        )

        cv2.imshow("LRCN Live Action Recognition", display_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('r') or key == ord('R'):
            frame_buffer = []
            pred_class = None
            probs = None
            print("Buffer reset!")

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == "__main__":
    main()
