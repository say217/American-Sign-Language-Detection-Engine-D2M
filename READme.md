# Sign Language Detection using 

<img width="1036" height="624" alt="image" src="https://github.com/user-attachments/assets/526e48ae-115d-4c5a-aca9-b1f6b11040db" />


## Overview

This repository presents an end-to-end computer vision pipeline for real-time American Sign Language (ASL) alphabet detection using the Ultralytics YOLOv8 object detection framework. The system is designed to accurately detect and classify hand gestures representing the 26 English alphabet letters (A–Z) from images and live video streams. To improve detection stability and reduce false positives, the project integrates Google's MediaPipe Hands framework for robust hand landmark estimation and hand region localization before object detection. By combining MediaPipe with YOLOv8, the system achieves more reliable sign recognition under varying lighting conditions, hand orientations, and backgrounds.

The repository includes the complete workflow for dataset preparation, visualization, preprocessing, model training, evaluation, and inference. The entire training pipeline was developed and executed on Kaggle using GPU acceleration.

**Kaggle Notebook:**  
ASL https://www.kaggle.com/code/sayaksamanta/sign-language-detection

ISL https://www.kaggle.com/code/sayaksamanta/indian-sign-language-detction/notebook


---

# Dataset Information

The model is trained on an annotated American Sign Language (ASL) alphabet dataset containing hand gesture images for all 26 English letters. Each image is annotated using the YOLO object detection format, enabling the model to learn precise hand localization together with alphabet classification.

### Dataset Statistics

| Property | Value |
|----------|-------|
| Total Classes | 26 (A–Z) |
| Training Images | 504 |
| Validation Images | 144 |
| Testing Images | 72 |
| Annotation Format | YOLO Bounding Boxes |
| Label Format | `class_id x_center y_center width height` |

Each annotation file contains normalized bounding box coordinates describing the location of the hand sign inside the image.

---
| | |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/249e6166-68ee-417e-acce-968c6f9b8aca" width="100%" /> | <img src="https://github.com/user-attachments/assets/13a29601-f543-4311-a7e4-a9c196a4bb20" width="100%" /> |


# Project Pipeline

The complete detection pipeline consists of multiple stages, beginning from dataset preparation and ending with real-time sign prediction.

### 1. Dataset Configuration

The dataset is organized according to the Ultralytics YOLO directory structure. The `data.yaml` configuration file defines:

- Dataset paths
- Number of classes
- Class names
- Train, validation, and test splits

The training script automatically validates these configurations before training begins.

---
 
### 2. Data Exploration & Visualization

Before training, the repository visualizes randomly selected samples from the dataset. Bounding boxes are drawn using OpenCV, while Matplotlib is used to display labeled images for manual verification.

This stage helps verify:

- Correct annotations
- Class distributions
- Image quality
- Bounding box placement

---
| | |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/8ed53478-c49a-4c6f-a76d-9da505d408cd" width="100%" /> | <img src="https://github.com/user-attachments/assets/0f90079f-5071-430d-9845-d6d08b7620f4" width="100%" /> |
| <img src="https://github.com/user-attachments/assets/30ac0366-7d65-4f00-9ad4-d69cb27e3f79" width="100%" /> | <img src="https://github.com/user-attachments/assets/745bff28-a7a0-47dc-a227-55f4c6b8d5f0" width="100%" /> |


### 3. Dataset Configuration Export

To improve portability between Kaggle, Google Colab, and local environments, the dataset configuration is automatically converted into a custom `data_custom.yaml` file containing absolute dataset paths.

This eliminates path-related errors during training and deployment.

---

### 4. Hand Detection using MediaPipe

Before object detection, Google's MediaPipe Hands framework is used to detect the hand region in each frame.

MediaPipe provides:

- High-speed hand detection
- 21 hand landmarks
- Hand bounding box estimation
- Robust tracking across consecutive frames

The detected hand region is used to improve localization and reduce unnecessary background information before passing the frame into the YOLOv8 detector.

Advantages include:

- Faster inference
- Improved robustness
- Better detection in cluttered backgrounds
- More stable real-time predictions

---

### 5. YOLOv8 Model Training

The project fine-tunes the pre-trained **YOLOv8 Nano (`yolov8n.pt`)** model on the custom ASL dataset.

Transfer learning allows the detector to converge quickly while maintaining high detection accuracy even with a relatively small dataset.

Training includes:

- Automatic image resizing
- Data augmentation
- Mosaic augmentation
- Random flipping
- Batch optimization
- Validation after each epoch
- Best model checkpoint saving

Training Configuration:

| Parameter | Value |
|-----------|------|
| Model | YOLOv8 Nano |
| Pretrained Weights | yolov8n.pt |
| Epochs | 50 |
| Framework | Ultralytics YOLOv8 |
| Task | Object Detection |

---

### 6. Model Evaluation

After training, the model is evaluated on the validation and testing datasets using standard object detection metrics.

Evaluation includes:

- Precision
- Recall
- mAP@0.50
- mAP@0.50:0.95
- Loss curves
- Precision–Recall curves
- Confusion Matrix

These metrics provide a comprehensive understanding of model performance across all ASL alphabet classes.

---

### 7. Real-Time Inference

The trained model supports real-time inference using:

- Webcam
- Images
- Video files

During inference:

1. MediaPipe detects the hand.
2. The detected hand region is passed to YOLOv8.
3. YOLO predicts the alphabet class.
4. Bounding boxes, confidence scores, and predicted letters are displayed on the output frame.

The pipeline is optimized for low-latency prediction, making it suitable for interactive sign language recognition applications.

---

# Installation

Install all required Python packages using pip:

```bash
pip install ultralytics opencv-python mediapipe pyyaml matplotlib numpy pandas
```

---

# Hardware Requirements

The project can run on both CPU and GPU, although GPU acceleration is highly recommended for faster training and inference.

| Component | Requirement |
|-----------|-------------|
| Python | 3.10 or above |
| GPU | NVIDIA CUDA-compatible GPU (Tesla T4, RTX Series, A100 recommended) |
| RAM | 8 GB minimum (16 GB recommended) |
| Storage | At least 5 GB free disk space |

---

# Technologies Used

- Python
- Ultralytics YOLOv8
- MediaPipe Hands
- OpenCV
- NumPy
- Matplotlib
- Pandas
- PyYAML
- Kaggle Notebooks
- CUDA (GPU Acceleration)

---

# Repository Features

- End-to-end ASL alphabet detection
- YOLOv8-based object detection
- MediaPipe hand tracking integration
- Automatic dataset visualization
- Transfer learning with pretrained weights
- Real-time webcam inference
- GPU-accelerated training
- Custom dataset support
- Easy deployment with custom YAML configuration

---

# Future Improvements

Future extensions of this project include:

- Word-level sign language recognition
- Sentence generation using Large Language Models (LLMs)
- Temporal gesture recognition with video sequences
- Sign-to-text conversion
- Text-to-speech synthesis
- Multi-hand detection
- Mobile and edge device deployment using TensorRT or ONNX
- Integration with conversational AI assistants for real-time communication between sign language users and non-signers
