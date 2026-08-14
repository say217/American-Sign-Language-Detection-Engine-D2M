# Sign Language Detection using 



![](https://i.postimg.cc/28xMsjKk/Screenshot-2026-08-14-221230.png)

## Overview


This repository presents an integrated real-time Sign Language Recognition system supporting both Indian Sign Language (ISL) and American Sign Language (ASL). The system combines the Ultralytics YOLOv8 object detection framework, Google MediaPipe Hands, and a FastAPI-powered web application to provide an end-to-end sign language detection and recognition pipeline.

The system is designed to detect and classify hand gestures from both Indian and American Sign Language alphabets, enabling users to perform sign gestures through images, uploaded media, or live video streams. MediaPipe Hands is used for robust hand detection and landmark estimation, helping localize hand regions and improve recognition stability, while YOLOv8 performs gesture detection and classification.

The project is implemented as an integrated web-based application using FastAPI, connecting the computer vision models with a real-time user interface. The architecture supports multiple recognition workflows within a single platform, allowing users to select the appropriate sign language and perform real-time inference. This combination of YOLOv8 + MediaPipe + FastAPI provides a scalable foundation for sign language recognition across different hand orientations, backgrounds, and lighting conditions.

The repository also includes the complete workflow for dataset preparation, preprocessing, visualization, model training, evaluation, and inference for both ISL and ASL recognition. Model development and training were performed using GPU-accelerated environments, while the trained models are integrated into the FastAPI application for real-time deployment and inference.



### 🇺🇸 American Sign Language (ASL) Model

For real-time **American Sign Language (ASL)** recognition, our system uses the **YOLOv8 object detection framework** integrated with computer vision-based hand gesture processing. The model detects and classifies hand gestures representing ASL alphabet signs from live video streams. **MediaPipe Hands** is used for robust hand detection and landmark tracking, improving stability across different hand positions, orientations, and backgrounds. The trained model is integrated into the **FastAPI web application** for real-time inference and visualization.

**Kaggle Notebook:**  
ASL https://www.kaggle.com/code/sayaksamanta/sign-language-detection


---
| American Sign Language|  American Sign Language detection |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/249e6166-68ee-417e-acce-968c6f9b8aca" width="100%" /> | <img src="https://github.com/user-attachments/assets/13a29601-f543-4311-a7e4-a9c196a4bb20" width="100%" /> |


### 🇮🇳 Indian Sign Language (ISL) Model

To enable real-time interpretation of **Indian Sign Language (ISL)**, our system adopts a **Hybrid CNN–Transformer Detection Framework** that combines convolutional feature extraction with Transformer-based attention mechanisms for robust spatial and contextual representation of hand gestures.

The model is designed to improve recognition across variations in **hand orientation, background, lighting conditions, and gesture appearance**, making it suitable for real-time sign language interpretation.

**Kaggle Notebook:**  

ISL https://www.kaggle.com/code/sayaksamanta/indian-sign-language-detction/notebook


> ** Detailed ISL Model Architecture & Implementation:**  
>  [View the Indian Sign Language Model Repository](https://github.com/say217/Indian-Sign-Language-Detection-Engine-D3M)

```bash

# Clone the repository
git clone https://github.com/say217/Sign-Language-Detection-Engine-.git
cd Sign-Language-Detection-Engine-

# Create virtual environment
python -m venv venv

# Activate virtual environment — Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI application
uvicorn src.Main.run:app --reload

```