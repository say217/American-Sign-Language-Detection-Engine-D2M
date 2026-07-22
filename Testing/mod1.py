import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- CPU Optimization Settings ---
# Set thread limits to prevent CPU thread thrashing
torch.set_num_threads(os.cpu_count() or 4)
cv2.setNumThreads(4)

# 1. Setup Relative Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

model_path = os.path.join(PROJECT_DIR, ".model", "ALpha_SIGN_M89.pt")
img_path = os.path.join(PROJECT_DIR, "assets", "B15_jpg.rf.0edce7630140f543226a42ebee3e747b.jpg")

# 2. Load Model & Image
model = YOLO(model_path)
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"Could not load image at path: {img_path}")

# 3. Fast CPU Inference
# Specifying imgsz=640 dramatically boosts CPU speed without quality loss
results = model.predict(source=img, imgsz=640, conf=0.25, device='cpu', verbose=False)[0]

# Color Palette (BGR)
CYAN = (255, 255, 0)         # Brackets & Box Border
NEON_GREEN = (0, 255, 128)   # Accuracy Bar Fill
DARK_BG = (20, 20, 20)       # Tag Background
WHITE = (255, 255, 255)      # Bar Text Color
BLACK = (0, 0, 0)            # High contrast text outline

detected_signs = []

# 4. Draw HUD Overlay
for box in results.boxes:
    cls_id = int(box.cls[0])
    label = model.names[cls_id]
    conf = float(box.conf[0])
    conf_pct = int(conf * 100)
    detected_signs.append(f"{label} ({conf_pct}%)")

    xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
    box_w, box_h = xmax - xmin, ymax - ymin

    # --- A. Corner Brackets ---
    line_len = max(25, int(min(box_w, box_h) * 0.25))
    t = 3  # Clean stroke

    # Outer outline
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), CYAN, 1)

    # Brackets
    cv2.line(img, (xmin, ymin), (xmin + line_len, ymin), CYAN, t)
    cv2.line(img, (xmin, ymin), (xmin, ymin + line_len), CYAN, t)
    cv2.line(img, (xmax, ymin), (xmax - line_len, ymin), CYAN, t)
    cv2.line(img, (xmax, ymin), (xmax, ymin + line_len), CYAN, t)
    cv2.line(img, (xmin, ymax), (xmin + line_len, ymax), CYAN, t)
    cv2.line(img, (xmin, ymax), (xmin, ymax - line_len), CYAN, t)
    cv2.line(img, (xmax, ymax), (xmax - line_len, ymax), CYAN, t)
    cv2.line(img, (xmax, ymax), (xmax, ymax - line_len), CYAN, t)

    # --- B. Much Larger HUD Tag & Longer Accuracy Bar ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    lbl_scale, lbl_thick = 1.0, 2  # Increased label text size & weight
    (lbl_w, lbl_h), _ = cv2.getTextSize(label, font, lbl_scale, lbl_thick)

    # Expanded Bar Dimensions
    bar_height = 36  # Taller 36px bar
    bar_text = f"Accuracy: {conf_pct}%"
    bar_font_scale, bar_font_thick = 0.85, 2  # Larger score font size
    (bt_w, bt_h), _ = cv2.getTextSize(bar_text, font, bar_font_scale, bar_font_thick)

    pad_x, pad_y = 16, 14
    # Increased minimum width to 720px for a substantially longer bar length
    tag_w = max(lbl_w + (pad_x * 2), bt_w + (pad_x * 2), 720)
    tag_h = lbl_h + bar_height + (pad_y * 3)
    
    tag_y1 = max(ymin - tag_h - 8, 0)
    tag_y2 = tag_y1 + tag_h
    
    # Background Box
    cv2.rectangle(img, (xmin, tag_y1), (xmin + tag_w, tag_y2), DARK_BG, -1)
    cv2.rectangle(img, (xmin, tag_y1), (xmin + tag_w, tag_y2), CYAN, 2)

    # Class Label Text
    cv2.putText(img, label, (xmin + pad_x, tag_y1 + lbl_h + pad_y), 
                font, lbl_scale, CYAN, lbl_thick, cv2.LINE_AA)

    # --- C. Long Accuracy Bar & Internal Score ---
    bar_y1 = tag_y2 - pad_y - bar_height
    bar_y2 = tag_y2 - pad_y
    bar_max_w = tag_w - (pad_x * 2)
    bar_fill_w = int(bar_max_w * conf)

    # Bar background & filled green region
    cv2.rectangle(img, (xmin + pad_x, bar_y1), (xmin + pad_x + bar_max_w, bar_y2), (50, 50, 50), -1)
    cv2.rectangle(img, (xmin + pad_x, bar_y1), (xmin + pad_x + bar_fill_w, bar_y2), NEON_GREEN, -1)
    cv2.rectangle(img, (xmin + pad_x, bar_y1), (xmin + pad_x + bar_max_w, bar_y2), CYAN, 1)

    # Centered Score Text inside Bar
    text_x = xmin + pad_x + (bar_max_w - bt_w) // 2
    text_y = bar_y1 + (bar_height + bt_h) // 2 - 2
    
    # Text with contrast border
    cv2.putText(img, bar_text, (text_x, text_y), font, bar_font_scale, BLACK, 3, cv2.LINE_AA)
    cv2.putText(img, bar_text, (text_x, text_y), font, bar_font_scale, WHITE, 1, cv2.LINE_AA)

# 5. Display Single Matplotlib Plot
title_str = f"Predicted Sign: {', '.join(detected_signs)}" if detected_signs else "No Sign Detected"

plt.figure(figsize=(12, 9))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(title_str, fontsize=14, pad=12)
plt.axis('off')
plt.tight_layout()
plt.show()