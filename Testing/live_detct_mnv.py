"""
ASL / Hand Gesture Recognition - "Super Engine"
=================================================
Real-time American Sign Language recognition pipeline combining:
  - MediaPipe Hands  -> 21-point keypoint tracking + skeleton overlay
  - Custom YOLO model -> gesture classification on cropped hand regions
  - Temporal smoothing -> majority-vote buffer to eliminate flicker
  - Full-frame YOLO fallback -> used only when MediaPipe misses a hand
  - Futuristic HUD -> cyan brackets, label tag, confidence bar, FPS readout

Requirements:
    pip install opencv-python mediapipe ultralytics torch

Run:
    python asl_super_engine.py
    (press 'q' to quit)
"""

import os
import sys
import time
import cv2
import torch
import numpy as np
from collections import deque, Counter
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_draw
from ultralytics import YOLO

class Config:
    # Model
    MODEL_RELATIVE_PATH = os.path.join(".model", "ALpha_SIGN_M89.pt")

    # Camera
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

    # MediaPipe
    MAX_NUM_HANDS = 2
    MIN_DETECTION_CONF = 0.5
    MIN_TRACKING_CONF = 0.5
    CROP_PADDING = 45          # pixels of padding around the hand bbox before cropping

    # YOLO inference
    PRIMARY_IMGSZ = 320        # inference size when we have a MediaPipe crop (small, fast)
    PRIMARY_CONF = 0.15
    FALLBACK_IMGSZ = 480       # inference size for full-frame fallback (needs more context)
    FALLBACK_CONF = 0.25

    # Temporal smoothing
    BUFFER_LEN = 5             # frames considered for majority-vote stabilization

    # Threading / performance
    # NOTE: using ALL cores for torch can cause oversubscription against OpenCV's
    # own thread pool and *hurt* FPS. Cap it - 4 is a good default for most CPUs.
    TORCH_THREADS = min(4, os.cpu_count() or 4)
    OPENCV_THREADS = 4

    # HUD colors (BGR)
    CYAN = (255, 255, 0)
    NEON_GREEN = (0, 255, 128)
    DARK_BG = (20, 20, 20)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)


# ============================================================
# STARTUP - thread tuning, model loading, MediaPipe init
# ============================================================
def resolve_model_path() -> str:
    """Build an absolute path to the model file and validate it exists."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    path = os.path.join(project_dir, Config.MODEL_RELATIVE_PATH)

    if not os.path.isfile(path):
        print(f"[FATAL] YOLO model not found at: {path}")
        print("        Check that ALpha_SIGN_M89.pt sits under <project>/.model/")
        sys.exit(1)
    return path


def init_pipeline():
    """Apply thread tuning, load YOLO, and construct the MediaPipe Hands solution."""
    torch.set_num_threads(Config.TORCH_THREADS)
    cv2.setNumThreads(Config.OPENCV_THREADS)

    model_path = resolve_model_path()
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=Config.MAX_NUM_HANDS,
        min_detection_confidence=Config.MIN_DETECTION_CONF,
        min_tracking_confidence=Config.MIN_TRACKING_CONF,
    )

    return model, hands


def open_camera():
    """Open the webcam and validate that it actually produced a frame."""
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[FATAL] Could not open camera index {Config.CAMERA_INDEX}.")
        sys.exit(1)

    ok, _ = cap.read()
    if not ok:
        print("[FATAL] Camera opened but returned no frames. Check permissions/drivers.")
        cap.release()
        sys.exit(1)

    return cap


# ============================================================
# CORE DETECTION HELPERS
# ============================================================
def clamp_bbox(xmin, ymin, xmax, ymax, w, h):
    """Keep a bounding box inside frame bounds, regardless of source."""
    xmin = max(0, min(int(xmin), w - 1))
    ymin = max(0, min(int(ymin), h - 1))
    xmax = max(0, min(int(xmax), w))
    ymax = max(0, min(int(ymax), h))
    return xmin, ymin, xmax, ymax


def best_box_from_results(yolo_results):
    """Return (label, confidence) for the highest-confidence detection, or (None, 0.0)."""
    if yolo_results.boxes is None or len(yolo_results.boxes) == 0:
        return None, 0.0
    best = max(yolo_results.boxes, key=lambda b: float(b.conf[0]))
    cls_id = int(best.cls[0])
    return yolo_results.names[cls_id] if hasattr(yolo_results, "names") else None, float(best.conf[0])


def run_mediapipe_pipeline(model, hand_landmarks_list, clean_frame, display_frame, w, h,
                            landmark_style, connection_style):
    """
    Primary pipeline: for each detected hand, crop from the CLEAN (unannotated) frame,
    run YOLO on just that region, then draw the skeleton on the DISPLAY frame afterward
    so annotations never leak into what YOLO sees.

    Returns (detected_sign, detected_conf, active_bbox) for the last processed hand.
    """
    detected_sign, detected_conf, active_bbox = None, 0.0, None

    for hand_landmarks in hand_landmarks_list:
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        pad = Config.CROP_PADDING
        xmin, ymin, xmax, ymax = clamp_bbox(
            min(x_coords) - pad, min(y_coords) - pad,
            max(x_coords) + pad, max(y_coords) + pad,
            w, h,
        )

        hand_crop = clean_frame[ymin:ymax, xmin:xmax]  # crop from UNANNOTATED frame

        if hand_crop.size > 0:
            yolo_results = model.predict(
                source=hand_crop,
                imgsz=Config.PRIMARY_IMGSZ,
                conf=Config.PRIMARY_CONF,
                device="cpu",
                verbose=False,
            )[0]

            label, conf = best_box_from_results(yolo_results)
            if label is not None:
                detected_sign, detected_conf = label, conf
                active_bbox = (xmin, ymin, xmax, ymax)

        # Draw skeleton on the DISPLAY frame only, after cropping is done
        mp_draw.draw_landmarks(
            display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style,
        )

    return detected_sign, detected_conf, active_bbox


def run_fallback_pipeline(model, clean_frame, w, h):
    """Full-frame YOLO pass used only when MediaPipe finds no hands at all."""
    yolo_results = model.predict(
        source=clean_frame,
        imgsz=Config.FALLBACK_IMGSZ,
        conf=Config.FALLBACK_CONF,
        device="cpu",
        verbose=False,
    )[0]

    label, conf = best_box_from_results(yolo_results)
    if label is None:
        return None, 0.0, None

    best_box = max(yolo_results.boxes, key=lambda b: float(b.conf[0]))
    xmin, ymin, xmax, ymax = clamp_bbox(*best_box.xyxy[0].tolist(), w, h)
    return label, conf, (xmin, ymin, xmax, ymax)


# ============================================================
# HUD RENDERING
# ============================================================
def draw_hud(frame, stable_label, active_bbox, detected_conf, fps):
    """Render corner brackets, gesture label tag with confidence bar, and FPS counter."""
    if stable_label and active_bbox:
        xmin, ymin, xmax, ymax = active_bbox
        box_w, box_h = xmax - xmin, ymax - ymin
        conf_pct = int(detected_conf * 100)
        line_len = max(20, int(min(box_w, box_h) * 0.25))
        t = 3

        # --- Corner brackets ---
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), Config.CYAN, 1)
        for (x1, y1), (x2, y2) in [
            ((xmin, ymin), (xmin + line_len, ymin)), ((xmin, ymin), (xmin, ymin + line_len)),
            ((xmax, ymin), (xmax - line_len, ymin)), ((xmax, ymin), (xmax, ymin + line_len)),
            ((xmin, ymax), (xmin + line_len, ymax)), ((xmin, ymax), (xmin, ymax - line_len)),
            ((xmax, ymax), (xmax - line_len, ymax)), ((xmax, ymax), (xmax, ymax - line_len)),
        ]:
            cv2.line(frame, (x1, y1), (x2, y2), Config.CYAN, t)

        # --- Label + confidence bar tag ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        lbl_scale, lbl_thick = 0.85, 2
        (lbl_w, lbl_h), _ = cv2.getTextSize(stable_label, font, lbl_scale, lbl_thick)

        bar_height = 28
        bar_text = f"Accuracy: {conf_pct}%"
        bar_font_scale, bar_font_thick = 0.7, 2
        (bt_w, bt_h), _ = cv2.getTextSize(bar_text, font, bar_font_scale, bar_font_thick)

        pad_x, pad_y = 14, 10
        tag_w = max(lbl_w + (pad_x * 2), bt_w + (pad_x * 2), 250)
        tag_h = lbl_h + bar_height + (pad_y * 3)

        tag_y1 = max(ymin - tag_h - 6, 0)
        tag_y2 = tag_y1 + tag_h

        cv2.rectangle(frame, (xmin, tag_y1), (xmin + tag_w, tag_y2), Config.DARK_BG, -1)
        cv2.rectangle(frame, (xmin, tag_y1), (xmin + tag_w, tag_y2), Config.CYAN, 1)

        cv2.putText(frame, stable_label, (xmin + pad_x, tag_y1 + lbl_h + pad_y),
                    font, lbl_scale, Config.CYAN, lbl_thick, cv2.LINE_AA)

        bar_y1 = tag_y2 - pad_y - bar_height
        bar_y2 = tag_y2 - pad_y
        bar_max_w = tag_w - (pad_x * 2)
        bar_fill_w = int(bar_max_w * detected_conf)

        cv2.rectangle(frame, (xmin + pad_x, bar_y1), (xmin + pad_x + bar_max_w, bar_y2), (50, 50, 50), -1)
        cv2.rectangle(frame, (xmin + pad_x, bar_y1), (xmin + pad_x + bar_fill_w, bar_y2), Config.NEON_GREEN, -1)
        cv2.rectangle(frame, (xmin + pad_x, bar_y1), (xmin + pad_x + bar_max_w, bar_y2), Config.CYAN, 1)

        text_x = xmin + pad_x + (bar_max_w - bt_w) // 2
        text_y = bar_y1 + (bar_height + bt_h) // 2 - 2

        # Black outline + white fill for legibility on any background
        cv2.putText(frame, bar_text, (text_x, text_y), font, bar_font_scale, Config.BLACK, 3, cv2.LINE_AA)
        cv2.putText(frame, bar_text, (text_x, text_y), font, bar_font_scale, Config.WHITE, 1, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                Config.NEON_GREEN, 2, cv2.LINE_AA)


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    model, hands = init_pipeline()
    cap = open_camera()

    landmark_style = mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3)
    connection_style = mp_draw.DrawingSpec(color=(255, 0, 128), thickness=2)

    prediction_buffer = deque(maxlen=Config.BUFFER_LEN)
    prev_time = time.time()
    fps = 0

    print("Super Engine started. Press 'q' to quit.")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("[WARN] Camera feed unavailable, stopping.")
                break

            # --- FPS ---
            curr_time = time.time()
            elapsed = curr_time - prev_time
            fps = int(1.0 / elapsed) if elapsed > 0 else fps
            prev_time = curr_time

            frame = cv2.flip(frame, 1)  # mirror for natural viewing
            h, w, _ = frame.shape
            clean_frame = frame.copy()  # unannotated copy used for all YOLO input

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_results = hands.process(rgb_frame)

            if mp_results.multi_hand_landmarks:
                detected_sign, detected_conf, active_bbox = run_mediapipe_pipeline(
                    model, mp_results.multi_hand_landmarks, clean_frame, frame, w, h,
                    landmark_style, connection_style,
                )
            else:
                detected_sign, detected_conf, active_bbox = run_fallback_pipeline(model, clean_frame, w, h)

            # --- Temporal smoothing (majority vote over last N frames) ---
            if detected_sign:
                prediction_buffer.append(detected_sign)
            elif prediction_buffer:
                prediction_buffer.popleft()

            stable_label = Counter(prediction_buffer).most_common(1)[0][0] if prediction_buffer else None

            draw_hud(frame, stable_label, active_bbox, detected_conf, fps)

            cv2.imshow("Super Engine - Sign Language Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()