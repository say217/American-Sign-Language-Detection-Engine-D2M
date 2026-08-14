import cv2
import time
import threading
from collections import deque, Counter
import mediapipe as mp
from ultralytics import YOLO

WEIGHTS_PATH = r"C:\PROJECTS\Sign Language Detection Engine\.model\Indian_SIgn_language_detection.pt"
CAM_INDEX = 0


CONF_THRES = 0.35                    # 0.15 was way too low -> tons of noisy false positives
                                      # raise once you confirm ROI cropping helps; tune from there

CAM_WIDTH, CAM_HEIGHT = 640, 480
YOLO_IMGSZ = 224                     # crop is already zoomed on the hand, so we don't need 640 here
DETECT_EVERY_N_FRAMES = 2            # ROI crops are cheap for YOLO, so we can afford more frequent runs
DEBUG = True

# --- ROI cropping around detected hand(s) ---
ROI_PAD_RATIO = 0.35                 # extra padding around the mediapipe hand bbox (fraction of bbox size)
MIN_ROI_SIZE = 120                   # px, avoid absurdly tiny crops when hand is far away

# --- temporal smoothing to kill flicker / stray false positives ---
SMOOTH_WINDOW = 5                    # frames of history per hand slot
SMOOTH_MIN_AGREEMENT = 3             # need at least this many frames agreeing before we show a label

WHITE        = (255, 255, 255)
FAINT_WHITE  = (200, 200, 200)
FONT         = cv2.FONT_HERSHEY_SIMPLEX
CORNER_LEN   = 22
THICKNESS    = 2
LINE_THIN    = 1

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
HAND_LANDMARK_STYLE = mp_draw.DrawingSpec(color=WHITE, thickness=2, circle_radius=3)
HAND_CONNECTION_STYLE = mp_draw.DrawingSpec(color=FAINT_WHITE, thickness=1)


class CamStream:
    def __init__(self, index, width, height):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera. Check CAM_INDEX.")
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


def draw_corner_box(img, x1, y1, x2, y2):
    cv2.rectangle(img, (x1, y1), (x2, y2), FAINT_WHITE, LINE_THIN, cv2.LINE_AA)
    for (cx, cy, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * CORNER_LEN, cy), WHITE, THICKNESS, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * CORNER_LEN), WHITE, THICKNESS, cv2.LINE_AA)


def draw_label_bar(img, x1, y1, label, conf, bar_w=130, bar_h=6):
    text = f"{label}  {conf * 100:.1f}%"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    pad = 6
    box_w, box_h = max(tw, bar_w) + pad * 2, th + bar_h + pad * 3
    ty1 = max(0, y1 - box_h)

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, ty1), (x1 + box_w, ty1 + box_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (x1, ty1), (x1 + box_w, ty1 + box_h), WHITE, 1, cv2.LINE_AA)

    cv2.putText(img, text, (x1 + pad, ty1 + th + pad // 2), FONT, 0.55, WHITE, 1, cv2.LINE_AA)

    bar_x, bar_y = x1 + pad, ty1 + th + pad + 4
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_w * conf), bar_y + bar_h), WHITE, -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), WHITE, 1)


def draw_hud_frame(img, fps):
    h, w = img.shape[:2]
    cv2.line(img, (0, 40), (w, 40), FAINT_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, "ISL LIVE TRACKING", (15, 28), FONT, 0.65, WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, f"FPS: {fps:.1f}", (w - 140, 28), FONT, 0.6, WHITE, 1, cv2.LINE_AA)
    cv2.line(img, (0, h - 25), (w, h - 25), FAINT_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, "Press 'q' to quit", (15, h - 8), FONT, 0.5, FAINT_WHITE, 1, cv2.LINE_AA)


def hand_bbox_from_landmarks(landmarks, img_w, img_h, pad_ratio):
    """Return a padded, clamped (x1, y1, x2, y2) box around one hand's landmarks."""
    xs = [lm.x * img_w for lm in landmarks.landmark]
    ys = [lm.y * img_h for lm in landmarks.landmark]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * pad_ratio, bh * pad_ratio
    x1 -= pad_x; x2 += pad_x
    y1 -= pad_y; y2 += pad_y

    # keep square-ish crops (YOLO/CNNs generally prefer this) and enforce a minimum size
    side = max(x2 - x1, y2 - y1, MIN_ROI_SIZE)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1, x2 = cx - side / 2, cx + side / 2
    y1, y2 = cy - side / 2, cy + side / 2

    x1 = int(max(0, x1)); y1 = int(max(0, y1))
    x2 = int(min(img_w, x2)); y2 = int(min(img_h, y2))
    return x1, y1, x2, y2


class LabelSmoother:
    """Per-hand-slot rolling vote so a single stray frame can't flash a wrong label."""
    def __init__(self, window, min_agree):
        self.window = window
        self.min_agree = min_agree
        self.history = deque(maxlen=window)

    def update(self, label, conf):
        self.history.append((label, conf))
        counts = Counter(l for l, _ in self.history)
        best_label, votes = counts.most_common(1)[0]
        if votes < self.min_agree:
            return None
        confs = [c for l, c in self.history if l == best_label]
        return best_label, sum(confs) / len(confs)


def main():
    model = YOLO(WEIGHTS_PATH)
    stream = CamStream(CAM_INDEX, CAM_WIDTH, CAM_HEIGHT)

    prev_time = time.time()
    frame_count = 0
    last_boxes = []          # cache: (x1, y1, x2, y2, cls_name, conf) in FULL-FRAME coords
    smoothers = []            # one LabelSmoother per currently-tracked hand slot

    with mp_hands.Hands(model_complexity=0, max_num_hands=2,
                         min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        while True:
            ret, raw_frame = stream.read()
            if not ret or raw_frame is None:
                continue

            frame_count += 1
            h, w = raw_frame.shape[:2]

            # --- run hand tracking + YOLO on the UNFLIPPED frame (matches training orientation) ---
            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb)

            hand_boxes_px = []
            if hand_results.multi_hand_landmarks:
                for lm in hand_results.multi_hand_landmarks:
                    hand_boxes_px.append(hand_bbox_from_landmarks(lm, w, h, ROI_PAD_RATIO))
            elif DEBUG and frame_count % 30 == 0:
                print("No hand landmarks detected this frame.")

            # keep smoother count in sync with number of hands currently tracked
            while len(smoothers) < len(hand_boxes_px):
                smoothers.append(LabelSmoother(SMOOTH_WINDOW, SMOOTH_MIN_AGREEMENT))
            smoothers = smoothers[:max(len(hand_boxes_px), 0)] or smoothers[:1]

            if frame_count % DETECT_EVERY_N_FRAMES == 0 and hand_boxes_px:
                new_boxes = []
                for i, (rx1, ry1, rx2, ry2) in enumerate(hand_boxes_px):
                    crop = raw_frame[ry1:ry2, rx1:rx2]
                    if crop.size == 0:
                        continue
                    res = model.predict(source=crop, conf=CONF_THRES, imgsz=YOLO_IMGSZ, verbose=False)[0]
                    if len(res.boxes) == 0:
                        continue
                    # take the single most confident detection in this hand's crop
                    best = max(res.boxes, key=lambda b: float(b.conf[0].item()))
                    cls_id = int(best.cls[0].item())
                    cls_name = model.names[cls_id]
                    conf = float(best.conf[0].item())

                    bx1, by1, bx2, by2 = map(int, best.xyxy[0].tolist())
                    # translate crop-local coords back to full-frame coords
                    fx1, fy1, fx2, fy2 = bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1

                    smoother = smoothers[i] if i < len(smoothers) else LabelSmoother(SMOOTH_WINDOW, SMOOTH_MIN_AGREEMENT)
                    if i >= len(smoothers):
                        smoothers.append(smoother)
                    smoothed = smoother.update(cls_name, conf)
                    if smoothed is not None:
                        s_label, s_conf = smoothed
                        new_boxes.append((fx1, fy1, fx2, fy2, s_label, s_conf))

                last_boxes = new_boxes
                if DEBUG:
                    print(f"[frame {frame_count}] hands: {len(hand_boxes_px)}  stable detections: {len(last_boxes)}")
            elif not hand_boxes_px:
                last_boxes = []  # no hand -> nothing to show, don't keep stale boxes on screen

            # --- now build the display frame: flip for a natural mirror view ---
            display = cv2.flip(raw_frame, 1)

            if hand_results.multi_hand_landmarks:
                # draw skeleton on the flipped display: mirror each landmark's x
                for lm in hand_results.multi_hand_landmarks:
                    mirrored = mp.solutions.hands.HandLandmark  # noqa: F841 (kept for clarity)
                    lm_mirrored = type(lm)()
                    lm_mirrored.landmark.extend(lm.landmark)
                    for point in lm_mirrored.landmark:
                        point.x = 1.0 - point.x
                    mp_draw.draw_landmarks(
                        display, lm_mirrored, mp_hands.HAND_CONNECTIONS,
                        HAND_LANDMARK_STYLE, HAND_CONNECTION_STYLE
                    )

            for (x1, y1, x2, y2, cls_name, conf) in last_boxes:
                # mirror the box x-coordinates to match the flipped display
                mx1, mx2 = w - x2, w - x1
                draw_corner_box(display, mx1, y1, mx2, y2)
                draw_label_bar(display, mx1, y1, cls_name, conf)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            draw_hud_frame(display, fps)

            cv2.imshow("ISL Live Detection", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()