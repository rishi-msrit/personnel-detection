"""
RealSense D415 — RGB Motion Detection + Object Detection + Direction Tracking
Stage 1+: Motion detection with YOLOv8 object classification and direction tracking.

Flicker Suppression (IR exposure → RGB luminance EMA):
    Periodic 50/60 Hz artificial lighting causes frame-to-frame brightness
    swings in the RGB stream.  A running EMA of mean-luminance is maintained
    and each frame is gain-corrected to that stable reference BEFORE background
    subtraction and video recording, eliminating the periodic intensity drift
    at the source without affecting colour balance or spatial detail.

Controls:
    Trackbars  — Threshold, Min Contour Area, Background Learning Rate
    r          — Start / stop video recording
    b          — Re-capture background model
    q          — Quit
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime
from collections import defaultdict

# ────────────────────────── YOLOv8 Setup ──────────────────────────
from ultralytics import YOLO

print("[YOLO] Loading YOLOv8n model (balanced accuracy/speed)...")
model = YOLO("yolov8n.pt")  # YOLOv8 nano — significantly better accuracy than v5
print("[YOLO] Model loaded.")

# ────────────────────────── Configuration ──────────────────────────
WIDTH, HEIGHT, FPS = 640, 480, 30
DEBOUNCE_FRAMES = 3
MAX_SAVED_FRAMES = 200

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
VIDEO_DIR = os.path.join(OUTPUT_DIR, "video")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames", "rgb_motion")
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

# ────────────────────────── Direction Labels ──────────────────────────
# ────────────────────────── Smooth Direction Tracker ──────────────────────────
# Uses EMA on the velocity vector so the arrow stays stable even when
# centroid jitters between frames. Hysteresis ensures the label only
# commits to a new direction after it holds for HYSTERESIS_FRAMES frames.

VELOCITY_ALPHA   = 0.22   # lower = more smoothing (0.0 = frozen, 1.0 = raw)
HYSTERESIS_FRAMES = 4     # frames new direction must hold before switching label

_ARROW_DIRS = {
    "Right": (1, 0), "Left": (-1, 0), "Up": (0, -1), "Down": (0, 1),
    "Up-Right": (1, -1), "Up-Left": (-1, -1),
    "Down-Right": (1, 1), "Down-Left": (-1, 1),
}


def _vec_to_direction(vx, vy, min_speed=1.2):
    if abs(vx) < min_speed and abs(vy) < min_speed:
        return "Stationary"
    angle = np.degrees(np.arctan2(-vy, vx))
    for lo, hi, label in [
        (-22.5,   22.5,  "Right"),    (22.5,  67.5, "Up-Right"),
        ( 67.5,  112.5,  "Up"),       (112.5, 157.5, "Up-Left"),
        (-157.5,-112.5,  "Down-Left"),(-112.5,-67.5, "Down"),
        ( -67.5, -22.5,  "Down-Right"),
    ]:
        if lo <= angle < hi:
            return label
    return "Left"


class SmoothTracker:
    """Nearest-neighbour tracker with EMA velocity + direction hysteresis."""

    def __init__(self):
        self.tracks = {}    # tid -> dict
        self.next_id = 0

    def update(self, detections, max_dist=100):
        global track_id_counter
        unmatched = set(self.tracks)
        new_centroids = {}

        for det in detections:
            cx, cy = det["cx"], det["cy"]
            best_id, best_d = None, max_dist
            for tid, t in self.tracks.items():
                d = np.hypot(cx - t["cx"], cy - t["cy"])
                if d < best_d:
                    best_d, best_id = d, tid

            if best_id is not None:
                t = self.tracks[best_id]
                t["vx"] = VELOCITY_ALPHA*(cx-t["cx"]) + (1-VELOCITY_ALPHA)*t["vx"]
                t["vy"] = VELOCITY_ALPHA*(cy-t["cy"]) + (1-VELOCITY_ALPHA)*t["vy"]
                t["cx"], t["cy"] = cx, cy
                cand = _vec_to_direction(t["vx"], t["vy"])
                if cand == t["cand"]:
                    t["hold"] += 1
                    if t["hold"] >= HYSTERESIS_FRAMES:
                        t["dir"] = cand
                else:
                    t["cand"], t["hold"] = cand, 1
                det["track_id"] = best_id
                det["direction"] = t["dir"]
                unmatched.discard(best_id)
            else:
                self.next_id += 1
                self.tracks[self.next_id] = {
                    "cx": cx, "cy": cy, "vx": 0.0, "vy": 0.0,
                    "dir": "New", "cand": "New", "hold": 0,
                }
                det["track_id"] = self.next_id
                det["direction"] = "New"

        for tid in unmatched:
            del self.tracks[tid]
        return detections

    @staticmethod
    def draw_arrow(frame, det):
        direction = det.get("direction", "")
        if direction in _ARROW_DIRS:
            dx, dy = _ARROW_DIRS[direction]
            cx, cy = det["cx"], det["cy"]
            cv2.arrowedLine(frame, (cx, cy),
                            (cx + dx*35, cy + dy*35),
                            (0, 200, 255), 2, tipLength=0.4)


# ────────────────────────── Pipeline Setup ──────────────────────────
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
pipeline.start(config)
time.sleep(1)

# ────────────────────────── Luminance Flicker Suppressor ──────────────────────────
# Periodic 50/60 Hz lighting produces a smooth brightness oscillation across
# consecutive RGB frames.  We track the per-frame mean luminance with an EMA
# and apply a scalar gain correction so each frame has the same effective
# brightness as the long-term average — neutralising the flicker before
# motion detection and before video is written.
#
# EMA_LUMA_ALPHA — controls how quickly the reference adapts.
#   ≈0.97 is slow (stable reference over ~33 frames), good for steady lighting.
#   Lower → faster adaptation but may track real brightness changes.
EMA_LUMA_ALPHA = 0.97
_ema_luma = None   # initialised on first frame


def apply_luma_stabilisation(bgr_frame):
    """Return a gain-corrected copy of *bgr_frame* with flicker removed."""
    global _ema_luma
    # Compute current mean luminance (use perceived-luminance weights)
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    current_luma = float(np.mean(gray))

    if _ema_luma is None:
        _ema_luma = current_luma
        return bgr_frame  # no correction on the very first frame

    # Update running EMA reference
    _ema_luma = EMA_LUMA_ALPHA * _ema_luma + (1.0 - EMA_LUMA_ALPHA) * current_luma

    if current_luma < 1.0:
        return bgr_frame  # avoid division by near-zero

    # Scale frame so its mean luminance matches the stable EMA reference
    gain = _ema_luma / current_luma
    # Clamp gain to ±15% so we never over-correct on a real scene change
    gain = float(np.clip(gain, 0.85, 1.15))
    stabilised = cv2.convertScaleAbs(bgr_frame, alpha=gain, beta=0)
    return stabilised

# ────────────────────────── Window & Trackbars ──────────────────────────
WINDOW = "RGB Motion + Object Detection"
cv2.namedWindow(WINDOW)
cv2.createTrackbar("Threshold", WINDOW, 25, 100, lambda x: None)
cv2.createTrackbar("Min Area", WINDOW, 5000, 50000, lambda x: None)
cv2.createTrackbar("BG Rate x1000", WINDOW, 5, 100, lambda x: None)

# ────────────────────────── State ──────────────────────────
background = None
motion_counter = 0
clear_counter = 0
motion_confirmed = False

recording = False
video_writer = None
saved_frame_count = 0

fps_counter = 0
fps_value = 0.0
fps_timer = time.time()

# Object tracking
tracker = SmoothTracker()
track_id_counter = 0
YOLO_INTERVAL = 3  # run YOLO every N frames for performance
last_detections = []  # cache last YOLO results
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # contrast boost for YOLO


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def start_recording(w, h):
    global video_writer, recording
    path = os.path.join(VIDEO_DIR, f"rgb_motion_{timestamp()}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(path, fourcc, FPS, (w, h))
    recording = True
    print(f"[REC] Started → {path}")


def stop_recording():
    global video_writer, recording
    if video_writer:
        video_writer.release()
        video_writer = None
    recording = False
    print("[REC] Stopped")



# ────────────────────────── Main Loop ──────────────────────────
frame_count = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame_raw = np.asanyarray(color_frame.get_data())

        # ── [1] Luminance flicker suppression ──────────────────────────────
        # Applied BEFORE background subtraction and display so that periodic
        # 50/60 Hz brightness oscillations are removed at the source.
        frame = apply_luma_stabilisation(frame_raw)
        display = frame.copy()

        # ── Motion detection (background subtraction) ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if background is None:
            background = gray.astype("float")
            continue

        alpha = cv2.getTrackbarPos("BG Rate x1000", WINDOW) / 1000.0
        if alpha < 0.001:
            alpha = 0.001
        cv2.accumulateWeighted(gray, background, alpha)

        diff = cv2.absdiff(gray, cv2.convertScaleAbs(background))
        thresh_val = cv2.getTrackbarPos("Threshold", WINDOW)
        min_area = cv2.getTrackbarPos("Min Area", WINDOW)

        _, thresh = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
        kernel_small = np.ones((5, 5), np.uint8)
        # Tighter morphology: prevents small movements (fan, curtain) from
        # flooding the entire frame — erode kills thin noise, dilate is minimal.
        thresh = cv2.erode(thresh, kernel_small, iterations=2)
        thresh = cv2.dilate(thresh, kernel_small, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_small)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        MAX_AREA = int(WIDTH * HEIGHT * 0.55)   # ignore frame-wide sensor floods
        raw_motion = any(min_area < cv2.contourArea(cnt) < MAX_AREA for cnt in contours)

        # Temporal debounce
        if raw_motion:
            motion_counter += 1
            clear_counter = 0
            if motion_counter >= DEBOUNCE_FRAMES:
                motion_confirmed = True
        else:
            clear_counter += 1
            motion_counter = 0
            if clear_counter >= DEBOUNCE_FRAMES:
                motion_confirmed = False

        # ── YOLOv8 Object Detection (every N frames) ──
        frame_count += 1
        if frame_count % YOLO_INTERVAL == 0:
            # CLAHE contrast enhancement for better detection in variable lighting
            gray_for_clahe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe_gray = _clahe.apply(gray_for_clahe)
            clahe_bgr = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)
            # Blend 60% original color with 40% contrast-boosted grayscale structure
            yolo_input = cv2.addWeighted(frame, 0.6, clahe_bgr, 0.4, 0)
            
            results = model(yolo_input, verbose=False, conf=0.30, iou=0.45)
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    detections.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": conf, "label": label,
                        "cx": cx, "cy": cy
                    })

            last_detections = tracker.update(detections)

        # ── Draw detections ──
        for det in last_detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            label = det["label"]
            conf = det["conf"]
            direction = det.get("direction", "")

            # Green box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label: "person (0.87) → Right"
            text = f"{label} ({conf:.2f})"
            if direction and direction != "New":
                text += f" | {direction}"

            # Background for text
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(display, text, (x1 + 2, y1 - 5),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Direction arrow (stable — drawn from smoothed velocity)
            SmoothTracker.draw_arrow(display, det)

        # ── Status overlay ──
        if motion_confirmed:
            cv2.putText(display, "MOTION DETECTED", (20, 35),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(display, "CLEAR", (20, 35),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

        # Object count
        if last_detections:
            obj_summary = {}
            for det in last_detections:
                obj_summary[det["label"]] = obj_summary.get(det["label"], 0) + 1
            summary_text = " | ".join(f"{k}: {v}" for k, v in obj_summary.items())
            cv2.putText(display, summary_text, (20, HEIGHT - 15),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # FPS
        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_value = fps_counter / elapsed
            fps_counter = 0
            fps_timer = time.time()

        cv2.putText(display, f"FPS: {fps_value:.1f}", (WIDTH - 110, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if recording:
            cv2.circle(display, (WIDTH - 20, 45), 8, (0, 0, 255), -1)

        # ── Recording & saving ──
        if recording and video_writer:
            video_writer.write(display)

        if motion_confirmed and saved_frame_count < MAX_SAVED_FRAMES:
            path = os.path.join(FRAMES_DIR, f"motion_{timestamp()}.png")
            cv2.imwrite(path, display)
            saved_frame_count += 1

        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            if recording:
                stop_recording()
            else:
                h_d, w_d = display.shape[:2]
                start_recording(w_d, h_d)
        elif key == ord("b"):
            background = gray.astype("float")
            motion_counter = 0
            clear_counter = 0
            motion_confirmed = False
            print("[BG] Background recaptured")

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    if recording:
        stop_recording()
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"Saved {saved_frame_count} motion frames to {FRAMES_DIR}")
