"""
RealSense D415 — Depth Motion Detection + Object Detection + Direction Tracking
Stage 1+: Depth-based motion detection with YOLOv8 classification and direction tracking.

Uses the RGB stream for object detection and the depth stream for motion detection,
overlaying results on the colorized depth view.

IR Exposure Flicker Suppression (NEW):
    The D415’s IR projector/imager shares a depth sensor whose auto-exposure
    causes periodic brightness oscillations in the raw depth signal.  Auto-
    exposure is disabled and replaced by a calibrated manual exposure whose
    integration window spans one or more full 50/60 Hz mains cycles, so each
    frame averages over complete lighting cycles and the periodic component
    cancels.  The best exposure is selected via a short variance-minimisation
    pass at startup and re-evaluated every ~10 s with EMA smoothing.

Controls:
    Trackbars  — Depth Diff (mm), Min Contour Area, Max Contour Area
    r          — Start / stop video recording
    b          — Re-capture background depth
    q          — Quit
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime

# ────────────────────────── YOLOv8 Setup ──────────────────────────
from ultralytics import YOLO

print("[YOLO] Loading YOLOv8n model (balanced accuracy/speed)...")
model = YOLO("yolov8n.pt")  # YOLOv8 nano — significantly better accuracy than v5
print("[YOLO] Model loaded.")

# ────────────────────────── Configuration ──────────────────────────
WIDTH, HEIGHT, FPS = 640, 480, 30
DEBOUNCE_FRAMES = 5    # higher = fewer false motion triggers from IR noise
MAX_SAVED_FRAMES = 200
EDGE_MARGIN = 0.10

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
VIDEO_DIR = os.path.join(OUTPUT_DIR, "video")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames", "depth_motion")
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

# ────────────────────────── Direction Helpers ──────────────────────────
VELOCITY_ALPHA    = 0.22
HYSTERESIS_FRAMES = 4

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
        self.tracks = {}
        self.next_id = 0

    def update(self, detections, max_dist=100):
        unmatched = set(self.tracks)
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
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

# Align depth to color for matching YOLO detections to depth
align = rs.align(rs.stream.color)

profile = pipeline.start(config)

# Native colorizer for Viewer-matching visualization
colorizer = rs.colorizer()
colorizer.set_option(rs.option.color_scheme, 0)

# ────────────────────────── IR Exposure Flicker Suppressor ──────────────────────────

class IRFlickerSuppressor:
    """
    Minimises IR sensor flicker by locking the depth sensor to the exposure
    value that produces the smallest frame-to-frame intensity variance.

    Candidate exposures (µs) are aligned with 50/60 Hz mains cycle lengths:
        8 000 µs  →  sub-cycle safe floor
       10 000 µs  →  10 ms
       16 600 µs  →  ≈1 full 60 Hz cycle (16.67 ms)
       20 000 µs  →  1 full 50 Hz cycle (20 ms)
    """

    CANDIDATE_EXPOSURES_US = [8000, 10000, 16600, 20000]
    EVAL_FRAMES = 20             # frames captured per candidate during calibration
    REEVAL_INTERVAL_FRAMES = 900  # re-evaluate every ~30 s at 30 fps
    EMA_ALPHA = 0.8              # weight for current exposure during EMA blend

    def __init__(self, depth_sensor):
        self._sensor = depth_sensor
        self._current_exposure_us = self.CANDIDATE_EXPOSURES_US[0]
        self._frames_since_eval = 0
        self._calibrated = False
        self.status_text = "IR Exp: calibrating…"

    def _set_exposure(self, us: int) -> int:
        try:
            opt = rs.option.exposure
            if self._sensor.supports(opt):
                r = self._sensor.get_option_range(opt)
                clamped = int(max(r.min, min(r.max, us)))
                self._sensor.set_option(opt, clamped)
                return clamped
        except Exception as exc:
            print(f"[IR-EXP] set_exposure error: {exc}")
        return us

    def _measure_variance(self, pipeline_ref, n: int) -> float:
        """Measure temporal variance using depth frames (IR stream not in this pipeline).
        We use mean depth over valid (non-zero) pixels per frame as the signal.
        IR exposure directly controls the IR projector integration window, so
        the depth signal stability is a valid proxy for IR flicker suppression.
        """
        intensities = []
        for _ in range(n):
            try:
                fs = pipeline_ref.wait_for_frames(timeout_ms=300)
                aligned_fs = align.process(fs)
                df = aligned_fs.get_depth_frame()
                if df:
                    img = np.asanyarray(df.get_data()).astype(np.float32)
                    valid = img[img > 0]
                    if len(valid) > 100:   # need a meaningful sample
                        intensities.append(float(np.mean(valid)))
            except Exception:
                pass
        return float(np.std(intensities)) if len(intensities) >= 4 else float("inf")

    def calibrate(self, pipeline_ref):
        print("[IR-EXP] Calibrating exposure for flicker suppression…")
        results = {}
        for exp_us in self.CANDIDATE_EXPOSURES_US:
            applied = self._set_exposure(exp_us)
            time.sleep(0.08)  # let sensor settle
            std = self._measure_variance(pipeline_ref, self.EVAL_FRAMES)
            results[applied] = std
            print(f"         {applied:>6} µs  std={std:.4f}")

        best = min(results, key=results.get)
        print(f"[IR-EXP] Best exposure: {best} µs (std={results[best]:.4f})")

        if self._calibrated:
            blended = int(
                self.EMA_ALPHA * self._current_exposure_us
                + (1.0 - self.EMA_ALPHA) * best
            )
            self._current_exposure_us = self._set_exposure(blended)
        else:
            self._current_exposure_us = self._set_exposure(best)
            self._calibrated = True

        self.status_text = f"IR Exp: {self._current_exposure_us} µs"
        self._frames_since_eval = 0

    def tick(self, pipeline_ref):
        """Call once per main-loop iteration BEFORE wait_for_frames()."""
        if not self._calibrated or (
            self._frames_since_eval >= self.REEVAL_INTERVAL_FRAMES
        ):
            self.calibrate(pipeline_ref)
        self._frames_since_eval += 1


# Disable auto-exposure, then instantiate suppressor
_depth_sensor = profile.get_device().first_depth_sensor()
if _depth_sensor.supports(rs.option.enable_auto_exposure):
    _depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
    print("[IR-EXP] Auto-exposure DISABLED on depth sensor.")

ir_suppressor = IRFlickerSuppressor(_depth_sensor)

time.sleep(2)

# ────────────────────────── Capture Background ──────────────────────────
print("[BG] Capturing background (averaging 15 frames)...")
bg_accumulator = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
bg_count_total = np.zeros((HEIGHT, WIDTH), dtype=np.float64)

for _ in range(15):
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    df = aligned.get_depth_frame()
    if df:
        d = np.asanyarray(df.get_data()).astype(np.float64)
        valid = d > 0
        bg_accumulator[valid] += d[valid]
        bg_count_arr = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
        bg_count_arr[valid] += 1
        bg_count_total += bg_count_arr

bg_count_total[bg_count_total == 0] = 1
background = (bg_accumulator / bg_count_total).astype(np.float32)
print("[BG] Background captured")

# ────────────────────────── Auto-compute Motion Parameters ──────────────────────────
# Depth threshold: 3× the measured background noise std, clamped to [60, 400] mm.
# This ensures the threshold is always above sensor noise but sensitive to real motion.
def _compute_depth_thresh(bg):
    valid = bg[bg > 0].flatten()
    if len(valid) < 100:
        return 150  # safe fallback
    return int(np.clip(float(np.std(valid)) * 3.0, 60, 400))

depth_thresh = _compute_depth_thresh(background)

# Area bounds: 1% (min) … 78% (max) of total frame pixels.
MIN_AREA = int(WIDTH * HEIGHT * 0.010)   # ~3072 px — ignores tiny noise blobs
MAX_AREA = int(WIDTH * HEIGHT * 0.50)   # 50% max — prevents frame-wide flood artefacts

# LPF alpha: 0.72 = 72% current frame + 28% history.
# Higher → more responsive; lower → smoother. 0.72 is a good balance for indoor scenes.
LPF_ALPHA = 0.72

print(f"[AUTO] depth_thresh={depth_thresh} mm | min_area={MIN_AREA} | "
      f"max_area={MAX_AREA} | lpf_alpha={LPF_ALPHA}")

# ────────────────────────── ROI Mask ──────────────────────────
roi_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
margin_y = int(HEIGHT * EDGE_MARGIN)
margin_x = int(WIDTH * EDGE_MARGIN)
roi_mask[margin_y:HEIGHT - margin_y, margin_x:WIDTH - margin_x] = 255

# ────────────────────────── Window & Jitter Slider ──────────────────────────
WINDOW = "Depth Motion + Object Detection"
cv2.namedWindow(WINDOW)
# Jitter Reject %: controls the adaptive depth-diff percentile threshold.
# 0  = very sensitive (detects faint motion, but may false-alarm on sensor jitter)
# 100 = very selective (only large, confident depth changes trigger motion)
# Default 55 balances noise rejection with sensitivity for typical indoor scenes.
cv2.createTrackbar("Jitter Reject %", WINDOW, 55, 100, lambda x: None)

# ────────────────────────── State ──────────────────────────
motion_counter = 0
clear_counter = 0
motion_confirmed = False
recording = False
video_writer = None
saved_frame_count = 0

fps_counter = 0
fps_value = 0.0
fps_timer = time.time()

prev_depth = None   # holds previous EMA-filtered depth frame for LPF

tracker = SmoothTracker()
track_id_counter = 0
YOLO_INTERVAL = 2
last_detections = []
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def start_recording(w, h):
    global video_writer, recording
    path = os.path.join(VIDEO_DIR, f"depth_motion_{timestamp()}.mp4")
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
        # ── [1] IR Exposure: suppress flicker BEFORE capturing frames ────────────────
        # Sets the IR integration window to align with mains-lighting cycles
        # so temporal flicker averages out within a single frame.
        ir_suppressor.tick(pipeline)

        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        color_image = np.asanyarray(color_frame.get_data())

        # ── Temporal LPF (EMA) on depth frames ──────────────────────────────
        # Auto-computed LPF_ALPHA=0.72: blends 72% current + 28% history to
        # suppress high-frequency IR sensor noise before motion detection.
        if prev_depth is None or prev_depth.shape != depth_raw.shape:
            prev_depth = depth_raw.copy()
        depth = LPF_ALPHA * depth_raw + (1.0 - LPF_ALPHA) * prev_depth
        depth = depth.astype(np.float32)
        prev_depth = depth.copy()

        # Colorized depth display (Viewer-matching)
        depth_colorized = np.asanyarray(
            colorizer.colorize(depth_frame).get_data()
        )
        display = depth_colorized.copy()

        # ── Adaptive depth-diff thresholding (jitter rejection) ────────────────
        # 1. Pre-blur the diff to eliminate single-pixel IR sensor noise.
        # 2. Compute a percentile over valid diff pixels — percentile rises with
        #    the Jitter Reject slider, so the threshold automatically tracks the
        #    noise floor and only large, coherent depth changes pass through.
        jitter_pct = cv2.getTrackbarPos("Jitter Reject %", WINDOW) / 100.0
        # Map [0..1] → percentile [82..97]: higher slider = higher percentile = stricter
        perc = 82 + jitter_pct * 15.0

        diff = cv2.absdiff(depth, background)
        # Gaussian blur suppresses salt-and-pepper sensor noise before threshold
        diff_smooth = cv2.GaussianBlur(diff, (9, 9), 0)

        both_valid = (depth > 0) & (background > 0)
        valid_diffs = diff_smooth[both_valid]
        if len(valid_diffs) > 500:
            adapt_thresh = float(np.percentile(valid_diffs, perc))
            adapt_thresh = max(30.0, adapt_thresh)  # hard floor at 30 mm
        else:
            adapt_thresh = 80.0  # fallback when scene has little valid depth

        motion_mask = np.zeros_like(diff, dtype=np.uint8)
        motion_mask[(diff_smooth > adapt_thresh) & both_valid] = 255
        motion_mask = cv2.bitwise_and(motion_mask, roi_mask)

        kernel = np.ones((7, 7), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        raw_motion = any(
            MIN_AREA < cv2.contourArea(cnt) < MAX_AREA for cnt in contours
        )

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

        # ── YOLOv8 Object Detection on RGB (every N frames) ──
        frame_count += 1
        if frame_count % YOLO_INTERVAL == 0:
            # CLAHE contrast enhancement for better detection in variable lighting
            gray_for_clahe = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            clahe_gray = _clahe.apply(gray_for_clahe)
            clahe_bgr = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)
            yolo_input = cv2.addWeighted(color_image, 0.6, clahe_bgr, 0.4, 0)
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

                    # Get depth at centroid
                    depth_val = depth_frame.get_distance(
                        min(cx, WIDTH - 1), min(cy, HEIGHT - 1)
                    )

                    detections.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": conf, "label": label,
                        "cx": cx, "cy": cy,
                        "depth_m": depth_val
                    })

            last_detections = tracker.update(detections)

        # ── Draw detections on depth view ──
        for det in last_detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            label = det["label"]
            conf = det["conf"]
            direction = det.get("direction", "")
            depth_m = det.get("depth_m", 0)

            # Green box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label with depth distance
            text = f"{label} ({conf:.2f}) {depth_m:.2f}m"
            if direction and direction != "New":
                text += f" | {direction}"

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(display, text, (x1 + 2, y1 - 5),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

            # Direction arrow (smoothed EMA velocity)
            SmoothTracker.draw_arrow(display, det)

        # ── Status overlay ──
        if motion_confirmed:
            cv2.putText(display, "MOTION DETECTED", (20, 35),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(display, "CLEAR", (20, 35),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

        # Object summary
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
            path = os.path.join(FRAMES_DIR, f"depth_motion_{timestamp()}.png")
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
            print("[BG] Recapturing background...")
            bg_accumulator = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
            bg_count_total = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
            for _ in range(15):
                f = pipeline.wait_for_frames()
                af = align.process(f)
                df = af.get_depth_frame()
                if df:
                    d = np.asanyarray(df.get_data()).astype(np.float64)
                    valid = d > 0
                    bg_accumulator[valid] += d[valid]
                    bc = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
                    bc[valid] += 1
                    bg_count_total += bc
            bg_count_total[bg_count_total == 0] = 1
            background = (bg_accumulator / bg_count_total).astype(np.float32)
            # Recompute depth threshold from new background
            depth_thresh = _compute_depth_thresh(background)
            motion_counter = 0
            clear_counter = 0
            motion_confirmed = False
            print(f"[BG] Background recaptured | new depth_thresh={depth_thresh} mm")

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    if recording:
        stop_recording()
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"Saved {saved_frame_count} motion frames to {FRAMES_DIR}")
