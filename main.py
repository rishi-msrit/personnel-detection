"""
RealSense D415 — Dual Stream Viewer (RGB + Depth)
Stage 1: Basic Viewer with correct depth colorization matching RealSense Viewer.

Controls:
    r  — Start / stop video recording
    s  — Save snapshot (RGB + Depth frame pair)
    q  — Quit
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime

# ────────────────────────── Configuration ──────────────────────────
WIDTH, HEIGHT, FPS = 640, 480, 30
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
VIDEO_DIR = os.path.join(OUTPUT_DIR, "video")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

# ────────────────────────── Pipeline Setup ──────────────────────────
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

# Align depth to color so both frames share the same viewport
align = rs.align(rs.stream.color)

# Native colorizer — uses the same algorithm as the RealSense Viewer
colorizer = rs.colorizer()
# Color scheme 0 = Jet (default Viewer look: blue=near, red=far)
colorizer.set_option(rs.option.color_scheme, 0)

profile = pipeline.start(config)

# ────────────────────────── State ──────────────────────────
recording = False
video_writer = None
frame_count = 0
fps_counter = 0
fps_value = 0.0
fps_timer = time.time()

WINDOW = "RealSense D415 — RGB | Depth"


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def start_recording(w, h):
    global video_writer, recording
    path = os.path.join(VIDEO_DIR, f"viewer_{timestamp()}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(path, fourcc, FPS, (w, h))
    recording = True
    print(f"[REC] Recording started → {path}")


def stop_recording():
    global video_writer, recording
    if video_writer:
        video_writer.release()
        video_writer = None
    recording = False
    print("[REC] Recording stopped")


# ────────────────────────── Main Loop ──────────────────────────
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # RGB image — raw, no processing
        color_image = np.asanyarray(color_frame.get_data())

        # Depth image — colorized using the native RealSense colorizer
        depth_colorized = np.asanyarray(
            colorizer.colorize(depth_frame).get_data()
        )

        # FPS calculation
        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_value = fps_counter / elapsed
            fps_counter = 0
            fps_timer = time.time()

        # Overlay info
        info_color = color_image.copy()
        info_depth = depth_colorized.copy()

        cv2.putText(info_color, f"RGB  |  FPS: {fps_value:.1f}",
                     (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(info_depth, "Depth (RealSense Colorizer)",
                     (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if recording:
            cv2.circle(info_color, (WIDTH - 20, 25), 8, (0, 0, 255), -1)
            cv2.putText(info_color, "REC", (WIDTH - 60, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        combined = np.hstack((info_color, info_depth))

        # Record
        if recording and video_writer:
            video_writer.write(combined)

        cv2.imshow(WINDOW, combined)

        # ── Key handling ──
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            if recording:
                stop_recording()
            else:
                h, w = combined.shape[:2]
                start_recording(w, h)

        elif key == ord("s"):
            ts = timestamp()
            rgb_path = os.path.join(FRAMES_DIR, f"rgb_{ts}.png")
            dep_path = os.path.join(FRAMES_DIR, f"depth_{ts}.png")
            cv2.imwrite(rgb_path, color_image)
            cv2.imwrite(dep_path, depth_colorized)
            print(f"[SNAP] Saved → {rgb_path}, {dep_path}")

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

        frame_count += 1

finally:
    if recording:
        stop_recording()
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"Total frames processed: {frame_count}")
