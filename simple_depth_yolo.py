import pyrealsense2 as rs
import numpy as np
import cv2
import time
from ultralytics import YOLO

print("[YOLO] Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")
print("[YOLO] Model loaded.")

WIDTH, HEIGHT, FPS = 640, 480, 30

def get_direction(dx, dy):
    """Convert centroid displacement to a direction string."""
    if abs(dx) < 3 and abs(dy) < 3:
        return "Stationary"
    angle = np.degrees(np.arctan2(-dy, dx))
    if -22.5 <= angle < 22.5:
        return "Right"
    elif 22.5 <= angle < 67.5:
        return "Up-Right"
    elif 67.5 <= angle < 112.5:
        return "Up"
    elif 112.5 <= angle < 157.5:
        return "Up-Left"
    elif angle >= 157.5 or angle < -157.5:
        return "Left"
    elif -157.5 <= angle < -112.5:
        return "Down-Left"
    elif -112.5 <= angle < -67.5:
        return "Down"
    elif -67.5 <= angle < -22.5:
        return "Down-Right"
    return ""

ARROW_DIRS = {
    "Right": (1, 0), "Left": (-1, 0), "Up": (0, -1), "Down": (0, 1),
    "Up-Right": (1, -1), "Up-Left": (-1, -1),
    "Down-Right": (1, 1), "Down-Left": (-1, 1),
}

def match_detections_to_tracks(detections, prev_cents, max_dist=80):
    global track_id_counter
    new_centroids = {}
    for det in detections:
        cx, cy = det["cx"], det["cy"]
        best_id = None
        best_dist = max_dist

        for tid, (px, py) in prev_cents.items():
            dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_id = tid

        if best_id is not None:
            new_centroids[best_id] = (cx, cy)
            px, py = prev_cents[best_id]
            dx, dy = cx - px, cy - py
            det["track_id"] = best_id
            det["direction"] = get_direction(dx, dy)
        else:
            track_id_counter += 1
            new_centroids[track_id_counter] = (cx, cy)
            det["track_id"] = track_id_counter
            det["direction"] = "New"

    return new_centroids, detections

track_id_counter = 0

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

align = rs.align(rs.stream.color)
profile = pipeline.start(config)

colorizer = rs.colorizer()
colorizer.set_option(rs.option.color_scheme, 0) # 0 is Jet colormap

prev_centroids = {}
WINDOW = "Simple Depth YOLO Tracking"
cv2.namedWindow(WINDOW)

print("\n--- Starting pipeline ---")
print("Press 'q' or close the window to exit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
            
        color_image = np.asanyarray(color_frame.get_data())
        depth_colorized = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        
        display = depth_colorized.copy()
        
        results = model(color_image, verbose=False, conf=0.30, iou=0.45)
        detections = []
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                depth_val = depth_frame.get_distance(min(cx, WIDTH - 1), min(cy, HEIGHT - 1))
                
                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": conf, "label": label, "cx": cx, "cy": cy,
                    "depth_m": depth_val
                })
                
        prev_centroids, tracked_detections = match_detections_to_tracks(detections, prev_centroids)
        
        for det in tracked_detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            text = f"{det['label']} ({det['conf']:.2f}) {det['depth_m']:.2f}m"
            direction = det.get('direction', '')
            if direction and direction != "New":
                text += f" | {direction}"
                
            # Draw nice text background and text
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display, (x1, max(y1 - th - 5, 0)), (x1 + tw, max(y1, th + 5)), (0, 255, 0), -1)
            cv2.putText(display, text, (x1, max(y1 - 3, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw an arrow for direction
            cx, cy = det["cx"], det["cy"]
            cv2.circle(display, (cx, cy), 3, (0, 0, 255), -1)
            if direction in ARROW_DIRS:
                adx, ady = ARROW_DIRS[direction]
                arrow_len = 30
                end_x = cx + adx * arrow_len
                end_y = cy + ady * arrow_len
                cv2.arrowedLine(display, (cx, cy), (end_x, end_y), (0, 200, 255), 2, tipLength=0.4)
            
        cv2.imshow(WINDOW, display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
