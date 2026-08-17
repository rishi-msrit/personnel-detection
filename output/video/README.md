# Video Outputs Directory

Recorded video files are not committed to this repository due to GitHub file size limits and large media storage constraints.

## Expected Video Recordings

When running the surveillance and personnel detection system (via `main.py`, `simple_depth_yolo.py`, `motion_rgb.py`, or `motion_depth.py`), the following video outputs are generated and saved to this folder:

1. **Dual-Stream RealSense Recordings (`.mp4` / `.avi`)**:
   - **Synchronized RGB & Depth Streams**: Side-by-side or overlaid 640x480 @ 30 FPS video recordings matching Intel RealSense Viewer colorization standards (Jet colormap: blue=near, red=far).
2. **YOLOv8 Personnel Detection Recordings**:
   - Live bounding box overlay tracking personnel, confidence scores, and distance metrics derived from depth frame alignment.
3. **Motion & Spatial Analysis Videos**:
   - Frame-by-frame temporal motion heatmaps and depth spatial variance streams used for frequency profiling.
