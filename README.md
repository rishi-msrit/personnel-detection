# Personnel Detection and Telemetry Analytics System

A real-time computer vision system built for Intel RealSense D415 (RGB-D) cameras. It handles sensor flicker suppression, motion tracking, YOLOv8 object detection with 3D depth measurement, FFT-based frequency analysis for ambient noise, and a Flask dashboard for live monitoring and performance telemetry.

## Key Features

- Dual-Stream Alignment: Synchronized 640x480 at 30 FPS depth (Z16) and color (BGR8) streams aligned to a single viewport.
- Sensor Noise and Flicker Suppression: Manual exposure control and integration window matching to reduce 50 Hz / 60 Hz ambient lighting flicker on IR depth sensors.
- Motion Detection: Depth-differencing (millimeter thresholding) and RGB MOG2 background subtraction.
- Personnel Detection with Depth: YOLOv8 object detection combined with median depth calculation to estimate distance to detected objects.
- Object Tracking and Direction: Centroid tracking across 8 compass directions with velocity smoothing and hysteresis filtering.
- Frequency Analysis: Fast Fourier Transform (FFT) analysis on IR and RGB streams to identify flickering frequencies from artificial lights.
- Web Dashboard: Flask web interface streaming live camera views, CPU/GPU power estimates, frame latency breakdown, and system performance stats.
- CSV Logging: Asynchronous background telemetry logging to save session metrics to CSV files without affecting camera frame rates.

## System Architecture

```text
Input (Intel RealSense D415 RGB-D Camera)
  │
  ├── Depth Processing (Exposure calibration, depth difference filtering)
  ├── RGB Processing (MOG2 background subtraction)
  │
  └── YOLOv8 Detection Engine (Bounding boxes, median depth calculation, centroid tracking)
        │
        ├── Spectral FFT Engine (Flicker detection and plot generation)
        ├── Web Dashboard (Flask web server at http://localhost:5000)
        └── Telemetry Logger (Background CSV output)
```

## Sensor Flicker Suppression and Frequency Analysis

Fluorescent and LED lighting flicker at 100 Hz or 120 Hz depending on AC mains frequency. Since the Intel RealSense D415 uses active IR projection and dual IR sensors, ambient lighting flicker causes temporal noise in raw depth data.

### 1. Depth Flicker Suppressor (DepthFlickerSuppressor)
Auto-exposure is disabled on the depth sensor. Exposure integration windows are matched to full lighting cycles (~20 ms for 50 Hz, ~16.6 ms for 60 Hz). At startup, exposure values are evaluated to minimize temporal variance, with an exponential moving average (EMA) smoothing updates over time.

### 2. Spectral Analysis Modules
- frequency_analysis.py: Captures depth intensity sequences, applies 1D FFT, and outputs spectral plots to output/plots/.
- frequency_analysis_rgb.py: Performs channel-separated (RGB) FFT to detect flicker in color feeds.

## Web Telemetry and Performance Dashboard

The web dashboard (dashboard.py) streams video feeds to http://localhost:5000 and reports performance data.

Metrics tracked include:
- Detections: Object count, average confidence, average distance in meters.
- Power and Energy: Estimated CPU/GPU wattage, energy per frame (mJ), energy per object.
- Latency Breakdown: Execution time in ms for capture, preprocessing, YOLO inference, tracking, and rendering.
- System Stats: FPS, frame drop count, CPU usage, GPU usage, and RAM/VRAM usage.

## Repository Structure

- main.py: RealSense viewer with aligned RGB and depth streams.
- simple_depth_yolo.py: YOLOv8 object detection with depth estimation.
- motion_depth.py: Depth-based motion detection and tracking script.
- motion_rgb.py: RGB background subtraction and motion detection script.
- frequency_analysis.py: FFT analysis script for IR depth sensor noise.
- frequency_analysis_rgb.py: FFT analysis script for RGB light flicker.
- dashboard.py: Flask web server for live feeds and telemetry monitoring.
- csv_logger.py: Async background CSV logger for telemetry metrics.
- profiler.py: Execution stage timer and performance breakdown helper.
- requirements.txt: Python package dependencies.
- yolov8n.pt: YOLOv8 nano model weights file.
- .gitignore: Configuration for ignoring caches, venv, and large media.
- templates/index.html: HTML dashboard template.
- output/: Directory for generated plots, CSV logs, and video notes.

## Setup and Installation

Prerequisites:
- Intel RealSense D415 camera connected via USB 3.0
- Python 3.8 to 3.11
- Optional: NVIDIA GPU with CUDA support for YOLO acceleration

Steps:
1. Clone the repository:
   git clone https://github.com/rishi-msrit/personnel-detection.git
   cd personnel-detection

2. Set up virtual environment:
   python -m venv .venv
   .venv\Scripts\activate

3. Install requirements:
   pip install -r requirements.txt

## Running the Components

1. Web Dashboard:
   python dashboard.py
   Open http://localhost:5000 in a browser.

2. RGB + Depth Stream Viewer:
   python main.py
   Press 'r' to toggle video recording, 's' to save snapshots, 'q' to exit.

3. YOLO Detection with Distance Measurement:
   python simple_depth_yolo.py

4. Depth Motion Detection:
   python motion_depth.py
   Press 'b' to recalibrate background depth, 'q' to exit.

5. Frequency Analysis (Flicker Check):
   python frequency_analysis.py
   python frequency_analysis_rgb.py

## Output Data

- Telemetry CSV files are saved in output/telemetry/ with timestamps.
- Frequency analysis plots are saved as PNG images in output/plots/.
- Recorded video details are described in output/video/README.md.

## License

MIT
