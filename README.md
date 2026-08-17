# RealSense D415 Personnel Detection, Depth Motion Estimation & Telemetry Analytics Platform

A high-performance, real-time computer vision and physical security telemetry system engineered for **Intel RealSense D415 (RGB-D)** cameras. The platform integrates hardware-level sensor flicker suppression, dual-stream motion tracking, YOLOv8 AI object detection with 3D spatial depth estimation, Fast Fourier Transform (FFT) frequency analysis, and an interactive Flask telemetry dashboard featuring real-time energy profiling and pipeline latency decomposition.

---

## 🌟 Key System Capabilities

- **Dual-Stream RGB-D Alignment**: Synchronized 640x480 @ 30 FPS depth ($Z16$) and color ($BGR8$) alignment sharing a unified viewport using `pyrealsense2`.
- **Flicker & Ambient Noise Suppression (`DepthFlickerSuppressor`)**: Hardware-level auto-exposure disabling and manual integration window calibration to eliminate $50\text{ Hz} / 60\text{ Hz}$ AC mains ambient light flicker on raw IR depth sensors using Exponential Moving Average (EMA) smoothing.
- **Dual-Domain Motion Engine**:
  - **Depth Motion Detection**: Absolute millimeter-level spatial depth differencing ($|Z_{\text{curr}} - Z_{\text{bg}}| > \text{threshold}$) immune to shadows and lighting changes.
  - **RGB Motion Subtraction**: Adaptive MOG2 Gaussian background modeling with morphological opening/closing and shadow rejection.
- **YOLOv8 AI Personnel Detection & 3D Depth Mapping**: Real-time object detection powered by YOLOv8n paired with bounding-box spatial median depth extraction ($Z_{\text{meters}}$) for accurate distance estimation.
- **Direction & Trajectory Tracking**: 8-compass direction tracking (Up, Down, Left, Right, Diagonals) with exponential velocity smoothing ($\alpha = 0.22 - 0.35$) and hysteresis state filtering to prevent direction jitter.
- **Spectral Frequency Analysis (FFT)**: Temporal Fast Fourier Transform analysis on IR and RGB intensity series to isolate ambient lighting harmonics (100 Hz / 120 Hz) and generate high-resolution frequency spectrum plots.
- **Web Telemetry & Power Profiling Dashboard**: Real-time browser GUI (`http://localhost:5000`) visualizing live MJPEG video streams, CPU/GPU TDP wattage consumption, energy per frame ($\text{mJ/frame}$), energy per object ($\text{mJ/object}$), stage-by-stage latency breakdowns, and system efficiency metrics.
- **Asynchronous Telemetry Logging**: Multithreaded queue-based CSV logger capturing 40+ system metrics per second without blocking camera throughput.

---

## 🏗 System Architecture

```text
                               ┌─────────────────────────────────────────┐
                               │  Intel RealSense D415 Depth Camera      │
                               │  (Aligned RGB Stream + Depth Stream)     │
                               └────────────────────┬────────────────────┘
                                                    │
                        ┌───────────────────────────┴───────────────────────────┐
                        ▼                                                       ▼
      ┌───────────────────────────────────┐                   ┌───────────────────────────────────┐
      │  Raw IR / Depth Sensor Pipeline   │                   │        RGB Frame Processing       │
      │  - Exposure Flicker Suppression   │                   │  - MOG2 Background Subtraction    │
      │  - Millimeter Depth Differencing  │                   │  - Contour Tracking               │
      └─────────────────┬─────────────────┘                   └─────────────────┬─────────────────┘
                        │                                                       │
                        └───────────────────────────┬───────────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │   YOLOv8 AI Personnel Detection Engine  │
                               │   - Object Bounding Box Estimation       │
                               │   - 3D Depth Median Query (Z-distance)  │
                               │   - Direction Vector & Centroid Tracking │
                               └────────────────────┬────────────────────┘
                                                    │
                        ┌───────────────────────────┼───────────────────────────┐
                        ▼                           ▼                           ▼
       ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
       │   Spectral FFT Engine   │     │  Web Telemetry Server   │     │   Async CSV Telemetry   │
       │  (Flicker Plots & FFT)  │     │ (Flask Live Dashboard)  │     │   (Session Logs Queue)  │
       └─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

---

## 🔬 Hardware Sensor Calibration & Frequency Analysis

Artificial room lighting (fluorescent ballast, LED drivers) creates subtle high-frequency brightness oscillations driven by $50\text{ Hz}$ or $60\text{ Hz}$ power grids ($100\text{ Hz} / 120\text{ Hz}$ optical harmonics). Because Intel RealSense D415 depth cameras use active infrared projection and dual IR sensors, ambient flicker distorts raw depth readings.

### 1. Exposure-Based Flicker Suppression (`DepthFlickerSuppressor`)
- **Mechanism**: Auto-exposure is programmatically disabled on the RealSense depth sensor (`rs.option.enable_auto_exposure = 0`).
- **Integration Matching**: Exposure integration windows are locked to integer multiples of full lighting cycles ($\approx 20\text{ ms}$ for $50\text{ Hz}$, $\approx 16.6\text{ ms}$ for $60\text{ Hz}$).
- **Dynamic Calibration**: Runs a variance-minimization pass at startup across candidate exposure steps and re-evaluates ambient noise periodically using Exponential Moving Average (EMA) smoothing.

### 2. Spectral Analysis Modules
- **IR Noise Frequency Analysis (`frequency_analysis.py`)**: Collects a rolling temporal buffer of raw depth frame intensities, computes 1D Fast Fourier Transforms (FFT), and outputs power spectrum plots into `output/plots/ir_noise_final_*.png`.
- **RGB Frequency Analysis (`frequency_analysis_rgb.py`)**: Runs channel-separated (Red, Green, Blue) temporal FFT to monitor ambient lighting stability and color-channel noise profiles.

---

## 📊 Web Telemetry & Power Dashboard

The interactive Web Dashboard (`dashboard.py`) streams live camera feeds and real-time telemetry metrics to `http://localhost:5000`.

### Monitored Metrics & Telemetry Parameters:

| Category | Parameter | Description |
| :--- | :--- | :--- |
| **Detection** | `obj_count`, `avg_confidence`, `avg_depth_m` | Live count of detected personnel, mean bounding box confidence, and spatial depth distance in meters. |
| **Power Profile** | `cpu_power_w`, `gpu_power_w`, `inst_power_w` | Real-time estimated system CPU and GPU wattage based on TDP utilization models (`psutil`, `GPUtil`). |
| **Energy** | `frame_power_mj`, `pixel_power_uj`, `energy_per_object_mj` | Energy consumed per processed frame ($\text{mJ}$), micro-joules per pixel ($\mu\text{J}$), and energy required per detected object. |
| **Pipeline Latency**| `capture_ms`, `preprocess_ms`, `yolo_ms`, `track_ms`, `render_ms` | Stage-by-stage execution latency profiling with dynamic bottleneck identification (`bottleneck_stage`). |
| **Performance** | `fps`, `fps_stability_std`, `dropped_frames` | Real-time frame throughput, FPS standard deviation stability, and dropped frame counts. |
| **Tracking** | `tracking_stability_pct` | Percentage ratio of sustained centroid tracking across frame sequences. |

---

## 📁 Repository Structure

```text
personnel-detection/
├── main.py                     # Primary RealSense dual-stream viewer (RGB + Aligned Depth)
├── simple_depth_yolo.py        # YOLOv8 object detection with bounding-box spatial depth estimation
├── motion_depth.py             # Depth-differencing motion tracker with flicker suppression & direction logic
├── motion_rgb.py               # RGB Gaussian MOG2 background subtraction & motion tracker
├── frequency_analysis.py       # IR sensor temporal FFT spectrum analysis & exposure flicker suppressor
├── frequency_analysis_rgb.py   # RGB temporal channel-separated FFT analysis
├── dashboard.py                # Flask web dashboard with live MJPEG stream & real-time telemetry
├── csv_logger.py               # Multithreaded asynchronous CSV telemetry logger
├── profiler.py                 # Pipeline latency profiler & execution stage breakdown timer
├── requirements.txt            # Python dependencies (pyrealsense2, ultralytics, opencv, flask, etc.)
├── yolov8n.pt                  # Pre-trained YOLOv8 Nano model weights
├── .gitignore                  # Git ignore rules for virtualenvs, cache, and large video binaries
├── templates/
│   └── index.html              # HTML5/JS dashboard interface with dynamic telemetry charts
└── output/                     # Generated telemetry data & outputs
    ├── plots/                  # Hi-res FFT spectral analysis plots (.png)
    ├── telemetry/              # Session-level CSV telemetry log files
    └── video/                  # Video output documentation & README
```

---

## 🛠 Installation & Setup

### Prerequisites
- **Hardware**: Intel RealSense D415 Depth Camera (connected via USB 3.0+).
- **OS**: Windows 10/11 or Linux.
- **Python**: Python 3.8 to 3.11 recommended.
- **CUDA (Optional)**: NVIDIA GPU with CUDA drivers for accelerated YOLOv8 inference.

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rishi-msrit/personnel-detection.git
   cd personnel-detection
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Execution & Usage Guide

### 1. Launch Interactive Web Dashboard
Run the Flask telemetry server and open `http://localhost:5000` in your web browser:
```bash
python dashboard.py
```

### 2. Run Basic Dual-Stream Viewer
Visualize synchronized RGB and RealSense-colorized Depth streams:
```bash
python main.py
```
- **Controls**: Press `r` to toggle recording, `s` to save frame snapshots, `q` to quit.

### 3. Run YOLOv8 Depth Detection Pipeline
Execute real-time personnel detection with distance estimation:
```bash
python simple_depth_yolo.py
```

### 4. Run Depth Motion Detection & Tracking
Track spatial motion using depth differences (immune to ambient lighting changes):
```bash
python motion_depth.py
```
- **Controls**: Press `b` to re-capture background depth, `q` to quit.

### 5. Run Spectral Frequency Analysis (FFT)
Perform IR flicker diagnosis and output spectral graphs to `output/plots/`:
```bash
# IR Depth Flicker FFT Analysis:
python frequency_analysis.py

# RGB Flicker FFT Analysis:
python frequency_analysis_rgb.py
```

---

## 📈 Data Output & Logging

- **Telemetry Log CSVs**: Written to `output/telemetry/session_YYYYMMDD_HHMMSS.csv` containing 1-second interval system snapshots.
- **FFT Spectral Plots**: Saved as high-resolution PNGs in `output/plots/` detailing frequency amplitudes up to Nyquist limits ($15\text{ Hz}$ for 30 FPS feeds).
- **Video Recordings**: Documented in `output/video/README.md` (video binary files `.mp4`/`.avi` are excluded from Git repository tracking due to storage size limits).

---

## 📜 License

This project is licensed under the MIT License.
