# Personnel Detection & Surveillance Analytics

Real-time personnel detection, motion depth estimation, and spatial surveillance analytics system using Intel RealSense D415 depth cameras, YOLOv8, and Flask.

## Features

- **Dual-Stream RealSense Viewer**: Synchronized RGB and depth stream visualization.
- **YOLOv8 Detection**: Real-time object and personnel detection using RGB + Depth alignment.
- **Motion & Depth Estimation**: Motion tracking and frequency analysis across RGB and depth layers.
- **Web Dashboard**: Interactive web GUI powered by Flask and Chart.js for telemetry and visual stream management.
- **Data Logging**: Automated CSV logging for telemetry, motion frequency, and spatial profiles.

## Project Structure

```text
├── csv_logger.py           # Telemetry and event logging utility
├── dashboard.py            # Flask web server dashboard
├── frequency_analysis.py   # Depth frequency spectrum & temporal analysis
├── frequency_analysis_rgb.py# RGB temporal motion frequency analysis
├── main.py                 # Core Intel RealSense dual-stream viewer
├── motion_depth.py         # Depth motion tracking pipeline
├── motion_rgb.py           # RGB motion detection pipeline
├── profiler.py             # Performance benchmark & profiling helper
├── requirements.txt        # Python package dependencies
├── simple_depth_yolo.py    # YOLOv8 object detection with depth estimation
└── templates/
    └── index.html          # Web dashboard user interface
```

## Setup & Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rishi-msrit/personnel-detection.git
   cd personnel-detection
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Main Viewer**:
   ```bash
   python main.py
   ```

4. **Launch Web Dashboard**:
   ```bash
   python dashboard.py
   ```
