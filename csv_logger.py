"""
Async CSV telemetry logger.
Writes one aggregate row every LOG_INTERVAL_SEC seconds.
Each row is a session-level snapshot, not per-detection.
"""

import csv
import os
import queue
import threading
import time
from datetime import datetime

LOG_INTERVAL_SEC = 1   # one row every second
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "telemetry")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIELDNAMES = [
    "timestamp", "frame_num", "mode",
    # Detection
    "obj_count", "obj_classes", "avg_confidence", "avg_depth_m",
    # Power
    "cpu_power_w", "gpu_power_w", "inst_power_w",
    "frame_power_mj", "pixel_power_uj",
    "total_energy_wh", "peak_power_w",
    "energy_per_frame_mj", "energy_per_object_mj",
    # Efficiency
    "fps_per_watt", "objects_per_joule", "yolo_efficiency",
    # Latency (ms)
    "frame_latency_ms", "capture_ms", "preprocess_ms",
    "yolo_ms", "track_ms", "render_ms",
    "bottleneck_stage",
    # FPS
    "fps", "fps_stability_std", "dropped_frames",
    # Resources
    "cpu_pct", "gpu_pct", "ram_mb", "gpu_mb",
    # Motion
    "motion_detected",
    # Tracking
    "tracking_stability_pct",
]


class CSVLogger:
    def __init__(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(OUTPUT_DIR, f"session_{ts}.csv")
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()
        # Accumulator for rolling stats between rows
        self._reset_accum()
        self._last_write = time.time()

    def _reset_accum(self):
        self._fps_samples = []
        self._conf_samples = []
        self._depth_samples = []
        self._obj_classes_seen = []

    def _writer(self):
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
            w.writeheader()
            f.flush()
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    row = self._queue.get(timeout=1.0)
                    w.writerow(row)
                    f.flush()
                except queue.Empty:
                    continue
        print(f"[CSV] Session saved to {self.path}")

    def maybe_write(self, stats: dict):
        """Call every frame. Writes a row every LOG_INTERVAL_SEC seconds."""
        # Accumulate rolling samples
        self._fps_samples.append(stats.get("fps", 0))
        c = stats.get("avg_confidence", 0)
        if c: self._conf_samples.append(c)
        d = stats.get("avg_depth_m", 0)
        if d: self._depth_samples.append(d)
        cls = stats.get("obj_classes", "")
        if cls: self._obj_classes_seen.append(cls)

        now = time.time()
        if now - self._last_write < LOG_INTERVAL_SEC:
            return
        self._last_write = now

        # Compute rolling aggregates
        import numpy as np
        fps_arr = np.array(self._fps_samples) if self._fps_samples else np.array([0])
        fps_std = round(float(np.std(fps_arr)), 2)

        stages = stats.get("stages", {})
        stage_vals = {s: stages.get(s, 0) for s in
                      ["capture","preprocess","yolo","track","render"]}
        bottleneck = max(stage_vals, key=stage_vals.get) if stage_vals else "—"

        row = {
            "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frame_num":           stats.get("frame_num", 0),
            "mode":                stats.get("mode", "rgb"),
            "obj_count":           stats.get("obj_count", 0),
            "obj_classes":         "|".join(set(self._obj_classes_seen)) or "—",
            "avg_confidence":      round(float(np.mean(self._conf_samples)), 3) if self._conf_samples else 0,
            "avg_depth_m":         round(float(np.mean(self._depth_samples)), 2) if self._depth_samples else 0,
            # Power
            "cpu_power_w":         stats.get("cpu_power_w", 0),
            "gpu_power_w":         stats.get("gpu_power_w", 0),
            "inst_power_w":        stats.get("inst_power_w", 0),
            "frame_power_mj":      stats.get("frame_power_mj", 0),
            "pixel_power_uj":      stats.get("pixel_power_uj", 0),
            "total_energy_wh":     stats.get("total_energy_wh", 0),
            "peak_power_w":        stats.get("peak_power_w", 0),
            "energy_per_frame_mj": stats.get("energy_per_frame_mj", 0),
            "energy_per_object_mj":stats.get("energy_per_object_mj", 0),
            # Efficiency
            "fps_per_watt":        stats.get("fps_per_watt", 0),
            "objects_per_joule":   stats.get("objects_per_joule", 0),
            "yolo_efficiency":     stats.get("yolo_efficiency", 0),
            # Latency
            "frame_latency_ms":    stats.get("frame_time_ms", 0),
            "capture_ms":          stage_vals.get("capture", 0),
            "preprocess_ms":       stage_vals.get("preprocess", 0),
            "yolo_ms":             stage_vals.get("yolo", 0),
            "track_ms":            stage_vals.get("track", 0),
            "render_ms":           stage_vals.get("render", 0),
            "bottleneck_stage":    bottleneck,
            # FPS
            "fps":                 stats.get("fps", 0),
            "fps_stability_std":   fps_std,
            "dropped_frames":      stats.get("dropped_frames", 0),
            # Resources
            "cpu_pct":             stats.get("cpu_pct", 0),
            "gpu_pct":             stats.get("gpu_pct", 0),
            "ram_mb":              stats.get("ram_mb", 0),
            "gpu_mb":              stats.get("gpu_mb", 0),
            # Status
            "motion_detected":     int(stats.get("motion", False)),
            "tracking_stability_pct": stats.get("tracking_stability_pct", 100),
        }
        self._queue.put(row)
        self._reset_accum()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
