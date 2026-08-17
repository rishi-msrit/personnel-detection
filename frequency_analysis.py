"""
RealSense D415 — Infrared Noise Frequency Analysis
Captures the IR stream and shows the FFT spectrum to identify
noise frequencies caused by artificial lighting (LED, fluorescent, etc.).

The IR sensor is directly affected by ambient light flicker —
this script isolates and visualizes that noise.

Exposure-Based Flicker Suppression (NEW):
    Auto-exposure is disabled and the IR sensor is locked to an
    exposure value whose integration window spans one or more complete
    lighting cycles (1/50 Hz ≈ 20 ms, 1/60 Hz ≈ 16.6 ms).
    A short calibration phase selects the candidate that minimises
    frame-to-frame intensity variance; the choice is re-evaluated
    every few seconds using exponential smoothing so the system
    adapts to changing ambient conditions without abrupt jumps.

Controls:
    q  — Quit
    s  — Save snapshot + hi-res plot

Output folder: output/
    video/    — 8s recorded video
    frames/   — individual frame snapshots
    plots/    — hi-res FFT plot
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import io
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

# ────────────────────────── Configuration ──────────────────────────
WIDTH, HEIGHT, FPS = 640, 480, 30
BUFFER_SIZE = 200
RECORD_SECONDS = 8
MAX_SAVED_FRAMES = 200

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames", "ir_noise")
VIDEO_DIR = os.path.join(OUTPUT_DIR, "video")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# ────────────────────────── Pipeline Setup ──────────────────────────
pipeline = rs.pipeline()
config = rs.config()
# Enable Depth stream (flicker noise impacts raw depth readings)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
profile = pipeline.start(config)

# Native colorizer for Viewer-matching visualization
colorizer = rs.colorizer()
colorizer.set_option(rs.option.color_scheme, 0)
# ── Disable Depth auto-exposure and lock manual control ──────────────
# We must set options on the *depth* sensor because the IR projector
# and both IR imagers share the same sensor on the D415.
_depth_sensor = profile.get_device().first_depth_sensor()
if _depth_sensor.supports(rs.option.enable_auto_exposure):
    _depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
    print("[DEPTH-EXP] Auto-exposure DISABLED — switching to manual control.")
else:
    print("[DEPTH-EXP] WARNING: auto-exposure option not supported on this device.")

# ────────────────────────── Exposure Suppressor ──────────────────────────

class DepthFlickerSuppressor:
    """
    Selects the Depth exposure value that minimises temporal flicker.

    Strategy
    --------
    Candidate exposures (µs) are aligned with common mains-lighting
    cycle lengths:
        8 000 µs  → ~8 ms  (sub-cycle, safe lower bound)
       10 000 µs  → 10 ms
       16 600 µs  → ≈1 full 60 Hz cycle (16.67 ms)
       20 000 µs  → 1 full 50 Hz cycle (20 ms)

    For each candidate a short burst of frames is captured and the
    standard deviation of mean-intensity across those frames is
    measured.  The exposure with the smallest std-dev wins.

    The winning value is re-evaluated periodically.  Transitions are
    softened with exponential smoothing:
        exposure_applied = α * exposure_current + (1-α) * exposure_new
    so that corrections are gradual and do not themselves introduce
    a transient.
    """

    # Candidate exposures in microseconds — cover 50 Hz and 60 Hz cycles
    CANDIDATE_EXPOSURES_US = [8000, 10000, 16600, 20000]

    # Number of Depth frames captured per candidate during calibration
    # (frequency_analysis.py uses the Depth stream directly)
    EVAL_FRAMES = 20

    # Re-evaluate every 60 wall-clock seconds (frame-count is unreliable
    # when calibration itself reduces effective FPS)
    REEVAL_INTERVAL_SECS = 60.0

    # EMA weight for smooth transitions (0.8 → keep 80 % of current value)
    EMA_ALPHA = 0.8

    def __init__(self, depth_sensor):
        self._sensor = depth_sensor
        self._current_exposure_us = self.CANDIDATE_EXPOSURES_US[0]
        self._last_eval_time = 0.0   # epoch seconds of last calibration
        self._calibrated = False
        self._status_text = "Depth-only | Exposure: calibrating…"

    # ── Internal helpers ──────────────────────────────────────────

    def _set_exposure(self, exposure_us: int):
        """Push an exposure value to the hardware (clamped to valid range)."""
        try:
            opt = rs.option.exposure
            if self._sensor.supports(opt):
                r = self._sensor.get_option_range(opt)
                clamped = int(max(r.min, min(r.max, exposure_us)))
                self._sensor.set_option(opt, clamped)
                return clamped
        except Exception as exc:
            print(f"[DEPTH-EXP] set_exposure failed: {exc}")
        return exposure_us

    def _measure_variance(self, pipeline_ref, n_frames: int) -> float:
        """
        Capture *n_frames* Depth frames and return the temporal std-dev of
        per-frame mean intensity.  Lower is better (less flicker).
        """
        intensities = []
        for _ in range(n_frames):
            try:
                fs = pipeline_ref.wait_for_frames(timeout_ms=300)
                df = fs.get_depth_frame()
                if df:
                    img = np.asanyarray(df.get_data()).astype(np.float32)
                    valid = img[img > 0]
                    if len(valid) > 100:
                        intensities.append(float(np.mean(valid)))
            except Exception:
                pass  # skip dropped frames
        if len(intensities) < 4:
            return float("inf")
        return float(np.std(intensities))

    # ── Public API ───────────────────────────────────────────────

    def calibrate(self, pipeline_ref):
        """
        Evaluate all exposure candidates and lock to the best one.
        Called automatically on first iteration and every REEVAL_INTERVAL_FRAMES.
        """
        print("[DEPTH-EXP] Running exposure calibration…")
        results = {}
        for exp_us in self.CANDIDATE_EXPOSURES_US:
            applied = self._set_exposure(exp_us)
            # Allow the sensor to settle (≥2 frames at 30 fps ≈ 67 ms)
            time.sleep(0.08)
            std = self._measure_variance(pipeline_ref, self.EVAL_FRAMES)
            results[applied] = std
            print(f"         exposure={applied:>6} µs  →  std={std:.4f}")

        best_exp = min(results, key=results.get)
        print(f"[DEPTH-EXP] Selected exposure: {best_exp} µs "
              f"(std={results[best_exp]:.4f})")

        # EMA blend: if already calibrated, smooth the transition
        if self._calibrated:
            blended = int(
                self.EMA_ALPHA * self._current_exposure_us
                + (1.0 - self.EMA_ALPHA) * best_exp
            )
            print(f"[DEPTH-EXP] EMA blend: {self._current_exposure_us} → "
                  f"{blended} µs (target={best_exp})")
            applied = self._set_exposure(blended)
            self._current_exposure_us = applied
        else:
            applied = self._set_exposure(best_exp)
            self._current_exposure_us = applied
            self._calibrated = True

        self._status_text = (
            f"Depth-only | Exp: {self._current_exposure_us} µs "
            f"(std={results[best_exp]:.3f})"
        )
        self._last_eval_time = time.time()

    def tick(self, pipeline_ref):
        """
        Called once per main-loop iteration (BEFORE frame intensity is read).
        Uses wall-clock time so the 60 s interval is accurate even when
        calibration itself reduces effective FPS.
        """
        now = time.time()
        if not self._calibrated or (
            now - self._last_eval_time >= self.REEVAL_INTERVAL_SECS
        ):
            self.calibrate(pipeline_ref)

    @property
    def status_text(self) -> str:
        """One-line status string suitable for OSD overlay."""
        return self._status_text


# ── Instantiate the suppressor (applied before any Depth data is read) ──
suppressor = DepthFlickerSuppressor(_depth_sensor)

# ────────────────────────── State ──────────────────────────
depth_intensities = []
timestamps_sec = []

frame_idx = 0
start_time = None
saved_frame_count = 0

video_writer = None
recording = True
record_start_time = None

WINDOW = "Depth Noise Analysis - LED Flicker Detection"
PLOT_W = 640
PLOT_H = 300  # plot height
DEPTH_H = 240    # Depth feed height

cached_plot = np.zeros((PLOT_H, PLOT_W, 3), dtype=np.uint8)


def fig_to_cv2(fig):
    """Render matplotlib figure to BGR OpenCV image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.2,
                facecolor=fig.get_facecolor(), dpi=100)
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def render_noise_plot(depth_sig, actual_fps, n):
    """Render Depth noise FFT plot — two panels: time domain + frequency spectrum."""
    fig, (ax_time, ax_fft) = plt.subplots(1, 2, figsize=(8, 3))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Depth Noise Analysis — LED Flicker Detection",
                  color="white", fontsize=10, fontweight="bold")

    t = np.arange(n) / actual_fps

    # ── Time domain: Depth intensity fluctuation ──
    mean_val = np.mean(depth_sig)
    ax_time.set_facecolor("#161b22")
    ax_time.plot(t, depth_sig, color="#58a6ff", linewidth=0.6, alpha=0.8)
    ax_time.axhline(mean_val, color="#f85149", linestyle="--", linewidth=0.5,
                     alpha=0.5, label=f"Mean: {mean_val:.1f}")
    ax_time.set_title("Depth Intensity Over Time", color="white", fontsize=9)
    ax_time.set_xlabel("Time (s)", color="gray", fontsize=7)
    ax_time.set_ylabel("Mean Pixel Intensity", color="gray", fontsize=7)
    ax_time.tick_params(colors="gray", labelsize=6)
    ax_time.legend(fontsize=6, facecolor="#161b22", edgecolor="gray",
                    labelcolor="white")
    ax_time.grid(True, alpha=0.15, color="gray")

    # ── FFT: Frequency spectrum ──
    centered = np.array(depth_sig) - mean_val

    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(n)
    windowed = centered * window

    fft_result = np.abs(np.fft.fft(windowed))[:n // 2]
    freqs = np.fft.fftfreq(n, d=1.0 / actual_fps)[:n // 2]

    # Skip DC and very low freq (< 0.5 Hz)
    start_idx = max(1, int(0.5 / (actual_fps / n))) if n > 1 else 1

    ax_fft.set_facecolor("#161b22")
    if len(fft_result) > start_idx:
        ax_fft.plot(freqs[start_idx:], fft_result[start_idx:],
                     color="#58a6ff", linewidth=1.0)

        # Fill under curve for visual clarity
        ax_fft.fill_between(freqs[start_idx:], fft_result[start_idx:],
                             alpha=0.15, color="#58a6ff")

        # Mark dominant frequency (noise peak)
        dom_idx = start_idx + np.argmax(fft_result[start_idx:])
        dom_freq = freqs[dom_idx]
        dom_mag = fft_result[dom_idx]
        ax_fft.axvline(dom_freq, color="#f85149", linestyle="--",
                        linewidth=1.0, alpha=0.9)
        ax_fft.annotate(f"  {dom_freq:.2f} Hz",
                          xy=(dom_freq, dom_mag),
                          color="#f85149", fontsize=8, fontweight="bold")

        ax_fft.set_title(f"Frequency Spectrum | Noise Peak: {dom_freq:.2f} Hz",
                          color="white", fontsize=9)
    else:
        ax_fft.set_title("Frequency Spectrum | collecting...",
                          color="white", fontsize=9)

    ax_fft.set_xlabel("Frequency (Hz)", color="gray", fontsize=7)
    ax_fft.set_ylabel("Magnitude", color="gray", fontsize=7)
    ax_fft.tick_params(colors="gray", labelsize=6)
    ax_fft.grid(True, alpha=0.15, color="gray")

    plt.tight_layout(pad=0.8)
    plot_bgr = fig_to_cv2(fig)
    plt.close(fig)
    return cv2.resize(plot_bgr, (PLOT_W, PLOT_H))


def identify_flicker_source(freq):
    """Link a detected frequency to its most likely source."""
    if freq < 0.5:
        return "Sensor drift / thermal noise"
    elif 0.5 <= freq < 2.0:
        return "Slow ambient light variation (cloud cover, dimming)"
    elif abs(freq - 8.33) < 1.0:
        return f"Aliased 50Hz mains flicker (50Hz aliased at {30}fps camera)"
    elif abs(freq - 10.0) < 1.0:
        return f"Aliased 60Hz mains flicker (60Hz aliased at {30}fps camera)"
    elif 2.0 <= freq < 5.0:
        return "Low-freq LED PWM dimming or flickering fluorescent"
    elif 5.0 <= freq < 15.0:
        return f"Periodic interference ({freq:.1f}Hz) — likely aliased LED/PWM flicker"
    else:
        return f"Unknown periodic source at {freq:.2f} Hz"


def save_hires_plot(depth_sig, actual_fps, n, path):
    """Save a high-res version of the FFT plot with source explanation."""
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    ax_time = fig.add_subplot(gs[0, 0])
    ax_fft = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, :])

    fig.suptitle(
        f"Temporal Frequency Analysis of Depth Image Sequence for Flicker Detection\n"
        f"({n} frames @ {actual_fps:.1f} FPS | Nyquist limit: {actual_fps/2:.1f} Hz)",
        fontsize=14, fontweight="bold"
    )

    t = np.arange(n) / actual_fps
    mean_val = np.mean(depth_sig)

    # Time domain
    ax_time.plot(t, depth_sig, color="#2196F3", linewidth=0.5)
    ax_time.axhline(mean_val, color="red", linestyle="--", linewidth=0.5,
                     alpha=0.5, label=f"Mean: {mean_val:.1f}")
    ax_time.set_title("Temporal Signal: Mean Depth Intensity Per Frame")
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Mean Pixel Intensity")
    ax_time.legend()
    ax_time.grid(True, alpha=0.3)

    # FFT
    centered = np.array(depth_sig) - mean_val
    window = np.hanning(n)
    windowed = centered * window
    fft_result = np.abs(np.fft.fft(windowed))[:n // 2]
    freqs = np.fft.fftfreq(n, d=1.0 / actual_fps)[:n // 2]
    si = max(1, int(0.5 / (actual_fps / n))) if n > 1 else 1

    ax_fft.plot(freqs[si:], fft_result[si:], color="#2196F3", linewidth=1.0)
    ax_fft.fill_between(freqs[si:], fft_result[si:], alpha=0.1, color="#2196F3")

    dom_freq = 0
    source_text = ""
    if len(fft_result) > si:
        di = si + np.argmax(fft_result[si:])
        dom_freq = freqs[di]
        dom_mag = fft_result[di]
        source_text = identify_flicker_source(dom_freq)
        ax_fft.axvline(dom_freq, color="red", linestyle="--", alpha=0.8,
                        label=f"Dominant: {dom_freq:.2f} Hz")
        ax_fft.annotate(f"  {dom_freq:.2f} Hz\n  {source_text}",
                         xy=(dom_freq, dom_mag * 0.9),
                         fontsize=9, color="red", fontweight="bold")
        ax_fft.legend(fontsize=11)
    ax_fft.set_title("Frequency Spectrum (Hanning Windowed FFT)")
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Magnitude")
    ax_fft.grid(True, alpha=0.3)

    # Explanation panel
    ax_info.axis("off")
    explanation = (
        f"METHOD: Extracted mean valid depth from each frame to construct a temporal signal "
        f"across {n} frames ({n/actual_fps:.1f}s). Applied Hanning window and computed FFT.\n"
        f"RESULT: Dominant frequency = {dom_freq:.2f} Hz | Likely source: {source_text}\n"
        f"NOTE: Camera samples at {actual_fps:.0f} FPS → Nyquist limit = {actual_fps/2:.1f} Hz. "
        f"Mains flicker at 50/60Hz appears aliased. Real-world LED PWM (100-1000Hz) cannot be "
        f"resolved at this frame rate — only low-frequency harmonics or aliased components are visible.\n"
        f"COMMON SOURCES: 50Hz regions → ~8.33Hz alias at 30fps | "
        f"60Hz regions → ~10Hz alias at 30fps | LED PWM → variable low-freq aliases"
    )
    ax_info.text(0.02, 0.5, explanation, transform=ax_info.transAxes,
                  fontsize=9, verticalalignment="center", fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                            edgecolor="gray", alpha=0.9))

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ────────────────────────── Main Loop ──────────────────────────
print("[DEPTH] Starting depth noise frequency analysis...")
print("[DEPTH] Keep the scene STATIC — we are measuring ambient light noise.")
print(f"[DEPTH] Auto-recording {RECORD_SECONDS}s video.\n")
print("Controls:  q = quit  |  s = save snapshot + hi-res plot\n")

try:
    while True:
        # ── [1] EXPOSURE CONTROL — applied BEFORE any frame is read ──────
        # The suppressor checks whether calibration is needed (first run or
        # periodic re-evaluation) and sets the hardware exposure accordingly.
        # This must happen before wait_for_frames() so the *next* frame
        # already uses the corrected integration window.
        suppressor.tick(pipeline)

        # ── [2] Capture depth frame ─────────────────────────────────────────
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        if start_time is None:
            start_time = frames.get_timestamp() / 1000.0
        current_time = frames.get_timestamp() / 1000.0 - start_time
        timestamps_sec.append(current_time)

        # ── [3] Depth extraction (post-exposure correction) ───────
        d_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        valid = d_raw[d_raw > 0]
        if len(valid) > 100:
            depth_intensities.append(float(np.mean(valid)))
        else:
            depth_intensities.append(0.0)

        # Rolling buffer
        if len(depth_intensities) > BUFFER_SIZE:
            depth_intensities.pop(0)
            timestamps_sec.pop(0)

        frame_idx += 1
        n = len(depth_intensities)

        # ── [4] Depth feed display ──────────────────────────────────────────
        depth_colorized = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        viz_display = cv2.resize(depth_colorized, (PLOT_W, DEPTH_H))

        cv2.putText(viz_display, "Depth Stream (Colorized)", (10, 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(viz_display, f"Mean: {depth_intensities[-1]:.1f}", (10, DEPTH_H - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Overlay exposure suppressor status
        cv2.putText(viz_display, suppressor.status_text,
                     (10, DEPTH_H - 28),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 215, 255), 1)

        # ── [5] Update FFT plot every 10 frames ─────────────────────────
        if n >= 10 and frame_idx % 10 == 0:
            actual_fps = (n - 1) / max(timestamps_sec[-1] - timestamps_sec[0], 0.001)
            try:
                cached_plot = render_noise_plot(
                    list(depth_intensities), actual_fps, n
                )
            except Exception as e:
                print(f"[WARN] Plot error: {e}")
        elif n < 10:
            placeholder = np.zeros((PLOT_H, PLOT_W, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"Collecting Depth data... {n}/{BUFFER_SIZE}",
                         (PLOT_W // 4, PLOT_H // 2),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
            cached_plot = placeholder

        # Stack: Depth feed on top, FFT plot on bottom
        combined = np.vstack((viz_display, cached_plot))
        h_c, w_c = combined.shape[:2]

        # ── Video recording ──
        if recording:
            if record_start_time is None:
                record_start_time = time.time()
                ts = timestamp()
                vid_path = os.path.join(VIDEO_DIR, f"depth_noise_{ts}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(vid_path, fourcc, FPS, (w_c, h_c))
                print(f"[REC] Recording → {vid_path}")

            if video_writer:
                video_writer.write(combined)

            elapsed_rec = time.time() - record_start_time
            cv2.circle(combined, (w_c - 20, 15), 6, (0, 0, 255), -1)
            cv2.putText(combined, f"REC {elapsed_rec:.1f}s/{RECORD_SECONDS}s",
                         (w_c - 170, 20),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            if elapsed_rec >= RECORD_SECONDS:
                video_writer.release()
                video_writer = None
                recording = False
                print(f"[REC] Done — {RECORD_SECONDS}s video saved.")

        # ── Save individual frames ──
        if saved_frame_count < MAX_SAVED_FRAMES and frame_idx % 3 == 0:
            fpath = os.path.join(FRAMES_DIR, f"ir_noise_{frame_idx:05d}.png")
            cv2.imwrite(fpath, combined)
            saved_frame_count += 1

        cv2.imshow(WINDOW, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts = timestamp()
            snap_path = os.path.join(FRAMES_DIR, f"snapshot_{ts}.png")
            cv2.imwrite(snap_path, combined)
            print(f"[SNAP] Frame → {snap_path}")
            if n >= 10:
                actual_fps = (n - 1) / max(timestamps_sec[-1] - timestamps_sec[0], 0.001)
                hires_path = os.path.join(PLOTS_DIR, f"ir_noise_plot_{ts}.png")
                save_hires_plot(list(ir_intensities), actual_fps, n, hires_path)
                print(f"[SNAP] Hi-res plot → {hires_path}")

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    if video_writer:
        video_writer.release()
    pipeline.stop()
    cv2.destroyAllWindows()

    # Save final hi-res plot
    n = len(ir_intensities)
    if n >= 10:
        actual_fps = (n - 1) / max(timestamps_sec[-1] - timestamps_sec[0], 0.001)
        final_path = os.path.join(PLOTS_DIR, f"ir_noise_final_{timestamp()}.png")
        save_hires_plot(list(ir_intensities), actual_fps, n, final_path)
        print(f"\n[IR] Final plot → {final_path}")

        # Print detailed summary
        centered = np.array(ir_intensities) - np.mean(ir_intensities)
        window = np.hanning(n)
        windowed = centered * window
        fft_result = np.abs(np.fft.fft(windowed))[:n // 2]
        freqs = np.fft.fftfreq(n, d=1.0 / actual_fps)[:n // 2]
        si = max(1, int(0.5 / (actual_fps / n)))
        if len(fft_result) > si:
            di = si + np.argmax(fft_result[si:])
            dom_freq = freqs[di]
            source = identify_flicker_source(dom_freq)
            print(f"\n{'='*60}")
            print(f"  TEMPORAL FREQUENCY ANALYSIS SUMMARY")
            print(f"{'='*60}")
            print(f"  Frames analyzed : {n}")
            print(f"  Effective FPS   : {actual_fps:.2f}")
            print(f"  Nyquist limit   : {actual_fps/2:.2f} Hz")
            print(f"  Freq resolution : {actual_fps/n:.3f} Hz")
            print(f"  Dominant freq   : {dom_freq:.2f} Hz")
            print(f"  Magnitude       : {fft_result[di]:.1f}")
            print(f"  Likely source   : {source}")
            print(f"  Active exposure : {suppressor._current_exposure_us} µs")
            print(f"{'='*60}")
            print(f"  NOTE: 50Hz mains aliased at 30fps → ~8.33Hz")
            print(f"        60Hz mains aliased at 30fps → ~10.0Hz")
            print(f"{'='*60}\n")

    print(f"[IR] Saved {saved_frame_count} frames to {FRAMES_DIR}")
    print("[IR] Done.")
