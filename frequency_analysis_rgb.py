"""
RealSense D415 — RGB Frequency Analysis for Flicker Detection
Captures the RGB stream and performs temporal FFT on mean intensity
to identify noise frequencies caused by artificial lighting.

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
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames", "rgb_noise")
VIDEO_DIR = os.path.join(OUTPUT_DIR, "video")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# ────────────────────────── Pipeline Setup ──────────────────────────
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
pipeline.start(config)

# ────────────────────────── State ──────────────────────────
rgb_intensities = []
r_intensities = []
g_intensities = []
b_intensities = []
timestamps_sec = []

frame_idx = 0
start_time = None
saved_frame_count = 0

video_writer = None
recording = True
record_start_time = None

WINDOW = "RGB Noise Analysis - LED Flicker Detection"
PLOT_W = 640
PLOT_H = 360
CAM_H = 240

cached_plot = np.zeros((PLOT_H, PLOT_W, 3), dtype=np.uint8)


def fig_to_cv2(fig):
    """Render matplotlib figure to BGR OpenCV image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.2,
                facecolor=fig.get_facecolor(), dpi=80)
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


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


def render_noise_plot(rgb_sig, r_sig, g_sig, b_sig, actual_fps, n):
    """Render RGB noise FFT plot — time domain + frequency spectrum."""
    fig, (ax_time, ax_fft) = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("RGB Temporal Noise Analysis — LED Flicker Detection",
                  color="white", fontsize=10, fontweight="bold")

    t = np.arange(n) / actual_fps

    # ── Time domain: per-channel intensity ──
    ax_time.set_facecolor("#161b22")
    ax_time.plot(t, r_sig, color="#ff6b6b", linewidth=0.5, alpha=0.6, label="R")
    ax_time.plot(t, g_sig, color="#51cf66", linewidth=0.5, alpha=0.6, label="G")
    ax_time.plot(t, b_sig, color="#339af0", linewidth=0.5, alpha=0.6, label="B")
    ax_time.plot(t, rgb_sig, color="white", linewidth=0.8, alpha=0.9, label="Gray")
    ax_time.set_title("Per-Channel Intensity Over Time", color="white", fontsize=9)
    ax_time.set_xlabel("Time (s)", color="gray", fontsize=7)
    ax_time.set_ylabel("Mean Intensity", color="gray", fontsize=7)
    ax_time.tick_params(colors="gray", labelsize=6)
    ax_time.legend(fontsize=5, facecolor="#161b22", edgecolor="gray",
                    labelcolor="white", loc="upper right")
    ax_time.grid(True, alpha=0.15, color="gray")

    # ── FFT: Grayscale frequency spectrum ──
    mean_val = np.mean(rgb_sig)
    centered = np.array(rgb_sig) - mean_val
    window = np.hanning(n)
    windowed = centered * window

    fft_result = np.abs(np.fft.fft(windowed))[:n // 2]
    freqs = np.fft.fftfreq(n, d=1.0 / actual_fps)[:n // 2]
    start_idx = max(1, int(0.5 / (actual_fps / n))) if n > 1 else 1

    ax_fft.set_facecolor("#161b22")
    if len(fft_result) > start_idx:
        ax_fft.plot(freqs[start_idx:], fft_result[start_idx:],
                     color="#74c0fc", linewidth=1.0)
        ax_fft.fill_between(freqs[start_idx:], fft_result[start_idx:],
                             alpha=0.15, color="#74c0fc")

        dom_idx = start_idx + np.argmax(fft_result[start_idx:])
        dom_freq = freqs[dom_idx]
        ax_fft.axvline(dom_freq, color="#f85149", linestyle="--",
                        linewidth=1.0, alpha=0.9)
        ax_fft.annotate(f"  {dom_freq:.2f} Hz",
                          xy=(dom_freq, fft_result[dom_idx]),
                          color="#f85149", fontsize=8, fontweight="bold")
        ax_fft.set_title(f"Frequency Spectrum | Peak: {dom_freq:.2f} Hz",
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


def save_hires_plot(rgb_sig, r_sig, g_sig, b_sig, actual_fps, n, path):
    """Save a high-res FFT plot with per-channel analysis and source explanation."""
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    ax_time = fig.add_subplot(gs[0, 0])
    ax_fft = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, :])

    fig.suptitle(
        f"Temporal Frequency Analysis of RGB Image Sequence for Flicker Detection\n"
        f"({n} frames @ {actual_fps:.1f} FPS | Nyquist limit: {actual_fps/2:.1f} Hz)",
        fontsize=14, fontweight="bold"
    )

    t = np.arange(n) / actual_fps

    # Time domain — all channels
    ax_time.plot(t, r_sig, color="red", linewidth=0.4, alpha=0.5, label="Red")
    ax_time.plot(t, g_sig, color="green", linewidth=0.4, alpha=0.5, label="Green")
    ax_time.plot(t, b_sig, color="blue", linewidth=0.4, alpha=0.5, label="Blue")
    ax_time.plot(t, rgb_sig, color="black", linewidth=0.7, alpha=0.9, label="Grayscale")
    ax_time.set_title("Temporal Signal: Mean RGB Intensity Per Frame")
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Mean Pixel Intensity")
    ax_time.legend(fontsize=8)
    ax_time.grid(True, alpha=0.3)

    # FFT — grayscale + per-channel overlays
    mean_val = np.mean(rgb_sig)
    centered = np.array(rgb_sig) - mean_val
    window = np.hanning(n)
    windowed = centered * window

    fft_gray = np.abs(np.fft.fft(windowed))[:n // 2]
    freqs = np.fft.fftfreq(n, d=1.0 / actual_fps)[:n // 2]
    si = max(1, int(0.5 / (actual_fps / n))) if n > 1 else 1

    # Per-channel FFT
    r_fft = np.abs(np.fft.fft((np.array(r_sig) - np.mean(r_sig)) * window))[:n // 2]
    g_fft = np.abs(np.fft.fft((np.array(g_sig) - np.mean(g_sig)) * window))[:n // 2]
    b_fft = np.abs(np.fft.fft((np.array(b_sig) - np.mean(b_sig)) * window))[:n // 2]

    ax_fft.plot(freqs[si:], r_fft[si:], color="red", linewidth=0.5, alpha=0.4, label="R")
    ax_fft.plot(freqs[si:], g_fft[si:], color="green", linewidth=0.5, alpha=0.4, label="G")
    ax_fft.plot(freqs[si:], b_fft[si:], color="blue", linewidth=0.5, alpha=0.4, label="B")
    ax_fft.plot(freqs[si:], fft_gray[si:], color="black", linewidth=1.0, alpha=0.8, label="Gray")
    ax_fft.fill_between(freqs[si:], fft_gray[si:], alpha=0.08, color="black")

    dom_freq = 0
    source_text = ""
    if len(fft_gray) > si:
        di = si + np.argmax(fft_gray[si:])
        dom_freq = freqs[di]
        dom_mag = fft_gray[di]
        source_text = identify_flicker_source(dom_freq)
        ax_fft.axvline(dom_freq, color="red", linestyle="--", alpha=0.8,
                        label=f"Dominant: {dom_freq:.2f} Hz")
        ax_fft.annotate(f"  {dom_freq:.2f} Hz\n  {source_text}",
                         xy=(dom_freq, dom_mag * 0.9),
                         fontsize=9, color="red", fontweight="bold")
    ax_fft.legend(fontsize=8)
    ax_fft.set_title("Frequency Spectrum (Hanning Windowed FFT)")
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Magnitude")
    ax_fft.grid(True, alpha=0.3)

    # Explanation panel
    ax_info.axis("off")
    explanation = (
        f"METHOD: Extracted mean pixel intensity (grayscale + R/G/B channels) from each frame to construct\n"
        f"a temporal signal across {n} frames ({n/actual_fps:.1f}s). Applied Hanning window and computed FFT.\n"
        f"RESULT: Dominant frequency = {dom_freq:.2f} Hz | Likely source: {source_text}\n"
        f"NOTE: Camera samples at {actual_fps:.0f} FPS → Nyquist limit = {actual_fps/2:.1f} Hz. "
        f"Mains flicker at 50/60Hz appears aliased.\n"
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
print("[RGB-FFT] Starting RGB frequency analysis...")
print("[RGB-FFT] Keep the scene STATIC — we are measuring ambient light noise.")
print(f"[RGB-FFT] Auto-recording {RECORD_SECONDS}s video.\n")
print("Controls:  q = quit  |  s = save snapshot + hi-res plot\n")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        if start_time is None:
            start_time = frames.get_timestamp() / 1000.0
        current_time = frames.get_timestamp() / 1000.0 - start_time
        timestamps_sec.append(current_time)

        # RGB image
        color_image = np.asanyarray(color_frame.get_data())

        # Per-channel and grayscale intensities
        b_mean = float(np.mean(color_image[:, :, 0]))
        g_mean = float(np.mean(color_image[:, :, 1]))
        r_mean = float(np.mean(color_image[:, :, 2]))
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray_mean = float(np.mean(gray))

        rgb_intensities.append(gray_mean)
        r_intensities.append(r_mean)
        g_intensities.append(g_mean)
        b_intensities.append(b_mean)

        # Rolling buffer
        if len(rgb_intensities) > BUFFER_SIZE:
            rgb_intensities.pop(0)
            r_intensities.pop(0)
            g_intensities.pop(0)
            b_intensities.pop(0)
            timestamps_sec.pop(0)

        frame_idx += 1
        n = len(rgb_intensities)

        # ── RGB feed display ──
        cam_display = cv2.resize(color_image, (PLOT_W, CAM_H))
        cv2.putText(cam_display, "RGB Stream", (10, 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(cam_display, f"Gray: {gray_mean:.1f}  R:{r_mean:.0f} G:{g_mean:.0f} B:{b_mean:.0f}",
                     (10, CAM_H - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # ── Update FFT plot every 10 frames ──
        if n >= 10 and frame_idx % 10 == 0:
            actual_fps = (n - 1) / max(timestamps_sec[-1] - timestamps_sec[0], 0.001)
            try:
                cached_plot = render_noise_plot(
                    list(rgb_intensities), list(r_intensities),
                    list(g_intensities), list(b_intensities),
                    actual_fps, n
                )
            except Exception as e:
                print(f"[WARN] Plot error: {e}")
        elif n < 10:
            placeholder = np.zeros((PLOT_H, PLOT_W, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"Collecting RGB data... {n}/{BUFFER_SIZE}",
                         (PLOT_W // 4, PLOT_H // 2),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
            cached_plot = placeholder

        # Stack: RGB feed on top, FFT plot on bottom
        combined = np.vstack((cam_display, cached_plot))
        h_c, w_c = combined.shape[:2]

        # ── Video recording ──
        if recording:
            if record_start_time is None:
                record_start_time = time.time()
                ts = timestamp()
                vid_path = os.path.join(VIDEO_DIR, f"rgb_noise_{ts}.mp4")
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
            fpath = os.path.join(FRAMES_DIR, f"rgb_noise_{frame_idx:05d}.png")
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
                hires_path = os.path.join(PLOTS_DIR, f"rgb_noise_plot_{ts}.png")
                save_hires_plot(list(rgb_intensities), list(r_intensities),
                                list(g_intensities), list(b_intensities),
                                actual_fps, n, hires_path)
                print(f"[SNAP] Hi-res plot → {hires_path}")

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    if video_writer:
        video_writer.release()
    pipeline.stop()
    cv2.destroyAllWindows()

    # Save final hi-res plot
    n = len(rgb_intensities)
    if n >= 10:
        actual_fps = (n - 1) / max(timestamps_sec[-1] - timestamps_sec[0], 0.001)
        final_path = os.path.join(PLOTS_DIR, f"rgb_noise_final_{timestamp()}.png")
        save_hires_plot(list(rgb_intensities), list(r_intensities),
                        list(g_intensities), list(b_intensities),
                        actual_fps, n, final_path)
        print(f"\n[RGB-FFT] Final plot → {final_path}")

        # Print detailed summary
        centered = np.array(rgb_intensities) - np.mean(rgb_intensities)
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
            print(f"  RGB TEMPORAL FREQUENCY ANALYSIS SUMMARY")
            print(f"{'='*60}")
            print(f"  Frames analyzed : {n}")
            print(f"  Effective FPS   : {actual_fps:.2f}")
            print(f"  Nyquist limit   : {actual_fps/2:.2f} Hz")
            print(f"  Freq resolution : {actual_fps/n:.3f} Hz")
            print(f"  Dominant freq   : {dom_freq:.2f} Hz")
            print(f"  Magnitude       : {fft_result[di]:.1f}")
            print(f"  Likely source   : {source}")
            print(f"{'='*60}")
            print(f"  NOTE: 50Hz mains aliased at 30fps → ~8.33Hz")
            print(f"        60Hz mains aliased at 30fps → ~10.0Hz")
            print(f"{'='*60}\n")

    print(f"[RGB-FFT] Saved {saved_frame_count} frames to {FRAMES_DIR}")
    print("[RGB-FFT] Done.")
