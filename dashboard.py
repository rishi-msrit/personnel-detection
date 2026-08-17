"""
RealSense Surveillance — Browser Dashboard
Run:  python dashboard.py
Open: http://localhost:5000
Stop: click "Stop System" in browser, or Ctrl+C in terminal
"""
import threading, time, json, collections, traceback
from datetime import datetime
import cv2, numpy as np, pyrealsense2 as rs
from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO
import psutil

try:
    import GPUtil; _HAS_GPU = True
except ImportError:
    _HAS_GPU = False

from profiler import FrameProfiler
from csv_logger import CSVLogger

# ── Config ────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT, FPS   = 640, 480, 30
YOLO_INTERVAL        = 3
VELOCITY_ALPHA       = 0.35   # more responsive to new movement
HYSTERESIS_FRAMES    = 2     # commit direction after 2 agreeing frames
MAX_LOG_ROWS         = 500
LOG_THROTTLE         = 3      # frames between per-track log entries (was 10)
CPU_TDP_W            = 45.0   # adjust for your CPU
GPU_TDP_W            = 60.0   # RTX 3050 Laptop ~60W TDP

# ── Shared state ─────────────────────────────────────────────────────────────
app = Flask(__name__)
_lock    = threading.Lock()
_stop_ev = threading.Event()   # signals pipeline to shut down
 
_state = {
    "frame_jpeg": None, "mode": "rgb",
    "motion": False, "fps": 0.0, "frame_time_ms": 0.0,
    "obj_count": 0,
    # Resources
    "cpu_pct": 0.0, "gpu_pct": 0.0, "ram_mb": 0.0, "gpu_mb": 0.0,
    # Power
    "cpu_power_w": 0.0, "gpu_power_w": 0.0, "inst_power_w": 0.0,
    "frame_power_mj": 0.0, "pixel_power_uj": 0.0,
    "total_energy_wh": 0.0, "peak_power_w": 0.0,
    "energy_per_frame_mj": 0.0, "energy_per_object_mj": 0.0,
    # Efficiency
    "fps_per_watt": 0.0, "objects_per_joule": 0.0, "yolo_efficiency": 0.0,
    # Latency
    "stages": {}, "bottleneck": "—",
    # FPS stability
    "fps_stability_std": 0.0, "dropped_frames": 0,
    # Tracking
    "tracking_stability_pct": 100.0,
    # Log
    "new_events": [], "log": collections.deque(maxlen=MAX_LOG_ROWS),
    # Status
    "running": True,
}

# ── Direction tracker ─────────────────────────────────────────────────────────
_ARROW_DIRS = {
    "Right":(1,0),"Left":(-1,0),"Up":(0,-1),"Down":(0,1),
    "Up-Right":(1,-1),"Up-Left":(-1,-1),"Down-Right":(1,1),"Down-Left":(-1,1),
}

def _vec_to_dir(vx, vy, min_speed=0.3):
    if abs(vx) < min_speed and abs(vy) < min_speed:
        return "Stationary"
    angle = np.degrees(np.arctan2(-vy, vx))
    for lo, hi, label in [
        (-22.5,22.5,"Right"),(22.5,67.5,"Up-Right"),(67.5,112.5,"Up"),
        (112.5,157.5,"Up-Left"),(-157.5,-112.5,"Down-Left"),
        (-112.5,-67.5,"Down"),(-67.5,-22.5,"Down-Right"),
    ]:
        if lo <= angle < hi: return label
    return "Left"


class SmoothTracker:
    def __init__(self):
        self.tracks  = {}
        self.next_id = 0
        self._log_cd = {}
        self._new_tracks_this_frame = 0
        self._total_tracks_this_frame = 0

    @property
    def tracking_stability_pct(self):
        """% of tracks that were matched (not newly created)."""
        if self._total_tracks_this_frame == 0: return 100.0
        matched = self._total_tracks_this_frame - self._new_tracks_this_frame
        return round(matched / self._total_tracks_this_frame * 100, 1)

    def update(self, detections, max_dist=100):
        self._new_tracks_this_frame = 0
        self._total_tracks_this_frame = len(detections)
        unmatched = set(self.tracks)
        for det in detections:
            cx, cy = det["cx"], det["cy"]
            best_id, best_d = None, max_dist
            for tid, t in self.tracks.items():
                d = np.hypot(cx-t["cx"], cy-t["cy"])
                if d < best_d: best_d, best_id = d, tid
            if best_id is not None:
                t = self.tracks[best_id]
                t["vx"] = VELOCITY_ALPHA*(cx-t["cx"]) + (1-VELOCITY_ALPHA)*t["vx"]
                t["vy"] = VELOCITY_ALPHA*(cy-t["cy"]) + (1-VELOCITY_ALPHA)*t["vy"]
                t["cx"], t["cy"] = cx, cy
                t["depth_m"] = det.get("depth_m", t.get("depth_m", 0))
                cand = _vec_to_dir(t["vx"], t["vy"])
                if cand == t["cand"]:
                    t["hold"] += 1
                    if t["hold"] >= HYSTERESIS_FRAMES: t["dir"] = cand
                else: t["cand"], t["hold"] = cand, 1
                det["track_id"] = best_id
                det["direction"] = t["dir"]
                det["speed_px_s"] = round(np.hypot(t["vx"], t["vy"]) * FPS, 1)
                unmatched.discard(best_id)
                self._log_cd[best_id] = self._log_cd.get(best_id, 0) + 1
            else:
                self.next_id += 1
                self._new_tracks_this_frame += 1
                self.tracks[self.next_id] = {
                    "cx":cx,"cy":cy,"vx":0.0,"vy":0.0,
                    "dir":"New","cand":"New","hold":0,"depth_m":det.get("depth_m",0),
                }
                det["track_id"] = self.next_id
                det["direction"] = "New"
                det["speed_px_s"] = 0.0
                self._log_cd[self.next_id] = 0
        for tid in unmatched:
            self.tracks.pop(tid, None); self._log_cd.pop(tid, None)
        return detections

    def should_log(self, tid):
        c = self._log_cd.get(tid, 0)
        if c >= LOG_THROTTLE:
            self._log_cd[tid] = 0; return True
        return False

    @staticmethod
    def draw_arrow(frame, det):
        d = det.get("direction","")
        if d in _ARROW_DIRS:
            dx, dy = _ARROW_DIRS[d]
            cx, cy = det["cx"], det["cy"]
            cv2.arrowedLine(frame,(cx,cy),(cx+dx*35,cy+dy*35),(0,200,255),2,tipLength=0.4)


# ── Pipeline thread ───────────────────────────────────────────────────────────
def pipeline_thread():
    csv_log  = CSVLogger()
    profiler = FrameProfiler()
    model    = YOLO("yolov8n.pt")
    tracker  = SmoothTracker()
    _clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 0)

    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    align = rs.align(rs.stream.color)
    pipeline.start(config)
    time.sleep(1)

    def capture_depth_bg():
        acc = np.zeros((HEIGHT,WIDTH),np.float64); cnt = np.zeros((HEIGHT,WIDTH),np.float64)
        for _ in range(15):
            fs=pipeline.wait_for_frames(); af=align.process(fs); df=af.get_depth_frame()
            if df:
                d=np.asanyarray(df.get_data()).astype(np.float64); v=d>0; acc[v]+=d[v]
                c=np.zeros_like(d); c[v]=1; cnt+=c
        cnt[cnt==0]=1; return (acc/cnt).astype(np.float32)

    depth_bg=capture_depth_bg(); rgb_bg=None; prev_depth=None; ema_luma=None
    roi=np.zeros((HEIGHT,WIDTH),np.uint8); roi[24:HEIGHT-24,32:WIDTH-32]=255

    # Motion
    motion_ctr=clear_ctr=0; motion_confirmed=False; DEBOUNCE=4
    MIN_AREA=int(WIDTH*HEIGHT*0.005)
    MAX_A_RGB=int(WIDTH*HEIGHT*0.55); MAX_A_DEP=int(WIDTH*HEIGHT*0.50)

    # YOLO / tracking state
    frame_count=0; last_dets=[]; dropped=0
    fps_ctr=0; fps_val=0.0; fps_timer=time.time()
    fps_history=collections.deque(maxlen=30)   # for std

    # System stats (polled every 30 frames)
    sys_stats={"cpu_pct":0.0,"ram_mb":0.0,"gpu_mb":0.0,"gpu_pct":0.0}
    sys_poll=0

    # Power accumulators
    total_energy_wh=0.0; peak_power_w=0.0

    # Confidence / depth rolling averages
    conf_history=collections.deque(maxlen=30)
    depth_history=collections.deque(maxlen=30)

    try:
        while not _stop_ev.is_set():
            mode=_state["mode"]
            profiler.frame_start()

            # ── Capture ──────────────────────────────────────────────────────
            profiler.stage_start("capture")
            try:
                fs=pipeline.wait_for_frames(timeout_ms=500)
            except RuntimeError:
                dropped+=1; profiler.stage_end("capture"); profiler.frame_end(); continue
            al=align.process(fs)
            cf=al.get_color_frame(); df=al.get_depth_frame()
            if not cf:
                dropped+=1; profiler.stage_end("capture"); profiler.frame_end(); continue
            color_raw=np.asanyarray(cf.get_data())
            profiler.stage_end("capture")

            # ── Preprocess ───────────────────────────────────────────────────
            profiler.stage_start("preprocess")
            if mode=="rgb":
                gr=cv2.cvtColor(color_raw,cv2.COLOR_BGR2GRAY); luma=float(np.mean(gr))
                if ema_luma is None: ema_luma=luma; frame=color_raw.copy()
                else:
                    ema_luma=0.97*ema_luma+0.03*luma
                    gain=float(np.clip(ema_luma/max(luma,1.0),0.85,1.15))
                    frame=cv2.convertScaleAbs(color_raw,alpha=gain,beta=0)
                gb=cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(21,21),0)
                if rgb_bg is None: rgb_bg=gb.astype(float)
                cv2.accumulateWeighted(gb,rgb_bg,0.005)
                diff=cv2.absdiff(gb,cv2.convertScaleAbs(rgb_bg))
                _,thresh=cv2.threshold(diff,25,255,cv2.THRESH_BINARY)
                k=np.ones((5,5),np.uint8)
                thresh=cv2.erode(thresh,k,iterations=2)
                thresh=cv2.dilate(thresh,k,iterations=1)
                thresh=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,k)
                display=frame.copy(); max_area=MAX_A_RGB
            else:
                if df:
                    dr=np.asanyarray(df.get_data()).astype(np.float32)
                    if prev_depth is None: prev_depth=dr.copy()
                    de=0.72*dr+0.28*prev_depth; prev_depth=de.copy()
                    dd=cv2.GaussianBlur(cv2.absdiff(de,depth_bg),(9,9),0)
                    bv=(de>0)&(depth_bg>0); vd=dd[bv]
                    at=max(30.0,float(np.percentile(vd,90)) if len(vd)>500 else 80.0)
                    mm=np.zeros_like(dd,np.uint8); mm[(dd>at)&bv]=255
                    mm=cv2.bitwise_and(mm,roi)
                    k7=np.ones((7,7),np.uint8)
                    mm=cv2.morphologyEx(mm,cv2.MORPH_OPEN,k7)
                    thresh=cv2.morphologyEx(mm,cv2.MORPH_CLOSE,k7)
                    display=np.asanyarray(colorizer.colorize(df).get_data())
                else:
                    thresh=np.zeros((HEIGHT,WIDTH),np.uint8)
                    display=np.zeros((HEIGHT,WIDTH,3),np.uint8)
                max_area=MAX_A_DEP
            profiler.stage_end("preprocess")

            # ── Motion state machine ──────────────────────────────────────────
            cnts,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            raw_motion=any(MIN_AREA<cv2.contourArea(c)<max_area for c in cnts)
            if raw_motion:
                motion_ctr+=1; clear_ctr=0
                if motion_ctr>=DEBOUNCE: motion_confirmed=True
            else:
                clear_ctr+=1; motion_ctr=0
                if clear_ctr>=DEBOUNCE: motion_confirmed=False

            # ── YOLO ─────────────────────────────────────────────────────────
            profiler.stage_start("yolo")
            frame_count+=1
            if frame_count%YOLO_INTERVAL==0:
                g=_clahe.apply(cv2.cvtColor(color_raw,cv2.COLOR_BGR2GRAY))
                yi=cv2.addWeighted(color_raw,0.6,cv2.cvtColor(g,cv2.COLOR_GRAY2BGR),0.4,0)
                results=model(yi,verbose=False,conf=0.30,iou=0.45)
                raw_dets=[]
                for r in results:
                    for box in r.boxes:
                        x1,y1,x2,y2=box.xyxy[0].cpu().numpy().astype(int)
                        cx,cy=(x1+x2)//2,(y1+y2)//2
                        dm=df.get_distance(min(cx,WIDTH-1),min(cy,HEIGHT-1)) if df and mode=="depth" else 0.0
                        c_val=float(box.conf[0])
                        conf_history.append(c_val)
                        if dm>0: depth_history.append(dm)
                        raw_dets.append({
                            "x1":x1,"y1":y1,"x2":x2,"y2":y2,
                            "conf":c_val,"label":model.names[int(box.cls[0])],
                            "cx":cx,"cy":cy,"depth_m":dm
                        })
                last_dets=tracker.update(raw_dets)
            profiler.stage_end("yolo")

            # ── Track + render ────────────────────────────────────────────────
            profiler.stage_start("track")
            new_log_rows=[]
            for det in last_dets:
                x1,y1,x2,y2=det["x1"],det["y1"],det["x2"],det["y2"]
                cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,0),2)
                direction=det.get("direction",""); dm=det.get("depth_m",0)
                txt=f"{det['label']} ({det['conf']:.2f})"
                if mode=="depth" and dm>0: txt+=f" {dm:.2f}m"
                if direction and direction not in ("New","Stationary"): txt+=f" | {direction}"
                (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.48,1)
                cv2.rectangle(display,(x1,y1-th-8),(x1+tw+4,y1),(0,255,0),-1)
                cv2.putText(display,txt,(x1+2,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,0,0),1)
                SmoothTracker.draw_arrow(display,det)
                tid=det.get("track_id",-1)
                if tracker.should_log(tid):
                    new_log_rows.append({
                        "ts":datetime.now().strftime("%H:%M:%S"),
                        "object":det["label"],
                        "direction":direction if direction not in ("New","") else "Stationary",
                        "conf":round(det["conf"],2),"mode":mode,
                    })
            profiler.stage_end("track")

            # ── Render overlays ───────────────────────────────────────────────
            profiler.stage_start("render")
            lbl="MOTION" if motion_confirmed else "CLEAR"
            cv2.putText(display,lbl,(20,35),cv2.FONT_HERSHEY_SIMPLEX,0.9,
                        (0,0,255) if motion_confirmed else (0,200,0),2)
            cv2.putText(display,f"{mode.upper()} {fps_val:.1f}fps",
                        (WIDTH-180,25),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
            fps_ctr+=1
            elapsed=time.time()-fps_timer
            if elapsed>=1.0:
                fps_val=fps_ctr/elapsed; fps_history.append(fps_val)
                fps_ctr=0; fps_timer=time.time()
            _,jpeg=cv2.imencode(".jpg",display,[cv2.IMWRITE_JPEG_QUALITY,80])
            profiler.stage_end("render")
            profiler.frame_end()

            # ── System stats (every 30 frames) ────────────────────────────────
            sys_poll+=1
            if sys_poll>=30:
                sys_poll=0
                proc=psutil.Process()
                cpu=psutil.cpu_percent(interval=None); ram=proc.memory_info().rss/1024**2
                gm=gp=0.0
                if _HAS_GPU:
                    try:
                        gpus=GPUtil.getGPUs()
                        if gpus: gm=gpus[0].memoryUsed; gp=gpus[0].load*100
                    except: pass
                sys_stats.update(cpu_pct=cpu,ram_mb=ram,gpu_mb=gm,gpu_pct=gp)

            # ── Power calculations ────────────────────────────────────────────
            summary=profiler.get_summary()
            ft=summary["frame_time_ms"]; stages=summary["stages"]
            cpu_p=sys_stats["cpu_pct"]/100.0*CPU_TDP_W
            gpu_p=sys_stats["gpu_pct"]/100.0*GPU_TDP_W
            inst_p=round(cpu_p+gpu_p,2)
            frame_p_mj=round(inst_p*(ft/1000)*1000,3)          # millijoules
            pixel_p_uj=round(inst_p*(ft/1000)*1e6/(WIDTH*HEIGHT),4)  # µJ/pixel
            n_obj=max(len(last_dets),1)
            total_energy_wh+=inst_p*(ft/3600000)
            peak_power_w=max(peak_power_w,inst_p)
            fps_per_watt=round(fps_val/max(inst_p,0.01),2)
            obj_per_j=round(len(last_dets)/max(frame_p_mj/1000,0.001),1)
            yolo_ms=stages.get("yolo",0)
            yolo_eff=round(len(last_dets)/max(yolo_ms,0.1),3)

            # Bottleneck
            bottleneck=max(stages,key=stages.get) if stages else "—"

            # FPS stability
            fps_arr=np.array(list(fps_history)) if fps_history else np.array([fps_val])
            fps_std=round(float(np.std(fps_arr)),2)

            # Averages
            avg_conf=round(float(np.mean(list(conf_history))),3) if conf_history else 0.0
            avg_dep=round(float(np.mean(list(depth_history))),2) if depth_history else 0.0

            # Build stats dict for CSV
            csv_stats={
                "frame_num":frame_count,"mode":mode,"obj_count":len(last_dets),
                "obj_classes":"|".join(set(d["label"] for d in last_dets)),
                "avg_confidence":avg_conf,"avg_depth_m":avg_dep,
                "cpu_power_w":round(cpu_p,2),"gpu_power_w":round(gpu_p,2),
                "inst_power_w":inst_p,"frame_power_mj":frame_p_mj,
                "pixel_power_uj":pixel_p_uj,"total_energy_wh":round(total_energy_wh,6),
                "peak_power_w":round(peak_power_w,2),
                "energy_per_frame_mj":frame_p_mj,
                "energy_per_object_mj":round(frame_p_mj/n_obj,3),
                "fps_per_watt":fps_per_watt,"objects_per_joule":obj_per_j,
                "yolo_efficiency":yolo_eff,"frame_time_ms":ft,"stages":stages,
                "fps":round(fps_val,1),"fps_stability_std":fps_std,
                "dropped_frames":dropped,"cpu_pct":sys_stats["cpu_pct"],
                "gpu_pct":sys_stats["gpu_pct"],"ram_mb":round(sys_stats["ram_mb"],1),
                "gpu_mb":round(sys_stats["gpu_mb"],1),"motion":motion_confirmed,
                "tracking_stability_pct":tracker.tracking_stability_pct,
            }
            csv_log.maybe_write(csv_stats)

            with _lock:
                _state["frame_jpeg"]=jpeg.tobytes(); _state["motion"]=motion_confirmed
                _state["fps"]=round(fps_val,1); _state["frame_time_ms"]=ft
                _state["obj_count"]=len(last_dets)
                _state["cpu_pct"]=sys_stats["cpu_pct"]; _state["gpu_pct"]=sys_stats["gpu_pct"]
                _state["ram_mb"]=round(sys_stats["ram_mb"],1); _state["gpu_mb"]=round(sys_stats["gpu_mb"],1)
                _state["cpu_power_w"]=round(cpu_p,2); _state["gpu_power_w"]=round(gpu_p,2)
                _state["inst_power_w"]=inst_p; _state["frame_power_mj"]=frame_p_mj
                _state["pixel_power_uj"]=pixel_p_uj
                _state["total_energy_wh"]=round(total_energy_wh,6)
                _state["peak_power_w"]=round(peak_power_w,2)
                _state["energy_per_frame_mj"]=frame_p_mj
                _state["energy_per_object_mj"]=round(frame_p_mj/n_obj,3)
                _state["fps_per_watt"]=fps_per_watt; _state["objects_per_joule"]=obj_per_j
                _state["yolo_efficiency"]=yolo_eff; _state["stages"]=stages
                _state["bottleneck"]=bottleneck; _state["fps_stability_std"]=fps_std
                _state["dropped_frames"]=dropped
                _state["tracking_stability_pct"]=tracker.tracking_stability_pct
                _state["avg_confidence"]=avg_conf; _state["avg_depth_m"]=avg_dep
                _state["new_events"]=new_log_rows
                _state["log"].extendleft(reversed(new_log_rows))

    except Exception as exc:
        print(f"\n[PIPELINE ERROR] {exc}"); traceback.print_exc()
    finally:
        csv_log.stop(); pipeline.stop()
        with _lock: _state["running"]=False
        print("[PIPELINE] Stopped cleanly.")


# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

def _gen_frames():
    while not _stop_ev.is_set():
        with _lock: frame=_state["frame_jpeg"]
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+frame+b"\r\n"
        time.sleep(0.033)

@app.route("/video")
def video_feed():
    return Response(_gen_frames(),mimetype="multipart/x-mixed-replace; boundary=frame")

_ANALYTICS_KEYS = [
    "motion","fps","frame_time_ms","obj_count","mode",
    "cpu_pct","gpu_pct","ram_mb","gpu_mb",
    "cpu_power_w","gpu_power_w","inst_power_w","frame_power_mj","pixel_power_uj",
    "total_energy_wh","peak_power_w","energy_per_frame_mj","energy_per_object_mj",
    "fps_per_watt","objects_per_joule","yolo_efficiency",
    "stages","bottleneck","fps_stability_std","dropped_frames",
    "tracking_stability_pct","avg_confidence","avg_depth_m",
    "new_events","running",
]

def _gen_stats():
    while True:
        with _lock:
            payload={k:_state[k] for k in _ANALYTICS_KEYS}
            _state["new_events"]=[]
        yield f"data: {json.dumps(payload)}\n\n"
        time.sleep(0.25)

@app.route("/stats")
def stats_stream():
    return Response(_gen_stats(),mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/set_mode/<mode>")
def set_mode(mode):
    if mode in ("rgb","depth"):
        with _lock: _state["mode"]=mode
    return "OK"

@app.route("/shutdown")
def shutdown():
    _stop_ev.set()
    return "Stopping…"

@app.route("/log")
def get_log():
    with _lock: rows=list(_state["log"])
    return jsonify(rows)

if __name__=="__main__":
    t=threading.Thread(target=pipeline_thread,daemon=True)
    t.start()
    print("\n[DASHBOARD] Open http://localhost:5000\n")
    app.run(host="0.0.0.0",port=5000,threaded=True,debug=False)
