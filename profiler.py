import time
import collections


class FrameProfiler:
    STAGES = ["capture", "preprocess", "yolo", "track", "render"]
    HISTORY = 60

    def __init__(self):
        self._t0 = {}
        self._history = {s: collections.deque(maxlen=self.HISTORY) for s in self.STAGES}
        self._frame_history = collections.deque(maxlen=self.HISTORY)
        self._frame_t0 = None

    def frame_start(self):
        self._frame_t0 = time.perf_counter()

    def stage_start(self, stage: str):
        self._t0[stage] = time.perf_counter()

    def stage_end(self, stage: str):
        if stage in self._t0:
            dt = (time.perf_counter() - self._t0[stage]) * 1000.0
            self._history[stage].append(dt)

    def frame_end(self):
        if self._frame_t0 is not None:
            dt = (time.perf_counter() - self._frame_t0) * 1000.0
            self._frame_history.append(dt)

    def get_summary(self) -> dict:
        def mean(dq):
            return round(sum(dq) / len(dq), 2) if dq else 0.0

        stage_means = {s: mean(self._history[s]) for s in self.STAGES}
        frame_time = mean(self._frame_history)
        fps = round(1000.0 / frame_time, 1) if frame_time > 0 else 0.0
        return {
            "stages": stage_means,
            "frame_time_ms": frame_time,
            "fps": fps,
        }


if __name__ == "__main__":
    profiler = FrameProfiler()

    print(f"Running self-test over {FrameProfiler.HISTORY} simulated frames...\n")

    for _ in range(FrameProfiler.HISTORY):
        profiler.frame_start()
        for stage in FrameProfiler.STAGES:
            profiler.stage_start(stage)
            time.sleep(0.002)
            profiler.stage_end(stage)
        profiler.frame_end()

    summary = profiler.get_summary()
    print(f"{'Stage':<12} {'Avg (ms)':>10}")
    print("-" * 24)
    for stage, ms in summary["stages"].items():
        print(f"{stage:<12} {ms:>10.2f}")
    print("-" * 24)
    print(f"{'frame_time':<12} {summary['frame_time_ms']:>10.2f} ms")
    print(f"{'fps':<12} {summary['fps']:>10.1f}")
