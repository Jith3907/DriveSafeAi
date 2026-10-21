"""
Monitor worker.

Streamlit reruns hold the GIL long enough to starve an in-process camera loop
(measured: 1.7 FPS at 10 UI refreshes/s). The detector pipeline therefore runs in
a separate process; the page only polls the newest snapshot. The detection loop
keeps its own frame rate (20+ FPS on this machine) while the UI refreshes at
whatever rate the browser can render.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from queue import Empty

JPEG_QUALITY = 78
FPS_WINDOW = 20
MAX_READ_FAILURES = 30
# No page polled for this long -> the browser is gone; release the webcam.
POLL_TIMEOUT_SEC = 120.0
MODEL_LOAD_TIMEOUT_SEC = 15.0


def _monitor_process(camera_index: int, queue, stop_event, poll_time: float) -> None:
    """Child process: owns the camera, runs the detectors, publishes snapshots."""
    import cv2

    from src.ui.camera import build_runtime, open_camera
    from src.ui.pipeline import process_frame

    cap, error = open_camera(camera_index)
    if cap is None:
        queue.put({"error": error or f"Camera {camera_index} could not be opened."})
        return

    try:
        runtime = build_runtime()
    except Exception as exc:
        cap.release()
        queue.put({"error": f"Could not load the detection models: {exc}"})
        return

    times: list[float] = []
    proc_times: list[float] = []
    seq = 0
    failures = 0

    def publish(snapshot: dict) -> None:
        # Keep only the freshest snapshot in the queue.
        try:
            queue.put_nowait(snapshot)
        except Exception:
            try:
                queue.get_nowait()
            except Empty:
                pass
            try:
                queue.put_nowait(snapshot)
            except Exception:
                pass

    try:
        while not stop_event.is_set():
            now = time.time()
            try:
                if now - float(poll_time.value) > POLL_TIMEOUT_SEC:
                    break
            except Exception:
                pass

            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= MAX_READ_FAILURES:
                    publish({"error": "Lost the camera feed. Click Retry to reconnect."})
                    break
                time.sleep(0.02)
                continue
            failures = 0
            frame = cv2.flip(frame, 1)

            started = time.perf_counter()
            try:
                result = process_frame(frame, runtime)
            except Exception as exc:
                publish({"error": f"Detector error: {exc}. Click Retry to restart the camera."})
                break
            process_ms = (time.perf_counter() - started) * 1000.0

            annotated = result.pop("frame", None)
            jpeg = None
            if annotated is not None:
                encoded, buffer = cv2.imencode(
                    ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                )
                if encoded:
                    jpeg = buffer.tobytes()

            now = time.time()
            if times and (now - times[-1]) > 2.0:
                times.clear()
                proc_times.clear()
            times.append(now)
            proc_times.append(process_ms)
            del times[:-FPS_WINDOW]
            del proc_times[:-FPS_WINDOW]

            fps = 0.0
            if len(times) >= 2:
                span = times[-1] - times[0]
                if span > 0:
                    fps = (len(times) - 1) / span

            seq += 1
            result.update(
                {
                    "jpeg": jpeg,
                    "fps": fps,
                    "process_ms": sum(proc_times) / len(proc_times),
                    "seq": seq,
                    "captured_at": now,
                    "camera_index": camera_index,
                }
            )
            publish(result)
    finally:
        try:
            cap.release()
        except Exception:
            pass


class MonitorWorker:
    """Parent-side handle for the detector process."""

    def __init__(self, camera_index: int):
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue(maxsize=1)
        self._stop_event = ctx.Event()
        self._poll_time = ctx.Value("d", time.time(), lock=False)
        self._process = ctx.Process(
            target=_monitor_process,
            args=(camera_index, self._queue, self._stop_event, self._poll_time),
            name="drivesafe-monitor",
            daemon=True,
        )
        self._snapshot: dict | None = None
        self._error: str | None = None

    def start(self) -> "MonitorWorker":
        self._process.start()
        return self

    def poll(self) -> dict | None:
        """Drains the snapshot queue and returns the newest complete snapshot."""
        while True:
            try:
                item = self._queue.get_nowait()
            except Empty:
                break
            except (EOFError, OSError):
                break
            if isinstance(item, dict) and item.get("error"):
                self._error = str(item["error"])
            elif isinstance(item, dict):
                self._snapshot = item
        try:
            self._poll_time.value = time.time()
        except Exception:
            pass
        return self._snapshot

    def snapshot(self) -> dict | None:
        return self.poll()

    @property
    def error(self) -> str | None:
        return self._error

    def alive(self) -> bool:
        try:
            return self._process.is_alive()
        except Exception:
            return False

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        try:
            if self._process.is_alive():
                self._process.join(timeout=timeout)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
        except Exception:
            pass
        try:
            while True:
                self._queue.get_nowait()
        except Exception:
            pass
        try:
            self._queue.close()
        except Exception:
            pass
