"""
Flask display layer for DriveSafe AI.

Only the display layer is replaced. Detection is the existing code:

    src/ui/pipeline.py   process_frame(frame, runtime) -> status dict + annotated frame
    src/drowsiness.py    EAR / MAR, drowsy, yawn, micro-sleep
    src/head_pose.py     solvePnP yaw / pitch / roll, distraction timers
    src/phone_detection.py  YOLO phone detection with the voting window
    src/driver_auth.py   driver identity from face-mesh similarity

`src/app.py` and the rest of `src/ui/` are untouched and still run under
Streamlit. The only thing this module cannot reuse is `src/ui/camera.py`,
because it imports Streamlit; the camera open / runtime setup below mirrors it.

Run with:

    python -m src.web.app        ->  http://localhost:5000

Threads:

    FrameGrabber      owns the webcam, keeps only the newest 640x480 frame
    AsyncPhone        runs the unchanged PhoneDetector every 3rd frame (416 px)
    ProcessingWorker  runs process_frame() on the newest frame and publishes
                      the newest annotated JPEG (quality 70) + status under a lock
"""

from __future__ import annotations

import csv
import os
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify

from src import config
from src.ui.pipeline import process_frame

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
HOST = "127.0.0.1"
PORT = 5000

JPEG_QUALITY = 70            # annotated frames published to /video_feed
YOLO_FRAME_INTERVAL = 3      # phone detection runs on every 3rd frame
YOLO_IMAGE_SIZE = 416        # smaller YOLO input for the web path
FPS_WINDOW = 20              # frames used for the FPS average
MAX_READ_FAILURES = 60       # consecutive camera read failures before giving up
CAMERA_START_TIMEOUT = 15.0  # how long startup waits for the first camera frame
PLACEHOLDER_INTERVAL = 0.5   # re-send the "starting" JPEG this often until frames flow

MESSAGE_HISTORY = 80         # alert lines kept in memory for the Messages box
EVENT_LOG_PATH = config.PROJECT_ROOT / "logs" / "events.csv"
EVENT_LOG_HEADER = ["timestamp", "driver", "event_type"]

# Pipeline tuning for this process only; config.py itself is not modified.
config.YOLO_FRAME_INTERVAL = 1  # AsyncPhone already feeds every 3rd frame
config.PHONE_IMGSZ = YOLO_IMAGE_SIZE
config.YOLO_IMGSZ = YOLO_IMAGE_SIZE

# Head-turn / head-up / head-down timeline, in seconds. The alert engine reads
# these back from config, so this is the only place to change them:
#   turned < HEAD_TURN_WARNING_SEC         SAFE
#   turned >= HEAD_TURN_WARNING_SEC        WARNING
#   turned >= HEAD_TURN_DANGER_SEC         DANGER
HEAD_TURN_WARNING_SEC = 3.0
HEAD_TURN_DANGER_SEC = 4.0
config.YAW_ATTENTION_TIME_SEC = HEAD_TURN_WARNING_SEC
config.YAW_DANGER_TIME_SEC = HEAD_TURN_DANGER_SEC


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def clock() -> str:
    return datetime.now().strftime("%H:%M:%S")


def fps_from(times: deque) -> float:
    if len(times) < 2:
        return 0.0
    span = times[-1] - times[0]
    return (len(times) - 1) / span if span > 0 else 0.0


def blank_status() -> dict:
    return {
        "ready": False,
        "camera_ok": False,
        "fps": 0.0,
        "stream_fps": 0.0,
        "process_ms": 0.0,
        "timestamp": clock(),
        "last_error": None,
        "error": None,  # kept for the page; mirrors last_error
    }


# ---------------------------------------------------------
# Camera thread
# ---------------------------------------------------------
def camera_candidates(preferred: int) -> list[int]:
    """The preferred index first, then the usual 0/1/2 slots, without repeats."""
    order: list[int] = []
    for index in [int(preferred), 0, 1, 2]:
        if index not in order:
            order.append(index)
    return order


def camera_backends() -> list[tuple[str, int | None]]:
    """DirectShow first on Windows, then whatever OpenCV picks by default."""
    backends: list[tuple[str, int | None]] = []
    dshow = getattr(cv2, "CAP_DSHOW", None)
    if os.name == "nt" and dshow is not None:
        backends.append(("DirectShow", dshow))
    backends.append(("default", None))
    return backends


def open_camera(preferred_index: int = 0):
    """Opens the first webcam that actually delivers a frame.

    Tries the preferred index, then 0/1/2, each with DirectShow and the default
    backend. Returns ``(cap, index, backend, attempts)``; ``cap`` is ``None``
    when every combination failed, and ``attempts`` then says why.
    """
    attempts: list[str] = []
    for index in camera_candidates(preferred_index):
        for name, backend in camera_backends():
            label = f"index {index} / {name}"
            try:
                cap = (
                    cv2.VideoCapture(index)
                    if backend is None
                    else cv2.VideoCapture(index, backend)
                )
            except Exception as exc:  # backend not available on this build
                attempts.append(f"{label}: {exc}")
                continue

            if cap is None or not cap.isOpened():
                attempts.append(f"{label}: not available")
                if cap is not None:
                    cap.release()
                continue

            # MJPG first: raw 640x480 frames saturate USB 2 bandwidth at 30 fps.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, config.FPS_TARGET)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ok, frame = cap.read()
            if not ok or frame is None:
                attempts.append(f"{label}: opened but delivered no frame")
                cap.release()
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame.shape[1]
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0]
            print(f"  [camera] using {label} at {width}x{height}")
            return cap, index, name, attempts

    return None, None, None, attempts


class FrameGrabber(threading.Thread):
    """Reads the webcam as fast as it delivers and keeps only the latest frame."""

    def __init__(self, index: int):
        super().__init__(name="drivesafe-camera", daemon=True)
        self.index = int(index)
        self.opened_index: int | None = None
        self.backend: str | None = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._stop = threading.Event()

    def snapshot(self):
        with self._lock:
            return self._seq, self._frame

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        print(f"  [camera] searching for a webcam (preferred index {self.index})...")
        cap, index, backend, attempts = open_camera(self.index)
        if cap is None:
            print("  [camera] ERROR: no camera could be opened. Tried:")
            for line in attempts:
                print(f"  [camera]   {line}")
            print(
                "  [camera] Check that the webcam is plugged in, that no other app "
                "(Teams, Zoom, the Streamlit fallback) is using it, and that "
                "Settings > Privacy > Camera allows desktop apps."
            )
            self.error = (
                "No camera could be opened (tried "
                + ", ".join(attempts[:6])
                + (", ..." if len(attempts) > 6 else "")
                + ")."
            )
            return

        self.opened_index = index
        self.backend = backend
        failures = 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= MAX_READ_FAILURES:
                        self.error = "Lost the camera feed."
                        break
                    time.sleep(0.02)
                    continue
                failures = 0
                frame = cv2.flip(frame, 1)  # pipeline expects a mirrored frame
                with self._lock:
                    self._frame = frame
                    self._seq += 1
        except Exception as exc:  # hardware / driver failure
            self.error = f"Camera error: {exc}"
            print(f"  [camera] ERROR: {self.error}")
        finally:
            try:
                cap.release()
            except Exception:
                pass


# ---------------------------------------------------------
# YOLO thread
# ---------------------------------------------------------
class AsyncPhone:
    """Runs the unchanged PhoneDetector off the video path.

    `process_frame` calls `detect(frame)` once per frame. This proxy hands every
    3rd frame to the detector thread and immediately returns the newest known
    result, so a slow YOLO pass never stalls MediaPipe or the video stream. The
    detector keeps its own confidence threshold, voting window and hold time.
    """

    def __init__(self, detector, every_n: int = YOLO_FRAME_INTERVAL):
        self.detector = detector
        self.every_n = max(1, int(every_n))
        self._count = 0
        self._pending = None
        self._pending_lock = threading.Lock()
        self._result_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._latest = {
            "detected": False,
            "detections": [],
            "phone_in_use": False,
            "true_votes": 0,
            "votes_needed": config.PHONE_VOTES_NEEDED,
            "voting_window": config.PHONE_VOTING_WINDOW,
            "vote_str": "0/0",
            "hold_remaining": 0.0,
            "hold_time": config.PHONE_HOLD_TIME_SEC,
            "last_confidence": 0.0,
            "alert_reason": "Waiting for the first YOLO pass",
        }
        self._thread = threading.Thread(target=self._run, name="drivesafe-yolo", daemon=True)

    def start(self) -> "AsyncPhone":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def detect(self, frame) -> dict:
        self._count += 1
        if self._count % self.every_n == 0:
            with self._pending_lock:
                self._pending = frame.copy()
            self._wake.set()
        with self._result_lock:
            return dict(self._latest)

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._wake.wait(0.1):
                continue
            self._wake.clear()
            with self._pending_lock:
                frame = self._pending
                self._pending = None
            if frame is None:
                continue
            try:
                result = self.detector.detect(frame)
            except Exception:
                continue
            if self._stop.is_set():
                break
            with self._result_lock:
                self._latest = result


# ---------------------------------------------------------
# Alerts (display layer only - the pipeline is not modified)
# ---------------------------------------------------------
_LEVEL_NONE = 0
_LEVEL_WARNING = 1
_LEVEL_DANGER = 2
_LEVEL_NAMES = {_LEVEL_WARNING: "WARNING", _LEVEL_DANGER: "DANGER"}

# Ties for the highest level are broken by this priority.
_ALERT_PRIORITY = {
    "no_driver": 100,
    "phone": 90,
    "drowsiness": 80,
    "looking_left": 50,
    "looking_right": 50,
    "looking_up": 45,
    "looking_down": 45,
    "yawning": 30,
    "face_not_visible": 20,
}

# Event type written to logs/events.csv for each alert key.
_ALERT_EVENT_TYPES = {
    "no_driver": "NO_DRIVER",
    "phone": "PHONE_USAGE",
    "drowsiness": "DROWSINESS",
    "looking_left": "LOOKING_LEFT",
    "looking_right": "LOOKING_RIGHT",
    "looking_up": "LOOKING_UP",
    "looking_down": "LOOKING_DOWN",
    "yawning": "YAWNING",
    "face_not_visible": "FACE_NOT_VISIBLE",
}

_LOG_LOCK = threading.Lock()
_LOG_COLUMNS: int | None = None


def _log_width() -> int:
    """Columns to write: 3, or the existing file's width if it has more.

    logs/events.csv may already exist from the Streamlit dashboard, which added
    a trailing ``duration`` column; those rows are padded so the file stays
    rectangular rather than gaining ragged lines.
    """
    global _LOG_COLUMNS
    if _LOG_COLUMNS is None:
        width = len(EVENT_LOG_HEADER)
        try:
            with open(EVENT_LOG_PATH, "r", encoding="utf-8", newline="") as handle:
                existing = next(csv.reader(handle), [])
            width = max(width, len(existing))
        except (OSError, StopIteration):
            pass
        _LOG_COLUMNS = width
    return _LOG_COLUMNS


def log_event(timestamp: str, driver: str, event_type: str) -> None:
    """Appends one alert row to logs/events.csv (created on first alert)."""
    try:
        with _LOG_LOCK:
            EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            is_new = not EVENT_LOG_PATH.exists()
            row = [timestamp, driver, event_type]
            row += [""] * max(0, _log_width() - len(row))
            with open(EVENT_LOG_PATH, "a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if is_new:
                    writer.writerow(
                        EVENT_LOG_HEADER
                        + [""] * max(0, len(row) - len(EVENT_LOG_HEADER))
                    )
                writer.writerow(row)
    except OSError as exc:
        print(f"  [log] could not append to {EVENT_LOG_PATH}: {exc}")


class AlertEngine:
    """Turns pipeline results into banner, flags, cards and alert history.

    Called once per processed frame from the processing thread (see the wrapper
    below) and read by the Flask threads through ``snapshot()``. It only reads
    the existing pipeline output: no detector logic lives here.

    Timing rules (all from config):
      head turned left/right, up or down  < warn = SAFE, >= warn = WARNING,
                                          >= danger = DANGER
      eyes closed                         >= warn = WARNING, >= danger = DANGER
      yawning / face hidden 1.5-3 s       WARNING
      phone / face missing > 3 s          DANGER
    A head-pose timer is reset only when the face is detected AND the head is
    back near neutral (80% hysteresis). A face lost within 1.5 s of a turn keeps
    the timer counting; after that the face-lost rules take over.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._yaw_start: float | None = None
        self._yaw_direction: str | None = None
        self._up_start: float | None = None
        self._down_start: float | None = None
        self._levels: dict[str, int] = {}
        self._messages: deque = deque(maxlen=MESSAGE_HISTORY)
        self._snapshot = self._blank()
        self._last_face: float | None = None
        self._first_frame: float | None = None

    @staticmethod
    def _blank() -> dict:
        return {
            "banner": "SAFE",
            "message": "Waiting for the first frame",
            "flags": {
                "drowsy": False,
                "yawn": False,
                "looking_left": False,
                "looking_right": False,
                "looking_up": False,
                "looking_down": False,
                "phone": False,
                "no_driver": False,
                "unknown_driver": False,
            },
            "alerts": {},
            "EAR": 0.0,
            "MAR": 0.0,
            "yaw": 0.0,
            "pitch": 0.0,
            "pose_source": "none",
            "head_pose": {
                "source": "none",
                "raw": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                "smoothed": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            },
            "driver": "Unknown",
            "unknown_driver": False,
            "head": {},
        }

    def snapshot(self) -> dict:
        with self._lock:
            data = dict(self._snapshot)
            data["messages"] = list(self._messages)
            return data

    # -- internals -------------------------------------------------------
    @staticmethod
    def _head_level(seconds: float, warn_after: float, danger_after: float) -> int:
        if seconds >= danger_after:
            return _LEVEL_DANGER
        if seconds >= warn_after:
            return _LEVEL_WARNING
        return _LEVEL_NONE

    def _record(self, now: float, key: str, level: int, text: str, driver: str) -> None:
        stamp = datetime.fromtimestamp(now)
        full = stamp.strftime("%Y-%m-%d %H:%M:%S")
        level_name = _LEVEL_NAMES[level]
        self._messages.append(
            {
                "time": full,
                "level": level_name,
                "message": text,
                "text": f"{stamp.strftime('%H:%M:%S')} {level_name}: {text}",
            }
        )
        log_event(full, driver or "Unknown", _ALERT_EVENT_TYPES.get(key, key.upper()))

    def update(self, result: dict) -> None:
        now = time.time()
        metrics = result.get("metrics") or {}
        flags = result.get("flags") or {}
        timers = result.get("timers") or {}
        phone = result.get("phone") or {}
        driver = result.get("driver") or {}
        face = bool(result.get("face_detected"))

        yaw = float(metrics.get("yaw", 0.0))
        pitch = float(metrics.get("pitch", 0.0))
        ear = float(metrics.get("ear", 0.0))
        mar = float(metrics.get("mar", 0.0))
        eyes_closed = float(timers.get("eyes_closed", 0.0))
        face_missing = float(timers.get("face_missing", 0.0))

        # The pipeline reports 0 while no face has ever been seen, so the missing
        # duration is also derived here: from the last seen face, or from the
        # first processed frame when the seat has been empty since startup.
        if self._first_frame is None:
            self._first_frame = now
        if face:
            self._last_face = now
        reference = self._last_face if self._last_face is not None else self._first_frame
        missing = max(face_missing, now - reference)

        yaw_thresh = float(config.YAW_ATTENTION_THRESHOLD)
        yaw_clear = yaw_thresh * float(config.HEAD_HYSTERESIS_RATIO)
        down_thresh = float(config.PITCH_DOWN_ATTENTION_THRESHOLD)
        down_clear = down_thresh * float(config.HEAD_HYSTERESIS_RATIO)
        up_thresh = float(config.PITCH_UP_THRESHOLD)
        up_clear = up_thresh * float(config.HEAD_HYSTERESIS_RATIO)
        warn_after = float(config.YAW_ATTENTION_TIME_SEC)
        danger_after = float(config.YAW_DANGER_TIME_SEC)
        face_warn = float(config.FACE_LOST_TURN_HOLD_SEC)
        face_danger = float(config.FACE_LOST_NO_DRIVER_SEC)

        # -- head-turn timer ---------------------------------------------
        if not face and missing >= face_warn:
            # Face gone for longer than the hold window: the face-lost rules
            # take over, so the turn/nod stop counting.
            self._yaw_start = None
            self._yaw_direction = None
            self._up_start = None
            self._down_start = None
        elif abs(yaw) > yaw_thresh:
            if self._yaw_start is None:
                self._yaw_start = now
            self._yaw_direction = "LEFT" if yaw < 0 else "RIGHT"
        elif face and self._yaw_start is not None:
            neutral = abs(yaw) <= yaw_clear and down_clear <= pitch <= up_clear
            if neutral:
                self._yaw_start = None
                self._yaw_direction = None

        # -- pitch timers (down / up) -------------------------------------
        # While the face is briefly lost the pitch value holds, so a nod keeps
        # counting; after the hold window the block above clears the timers.
        if pitch < down_thresh:
            if self._down_start is None:
                self._down_start = now
        elif face and pitch >= down_clear:
            self._down_start = None

        if pitch > up_thresh:
            if self._up_start is None:
                self._up_start = now
        elif face and pitch <= up_clear:
            self._up_start = None

        yaw_seconds = (now - self._yaw_start) if self._yaw_start is not None else 0.0
        up_seconds = (now - self._up_start) if self._up_start is not None else 0.0
        down_seconds = (now - self._down_start) if self._down_start is not None else 0.0

        alerts: dict[str, int] = {}
        texts: dict[str, str] = {}

        def raise_alert(key: str, level: int, text: str) -> None:
            if level <= _LEVEL_NONE:
                return
            if level >= alerts.get(key, _LEVEL_NONE):
                alerts[key] = level
                texts[key] = text

        if self._yaw_start is not None and self._yaw_direction:
            level = self._head_level(yaw_seconds, warn_after, danger_after)
            side = self._yaw_direction.lower()
            raise_alert(
                "looking_left" if self._yaw_direction == "LEFT" else "looking_right",
                level,
                f"Looking {side} for {yaw_seconds:.1f}s",
            )

        down_level = self._head_level(down_seconds, warn_after, danger_after)
        raise_alert("looking_down", down_level, f"Head down for {down_seconds:.1f}s")

        up_level = self._head_level(up_seconds, warn_after, danger_after)
        raise_alert("looking_up", up_level, f"Head up for {up_seconds:.1f}s")

        microsleep = bool(flags.get("microsleep"))
        drowsy = bool(flags.get("drowsy"))
        if microsleep or drowsy:
            level = _LEVEL_DANGER if microsleep else _LEVEL_WARNING
            suffix = " - micro-sleep" if microsleep else ""
            raise_alert(
                "drowsiness",
                level,
                f"Eyes closed {eyes_closed:.1f}s (EAR {ear:.2f}){suffix}",
            )

        if flags.get("yawning"):
            raise_alert("yawning", _LEVEL_WARNING, f"Yawning (MAR {mar:.2f})")

        if flags.get("phone"):
            raise_alert(
                "phone",
                _LEVEL_DANGER,
                f"Phone usage detected (votes {phone.get('vote_str', '0/0')})",
            )

        if missing >= face_danger:
            raise_alert(
                "no_driver",
                _LEVEL_DANGER,
                f"NO DRIVER DETECTED - face missing {missing:.1f}s",
            )
        elif missing >= face_warn:
            raise_alert(
                "face_not_visible",
                _LEVEL_WARNING,
                f"FACE NOT VISIBLE - face missing {missing:.1f}s",
            )

        # -- banner = highest active level, ties by priority ---------------
        banner_level = _LEVEL_NONE
        banner_message = "Driver attentive - all detectors clear"
        best_key = ""
        for key, level in alerts.items():
            if level > banner_level or (
                level == banner_level
                and _ALERT_PRIORITY.get(key, 0) > _ALERT_PRIORITY.get(best_key, 0)
            ):
                banner_level = level
                banner_message = texts[key]
                best_key = key

        driver_name = str(driver.get("name") or "Unknown")
        unknown_driver = bool(flags.get("unknown_driver"))
        if not face:
            driver_name = "Unknown"

        out_flags = {
            "drowsy": "drowsiness" in alerts,
            "yawn": "yawning" in alerts,
            "looking_left": "looking_left" in alerts,
            "looking_right": "looking_right" in alerts,
            "looking_up": "looking_up" in alerts,
            "looking_down": "looking_down" in alerts,
            "phone": "phone" in alerts,
            "no_driver": "no_driver" in alerts,
            "unknown_driver": unknown_driver,
        }

        with self._lock:
            for key, level in alerts.items():
                if level > self._levels.get(key, _LEVEL_NONE):
                    self._record(now, key, level, texts[key], driver_name)
            self._levels = dict(alerts)
            self._snapshot = {
                "banner": _LEVEL_NAMES.get(banner_level, "SAFE"),
                "message": banner_message,
                "flags": out_flags,
                "alerts": {k: _LEVEL_NAMES[v] for k, v in alerts.items()},
                "EAR": round(ear, 3),
                "MAR": round(mar, 3),
                "yaw": round(yaw, 1),
                "pitch": round(pitch, 1),
                # Debug data: raw per-frame angles, smoothed angles and whether
                # they came from the facial transformation matrix or solvePnP.
                "pose_source": str(metrics.get("pose_source", "none")),
                "head_pose": {
                    "source": str(metrics.get("pose_source", "none")),
                    "raw": {
                        "yaw": round(float(metrics.get("raw_yaw", 0.0)), 1),
                        "pitch": round(float(metrics.get("raw_pitch", 0.0)), 1),
                        "roll": round(float(metrics.get("raw_roll", 0.0)), 1),
                    },
                    "smoothed": {
                        "yaw": round(yaw, 1),
                        "pitch": round(pitch, 1),
                        "roll": round(float(metrics.get("roll", 0.0)), 1),
                    },
                },
                "driver": driver_name,
                "unknown_driver": unknown_driver,
                "head": {
                    "yaw_seconds": round(yaw_seconds, 1),
                    "up_seconds": round(up_seconds, 1),
                    "down_seconds": round(down_seconds, 1),
                    "face_missing": round(missing, 1),
                },
            }


ALERTS = AlertEngine()

_raw_process_frame = process_frame


def _process_frame_with_alerts(frame, runtime: dict) -> dict:
    """Runs the unchanged pipeline, then feeds its result to the alert engine."""
    result = _raw_process_frame(frame, runtime)
    ALERTS.update(result)
    return result


# ProcessingWorker looks this name up at call time, so the display copy becomes
# the one with the alert hook; src/ui/pipeline.py itself is untouched.
process_frame = _process_frame_with_alerts


# ---------------------------------------------------------
# Processing thread
# ---------------------------------------------------------
class ProcessingWorker(threading.Thread):
    """Runs the existing pipeline on the newest frame; publishes JPEG + status."""

    def __init__(self, camera: FrameGrabber, runtime: dict, phone: AsyncPhone):
        super().__init__(name="drivesafe-processing", daemon=True)
        self.camera = camera
        self.runtime = runtime
        self.phone = phone
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._status = blank_status()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()
        self.phone.stop()

    def frame(self):
        with self._lock:
            return self._jpeg, self._seq

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    @staticmethod
    def _encode(frame) -> bytes | None:
        try:
            ok, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
        except Exception:
            return None
        return buffer.tobytes() if ok else None

    def _set_error(self, message: str) -> None:
        with self._lock:
            status = dict(self._status)
            status["error"] = message
            self._status = status

    def run(self) -> None:
        last_seq = 0
        times: deque = deque(maxlen=FPS_WINDOW)

        while not self._stop.is_set():
            seq, frame = self.camera.snapshot()
            if frame is None or seq == last_seq:
                time.sleep(0.005)  # wait for the next camera frame
                continue
            last_seq = seq

            started = time.perf_counter()
            try:
                result = process_frame(frame, self.runtime)
            except Exception as exc:
                self._set_error(f"Detector error: {exc}")
                time.sleep(0.2)
                continue
            process_ms = (time.perf_counter() - started) * 1000.0

            annotated = result.pop("frame", None)
            jpeg = self._encode(annotated) if annotated is not None else None

            times.append(time.time())
            fps = fps_from(times)

            self._seq += 1
            status = {
                "ready": True,
                "fps": round(fps, 1),
                "stream_fps": 0.0,  # filled in by Monitor.status()
                "process_ms": round(process_ms, 1),
                "timestamp": clock(),
                "error": None,
            }
            with self._lock:
                if jpeg is not None:
                    self._jpeg = jpeg
                self._status = status


# ---------------------------------------------------------
# Monitor
# ---------------------------------------------------------
class StreamMeter:
    """Annotated frames per second actually delivered to the browser."""

    def __init__(self, window: float = 2.0):
        self.window = window
        self._lock = threading.Lock()
        self._times: deque = deque()

    def tick(self) -> None:
        now = time.time()
        with self._lock:
            self._times.append(now)
            while self._times and now - self._times[0] > self.window:
                self._times.popleft()

    def fps(self) -> float:
        with self._lock:
            return fps_from(self._times)


STREAM_METER = StreamMeter()


def build_runtime() -> dict:
    """Detector instances, wired exactly like src/ui/camera.build_runtime()."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    from src.driver_auth import DriverAuthenticator
    from src.drowsiness import DrowsinessDetector
    from src.head_pose import HeadPoseEstimator
    from src.phone_detection import PhoneDetector

    base_options = mp_python.BaseOptions(
        model_asset_path=str(config.FACE_LANDMARKER_MODEL_PATH)
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    return {
        "landmarker": landmarker,
        "drowsiness": DrowsinessDetector(),
        "head_pose": HeadPoseEstimator(),
        "phone": PhoneDetector(),
        "auth": DriverAuthenticator(face_landmarker=landmarker),
    }


class Monitor:
    """Owns the camera, YOLO and processing threads of this server process."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = int(camera_index)
        self.camera: FrameGrabber | None = None
        self.processing: ProcessingWorker | None = None
        self.error: str | None = None

    def start(self) -> None:
        """Starts the camera thread, then the detector + processing threads.

        Called once from ``main()`` before ``app.run()`` so nothing waits for a
        request: the page can be opened the moment the port answers.
        """
        print("  [startup] starting the camera thread")
        self.camera = FrameGrabber(self.camera_index)
        self.camera.start()

        deadline = time.time() + CAMERA_START_TIMEOUT
        while time.time() < deadline:
            _, frame = self.camera.snapshot()
            if frame is not None or self.camera.error:
                break
            time.sleep(0.05)

        if self.camera.error:
            self.error = self.camera.error
            print(f"  [startup] {self.error}")
            self.camera.stop()
            return

        if self.camera.snapshot()[1] is None:
            # Not fatal: the grabber keeps trying and the stream sends the
            # placeholder until a real frame arrives.
            print(
                "  [startup] no camera frame yet - the page will show "
                "'Starting camera...' until one arrives"
            )

        if not config.FACE_LANDMARKER_MODEL_PATH.exists():
            self.error = (
                "The face landmark model is missing. Expected "
                f"{config.FACE_LANDMARKER_MODEL_PATH} - restore the file and restart."
            )
            print(f"  [startup] {self.error}")
            self.camera.stop()
            return

        print("  [startup] loading the detection models...")
        try:
            runtime = build_runtime()
        except Exception as exc:
            self.error = f"Could not load the detection models: {exc}"
            print(f"  [startup] {self.error}")
            self.camera.stop()
            return

        phone = AsyncPhone(runtime["phone"]).start()
        runtime["phone"] = phone
        self.processing = ProcessingWorker(self.camera, runtime, phone)
        self.processing.start()
        print("  [startup] processing thread running")

    def camera_ok(self) -> bool:
        if self.camera is None or self.camera.error is not None:
            return False
        _, frame = self.camera.snapshot()
        return frame is not None

    def status(self) -> dict:
        camera_error = self.camera.error if self.camera else None
        camera_ok = self.camera_ok()

        if self.processing is None:
            status = blank_status()
        else:
            status = self.processing.status()

        # Banner, flags, EAR/MAR/yaw/pitch, driver and the Messages history.
        status.update(ALERTS.snapshot())

        status["stream_fps"] = round(STREAM_METER.fps(), 1)
        status["camera_ok"] = camera_ok

        last_error = self.error or camera_error or status.get("error")
        status["last_error"] = last_error
        status["error"] = last_error  # the page reads this key
        if not camera_ok:
            status["ready"] = False  # whatever was published is now stale
        status["timestamp"] = clock()
        return status

    def frame(self):
        if self.processing is None:
            return None, 0
        return self.processing.frame()

    def stop(self) -> None:
        if self.processing is not None:
            self.processing.stop()
            self.processing.join(timeout=2.0)
        if self.camera is not None:
            self.camera.stop()
            self.camera.join(timeout=2.0)


# ---------------------------------------------------------
# Flask
# ---------------------------------------------------------
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>DriveSafe AI - Live Monitor</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #0a1120; color: #e6ecf7;
         font: 15px/1.45 system-ui, "Segoe UI", sans-serif; }
  main { max-width: 1080px; margin: 0 auto; padding: 20px 20px 36px; }
  .bar { display: flex; gap: 12px; align-items: center; justify-content: space-between;
         flex-wrap: wrap; margin-bottom: 14px; }
  h1 { margin: 0; font-size: 17px; letter-spacing: .3px; }
  h1 small { color: #8ea1bd; font-weight: 400; }
  .sound { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  button { font: inherit; font-size: 13px; padding: 7px 12px; border-radius: 8px;
           border: 1px solid #2b3d5c; background: #16233a; color: #e6ecf7; cursor: pointer; }
  button:hover { background: #1d2d49; }
  button.on { border-color: #16a34a; color: #86efac; }
  #audio-state { font-size: 12px; color: #8ea1bd; }
  .chip { font-size: 11px; letter-spacing: .8px; text-transform: uppercase;
          padding: 4px 9px; border-radius: 999px; border: 1px solid #2b3d5c; color: #8ea1bd; }
  .chip.ok { border-color: #15803d; color: #86efac; }
  .chip.bad { border-color: #b91c1c; color: #fca5a5; }

  .banner { display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap;
            padding: 18px 22px; border-radius: 14px; border: 1px solid transparent;
            margin-bottom: 16px; transition: background .18s ease, border-color .18s ease; }
  .banner strong { font-size: 34px; letter-spacing: 3px; line-height: 1; }
  .banner span { font-size: 15px; }
  .banner.safe { background: #0d2618; border-color: #15803d; color: #86efac; }
  .banner.warning { background: #2b2109; border-color: #b45309; color: #fcd34d; }
  .banner.danger { background: #2b0b0b; border-color: #b91c1c; color: #fca5a5; }

  .cols { display: grid; grid-template-columns: minmax(0, 2fr) minmax(220px, 1fr);
          gap: 16px; align-items: start; }
  @media (max-width: 820px) { .cols { grid-template-columns: 1fr; } }
  .video { aspect-ratio: 4 / 3; background: #000; border: 1px solid #24344f;
           border-radius: 12px; overflow: hidden; }
  .video img { display: block; width: 100%; height: 100%; object-fit: contain; }
  .numbers { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .num { background: #121b2c; border: 1px solid #24344f; border-radius: 10px; padding: 9px 11px; }
  .num span { display: block; font-size: 10px; letter-spacing: .8px; text-transform: uppercase; color: #8ea1bd; }
  .num strong { font-size: 20px; font-variant-numeric: tabular-nums; }

  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
           gap: 10px; margin-top: 16px; }
  .card { background: #121b2c; border: 1px solid #24344f; border-left-width: 4px;
          border-radius: 10px; padding: 10px 12px; }
  .card span { display: block; font-size: 10px; letter-spacing: .8px; text-transform: uppercase; color: #8ea1bd; }
  .card strong { font-size: 16px; font-variant-numeric: tabular-nums; }
  .card em { display: block; font-style: normal; font-size: 10px; letter-spacing: .6px; color: #fcd34d; }
  .card.off { border-left-color: #33445f; color: #7e90ad; }
  .card.warning { border-left-color: #d97706; background: #241c0a; color: #fcd34d; }
  .card.danger { border-left-color: #dc2626; background: #2a0f10; color: #fca5a5; }

  .log { margin-top: 18px; background: #0e1728; border: 1px solid #24344f; border-radius: 12px; }
  .log h2 { margin: 0; padding: 12px 14px; font-size: 12px; letter-spacing: .9px;
            text-transform: uppercase; color: #8ea1bd; border-bottom: 1px solid #24344f; }
  #messages { list-style: none; margin: 0; padding: 8px 14px 12px; max-height: 210px;
              overflow-y: auto; font: 13px/1.7 ui-monospace, "Cascadia Mono", Consolas, monospace; }
  #messages li { color: #9fb0c9; }
  #messages li.warning { color: #fcd34d; }
  #messages li.danger { color: #fca5a5; }
  #messages li.empty { color: #5b6b83; }
  #error { margin: 10px 0 0; color: #fca5a5; font-size: 13px; }

  body.danger::after { content: ""; position: fixed; inset: 0; pointer-events: none;
    box-shadow: inset 0 0 0 8px rgba(239, 68, 68, .95);
    animation: pulse .7s ease-in-out infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1 } 50% { opacity: .2 } }
</style>
</head>
<body>
<main>
  <header class="bar">
    <h1>DriveSafe AI <small>/ live monitor</small></h1>
    <div class="sound">
      <span id="camera" class="chip">camera unknown</span>
      <button id="enable" type="button">Enable sound</button>
      <button id="mute" type="button" hidden>Mute</button>
      <span id="audio-state">sound off</span>
    </div>
  </header>

  <section id="banner" class="banner safe">
    <strong id="banner-level">SAFE</strong>
    <span id="banner-message">Waiting for the first frame</span>
  </section>

  <div class="cols">
    <div class="video"><img id="video" src="/video_feed" alt="Live camera feed" /></div>
    <div class="numbers">
      <div class="num"><span>EAR</span><strong id="ear">-</strong></div>
      <div class="num"><span>MAR</span><strong id="mar">-</strong></div>
      <div class="num"><span>Yaw</span><strong id="yaw">-</strong></div>
      <div class="num"><span>Pitch</span><strong id="pitch">-</strong></div>
      <div class="num"><span>Processing FPS</span><strong id="fps">-</strong></div>
      <div class="num"><span>Stream FPS</span><strong id="stream">-</strong></div>
    </div>
  </div>

  <section class="cards" id="cards">
    <div class="card off" data-key="drowsiness"><span>Drowsy</span><strong>NO</strong></div>
    <div class="card off" data-key="yawning"><span>Yawn</span><strong>NO</strong></div>
    <div class="card off" data-key="looking_left"><span>Looking Left</span><strong>NO</strong></div>
    <div class="card off" data-key="looking_right"><span>Looking Right</span><strong>NO</strong></div>
    <div class="card off" data-key="looking_up"><span>Looking Up</span><strong>NO</strong></div>
    <div class="card off" data-key="looking_down"><span>Looking Down</span><strong>NO</strong></div>
    <div class="card off" data-key="phone"><span>Phone</span><strong>NO</strong></div>
    <div class="card off" data-key="no_driver"><span>No Driver</span><strong>NO</strong></div>
    <div class="card off" data-key="driver"><span>Driver</span><strong>-</strong><em id="driver-badge" hidden>UNKNOWN DRIVER</em></div>
  </section>

  <section class="log">
    <h2>Messages</h2>
    <ol id="messages"><li class="empty">No alerts yet.</li></ol>
  </section>

  <p id="error" hidden></p>
</main>
<script>
(function () {
  "use strict";

  var bannerEl = document.getElementById("banner");
  var levelEl = document.getElementById("banner-level");
  var messageEl = document.getElementById("banner-message");
  var cameraEl = document.getElementById("camera");
  var messagesEl = document.getElementById("messages");
  var cardsEl = document.getElementById("cards");
  var errorEl = document.getElementById("error");
  var enableBtn = document.getElementById("enable");
  var muteBtn = document.getElementById("mute");
  var audioState = document.getElementById("audio-state");
  var badgeEl = document.getElementById("driver-badge");

  var readout = {
    ear: document.getElementById("ear"),
    mar: document.getElementById("mar"),
    yaw: document.getElementById("yaw"),
    pitch: document.getElementById("pitch"),
    fps: document.getElementById("fps"),
    stream: document.getElementById("stream")
  };

  var CARD_KEYS = [
    "drowsiness", "yawning", "looking_left", "looking_right",
    "looking_up", "looking_down", "phone", "no_driver"
  ];

  var state = { banner: "SAFE", messages: -1, lastMessage: "" };

  // ---------------- sound ----------------
  var audio = { ctx: null, enabled: false, muted: false, loop: null };

  function tone(kind) {
    if (!audio.enabled || audio.muted || !audio.ctx) { return; }
    var ctx = audio.ctx;
    if (ctx.state === "suspended") { ctx.resume(); }
    var soft = kind === "warning";
    var now = ctx.currentTime;
    var osc = ctx.createOscillator();
    var gain = ctx.createGain();
    osc.type = soft ? "sine" : "square";
    osc.frequency.setValueAtTime(soft ? 620 : 960, now);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(soft ? 0.08 : 0.26, now + 0.012);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + (soft ? 0.16 : 0.22));
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now);
    osc.stop(now + (soft ? 0.2 : 0.26));
  }

  function stopLoop() {
    if (audio.loop !== null) {
      window.clearInterval(audio.loop);
      audio.loop = null;
    }
  }

  function startLoop() {
    stopLoop();
    if (!audio.enabled || audio.muted) { return; }
    tone("danger");
    audio.loop = window.setInterval(function () { tone("danger"); }, 1000);
  }

  function audioLabel() {
    if (!audio.enabled) { audioState.textContent = "sound off"; return; }
    audioState.textContent = audio.muted ? "muted" : "sound on";
  }

  enableBtn.addEventListener("click", function () {
    var Ctor = window.AudioContext || window.webkitAudioContext;
    if (!Ctor) {
      audioState.textContent = "audio unsupported";
      enableBtn.disabled = true;
      return;
    }
    if (!audio.ctx) { audio.ctx = new Ctor(); }
    if (audio.ctx.state === "suspended") { audio.ctx.resume(); }
    audio.enabled = true;
    audio.muted = false;
    enableBtn.hidden = true;
    enableBtn.classList.add("on");
    muteBtn.hidden = false;
    muteBtn.textContent = "Mute";
    muteBtn.classList.remove("on");
    audioLabel();
    tone("warning");
    if (state.banner === "DANGER") { startLoop(); }
  });

  muteBtn.addEventListener("click", function () {
    audio.muted = !audio.muted;
    muteBtn.textContent = audio.muted ? "Unmute" : "Mute";
    muteBtn.classList.toggle("on", audio.muted);
    audioLabel();
    if (audio.muted) { stopLoop(); }
    else if (state.banner === "DANGER") { startLoop(); }
  });

  // ---------------- banner ----------------
  function applyBanner(level, message) {
    bannerEl.className = "banner " + level.toLowerCase();
    levelEl.textContent = level;
    messageEl.textContent = message;
    document.body.classList.toggle("danger", level === "DANGER");

    if (level === state.banner) { return; }
    if (level === "DANGER") {
      startLoop();
      if (navigator.vibrate) { navigator.vibrate([300, 150, 300]); }
    } else {
      stopLoop();
      if (level === "WARNING") { tone("warning"); }
    }
    state.banner = level;
  }

  // ---------------- cards and numbers ----------------
  function setCard(key, level) {
    var el = cardsEl.querySelector('[data-key="' + key + '"]');
    if (!el) { return; }
    var cls = level === "DANGER" ? "danger" : (level === "WARNING" ? "warning" : "off");
    el.className = "card " + cls;
    el.querySelector("strong").textContent = level ? level : "NO";
  }

  function setDriver(name, unknown) {
    var el = cardsEl.querySelector('[data-key="driver"]');
    if (!el) { return; }
    var label = name && name !== "Unknown" ? name : "NO FACE";
    el.className = "card off";
    el.querySelector("strong").textContent = label;
    if (badgeEl) { badgeEl.hidden = !unknown; }
  }

  function fmt(value, digits) {
    var n = Number(value);
    return isFinite(n) ? n.toFixed(digits) : "-";
  }

  // ---------------- messages ----------------
  function escapeHtml(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function renderMessages(list) {
    var last = list.length ? list[list.length - 1].text : "";
    if (list.length === state.messages && last === state.lastMessage) { return; }
    state.messages = list.length;
    state.lastMessage = last;

    if (!list.length) {
      messagesEl.innerHTML = '<li class="empty">No alerts yet.</li>';
      return;
    }

    var html = "";
    for (var i = 0; i < list.length; i++) {
      var item = list[i];
      var cls = item.level === "DANGER" ? "danger" : (item.level === "WARNING" ? "warning" : "");
      html += '<li class="' + cls + '">' + escapeHtml(item.text) + "</li>";
    }
    messagesEl.innerHTML = html;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // ---------------- polling ----------------
  function poll() {
    fetch("/api/status", { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) { throw new Error("HTTP " + response.status); }
        return response.json();
      })
      .then(function (status) {
        applyBanner(status.banner || "SAFE", status.message || "");

        var alerts = status.alerts || {};
        for (var i = 0; i < CARD_KEYS.length; i++) {
          setCard(CARD_KEYS[i], alerts[CARD_KEYS[i]] || "");
        }
        setDriver(status.driver || "Unknown", !!status.unknown_driver);

        readout.ear.textContent = fmt(status.EAR, 2);
        readout.mar.textContent = fmt(status.MAR, 2);
        readout.yaw.textContent = fmt(status.yaw, 1) + "\u00b0";
        readout.pitch.textContent = fmt(status.pitch, 1) + "\u00b0";
        readout.fps.textContent = fmt(status.fps, 1);
        readout.stream.textContent = fmt(status.stream_fps, 1);

        cameraEl.textContent = status.camera_ok ? "camera ok" : "no camera";
        cameraEl.className = "chip " + (status.camera_ok ? "ok" : "bad");

        if (status.last_error) {
          errorEl.textContent = status.last_error;
          errorEl.hidden = false;
        } else {
          errorEl.hidden = true;
        }

        renderMessages(status.messages || []);
      })
      .catch(function () {
        cameraEl.textContent = "server unreachable";
        cameraEl.className = "chip bad";
      })
      .finally(function () {
        window.setTimeout(poll, 300);
      });
  }

  poll();
})();
</script>
</body>
</html>
"""

app = Flask(__name__, static_folder=None)
app.json.sort_keys = False

monitor = Monitor(config.CAMERA_INDEX)


_PLACEHOLDER_JPEG: bytes | None = None


def placeholder_jpeg() -> bytes:
    """A dark 'Starting camera...' frame, encoded once and reused."""
    global _PLACEHOLDER_JPEG
    if _PLACEHOLDER_JPEG is not None:
        return _PLACEHOLDER_JPEG

    frame = np.full((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), 16, dtype=np.uint8)
    for text, scale, position in (
        ("Starting camera...", 0.9, 0.46),
        ("waiting for the first frame", 0.5, 0.58),
    ):
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        origin = ((config.FRAME_WIDTH - size[0]) // 2, int(config.FRAME_HEIGHT * position))
        cv2.putText(
            frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
            (214, 226, 240), 2, cv2.LINE_AA,
        )

    ok, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    _PLACEHOLDER_JPEG = buffer.tobytes() if ok else b""
    return _PLACEHOLDER_JPEG


def mjpeg_part(jpeg: bytes) -> bytes:
    """One part of a multipart/x-mixed-replace response."""
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
        + jpeg
        + b"\r\n"
    )


def mjpeg_stream():
    """Sends a frame immediately, then one per newly processed frame.

    The very first part is a placeholder when the camera or detector is not
    ready yet, so the browser paints instead of waiting on an empty response.
    Real frames come from the worker's publish slot (JPEG + seq under one lock),
    so this loop never blocks on MediaPipe or YOLO.
    """
    placeholder = placeholder_jpeg()
    last_seq = -1
    last_sent = 0.0  # forces the first yield on connect

    while True:
        jpeg, seq = monitor.frame()
        now = time.time()

        if jpeg is not None and seq != last_seq:
            last_seq = seq
            last_sent = now
            STREAM_METER.tick()
            yield mjpeg_part(jpeg)
            continue

        # Nothing new: keep the connection alive with the placeholder.
        if placeholder and now - last_sent >= PLACEHOLDER_INTERVAL:
            last_sent = now
            yield mjpeg_part(placeholder)

        time.sleep(0.01)


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.get("/video_feed")
def video_feed():
    return Response(
        mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/status")
def api_status():
    response = jsonify(monitor.status())
    response.headers["Cache-Control"] = "no-store"
    return response


def main() -> None:
    print("DriveSafe AI - Flask live view")
    print(f"  camera : index {config.CAMERA_INDEX} (falls back to 0/1/2)")
    print(f"  models : {config.FACE_LANDMARKER_MODEL_PATH}")

    # Threads start here, once, before the server accepts requests.
    monitor.start()

    print(f"  url    : http://{HOST}:{PORT}")
    try:
        app.run(
            host=HOST,
            port=PORT,
            threaded=True,
            debug=False,
            use_reloader=False,  # a reloader would start the camera twice
        )
    finally:
        print("  [shutdown] releasing the camera")
        monitor.stop()


if __name__ == "__main__":
    main()
