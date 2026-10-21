"""
Camera + detector lifecycle for the Streamlit app.

Monitors run in a background ``MonitorWorker``; this module only owns opening the
webcam, building detector instances and releasing the hardware. The webcam is
released as soon as the Live Monitor page is left, when Stop is pressed, or when
the process exits.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time

import cv2
import streamlit as st

from src import config
from src.alerts import AlertSystem
from src.driver_auth import DriverAuthenticator
from src.drowsiness import DrowsinessDetector
from src.head_pose import HeadPoseEstimator
from src.phone_detection import PhoneDetector
from src.ui.worker import MonitorWorker

# Only one webcam capture may be live per server process. This also cleans up workers
# left behind by a previous browser session (a page reload creates a new session).
_GLOBAL_WORKER = None


def model_available() -> bool:
    return config.FACE_LANDMARKER_MODEL_PATH.exists()


def open_camera(index: int | None = None):
    """Opens the webcam, trying the DirectShow backend first on Windows."""
    index = config.CAMERA_INDEX if index is None else index
    backends = []
    if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    backends.append(None)

    for backend in backends:
        try:
            cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
        except Exception:
            cap = None
        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap, None
        if cap is not None:
            cap.release()

    return None, (
        f"Camera {index} could not be opened. Check that it is plugged in, "
        "not used by another app, and that camera access is allowed."
    )


def build_runtime() -> dict:
    """Builds a fresh set of detector instances (thread-safe, no session state)."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base_options = mp_python.BaseOptions(model_asset_path=str(config.FACE_LANDMARKER_MODEL_PATH))
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


def get_runtime() -> dict:
    """Session-cached runtime for pages that need detectors outside the worker."""
    runtime = st.session_state.get("_runtime")
    if runtime is None:
        runtime = build_runtime()
        st.session_state["_runtime"] = runtime
    return runtime


def release_camera() -> None:
    st.session_state["cap"] = None


def _build_alert_system():
    try:
        return AlertSystem()
    except Exception:
        return None


def _terminate_stale_monitors() -> None:
    """Kills detector child processes that no session owns any more.

    A browser reload creates a new Streamlit session without stopping the old
    session's process, which would keep the webcam busy. All monitor processes are
    children of this server process, so they can always be found this way.
    """
    for child in mp.active_children():
        if child.name != "drivesafe-monitor":
            continue
        try:
            if child.is_alive():
                child.terminate()
                child.join(timeout=1.0)
        except Exception:
            pass


def start_camera() -> bool:
    """Opens the camera, starts the worker and marks monitoring as running."""
    if not model_available():
        st.session_state.camera_error = (
            "The face landmark model is missing. Expected "
            f"`{config.FACE_LANDMARKER_MODEL_PATH}` - restore the file and retry."
        )
        st.session_state.camera_running = False
        return False

    _stop_worker()
    _stop_orphaned_worker()
    _terminate_stale_monitors()
    release_camera()

    worker = MonitorWorker(config.CAMERA_INDEX).start()
    deadline = time.time() + 20.0
    while time.time() < deadline:
        worker.poll()
        if worker.error or worker.snapshot():
            break
        time.sleep(0.05)

    if worker.error:
        message = worker.error
        worker.stop()
        st.session_state.camera_error = message
        st.session_state.camera_running = False
        return False

    if not worker.snapshot():
        worker.stop()
        st.session_state.camera_error = (
            f"Camera {config.CAMERA_INDEX} did not produce any frames. "
            "Close other apps that use the camera and click Retry."
        )
        st.session_state.camera_running = False
        return False

    global _GLOBAL_WORKER
    _GLOBAL_WORKER = worker
    st.session_state["_worker"] = worker
    st.session_state["_alerts"] = _build_alert_system()
    st.session_state.camera_error = None
    st.session_state.camera_running = True
    st.session_state.monitor_started_at = time.time()
    st.session_state.status_level = "SAFE"
    st.session_state.last_seq = 0
    return True


def _stop_worker() -> None:
    global _GLOBAL_WORKER
    worker = st.session_state.pop("_worker", None)
    if _GLOBAL_WORKER is worker:
        _GLOBAL_WORKER = None
    if worker is not None:
        try:
            worker.stop()
        except Exception:
            pass


def _stop_orphaned_worker() -> None:
    """Stops a worker left running by a previous (closed/reloaded) browser session."""
    global _GLOBAL_WORKER
    worker = _GLOBAL_WORKER
    _GLOBAL_WORKER = None
    if worker is not None and st.session_state.get("_worker") is not worker:
        try:
            worker.stop()
        except Exception:
            pass


def stop_camera(reason: str | None = None) -> None:
    """Stops monitoring, releases the webcam and clears live timers."""
    _stop_worker()
    release_camera()
    st.session_state.camera_running = False
    st.session_state.status_level = "IDLE"
    st.session_state.active_events = {}
    st.session_state.unknown_since = None
    st.session_state.last_seq = 0
    if reason:
        st.session_state.camera_error = reason


def cleanup_stale_worker() -> None:
    """Stops a monitor worker that belongs to no current session (page reload)."""
    global _GLOBAL_WORKER
    worker = _GLOBAL_WORKER
    if worker is None:
        return
    if st.session_state.get("_worker") is worker:
        return
    try:
        worker.stop()
    except Exception:
        pass
    _GLOBAL_WORKER = None


def camera_ready() -> bool:
    worker = st.session_state.get("_worker")
    if worker is None:
        return False
    if worker.error:
        return False
    return worker.alive()


def worker_snapshot() -> dict | None:
    worker = st.session_state.get("_worker")
    if worker is None:
        return None
    return worker.snapshot()


def worker_error() -> str | None:
    worker = st.session_state.get("_worker")
    return worker.error if worker is not None else None


def streaming_score(fps: float) -> str:
    return "good" if fps >= 15.0 else "bad"
