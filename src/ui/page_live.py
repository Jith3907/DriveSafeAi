"""
PAGE 1 - Live Monitor.

Capture and detection run in a background worker at the camera's own pace, so the
detector loop stays above 15 FPS. The page fragment samples the newest snapshot,
renders the banner, cards, metrics and message box, and logs every event exactly
once when it starts. Leaving the page stops the worker and releases the webcam.
"""

from __future__ import annotations

import time

import streamlit as st

from src import config
from src.ui import camera, pipeline, state, theme

FRAGMENT_PERIOD_SEC = 0.1

CARD_LABELS = ["Drowsy", "Yawning", "Looking Away", "Phone", "No Driver", "Driver"]


def render() -> None:
    """Entry point called by app.py when the Live Monitor page is selected."""
    state.apply_settings()

    if st.session_state.camera_running:
        error = camera.worker_error()
        if error:
            camera.stop_camera(error)
        elif not camera.camera_ready():
            camera.stop_camera("The camera stopped responding. Click Retry to reconnect.")

    if st.session_state.camera_running:
        _live_fragment()
    else:
        _render_page(None, fps=0.0, process_ms=0.0)


# ---------------------------------------------------------
# Live fragment
# ---------------------------------------------------------
@st.fragment(run_every=FRAGMENT_PERIOD_SEC)
def _live_fragment() -> None:
    result = camera.worker_snapshot()
    error = camera.worker_error()
    if error:
        camera.stop_camera(error)
        st.rerun()
    if result is None:
        st.spinner("Starting camera and loading models…")
        return

    if result["seq"] != st.session_state.get("last_seq"):
        st.session_state.last_seq = result["seq"]
        _handle_transitions(result)
        st.session_state.last_result = result
        st.session_state.status_level = result["level"]
    else:
        result = st.session_state.get("last_result") or result

    _render_page(result, result["fps"], result["process_ms"])


# ---------------------------------------------------------
# Alerts, messages and event logging (once per event start)
# ---------------------------------------------------------
def _event_copy(event_id: str, result: dict) -> tuple[str, str, str]:
    level, text, icon = pipeline.EVENT_MESSAGES[event_id]
    flags = result["flags"]
    if event_id == pipeline.EVENT_DROWSY and flags["microsleep"]:
        return theme.LEVEL_DANGER, "Micro-sleep detected - eyes closed", "🚨"
    if event_id == pipeline.EVENT_LOOKING_AWAY and result["pose"]["severity"] == config.SEVERITY_CRITICAL:
        duration = max(result["timers"]["looking_away"], result["timers"]["head_down"])
        return theme.LEVEL_DANGER, f"Looking away from the road for {duration:.1f}s", "🚨"
    return level, text, icon


def _play_alert(level: str, text: str, event_id: str) -> None:
    alerts = st.session_state.get("_alerts")
    if alerts is None:
        return
    severity = config.SEVERITY_CRITICAL if level == theme.LEVEL_DANGER else config.SEVERITY_WARNING
    play_sound = level == theme.LEVEL_DANGER or (
        event_id == pipeline.EVENT_NO_DRIVER and config.NO_DRIVER_BEEP
    )
    try:
        alerts.trigger_alert(text, severity, play_sound=play_sound)
    except Exception:
        pass


def _handle_transitions(result: dict) -> None:
    """Logs and toasts only on state changes, never per frame."""
    now = float(result.get("captured_at") or time.time())
    active = st.session_state.active_events
    cooldowns = st.session_state.event_cooldowns
    current = set(result["events"])

    # Unknown driver only counts once it persists for the configured timeout.
    if result["flags"]["unknown_driver"]:
        if st.session_state.unknown_since is None:
            st.session_state.unknown_since = now
        elif now - st.session_state.unknown_since >= float(config.UNKNOWN_DRIVER_TIMEOUT_SEC):
            current.add(pipeline.EVENT_UNKNOWN_DRIVER)
    else:
        st.session_state.unknown_since = None

    for event_id in [event for event in active if event not in current]:
        info = active.pop(event_id)
        if not info.get("silent"):
            state.log_event_end(event_id, max(0.0, now - float(info.get("started", now))))
        cooldowns[event_id] = now

    for event_id in sorted(current - set(active)):
        if now - float(cooldowns.get(event_id, 0.0)) < state.EVENT_REARM_COOLDOWN_SEC:
            active[event_id] = {"started": now, "silent": True}
            continue

        level, text, icon = _event_copy(event_id, result)
        driver = result["driver"]["name"] if result["driver"]["authorized"] else "Unknown"
        state.log_event_start(event_id, driver, result["reason"])
        active[event_id] = {"started": now, "driver": driver}
        state.push_message(level, text)
        st.toast(f"{level}: {text}", icon=icon)
        _play_alert(level, text, event_id)


# ---------------------------------------------------------
# Rendering
# ---------------------------------------------------------
def _banner_sub(result: dict) -> str:
    flags = result["flags"]
    if flags["phone"]:
        return "Phone usage detected"
    if flags["microsleep"]:
        return "Micro-sleep detected - eyes closed"
    if result["pose"]["severity"] == config.SEVERITY_CRITICAL:
        return "Looking away from the road"
    if flags["no_driver"]:
        return "No driver detected"
    if flags["yawning"]:
        return "Yawning detected - possible fatigue"
    if flags["drowsy"]:
        return "Drowsiness detected - eyes closing"
    if flags["face_not_visible"]:
        return "Face not visible"
    if flags["looking_away"]:
        return "Looking away from the road"
    if flags["unknown_driver"]:
        return "Driver attentive - identity not recognised"
    return "Driver attentive & road focused"


def _metrics(result: dict | None, fps: float) -> list[dict]:
    if result is None:
        return [
            {"label": "EAR", "value": "—"},
            {"label": "MAR", "value": "—"},
            {"label": "Yaw", "value": "—"},
            {"label": "Pitch", "value": "—"},
            {"label": "FPS", "value": "—"},
        ]
    metrics = result["metrics"]
    return [
        {"label": "EAR", "value": f"{metrics['ear']:.2f}"},
        {"label": "MAR", "value": f"{metrics['mar']:.2f}"},
        {"label": "Yaw", "value": f"{metrics['yaw']:.0f}°"},
        {"label": "Pitch", "value": f"{metrics['pitch']:.0f}°"},
        {
            "label": "FPS",
            "value": f"{fps:.1f}",
            "tone": camera.streaming_score(fps) if fps > 0 else "",
        },
    ]


def _idle_cards() -> list[dict]:
    return [
        {"label": label, "value": "—", "detail": "Camera offline", "state": "off"}
        for label in CARD_LABELS
    ]


def _debug_rows(result: dict, fps: float, process_ms: float) -> list[tuple[str, str, str]]:
    metrics = result["metrics"]
    timers = result["timers"]
    phone = result["phone"]
    pose = result["pose"]
    flags = result["flags"]

    def yes_no(value: bool) -> str:
        return "YES" if value else "NO"

    return [
        ("face_detected", yes_no(result["face_detected"]), ""),
        (
            "driver",
            f"{result['driver']['name']} (sim {result['driver']['similarity']:.2f})",
            "authorized" if result["driver"]["authorized"] else "unknown",
        ),
        ("EAR raw / smooth", f"{metrics['raw_ear']:.3f} / {metrics['ear']:.3f}", f"< {config.EAR_THRESHOLD:.2f}"),
        ("MAR raw / smooth", f"{metrics['raw_mar']:.3f} / {metrics['mar']:.3f}", f"> {config.MAR_YAWN_THRESHOLD:.2f}"),
        ("yaw raw / smooth", f"{metrics['raw_yaw']:.1f}° / {metrics['yaw']:.1f}°", f"> {config.YAW_ATTENTION_THRESHOLD:.0f}°"),
        ("pitch raw / smooth", f"{metrics['raw_pitch']:.1f}° / {metrics['pitch']:.1f}°", f"< {config.PITCH_DOWN_ATTENTION_THRESHOLD:.0f}°"),
        ("roll raw / smooth", f"{metrics['raw_roll']:.1f}° / {metrics['roll']:.1f}°", ""),
        ("pose source", metrics.get("pose_source", "none"), "matrix preferred, solvePnP fallback"),
        ("timer: eyes closed", f"{timers['eyes_closed']:.2f} s", f">= {config.DROWSY_TIME_THRESHOLD_SEC:.1f} s"),
        ("timer: looking away", f"{timers['looking_away']:.2f} s", f">= {config.YAW_ATTENTION_TIME_SEC:.1f} s"),
        ("timer: head down", f"{timers['head_down']:.2f} s", f">= {config.PITCH_DOWN_ATTENTION_TIME_SEC:.1f} s"),
        ("timer: face missing", f"{timers['face_missing']:.2f} s", f">= {config.FACE_LOST_NO_DRIVER_SEC:.1f} s"),
        (
            "phone votes",
            f"{phone['vote_str']} (need {phone['votes_needed']})",
            "ALERT" if phone["votes_needed"] and phone["votes"] >= phone["votes_needed"] else "clear",
        ),
        ("phone confidence", f"{phone['confidence']:.2f}", f">= {config.PHONE_CONF:.2f}"),
        ("phone hold remaining", f"{phone['hold_remaining']:.2f} s", f"of {phone['hold_time']:.1f} s"),
        ("pose status", pose["status"], pose["severity"]),
        ("pose state reason", pose["reason"], ""),
        ("derived flags", ", ".join(name for name, on in flags.items() if on) or "none", result["level"]),
        ("threshold EAR / MAR", f"{config.EAR_THRESHOLD:.2f} / {config.MAR_YAWN_THRESHOLD:.2f}", ""),
        ("threshold yaw / pitch", f"{config.YAW_ATTENTION_THRESHOLD:.0f}° / {config.PITCH_DOWN_ATTENTION_THRESHOLD:.0f}°", ""),
        ("active events", ", ".join(sorted(st.session_state.active_events)) or "none", ""),
        ("loop", f"{fps:.1f} fps / {process_ms:.0f} ms per frame", ""),
        ("reason for state", result["reason"], result["level"]),
        ("phone reason", phone["reason"], ""),
    ]


def _render_controls(running: bool, fps: float) -> None:
    col_start, col_stop, col_info = st.columns([1.0, 1.0, 3.0])
    with col_start:
        if st.button(
            "▶ Start monitoring",
            key="start_camera",
            type="primary",
            width="stretch",
            disabled=running,
        ):
            if camera.start_camera():
                st.rerun()
    with col_stop:
        if st.button("⏹ Stop", key="stop_camera", width="stretch", disabled=not running):
            state.close_active_events()
            camera.stop_camera()
            st.rerun()
    with col_info:
        if running:
            st.caption(
                f"Camera {config.CAMERA_INDEX} · {config.FRAME_WIDTH}×{config.FRAME_HEIGHT} · "
                f"MediaPipe FaceLandmarker + YOLOv8n · {fps:.1f} FPS"
            )
        else:
            st.caption("Camera idle. Start monitoring to run the detectors.")


def _render_page(result: dict | None, fps: float, process_ms: float) -> None:
    level = result["level"] if result else theme.LEVEL_IDLE
    running = bool(st.session_state.camera_running)

    theme.page_header(level, live=running)
    _render_controls(running, fps)

    error = st.session_state.get("camera_error")
    if error:
        err_col, retry_col = st.columns([4.0, 1.0])
        with err_col:
            st.error(error, icon="📷")
        with retry_col:
            if st.button("🔄 Retry", key="retry_camera", width="stretch"):
                if camera.start_camera():
                    st.rerun()

    left, right = st.columns([1.15, 1.0], gap="medium")

    with left:
        st.markdown("<div class='ds-section-title'>Live camera</div>", unsafe_allow_html=True)
        video_slot = st.empty()
        if result is None or not result.get("jpeg"):
            video_slot.info("Click **▶ Start monitoring** to activate the webcam and detectors.")
        else:
            video_slot.markdown(theme.video_html(result["jpeg"]), unsafe_allow_html=True)
            st.caption(
                f"Driver: {result['driver']['name']} · {result['pose']['status']} · "
                f"{process_ms:.0f} ms/frame · {fps:.1f} FPS"
            )

    with right:
        st.markdown("<div class='ds-section-title'>Overall status</div>", unsafe_allow_html=True)
        banner_slot = st.empty()
        if result is None:
            banner_slot.markdown(
                theme.banner_html(
                    theme.LEVEL_IDLE,
                    "MONITORING PAUSED",
                    "Start the camera to begin monitoring",
                ),
                unsafe_allow_html=True,
            )
        else:
            banner_slot.markdown(
                theme.banner_html(result["level"], result["level"], _banner_sub(result)),
                unsafe_allow_html=True,
            )

        st.markdown("<div class='ds-section-title'>Indicators</div>", unsafe_allow_html=True)
        cards_slot = st.empty()
        cards_slot.markdown(
            theme.indicator_grid_html(result["cards"] if result else _idle_cards()),
            unsafe_allow_html=True,
        )

        st.markdown("<div class='ds-section-title'>Live metrics</div>", unsafe_allow_html=True)
        metrics_slot = st.empty()
        metrics_slot.markdown(theme.metrics_html(_metrics(result, fps)), unsafe_allow_html=True)

        st.markdown("<div class='ds-section-title'>Messages</div>", unsafe_allow_html=True)
        messages_slot = st.empty()
        messages_slot.markdown(
            theme.messages_html(st.session_state.live_messages),
            unsafe_allow_html=True,
        )

    st.markdown("---")
    show_debug = st.toggle("🐞 Debug telemetry", key="show_debug")
    if show_debug:
        if result is None:
            st.info("Start monitoring to inspect live detector telemetry.")
        else:
            st.markdown(
                theme.debug_table_html(_debug_rows(result, fps, process_ms)),
                unsafe_allow_html=True,
            )
            st.caption(
                "Debug shows raw/calibrated signals, timers, vote counts and the reason "
                "for the current overall state."
            )
    else:
        st.caption("Enable **Debug telemetry** to inspect face detection, timers, votes and state reasons.")
