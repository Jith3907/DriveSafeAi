"""
Session state, runtime settings and event-log persistence for DriveSafe AI.

The event log is the single source of truth for the Reports page:

    logs/events.csv -> timestamp, driver, event_type, duration

Rows are appended once when an event *starts*; the duration is filled in when the
event ends. Nothing is written per frame.
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config

EVENTS_PATH: Path = config.PROJECT_ROOT / "logs" / "events.csv"
EVENT_COLUMNS = ["timestamp", "driver", "event_type", "duration"]

# Settings surfaced on the Settings / Calibration pages. Page 1 only applies them.
DEFAULT_SETTINGS: dict = {
    "EAR_THRESHOLD": 0.22,
    "MAR_YAWN_THRESHOLD": 0.52,
    "DROWSY_TIME_THRESHOLD_SEC": 1.5,
    "YAW_ATTENTION_THRESHOLD": 28.0,
    "YAW_ATTENTION_TIME_SEC": 2.0,
    "YAW_DANGER_TIME_SEC": 4.0,
    "PITCH_DOWN_ATTENTION_THRESHOLD": -18.0,
    "PITCH_DOWN_ATTENTION_TIME_SEC": 2.5,
    "PITCH_UP_THRESHOLD": 22.0,
    "PITCH_DOWN_DANGER_TIME_SEC": 4.0,
    "FACE_LOST_TURN_HOLD_SEC": 1.5,
    "FACE_LOST_NO_DRIVER_SEC": 3.0,
    "PHONE_CONF": 0.28,
    "PHONE_VOTING_WINDOW": 6,
    "PHONE_VOTES_NEEDED": 3,
    "PHONE_HOLD_TIME_SEC": 1.5,
}

SETTING_LABELS: dict = {
    "EAR_THRESHOLD": "EAR threshold",
    "MAR_YAWN_THRESHOLD": "MAR threshold",
    "DROWSY_TIME_THRESHOLD_SEC": "Drowsy timer (s)",
    "YAW_ATTENTION_THRESHOLD": "Head-turn angle (deg)",
    "YAW_ATTENTION_TIME_SEC": "Looking-away attention (s)",
    "YAW_DANGER_TIME_SEC": "Looking-away danger (s)",
    "PITCH_DOWN_ATTENTION_THRESHOLD": "Head-down angle (deg)",
    "PITCH_UP_THRESHOLD": "Head-up angle (deg)",
    "PITCH_DOWN_ATTENTION_TIME_SEC": "Head-down attention (s)",
    "PITCH_DOWN_DANGER_TIME_SEC": "Head-down danger (s)",
    "FACE_LOST_TURN_HOLD_SEC": "Face-lost hold (s)",
    "FACE_LOST_NO_DRIVER_SEC": "No-driver timeout (s)",
    "PHONE_CONF": "PHONE_CONF",
    "PHONE_VOTING_WINDOW": "Phone vote window",
    "PHONE_VOTES_NEEDED": "Phone votes needed",
    "PHONE_HOLD_TIME_SEC": "Phone hold time (s)",
}


# ---------------------------------------------------------
# Event log
# ---------------------------------------------------------
def _write_empty_log() -> None:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=EVENT_COLUMNS).to_csv(EVENTS_PATH, index=False)


def ensure_events_csv() -> None:
    """Creates logs/events.csv and normalises legacy headers."""
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not EVENTS_PATH.exists():
        _write_empty_log()
        return
    try:
        df = pd.read_csv(EVENTS_PATH)
    except Exception:
        _write_empty_log()
        return
    if list(df.columns) == EVENT_COLUMNS:
        return
    if df.empty:
        _write_empty_log()
        return
    df = df.rename(
        columns={
            "Timestamp": "timestamp",
            "Event Type": "event_type",
            "Driver": "driver",
            "Details": "details",
        }
    )
    for column in EVENT_COLUMNS:
        if column not in df.columns:
            df[column] = 0.0 if column == "duration" else ""
    df[EVENT_COLUMNS].to_csv(EVENTS_PATH, index=False)


def log_event_start(event_type: str, driver: str = "Unknown", details: str = "") -> str:
    """Appends one row for a newly started event and returns its timestamp string."""
    ensure_events_csv()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": timestamp,
        "driver": driver or "Unknown",
        "event_type": event_type,
        "duration": "",
    }
    with open(EVENTS_PATH, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
        writer.writerow(row)
    return timestamp


def log_event_end(event_type: str, duration: float) -> None:
    """Fills in the duration of the most recent still-open row for this event type."""
    if not EVENTS_PATH.exists():
        return
    try:
        df = pd.read_csv(EVENTS_PATH)
    except Exception:
        return
    if df.empty or "duration" not in df.columns:
        return
    open_rows = df[
        (df["event_type"].astype(str) == str(event_type))
        & (df["duration"].isna() | (df["duration"].astype(str).str.strip() == ""))
    ]
    if open_rows.empty:
        return
    df.loc[open_rows.index[-1], "duration"] = round(float(duration), 2)
    df.to_csv(EVENTS_PATH, index=False)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
def load_settings() -> dict:
    """Reads persisted calibration/threshold values, falling back to defaults."""
    saved = config.load_calibration()
    settings = {}
    for key, default in DEFAULT_SETTINGS.items():
        value = saved.get(key, default)
        try:
            settings[key] = int(value) if isinstance(default, int) else float(value)
        except (TypeError, ValueError):
            settings[key] = default
    return settings


def get_settings() -> dict:
    if "settings" not in st.session_state:
        st.session_state.settings = load_settings()
    return st.session_state.settings


def apply_settings(settings: dict | None = None) -> None:
    """Pushes session settings into the config module used by the detectors."""
    settings = settings or get_settings()
    for key, value in settings.items():
        setattr(config, key, value)
    config.PHONE_CONF_THRESHOLD = float(settings.get("PHONE_CONF", config.PHONE_CONF_THRESHOLD))
    config.PHONE_CONF = config.PHONE_CONF_THRESHOLD


# ---------------------------------------------------------
# Messages + session state
# ---------------------------------------------------------
MAX_MESSAGES = 25


def push_message(level: str, text: str) -> dict:
    """Adds a timestamped alert message to the Live Monitor message box."""
    message = {
        "ts": datetime.now().strftime("%H:%M:%S"),
        "level": level,
        "text": text,
    }
    messages = st.session_state.setdefault("live_messages", [])
    messages.insert(0, message)
    del messages[MAX_MESSAGES:]
    return message


EVENT_REARM_COOLDOWN_SEC = 2.0


def close_active_events(now: float | None = None) -> None:
    """Closes every event still marked active, writing its duration to the log."""
    now = time.time() if now is None else now
    active = st.session_state.get("active_events") or {}
    cooldowns = st.session_state.setdefault("event_cooldowns", {})
    for event_id, info in list(active.items()):
        log_event_end(event_id, now - float(info.get("started", now)))
        cooldowns[event_id] = now
    st.session_state.active_events = {}


def init_state() -> None:
    """Initialises every session key the app relies on (idempotent)."""
    defaults = {
        "nav_page": "live",
        "last_page": None,
        "camera_running": False,
        "camera_error": None,
        "camera_retry_token": 0,
        "cap": None,
        "status_level": "IDLE",
        "live_messages": [],
        "active_events": {},
        "event_cooldowns": {},
        "unknown_since": None,
        "frame_times": [],
        "process_ms": 0.0,
        "show_debug": False,
        "monitor_started_at": None,
        "camera_restarts": 0,
        "read_failures": 0,
        "last_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    get_settings()
