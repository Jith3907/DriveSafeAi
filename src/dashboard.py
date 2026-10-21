"""
DriveSafe AI - Interactive Streamlit Verification & Monitoring Dashboard.
Integrates live camera feed, real-time indicator cards, rolling telemetry charts,
neutral pose calibration, auto-suggest EAR/MAR calibration, debug telemetry expander,
and manual test checklist with event logging.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.drowsiness import DrowsinessDetector
from src.head_pose import HeadPoseEstimator
from src.phone_detection import PhoneDetector
from src.driver_auth import DriverAuthenticator
from src.alerts import AlertSystem

# =========================================================
# Page Configuration & Modern Dark Styling
# =========================================================
st.set_page_config(
    page_title="DriveSafe AI - Verification Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    
    /* Indicator Cards */
    .indicator-card {
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
        color: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: all 0.3s ease;
    }
    .indicator-inactive {
        background: #1e222d;
        border: 1px solid #2d3343;
        color: #94a3b8;
    }
    .indicator-active-red {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        border: 1px solid #ef4444;
        color: #ffffff;
        animation: pulse 1.5s infinite;
    }
    .indicator-active-amber {
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
        border: 1px solid #f59e0b;
        color: #ffffff;
    }
    .indicator-active-green {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
        border: 1px solid #22c55e;
        color: #ffffff;
    }
    .indicator-driver-unknown {
        background: #1e293b;
        border: 1px solid #475569;
        color: #cbd5e1;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    /* Overall Status Banners */
    .banner-safe {
        background: linear-gradient(90deg, #14532d 0%, #166534 100%);
        color: #bbf7d0;
        padding: 12px 20px;
        border-radius: 10px;
        border: 1px solid #22c55e;
        font-weight: 700;
        font-size: 1.15rem;
        text-align: center;
        margin-bottom: 14px;
    }
    .banner-warning {
        background: linear-gradient(90deg, #78350f 0%, #92400e 100%);
        color: #fef08a;
        padding: 12px 20px;
        border-radius: 10px;
        border: 1px solid #eab308;
        font-weight: 700;
        font-size: 1.15rem;
        text-align: center;
        margin-bottom: 14px;
    }
    .banner-danger {
        background: linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%);
        color: #fecaca;
        padding: 12px 20px;
        border-radius: 10px;
        border: 1px solid #ef4444;
        font-weight: 700;
        font-size: 1.15rem;
        text-align: center;
        margin-bottom: 14px;
        animation: pulse 1.5s infinite;
    }
    .banner-calib {
        background: linear-gradient(90deg, #1e3a8a 0%, #1d4ed8 100%);
        color: #dbeafe;
        padding: 12px 20px;
        border-radius: 10px;
        border: 1px solid #3b82f6;
        font-weight: 700;
        font-size: 1.15rem;
        text-align: center;
        margin-bottom: 14px;
    }

    /* Metric display tiles */
    .metric-badge {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 8px 10px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        color: #9ca3af;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #38bdf8;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================================================
# File Paths & Directories
# =========================================================
LOGS_DIR = config.PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_CSV_PATH = LOGS_DIR / "events.csv"
CALIBRATION_JSON_PATH = config.CALIBRATION_FILE_PATH

if not EVENTS_CSV_PATH.exists():
    pd.DataFrame(columns=["Timestamp", "Event Type", "Driver", "Details"]).to_csv(EVENTS_CSV_PATH, index=False)

# Load existing calibration into config
calib_data = config.load_calibration()

# =========================================================
# Session State Initialization
# =========================================================
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if "calib_mode" not in st.session_state:
    st.session_state.calib_mode = None  # None, "neutral_pose", "ear_open", "ear_closed"

if "calib_start_time" not in st.session_state:
    st.session_state.calib_start_time = None

if "calib_samples" not in st.session_state:
    st.session_state.calib_samples = []

if "calib_open_ear_samples" not in st.session_state:
    st.session_state.calib_open_ear_samples = []

if "calib_resting_mar_samples" not in st.session_state:
    st.session_state.calib_resting_mar_samples = []

if "calib_closed_ear_samples" not in st.session_state:
    st.session_state.calib_closed_ear_samples = []

if "startup_calibrated" not in st.session_state:
    st.session_state.startup_calibrated = False

if "checklist_results" not in st.session_state:
    st.session_state.checklist_results = {
        "1. Eyes closed (>=1.5s)": "PENDING",
        "2. Blink (quick blink does NOT alert)": "PENDING",
        "3. Yawn (wide mouth)": "PENDING",
        "4. Head turn (>35° for 2s)": "PENDING",
        "5. Phone in hand (>=5 frames)": "PENDING",
        "6. Unknown driver (unregistered face)": "PENDING",
    }

if "active_events" not in st.session_state:
    st.session_state.active_events = set()

if "rolling_history" not in st.session_state:
    st.session_state.rolling_history = []

def log_event_start(event_type, driver, details=""):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = pd.DataFrame([{
        "Timestamp": now_str,
        "Event Type": event_type,
        "Driver": driver,
        "Details": details
    }])
    row.to_csv(EVENTS_CSV_PATH, mode="a", header=False, index=False)

# =========================================================
# Sidebar Controls & Calibration
# =========================================================
st.sidebar.title("🛡️ Calibration & Controls")

st.sidebar.subheader("🎯 Pose & Eye Calibration")
col_cal1, col_cal2 = st.sidebar.columns(2)
with col_cal1:
    if st.button("🎯 Calibrate Neutral (3s)", use_container_width=True, help="Look straight for 3s to zero out yaw/pitch/roll"):
        st.session_state.calib_mode = "neutral_pose"
        st.session_state.calib_start_time = None
        st.session_state.calib_samples = []
with col_cal2:
    if st.button("👁️ Calibrate EAR/MAR", use_container_width=True, help="Auto-suggest EAR/MAR (5s open + 5s closed)"):
        st.session_state.calib_mode = "ear_open"
        st.session_state.calib_start_time = None
        st.session_state.calib_open_ear_samples = []
        st.session_state.calib_resting_mar_samples = []
        st.session_state.calib_closed_ear_samples = []

auto_calib_startup = st.sidebar.checkbox("Auto-calibrate neutral pose at startup", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Live Detection Thresholds")

ear_thresh_slider = st.sidebar.slider(
    "EAR Threshold (Eye Closure)",
    min_value=0.10,
    max_value=0.40,
    value=float(calib_data.get("EAR_THRESHOLD", config.EAR_THRESHOLD)),
    step=0.01,
    help="Eyes below this EAR are marked closed. Hysteresis clears at +11%."
)

mar_thresh_slider = st.sidebar.slider(
    "MAR Yawn Threshold",
    min_value=0.30,
    max_value=0.80,
    value=float(calib_data.get("MAR_YAWN_THRESHOLD", config.MAR_YAWN_THRESHOLD)),
    step=0.01,
    help="Mouth aspect ratio above this value triggers yawning. Hysteresis clears at 90%."
)

drowsy_duration_slider = st.sidebar.slider(
    "Drowsy Closure Duration (s)",
    min_value=0.5,
    max_value=4.0,
    value=float(calib_data.get("DROWSY_TIME_THRESHOLD_SEC", config.DROWSY_TIME_THRESHOLD_SEC)),
    step=0.1,
    help="Sustained seconds of closed eyes before triggering Drowsy alert."
)

st.sidebar.markdown("**Head Turn Tolerances (Normal Driving)**")
yaw_angle_slider = st.sidebar.slider(
    "Yaw Turn Angle Threshold (°)",
    min_value=20.0,
    max_value=60.0,
    value=float(calib_data.get("YAW_ATTENTION_THRESHOLD", config.YAW_ATTENTION_THRESHOLD)),
    step=1.0,
    help="Head turn beyond this angle starts the distraction timer."
)

yaw_attention_time_slider = st.sidebar.slider(
    "Yaw Attention Duration (s)",
    min_value=1.0,
    max_value=5.0,
    value=float(calib_data.get("YAW_ATTENTION_TIME_SEC", config.YAW_ATTENTION_TIME_SEC)),
    step=0.5,
    help="Sustained turn duration for ATTENTION (yellow status, silent)."
)

yaw_danger_time_slider = st.sidebar.slider(
    "Yaw Danger Duration (s)",
    min_value=2.0,
    max_value=8.0,
    value=float(calib_data.get("YAW_DANGER_TIME_SEC", config.YAW_DANGER_TIME_SEC)),
    step=0.5,
    help="Sustained turn duration for DANGER (red status, audio chime)."
)

st.sidebar.markdown("**Looking Down Tolerances (Nodding Off)**")
pitch_down_slider = st.sidebar.slider(
    "Pitch Down Threshold (°)",
    min_value=-50.0,
    max_value=-15.0,
    value=float(calib_data.get("PITCH_DOWN_ATTENTION_THRESHOLD", config.PITCH_DOWN_ATTENTION_THRESHOLD)),
    step=1.0,
    help="Looking down angle below this threshold starts the nodding timer."
)

pitch_down_attention_slider = st.sidebar.slider(
    "Pitch Down Attention Duration (s)",
    min_value=1.0,
    max_value=5.0,
    value=float(calib_data.get("PITCH_DOWN_ATTENTION_TIME_SEC", config.PITCH_DOWN_ATTENTION_TIME_SEC)),
    step=0.5,
    help="Sustained head down duration for ATTENTION (yellow status, silent)."
)

pitch_down_danger_slider = st.sidebar.slider(
    "Pitch Down Danger Duration (s)",
    min_value=2.0,
    max_value=8.0,
    value=float(calib_data.get("PITCH_DOWN_DANGER_TIME_SEC", config.PITCH_DOWN_DANGER_TIME_SEC)),
    step=0.5,
    help="Sustained head down duration for DANGER (red status, audio chime)."
)

st.sidebar.markdown("**Face-Lost Timers (Part A/B)**")
face_lost_hold_slider = st.sidebar.slider(
    "Face-Lost Turn Hold (s)",
    min_value=0.5, max_value=3.0,
    value=float(calib_data.get("FACE_LOST_TURN_HOLD_SEC", config.FACE_LOST_TURN_HOLD_SEC)),
    step=0.1,
    help="If face lost within this many seconds of a sharp turn, keep looking-away state (Part A)."
)
face_lost_no_driver_slider = st.sidebar.slider(
    "No-Driver Timeout (s)",
    min_value=1.5, max_value=8.0,
    value=float(calib_data.get("FACE_LOST_NO_DRIVER_SEC", config.FACE_LOST_NO_DRIVER_SEC)),
    step=0.5,
    help="Face missing longer than this shows NO DRIVER DETECTED (gray, not CRITICAL)."
)
no_driver_beep = st.sidebar.checkbox(
    "🔔 Soft beep when NO DRIVER detected",
    value=bool(calib_data.get("NO_DRIVER_BEEP", config.NO_DRIVER_BEEP)),
    help="Play a soft audio beep when face has been absent > No-Driver Timeout."
)

st.sidebar.markdown("**📱 Phone Detection (Part C)**")
phone_conf_slider = st.sidebar.slider(
    "Phone Conf Threshold",
    min_value=0.10, max_value=0.70,
    value=float(calib_data.get("PHONE_CONF", config.PHONE_CONF_THRESHOLD)),
    step=0.01,
    help="YOLO confidence threshold. Lower = more sensitive (default 0.28)."
)
phone_votes_needed_slider = st.sidebar.slider(
    "Phone Votes Needed",
    min_value=1, max_value=6,
    value=int(calib_data.get("PHONE_VOTES_NEEDED", config.PHONE_VOTES_NEEDED)),
    step=1,
    help="Number of True votes in the voting window to trigger alert."
)
phone_voting_window_slider = st.sidebar.slider(
    "Phone Voting Window Size",
    min_value=2, max_value=10,
    value=int(calib_data.get("PHONE_VOTING_WINDOW", config.PHONE_VOTING_WINDOW)),
    step=1,
    help="How many YOLO runs to keep in the voting window."
)
phone_hold_time_slider = st.sidebar.slider(
    "Phone Hold Time (s)",
    min_value=0.5, max_value=5.0,
    value=float(calib_data.get("PHONE_HOLD_TIME_SEC", config.PHONE_HOLD_TIME_SEC)),
    step=0.1,
    help="After last positive detection, keep alert active for this many seconds."
)
phone_imgsz_slider = st.sidebar.select_slider(
    "Phone YOLO imgsz",
    options=[320, 416, 480, 640],
    value=int(calib_data.get("PHONE_IMGSZ", config.PHONE_IMGSZ)),
    help="YOLO input resolution. Higher = better but slower."
)

if st.sidebar.button("💾 Save Settings to calibration.json", use_container_width=True):
    saved_cfg = {
        "EAR_THRESHOLD": round(ear_thresh_slider, 3),
        "MAR_YAWN_THRESHOLD": round(mar_thresh_slider, 3),
        "DROWSY_TIME_THRESHOLD_SEC": round(drowsy_duration_slider, 1),
        "YAW_ATTENTION_THRESHOLD": round(yaw_angle_slider, 1),
        "YAW_ATTENTION_TIME_SEC": round(yaw_attention_time_slider, 1),
        "YAW_DANGER_TIME_SEC": round(yaw_danger_time_slider, 1),
        "PITCH_DOWN_ATTENTION_THRESHOLD": round(pitch_down_slider, 1),
        "PITCH_DOWN_ATTENTION_TIME_SEC": round(pitch_down_attention_slider, 1),
        "PITCH_DOWN_DANGER_TIME_SEC": round(pitch_down_danger_slider, 1),
        "neutral_yaw": round(config.NEUTRAL_YAW, 2),
        "neutral_pitch": round(config.NEUTRAL_PITCH, 2),
        "neutral_roll": round(config.NEUTRAL_ROLL, 2),
        "FACE_LOST_TURN_HOLD_SEC": round(face_lost_hold_slider, 1),
        "FACE_LOST_NO_DRIVER_SEC": round(face_lost_no_driver_slider, 1),
        "NO_DRIVER_BEEP": no_driver_beep,
        "PHONE_CONF": round(phone_conf_slider, 2),
        "PHONE_CONF_THRESHOLD": round(phone_conf_slider, 2),
        "PHONE_VOTES_NEEDED": phone_votes_needed_slider,
        "PHONE_VOTING_WINDOW": phone_voting_window_slider,
        "PHONE_HOLD_TIME_SEC": round(phone_hold_time_slider, 1),
        "PHONE_IMGSZ": phone_imgsz_slider,
    }
    config.save_calibration(saved_cfg)
    st.sidebar.success("Saved to data/calibration.json!")

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Driver Registration")
reg_name_input = st.sidebar.text_input("New Driver Name", value="Primary Driver")
register_requested = st.sidebar.button("📸 Register Current Face as Driver", use_container_width=True)

# =========================================================
# Header & Top Banner
# =========================================================
st.title("🚗 DriveSafe AI - Real-time Verification Dashboard")
st.caption("Edge monitoring with pitch normalization, neutral calibration, smoothing, hysteresis & driving tolerances")

banner_placeholder = st.empty()

# Layout: Left column (Webcam feed), Right column (Indicators, Live Stats & Chart)
col_left, col_right = st.columns([1.15, 1.0], gap="medium")

with col_left:
    st.subheader("📹 Live Monitoring Feed")
    cam_btn_col1, cam_btn_col2 = st.columns([1, 1])
    with cam_btn_col1:
        if st.button("▶️ Start Camera", use_container_width=True, type="primary"):
            st.session_state.camera_running = True
    with cam_btn_col2:
        if st.button("⏹️ Stop Camera", use_container_width=True):
            st.session_state.camera_running = False

    video_placeholder = st.empty()
    fps_badge_placeholder = st.empty()

with col_right:
    st.subheader("📊 Hazard Status Panel")
    
    # Indicator Card Placeholders
    card_drowsy = st.empty()
    card_yawn = st.empty()
    card_pose = st.empty()
    card_phone = st.empty()
    card_driver = st.empty()  # Requirement 5: Separate identity badge

    # Live telemetry numbers row
    st.markdown("##### ⏱️ Telemetry Metrics")
    metrics_row = st.empty()

    # Rolling chart section
    st.markdown("##### 📈 Rolling EAR & MAR Telemetry (Last 10 Seconds)")
    chart_placeholder = st.empty()

# =========================================================
# Requirement 1: Debug Expander
# =========================================================
st.markdown("---")
debug_expander = st.expander("🛠️ Debug Telemetry & Raw Detector States", expanded=True)
with debug_expander:
    st.caption("Live raw signals, calibrated values, active thresholds, and detector state breakdown.")
    debug_metrics_placeholder = st.empty()
    debug_status_placeholder = st.empty()

# =========================================================
# Verification Checklist Section
# =========================================================
st.markdown("---")
st.subheader("📋 Verification Checklist")
st.caption("Perform each test in front of the camera, verify the live detection indicator, and mark PASS or FAIL.")

checklist_specs = [
    {
        "id": "1. Eyes closed (>=1.5s)",
        "instruction": "Close both eyes continuously for 1.5 seconds. Drowsiness alert card turns RED.",
        "key": "drowsy"
    },
    {
        "id": "2. Blink (quick blink does NOT alert)",
        "instruction": "Blink normally. Smoothing & 1.5s threshold prevent alert. Status stays SAFE.",
        "key": "blink_safe"
    },
    {
        "id": "3. Yawn (wide mouth)",
        "instruction": "Open mouth wide to simulate a yawn. Yawning card turns AMBER.",
        "key": "yawn"
    },
    {
        "id": "4. Head turn (>35° for 2s)",
        "instruction": "Turn head > 35°. Brief check (<2s) is SAFE; >= 2s turns ATTENTION; >= 4s turns DANGER.",
        "key": "head_turn"
    },
    {
        "id": "5. Phone in hand (>=5 frames)",
        "instruction": "Hold a smartphone in view in front of the camera. Phone alert card turns RED.",
        "key": "phone"
    },
    {
        "id": "6. Unknown driver (unregistered face)",
        "instruction": "Displays 'Driver: Unknown' badge. Does NOT turn the overall banner to Danger.",
        "key": "unknown_driver"
    }
]

check_cols = st.columns([2.5, 1.2, 0.8, 0.8, 1.0])
check_cols[0].markdown("**Test Case & Instruction**")
check_cols[1].markdown("**Live Status**")
check_cols[2].markdown("**Action**")
check_cols[3].markdown("**Action**")
check_cols[4].markdown("**Marked Result**")

live_test_badges = {}
for test in checklist_specs:
    r_cols = st.columns([2.5, 1.2, 0.8, 0.8, 1.0])
    r_cols[0].markdown(f"**{test['id']}**<br><span style='font-size:0.85rem; color:#9ca3af;'>{test['instruction']}</span>", unsafe_allow_html=True)
    live_test_badges[test['key']] = r_cols[1].empty()
    live_test_badges[test['key']].markdown("<span style='color:#6b7280;'>⚪ Inactive</span>", unsafe_allow_html=True)
    
    if r_cols[2].button("✅ PASS", key=f"pass_{test['id']}", use_container_width=True):
        st.session_state.checklist_results[test['id']] = "PASS"
    if r_cols[3].button("❌ FAIL", key=f"fail_{test['id']}", use_container_width=True):
        st.session_state.checklist_results[test['id']] = "FAIL"

    res_val = st.session_state.checklist_results.get(test['id'], "PENDING")
    res_color = "#22c55e" if res_val == "PASS" else ("#ef4444" if res_val == "FAIL" else "#9ca3af")
    r_cols[4].markdown(f"<span style='font-weight:700; color:{res_color};'>{res_val}</span>", unsafe_allow_html=True)

# Download checklist button
chk_df = pd.DataFrame([
    {"Test": k, "Result": v} for k, v in st.session_state.checklist_results.items()
])
st.download_button(
    label="📥 Download Checklist Results CSV",
    data=chk_df.to_csv(index=False).encode('utf-8'),
    file_name=f"drivesafe_checklist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
    use_container_width=False
)

# =========================================================
# Event Log Section
# =========================================================
st.markdown("---")
st.subheader("📜 Event Log (logs/events.csv)")
events_table_placeholder = st.empty()

def update_events_table():
    if EVENTS_CSV_PATH.exists():
        try:
            df = pd.read_csv(EVENTS_CSV_PATH)
            if not df.empty:
                events_table_placeholder.dataframe(df.tail(15), use_container_width=True)
            else:
                events_table_placeholder.info("No events logged yet.")
        except Exception:
            pass

update_events_table()

# =========================================================
# Video Processing & Real-time Loop
# =========================================================
if st.session_state.camera_running:
    # Initialize models
    base_options = mp_python.BaseOptions(model_asset_path=str(config.FACE_LANDMARKER_MODEL_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    drowsiness_detector = DrowsinessDetector()
    head_pose_estimator = HeadPoseEstimator()
    phone_detector = PhoneDetector()
    authenticator = DriverAuthenticator(face_landmarker=face_landmarker)
    alert_system = AlertSystem()

    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        st.error(f"Could not open camera at index {config.CAMERA_INDEX}.")
        st.session_state.camera_running = False
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        prev_time = time.time()

        # Check startup auto-calibration
        if auto_calib_startup and not st.session_state.startup_calibrated:
            if config.NEUTRAL_YAW == 0.0 and config.NEUTRAL_PITCH == 0.0:
                st.session_state.calib_mode = "neutral_pose"
                st.session_state.calib_start_time = None
                st.session_state.calib_samples = []

        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 30.0
            prev_time = current_time

            # Synchronize active thresholds from sidebar sliders
            config.EAR_THRESHOLD = ear_thresh_slider
            config.MAR_YAWN_THRESHOLD = mar_thresh_slider
            config.DROWSY_TIME_THRESHOLD_SEC = drowsy_duration_slider
            config.YAW_ATTENTION_THRESHOLD = yaw_angle_slider
            config.YAW_ATTENTION_TIME_SEC = yaw_attention_time_slider
            config.YAW_DANGER_TIME_SEC = yaw_danger_time_slider
            config.PITCH_DOWN_ATTENTION_THRESHOLD = pitch_down_slider
            config.PITCH_DOWN_ATTENTION_TIME_SEC = pitch_down_attention_slider
            config.PITCH_DOWN_DANGER_TIME_SEC = pitch_down_danger_slider
            # Part A/B: face-lost timers
            config.FACE_LOST_TURN_HOLD_SEC = face_lost_hold_slider
            config.FACE_LOST_NO_DRIVER_SEC = face_lost_no_driver_slider
            config.NO_DRIVER_BEEP = no_driver_beep
            # Part C: phone settings
            config.PHONE_CONF_THRESHOLD = phone_conf_slider
            config.PHONE_CONF = phone_conf_slider
            config.PHONE_VOTES_NEEDED = phone_votes_needed_slider
            config.PHONE_VOTING_WINDOW = phone_voting_window_slider
            config.PHONE_HOLD_TIME_SEC = phone_hold_time_slider
            config.PHONE_IMGSZ = phone_imgsz_slider
            config.YOLO_IMGSZ = phone_imgsz_slider

            # 1. MediaPipe Face Detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = face_landmarker.detect(mp_image)

            raw_ear = 0.0
            raw_mar = 0.0
            raw_pitch = 0.0
            raw_yaw = 0.0
            raw_roll = 0.0
            norm_pitch = 0.0

            smoothed_ear = 0.0
            smoothed_mar = 0.0
            smoothed_pitch = 0.0
            smoothed_yaw = 0.0
            smoothed_roll = 0.0

            is_drowsy_warn = False
            is_drowsy_crit = False
            is_yawn = False
            pose_status = "FORWARD"
            pose_severity = config.SEVERITY_NORMAL
            pose_alert = ""
            pose_play_sound = False
            turn_duration = 0.0
            pitch_duration = 0.0

            driver_name = "Scanning..."
            is_auth = False
            current_landmarks = None

            is_no_face = False
            face_state_reason = ""
            h_res = {}

            if detection_result.face_landmarks:
                current_landmarks = detection_result.face_landmarks[0]
                d_res = drowsiness_detector.process(current_landmarks, w, h)
                h_res = head_pose_estimator.estimate(current_landmarks, w, h)
                a_res = authenticator.authenticate(current_landmarks, frame)

                raw_ear = d_res.get("raw_ear", 0.0)
                raw_mar = d_res.get("raw_mar", 0.0)
                smoothed_ear = d_res.get("ear", 0.0)
                smoothed_mar = d_res.get("mar", 0.0)

                raw_pitch = h_res.get("raw_pitch", 0.0)
                raw_yaw = h_res.get("raw_yaw", 0.0)
                raw_roll = h_res.get("raw_roll", 0.0)
                norm_pitch = h_res.get("norm_pitch", 0.0)

                smoothed_pitch = h_res.get("pitch", 0.0)
                smoothed_yaw = h_res.get("yaw", 0.0)
                smoothed_roll = h_res.get("roll", 0.0)

                pose_status = h_res.get("status", "FORWARD")
                pose_severity = h_res.get("severity", config.SEVERITY_NORMAL)
                pose_alert = h_res.get("alert", "")
                pose_play_sound = h_res.get("play_sound", False)
                turn_duration = h_res.get("turn_duration", 0.0)
                pitch_duration = h_res.get("pitch_duration", 0.0)

                is_drowsy_crit = (d_res.get("severity") == config.SEVERITY_CRITICAL)
                is_drowsy_warn = (d_res.get("severity") == config.SEVERITY_WARNING and "DROWSINESS" in d_res.get("alert", ""))
                is_yawn = (d_res.get("status") == "YAWNING")

                driver_name = a_res.get("driver_name", "Unknown")
                is_auth = a_res.get("is_authorized", False)

                # Registration action from sidebar
                if register_requested and current_landmarks:
                    authenticator.register_face(reg_name_input, current_landmarks, frame)
                    st.sidebar.success(f"Driver '{reg_name_input}' registered!")
                    register_requested = False

                # Draw head pose projection line
                if h_res.get("nose_proj"):
                    p1, p2 = h_res["nose_proj"]
                    cv2.arrowedLine(frame, p1, p2, (0, 255, 255), 2, tipLength=0.25)

                # Draw facial landmarks points
                for idx in config.LEFT_EYE_LANDMARKS + config.RIGHT_EYE_LANDMARKS:
                    pt = (int(current_landmarks[idx].x * w), int(current_landmarks[idx].y * h))
                    cv2.circle(frame, pt, 2, (0, 255, 0), -1)
                for key_idx in config.MOUTH_LANDMARKS.values():
                    pt = (int(current_landmarks[key_idx].x * w), int(current_landmarks[key_idx].y * h))
                    cv2.circle(frame, pt, 2, (0, 200, 255), -1)

            else:
                # Part B: NO FACE state machine — call estimator with None to get correct status
                h_res = head_pose_estimator.estimate(None, w, h)
                pose_status = h_res.get("status", "FORWARD")
                pose_severity = h_res.get("severity", config.SEVERITY_NORMAL)
                pose_alert = h_res.get("alert", "")
                pose_play_sound = h_res.get("play_sound", False)
                turn_duration = h_res.get("turn_duration", 0.0)
                pitch_duration = h_res.get("pitch_duration", 0.0)
                smoothed_yaw = h_res.get("yaw", 0.0)
                smoothed_pitch = h_res.get("pitch", 0.0)
                smoothed_roll = h_res.get("roll", 0.0)
                is_no_face = True
                face_state_reason = h_res.get("state_change_reason", "")

                # Part B: clear drowsiness/yawn timers when no face — don't run on stale values
                drowsiness_detector.is_eye_closed = False
                drowsiness_detector.closed_start_time = None
                drowsiness_detector.closed_eye_frames = 0
                drowsiness_detector.is_yawning = False
                drowsiness_detector.yawn_frames = 0
                is_drowsy_crit = False
                is_drowsy_warn = False
                is_yawn = False

            # 2. YOLO Phone Detection (with voting window + hold time)
            phone_res = phone_detector.detect(frame)
            is_phone = phone_res.get("phone_in_use", False)

            for det in phone_res.get("detections", []):
                bx1, by1, bx2, by2 = det["bbox"]
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                cv2.putText(frame, f"PHONE {det['confidence']*100:.0f}%", (bx1, max(20, by1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # =========================================================
            # Calibration Routine State Machine
            # =========================================================
            is_calibrating = False
            calib_banner_text = ""

            if st.session_state.calib_mode == "neutral_pose":
                is_calibrating = True
                if st.session_state.calib_start_time is None:
                    st.session_state.calib_start_time = current_time
                    st.session_state.calib_samples = []

                elapsed = current_time - st.session_state.calib_start_time
                remaining = max(0.0, 3.0 - elapsed)
                calib_banner_text = f"🎯 CALIBRATING NEUTRAL POSE: Keep looking straight ahead! ({remaining:.1f}s remaining)"

                if current_landmarks:
                    st.session_state.calib_samples.append((norm_pitch, raw_yaw, raw_roll))

                if elapsed >= 3.0:
                    if st.session_state.calib_samples:
                        med_pitch = float(np.median([s[0] for s in st.session_state.calib_samples]))
                        med_yaw = float(np.median([s[1] for s in st.session_state.calib_samples]))
                        med_roll = float(np.median([s[2] for s in st.session_state.calib_samples]))
                        config.save_calibration({
                            "neutral_yaw": round(med_yaw, 2),
                            "neutral_pitch": round(med_pitch, 2),
                            "neutral_roll": round(med_roll, 2)
                        })
                        head_pose_estimator.update_calibration(med_yaw, med_pitch, med_roll)
                        st.sidebar.success(f"Neutral calibrated! Yaw: {med_yaw:.1f}°, Pitch: {med_pitch:.1f}°, Roll: {med_roll:.1f}°")
                    st.session_state.calib_mode = None
                    st.session_state.startup_calibrated = True

            elif st.session_state.calib_mode == "ear_open":
                is_calibrating = True
                if st.session_state.calib_start_time is None:
                    st.session_state.calib_start_time = current_time
                    st.session_state.calib_open_ear_samples = []
                    st.session_state.calib_resting_mar_samples = []

                elapsed = current_time - st.session_state.calib_start_time
                remaining = max(0.0, 5.0 - elapsed)
                calib_banner_text = f"👁️ STEP 1/2: Keep eyes OPEN normally and mouth relaxed! ({remaining:.1f}s)"

                if current_landmarks:
                    st.session_state.calib_open_ear_samples.append(raw_ear)
                    st.session_state.calib_resting_mar_samples.append(raw_mar)

                if elapsed >= 5.0:
                    st.session_state.calib_mode = "ear_closed"
                    st.session_state.calib_start_time = current_time
                    st.session_state.calib_closed_ear_samples = []

            elif st.session_state.calib_mode == "ear_closed":
                is_calibrating = True
                elapsed = current_time - st.session_state.calib_start_time
                remaining = max(0.0, 5.0 - elapsed)
                calib_banner_text = f"😴 STEP 2/2: Keep eyes CLOSED naturally! ({remaining:.1f}s)"

                if current_landmarks:
                    st.session_state.calib_closed_ear_samples.append(raw_ear)

                if elapsed >= 5.0:
                    if st.session_state.calib_open_ear_samples and st.session_state.calib_closed_ear_samples:
                        open_ear = float(np.median(st.session_state.calib_open_ear_samples))
                        closed_ear = float(np.median(st.session_state.calib_closed_ear_samples))
                        resting_mar = float(np.median(st.session_state.calib_resting_mar_samples)) if st.session_state.calib_resting_mar_samples else 0.25
                        sugg_ear = round(float((open_ear + closed_ear) / 2.0), 3)
                        sugg_mar = round(max(0.48, float(resting_mar * 1.8)), 3)
                        config.save_calibration({
                            "EAR_THRESHOLD": sugg_ear,
                            "MAR_YAWN_THRESHOLD": sugg_mar
                        })
                        st.sidebar.success(f"EAR/MAR Calibrated! Open={open_ear:.3f}, Closed={closed_ear:.3f} -> Sugg EAR={sugg_ear:.3f}, MAR={sugg_mar:.3f}")
                    st.session_state.calib_mode = None

            # =========================================================
            # Requirement 6: Overall Status Banner
            # Overall banner is SAFE unless a drowsy, yawn, sustained looking-away or phone event is active.
            # Requirement 5: Unknown driver does NOT alter overall banner!
            # =========================================================
            is_pose_danger = (pose_severity == config.SEVERITY_CRITICAL)
            is_pose_attention = (pose_severity == config.SEVERITY_WARNING)

            if is_calibrating:
                banner_placeholder.markdown(f"<div class='banner-calib'>{calib_banner_text}</div>", unsafe_allow_html=True)
            elif is_drowsy_crit or is_phone or is_pose_danger:
                banner_placeholder.markdown("<div class='banner-danger'>🚨 DANGER: IMMEDIATE DRIVER ATTENTION REQUIRED!</div>", unsafe_allow_html=True)
            elif is_drowsy_warn or is_yawn or is_pose_attention:
                banner_placeholder.markdown("<div class='banner-warning'>⚠️ WARNING / ATTENTION: DRIVER DISTRACTION DETECTED</div>", unsafe_allow_html=True)
            else:
                banner_placeholder.markdown("<div class='banner-safe'>✅ SAFE: DRIVER ATTENTIVE & ROAD FOCUSED</div>", unsafe_allow_html=True)

            # Audio alert management
            if is_no_driver:
                if no_driver_beep:
                    alert_system.trigger_alert("NO DRIVER DETECTED", config.SEVERITY_WARNING, play_sound=True)
            elif is_pose_danger and pose_play_sound:
                alert_system.trigger_alert(pose_alert, config.SEVERITY_CRITICAL, play_sound=True)
            elif is_drowsy_crit:
                alert_system.trigger_alert("CRITICAL: WAKE UP! EYES CLOSED!", config.SEVERITY_CRITICAL, play_sound=True)
            elif is_phone:
                alert_system.trigger_alert("CRITICAL: PHONE IN HAND DETECTED!", config.SEVERITY_CRITICAL, play_sound=True)
            elif is_drowsy_warn:
                alert_system.trigger_alert("WARNING: DROWSINESS DETECTED!", config.SEVERITY_WARNING, play_sound=True)
            elif is_yawn:
                alert_system.trigger_alert("WARNING: YAWNING DETECTED!", config.SEVERITY_WARNING, play_sound=False)
            elif is_pose_attention:
                alert_system.trigger_alert(pose_alert, config.SEVERITY_WARNING, play_sound=False)

            # =========================================================
            # Render Indicator Cards
            # =========================================================
            # 1. Drowsy Card
            if is_drowsy_crit or is_drowsy_warn:
                card_drowsy.markdown(f"<div class='indicator-card indicator-active-red'><b>😴 DROWSY</b><br><small>{'MICRO-SLEEP (EYES CLOSED > 3s)' if is_drowsy_crit else 'EYES CLOSED > 1.5s — WAKE UP!'}</small></div>", unsafe_allow_html=True)
            else:
                card_drowsy.markdown("<div class='indicator-card indicator-inactive'><b>😴 DROWSY</b><br><small>Eyes open & attentive</small></div>", unsafe_allow_html=True)

            # 2. Yawn Card
            if is_yawn:
                card_yawn.markdown("<div class='indicator-card indicator-active-amber'><b>🥱 YAWNING</b><br><small>Wide mouth fatigue detected</small></div>", unsafe_allow_html=True)
            else:
                card_yawn.markdown("<div class='indicator-card indicator-inactive'><b>🥱 YAWNING</b><br><small>Mouth posture normal</small></div>", unsafe_allow_html=True)

            # 3. Looking Away / Face Card
            if is_no_driver:
                card_pose.markdown("<div class='indicator-card indicator-inactive'><b>\U0001f464 NO DRIVER</b><br><small>Seat empty or face absent &gt; 3s</small></div>", unsafe_allow_html=True)
            elif is_face_not_visible:
                card_pose.markdown("<div class='indicator-card indicator-active-amber'><b>\U0001f441\ufe0f FACE NOT VISIBLE</b><br><small>Face hidden 1.5-3s</small></div>", unsafe_allow_html=True)
            elif is_pose_danger:
                card_pose.markdown(f"<div class='indicator-card indicator-active-red'><b>\U0001f440 LOOKING AWAY</b><br><small>{pose_status} ({turn_duration:.1f}s DANGER!)</small></div>", unsafe_allow_html=True)
            elif is_pose_attention:
                card_pose.markdown(f"<div class='indicator-card indicator-active-amber'><b>\U0001f440 ATTENTION</b><br><small>{pose_status} ({turn_duration if turn_duration>0 else pitch_duration:.1f}s)</small></div>", unsafe_allow_html=True)
            else:
                if "TURNING" in pose_status or "BRIEF" in pose_status:
                    card_pose.markdown(f"<div class='indicator-card indicator-inactive'><b>\U0001f440 HEAD POSE</b><br><small>{pose_status} (Normal &lt;2s)</small></div>", unsafe_allow_html=True)
                else:
                    card_pose.markdown("<div class='indicator-card indicator-inactive'><b>\U0001f440 LOOKING AWAY</b><br><small>Forward road attention</small></div>", unsafe_allow_html=True)

            # 4. Phone Card (with vote count)
            vote_str = phone_res.get("vote_str", "0/6")
            if is_phone:
                card_phone.markdown(f"<div class='indicator-card indicator-active-red'><b>\U0001f4f1 PHONE IN HAND</b><br><small>Votes {vote_str} \u2014 ALERT!</small></div>", unsafe_allow_html=True)
            else:
                card_phone.markdown(f"<div class='indicator-card indicator-inactive'><b>\U0001f4f1 PHONE IN HAND</b><br><small>Hands on wheel (votes {vote_str})</small></div>", unsafe_allow_html=True)

            # 5. Requirement 5: Driver Identity as a Separate Badge
            if is_auth:
                card_driver.markdown(f"<div class='indicator-card indicator-active-green'><b>👤 DRIVER: {driver_name}</b><br><small>Authorized driver profile</small></div>", unsafe_allow_html=True)
            else:
                card_driver.markdown("<div class='indicator-card indicator-driver-unknown'><b>👤 DRIVER: Unknown</b><br><small>Unregistered face (identity badge only)</small></div>", unsafe_allow_html=True)

            # =========================================================
            # Live Metrics Row
            # =========================================================
            metrics_row.markdown(f"""
            <div style='display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px;'>
                <div class='metric-badge'><div class='metric-label'>EAR</div><div class='metric-value'>{smoothed_ear:.2f}</div></div>
                <div class='metric-badge'><div class='metric-label'>MAR</div><div class='metric-value'>{smoothed_mar:.2f}</div></div>
                <div class='metric-badge'><div class='metric-label'>Yaw</div><div class='metric-value'>{smoothed_yaw:.0f}°</div></div>
                <div class='metric-badge'><div class='metric-label'>Pitch</div><div class='metric-value'>{smoothed_pitch:.0f}°</div></div>
                <div class='metric-badge'><div class='metric-label'>Roll</div><div class='metric-value'>{smoothed_roll:.0f}°</div></div>
                <div class='metric-badge'><div class='metric-label'>FPS</div><div class='metric-value' style='color: {"#4ade80" if fps>=15 else "#f87171"};'>{fps:.1f}</div></div>
            </div>
            """, unsafe_allow_html=True)

            # =========================================================
            # Requirement 1: Live Debug Expander Telemetry
            # =========================================================
            # Detect which detectors are out of range
            out_of_range = []
            if smoothed_ear < config.EAR_THRESHOLD:
                out_of_range.append(f"EAR ({smoothed_ear:.3f} < {config.EAR_THRESHOLD:.2f}) [EYES CLOSED]")
            if smoothed_mar > config.MAR_YAWN_THRESHOLD:
                out_of_range.append(f"MAR ({smoothed_mar:.3f} > {config.MAR_YAWN_THRESHOLD:.2f}) [YAWNING]")
            if abs(smoothed_yaw) > config.YAW_ATTENTION_THRESHOLD:
                out_of_range.append(f"Yaw ({smoothed_yaw:.1f}° > {config.YAW_ATTENTION_THRESHOLD:.0f}°) [HEAD TURN, {turn_duration:.1f}s]")
            if smoothed_pitch < config.PITCH_DOWN_ATTENTION_THRESHOLD:
                out_of_range.append(f"Pitch ({smoothed_pitch:.1f}° < {config.PITCH_DOWN_ATTENTION_THRESHOLD:.0f}°) [LOOKING DOWN, {pitch_duration:.1f}s]")
            if is_phone:
                out_of_range.append("Phone Detected in Hand (>=5 frames)")

            debug_table_html = f"""
            <table style='width:100%; font-size:0.85rem; border-collapse:collapse; color:#cbd5e1;'>
                <thead>
                    <tr style='border-bottom: 1px solid #374151; color:#9ca3af;'>
                        <th style='text-align:left; padding:4px;'>Signal</th>
                        <th style='text-align:right; padding:4px;'>Raw Value</th>
                        <th style='text-align:right; padding:4px;'>Normalized / Calibrated</th>
                        <th style='text-align:right; padding:4px;'>Active Threshold</th>
                        <th style='text-align:right; padding:4px;'>Active Flag</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style='padding:4px;'><b>EAR</b></td>
                        <td style='text-align:right; padding:4px;'>{raw_ear:.3f}</td>
                        <td style='text-align:right; padding:4px;'>{smoothed_ear:.3f}</td>
                        <td style='text-align:right; padding:4px;'>&lt; {config.EAR_THRESHOLD:.2f}</td>
                        <td style='text-align:right; padding:4px; color:{"#ef4444" if (is_drowsy_crit or is_drowsy_warn) else "#22c55e"};'>{'CLOSED' if (is_drowsy_crit or is_drowsy_warn) else 'OPEN'}</td>
                    </tr>
                    <tr>
                        <td style='padding:4px;'><b>MAR</b></td>
                        <td style='text-align:right; padding:4px;'>{raw_mar:.3f}</td>
                        <td style='text-align:right; padding:4px;'>{smoothed_mar:.3f}</td>
                        <td style='text-align:right; padding:4px;'>&gt; {config.MAR_YAWN_THRESHOLD:.2f}</td>
                        <td style='text-align:right; padding:4px; color:{"#f59e0b" if is_yawn else "#22c55e"};'>{'YAWN' if is_yawn else 'NORMAL'}</td>
                    </tr>
                    <tr>
                        <td style='padding:4px;'><b>Yaw</b></td>
                        <td style='text-align:right; padding:4px;'>{raw_yaw:.1f}°</td>
                        <td style='text-align:right; padding:4px;'>{smoothed_yaw:.1f}° (zero={config.NEUTRAL_YAW:.1f}°)</td>
                        <td style='text-align:right; padding:4px;'>&gt; {config.YAW_ATTENTION_THRESHOLD:.0f}°</td>
                        <td style='text-align:right; padding:4px; color:{"#ef4444" if is_pose_danger else ("#f59e0b" if is_pose_attention else "#22c55e")};'>{pose_status}</td>
                    </tr>
                    <tr>
                        <td style='padding:4px;'><b>Pitch</b></td>
                        <td style='text-align:right; padding:4px;'>{raw_pitch:.1f}° (norm={norm_pitch:.1f}°)</td>
                        <td style='text-align:right; padding:4px;'>{smoothed_pitch:.1f}° (zero={config.NEUTRAL_PITCH:.1f}°)</td>
                        <td style='text-align:right; padding:4px;'>&lt; {config.PITCH_DOWN_ATTENTION_THRESHOLD:.0f}°</td>
                        <td style='text-align:right; padding:4px; color:{"#ef4444" if "HEAD DOWN" in pose_status else "#22c55e"};'>{'DOWN' if "HEAD DOWN" in pose_status else 'NORMAL'}</td>
                    </tr>
                    <tr>
                        <td style='padding:4px;'><b>Roll</b></td>
                        <td style='text-align:right; padding:4px;'>{raw_roll:.1f}°</td>
                        <td style='text-align:right; padding:4px;'>{smoothed_roll:.1f}° (zero={config.NEUTRAL_ROLL:.1f}°)</td>
                        <td style='text-align:right; padding:4px;'>-</td>
                        <td style='text-align:right; padding:4px; color:#22c55e;'>LEVEL</td>
                    </tr>
                </tbody>
            </table>
            """
            debug_metrics_placeholder.markdown(debug_table_html, unsafe_allow_html=True)

            status_summary = (
                f"<span style='color:#ef4444; font-weight:700;'>⚠️ Out of Range:</span> {', '.join(out_of_range)}"
                if out_of_range else
                "<span style='color:#22c55e; font-weight:700;'>✅ All detector values within normal range. Driver attentive & looking straight.</span>"
            )
            debug_status_placeholder.markdown(
                f"<div style='margin-top:8px; font-size:0.85rem;'><b>Active Detector Assessment:</b> {status_summary}</div>",
                unsafe_allow_html=True
            )

            # =========================================================
            # Rolling 10-Second Telemetry Chart
            # =========================================================
            st.session_state.rolling_history.append({
                "time": current_time,
                "EAR": round(smoothed_ear, 3),
                "MAR": round(smoothed_mar, 3),
                "EAR Threshold": round(ear_thresh_slider, 3),
                "MAR Threshold": round(mar_thresh_slider, 3)
            })
            st.session_state.rolling_history = [
                pt for pt in st.session_state.rolling_history if current_time - pt["time"] <= 10.0
            ]
            if len(st.session_state.rolling_history) > 2:
                chart_df = pd.DataFrame(st.session_state.rolling_history).set_index("time")
                chart_placeholder.line_chart(
                    chart_df[["EAR", "MAR", "EAR Threshold", "MAR Threshold"]],
                    height=180,
                    use_container_width=True
                )

            # =========================================================
            # Live Checklist "Detected Now" Indicators
            # =========================================================
            live_test_badges["drowsy"].markdown(
                "<span style='color:#ef4444; font-weight:700;'>🔴 DETECTED NOW</span>" if (is_drowsy_crit or is_drowsy_warn) else "<span style='color:#9ca3af;'>⚪ Inactive</span>",
                unsafe_allow_html=True
            )
            live_test_badges["blink_safe"].markdown(
                "<span style='color:#22c55e; font-weight:700;'>🟢 SAFE (No Alert)</span>" if not (is_drowsy_crit or is_drowsy_warn) else "<span style='color:#ef4444;'>🔴 Alerting</span>",
                unsafe_allow_html=True
            )
            live_test_badges["yawn"].markdown(
                "<span style='color:#f59e0b; font-weight:700;'>🟡 DETECTED NOW</span>" if is_yawn else "<span style='color:#9ca3af;'>⚪ Inactive</span>",
                unsafe_allow_html=True
            )
            live_test_badges["head_turn"].markdown(
                "<span style='color:#ef4444; font-weight:700;'>🔴 DETECTED NOW</span>" if (is_pose_danger or is_pose_attention) else "<span style='color:#9ca3af;'>⚪ Inactive</span>",
                unsafe_allow_html=True
            )
            live_test_badges["phone"].markdown(
                "<span style='color:#ef4444; font-weight:700;'>🔴 DETECTED NOW</span>" if is_phone else "<span style='color:#9ca3af;'>⚪ Inactive</span>",
                unsafe_allow_html=True
            )
            live_test_badges["unknown_driver"].markdown(
                "<span style='color:#f59e0b; font-weight:700;'>🟡 UNKNOWN FACE</span>" if not is_auth else "<span style='color:#22c55e;'>🟢 Authorized</span>",
                unsafe_allow_html=True
            )

            # =========================================================
            # Event Logging (one row per event start)
            # =========================================================
            current_active_events = set()
            if is_drowsy_crit or is_drowsy_warn:
                current_active_events.add("DROWSINESS_STARTED")
            if is_yawn:
                current_active_events.add("YAWN_STARTED")
            if is_pose_danger or is_pose_attention:
                current_active_events.add(f"HEAD_DISTRACTION_{pose_status.replace(' ', '_')}")
            if is_phone:
                current_active_events.add("PHONE_USAGE_STARTED")

            new_events = current_active_events - st.session_state.active_events
            for ev in new_events:
                log_event_start(ev, driver_name, f"EAR={smoothed_ear:.2f}, MAR={smoothed_mar:.2f}, Yaw={smoothed_yaw:.0f}°, Pitch={smoothed_pitch:.0f}°")
                update_events_table()

            st.session_state.active_events = current_active_events

            # Display video frame
            disp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(disp_frame, channels="RGB", use_container_width=True)

        cap.release()
else:
    # Idle / Camera stopped view
    banner_placeholder.markdown("<div class='banner-safe'>⏸️ SYSTEM PAUSED — CLICK 'START CAMERA' TO BEGIN</div>", unsafe_allow_html=True)
    card_drowsy.markdown("<div class='indicator-card indicator-inactive'><b>😴 DROWSY</b><br><small>Camera offline</small></div>", unsafe_allow_html=True)
    card_yawn.markdown("<div class='indicator-card indicator-inactive'><b>🥱 YAWNING</b><br><small>Camera offline</small></div>", unsafe_allow_html=True)
    card_pose.markdown("<div class='indicator-card indicator-inactive'><b>👀 LOOKING AWAY</b><br><small>Camera offline</small></div>", unsafe_allow_html=True)
    card_phone.markdown("<div class='indicator-card indicator-inactive'><b>📱 PHONE IN HAND</b><br><small>Camera offline</small></div>", unsafe_allow_html=True)
    card_driver.markdown("<div class='indicator-card indicator-inactive'><b>👤 DRIVER</b><br><small>Camera offline</small></div>", unsafe_allow_html=True)
    video_placeholder.info("Click **'▶️ Start Camera'** above to activate the live feed and neural detectors.")
