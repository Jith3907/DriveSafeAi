"""
Central configuration settings and detection thresholds for DriveSafe AI.
"""

from pathlib import Path

# ==========================================
# Paths & Directories
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
KNOWN_DRIVERS_DIR = DATA_DIR / "known_drivers"
ASSETS_DIR = PROJECT_ROOT / "assets"
ALERT_SOUND_PATH = ASSETS_DIR / "alert.wav"
FACE_LANDMARKER_MODEL_PATH = ASSETS_DIR / "face_landmarker.task"

# Create directories if they do not exist
KNOWN_DRIVERS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# Camera & Video Feed Settings
# ==========================================
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 30

# ==========================================
# Drowsiness Detection Thresholds
# ==========================================
# Eye Aspect Ratio (EAR)
EAR_THRESHOLD = 0.22                   # Eyes below this ratio are considered closed
DROWSY_TIME_THRESHOLD_SEC = 1.5       # Eyes closed for 1.5s triggers drowsy alert
MICROSLEEP_TIME_THRESHOLD_SEC = 3.0   # Prolonged closure triggers micro-sleep alert
EAR_CONSEC_FRAMES = 18                 # Fallback frame threshold if time not used

# Mouth Aspect Ratio (MAR) / Yawn Detection
MAR_YAWN_THRESHOLD = 0.52              # Mouth opening aspect ratio threshold for yawn
MAR_CONSEC_FRAMES = 12                 # Consecutive frames indicating a distinct yawn (~0.6-0.8s)

# Eye Landmark Indices (MediaPipe Face Mesh 468/478 landmarks)
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

# Mouth landmarks for vertical and horizontal distance:
# upper lip inner (13), lower lip inner (14), left corner (78), right corner (308)
MOUTH_LANDMARKS = {
    "upper_lip": 13,
    "lower_lip": 14,
    "left_corner": 78,
    "right_corner": 308
}

# Calibration File Path
CALIBRATION_FILE_PATH = DATA_DIR / "calibration.json"

# ==========================================
# Smoothing & Hysteresis Settings
# ==========================================
SMOOTHING_WINDOW_SIZE = 5               # Moving average over last 5 frames
HYSTERESIS_RATIO = 0.9                  # Trigger at threshold, clear at 90% of threshold (EAR/MAR)
HEAD_HYSTERESIS_RATIO = 0.8             # Clear head turn at 80% of trigger threshold

# Head pose stabilisation (angles come from MediaPipe's transformation matrix)
HEAD_MAX_JUMP_DEG = 25.0                # Ignore single-frame jumps larger than this (degrees)
HEAD_JUMP_CONFIRM_FRAMES = 3            # ... unless the jump repeats on this many consecutive frames
HEAD_MEDIAN_WINDOW = 5                  # Median over the last N accepted angles
HEAD_SMOOTHING_ALPHA = 0.3              # Exponential smoothing applied after the median

# ==========================================
# Neutral Head Pose Offsets (Calibrated)
# ==========================================
NEUTRAL_YAW = 0.0
NEUTRAL_PITCH = 0.0
NEUTRAL_ROLL = 0.0

# ==========================================
# Head Pose & Distraction Thresholds (Normal Driving Tolerances)
# ==========================================
# Head turn (yaw) thresholds. yaw is negative looking left, positive looking right.
YAW_ATTENTION_THRESHOLD = 28.0          # |yaw| > 28° considered head turn
YAW_ATTENTION_TIME_SEC = 2.0           # 2.0s sustained -> ATTENTION (yellow, no sound)
YAW_DANGER_TIME_SEC = 4.0              # 4.0s sustained -> DANGER (red, sound)

# Looking down (pitch down) thresholds. pitch is negative looking down.
PITCH_DOWN_ATTENTION_THRESHOLD = -18.0  # pitch < -18° considered looking down
PITCH_DOWN_ATTENTION_TIME_SEC = 2.5    # 2.5s sustained -> ATTENTION (yellow, no sound)
PITCH_DOWN_DANGER_TIME_SEC = 4.0       # 4.0s sustained -> DANGER (red, sound)

# Looking up threshold. pitch is positive looking up.
PITCH_UP_THRESHOLD = 22.0               # pitch > 22° considered looking up

# Face-Lost & No-Driver Timers
FACE_LOST_TURN_HOLD_SEC = 1.5           # If face lost within 1.5s of sharp turn, hold looking-away state
FACE_LOST_NO_DRIVER_SEC = 3.0           # If face missing > 3.0s -> NO DRIVER DETECTED
NO_DRIVER_BEEP = False                  # Sidebar option: soft beep when no driver detected

# Backward compatibility aliases
HEAD_POSE_ANGLE_THRESHOLD = 28.0
HEAD_DISTRACTION_TIME_SEC = 2.0
PITCH_DOWN_THRESHOLD = -18.0
YAW_LEFT_THRESHOLD = -28.0
YAW_RIGHT_THRESHOLD = 28.0
ROLL_TILT_THRESHOLD = 25.0
HEAD_DISTRACTION_CONSEC_FRAMES = 25

# Key 3D facial landmarks for SolvePnP head pose estimation:
# Nose tip, Chin, Left eye corner, Right eye corner, Left mouth corner, Right mouth corner
POSE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

# ==========================================
# Phone Usage & Object Detection (Ultralytics YOLO)
# ==========================================
YOLO_MODEL_NAME = "yolov8n.pt"         # Lightweight pretrained COCO model
PHONE_CLASS_ID = 67                    # 'cell phone' in COCO dataset
PHONE_CONF_THRESHOLD = 0.28            # Lowered confidence threshold from 0.4 to 0.28
PHONE_CONF = 0.28                      # Alias for PHONE_CONF_THRESHOLD
PHONE_CONSEC_FRAMES = 5                # Backward compatibility
PHONE_IMGSZ = 480                      # Raised YOLO imgsz from 320 to 480
YOLO_IMGSZ = 480                       # Alias
YOLO_FRAME_INTERVAL = 2                # Run YOLO every 2nd frame
PHONE_VOTING_WINDOW = 6                # Keep last 6 YOLO results
PHONE_VOTES_NEEDED = 3                 # True when >= 3 of last 6 are True
PHONE_HOLD_TIME_SEC = 1.5              # Keep alert True for 1.5s after last positive detection
PHONE_MIN_AREA_RATIO = 0.01            # Ignore detections smaller than 1% of frame area

# ==========================================
# Driver Authentication Settings
# ==========================================
FACE_AUTH_SIMILARITY_THRESHOLD = 0.94  # Cosine similarity for face matching
UNKNOWN_DRIVER_TIMEOUT_SEC = 3.0       # Alert if unknown driver persists

# ==========================================
# Alert System Settings
# ==========================================
AUDIO_ALERT_ENABLED = True
AUDIO_COOLDOWN_SECONDS = 2.5      # Minimum seconds between repeated sound alarms

# Alert Severities
SEVERITY_NORMAL = "NORMAL"
SEVERITY_WARNING = "WARNING"
SEVERITY_CRITICAL = "CRITICAL"

ALERT_COLORS = {
    SEVERITY_NORMAL: (50, 205, 50),     # Lime Green (BGR)
    SEVERITY_WARNING: (0, 215, 255),    # Amber/Yellow (BGR)
    SEVERITY_CRITICAL: (50, 50, 240),   # Vivid Red (BGR)
}

# ==========================================
# Calibration File Persistence
# ==========================================
import json

def load_calibration():
    """Loads calibration settings from data/calibration.json if it exists."""
    global NEUTRAL_YAW, NEUTRAL_PITCH, NEUTRAL_ROLL
    global EAR_THRESHOLD, MAR_YAWN_THRESHOLD, DROWSY_TIME_THRESHOLD_SEC
    global YAW_ATTENTION_THRESHOLD, YAW_ATTENTION_TIME_SEC, YAW_DANGER_TIME_SEC
    global PITCH_DOWN_ATTENTION_THRESHOLD, PITCH_DOWN_ATTENTION_TIME_SEC, PITCH_DOWN_DANGER_TIME_SEC
    global PITCH_UP_THRESHOLD
    global FACE_LOST_TURN_HOLD_SEC, FACE_LOST_NO_DRIVER_SEC, NO_DRIVER_BEEP
    global PHONE_CONF_THRESHOLD, PHONE_CONF, PHONE_VOTING_WINDOW, PHONE_VOTES_NEEDED, PHONE_HOLD_TIME_SEC, PHONE_IMGSZ, YOLO_IMGSZ

    data = {
        "neutral_yaw": NEUTRAL_YAW,
        "neutral_pitch": NEUTRAL_PITCH,
        "neutral_roll": NEUTRAL_ROLL,
        "EAR_THRESHOLD": EAR_THRESHOLD,
        "MAR_YAWN_THRESHOLD": MAR_YAWN_THRESHOLD,
        "DROWSY_TIME_THRESHOLD_SEC": DROWSY_TIME_THRESHOLD_SEC,
        "YAW_ATTENTION_THRESHOLD": YAW_ATTENTION_THRESHOLD,
        "YAW_ATTENTION_TIME_SEC": YAW_ATTENTION_TIME_SEC,
        "YAW_DANGER_TIME_SEC": YAW_DANGER_TIME_SEC,
        "PITCH_DOWN_ATTENTION_THRESHOLD": PITCH_DOWN_ATTENTION_THRESHOLD,
        "PITCH_DOWN_ATTENTION_TIME_SEC": PITCH_DOWN_ATTENTION_TIME_SEC,
        "PITCH_DOWN_DANGER_TIME_SEC": PITCH_DOWN_DANGER_TIME_SEC,
        "PITCH_UP_THRESHOLD": PITCH_UP_THRESHOLD,
        "FACE_LOST_TURN_HOLD_SEC": FACE_LOST_TURN_HOLD_SEC,
        "FACE_LOST_NO_DRIVER_SEC": FACE_LOST_NO_DRIVER_SEC,
        "NO_DRIVER_BEEP": NO_DRIVER_BEEP,
        "PHONE_CONF": PHONE_CONF_THRESHOLD,
        "PHONE_CONF_THRESHOLD": PHONE_CONF_THRESHOLD,
        "PHONE_VOTING_WINDOW": PHONE_VOTING_WINDOW,
        "PHONE_VOTES_NEEDED": PHONE_VOTES_NEEDED,
        "PHONE_HOLD_TIME_SEC": PHONE_HOLD_TIME_SEC,
        "PHONE_IMGSZ": PHONE_IMGSZ,
    }
    if CALIBRATION_FILE_PATH.exists():
        try:
            with open(CALIBRATION_FILE_PATH, "r") as f:
                saved = json.load(f)
                data.update(saved)
                NEUTRAL_YAW = float(data.get("neutral_yaw", NEUTRAL_YAW))
                NEUTRAL_PITCH = float(data.get("neutral_pitch", NEUTRAL_PITCH))
                NEUTRAL_ROLL = float(data.get("neutral_roll", NEUTRAL_ROLL))
                EAR_THRESHOLD = float(data.get("EAR_THRESHOLD", EAR_THRESHOLD))
                MAR_YAWN_THRESHOLD = float(data.get("MAR_YAWN_THRESHOLD", MAR_YAWN_THRESHOLD))
                DROWSY_TIME_THRESHOLD_SEC = float(data.get("DROWSY_TIME_THRESHOLD_SEC", DROWSY_TIME_THRESHOLD_SEC))
                YAW_ATTENTION_THRESHOLD = float(data.get("YAW_ATTENTION_THRESHOLD", YAW_ATTENTION_THRESHOLD))
                YAW_ATTENTION_TIME_SEC = float(data.get("YAW_ATTENTION_TIME_SEC", YAW_ATTENTION_TIME_SEC))
                YAW_DANGER_TIME_SEC = float(data.get("YAW_DANGER_TIME_SEC", YAW_DANGER_TIME_SEC))
                PITCH_DOWN_ATTENTION_THRESHOLD = float(data.get("PITCH_DOWN_ATTENTION_THRESHOLD", PITCH_DOWN_ATTENTION_THRESHOLD))
                PITCH_DOWN_ATTENTION_TIME_SEC = float(data.get("PITCH_DOWN_ATTENTION_TIME_SEC", PITCH_DOWN_ATTENTION_TIME_SEC))
                PITCH_DOWN_DANGER_TIME_SEC = float(data.get("PITCH_DOWN_DANGER_TIME_SEC", PITCH_DOWN_DANGER_TIME_SEC))
                PITCH_UP_THRESHOLD = float(data.get("PITCH_UP_THRESHOLD", PITCH_UP_THRESHOLD))
                FACE_LOST_TURN_HOLD_SEC = float(data.get("FACE_LOST_TURN_HOLD_SEC", FACE_LOST_TURN_HOLD_SEC))
                FACE_LOST_NO_DRIVER_SEC = float(data.get("FACE_LOST_NO_DRIVER_SEC", FACE_LOST_NO_DRIVER_SEC))
                NO_DRIVER_BEEP = bool(data.get("NO_DRIVER_BEEP", NO_DRIVER_BEEP))
                PHONE_CONF_THRESHOLD = float(data.get("PHONE_CONF", data.get("PHONE_CONF_THRESHOLD", PHONE_CONF_THRESHOLD)))
                PHONE_CONF = PHONE_CONF_THRESHOLD
                PHONE_VOTING_WINDOW = int(data.get("PHONE_VOTING_WINDOW", PHONE_VOTING_WINDOW))
                PHONE_VOTES_NEEDED = int(data.get("PHONE_VOTES_NEEDED", PHONE_VOTES_NEEDED))
                PHONE_HOLD_TIME_SEC = float(data.get("PHONE_HOLD_TIME_SEC", PHONE_HOLD_TIME_SEC))
                PHONE_IMGSZ = int(data.get("PHONE_IMGSZ", PHONE_IMGSZ))
                YOLO_IMGSZ = PHONE_IMGSZ
        except Exception as e:
            print(f"[Config] Error loading calibration file: {e}")
    return data

def save_calibration(updates: dict):
    """Saves updated calibration dictionary to data/calibration.json."""
    current = load_calibration()
    current.update(updates)
    CALIBRATION_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_FILE_PATH, "w") as f:
        json.dump(current, f, indent=4)
    # Refresh in-memory config values
    load_calibration()
    return current

# Load existing calibration on module import
load_calibration()

