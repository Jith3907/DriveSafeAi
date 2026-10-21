# DriveSafe AI 🚗👁️

A real-time driver monitoring system powered by lightweight, pretrained computer
vision models. It detects drowsiness, yawning, distraction, phone usage and driver
identity, and now ships as a multi-page, website-style Streamlit app.

---

## 🚀 Launch

```bash
# 1. Activate the virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 2. Install dependencies (first run only)
pip install -r requirements.txt

# 3. Start the app
streamlit run src/app.py
```

Open **http://localhost:8501** (Streamlit opens it automatically).

Requirements: Python 3.10-3.12, a working webcam, `assets/face_landmarker.task`
(MediaPipe face landmark model) and `yolov8n.pt` (Ultralytics COCO weights) in the
project root. If the landmark model is missing the Live Monitor shows a friendly
error instead of starting.

The app is organised as a single entry point (`src/app.py`) with a sidebar for page
navigation, a light theme (`.streamlit/config.toml`), a shared header with the
project name and a live **SAFE / ATTENTION / DANGER** status chip.

---

## 🗂️ Pages

| Page | What it does | Status |
| :--- | :--- | :--- |
| **1. Live Monitor** | Webcam feed with Start/Stop, large status banner, six indicator cards, live EAR/MAR/yaw/pitch/FPS, timestamped Messages box, debug telemetry panel | ✅ Implemented |
| **2. Driver Registration** | Driver form, 3-second capture countdown, saves `data/known_drivers/<name>.npy`, registered-driver table with delete | 🚧 Next step |
| **3. Calibration** | Guided neutral pose / eyes open / eyes closed / yawn captures with progress bars, measured values and suggested thresholds saved to `data/calibration.json` | 🚧 Next step |
| **4. Settings** | Sliders for every threshold and timer with Save and Reset-to-default | 🚧 Next step |
| **5. Test Checklist** | One row per test with a live "detected now" indicator, PASS/FAIL buttons you click yourself, results table and CSV download | 🚧 Next step |
| **6. Reports** | Filter `logs/events.csv` by date, event type and driver, summary cards, charts, event table and CSV/HTML downloads | 🚧 Next step |

### Page 1 - Live Monitor

- **Start / Stop** controls; leaving the page stops the detector loop and releases
  the webcam (the sidebar footer shows `Camera: live/off`).
- **Banner:** green `SAFE`, yellow `ATTENTION`, red `DANGER`.
- **Indicator cards:** Drowsy, Yawning, Looking Away, Phone, No Driver, Driver identity.
- **Live numbers:** EAR, MAR, yaw, pitch, FPS (green when ≥ 15 FPS).
- **Messages box:** newest alert first with its timestamp and level, e.g.
  `12:04:31 DANGER Phone usage detected`.
- **Debug telemetry:** `face_detected`, raw/smoothed signals, all timers, phone vote
  count and hold time, pose state reason, derived flags, reason for the current state
  and the active event ids.
- New alerts also raise an `st.toast` and, for DANGER, the audio chime from `alerts.py`.
- Camera problems show an `st.error` with a **Retry** button.

**Performance design.** Capture plus detection run in a separate process
(`src/ui/worker.py`) that publishes JPEG snapshots; the page polls the newest snapshot
from an `st.fragment`. Keeping the loop out of Streamlit's rerun thread is what makes
the 15+ FPS requirement achievable: a same-process loop measured 1.7 FPS at 10 UI
refreshes/s, while the process worker holds ~20-30 FPS on the same machine.

---

## 📁 Project Structure

```text
DriveSafeAi/
├── .streamlit/config.toml      # Light theme
├── assets/
│   ├── alert.wav               # Audio tone played on alerts
│   └── face_landmarker.task    # MediaPipe face landmark model
├── data/
│   ├── calibration.json        # Persisted thresholds and neutral-pose offsets
│   └── known_drivers/          # Registered driver references
├── logs/
│   └── events.csv              # timestamp, driver, event_type, duration
├── src/
│   ├── app.py                  # Streamlit entry point: nav, header, page dispatch
│   ├── config.py               # Central thresholds, paths, calibration persistence
│   ├── drowsiness.py           # EAR / MAR detection with smoothing + hysteresis
│   ├── head_pose.py            # Pitch / yaw / roll via SolvePnP, timers, face-lost states
│   ├── phone_detection.py      # YOLOv8n phone detection, voting window, hold time
│   ├── driver_auth.py          # Face descriptor matching against known drivers
│   ├── alerts.py               # Audio chime with cooldown
│   ├── dashboard.py            # Legacy single-page dashboard (kept for reference)
│   ├── demo.py                 # Standalone OpenCV HUD runner
│   └── ui/
│       ├── theme.py            # Light-theme CSS and HTML components
│       ├── state.py            # Session state, settings, event-log helpers
│       ├── camera.py           # Camera + detector lifecycle
│       ├── pipeline.py         # Per-frame orchestration of the detector modules
│       ├── worker.py           # Background detector process
│       ├── page_live.py        # Page 1 - Live Monitor
│       └── page_placeholder.py # Placeholders for pages 2-6
├── requirements.txt
└── README.md
```

The detector modules are reused unchanged; the UI layer only orchestrates them.

---

## ⚙️ Key Configuration & Thresholds

Defaults live in [`src/config.py`](src/config.py); saved values in
`data/calibration.json` override them at startup.

Head pose comes from MediaPipe's `facial_transformation_matrixes` (with
`cv2.solvePnP` as a fallback) and is reported as yaw (negative left, positive
right), pitch (negative down, positive up) and roll (negative left, positive
right), all normalized to −90°..90° after the neutral calibration offsets.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `EAR_THRESHOLD` | `0.22` | Eye Aspect Ratio below which eyes count as closed |
| `DROWSY_TIME_THRESHOLD_SEC` | `1.5` s | Closed-eye duration for the drowsy alert |
| `MICROSLEEP_TIME_THRESHOLD_SEC` | `3.0` s | Closed-eye duration for the critical micro-sleep alert |
| `MAR_YAWN_THRESHOLD` | `0.52` | Mouth Aspect Ratio above which a yawn is detected |
| `YAW_ATTENTION_THRESHOLD` | `28°` | Head turn angle that starts the distraction timer |
| `YAW_ATTENTION_TIME_SEC` / `YAW_DANGER_TIME_SEC` | `2.0` / `4.0` s | Sustained turn → ATTENTION / DANGER (the Flask monitor uses 3.0 / 4.0 s) |
| `PITCH_DOWN_ATTENTION_THRESHOLD` | `-18°` | Head-down angle for nodding detection |
| `PITCH_UP_THRESHOLD` | `22°` | Head-up angle for looking-up detection |
| `HEAD_MAX_JUMP_DEG` / `HEAD_JUMP_CONFIRM_FRAMES` | `25°` / `3` | Reject a per-frame jump unless it repeats on 3 frames |
| `HEAD_MEDIAN_WINDOW` / `HEAD_SMOOTHING_ALPHA` | `5` / `0.3` | Median window, then exponential smoothing |
| `FACE_LOST_TURN_HOLD_SEC` / `FACE_LOST_NO_DRIVER_SEC` | `1.5` / `3.0` s | Face hidden → FACE NOT VISIBLE / NO DRIVER |
| `PHONE_CONF` | `0.28` | YOLO confidence threshold for cell phones |
| `PHONE_VOTING_WINDOW` / `PHONE_VOTES_NEEDED` | `6` / `3` | Votes needed within the window to alert |
| `PHONE_HOLD_TIME_SEC` | `1.5` s | Keep the phone alert after the last positive frame |
| `FACE_AUTH_SIMILARITY_THRESHOLD` | `0.94` | Cosine similarity for a registered driver |

---

## 📝 Event logging

`logs/events.csv` has one row per event **start** (never per frame):

```csv
timestamp,driver,event_type,duration
2026-10-20 16:25:03,Unknown,NO_DRIVER_STARTED,46.87
```

Event types: `DROWSINESS_STARTED`, `YAWN_STARTED`, `LOOKING_AWAY_STARTED`,
`PHONE_USAGE_STARTED`, `NO_DRIVER_STARTED`, `UNKNOWN_DRIVER_STARTED`.
`duration` is filled in when the event ends (or when monitoring stops).

---

## 🧪 Manual verification

The in-app Test Checklist (page 5) is where PASS/FAIL is recorded by the operator.
For automated checks of the detector logic run:

```bash
python test_verification.py
```

Suggested manual checks for page 1: close your eyes ≥ 1.5 s (Drowsy + DANGER), yawn
(Yawning), turn your head ≥ 2 s (Looking Away), hold a phone (Phone), cover the camera
> 3 s (No Driver), sit still (SAFE, green chip, ≥ 15 FPS).
