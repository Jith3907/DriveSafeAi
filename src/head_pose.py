"""
Head Pose and Distraction Estimation Module for DriveSafe AI.

Angles come from MediaPipe's ``facial_transformation_matrixes`` (one rigid
transform per face, fitted by the landmarker) and fall back to
``cv2.solvePnP`` only when that matrix is missing.

Conventions, after normalization and neutral calibration:
    yaw    negative = looking left,  positive = looking right
    pitch  negative = looking down,  positive = looking up
    roll   negative = tilted left,   positive = tilted right

Stabilisation: a frame-to-frame jump larger than HEAD_MAX_JUMP_DEG is ignored
unless it persists for HEAD_JUMP_CONFIRM_FRAMES frames, then a median over the
last HEAD_MEDIAN_WINDOW accepted readings is followed by exponential smoothing
(HEAD_SMOOTHING_ALPHA). The distraction timers, hysteresis bands and face-lost
logic are unchanged.
"""

import math
import time
from collections import deque

import cv2
import numpy as np

from src import config

# 3D generic facial reference model points (fallback solvePnP path only).
# +x is image right, +y is up, +z points out of the face towards the camera.
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (landmark 1)
    (0.0, -330.0, -65.0),        # Chin (landmark 152)
    (-225.0, 170.0, -135.0),     # Left eye corner (landmark 33)
    (225.0, 170.0, -135.0),      # Right eye corner (landmark 263)
    (-150.0, -150.0, -125.0),    # Left mouth corner (landmark 61)
    (150.0, -150.0, -125.0)      # Right mouth corner (landmark 291)
], dtype=np.float64)

# Which matrix the angle came from, exposed as ``pose_source``.
POSE_SOURCE_MATRIX = "matrix"
POSE_SOURCE_PNP = "solvePnP"
POSE_SOURCE_NONE = "none"

# MediaPipe's canonical face model is x right, y up, z out of the face, and the
# transformation matrix is expressed in an OpenGL-style camera frame (y up,
# +z towards the camera). The face's forward and up axes are therefore the
# second and third columns of the rotation, unchanged.
FORWARD_COLUMN = 2
UP_COLUMN = 1


# =========================================================
# Angle helpers
# =========================================================
def normalize_angle(angle: float, limit: float = 90.0) -> float:
    """Folds any angle into ``-limit..limit`` degrees.

    Readings are 180-degree periodic for a head pose, so 170, -190 or 540 are
    folded back instead of leaking into the timers as a fake large turn.
    """
    try:
        value = float(angle)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    value = (value + 180.0) % 360.0 - 180.0
    if value > limit:
        value -= 180.0
    elif value < -limit:
        value += 180.0
    return value


def normalize_pitch(pitch: float) -> float:
    """Backwards-compatible alias for :func:`normalize_angle`."""
    return normalize_angle(pitch)


def rotation_from_transform(matrix) -> np.ndarray | None:
    """Returns the 3x3 rotation part of a MediaPipe transformation matrix.

    Accepts a 4x4 array, a flat 16-element sequence or ``None``. Malformed or
    degenerate matrices return ``None`` so the caller can fall back to solvePnP.
    """
    if matrix is None:
        return None
    try:
        flat = np.asarray(matrix, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if flat.size != 16 or not np.all(np.isfinite(flat)):
        return None

    rotation = flat.reshape(4, 4)[:3, :3]
    # The landmarker's matrix is already rigid: all three singular values are 1.
    # Anything else (zeroed/transposed/serialised badly) is rejected so the
    # caller falls back to solvePnP instead of trusting nonsense.
    try:
        u, singular, vt = np.linalg.svd(rotation)
    except np.linalg.LinAlgError:
        return None
    if singular.min() < 0.5 or singular.max() > 1.5:
        return None
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        return None
    return rotation


def _angles_from_axes(forward: np.ndarray, up: np.ndarray, vertical_sign: float) -> tuple:
    """Turns the head's forward/up vectors into yaw, pitch and roll (degrees).

    ``vertical_sign`` maps the camera's vertical axis onto "up" (+1 when the
    frame is y-up, -1 for OpenCV's y-down camera coordinates).
    """
    forward = np.asarray(forward, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    # Signed component of the face normal pointing at the camera.
    depth = float(forward[2]) * (1.0 if vertical_sign > 0 else -1.0)
    # Length of the normal's projection on the camera's horizontal plane, so
    # pitch does not shrink as the head turns sideways.
    horizontal = math.hypot(float(forward[0]), float(forward[2])) or 1e-6
    forward_up = float(forward[1]) * vertical_sign
    up_up = float(up[1]) * vertical_sign

    yaw = math.degrees(math.atan2(float(forward[0]), depth))
    pitch = math.degrees(math.atan2(forward_up, horizontal))
    roll = math.degrees(math.atan2(float(up[0]), up_up))
    return normalize_angle(yaw), normalize_angle(pitch), normalize_angle(roll)


def angles_from_matrix(matrix) -> tuple | None:
    """New angle source: yaw, pitch, roll from the facial transformation matrix.

    The matrix maps the canonical face (x right, y up, z out of the face) into
    an OpenGL-style camera frame (x right, y up, +z towards the camera), so the
    face normal is the third column and the head's up axis the second one.
    Returns ``None`` when the matrix is missing or unusable.
    """
    rotation = rotation_from_transform(matrix)
    if rotation is None:
        return None
    return _angles_from_axes(
        rotation[:, FORWARD_COLUMN],
        rotation[:, UP_COLUMN],
        vertical_sign=1.0,
    )


def solve_pnp_rotation(landmarks, img_w: float, img_h: float) -> np.ndarray | None:
    """Fallback: head rotation (model -> OpenCV camera) from six face points."""
    try:
        image_points = np.array(
            [
                [landmarks[idx].x * img_w, landmarks[idx].y * img_h]
                for idx in config.POSE_LANDMARK_INDICES
            ],
            dtype=np.float64,
        )
    except (IndexError, AttributeError):
        return None
    if image_points.shape != (len(config.POSE_LANDMARK_INDICES), 2):
        return None

    focal = float(max(img_w, 1.0))
    camera_matrix = np.array([
        [focal, 0.0, img_w / 2.0],
        [0.0, focal, img_h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    try:
        success, rot_vec, _ = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            np.zeros((4, 1), dtype=np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None
    if not success:
        return None
    rotation, _ = cv2.Rodrigues(rot_vec)
    return rotation


def angles_from_pnp(rotation: np.ndarray) -> tuple:
    """Same yaw/pitch/roll convention as :func:`angles_from_matrix`, via solvePnP."""
    return _angles_from_axes(
        rotation[:, FORWARD_COLUMN],
        rotation[:, UP_COLUMN],
        vertical_sign=-1.0,  # OpenCV camera coordinates: y down, +z away
    )


class _AngleStabilizer:
    """Spike rejection, then median-of-N, then exponential smoothing.

    A reading further than ``max_jump`` degrees from the last accepted value is
    treated as noise and replaced by that value until it has been seen on
    ``jump_frames`` consecutive frames (a real, fast head movement).
    """

    def __init__(self, max_jump: float, jump_frames: int, window: int, alpha: float):
        self.max_jump = max(0.0, float(max_jump))
        self.jump_frames = max(1, int(jump_frames))
        self.alpha = min(max(float(alpha), 0.0), 1.0)
        self._window: deque = deque(maxlen=max(1, int(window)))
        self._last_accepted: float | None = None
        self._pending = 0
        self._smoothed: float | None = None
        self.rejected = False

    @staticmethod
    def _distance(a: float, b: float) -> float:
        """Shortest distance between two 180-periodic angles."""
        return abs((a - b + 90.0) % 180.0 - 90.0)

    def update(self, raw: float) -> float:
        value = normalize_angle(raw)
        self.rejected = False

        if (self._last_accepted is not None
                and self._distance(value, self._last_accepted) > self.max_jump):
            self._pending += 1
            if self._pending < self.jump_frames:
                value = self._last_accepted
                self.rejected = True
        else:
            self._pending = 0

        self._last_accepted = value
        self._window.append(value)
        median = float(np.median(self._window))

        if self._smoothed is None:
            self._smoothed = median
        else:
            self._smoothed += self.alpha * (median - self._smoothed)

        return normalize_angle(self._smoothed)

    def reset(self) -> None:
        self._window.clear()
        self._last_accepted = None
        self._pending = 0
        self._smoothed = None
        self.rejected = False


def _config_float(name: str, default: float) -> float:
    try:
        return float(getattr(config, name, default))
    except (TypeError, ValueError):
        return float(default)


def _config_int(name: str, default: int) -> int:
    try:
        return int(getattr(config, name, default))
    except (TypeError, ValueError):
        return int(default)


class HeadPoseEstimator:
    def __init__(self, neutral_yaw=None, neutral_pitch=None, neutral_roll=None):
        # Calibration offsets (subtracted from readings)
        self.neutral_yaw = neutral_yaw if neutral_yaw is not None else config.NEUTRAL_YAW
        self.neutral_pitch = neutral_pitch if neutral_pitch is not None else config.NEUTRAL_PITCH
        self.neutral_roll = neutral_roll if neutral_roll is not None else config.NEUTRAL_ROLL

        # Spike rejection + median + exponential smoothing, one filter per axis
        filters = {
            "max_jump": _config_float("HEAD_MAX_JUMP_DEG", 25.0),
            "jump_frames": _config_int("HEAD_JUMP_CONFIRM_FRAMES", 3),
            "window": _config_int("HEAD_MEDIAN_WINDOW", 5),
            "alpha": _config_float("HEAD_SMOOTHING_ALPHA", 0.3),
        }
        self.yaw_filter = _AngleStabilizer(**filters)
        self.pitch_filter = _AngleStabilizer(**filters)
        self.roll_filter = _AngleStabilizer(**filters)

        # Latest angles: *_raw is the per-frame value, the others are filtered.
        self.raw_yaw = 0.0
        self.raw_pitch = 0.0
        self.raw_roll = 0.0
        self.smoothed_yaw = 0.0
        self.smoothed_pitch = 0.0
        self.smoothed_roll = 0.0
        self.pose_source = POSE_SOURCE_NONE

        # Hysteresis state tracking
        self.yaw_active = False
        self.yaw_start_time = None

        self.pitch_down_active = False
        self.pitch_down_start_time = None

        self.pitch_up_active = False
        self.pitch_up_start_time = None

        # Tracking for Part A & B
        self.last_valid_yaw = 0.0
        self.last_valid_pitch = 0.0
        self.last_valid_roll = 0.0
        self.last_yaw_direction = "RIGHT"
        self.last_yaw_above_thresh_time = None
        self.last_face_time = None
        self.state_change_reason = "Initialized"
        self.face_detected = False

    def update_calibration(self, neutral_yaw: float, neutral_pitch: float, neutral_roll: float):
        """Update neutral pose offsets dynamically."""
        self.neutral_yaw = float(neutral_yaw)
        self.neutral_pitch = float(neutral_pitch)
        self.neutral_roll = float(neutral_roll)

    # -----------------------------------------------------
    # Debug / status payload
    # -----------------------------------------------------
    def _pose_fields(self) -> dict:
        """Raw angles, smoothed angles and the source that produced them."""
        return {
            "raw_yaw": round(self.raw_yaw, 2),
            "raw_pitch": round(self.raw_pitch, 2),
            "raw_roll": round(self.raw_roll, 2),
            "norm_pitch": round(self.raw_pitch, 2),
            "smoothed_yaw": round(self.smoothed_yaw, 2),
            "smoothed_pitch": round(self.smoothed_pitch, 2),
            "smoothed_roll": round(self.smoothed_roll, 2),
            "pose_source": self.pose_source,
            "pose": {
                "source": self.pose_source,
                "raw": {
                    "yaw": round(self.raw_yaw, 2),
                    "pitch": round(self.raw_pitch, 2),
                    "roll": round(self.raw_roll, 2),
                },
                "smoothed": {
                    "yaw": round(self.smoothed_yaw, 2),
                    "pitch": round(self.smoothed_pitch, 2),
                    "roll": round(self.smoothed_roll, 2),
                },
                "jump_rejected": bool(
                    self.yaw_filter.rejected
                    or self.pitch_filter.rejected
                    or self.roll_filter.rejected
                ),
            },
        }

    # -----------------------------------------------------
    # HUD arrow
    # -----------------------------------------------------
    @staticmethod
    def _nose_pointer(landmarks, img_w: float, img_h: float, yaw: float, pitch: float):
        """Arrow origin + tip derived from the *smoothed* angles.

        The tip is offset in the image plane in proportion to sin(yaw) and
        sin(pitch), scaled by the inter-eye distance. Projecting a long 3D ray
        from the raw rotation amplified every jitter, which is what made the
        pointer shake while looking left.
        """
        try:
            nose = landmarks[1]
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            p1 = (int(nose.x * img_w), int(nose.y * img_h))
            eye_span = math.hypot(
                (right_eye.x - left_eye.x) * img_w,
                (right_eye.y - left_eye.y) * img_h,
            )
        except (IndexError, AttributeError):
            return None
        if not math.isfinite(eye_span) or eye_span < 1.0:
            eye_span = 0.12 * img_w

        yaw_rad = math.radians(normalize_angle(yaw))
        pitch_rad = math.radians(normalize_angle(pitch))
        length = 0.85 * eye_span
        dx = math.sin(yaw_rad) * math.cos(pitch_rad) * length
        dy = -math.sin(pitch_rad) * math.cos(yaw_rad) * length
        p2 = (int(p1[0] + dx), int(p1[1] + dy))
        return p1, p2

    # -----------------------------------------------------
    # Per-frame update
    # -----------------------------------------------------
    def estimate(self, landmarks, img_w, img_h, facial_transform=None):
        """Runs one frame and returns the pose, distraction state and HUD data.

        ``facial_transform`` is the MediaPipe 4x4 transformation matrix for this
        face (``FaceLandmarkerResult.facial_transformation_matrixes[0]``).
        When it is absent or unusable, ``cv2.solvePnP`` is used instead.
        """
        now = time.time()
        turn_hold_window = float(getattr(config, "FACE_LOST_TURN_HOLD_SEC", 1.5))
        no_driver_sec = float(getattr(config, "FACE_LOST_NO_DRIVER_SEC", 3.0))

        # Check if landmarks are present
        if landmarks is None or len(landmarks) == 0:
            self.face_detected = False
            missing_duration = (now - self.last_face_time) if self.last_face_time is not None else 999.0

            # Part A: If face is lost within 1.5s of the last reading that had |yaw| above threshold,
            # keep the timer running and keep current state (treat it as still looking away).
            had_recent_sharp_turn = (
                self.last_yaw_above_thresh_time is not None and
                (now - self.last_yaw_above_thresh_time) < turn_hold_window
            )

            if had_recent_sharp_turn:
                if not self.yaw_active:
                    self.yaw_active = True
                    self.yaw_start_time = self.last_yaw_above_thresh_time
                turn_duration = (now - self.yaw_start_time) if self.yaw_start_time else 0.0
                direction = self.last_yaw_direction

                if turn_duration >= config.YAW_DANGER_TIME_SEC:
                    status = f"DANGER - LOOKING {direction}"
                    severity = config.SEVERITY_CRITICAL
                    alert_msg = f"DANGER: PROLONGED LOOKING {direction} ({turn_duration:.1f}s)!"
                    play_sound = True
                elif turn_duration >= config.YAW_ATTENTION_TIME_SEC:
                    status = f"ATTENTION - LOOKING {direction}"
                    severity = config.SEVERITY_WARNING
                    alert_msg = f"ATTENTION: LOOKING {direction} ({turn_duration:.1f}s)"
                    play_sound = False
                else:
                    status = f"TURNING {direction} (MIRROR/TURN)"
                    severity = config.SEVERITY_NORMAL
                    alert_msg = ""
                    play_sound = False

                self.state_change_reason = (
                    f"Face lost within {turn_hold_window:.1f}s of sharp turn "
                    f"({now - self.last_yaw_above_thresh_time:.2f}s ago). Holding LOOKING {direction}."
                )

                return {
                    **self._pose_fields(),
                    "pitch": self.last_valid_pitch, "yaw": self.last_valid_yaw, "roll": self.last_valid_roll,
                    "status": status, "severity": severity, "alert": alert_msg, "play_sound": play_sound,
                    "turn_duration": turn_duration, "pitch_duration": 0.0,
                    "yaw_active": self.yaw_active, "pitch_down_active": False,
                    "face_detected": False,
                    "last_yaw": self.last_valid_yaw,
                    "last_pitch": self.last_valid_pitch,
                    "timer_value": turn_duration,
                    "state_change_reason": self.state_change_reason,
                    "nose_proj": None
                }

            # Part B: Separate NO FACE state machine
            if missing_duration >= no_driver_sec:
                # Face missing > 3s: status "NO DRIVER DETECTED" (gray, no sound, not counted as DANGER)
                # Option for soft beep
                self.yaw_active = False
                self.yaw_start_time = None
                status = "NO DRIVER DETECTED"
                severity = "NO_DRIVER"  # Not DANGER, not CRITICAL
                alert_msg = "NO DRIVER DETECTED"
                play_sound = bool(getattr(config, "NO_DRIVER_BEEP", False))
                self.state_change_reason = f"Face missing {missing_duration:.1f}s (> {no_driver_sec:.1f}s): NO DRIVER DETECTED"
            elif missing_duration >= turn_hold_window:
                # Face missing 1.5 to 3 s: status "FACE NOT VISIBLE" (yellow, no sound)
                self.yaw_active = False
                self.yaw_start_time = None
                status = "FACE NOT VISIBLE"
                severity = config.SEVERITY_WARNING
                alert_msg = "FACE NOT VISIBLE"
                play_sound = False
                self.state_change_reason = f"Face missing {missing_duration:.1f}s ({turn_hold_window:.1f}s - {no_driver_sec:.1f}s): FACE NOT VISIBLE"
            else:
                # Face missing < 1.5s without sharp turn
                status = "FORWARD"
                severity = config.SEVERITY_NORMAL
                alert_msg = ""
                play_sound = False
                self.state_change_reason = f"Face missing briefly ({missing_duration:.1f}s < {turn_hold_window:.1f}s, no sharp turn)"

            return {
                **self._pose_fields(),
                "pitch": self.last_valid_pitch, "yaw": self.last_valid_yaw, "roll": self.last_valid_roll,
                "status": status, "severity": severity, "alert": alert_msg, "play_sound": play_sound,
                "turn_duration": 0.0, "pitch_duration": 0.0,
                "yaw_active": False, "pitch_down_active": False,
                "face_detected": False,
                "last_yaw": self.last_valid_yaw,
                "last_pitch": self.last_valid_pitch,
                "timer_value": 0.0,
                "state_change_reason": self.state_change_reason,
                "nose_proj": None
            }

        # Face is detected
        self.face_detected = True
        self.last_face_time = now

        # ---- 1. Head orientation: matrix first, solvePnP only as fallback ----
        angles = angles_from_matrix(facial_transform)
        if angles is not None:
            raw_yaw_val, raw_pitch_val, raw_roll_val = angles
            self.pose_source = POSE_SOURCE_MATRIX
        else:
            rotation = solve_pnp_rotation(landmarks, img_w, img_h)
            if rotation is None:
                self.pose_source = POSE_SOURCE_NONE
                return {
                    **self._pose_fields(),
                    "pitch": self.last_valid_pitch, "yaw": self.last_valid_yaw, "roll": self.last_valid_roll,
                    "status": "FORWARD", "severity": config.SEVERITY_NORMAL,
                    "alert": "", "play_sound": False, "nose_proj": None,
                    "turn_duration": 0.0, "pitch_duration": 0.0,
                    "yaw_active": self.yaw_active, "pitch_down_active": self.pitch_down_active,
                    "face_detected": True,
                    "last_yaw": self.last_valid_yaw,
                    "last_pitch": self.last_valid_pitch,
                    "timer_value": 0.0,
                    "state_change_reason": "No usable pose (matrix missing, solvePnP failed); holding last valid state"
                }
            raw_yaw_val, raw_pitch_val, raw_roll_val = angles_from_pnp(rotation)
            self.pose_source = POSE_SOURCE_PNP

        # 2. Sync calibration offsets from config if not manually overwritten
        if self.neutral_yaw == 0.0 and config.NEUTRAL_YAW != 0.0:
            self.neutral_yaw = config.NEUTRAL_YAW
        if self.neutral_pitch == 0.0 and config.NEUTRAL_PITCH != 0.0:
            self.neutral_pitch = config.NEUTRAL_PITCH
        if self.neutral_roll == 0.0 and config.NEUTRAL_ROLL != 0.0:
            self.neutral_roll = config.NEUTRAL_ROLL

        # 3. Normalize, then apply the neutral calibration offsets
        calibrated_yaw = normalize_angle(normalize_angle(raw_yaw_val) - self.neutral_yaw)
        calibrated_pitch = normalize_angle(normalize_angle(raw_pitch_val) - self.neutral_pitch)
        calibrated_roll = normalize_angle(normalize_angle(raw_roll_val) - self.neutral_roll)

        self.raw_yaw = calibrated_yaw
        self.raw_pitch = calibrated_pitch
        self.raw_roll = calibrated_roll

        # 4. Stabilise: spike rejection -> median of last 5 -> exponential smoothing
        smoothed_yaw = self.yaw_filter.update(calibrated_yaw)
        smoothed_pitch = self.pitch_filter.update(calibrated_pitch)
        smoothed_roll = self.roll_filter.update(calibrated_roll)

        self.smoothed_yaw = smoothed_yaw
        self.smoothed_pitch = smoothed_pitch
        self.smoothed_roll = smoothed_roll
        self.last_valid_yaw = smoothed_yaw
        self.last_valid_pitch = smoothed_pitch
        self.last_valid_roll = smoothed_roll

        # 5. Hysteresis & Distraction Logic (Part A: clear at 80% of trigger threshold)
        hysteresis_ratio = float(getattr(config, "HEAD_HYSTERESIS_RATIO", 0.8))
        yaw_thresh = float(config.YAW_ATTENTION_THRESHOLD)
        yaw_clear_thresh = yaw_thresh * hysteresis_ratio  # 80%

        pitch_down_thresh = float(config.PITCH_DOWN_ATTENTION_THRESHOLD)
        pitch_down_clear_thresh = pitch_down_thresh * hysteresis_ratio  # e.g. -25 * 0.8 = -20

        pitch_up_thresh = float(config.PITCH_UP_THRESHOLD)
        pitch_up_clear_thresh = pitch_up_thresh * hysteresis_ratio

        # Check Yaw Turn (Left / Right)
        if abs(smoothed_yaw) > yaw_thresh:
            self.last_yaw_above_thresh_time = now
            self.last_yaw_direction = "LEFT" if smoothed_yaw < 0 else "RIGHT"
            if not self.yaw_active:
                self.yaw_active = True
                self.yaw_start_time = now
                self.state_change_reason = f"Head turn started: |yaw|={abs(smoothed_yaw):.1f}° > {yaw_thresh:.1f}° ({self.last_yaw_direction})"
            else:
                self.state_change_reason = f"Looking {self.last_yaw_direction} sustained ({now - self.yaw_start_time:.1f}s)"
        else:
            # Under trigger threshold:
            # Reset looking-away timer ONLY when face is detected AND smoothed yaw/pitch is back within neutral range (80% hysteresis)
            if self.yaw_active:
                in_neutral = (
                    abs(smoothed_yaw) <= yaw_clear_thresh and
                    smoothed_pitch >= pitch_down_clear_thresh and
                    smoothed_pitch <= pitch_up_clear_thresh
                )
                if in_neutral:
                    self.yaw_active = False
                    self.yaw_start_time = None
                    self.state_change_reason = f"Face back in neutral: yaw={smoothed_yaw:.1f}° (<= {yaw_clear_thresh:.1f}° [80% clear]), timer reset"
                else:
                    self.state_change_reason = f"Yaw {abs(smoothed_yaw):.1f}° in hysteresis band (>{yaw_clear_thresh:.1f}°), maintaining turn"
            else:
                self.state_change_reason = f"Looking straight / attentive (yaw={smoothed_yaw:.1f}°, pitch={smoothed_pitch:.1f}°)"

        # Check Looking Down (Pitch Down)
        if not self.pitch_down_active:
            if smoothed_pitch < pitch_down_thresh:
                self.pitch_down_active = True
                self.pitch_down_start_time = now
                self.state_change_reason = f"Looking down started: pitch={smoothed_pitch:.1f}° < {pitch_down_thresh:.1f}°"
        else:
            if smoothed_pitch >= pitch_down_clear_thresh:
                self.pitch_down_active = False
                self.pitch_down_start_time = None
                self.state_change_reason = f"Pitch recovered: pitch={smoothed_pitch:.1f}° >= {pitch_down_clear_thresh:.1f}°"

        # Check Looking Up (Pitch Up)
        if not self.pitch_up_active:
            if smoothed_pitch > pitch_up_thresh:
                self.pitch_up_active = True
                self.pitch_up_start_time = now
        else:
            if smoothed_pitch <= pitch_up_clear_thresh:
                self.pitch_up_active = False
                self.pitch_up_start_time = None

        # 6. Determine Status, Severity & Sound
        status = "FORWARD"
        severity = config.SEVERITY_NORMAL
        alert_msg = ""
        play_sound = False
        turn_duration = 0.0
        pitch_duration = 0.0

        if self.yaw_active and self.yaw_start_time is not None:
            turn_duration = now - self.yaw_start_time
            direction = self.last_yaw_direction

            if turn_duration >= config.YAW_DANGER_TIME_SEC:
                status = f"DANGER - LOOKING {direction}"
                severity = config.SEVERITY_CRITICAL
                alert_msg = f"DANGER: PROLONGED LOOKING {direction} ({turn_duration:.1f}s)!"
                play_sound = True
            elif turn_duration >= config.YAW_ATTENTION_TIME_SEC:
                status = f"ATTENTION - LOOKING {direction}"
                severity = config.SEVERITY_WARNING
                alert_msg = f"ATTENTION: LOOKING {direction} ({turn_duration:.1f}s)"
                play_sound = False
            else:
                # Brief turn (<2.0s) -> Mirror check / normal driving
                status = f"TURNING {direction} (MIRROR/TURN)"
                severity = config.SEVERITY_NORMAL
                alert_msg = ""
                play_sound = False

        elif self.pitch_down_active and self.pitch_down_start_time is not None:
            pitch_duration = now - self.pitch_down_start_time
            if pitch_duration >= config.PITCH_DOWN_DANGER_TIME_SEC:
                status = "DANGER - HEAD DOWN"
                severity = config.SEVERITY_CRITICAL
                alert_msg = f"DANGER: HEAD DOWN NODDING OFF ({pitch_duration:.1f}s)!"
                play_sound = True
            elif pitch_duration >= config.PITCH_DOWN_ATTENTION_TIME_SEC:
                status = "ATTENTION - LOOKING DOWN"
                severity = config.SEVERITY_WARNING
                alert_msg = f"ATTENTION: LOOKING DOWN ({pitch_duration:.1f}s)"
                play_sound = False
            else:
                status = "LOOKING DOWN (BRIEF)"
                severity = config.SEVERITY_NORMAL
                alert_msg = ""
                play_sound = False

        elif self.pitch_up_active and self.pitch_up_start_time is not None:
            pitch_up_duration = now - self.pitch_up_start_time
            if pitch_up_duration >= config.YAW_DANGER_TIME_SEC:
                status = "DANGER - LOOKING UP"
                severity = config.SEVERITY_CRITICAL
                alert_msg = f"DANGER: LOOKING UP ({pitch_up_duration:.1f}s)!"
                play_sound = True
            elif pitch_up_duration >= config.YAW_ATTENTION_TIME_SEC:
                status = "ATTENTION - LOOKING UP"
                severity = config.SEVERITY_WARNING
                alert_msg = f"ATTENTION: LOOKING UP ({pitch_up_duration:.1f}s)"
                play_sound = False
            else:
                status = "LOOKING UP (BRIEF)"
                severity = config.SEVERITY_NORMAL
                alert_msg = ""
                play_sound = False

        # Nose pointer from the *smoothed* angles, so it cannot twitch with raw
        # frame-to-frame noise while the head is turned.
        nose_proj = self._nose_pointer(landmarks, img_w, img_h, smoothed_yaw, smoothed_pitch)

        timer_val = turn_duration if self.yaw_active else (pitch_duration if self.pitch_down_active else 0.0)

        return {
            **self._pose_fields(),
            "pitch": smoothed_pitch,
            "yaw": smoothed_yaw,
            "roll": smoothed_roll,
            "status": status,
            "severity": severity,
            "alert": alert_msg,
            "play_sound": play_sound,
            "turn_duration": turn_duration,
            "pitch_duration": pitch_duration,
            "yaw_active": self.yaw_active,
            "pitch_down_active": self.pitch_down_active,
            "face_detected": True,
            "last_yaw": self.last_valid_yaw,
            "last_pitch": self.last_valid_pitch,
            "timer_value": timer_val,
            "state_change_reason": self.state_change_reason,
            "nose_proj": nose_proj
        }
