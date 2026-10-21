"""
Drowsiness and Yawn Detection Module for DriveSafe AI.
Uses Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) computed from facial landmarks.
"""

import time
import numpy as np
from src import config

def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_ear(landmarks, eye_indices, img_w, img_h):
    """
    Computes Eye Aspect Ratio (EAR) for 6 eye landmarks:
    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    """
    coords = [(landmarks[idx].x * img_w, landmarks[idx].y * img_h) for idx in eye_indices]
    p1, p2, p3, p4, p5, p6 = coords
    
    vertical_1 = euclidean_dist(p2, p6)
    vertical_2 = euclidean_dist(p3, p5)
    horizontal = euclidean_dist(p1, p4)
    
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def compute_mar(landmarks, img_w, img_h):
    """
    Computes Mouth Aspect Ratio (MAR):
    MAR = |upper_lip - lower_lip| / |left_corner - right_corner|
    """
    m = config.MOUTH_LANDMARKS
    p_upper = (landmarks[m["upper_lip"]].x * img_w, landmarks[m["upper_lip"]].y * img_h)
    p_lower = (landmarks[m["lower_lip"]].x * img_w, landmarks[m["lower_lip"]].y * img_h)
    p_left = (landmarks[m["left_corner"]].x * img_w, landmarks[m["left_corner"]].y * img_h)
    p_right = (landmarks[m["right_corner"]].x * img_w, landmarks[m["right_corner"]].y * img_h)
    
    vertical = euclidean_dist(p_upper, p_lower)
    horizontal = euclidean_dist(p_left, p_right)
    
    if horizontal == 0:
        return 0.0
    return vertical / horizontal

from collections import deque

class DrowsinessDetector:
    def __init__(self):
        self.closed_eye_frames = 0
        self.closed_start_time = None
        self.yawn_frames = 0

        # Smoothing buffers (last 5 frames)
        self.ear_history = deque(maxlen=config.SMOOTHING_WINDOW_SIZE)
        self.mar_history = deque(maxlen=config.SMOOTHING_WINDOW_SIZE)

        # Hysteresis states
        self.is_eye_closed = False
        self.is_yawning = False

    def process(self, landmarks, img_w, img_h):
        left_ear = compute_ear(landmarks, config.LEFT_EYE_LANDMARKS, img_w, img_h)
        right_ear = compute_ear(landmarks, config.RIGHT_EYE_LANDMARKS, img_w, img_h)
        raw_ear = float((left_ear + right_ear) / 2.0)
        raw_mar = float(compute_mar(landmarks, img_w, img_h))

        # 1. 5-frame moving average smoothing
        self.ear_history.append(raw_ear)
        self.mar_history.append(raw_mar)
        smoothed_ear = float(np.mean(self.ear_history))
        smoothed_mar = float(np.mean(self.mar_history))

        # 2. Hysteresis for Eye Closure
        # Trigger when EAR < threshold; Clear when EAR >= threshold / 0.9 (approx 11% higher)
        ear_thresh = float(config.EAR_THRESHOLD)
        ear_clear_thresh = ear_thresh / config.HYSTERESIS_RATIO

        now = time.time()
        if not self.is_eye_closed:
            if smoothed_ear < ear_thresh:
                self.is_eye_closed = True
                self.closed_start_time = now
        else:
            if smoothed_ear >= ear_clear_thresh:
                self.is_eye_closed = False
                self.closed_start_time = None

        closed_duration = 0.0
        if self.is_eye_closed and self.closed_start_time is not None:
            closed_duration = now - self.closed_start_time
            self.closed_eye_frames += 1
        else:
            self.closed_eye_frames = max(0, self.closed_eye_frames - 2)

        # 3. Hysteresis for Yawn
        # Trigger when MAR > threshold; Clear when MAR <= 0.9 * threshold
        mar_thresh = float(config.MAR_YAWN_THRESHOLD)
        mar_clear_thresh = mar_thresh * config.HYSTERESIS_RATIO

        if not self.is_yawning:
            if smoothed_mar > mar_thresh:
                self.yawn_frames += 1
                if self.yawn_frames >= config.MAR_CONSEC_FRAMES:
                    self.is_yawning = True
            else:
                self.yawn_frames = max(0, self.yawn_frames - 1)
        else:
            if smoothed_mar <= mar_clear_thresh:
                self.is_yawning = False
                self.yawn_frames = 0

        status = "AWAKE"
        severity = config.SEVERITY_NORMAL
        alert_msg = ""

        # Micro-sleep vs Drowsy warning
        if closed_duration >= config.MICROSLEEP_TIME_THRESHOLD_SEC:
            status = "MICRO-SLEEP DETECTED!"
            severity = config.SEVERITY_CRITICAL
            alert_msg = "CRITICAL: WAKE UP! EYES CLOSED!"
        elif closed_duration >= config.DROWSY_TIME_THRESHOLD_SEC:
            status = "DROWSY"
            severity = config.SEVERITY_WARNING
            alert_msg = "WARNING: DROWSINESS DETECTED!"
        elif self.is_yawning:
            status = "YAWNING"
            severity = config.SEVERITY_WARNING
            alert_msg = "WARNING: YAWNING DETECTED!"

        return {
            "raw_ear": raw_ear,
            "raw_mar": raw_mar,
            "ear": smoothed_ear,
            "mar": smoothed_mar,
            "status": status,
            "severity": severity,
            "alert": alert_msg,
            "closed_duration": closed_duration,
            "closed_frames": self.closed_eye_frames,
            "yawn_frames": self.yawn_frames,
            "is_eye_closed": self.is_eye_closed,
            "is_yawning": self.is_yawning
        }
