"""
Per-frame detection pipeline for DriveSafe AI.

This is a thin orchestration layer on top of the existing, unchanged detector
modules (`drowsiness.py`, `head_pose.py`, `phone_detection.py`, `driver_auth.py`).
It returns a single structured result that pages render, and the annotated frame.
"""

from __future__ import annotations

import time

import cv2

from src import config

# Event ids that are logged / toasted exactly once per occurrence.
EVENT_DROWSY = "DROWSINESS_STARTED"
EVENT_YAWN = "YAWN_STARTED"
EVENT_LOOKING_AWAY = "LOOKING_AWAY_STARTED"
EVENT_PHONE = "PHONE_USAGE_STARTED"
EVENT_NO_DRIVER = "NO_DRIVER_STARTED"
EVENT_UNKNOWN_DRIVER = "UNKNOWN_DRIVER_STARTED"

EVENT_LABELS = {
    EVENT_DROWSY: "Drowsiness",
    EVENT_YAWN: "Yawning",
    EVENT_LOOKING_AWAY: "Looking away",
    EVENT_PHONE: "Phone usage",
    EVENT_NO_DRIVER: "No driver",
    EVENT_UNKNOWN_DRIVER: "Unknown driver",
}

# Toast / message copy per event: (level, message, icon)
EVENT_MESSAGES = {
    EVENT_DROWSY: ("ATTENTION", "Drowsiness detected - eyes closing", "😴"),
    EVENT_YAWN: ("ATTENTION", "Yawning detected", "🥱"),
    EVENT_LOOKING_AWAY: ("ATTENTION", "Looking away from the road", "👀"),
    EVENT_PHONE: ("DANGER", "Phone usage detected", "📱"),
    EVENT_NO_DRIVER: ("ATTENTION", "No driver detected", "🪑"),
    EVENT_UNKNOWN_DRIVER: ("ATTENTION", "Unknown driver detected", "🙋"),
}


def _card(label: str, value: str, detail: str, state: str = "off") -> dict:
    return {"label": label, "value": value, "detail": detail, "state": state}


def _draw_overlays(frame, landmarks, w: int, h: int, head_res: dict, phone_res: dict) -> None:
    """Draws the eye/mouth landmarks, head-pose arrow and phone boxes in place."""
    if landmarks is not None:
        for idx in config.LEFT_EYE_LANDMARKS + config.RIGHT_EYE_LANDMARKS:
            point = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            cv2.circle(frame, point, 2, (34, 197, 94), -1)
        for idx in config.MOUTH_LANDMARKS.values():
            point = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            cv2.circle(frame, point, 2, (245, 158, 11), -1)
        if head_res.get("nose_proj"):
            p1, p2 = head_res["nose_proj"]
            cv2.arrowedLine(frame, p1, p2, (234, 179, 8), 2, tipLength=0.25)

    for det in phone_res.get("detections", []):
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 38, 38), 2)
        cv2.putText(
            frame,
            f"PHONE {det['confidence'] * 100:.0f}%",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 38, 38),
            2,
        )


def process_frame(frame, runtime: dict) -> dict:
    """
    Runs one frame through the detector pipeline.

    ``frame`` must already be mirrored/flipped. The same array is annotated and
    returned as ``result["frame"]``.
    """
    h, w = frame.shape[:2]
    land_detector = runtime["landmarker"]
    drowsiness = runtime["drowsiness"]
    head_pose = runtime["head_pose"]
    phone_detector = runtime["phone"]
    authenticator = runtime["auth"]

    import mediapipe as mp

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection = land_detector.detect(mp_image)
    landmarks = detection.face_landmarks[0] if detection.face_landmarks else None
    face_detected = landmarks is not None

    # Rigid face transform from the landmarker: the head-pose estimator turns
    # this into yaw/pitch/roll and only uses solvePnP when it is unavailable.
    transform = None
    transforms = getattr(detection, "facial_transformation_matrixes", None)
    if transforms:
        transform = transforms[0]

    ear = mar = 0.0
    raw_ear = raw_mar = 0.0
    yaw = pitch = roll = raw_yaw = raw_pitch = raw_roll = norm_pitch = 0.0
    pose_source = "none"
    eyes_closed_duration = 0.0
    drowsy = False
    microsleep = False
    yawning = False
    driver_name = "Unknown"
    similarity = 0.0
    authorized = False

    if face_detected:
        d_res = drowsiness.process(landmarks, w, h)
        head_res = head_pose.estimate(landmarks, w, h, transform)
        auth_res = authenticator.authenticate(landmarks, frame)

        raw_ear = float(d_res.get("raw_ear", 0.0))
        raw_mar = float(d_res.get("raw_mar", 0.0))
        ear = float(d_res.get("ear", 0.0))
        mar = float(d_res.get("mar", 0.0))
        eyes_closed_duration = float(d_res.get("closed_duration", 0.0))

        status = d_res.get("status", "AWAKE")
        microsleep = status == "MICRO-SLEEP DETECTED!"
        drowsy = status in ("DROWSY", "MICRO-SLEEP DETECTED!")
        yawning = status == "YAWNING"

        raw_pitch = float(head_res.get("raw_pitch", 0.0))
        raw_yaw = float(head_res.get("raw_yaw", 0.0))
        raw_roll = float(head_res.get("raw_roll", 0.0))
        norm_pitch = float(head_res.get("norm_pitch", 0.0))
        pose_source = str(head_res.get("pose_source", "none"))
        pitch = float(head_res.get("pitch", 0.0))
        yaw = float(head_res.get("yaw", 0.0))
        roll = float(head_res.get("roll", 0.0))

        driver_name = auth_res.get("driver_name", "Unknown")
        similarity = float(auth_res.get("similarity", 0.0))
        authorized = bool(auth_res.get("is_authorized", False))
    else:
        # No face: let the head-pose state machine produce FACE NOT VISIBLE / NO DRIVER.
        head_res = head_pose.estimate(None, w, h)
        pitch = float(head_res.get("pitch", 0.0))
        yaw = float(head_res.get("yaw", 0.0))
        roll = float(head_res.get("roll", 0.0))

        # Clear drowsiness/yawn timers so stale values cannot fire while nobody is there.
        drowsiness.is_eye_closed = False
        drowsiness.closed_start_time = None
        drowsiness.closed_eye_frames = 0
        drowsiness.is_yawning = False
        drowsiness.yawn_frames = 0

    phone_res = phone_detector.detect(frame)
    phone_active = bool(phone_res.get("phone_in_use", False))

    _draw_overlays(frame, landmarks, w, h, head_res, phone_res)

    pose_status = head_res.get("status", "FORWARD")
    pose_severity = head_res.get("severity", config.SEVERITY_NORMAL)
    turn_duration = float(head_res.get("turn_duration", 0.0))
    pitch_duration = float(head_res.get("pitch_duration", 0.0))

    looking_away = ("LOOKING" in pose_status or "HEAD DOWN" in pose_status) and (
        pose_severity in (config.SEVERITY_WARNING, config.SEVERITY_CRITICAL)
    )
    face_not_visible = pose_status == "FACE NOT VISIBLE"
    no_driver = pose_status == "NO DRIVER DETECTED"

    face_missing = 0.0
    last_face_time = getattr(head_pose, "last_face_time", None)
    if not face_detected and last_face_time is not None:
        face_missing = max(0.0, time.time() - last_face_time)

    is_danger = microsleep or phone_active or pose_severity == config.SEVERITY_CRITICAL
    is_attention = (
        drowsy
        or yawning
        or no_driver
        or pose_severity == config.SEVERITY_WARNING
    )
    level = "DANGER" if is_danger else ("ATTENTION" if is_attention else "SAFE")

    # ---------------- reasons ----------------
    reasons = []
    if phone_active:
        reasons.append(
            "Phone in use (votes {} >= {})".format(
                phone_res.get("true_votes", 0), phone_res.get("votes_needed", 0)
            )
        )
    if microsleep:
        reasons.append(f"Micro-sleep: eyes closed {eyes_closed_duration:.1f}s")
    elif drowsy:
        reasons.append(
            f"Eyes closed {eyes_closed_duration:.1f}s "
            f"(EAR {ear:.2f} < {config.EAR_THRESHOLD:.2f})"
        )
    if yawning:
        reasons.append(f"Yawning (MAR {mar:.2f} > {config.MAR_YAWN_THRESHOLD:.2f})")
    if looking_away:
        reasons.append(f"{pose_status} for {max(turn_duration, pitch_duration):.1f}s")
    if face_not_visible:
        reasons.append(f"Face not visible {face_missing:.1f}s")
    if no_driver:
        reasons.append(f"Face missing {face_missing:.1f}s - no driver in seat")
    if face_detected and not authorized and driver_name == "Unknown":
        reasons.append(f"Unknown driver (similarity {similarity:.2f})")
    if not reasons:
        reasons.append("All detectors within range - driver attentive")

    # ---------------- cards ----------------
    if microsleep:
        drowsy_card = _card("Drowsy", "MICRO-SLEEP", "Eyes closed > 3 s", "red")
    elif drowsy:
        drowsy_card = _card("Drowsy", "YES", f"Eyes closed {eyes_closed_duration:.1f} s", "red")
    else:
        drowsy_card = _card("Drowsy", "NO", "Eyes open", "green")

    if yawning:
        yawn_card = _card("Yawning", "YES", f"MAR {mar:.2f}", "amber")
    else:
        yawn_card = _card("Yawning", "NO", "Mouth relaxed", "off")

    if pose_severity == config.SEVERITY_CRITICAL and looking_away:
        looking_card = _card("Looking Away", "YES", pose_status, "red")
    elif face_not_visible:
        looking_card = _card("Looking Away", "FACE HIDDEN", f"{face_missing:.1f} s", "amber")
    elif looking_away:
        looking_card = _card("Looking Away", "YES", pose_status, "amber")
    else:
        looking_card = _card("Looking Away", "NO", "Forward road attention", "off")

    if phone_active:
        phone_card = _card(
            "Phone",
            "DETECTED",
            f"Votes {phone_res.get('vote_str', '0/0')}",
            "red",
        )
    else:
        phone_card = _card(
            "Phone",
            "NO",
            f"Votes {phone_res.get('vote_str', '0/0')}",
            "off",
        )

    if no_driver:
        no_driver_card = _card("No Driver", "DETECTED", f"Face absent {face_missing:.1f} s", "amber")
    elif not face_detected:
        no_driver_card = _card("No Driver", "NO", "Face partially visible", "off")
    else:
        no_driver_card = _card("No Driver", "NO", "Driver in seat", "off")

    if not face_detected:
        driver_card = _card("Driver", "NO FACE", "Waiting for a face", "off")
    elif authorized:
        driver_card = _card("Driver", driver_name.upper(), "Registered driver", "green")
    else:
        driver_card = _card("Driver", "UNKNOWN", f"Similarity {similarity:.2f}", "off")

    cards = [drowsy_card, yawn_card, looking_card, phone_card, no_driver_card, driver_card]

    # ---------------- events (state transitions) ----------------
    events = set()
    if drowsy:
        events.add(EVENT_DROWSY)
    if yawning:
        events.add(EVENT_YAWN)
    if looking_away:
        events.add(EVENT_LOOKING_AWAY)
    if phone_active:
        events.add(EVENT_PHONE)
    if no_driver:
        events.add(EVENT_NO_DRIVER)

    return {
        "level": level,
        "reason": "; ".join(reasons),
        "face_detected": face_detected,
        "driver": {
            "name": driver_name,
            "authorized": authorized,
            "similarity": similarity,
            "unknown": face_detected and not authorized,
        },
        "metrics": {
            "ear": ear,
            "mar": mar,
            "raw_ear": raw_ear,
            "raw_mar": raw_mar,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "raw_yaw": raw_yaw,
            "raw_pitch": raw_pitch,
            "raw_roll": raw_roll,
            "norm_pitch": norm_pitch,
            "pose_source": pose_source,
            "eyes_closed": eyes_closed_duration,
        },
        "head_pose": head_res.get("pose", {}),
        "timers": {
            "eyes_closed": eyes_closed_duration,
            "looking_away": turn_duration,
            "head_down": pitch_duration,
            "face_missing": face_missing,
        },
        "flags": {
            "drowsy": drowsy,
            "microsleep": microsleep,
            "yawning": yawning,
            "looking_away": looking_away,
            "face_not_visible": face_not_visible,
            "no_driver": no_driver,
            "phone": phone_active,
            "unknown_driver": face_detected and not authorized,
        },
        "pose": {
            "status": pose_status,
            "severity": pose_severity,
            "reason": head_res.get("state_change_reason", ""),
            "source": pose_source,
        },
        "phone": {
            "detected": bool(phone_res.get("detected", False)),
            "votes": int(phone_res.get("true_votes", 0)),
            "votes_needed": int(phone_res.get("votes_needed", 0)),
            "window": int(phone_res.get("voting_window", 0)),
            "vote_str": phone_res.get("vote_str", "0/0"),
            "hold_remaining": float(phone_res.get("hold_remaining", 0.0)),
            "hold_time": float(phone_res.get("hold_time", 0.0)),
            "confidence": float(phone_res.get("last_confidence", 0.0)),
            "reason": phone_res.get("alert_reason", ""),
        },
        "events": events,
        "cards": cards,
        "frame": frame,
    }
