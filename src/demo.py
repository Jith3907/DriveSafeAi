"""
DriveSafe AI - Real-time Driver Monitoring System Demo.
Integrates face mesh, drowsiness, head pose, phone detection, and visual HUD overlays.
"""
# Ensure src package is importable when running demo directly
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from src import config
from src.drowsiness import DrowsinessDetector
from src.head_pose import HeadPoseEstimator
from src.phone_detection import PhoneDetector
from src.driver_auth import DriverAuthenticator
from src.alerts import AlertSystem

def draw_hud(frame, drowsiness_res, head_pose_res, phone_res, auth_res, active_alert, fps):
    h, w, _ = frame.shape

    # Semi-transparent top HUD dashboard
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 88), (20, 20, 24), -1)
    # Semi-transparent bottom banner
    cv2.rectangle(overlay, (0, h - 48), (w, h), (20, 20, 24), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Top Header Title & FPS
    cv2.putText(frame, "DriveSafe AI", (18, 28), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 230, 255), 2)
    fps_color = (100, 255, 100) if fps >= 15.0 else (50, 50, 240)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 115, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 2)

    # Telemetry metrics row 1 (Driver & Eyes)
    driver_name = auth_res.get("driver_name", "Unknown")
    is_auth = auth_res.get("is_authorized", False)
    auth_color = (100, 230, 100) if is_auth else (0, 140, 255)
    cv2.putText(frame, f"Driver: {driver_name}", (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, auth_color, 1)

    ear_val = drowsiness_res.get("ear", 0.0)
    mar_val = drowsiness_res.get("mar", 0.0)
    drowsy_color = config.ALERT_COLORS.get(drowsiness_res.get("severity"), (200, 200, 200))
    cv2.putText(frame, f"EAR: {ear_val:.2f} | MAR: {mar_val:.2f}", (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, drowsy_color, 1)

    # Telemetry metrics row 2 (Head Pose & Phone)
    pitch = head_pose_res.get("pitch", 0.0)
    yaw = head_pose_res.get("yaw", 0.0)
    pose_status = head_pose_res.get("status", "FORWARD")
    pose_color = config.ALERT_COLORS.get(head_pose_res.get("severity"), (200, 200, 200))

    cv2.putText(frame, f"Head: {pose_status} (P:{pitch:.0f} Y:{yaw:.0f})", (w // 2 - 40, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pose_color, 1)
    phone_color = (0, 0, 255) if phone_res["detected"] else (100, 220, 100)
    phone_text = f"Phone: {'DETECTED (' + str(phone_res.get('consec_frames', 0)) + 'f)' if phone_res['detected'] else 'CLEAR'}"
    cv2.putText(frame, phone_text, (w // 2 - 40, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, phone_color, 1)

    # Draw Head Pose 3D orientation axis line
    if head_pose_res.get("nose_proj"):
        p1, p2 = head_pose_res["nose_proj"]
        cv2.arrowedLine(frame, p1, p2, (0, 255, 255), 2, tipLength=0.25)

    # Draw Phone bounding boxes
    for det in phone_res.get("detections", []):
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"PHONE {det['confidence']*100:.0f}%", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Active alert display in bottom banner
    if active_alert and active_alert.get("message"):
        msg = active_alert["message"]
        color = active_alert["color"]
        cv2.putText(frame, f"[ ! ] {msg}", (20, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 2)
    else:
        cv2.putText(frame, "STATUS: ALL SYSTEMS NORMAL - ATTENTIVE  |  [R] Register Driver", (20, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (50, 220, 50), 1)

    return frame

def run_demo():
    print("[DriveSafe AI] Initializing neural detectors...")

    # Initialize MediaPipe FaceLandmarker Task
    base_options = mp_python.BaseOptions(model_asset_path=str(config.FACE_LANDMARKER_MODEL_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    # Initialize Submodules
    drowsiness_detector = DrowsinessDetector()
    head_pose_estimator = HeadPoseEstimator()
    phone_detector = PhoneDetector()
    authenticator = DriverAuthenticator(face_landmarker=face_landmarker)
    alert_system = AlertSystem()

    print(f"[DriveSafe AI] Opening Camera (index {config.CAMERA_INDEX})...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print("[DriveSafe AI] Error: Could not access camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    print("[DriveSafe AI] Pipeline running. Press 'q' or 'ESC' on the video window to quit. Press 'r' to register current driver.")

    prev_time = time.time()
    session_start_time = time.time()
    total_frames = 0
    unknown_driver_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[DriveSafe AI] Warning: Empty frame received from camera.")
            time.sleep(0.03)
            continue

        frame = cv2.flip(frame, 1)  # Mirror view
        h, w, _ = frame.shape

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 30.0
        prev_time = current_time
        total_frames += 1

        # 1. MediaPipe Face Detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = face_landmarker.detect(mp_image)

        drowsiness_res = {"ear": 0.0, "mar": 0.0, "status": "NO FACE", "severity": config.SEVERITY_NORMAL, "alert": ""}
        head_pose_res = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "status": "NO FACE", "severity": config.SEVERITY_NORMAL, "alert": ""}
        auth_res = {"driver_name": "Scanning...", "is_authorized": False, "status": "SCANNING"}
        current_landmarks = None

        if detection_result.face_landmarks:
            current_landmarks = detection_result.face_landmarks[0]
            drowsiness_res = drowsiness_detector.process(current_landmarks, w, h)
            head_pose_res = head_pose_estimator.estimate(current_landmarks, w, h)
            auth_res = authenticator.authenticate(current_landmarks, frame)

        # 2. YOLO Phone Detection (optimized with imgsz=320 for 22ms inference)
        phone_res = phone_detector.detect(frame)

        # 3. Determine Active Driving Hazard Alert (Safe unless active hazard)
        active_alert = None
        alerts_list = [
            (phone_res.get("severity"), phone_res.get("alert"), True),
            (drowsiness_res.get("severity"), drowsiness_res.get("alert"), True),
            (head_pose_res.get("severity"), head_pose_res.get("alert"), head_pose_res.get("play_sound", True))
        ]

        # Prioritize Critical alerts over Warning alerts
        for severity, alert_msg, play_snd in alerts_list:
            if severity == config.SEVERITY_CRITICAL and alert_msg:
                active_alert = alert_system.trigger_alert(alert_msg, severity, play_sound=play_snd)
                break
        if not active_alert:
            for severity, alert_msg, play_snd in alerts_list:
                if severity == config.SEVERITY_WARNING and alert_msg:
                    active_alert = alert_system.trigger_alert(alert_msg, severity, play_sound=play_snd)
                    break

        # 5. Render HUD Overlay
        output_frame = draw_hud(frame, drowsiness_res, head_pose_res, phone_res, auth_res, active_alert, fps)

        # 6. Display Window
        cv2.imshow("DriveSafe AI - Real-time Driver Monitoring", output_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r') and current_landmarks is not None:
            # Register current driver on key 'r'
            authenticator.register_face("Registered Driver", current_landmarks, frame)

    elapsed_total = time.time() - session_start_time
    avg_session_fps = total_frames / elapsed_total if elapsed_total > 0 else 0.0

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DriveSafe AI] Demo ended cleanly. Total frames: {total_frames}, Elapsed: {elapsed_total:.2f}s, Average FPS: {avg_session_fps:.2f}")

if __name__ == "__main__":
    run_demo()
